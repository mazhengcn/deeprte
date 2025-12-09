import dataclasses
import datetime
import json
import pathlib

import jax
import jax.numpy as jnp
from absl import logging
from flax import nnx
from jax.sharding import AxisType

from deeprte.configs import config
from deeprte.input_pipeline import input_pipeline_interface
from deeprte.model import features
from deeprte.model.model import DeepRTE
from deeprte.train_lib import checkpointing, optimizers
from deeprte.train_lib import utils as train_utils
from deeprte.train_lib.gradient_accumulation import gradient_accumulation_loss_and_grad
from deeprte.train_lib.metric_logger import MetricLogger


@jax.jit(static_argnames=("gradient_accumulation_steps",))
def train_step(
    model: nnx.Module, optimizer: nnx.Optimizer, batch, gradient_accumulation_steps: int
):
    """Perform a single training step."""
    graphdef, params, nondiff = nnx.split(model, nnx.Param, ...)

    def loss_fn(params, data):
        """Loss function used for training."""
        model = nnx.merge(graphdef, params)
        labels = data["psi_label"]
        loss = jnp.mean((model(data) - labels) ** 2)
        return loss, {"mean_squared_labels": jnp.mean(labels**2)}

    (loss, aux), grads = gradient_accumulation_loss_and_grad(
        loss_fn, gradient_accumulation_steps, nnx.as_immutable_vars(params), batch
    )
    optimizer.update(model, grads)
    scalar_metrics = {
        "learning/loss": loss,
        "learning/relative_loss": jnp.sqrt(loss / aux["mean_squared_labels"]),
    }
    metrics = {"scalar": scalar_metrics, "scalars": {}}
    return metrics


@jax.jit
def eval_step(model: nnx.Module, metrics: nnx.MultiMetric, batch):
    """Calculate evaluation metrics on a batch."""
    labels = batch["psi_label"]
    predictions = model(batch)  # ty:ignore
    loss = jnp.mean((predictions - labels) ** 2)
    metrics.update(loss=loss, mean_squared_labels=jnp.mean(batch["psi_label"] ** 2))


def evaluate(model, metrics, eval_iter, subcollocation_size: int = 128):
    """Evaluate the target an return a dictionary with the metrics."""
    logging.info("Gathering evaluation metrics.")
    for eval_batch in eval_iter:
        phase_feat, other_feat = features.split_feature(eval_batch)
        phase_feat["psi_label"] = other_feat.pop("psi_label")
        num_subcollocations, last_subcollocation_size = divmod(
            phase_feat["psi_label"].shape[1], subcollocation_size
        )
        for i in range(num_subcollocations):
            subcollocation_feat = jax.tree.map(
                lambda x: x[:, i * subcollocation_size : (i + 1) * subcollocation_size],
                phase_feat,
            )
            eval_step(model, metrics, subcollocation_feat | other_feat)
        if last_subcollocation_size != 0:
            subcollocation_feat = jax.tree.map(
                lambda x: x[:, -last_subcollocation_size:], phase_feat
            )
            eval_step(model, metrics, subcollocation_feat | other_feat)


def train_loop(config: config.Config, workdir: str | pathlib.Path):
    """Runs a training and evaluation loop.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.

    """
    workdir = pathlib.Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    logging.info("Initializing mesh as global context.")
    mesh = jax.make_mesh(
        config.mesh_shape,
        config.mesh_axis_names,
        len(config.mesh_shape) * (AxisType.Explicit,),
    )
    logging.info("Mesh info: %s", mesh)

    with jax.set_mesh(mesh):
        # Build model constructor, optimizer and checkpoint manager
        # ---------------------------------------------------------------------------
        nnx.use_hijax(True)

        logging.info(
            f"Initializing optimizer, model and checkpointer with Hijax {'enabled' if nnx.using_hijax() else 'disabled'}."
        )
        lr_schedule = optimizers.create_learning_rate_schedule(config)
        tx = optimizers.create_optimizer(config, lr_schedule)
        # tx = optax.MultiSteps(tx, every_k_schedule=config.micro_steps)

        # accumulated_train_step = accumulate_gradent(
        #     config.micro_steps, config.global_batch_size
        # )

        checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
            workdir,
            config.save_checkpoints,
            config.async_checkpointing,
            config.checkpoint_every_steps,
            config.dataset_type,
        )

        # Setup metric logger
        # ---------------------------------------------------------------------------
        metric_logger = MetricLogger(config, lr_schedule)

        # Load Dataset
        # ---------------------------------------------------------------------------
        logging.info("Initializing dataset.")
        train_iter, eval_iter = input_pipeline_interface.create_data_iterator(config)

        # Initialize train state
        # ---------------------------------------------------------------------------
        logging.info("Initializing train state.")
        init_rng = jax.random.key(config.seed)

        model, optimizer, train_iter = train_utils.setup_training_state(
            model_class=DeepRTE,
            config=config,
            rng=init_rng,
            tx=tx,
            data_iterator=train_iter,
            checkpoint_manager=checkpoint_manager,
        )

        start_step = int(optimizer.step.get_value()) // config.micro_steps
        logging.info(f"Starting from step: {start_step}")
        if start_step == 0:
            metric_logger.write_setup_info_to_tensorboard(nnx.state(model))
            with (workdir / "config.json").open("w") as f:
                json.dump(dataclasses.asdict(config), f, indent=2)

        # Main Train Loop
        # ---------------------------------------------------------------------------
        logging.info("Starting training loop.")
        try:
            last_step_completion = datetime.datetime.now()
            # with checkpoint_manager as ckpt_mngr:
            for step in range(start_step, config.num_train_steps):
                is_last_step = step == config.num_train_steps - 1

                with jax.profiler.StepTraceAnnotation("train", step_num=step):
                    batch = next(train_iter)
                    # metrics = accumulated_train_step(model, optimizer, batch)
                    metrics = train_step(model, optimizer, batch, config.micro_steps)

                step_time_delta = datetime.datetime.now() - last_step_completion
                last_step_completion = datetime.datetime.now()
                # Periodic metric handling.
                if step % config.log_every_steps == 0 or is_last_step:
                    metric_logger.buffer_and_write_train_metrics(
                        metrics, step, step_time_delta
                    )

                    # if eval_iter:
                    #     if step % config.eval_every_steps == 0 or is_last_step:
                    #         with report_progress.timed("eval"):
                    #             evaluate(model, metrics, eval_iter)
                    #             writer.write_scalars(step, metrics.compute())
                    #         metrics.reset()

                    # if config.save_checkpoints:
                    #     with report_progress.timed("checkpoint"):
                    #         save_checkpoint(
                    #             ckpt_mngr,
                    #             step,
                    #             nnx.state(optimizer),
                    #             config.dataset_type,
                    #             train_iter,
                    #         )
        except train_utils.StopTraining as e:
            logging.info(f"Training stopped: {str(e)}")
        finally:
            metric_logger.flush_metrics_and_cleanup()
