import dataclasses

import jax
import jax.numpy as jnp
import tensorflow as tf
import yaml
from absl import logging
from clu import metric_writers, periodic_actions
from flax import nnx
from jax.sharding import Mesh

from deeprte.configs import default
from deeprte.input_pipeline import input_pipeline_interface
from deeprte.model import features, mapping
from deeprte.model.modules import DeepRTE
from deeprte.train_lib import checkpointing, optimizers
from deeprte.train_lib import utils as train_utils
from deeprte.train_lib.checkpointing import save_checkpoint
from deeprte.train_lib.metrics import RelativeError


# Primary training / eval / decode step functions.
# -----------------------------------------------------------------------------
def loss_fn(model: nnx.Module, batch):
    """Loss function used for training."""
    labels = batch["psi_label"]
    predictions = model(batch)
    return ((predictions - labels) ** 2).mean()


@nnx.jit
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    batch,
    micro_steps: int = 1,
):
    """Perform a single training step."""

    if micro_steps > 1:

        def accumulate_gradient(acc_grad_and_loss, data):
            grad_fn = nnx.value_and_grad(loss_fn)
            loss, cur_batch_gradient = grad_fn(model, data)
            acc_grad_and_loss["loss"] += loss
            acc_grad_and_loss["grads"] += cur_batch_gradient
            return acc_grad_and_loss

        def reshape_to_microbatch_accumulations(batch_arr):
            """Reshape global batch to microbatches, assuming batch axis is leading."""
            microbatch_shape = (
                micro_steps,
                batch_arr.shape[0] // micro_steps,
            ) + batch_arr.shape[1:]
            return jnp.reshape(batch_arr, microbatch_shape)

    batch = jax.tree.map(reshape_to_microbatch_accumulations, batch)
    init_grad = jax.tree.map(jnp.zeros_like, nnx.state(model, nnx.Param))
    init_grad_and_loss = {"loss": 0.0, "grads": init_grad}
    grad_and_loss = jax.lax.scan(
        accumulate_gradient, init_grad_and_loss, batch, length=micro_steps
    )
    loss = grad_and_loss["loss"] / micro_steps
    optimizer.update(grad_and_loss["grads"])

    metrics.update(loss=loss, mean_squared_labels=jnp.mean(batch["psi_label"] ** 2))


@nnx.jit
def eval_step(model: nnx.Module, metrics: nnx.MultiMetric, batch):
    """Calculate evaluation metrics on a batch."""
    loss = loss_fn(model, batch)
    metrics.update(loss=loss, mean_squred_labels=jnp.mean(batch["psi_labels"] ** 2))
    return loss


def evaluate(model, metrics, eval_iter):
    """Evaluate the target an return a dictionary with the metrics."""
    logging.info("Gathering evaluation metrics.")
    for eval_batch in eval_iter:
        phase_feat, other_feat = features.split_feature(eval_batch)
        phase_feat["psi_label"] = other_feat.pop("psi_label")
        _ = mapping.inference_subbatch(
            module=lambda feat: eval_step(model, metrics, feat),
            subbatch_size=128,
            batched_args=phase_feat,
            nonbatched_args=other_feat,
            low_memory=True,
            input_subbatch_dim=1,
        )


def train_and_evaluate(config: default.Config, workdir: str):
    """Runs a training and evaluation loop.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """
    tf.io.gfile.makedirs(workdir)

    init_rng = jax.random.key(config.seed)

    start_step = 0

    # Mesh definition
    # ---------------------------------------------------------------------------
    logging.info("Initializing mesh.")

    devices_array = train_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    # Build model constructor, optimizer and checkpoint manager
    # ---------------------------------------------------------------------------
    logging.info("Initializing optimizer, model and checkpointer.")

    lr_schedule = optimizers.create_learning_rate_schedule(config)
    tx = optimizers.create_optimizer(config, lr_schedule)

    checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
        workdir,
        config.save_checkpoints,
        config.async_checkpointing,
        config.checkpoint_every_steps,
        config.dataset_type,
    )

    # Setup Metrics
    # ---------------------------------------------------------------------------
    metrics: nnx.MultiMetric = nnx.MultiMetric(
        mse=nnx.metrics.Average("loss"),
        rmse=RelativeError("loss", "mean_squred_labels"),
    )

    # Create metric writers
    writer = metric_writers.create_default_writer(
        workdir, just_logging=jax.process_index() > 0
    )

    # Load Dataset
    # ---------------------------------------------------------------------------
    logging.info("Initializing dataset.")
    (train_iter, eval_iter), data_sharding = (
        input_pipeline_interface.create_data_iterator(config, mesh)
    )

    # Initialize train state
    # ---------------------------------------------------------------------------
    logging.info("Initializing train state.")
    model, optimizer, train_iter = train_utils.setup_training_state(
        DeepRTE, train_iter, tx, config, init_rng, mesh, checkpoint_manager
    )
    num_params = train_utils.calculate_num_params_from_pytree(nnx.state(model))
    logging.info(f"Number of model params={num_params}")

    start_step = optimizer.step.value // config.micro_steps
    if start_step == 0:
        writer.write_hparams(dataclasses.asdict(config))
        with open(f"{workdir}/config.yaml", "w") as f:
            yaml.dump(dataclasses.asdict(config), f)

    # Main Train Loop
    # ---------------------------------------------------------------------------
    logging.info("Starting training loop.")
    hooks = []
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=config.num_train_steps, writer=writer
    )
    if jax.process_index() == 0:
        hooks += [
            report_progress,
            periodic_actions.Profile(logdir=workdir, num_profile_steps=5),
        ]
    with metric_writers.ensure_flushes(writer), checkpoint_manager as ckpt_mngr:
        for step in range(start_step, config.num_train_steps):
            is_last_step = step == config.num_train_steps - 1

            with jax.profiler.StepTraceAnnotation("train", step_num=step):
                batch = next(train_iter)
                train_step(model, optimizer, metrics, batch)

            # Quick indication that training is happening.
            logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
            for h in hooks:
                h(step)

            # Periodic metric handling.
            if step % config.log_every_steps == 0 or is_last_step:
                with report_progress.timed("training_metrics"):
                    logging.info("Gathering training metrics.")
                    train_metrics = metrics.compute()
                    writer.write_scalars(step, train_metrics)
                    metrics.reset()

            if eval_iter and step % config.eval_every_steps == 0 or is_last_step:
                with report_progress.timed("eval"):
                    evaluate(model, metrics, eval_iter)
                    eval_metrics = metrics.compute()
                    writer.write_scalars(step, eval_metrics)
                    metrics.reset()

            if config.save_checkpoints:
                with report_progress.timed("checkpoint"):
                    save_checkpoint(
                        ckpt_mngr,
                        step,
                        nnx.state(optimizer),
                        config.dataset_type,
                        train_iter,
                    )
