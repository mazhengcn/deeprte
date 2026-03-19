import dataclasses
import datetime
import json
import pathlib

import jax
import jax.numpy as jnp
from absl import app, flags
from flax import nnx
from jax.sharding import AxisType

from deeprte.configs import base
from deeprte.input_pipeline import input_pipeline_interface
from deeprte.model.model import DeepRTE
from deeprte.train_lib import checkpointing, logging, optimizers, profiler
from deeprte.train_lib import utils as train_utils
from deeprte.train_lib.gradient_accumulation import gradient_accumulation_loss_and_grad
from deeprte.train_lib.metric_logger import MetricLogger

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data.")
flags.DEFINE_string(
    "config", None, "File path to the training hyperparameter configuration."
)
flags.mark_flags_as_required(["config", "workdir"])


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

    params = nnx.as_immutable_vars(params)
    if gradient_accumulation_steps > 1:
        (loss, aux), grads = gradient_accumulation_loss_and_grad(
            loss_fn, gradient_accumulation_steps, params, batch
        )
    else:
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
    optimizer.update(model, grads)
    scalar_metrics = {
        "learning/loss": loss,
        "learning/relative_loss": jnp.sqrt(loss / aux["mean_squared_labels"]),
    }
    metrics = {"scalar": scalar_metrics, "scalars": {}}
    return metrics


@jax.jit
def eval_step(model: nnx.Module, batch):
    """Calculate evaluation metrics on a batch."""
    labels = batch["psi_label"]
    predictions = model(batch)
    loss = jnp.mean((predictions - labels) ** 2)
    metrics = {
        "scalar": {
            "evaluation/loss": loss,
            "evaluation/mean_squared_labels": jnp.mean(labels**2),
        }
    }

    return metrics


def evaluate(model, eval_iter, step, metric_logger):
    """Evaluate the target an return a dictionary with the metrics."""
    logging.log("Gathering evaluation metrics.")
    eval_step_count = 0
    eval_model = nnx.as_immutable_vars(model)
    for eval_batch in eval_iter:
        t0 = datetime.datetime.now()
        metrics = eval_step(eval_model, eval_batch)
        t1 = datetime.datetime.now()
        metric_logger.record_eval_metrics(step, metrics)
        logging.log(
            f"Completed eval step {eval_step_count}, time taken: {(t1 - t0).total_seconds():.3f}"
        )
        eval_step_count += 1


def train_loop(config: base.Config, workdir: str | pathlib.Path):
    """Runs a training and evaluation loop.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.

    """
    workdir = pathlib.Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    logging.log("Initializing mesh as global context.")
    mesh = jax.make_mesh(
        config.mesh_shape,
        config.mesh_axis_names,
        len(config.mesh_shape) * (AxisType.Explicit,),
    )
    logging.log(f"Mesh info: {mesh}")

    with jax.set_mesh(mesh):
        # Build model constructor, optimizer and checkpoint manager
        # ---------------------------------------------------------------------------
        nnx.use_hijax(True)

        logging.log(
            f"Initializing optimizer, model and checkpointer with Hijax {'enabled' if nnx.using_hijax() else 'disabled'}."
        )
        lr_schedule = optimizers.create_learning_rate_schedule(config)
        tx = optimizers.create_optimizer(config, lr_schedule)

        checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
            workdir,
            config.save_checkpoints,
            config.async_checkpointing,
            config.checkpoint_every_steps,
        )

        # Load Dataset
        # ---------------------------------------------------------------------------
        logging.log("Initializing dataset.")
        train_iter, eval_iter = input_pipeline_interface.create_data_iterator(config)

        # Initialize train state
        # ---------------------------------------------------------------------------
        logging.log("Initializing train state.")
        init_rng = jax.random.key(config.seed)

        model, optimizer, train_iter = train_utils.setup_training_state(
            model_class=DeepRTE,
            config=config,
            rng=init_rng,
            tx=tx,
            data_iterator=train_iter,
            checkpoint_manager=checkpoint_manager,
        )
        eval_model = nnx.merge(*nnx.split(model))
        eval_model.set_attributes(low_memory_mode=True)

        start_step = int(optimizer.step.get_value()) // config.micro_steps
        # Setup metric logger and profiler
        # ---------------------------------------------------------------------------
        metric_logger = MetricLogger(config, lr_schedule)
        prof = profiler.Profiler(config, offset_step=start_step)

        logging.log(f"Starting from step: {start_step}")
        if start_step == 0:
            metric_logger.write_setup_info_to_tensorboard(nnx.state(model))
            with (workdir / "config.json").open("w") as f:
                json.dump(dataclasses.asdict(config), f, indent=2)

        # Main Train Loop
        # ---------------------------------------------------------------------------
        logging.log("Starting training loop.")
        try:
            last_step_completion = datetime.datetime.now()
            # with checkpoint_manager as ckpt_mngr:
            for step in range(start_step, config.num_train_steps):
                is_last_step = step == config.num_train_steps - 1
                prof.maybe_activate_profiler(step, state=nnx.state((model, optimizer)))

                with jax.profiler.StepTraceAnnotation("train", step_num=step):
                    batch = next(train_iter)
                    metrics = train_step(model, optimizer, batch, config.micro_steps)

                step_time_delta = datetime.datetime.now() - last_step_completion
                last_step_completion = datetime.datetime.now()

                checkpointing.maybe_save_checkpoint(
                    checkpoint_manager,
                    nnx.state((model, optimizer)),
                    config,
                    train_iter,
                    step,
                )
                # Periodic metric handling.
                # if step % config.log_every_steps == 0 or is_last_step:
                metric_logger.buffer_and_write_train_metrics(
                    metrics, step, step_time_delta
                )

                if eval_iter:
                    if step % config.eval_every_steps == 0 or is_last_step:
                        # metric_logger.reset_eval_metrics()
                        evaluate(eval_model, eval_iter, step, metric_logger)

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
            logging.log(f"Training stopped: {str(e)}")
        finally:
            metric_logger.flush_metrics_and_cleanup()


def run(argv) -> None:  # noqa: ANN001, D103
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")  # noqa: EM101, TRY003

    logging.log(f"JAX process: {jax.process_index()} / {jax.process_count()}")
    logging.log(f"JAX local devices: {jax.local_devices()}")
    # Load the configuration.
    cfg = base.get_config(FLAGS.config)
    # Train and evaluate
    train_loop(cfg, FLAGS.workdir)


def main() -> None:
    jax.config.config_with_absl()
    app.run(run)
