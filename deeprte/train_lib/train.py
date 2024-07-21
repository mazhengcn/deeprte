import dataclasses
import functools
from collections.abc import Callable

import grain.python as grain
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import tensorflow as tf
from absl import logging
from clu import metric_writers, periodic_actions
from etils import epath
from flax import nnx
from jax.sharding import Mesh

from deeprte.configs import default
from deeprte.input_pipeline import input_pipeline_interface
from deeprte.model.modules import constructor
from deeprte.train_lib import checkpointing, optimizers
from deeprte.train_lib import utils as train_utils
from deeprte.train_lib.checkpointing import emergency_checkpoint_manager


def get_first_step(state):
    with jax.spmd_mode("allow_all"):
        return int(state.step)


def save_checkpoint(
    checkpoint_manager: ocp.Checkpointer,
    step,
    state,
    dataset_type="tfds",
    data_iterator=None,
):
    """Wrapper for saving checkpoint"""
    if isinstance(checkpoint_manager, emergency_checkpoint_manager.CheckpointManager):
        return checkpoint_manager.save(step, args=ocp.args.PyTreeSave(state))

    if dataset_type == "grain":
        return checkpoint_manager.save(
            step,
            args=ocp.args.Composite(
                train_state=ocp.args.PyTreeSave(item=state),
                data_iter=grain.PyGrainCheckpointSave(data_iterator.local_iterator),
            ),
        )
    else:
        return checkpoint_manager.save(
            step,
            args=ocp.args.Composite(train_state=ocp.args.PyTreeSave(item=state)),
        )


def compute_mean_squared_error(predictions, labels):
    return jnp.mean(jnp.square(predictions - labels))


def compute_metrics(predictions, labels):
    mse = compute_mean_squared_error(predictions, labels)
    metrics = {
        "evaluation/mse": mse,
        "evaluation/denominator": jnp.mean(labels**2),
    }
    return metrics


# Primary training / eval / decode step functions.
# -----------------------------------------------------------------------------
def train_step(state, batch, accumulate_grads: Callable = None):
    """Perform a single training step."""

    def loss_fn(params, batch):
        """loss function used for training."""
        module = nnx.merge(state.graphdef, params)
        module.set_attributes(low_memory=False)
        predictions = module(batch)
        labels = batch["psi_label"]
        loss = compute_mean_squared_error(predictions, labels)
        rmse = jnp.sqrt(loss / jnp.mean(labels**2))
        metrics = {"training/loss": loss, "training/rmse": rmse}
        return loss, metrics

    grads_and_metrics_fn = jax.grad(loss_fn, has_aux=True)
    if accumulate_grads:
        grads_and_metrics_fn = accumulate_grads(grads_and_metrics_fn)
    grads, metrics = grads_and_metrics_fn(state.params, batch)
    new_state = state.apply_gradients(grads=grads)

    return new_state, metrics


def eval_step(params: nnx.State, batch, graphdef: nnx.GraphDef):
    """Calculate evaluation metrics on a batch."""
    labels = batch["psi_label"]
    module = nnx.merge(graphdef, params)
    module.set_attributes(low_memory=True)
    predictions = module(batch)
    metrics = compute_metrics(predictions, labels)
    return metrics


def evaluate(jit_eval_step, state, eval_iter):
    """Evaluate the target an return a dictionary with the metrics."""
    logging.info("Gathering evaluation metrics.")
    eval_metrics = {"evaluation/mse": 0.0, "evaluation/denominator": 0.0}
    eval_batch_count = 0
    for eval_batch in eval_iter:
        metrics = jit_eval_step(state.params, eval_batch, state.graphdef)
        eval_metrics = jax.tree.map(lambda x, y: x + float(y), eval_metrics, metrics)
        eval_batch_count += 1
    # eval_metrics = jax.tree.map(lambda x: x / eval_batch_count, eval_metrics)
    denominator = eval_metrics.pop("evaluation/denominator")
    eval_metrics["evaluation/rmse"] = jnp.sqrt(
        eval_metrics["evaluation/mse"] / denominator
    )
    return eval_metrics


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

    # Create metric writers
    writer = metric_writers.create_default_writer(
        workdir, just_logging=jax.process_index() > 0
    )

    # Mesh definition
    # ---------------------------------------------------------------------------
    logging.info("Initializing mesh.")

    devices_array = train_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    if emergency_checkpoint_manager.should_restore_mesh_from_metadata(
        epath.Path(workdir)
    ):
        mesh = emergency_checkpoint_manager.consistent_restore_mesh_from_metadata(
            epath.Path(workdir), mesh
        )

    # Build Model and Optimizer
    # ---------------------------------------------------------------------------
    logging.info("Initializing optimizer, model and checkpointer.")

    learning_rate_dict = {
        "schedule": config.schedule,
        "init_value": config.learning_rate,
        "decay_rate": config.decay_rate,
        "transition_steps": config.transition_steps,
    }

    tx, lr_schedule = optimizers.create_optimizer(
        name="adam",
        total_steps=config.num_train_steps,
        learning_rate=learning_rate_dict,
    )

    if config.enable_emergency_checkpoint:
        abstract_state, _ = train_utils.get_abstract_state(
            constructor, tx, config, init_rng, mesh
        )
        checkpoint_manager = checkpointing.create_orbax_emergency_checkpoint_manager(
            config.local_checkpoint_dir,
            workdir,
            mesh,
            abstract_state,
            config.local_checkpoint_every_steps,
            config.checkpoint_every_steps,
        )
    else:
        logger = checkpointing.setup_checkpoint_logger(config)
        checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
            workdir,
            config.save_checkpoints,
            config.async_checkpointing,
            config.checkpoint_every_steps,
            config.dataset_type,
            logger,
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
    state, state_sharding, train_iter = train_utils.setup_training_state(
        constructor, train_iter, tx, config, init_rng, mesh, checkpoint_manager
    )
    start_step = get_first_step(state)

    if start_step == 0:
        writer.write_hparams(dataclasses.asdict(config))

    # Compile multidevice versions of train/eval/predict step fn.
    # ---------------------------------------------------------------------------
    if config.microsteps:
        accum_grads = functools.partial(
            train_utils.accumulate_gradients_and_metrics,
            microsteps=config.microsteps,
            global_batch_size=config.global_batch_size,
            data_sharding=data_sharding,
        )
        train_step_fn = functools.partial(train_step, accum_grads_fn=accum_grads)
    else:
        train_step_fn = train_step

    jit_train_step = jax.jit(
        train_step_fn,
        in_shardings=(state_sharding, data_sharding),  # type: ignore
        out_shardings=(state_sharding, None),  # type: ignore
        donate_argnums=0,
    )

    if eval_iter:
        jit_eval_step = jax.jit(
            eval_step,
            in_shardings=(state_sharding.params, data_sharding),  # type: ignore
            out_shardings=None,  # type: ignore
            static_argnums=2,
        )

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

            # Shard data to devices and do a training step.
            with jax.profiler.StepTraceAnnotation("train", step_num=step):
                batch = next(train_iter)
                state, train_metrics = jit_train_step(state, batch)

            # Quick indication that training is happening.
            logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
            for h in hooks:
                h(step)

            # Periodic metric handling.
            if step % config.log_every_steps == 0 or is_last_step:
                with report_progress.timed("training_metrics"):
                    logging.info("Gathering training metrics.")
                    train_metrics["learning_rate"] = lr_schedule(step)
                    writer.write_scalars(step, train_metrics)

            if step % config.eval_every_steps == 0 or is_last_step:
                with report_progress.timed("eval"):
                    eval_metrics = evaluate(
                        jit_eval_step=jit_eval_step, state=state, eval_iter=eval_iter
                    )
                    writer.write_scalars(step, eval_metrics)

            # Save a checkpoint on one host after every checkpoint_freq steps.
            is_saving_checkpoint = (
                step % config.checkpoint_every_steps == 0 or is_last_step
            )
            if config.save_checkpoints and is_saving_checkpoint:
                logging.info("Saving checkpoint step %d.", step)
                with report_progress.timed("checkpoint"):
                    save_checkpoint(ckpt_mngr, step, state)
