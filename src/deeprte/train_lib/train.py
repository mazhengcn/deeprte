import dataclasses

import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import yaml
from absl import logging
from clu import metric_writers, periodic_actions
from flax import nnx
from jax.sharding import Mesh

from deeprte.configs import default
from deeprte.input_pipeline import input_pipeline_interface
from deeprte.model import features
from deeprte.model.model import DeepRTE
from deeprte.train_lib import checkpointing, optimizers
from deeprte.train_lib import utils as train_utils
from deeprte.train_lib.checkpointing import save_checkpoint
from deeprte.train_lib.metrics import RelativeError


def loss_fn(model: nnx.Module, batch):
    """Loss function used for training."""
    labels = batch["psi_label"]
    predictions = model(batch)
    return jnp.mean((predictions - labels) ** 2)


@nnx.jit
def train_step(
    model: nnx.Module, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch
):
    """Perform a single training step."""
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, batch)
    optimizer.update(grads)
    metrics.update(loss=loss, mean_squared_labels=jnp.mean(batch["psi_label"] ** 2))


def accumulate_gradent(micro_steps: int, global_batch_size: int):
    if not micro_steps or micro_steps < 0:
        return train_step

    batch_size_per_device = global_batch_size // jax.device_count()
    assert batch_size_per_device % micro_steps == 0

    def accumulated_train_step(model, optimizer, metrics, batch):
        batch = jax.tree.map(
            lambda x: x.reshape((-1, micro_steps) + x.shape[1:]), batch
        )
        for i in range(micro_steps):
            micro_batch = jax.tree.map(lambda x: x[:, i], batch)
            train_step(model, optimizer, metrics, micro_batch)

    return accumulated_train_step


@nnx.jit
def eval_step(model: nnx.Module, metrics: nnx.MultiMetric, batch):
    """Calculate evaluation metrics on a batch."""
    loss = loss_fn(model, batch)
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
    tx = optax.MultiSteps(tx, every_k_schedule=config.micro_steps)

    accumulated_train_step = accumulate_gradent(
        config.micro_steps, config.global_batch_size
    )

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
        rmse=RelativeError("loss", "mean_squared_labels"),
    )

    # Create metric writers
    writer = metric_writers.create_default_writer(
        workdir, just_logging=jax.process_index() > 0
    )

    # Load Dataset
    # ---------------------------------------------------------------------------
    logging.info("Initializing dataset.")
    train_iter, eval_iter = input_pipeline_interface.create_data_iterator(config, mesh)

    # Initialize train state
    # ---------------------------------------------------------------------------
    logging.info("Initializing train state.")
    model, optimizer, train_iter = train_utils.setup_training_state(
        model_class=DeepRTE,
        config=config,
        rng=init_rng,
        tx=tx,
        mesh=mesh,
        data_iterator=train_iter,
        checkpoint_manager=checkpoint_manager,
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
                accumulated_train_step(model, optimizer, metrics, batch)

            # Quick indication that training is happening.
            logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
            for h in hooks:
                h(step)

            # Periodic metric handling.
            if step % config.log_every_steps == 0 or is_last_step:
                with report_progress.timed("training_metrics"):
                    logging.info("Gathering training metrics.")
                    writer.write_scalars(step, metrics.compute())
                metrics.reset()

            if eval_iter:
                if step % config.eval_every_steps == 0 or is_last_step:
                    with report_progress.timed("eval"):
                        evaluate(model, metrics, eval_iter)
                        writer.write_scalars(step, metrics.compute())
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
