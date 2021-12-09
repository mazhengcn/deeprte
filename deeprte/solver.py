import datetime
import functools
import os
import pathlib
import signal
import threading
import time
from collections.abc import Generator, Mapping, Sequence
from typing import Union

import dill
import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf
from absl import flags, logging
from jaxline import experiment, platform
from jaxline import utils as jl_utils

from deeprte import dataset, optimizers

FLAGS = flags.FLAGS

OptState = tuple[
    optax.TraceState, optax.ScaleByScheduleState, optax.ScaleState
]
Scalars = Mapping[str, jnp.ndarray]


def _log_results(prefix, results):
    logging_str = ", ".join(
        [
            "{}={:.4f}".format(k, float(results[k]))
            for k in sorted(results.keys())
        ]
    )
    logging.info("%s: %s", prefix, logging_str)


class Solver(experiment.AbstractExperiment):
    """RTE solver."""

    # A map from object properties that will be checkpointed to their name
    # in a checkpoint. Currently we assume that these are all sharded
    # device arrays.
    CHECKPOINT_ATTRS = {
        "_params": "params",
        "_state": "state",
        "_opt_state": "opt_state",
    }

    def __init__(
        self,
        mode: str,
        init_rng: jnp.ndarray,
        config: ml_collections.ConfigDict,
    ):
        """Initializes solver."""
        super().__init__(mode=mode, init_rng=init_rng)

        if mode not in ("train", "eval", "train_eval_multithreaded"):
            raise ValueError(f"Invalid mode {mode}.")

        self.mode = mode
        self.init_rng = init_rng
        self.config = config

        # Checkpointed experiment state.
        self._params = None
        self._state = None
        self._opt_state = None

        # Initialize solution and model functions
        self.solution = None
        self.model = None
        self._init_solution_and_model()

        # Track what has started
        self._training = False
        self._evaluating = False

    #  _             _
    # | |_ _ __ __ _(_)_ __
    # | __| '__/ _` | | '_ \
    # | |_| | | (_| | | | | |
    #  \__|_|  \__,_|_|_| |_|
    #

    def step(
        self,
        global_step: jnp.ndarray,
        rng: jnp.ndarray,
        *unused_args,
        **unused_kwargs,
    ) -> Scalars:
        """See base class."""
        if not self._training:
            self._initialize_training()

        # Get next batch
        batch = next(self._train_input)

        # Update parameters
        self._params, self._state, self._opt_state, scalars = self.update_fn(
            self._params,
            self._state,
            self._opt_state,
            global_step,
            rng,
            batch,
        )

        # We only return the loss scalars on the first devict for logging
        scalars = jl_utils.get_first(scalars)

        return scalars

    def _update_fn(
        self,
        params: hk.Params,
        state: hk.State,
        opt_state: OptState,
        global_step: jnp.ndarray,
        rng: jnp.ndarray,
        batch: dataset.Batch,
    ) -> tuple[hk.Params, hk.State, optax.OptState, Scalars]:
        # Logging dict.
        scalars = {}

        # Gradient function w.r.t. params
        grad_fn = jax.grad(self._loss, has_aux=True)
        # Compute loss and gradients.
        scaled_grads, (loss_scalars, state) = grad_fn(
            params, state, rng, batch
        )
        grads = jax.lax.psum(scaled_grads, axis_name="i")

        # Grab the learning rate to log before performing the step.
        learning_rate = self._lr_schedule(global_step)
        # scalars["learning_rate"] = learning_rate

        # Update params
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Update scalars dict
        loss_scalars = {f"train_{k}": v for k, v in loss_scalars.items()}
        scalars.update(loss_scalars)
        scalars = jax.lax.pmean(scalars, axis_name="i")

        return params, state, opt_state, scalars

    def _loss(
        self,
        params: hk.Params,
        state: hk.State,
        rng: jnp.ndarray,
        batch: dataset.Batch,
    ) -> tuple[jnp.ndarray, tuple[Scalars, hk.State]]:

        # Get solution_op function
        solution_fn = functools.partial(
            self.solution.apply, params, state, rng, is_training=True
        )
        # Return loss and loss_scalars dict for logging.
        loss, loss_scalars = self.model.loss(solution_fn, batch)
        # Divided by device count since we have summed across all devices
        scaled_loss = loss / jax.local_device_count()

        return scaled_loss, (loss_scalars, state)

    def _initialize_training(self):
        # Less verbose
        c = self.config

        # Performs prefetching of elements from an iterable in a separate thread.
        train_input = jl_utils.py_prefetch(self._build_train_input)
        # This keeps two batches per-device in memory at all times, allowing
        # h2d transfers to overlap with execution (see b/173483287 for details).
        self._train_input = jl_utils.double_buffer_on_gpu(train_input)

        # Total batch size
        total_batch_size = c.training.batch_size
        # NOTE: Since we may have repeat number for the same batch
        # with different collocation points, stpes_per_epoch should be
        # multiplied by repeat.
        steps_per_epoch = (
            c.training.num_train_examples
            / c.training.batch_size
            * c.training.repeat
        )
        total_steps = c.training.num_epochs * steps_per_epoch

        # Get learning rate schedule.
        self._lr_schedule = optimizers.get_learning_rate_schedule(
            total_batch_size, steps_per_epoch, total_steps, c.optimizer
        )
        # Optimizer
        self.optimizer = optimizers.make_optimizer(
            c.optimizer, self._lr_schedule
        )

        # Initialize net if no params available.
        if self._params is None:
            logging.info("Initializing parameters.")

            # Pmap initial functions
            # init_net = jax.pmap(lambda *a: self.solution.init(*a))
            init_net = jax.pmap(self.solution.init)
            init_opt = jax.pmap(self.optimizer.init)

            # Init uses the same RNG key on all hosts+devices to ensure
            # everyone computes the same initial state.
            init_rng = jl_utils.bcast_local_devices(self.init_rng)

            # Load initial inputs
            dummy_inputs = self._load_dummy_data()
            self._params, self._state = init_net(init_rng, *dummy_inputs)
            self._opt_state = init_opt(self._params)

            # Log total number of parameters
            num_params = hk.data_structures.tree_size(self._params)
            logging.info(
                f"Net parameters: {num_params / jax.local_device_count()}"
            )

        # NOTE: We "donate" the `params, state, opt_state` arguments which
        # allows JAX (on some backends) to reuse the device memory associated
        # with these inputs to store the outputs of our function (which also
        # start with `params, state, opt_state`).
        self.update_fn = jax.pmap(self._update_fn, axis_name="i")

        # Set training state to True after initialization
        self._training = True

    def _build_train_input(self) -> Generator[dataset.Batch, None, None]:
        """Build train input as generator/iterator."""

        # Get number of devices (GPUs/TPUs).
        num_devices = jax.local_device_count()

        # Global batch size (without multiplied by repeat)
        global_batch_size = self.config.training.batch_size
        # Batch size on each device
        per_device_batch_size, ragged = divmod(global_batch_size, num_devices)
        # Raise error if not divisible
        if ragged:
            raise ValueError(
                f"Global batch size {global_batch_size} must be divisible by "
                f"number of devices {num_devices}"
            )
        # Get collocation sizes (number of sampled grid points)
        collocation_sizes = self.config.training.collocation_sizes
        # Get repeat
        # number of steps to repeat with different collocation points for a batch
        repeat = self.config.training.repeat

        # Split for training
        split = dataset.Split.TRAIN_AND_VALID

        return self._load_data(
            split=split,
            is_training=True,
            batch_sizes=[num_devices, per_device_batch_size],
            collocation_sizes=collocation_sizes,
            repeat=repeat,
        )

    #                  _
    #   _____   ____ _| |
    #  / _ \ \ / / _` | |
    # |  __/\ V / (_| | |
    #  \___| \_/ \__,_|_|
    #

    def evaluate(
        self, global_step, rng: jnp.ndarray, **unused_args
    ) -> Scalars:
        """See base class."""
        if not self._evaluating:
            self._initialize_evaluation()

        # Current time
        start_time = time.time()

        # Run evaluation for an epoch
        metrics = self._eval_epoch(self._params, self._state, rng)
        # Covert jnp.ndarry to list to have less verbose.
        metrics = jax.tree_map(
            lambda x: x.tolist() if isinstance(x, jnp.ndarray) else x, metrics
        )

        # End time for measure running time
        time_sec = time.time() - start_time

        # Get global step value on the first device for logging.
        global_step_value = jl_utils.get_first(global_step)

        logging.info(
            f"[Evaluation at step {global_step_value} takes {time_sec}], Eval metrics: {metrics}"
        )

        return metrics

    def _eval_epoch(
        self, params: hk.Params, state: hk.State, rng: jnp.ndarray
    ) -> Scalars:
        """Evaluates an epoch."""
        num_examples = 0.0
        summed_metrics = None

        for batch in self._eval_input():
            # Account for pmaps
            num_examples += np.prod(batch["labels"].shape[:2])
            metrics = self.eval_fn(params, state, rng, batch)
            # Accumulate the sum of scalars for each step.
            metrics = jax.tree_map(lambda x: jnp.sum(x[0], axis=0), metrics)
            if summed_metrics is None:
                summed_metrics = metrics
            else:
                summed_metrics = jax.tree_multimap(
                    jnp.add, summed_metrics, metrics
                )

        mean_metrics = jax.tree_map(lambda x: x / num_examples, summed_metrics)
        # Format as percentage of sqrt for some metrics.
        for k, v in mean_metrics.items():
            if "pr" in k:
                mean_metrics[k] = jnp.sqrt(v) * 100.0

        return mean_metrics

    def _eval_fn(
        self,
        params: hk.Params,
        state: hk.State,
        rng: jnp.ndarray,
        batch: dataset.Batch,
    ) -> Scalars:
        """Evaluates a batch."""
        metrics = {}

        predict_fn = functools.partial(
            self.solution.apply, params, state, rng, is_training=False
        )

        metrics.update(self.model.metrics(predict_fn, batch))

        # NOTE: Returned values will be summed and finally divided
        # by num_samples.
        return jax.lax.psum(metrics, axis_name="i")

    def _initialize_evaluation(self):

        # Initialize prefetch and double buffer function
        def prefetch_and_double_buffer_input():
            # Performs prefetching of elements from an iterable in a separate thread.
            eval_input = jl_utils.py_prefetch(self._build_eval_input)
            # This keeps two batches per-device in memory at all times, allowing
            # h2d transfers to overlap with execution (see b/173483287 for details).
            # return jl_utils.double_buffer_on_gpu(eval_input)
            return eval_input

        # Evaluation input as a Generator
        self._eval_input = prefetch_and_double_buffer_input

        # We pmap the evaluation function
        self.eval_fn = jax.pmap(self._eval_fn, axis_name="i")

        # Set evaluating state to True after initialization.
        self._evaluating = True

    def _build_eval_input(self) -> Generator[dataset.Batch, None, None]:
        # Split for evaluation
        split = dataset.Split.VALID

        # Global batch size
        global_batch_size = self.config.evaluation.batch_size
        # Number of local devices
        num_devices = jax.local_device_count()
        # Batch size on each device
        per_device_batch_size, ragged = divmod(global_batch_size, num_devices)
        # Raise error if not divisible
        if ragged:
            raise ValueError(
                f"Global batch size {global_batch_size} must be divisible by "
                f"number of devices {num_devices}"
            )

        return self._load_data(
            split=split,
            is_training=False,
            batch_sizes=[num_devices, per_device_batch_size],
            collocation_sizes=None,
            repeat=None,
        )

    def _init_solution_and_model(self):
        # Create solution instance.
        solution_config = self.config.solution
        if not self.solution:
            self.solution = solution_config.constructor(
                **solution_config.kwargs
            )
        else:
            raise ValueError("Solution instance is already initialized.")

        # Create model instance
        model_config = self.config.model
        if not self.model:
            self.model = model_config.constructor(**model_config.kwargs)
        else:
            raise ValueError("Model instance is already initialized.")

    def _load_data(
        self,
        split: dataset.Split,
        is_training: bool,
        batch_sizes: Sequence[int],
        collocation_sizes: Union[int, Sequence[int]],
        repeat: int,
    ) -> Generator[dataset.Batch, None, None]:
        """Wrapper for dataset loading."""

        return dataset.load(
            data_path=self.config.dataset.data_path,
            split=split,
            is_training=is_training,
            batch_sizes=batch_sizes,
            collocation_sizes=collocation_sizes,
            repeat=repeat,
            buffer_size=self.config.dataset.buffer_size,
            threadpool_size=self.config.dataset.threadpool_size,
            max_intra_op_parallelism=self.config.dataset.max_intra_op_parallelism,
        )

    def _load_dummy_data(self) -> tuple[jnp.ndarray]:
        """Load dummy data for initializing network parameters."""

        ds = self._load_data(
            split=dataset.Split.TRAIN_AND_VALID,
            is_training=True,
            batch_sizes=[jax.local_device_count(), 1],
            collocation_sizes=1,
            repeat=1,
        )
        dummy_inputs = jax.tree_map(lambda x: x.squeeze(), next(ds)["inputs"])

        if jax.local_device_count() == 1:
            dummy_inputs = jax.tree_map(lambda x: x[None, ...], dummy_inputs)

        return dummy_inputs

    def save_final_checkpoint(self, global_step_value: int):
        # Save final checkpoint.
        if global_step_value == FLAGS.config.get("training_steps", 1) - 1:

            def f_np(x):
                return np.array(jax.device_get(jl_utils.get_first(x)))

            np_params = jax.tree_map(f_np, self._params)
            np_state = jax.tree_map(f_np, self._state)
            path_npy = FLAGS.config.checkpoint_dir / "checkpoint.npy"
            with tf.io.gfile.GFile(path_npy, "wb") as fp:
                np.save(fp, (np_params, np_state))
            logging.info(f"Saved final checkpoint at {path_npy}")


def _get_step_date_label(global_step):
    # Date removing microseconds.
    date_str = datetime.datetime.now().isoformat().split(".")[0]
    return f"step_{global_step}_{date_str}"


def _restore_state_to_in_memory_checkpointer(restore_path):
    """Initializes experiment state from a checkpoint."""
    if not isinstance(restore_path, pathlib.Path):
        restore_path = pathlib.Path(restore_path)

    # Load pretrained experiment state.
    python_state_path = restore_path / "checkpoint.dill"
    with open(python_state_path, "rb") as f:
        pretrained_state = dill.load(f)
    logging.info(f"Restored checkpoint from {python_state_path}")

    # Assign state to a dummy experiment instance for the in-memory checkpointer,
    # broadcasting to devices.
    dummy_solver = Solver(
        mode="train", init_rng=0, config=FLAGS.config.experiment_kwargs.config
    )
    for attribute, key in Solver.CHECKPOINT_ATTRS.items():
        setattr(
            dummy_solver,
            attribute,
            jl_utils.bcast_local_devices(pretrained_state[key]),
        )

    jaxline_state = dict(
        global_step=pretrained_state["global_step"],
        experiment_module=dummy_solver,
    )
    snapshot = jl_utils.SnapshotNT(0, jaxline_state)

    # Finally, seed the jaxline `utils.InMemoryCheckpointer` global dict.
    jl_utils.GLOBAL_CHECKPOINT_DICT["latest"] = jl_utils.CheckpointNT(
        threading.local(), [snapshot]
    )


def _save_state_from_in_memory_checkpointer(
    save_path, experiment_class: experiment.AbstractExperiment
):
    """Saves experiment state to a checkpoint."""
    if not isinstance(save_path, pathlib.Path):
        save_path = pathlib.Path(save_path)

    logging.info("Saving model.")
    for checkpoint_name, checkpoint in jl_utils.GLOBAL_CHECKPOINT_DICT.items():
        if not checkpoint.history:
            logging.info(f'Nothing to save in "{checkpoint_name}"')
            continue

        pickle_nest = checkpoint.history[-1].pickle_nest
        global_step = pickle_nest["global_step"]

        state_dict = {"global_step": global_step}
        for attribute, key in experiment_class.CHECKPOINT_ATTRS.items():
            state_dict[key] = jl_utils.get_first(
                getattr(pickle_nest["experiment_module"], attribute)
            )
        save_dir = (
            save_path / checkpoint_name / _get_step_date_label(global_step)
        )

        python_state_path = save_dir / "checkpoint.dill"
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(python_state_path, "wb") as f:
            dill.dump(state_dict, f)

        logging.info(
            f'Saved "{checkpoint_name}" checkpoint to {python_state_path}'
        )


def _setup_signals(save_model_fn):
    """Sets up a signal for model saving."""
    # Save a model on Ctrl+C.
    def sigint_handler(unused_sig, unused_frame):
        # Ideally, rather than saving immediately, we would then "wait" for a good
        # time to save. In practice this reads from an in-memory checkpoint that
        # only saves every 30 seconds or so, so chances of race conditions are very
        # small.
        save_model_fn()
        logging.info(r"Use `Ctrl+\` to save and exit.")

    # Exit on `Ctrl+\`, saving a model.
    prev_sigquit_handler = signal.getsignal(signal.SIGQUIT)

    def sigquit_handler(unused_sig, unused_frame):
        # Restore previous handler early, just in case something goes wrong in the
        # next lines, so it is possible to press again and exit.
        signal.signal(signal.SIGQUIT, prev_sigquit_handler)
        save_model_fn()
        logging.info(r"Exiting on `Ctrl+\`")

        # Re-raise for clean exit.
        os.kill(os.getpid(), signal.SIGQUIT)

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGQUIT, sigquit_handler)


def main(experiment_class, argv):

    # Maybe restore a model.
    restore_path = FLAGS.config.restore_path
    if restore_path:
        _restore_state_to_in_memory_checkpointer(restore_path)

    # Maybe save a model.
    save_dir = os.path.join(FLAGS.config.checkpoint_dir, "models")
    if FLAGS.config.one_off_evaluate:
        save_model_fn = (
            lambda: None
        )  # No need to save checkpoint in this case.
    else:
        save_model_fn = functools.partial(
            _save_state_from_in_memory_checkpointer, save_dir, experiment_class
        )
    _setup_signals(
        save_model_fn
    )  # Save on Ctrl+C (continue) or Ctrl+\ (exit).

    try:
        platform.main(experiment_class, argv)
    finally:
        save_model_fn()  # Save at the end of training or in case of exception.
