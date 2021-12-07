import functools
from collections.abc import Generator, Iterator, Mapping, Sequence
from typing import Union

import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow as tf
from absl import flags, logging
from jax.interpreters.batching import batch
from jaxline import experiment
from jaxline import utils as jl_utils

from deeprte import dataset, optimizers
from deeprte.typing import F

FLAGS = flags.FLAGS

OptState = tuple[
    optax.TraceState, optax.ScaleByScheduleState, optax.ScaleState
]
Scalars = Mapping[str, jnp.ndarray]


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

    solution = None
    model = None

    @classmethod
    def from_solution_and_model(cls, solution, model):
        class InitializedSolver(cls):
            cls.solution = solution
            cls.model = model

        return InitializedSolver

    def __init__(
        self,
        mode: str,
        init_rng: jnp.ndarray,
        config: ml_collections.ConfigDict,
    ):
        """Initializes solver."""
        super().__init__(mode=mode, init_rng=init_rng)

        self.mode = mode
        self.init_rng = init_rng
        self.config = config

        # Checkpointed experiment state.
        self._params = None
        self._state = None
        self._opt_state = None

        # Input pipelines.
        self._train_input = None
        self._eval_input = None

        if mode == "train":
            self._initialize_training()
            if self.config.evaluation.interval > 0:
                self._last_evaluation_scalars = {}
                self._initialize_evaluation()
        elif mode == "eval":
            self._initialize_evaluation()
        elif mode == "train_eval_multithreaded":
            self._initialize_training()
            self._initialize_evaluation()
        else:
            raise ValueError(f'Invalid mode: "{mode}"')

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

        if self._train_input is None:
            self._initialize_training()

        batch = next(self._train_input)

        self._params, self._state, self._opt_state, scalars = self.update_fn(
            self._params,
            self._state,
            self._opt_state,
            global_step,
            rng,
            batch,
        )

        scalars = jl_utils.get_first(scalars)

        # Save final checkpoint.
        if self.config.save_final_checkpoint_as_npy:
            global_step_value = jl_utils.get_first(global_step)
            if global_step_value == FLAGS.config.get("training_steps", 1) - 1:

                def f_np(x):
                    return np.array(jax.device_get(jl_utils.get_first(x)))

                np_params = jax.tree_map(f_np, self._params)
                np_state = jax.tree_map(f_np, self._state)
                path_npy = FLAGS.config.checkpoint_dir / "checkpoint.npy"
                with tf.io.gfile.GFile(path_npy, "wb") as fp:
                    np.save(fp, (np_params, np_state))
                logging.info(f"Saved final checkpoint at {path_npy}")

        # Run synchronous evaluation.
        if self.config.evaluation.interval <= 0:
            return scalars

        global_step_value = jl_utils.get_first(global_step)
        if (
            global_step_value % self.config.evaluation.interval != 0
            and global_step_value != FLAGS.config.get("training_steps", 1) - 1
        ):
            return _merge_eval_scalars(scalars, self._last_evaluation_scalars)

        logging.info("Running synchronous evaluation...")
        self._last_evaluation_scalars = self.evaluate(rng)
        logging.info(
            f"[Step {global_step_value}] Eval scalars: "
            f"{self._last_evaluation_scalars}"
        )

        return _merge_eval_scalars(scalars, self._last_evaluation_scalars)

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
        scalars["learning_rate"] = learning_rate

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
        self.update_fn = jax.pmap(
            self._update_fn, axis_name="i", donate_argnums=(0, 1, 2)
        )

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

    #                  _
    #   _____   ____ _| |
    #  / _ \ \ / / _` | |
    # |  __/\ V / (_| | |
    #  \___| \_/ \__,_|_|
    #

    def evaluate(self, rng: jnp.ndarray, **unused_args) -> Scalars:
        """See base class."""

        metrics = self._eval_epoch(self._params, self._state, rng)

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


def _merge_eval_scalars(a, b):
    if b is None:
        return a
    for k, v in b.items():
        a["eval_" + k] = v
    return a
