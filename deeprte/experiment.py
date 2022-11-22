# Copyright 2022 Zheng Ma
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepRTE experiment."""

import functools
import time
from collections.abc import Generator, Mapping, Sequence

import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from absl import logging
from jaxline import experiment
from jaxline import utils as jl_utils

from deeprte import optimizers
from deeprte.model.tf import dataset

OptState = tuple[optax.TraceState, optax.ScaleByScheduleState, optax.ScaleState]
Scalars = Mapping[str, jnp.ndarray]


def _format_logs(prefix, results):
    # f_list for less verbosity; e.g., "4." instead of
    # "array(4., dtype=float32)".
    logging_str = " - ".join(
        [
            f"{k}: {results[k]:.2%}" if k[-2:] == "pe" else f"{k}: {results[k]}"
            for k in sorted(results.keys())
        ]
    )
    logging.info("%s - %s", prefix, logging_str)


class Experiment(experiment.AbstractExperiment):
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
        self, mode: str, init_rng: jnp.ndarray, config: ml_collections.ConfigDict
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

        # Initialize train and eval functions
        self._train_input = None
        self._eval_input = None
        self.eval_fn = None
        self.optimizer = None
        self._lr_schedule = None
        self.update_fn = None

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
        self, global_step: jnp.ndarray, rng: jnp.ndarray, *unused_args, **unused_kwargs
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
        scaled_grads, (loss_scalars, state) = grad_fn(params, state, rng, batch)
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
        self, params: hk.Params, state: hk.State, rng: jnp.ndarray, batch: dataset.Batch
    ) -> tuple[jnp.ndarray, tuple[Scalars, hk.State]]:

        # Get solution_op function
        solution_func = functools.partial(self.solution.apply, params, state, rng)
        # Return loss and loss_scalars dict for logging.
        loss, loss_scalars = self.model.loss(solution_func, batch)
        # Divided by device count since we have summed across all devices
        scaled_loss = loss / jax.local_device_count()

        return scaled_loss, (loss_scalars, state)

    def _initialize_training(self):
        # Less verbose
        # pylint: disable=invalid-name
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
            c.training.num_train_examples / c.training.batch_size * c.training.repeat
        )
        total_steps = c.training.num_epochs * steps_per_epoch

        # Get learning rate schedule.
        self._lr_schedule = optimizers.get_learning_rate_schedule(
            total_batch_size, steps_per_epoch, total_steps, c.optimizer
        )
        # Optimizer
        self.optimizer = optimizers.make_optimizer(c.optimizer, self._lr_schedule)

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
            logging.info("Net parameters: %d", num_params // jax.local_device_count())

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

    def evaluate(self, global_step, rng: jnp.ndarray, **unused_args) -> Scalars:
        """See base class."""
        if not self._evaluating:
            self._initialize_evaluation()

        # Get global step value on the first device for logging.
        global_step_value = jl_utils.get_first(global_step)
        logging.info("Running evaluation at global_step %s...", global_step_value)

        t_0 = time.time()
        # Run evaluation for an epoch
        metrics = self._eval_epoch(self._params, self._state, rng)
        # Covert jnp.ndarry to list to have less verbose.
        metrics = jax.tree_util.tree_map(
            lambda x: x.tolist() if isinstance(x, jnp.ndarray) else x, metrics
        )
        t_diff = time.time() - t_0

        _format_logs(
            f"(Evaluation time {t_diff:.1f}s, global_step {global_step_value})",
            metrics,
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
            metrics = jax.tree_util.tree_map(lambda x: jnp.sum(x[0], axis=0), metrics)
            if summed_metrics is None:
                summed_metrics = metrics
            else:
                summed_metrics = jax.tree_util.tree_map(
                    jnp.add, summed_metrics, metrics
                )

        # Compute mean_metrics
        mean_metrics = jax.tree_util.tree_map(
            lambda x: x / num_examples, summed_metrics
        )

        # Eval metrics dict
        metrics = {}
        # Take sqrt if it is squared
        for k, v in mean_metrics.items():  # pylint: disable=invalid-name
            metrics["eval_" + k] = jnp.sqrt(v) if k.split("_")[-1][0] == "r" else v

        return metrics

    def _eval_fn(
        self, params: hk.Params, state: hk.State, rng: jnp.ndarray, batch: dataset.Batch
    ) -> Scalars:
        """Evaluates a batch."""
        metrics = {}

        solution_func = functools.partial(self.solution.apply, params, state, rng)

        metrics.update(self.model.metrics(solution_func, batch))

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
            return jl_utils.double_buffer_on_gpu(eval_input)
            # return eval_input

        # Evaluation input as a Generator
        self._eval_input = prefetch_and_double_buffer_input

        # We pmap the evaluation function
        self.eval_fn = jax.pmap(self._eval_fn, axis_name="i")

        # Set evaluating state to True after initialization.
        self._evaluating = True

    def _build_eval_input(self) -> Generator[dataset.Batch, None, None]:
        # Split for evaluation
        split = dataset.Split.TEST

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
            self.solution = solution_config.constructor(**solution_config.kwargs)
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
        collocation_sizes: int | Sequence[int],
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

        ds = self._load_data(  # pylint: disable=invalid-name
            split=dataset.Split.TRAIN_AND_VALID,
            is_training=True,
            batch_sizes=[jax.local_device_count(), 1],
            collocation_sizes=1,
            repeat=1,
        )
        dummy_inputs = jax.tree_util.tree_map(lambda x: x.squeeze(), next(ds)["inputs"])

        if jax.local_device_count() == 1:
            dummy_inputs = jax.tree_util.tree_map(lambda x: x[None, ...], dummy_inputs)

        return dummy_inputs
