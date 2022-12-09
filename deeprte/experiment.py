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
from collections.abc import Generator, Mapping

import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections
import optax
from absl import logging
from jaxline import experiment
from jaxline import utils as jl_utils

from deeprte import optimizers
from deeprte.data.pipeline import FeatureDict
from deeprte.model.modules_v2 import DeepRTE
from deeprte.model.tf.input_pipeline import (
    load_tf_data,
    make_device_batch,
    tf_data_to_generator,
)

OptState = tuple[optax.TraceState, optax.ScaleByScheduleState, optax.ScaleState]
Scalars = Mapping[str, jax.Array]


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


class Trainer(experiment.AbstractExperiment):
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
        init_rng: jax.Array,
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

        # Initialize model functions
        self.model = None
        self._construct_model()

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

        self._process_data()

    #  _             _
    # | |_ _ __ __ _(_)_ __
    # | __| '__/ _` | | '_ \
    # | |_| | | (_| | | | | |
    #  \__|_|  \__,_|_|_| |_|
    #

    def step(
        self, global_step: jax.Array, rng: jax.Array, *unused_args, **unused_kwargs
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
        global_step: jax.Array,
        rng: jax.Array,
        batch: FeatureDict,
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
        self,
        params: hk.Params,
        state: hk.State,
        rng: jax.Array,
        batch: FeatureDict,
    ) -> tuple[jax.Array, tuple[Scalars, hk.State]]:
        # Get solution_op function
        rte_model_fn = functools.partial(
            self.model.apply,
            params,
            state,
            rng,
            is_training=True,
            compute_loss=True,
            compute_metrics=False,
        )
        # Return loss and loss_scalars dict for logging.
        (loss, ret), state = rte_model_fn(batch)
        # Divided by device count since we have summed across all devices
        loss_scalars = ret["loss"]
        scaled_loss = loss / jax.local_device_count()

        return scaled_loss, (loss_scalars, state)

    def _build_train_input(self) -> Generator[FeatureDict, None, None]:
        """Build train input as generator/iterator."""
        c = self.config.dataset
        batch_sizes = make_device_batch(
            c.train.batch_size,
            jax.device_count(),
        )
        train_ds = tf_data_to_generator(
            tf_data=self.tf_data,
            is_training=True,
            batch_sizes=batch_sizes,
            split_rate=c.data_split.train_validation_split_rate,
            collocation_sizes=c.train.collocation_sizes,
            repeat=c.train.repeat,
            buffer_size=c.buffer_size,
            threadpool_size=c.threadpool_size,
            max_intra_op_parallelism=c.max_intra_op_parallelism,
        )
        return train_ds

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
        total_batch_size = c.dataset.train.batch_size
        # NOTE: Since we may have repeat number for the same batch
        # with different collocation points, stpes_per_epoch should be
        # multiplied by repeat.

        steps_per_epoch = (
            c.dataset.data_split.num_train_samples
            / c.dataset.train.batch_size
            * c.dataset.train.repeat
        )
        total_steps = c.training.num_epochs * steps_per_epoch

        # Get learning rate schedule.
        self._lr_schedule = optimizers.get_learning_rate_schedule(
            total_batch_size, steps_per_epoch, total_steps, c.training.optimizer
        )
        # Optimizer
        self.optimizer = optimizers.make_optimizer(
            c.training.optimizer,
            self._lr_schedule,
        )

        # Initialize net if no params available.
        if self._params is None:
            logging.info("Initializing parameters.")

            # Pmap initial functions
            # init_net = jax.pmap(lambda *a: self.solution.init(*a))
            init_net = functools.partial(
                self.model.init,
                is_training=True,
                compute_loss=False,
                compute_metrics=False,
            )
            init_net = jax.pmap(init_net)
            init_opt = jax.pmap(self.optimizer.init)

            # Init uses the same RNG key on all hosts+devices to ensure
            # everyone computes the same initial state.
            init_rng = jl_utils.bcast_local_devices(self.init_rng)

            # Load initial inputs
            dummy_inputs = self._build_dummy_input()
            self._params, self._state = init_net(init_rng, dummy_inputs)
            self._opt_state = init_opt(self._params)

            # Log total number of parameters
            num_params = hk.data_structures.tree_size(self._params)
            logging.info(
                "Net parameters: %d",
                num_params // jax.local_device_count(),
            )

        # NOTE: We "donate" the `params, state, opt_state` arguments which
        # allows JAX (on some backends) to reuse the device memory associated
        # with these inputs to store the outputs of our function (which also
        # start with `params, state, opt_state`).
        self.update_fn = jax.pmap(self._update_fn, axis_name="i")

        # Set training state to True after initialization
        self._training = True

    #                  _
    #   _____   ____ _| |
    #  / _ \ \ / / _` | |
    # |  __/\ V / (_| | |
    #  \___| \_/ \__,_|_|
    #

    def evaluate(self, global_step, rng: jax.Array, **unused_args) -> Scalars:
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
            lambda x: x.tolist() if isinstance(x, jax.Array) else x, metrics
        )
        t_diff = time.time() - t_0

        _format_logs(
            f"(Evaluation time {t_diff:.1f}s, global_step {global_step_value})",
            metrics,
        )

        return metrics

    def _eval_epoch(
        self, params: hk.Params, state: hk.State, rng: jax.Array
    ) -> Scalars:
        """Evaluates an epoch."""
        num_examples = 0.0
        summed_metrics = None

        for batch in self._eval_input():
            # Account for pmaps
            num_examples += jnp.prod(jnp.array(batch["psi_label"].shape[:2]))
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
        self,
        params: hk.Params,
        state: hk.State,
        rng: jax.Array,
        batch: FeatureDict,
    ) -> Scalars:
        """Evaluates a batch."""
        metrics = {}

        eval_func = functools.partial(
            self.model.apply,
            params,
            state,
            rng,
            is_training=False,
            compute_loss=False,
            compute_metrics=True,
        )

        # metrics.update(self.model.metrics(eval_func, batch))

        ret, state = eval_func(batch)
        # Divided by device count since we have summed across all devices
        metrics.update(ret["metrics"])

        # NOTE: Returned values will be summed and finally divided
        # by num_samples.
        return jax.lax.psum(metrics, axis_name="i")

    def _initialize_evaluation(self):
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

    def _build_eval_input(self) -> Generator[FeatureDict, None, None]:
        c = self.config.dataset
        batch_sizes = make_device_batch(
            c.validation.batch_size,
            jax.device_count(),
        )
        val_ds = tf_data_to_generator(
            tf_data=self.tf_data,
            is_training=False,
            batch_sizes=batch_sizes,
            split_rate=c.data_split.train_validation_split_rate,
            buffer_size=c.buffer_size,
            threadpool_size=c.threadpool_size,
            max_intra_op_parallelism=c.max_intra_op_parallelism,
        )
        return val_ds

    def _process_data(
        self,
    ):
        """dataset loading."""
        c = self.config.dataset
        self.tf_data = load_tf_data(
            data_path=c.data_path,
            pre_shuffle=c.pre_shuffle,
            pre_shuffle_seed=c.pre_shuffle_seed,
            is_split_test_samples=c.data_split.is_split_datasets,
            num_test_samples=c.data_split.num_test_samples,
            save_path=c.data_split.save_path,
        )

    def _build_dummy_input(self) -> tuple[jax.Array]:
        """Load dummy data for initializing network parameters."""

        ds = tf_data_to_generator(
            tf_data=self.tf_data,
            is_training=True,
            batch_sizes=[jax.local_device_count(), 1],
            collocation_sizes=1,
            repeat=1,
        )

        dummy_inputs = next(ds)

        return dummy_inputs

    def _construct_model(self):
        # Create solution instance.
        if not self.model:

            def _forward_fn(batch, is_training, compute_loss, compute_metrics):
                model = DeepRTE(self.config.model, self.config.model)
                return model(
                    batch,
                    is_training=is_training,
                    compute_loss=compute_loss,
                    compute_metrics=compute_metrics,
                )

            self.model = hk.transform_with_state(_forward_fn)
        else:
            raise ValueError("Model instance is already initialized.")
