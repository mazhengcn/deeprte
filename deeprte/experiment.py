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

"""DeepRTE trainer."""

import functools
import time
from collections.abc import Generator, Mapping

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from absl import logging
from jaxline import experiment
from jaxline import utils as jl_utils
from ml_collections import config_dict

from deeprte import input_pipeline, optimizers, utils
from deeprte.data.pipeline import FeatureDict
from deeprte.model import modules

Scalars = dict[str, jax.Array]


class Experiment(experiment.AbstractExperiment):
    """RTE Trainer."""

    # A map from object properties that will be checkpointed to their name
    # in a checkpoint. Currently we assume that these are all sharded
    # device arrays.
    CHECKPOINT_ATTRS = {
        "_params": "params",
        "_state": "state",
        "_opt_state": "opt_state",
    }

    def __init__(
        self, mode: str, init_rng: chex.PRNGKey, config: config_dict.ConfigDict
    ) -> None:
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
        def _forward_fn(*args, **kwargs):
            model = modules.DeepRTE(self.config.model)
            return model(*args, **kwargs)

        self.model = hk.transform_with_state(_forward_fn)

        # Initialize train and eval functions
        self._train_input = None
        self._eval_input = None
        self._lr_schedule = None

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
        self, global_step: chex.Array, rng: chex.PRNGKey, *unused_args, **unused_kwargs
    ) -> Scalars:
        """See base class."""
        if not self._training:
            self._initialize_training()

        # Get next batch
        batch = next(self._train_input)

        # Update parameters
        self._params, self._state, self._opt_state, scalars = self.train_fn(
            self._params, self._state, self._opt_state, global_step, rng, batch
        )

        # We only return the loss scalars on the first devict for logging
        scalars = jl_utils.get_first(scalars)

        return scalars

    def _train_fn(
        self,
        params: hk.Params,
        state: hk.State,
        opt_state: optax.OptState,
        global_step: chex.Array,
        rng: chex.PRNGKey,
        batch: Mapping[str, chex.Array],
    ) -> tuple[hk.Params, hk.State, optax.OptState, Scalars]:
        # Logging dict.
        scalars = {}

        def loss(params, batch):
            (total_loss, ret), out_state = self.model.apply(
                params,
                state,
                rng,
                batch,
                is_training=True,
                compute_loss=True,
                compute_metrics=False,
            )
            loss_scalars = ret["loss"]
            scaled_loss = total_loss / jax.local_device_count()
            return scaled_loss, (loss_scalars, out_state)

        # Gradient function w.r.t. params
        scaled_grads, (loss_scalars, new_state) = self._accum_grads(
            jax.grad(loss, has_aux=True), params, batch
        )
        # Compute loss and gradients.
        # scaled_grads, (loss_scalars, new_state) = grad_fn(params, batch)
        grads = jax.lax.psum(scaled_grads, axis_name="i")

        # Grab the learning rate to log before performing the step.
        learning_rate = self._lr_schedule(global_step)
        scalars["learning_rate"] = learning_rate

        # Update params
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        # Update scalars dict
        loss_scalars = {f"train_{k}": v for k, v in loss_scalars.items()}
        scalars.update(loss_scalars)
        scalars = jax.lax.pmean(scalars, axis_name="i")

        return new_params, new_state, new_opt_state, scalars

    def _build_train_input(self) -> Generator[FeatureDict, None, None]:
        """Build train input as generator/iterator."""
        c = self.config
        global_batch_size = c.training.batch_size
        per_device_batch_size, ragged = divmod(
            global_batch_size, jax.local_device_count()
        )
        # Raise error if not divisible
        if ragged:
            raise ValueError(
                f"Global batch size {global_batch_size} must be divisible by "
                f"number of devices {jax.local_device_count()}"
            )
        return input_pipeline.load(
            name=c.dataset.name,
            split=input_pipeline.Split.TRAIN,
            split_percentage=c.dataset.split_percentage,
            tfds_dir=c.dataset.tfds_dir,
            is_training=True,
            batch_sizes=[jax.local_device_count(), per_device_batch_size],
            collocation_sizes=c.training.collocation_sizes,
            batch_repeat=c.training.batch_repeat,
        )

    def _initialize_training(self) -> None:
        # Less verbose
        c = self.config

        # Performs prefetching of elements from an iterable
        # in a separate thread.
        train_input = jl_utils.py_prefetch(self._build_train_input)
        # This keeps two batches per-device in memory at all times, allowing
        # h2d transfers to overlap with execution.
        self._train_input = jl_utils.double_buffer_on_gpu(train_input)

        global_batch_size = c.training.batch_size
        per_device_batch_size, ragged = divmod(
            global_batch_size, jax.local_device_count()
        )
        # Raise error if not divisible
        if ragged:
            raise ValueError(
                f"Global batch size {global_batch_size} must be divisible by "
                f"number of devices {jax.local_device_count()}"
            )
        self._accum_grads = functools.partial(
            utils.accumulate_gradient,
            batch_size=per_device_batch_size,
            accum_steps=c.training.accum_grads_steps,
        )

        # NOTE: Since we may have repeat number for the same batch
        # with different collocation points, stpes_per_epoch should be
        # multiplied by repeat.
        steps_per_epoch = (
            c.dataset.num_train_examples
            / c.training.batch_size
            * c.training.batch_repeat
        )
        total_steps = c.training.num_epochs * steps_per_epoch
        # Get learning rate schedule.
        self._lr_schedule = optimizers.get_learning_rate_schedule(
            global_batch_size,
            steps_per_epoch,
            total_steps,
            c.optimizer,
        )
        # Optimizer
        self.optimizer = optimizers.make_optimizer(c.optimizer, self._lr_schedule)

        # Initialize net if no params available.
        if self._params is None:
            logging.info("Initializing parameters.")

            # Pmap initial functions
            init_net = jax.pmap(lambda *a: self.model.init(*a, is_training=True))
            init_opt = jax.pmap(self.optimizer.init)

            # Init uses the same RNG key on all hosts+devices to ensure
            # everyone computes the same initial state.
            init_rng = jl_utils.bcast_local_devices(self.init_rng)

            # Load initial inputs
            batch = next(self._train_input)
            self._params, self._state = init_net(init_rng, batch)
            self._opt_state = init_opt(self._params)

            # Log total number of parameters
            num_params = hk.data_structures.tree_size(self._params)
            logging.info("Net parameters: %d", num_params // jax.local_device_count())
        # NOTE: We "donate" the `params, state, opt_state` arguments which
        # allows JAX (on some backends) to reuse the device memory associated
        # with these inputs to store the outputs of our function (which also
        # start with `params, state, opt_state`).
        self.train_fn = jax.pmap(self._train_fn, axis_name="i")

        # Set training state to True after initialization
        self._training = True

    #                  _
    #   _____   ____ _| |
    #  / _ \ \ / / _` | |
    # |  __/\ V / (_| | |
    #  \___| \_/ \__,_|_|
    #

    def evaluate(
        self, global_step: chex.Array, rng: chex.PRNGKey, **unused_args
    ) -> Scalars:
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

        (
            logging.info(
                "Evaluation time: %.1f at global_step: %d, %s",
                t_diff,
                global_step_value,
                metrics,
            ),
        )

        return metrics

    def _eval_epoch(
        self, params: hk.Params, state: hk.State, rng: chex.PRNGKey
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
        for k, v in mean_metrics.items():
            metrics["eval_" + k] = (
                100 * jnp.sqrt(v) if k.split("_")[-1][0] == "r" else v
            )

        return metrics

    def _eval_fn(
        self,
        params: hk.Params,
        state: hk.State,
        rng: chex.PRNGKey,
        batch: Mapping[str, chex.Array],
    ) -> Scalars:
        """Evaluates a batch."""
        outputs, state = self.model.apply(
            params, state, rng, batch, is_training=False, compute_metrics=True
        )

        # NOTE: Returned values will be summed and finally divided
        # by num_samples.
        return jax.lax.psum(outputs["metrics"], axis_name="i")

    def _initialize_evaluation(self) -> None:
        def prefetch_and_double_buffer_input():
            # Performs prefetching of elements from an iterable
            # in a separate thread.
            eval_input = jl_utils.py_prefetch(self._build_eval_input)
            # This keeps two batches per-device in memory at all times,
            # allowing h2d transfers to overlap with execution.
            return jl_utils.double_buffer_on_gpu(eval_input)
            # return eval_input

        # Evaluation input as a Generator
        self._eval_input = prefetch_and_double_buffer_input

        # We pmap the evaluation function
        self.eval_fn = jax.pmap(self._eval_fn, axis_name="i")

        # Set evaluating state to True after initialization.
        self._evaluating = True

    def _build_eval_input(self) -> Generator[FeatureDict, None, None]:
        c = self.config
        global_batch_size = c.evaluation.batch_size
        per_device_batch_size, ragged = divmod(
            global_batch_size, jax.local_device_count()
        )
        # Raise error if not divisible
        if ragged:
            raise ValueError(
                f"Global batch size {global_batch_size} must be divisible by "
                f"number of devices {jax.local_device_count()}"
            )
        return input_pipeline.load(
            name=c.dataset.name,
            split=input_pipeline.Split.VALID,
            split_percentage=c.dataset.split_percentage,
            tfds_dir=c.dataset.tfds_dir,
            batch_sizes=[jax.local_device_count(), per_device_batch_size],
        )
