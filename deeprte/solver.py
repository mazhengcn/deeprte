import functools
from collections.abc import Generator, Mapping

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from absl import flags, logging
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

    def __init__(self, mode, init_rng, config):
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
            raise ValueError(f'Unknown mode: "{mode}"')

    #  _             _
    # | |_ _ __ __ _(_)_ __
    # | __| '__/ _` | | '_ \
    # | |_| | | (_| | | | | |
    #  \__|_|  \__,_|_|_| |_|
    #

    def step(
        self, global_step: int, rng: jnp.ndarray, *unused_args, **unused_kwargs
    ):
        """See base class."""

        if self._train_input is None:
            self._initialize_train()

        inputs = next(self._train_input)

        self._params, self._state, self._opt_state, scalars = self.update_fn(
            self._params,
            self._state,
            self._opt_state,
            inputs,
            global_step,
            rng,
        )

        scalars = jl_utils.get_first(scalars)

        # Save final checkpoint.
        if (
            self.config.save_final_checkpoint_as_npy
            and not self.config.dry_run
        ):
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
        eval_scalars = self.evaluate(global_step, rng)

        def f_list(x):
            return x.tolist() if isinstance(x, jnp.ndarray) else x

        self._last_evaluation_scalars = jax.tree_map(f_list, eval_scalars)
        logging.info(
            f"(eval) global_step: {global_step_value}, "
            f"{self._last_evaluation_scalars}"
        )
        return _merge_eval_scalars(scalars, self._last_evaluation_scalars)

    def _update_fn(
        self,
        params: hk.Params,
        state: hk.State,
        opt_state: OptState,
        inputs: dataset.Batch,
        global_step: int,
        rng: jnp.ndarray,
    ):
        scalars = {}

        grad_fn = jax.grad(self._loss_fn, has_aux=True)

        # Compute loss and gradients.
        scaled_grads, (loss_scalars, state) = grad_fn(
            params, state, inputs, rng
        )
        grads = jax.lax.psum(scaled_grads, axis_name="i")

        # Grab the learning rate to log before performing the step.
        learning_rate = self._lr_schedule(global_step)
        scalars["learning_rate"] = learning_rate

        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        loss_scalars = {f"train_{k}": v for k, v in loss_scalars.items()}
        scalars.update(loss_scalars)
        scalars = jax.lax.pmean(scalars, axis_name="i")

        return params, state, opt_state, scalars

    def _loss_fn(
        self,
        params: hk.Params,
        state: hk.State,
        inputs: dataset.Batch,
        rng: jnp.ndarray,
    ) -> tuple[jnp.ndarray, tuple[Scalars, hk.State]]:

        _solution_fn = functools.partial(
            self.solution.apply, params, state, rng, is_training=True
        )

        loss, loss_scalars = self.model.loss(_solution_fn, inputs)

        scaled_loss = loss / jax.local_device_count()

        return scaled_loss, (loss_scalars, state)

    def _initialize_training(self):
        _train_input = jl_utils.py_prefetch(self._build_train_input)
        self._train_input = jl_utils.double_buffer_on_gpu(_train_input)

        total_batch_size = self.config.training.batch_size
        steps_per_epoch = (
            self.config.training.num_train_examples
            / self.config.training.batch_size
        )
        total_steps = self.config.training.num_epochs * steps_per_epoch
        # Scale by the (negative) learning rate.
        self._lr_schedule = optimizers.get_learning_rate_schedule(
            total_batch_size,
            steps_per_epoch,
            total_steps,
            self.config.optimizer,
        )

        self.optimizer = optimizers.make_optimizer(
            self.config.optimizer, self._lr_schedule
        )
        # Initialize net and EMA copy of net if no params available.
        if self._params is None:
            logging.info("Initializing parameters.")

            inputs = next(self._train_input)
            (r, v, sigma, psi_bc) = inputs["inputs"]

            dummy_data = (
                r[:, 0],
                v[:, 0],
                F(x=sigma.x, y=sigma.y[:, 0]),
                F(x=psi_bc.x, y=psi_bc.y[:, 0]),
            )

            init_net = jax.pmap(lambda *a: self.solution.init(*a))
            init_opt = jax.pmap(self.optimizer.init)

            # Init uses the same RNG key on all hosts+devices to ensure
            # everyone computes the same initial state.
            init_rng = jl_utils.bcast_local_devices(self.init_rng)

            self._params, self._state = init_net(init_rng, *dummy_data)
            self._opt_state = init_opt(self._params)

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
        """See base class."""

        num_devices = jax.local_device_count()

        global_batch_size = self.config.training.batch_size
        per_device_batch_size, ragged = divmod(global_batch_size, num_devices)

        if ragged:
            raise ValueError(
                f"Global batch size {global_batch_size} must be divisible by "
                f"num devices {num_devices}"
            )

        split = dataset.Split.TRAIN_AND_VALID

        return self._load_data(
            split=split,
            is_training=True,
            batch_dims=[jax.local_device_count(), per_device_batch_size, 500],
        )

    def _load_data(self, split, is_training, batch_dims):
        """Wrapper for dataset loading."""

        return dataset.load(
            self.config.dataset.data_path,
            split=split,
            is_training=is_training,
            batch_dims=batch_dims,
            data_dir="./data/rte/rte_2d_converted.npz",
        )

    #                  _
    #   _____   ____ _| |
    #  / _ \ \ / / _` | |
    # |  __/\ V / (_| | |
    #  \___| \_/ \__,_|_|
    #

    def evaluate(self, global_step, rng, **unused_args):
        """See base class."""

        metrics = jax.device_get(
            self._eval_epoch(self._params, self._state, rng)
        )
        logging.info(f"[Step {global_step}] Eval scalars: {metrics}")

        return metrics

    def _eval_fn(
        self,
        params: hk.Params,
        state: hk.State,
        inputs: dataset.Batch,
        rng: jnp.ndarray,
    ) -> Scalars:
        """Evaluates a batch."""
        metrics = {}

        labels = inputs["labels"]
        pred = self.solution.apply(
            params, state, rng, *inputs["interior"], is_training=False
        )
        metrics["rmse"] = jnp.sqrt(
            jnp.mean((pred - labels) ** 2) / jnp.mean(labels ** 2)
        )

        # NOTE: Returned values will be summed and finally divided
        # by num_samples.
        return jax.lax.pmean(metrics, axis_name="i")

    def _eval_epoch(self, params, state, rng):
        """Evaluates an epoch."""
        num_samples = 0.0
        summed_scalars = None

        # params = jl_utils.get_first(self._params)
        # state = jl_utils.get_first(self._state)

        for inputs in self._eval_input:
            num_samples += jnp.prod(inputs["labels"].shape[:2])
            metrics = self.eval_fn(params, state, inputs, rng)
            # Accumulate the sum of scalars for each step.
            metrics = jax.tree_map(lambda x: jnp.sum(x[0], axis=0), metrics)
            if summed_scalars is None:
                summed_scalars = metrics
            else:
                summed_metrics = jax.tree_multimap(
                    jnp.add, summed_scalars, metrics
                )

        mean_metrics = jax.tree_map(lambda x: x / num_samples, summed_metrics)
        return jax.device_get(mean_metrics)

    def _initialize_evaluation(self):
        _eval_input = jl_utils.py_prefetch(self._build_eval_input)
        self._eval_input = jl_utils.double_buffer_on_gpu(_eval_input)
        self.eval_fn = jax.pmap(self._eval_fn, axis_name="i")

    def _build_eval_input(self) -> Generator[dataset.Batch, None, None]:
        split = dataset.Split.VALID

        return self._load_data(
            split=split,
            is_training=False,
            batch_dims=[self.config.evaluation.batch_size],
        )


def _merge_eval_scalars(a, b):
    if b is None:
        return a
    for k, v in b.items():
        a["eval_" + k] = v
    return a
