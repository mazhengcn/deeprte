import functools
from typing import Any, Callable, Mapping, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from tensorflow.keras.utils import Progbar

from deeprte.models.base_model import BaseModel
from deeprte.typing import Data, Logs, Metric, State


class Solver(object):
    def __init__(self, sol: hk.Transformed, model: BaseModel) -> None:

        self._sol = sol
        self._model = model
        self._state = None

    def compile(
        self,
        loss_fn: Callable[[jnp.ndarray], jnp.float32],
        optimizer: optax.GradientTransformation,
        regularizers: Mapping[str, jnp.float32] = {},
    ) -> None:

        self._loss_fn = loss_fn
        self._regs = regularizers
        self._opt = optimizer
        self._model.compile(self._loss_fn, self._regs)

    @functools.partial(jax.jit, static_argnums=0)
    def init(self, rng: jnp.ndarray, data: Data) -> Logs:
        out_rng, init_rng = jax.random.split(rng)
        params = self._sol.init(init_rng, *data)
        opt_state = self._opt.init(params)
        out = dict(
            epoch=np.int32(0),
            step=np.int32(0),
            rng=out_rng,
            opt_state=opt_state,
            params=params,
        )
        return out

    def loss(
        self, params: hk.Params, rng: jnp.ndarray, data: Data
    ) -> Tuple[jnp.float32, Logs]:
        sol = self._sol.apply(partial_args=(params, None))
        return self._model.loss(sol, data)

    @functools.partial(jax.jit, static_argnums=0)
    def train_step(self, state: State, data: Data) -> Tuple[State, Metric]:
        rng, new_rng = jax.random.split(state["rng"])
        params = state["params"]
        (_, logs), grad = jax.value_and_grad(self.loss, has_aux=True)(
            params, rng, data
        )

        updates, opt_state = self._opt.update(grad, state["opt_state"])
        params = optax.apply_updates(params, updates)

        new_state = {
            "epoch": state["epoch"],
            "step": state["step"] + 1,
            "rng": new_rng,
            "opt_state": opt_state,
            "params": params,
        }

        metrics = {"step": state["step"], "logs": logs}

        return new_state, metrics

    @functools.partial(jax.jit, static_argnums=0)
    def test_step(
        self, params: hk.Params, data: Tuple[jnp.ndarray, jnp.ndarray]
    ):
        inputs, labels = data
        predictions = self._sol.predict(params, None, *inputs)
        return jnp.sqrt(
            jnp.mean((predictions - labels) ** 2) / jnp.mean(labels ** 2)
        )

    def solve(
        self,
        dataset: tf.data.Dataset,
        init_data: Any = None,
        num_epochs: int = 1,
        steps_per_epoch: int = None,
        val_data: tf.data.Dataset = None,
        val_freq: int = 1,
        restart: bool = False,
        seed: int = 0,
    ) -> None:
        if not self._state or restart:
            print("Initializing parameters...")
            rng = jax.random.PRNGKey(seed)
            # dummy_data = next(dataset)["interior"]
            # if not isinstance(dummy_data, (list, tuple)):
            #     dummy_data = [dummy_data]
            # dummy_data = jax.tree_map(lambda x: x[0], dummy_data)
            self._state = self.init(rng, init_data)
            print(
                "Parameters are: "
                f"{jax.tree_map(lambda x: x.shape, self._state['params'])}"
            )

        end_epoch = self.epoch + num_epochs

        if not steps_per_epoch:
            steps_per_epoch = dataset.cardinality().numpy()

        for _ in range(num_epochs):
            pbar = Progbar(
                steps_per_epoch,
                stateful_metrics=[
                    "total_loss",
                    "residual",
                    "boundary",
                    "initial",
                ],
            )
            self._state["epoch"] += 1
            print(f"Epoch {self.epoch:d}/{end_epoch:d}")

            for (step, data) in enumerate(dataset.as_numpy_iterator()):
                self._state, metrics = self.train_step(self._state, data)
                pbar.update(
                    step + 1,
                    # values=list(metrics["logs"].items()),
                    finalize=False,
                )

            if val_data and self._state["epoch"] % val_freq == 0:
                val_loss = self.test_step(self.params, val_data)
                metrics["logs"].update({"val_loss": val_loss})

            pbar.update(
                steps_per_epoch,
                values=list(metrics["logs"].items()),
                finalize=True,
            )

    @property
    def epoch(self):
        return self._state["epoch"]

    @property
    def optimizer(self):
        return self._opt

    @property
    def loss_fn(self):
        return self._loss_fn

    @property
    def regularizers(self):
        return self._regularizers

    @property
    def params(self):
        return self._state["params"]
