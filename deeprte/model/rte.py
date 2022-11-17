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

"""RTE model of solutio operator and loss."""

from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from deeprte import dataset
from deeprte.model.base import Model, Solution
from deeprte.model.integrate import quad
from deeprte.model.mapping import vmap
from deeprte.model.modules import (
    FunctionInputs,
    GreenFunctionNet,
    GreenFunctionResBlock,
)


def mean_squared_loss_fn(x, y, axis=None):
    return jnp.mean(jnp.square(x - y), axis=axis)


class RTEOperator(Solution):
    def __init__(self, config, use_jit=False, name="rte_operator"):
        super().__init__(config, name)

        self.vapply = vmap(
            vmap(self.apply, shard_size=128, argnums={3, 4}),
            argnums={5, 6},
            in_axes=(FunctionInputs(None, 0), FunctionInputs(None, 0)),
        )
        self.rho = lambda *args: quad(self.apply, args[-1], argnum=4, has_aux=True)(
            *args[:-1]
        )
        self.vrho = vmap(
            vmap(self.rho, shard_size=128, argnums={3}),
            argnums={4, 5},
            in_axes=(FunctionInputs(None, 0), FunctionInputs(None, 0)),
        )

        if use_jit:
            self.vapply = jax.jit(self.vapply)
            self.rho = jax.jit(self.rho)
            self.vrho = jax.jit(self.vrho)

    def forward(
        self,
        r: jnp.ndarray,
        v: jnp.ndarray,
        sigma: FunctionInputs,
        psi_bc: FunctionInputs,
        # scattering_kernel: FunctionInputs,
    ) -> jnp.ndarray:
        """Compute solution with Green's function as kernel.

        Args:
            x: (x_dim,).
            v: (v_dim,).
            sigma: (num_coeff_values, xdim) and (num_coeff_values, num_coeffs).
            bc: (num_quads, xdim - 1) and (num_quads,).

        Returns:
            Solution outputs.
        """
        rv = jnp.concatenate([r, v])
        # green_func_module = GreenFunctionResBlock(self.config.green_function_block)
        green_func_module = GreenFunctionNet(self.config.green_function)
        # def
        # sol = quad(green_func_module, (psi_bc.x, psi_bc.f), argnum=1, use_hk=True)(
        #     rv, sigma, scattering_kernel
        # )
        sol = quad(green_func_module, (psi_bc.x, psi_bc.f), argnum=1, use_hk=True)(
            rv, sigma
        )
        return sol


class RTESupervised(Model):
    def loss(  # pylint: disable=arguments-differ
        self, func: Callable[..., jnp.float32], batch: dataset.Batch
    ) -> tuple[jnp.float32, Mapping[str, jnp.ndarray]]:
        # We vmap the fn since it supposed to be unvmapped before
        # in order to use other logic.
        predict_func = vmap(
            vmap(func, argnums={0, 1}),
            excluded={0, 1},
            in_axes=(FunctionInputs(None, 0), FunctionInputs(None, 0)),
        )

        # Predictions and labels
        predictions, _ = predict_func(*batch["inputs"])
        labels = batch["labels"]

        # Loss
        loss = mean_squared_loss_fn(predictions, labels)

        return loss, {
            "mse": loss,
            "rmspe": jnp.sqrt(loss / jnp.mean(labels**2)),
        }

    def metrics(
        self, func: Callable[..., jnp.ndarray], batch: dataset.Batch
    ) -> Mapping[str, jnp.ndarray]:

        predict_func = vmap(
            vmap(func, shard_size=128, argnums={0, 1}),
            excluded={0, 1},
            in_axes=(FunctionInputs(None, 0), FunctionInputs(None, 0)),
        )
        # Predictions
        predictions, _ = predict_func(*batch["inputs"])
        # Labels
        labels = batch["labels"]

        mse = mean_squared_loss_fn(predictions, labels, axis=-1)
        # label_scale = jnp.mean(labels**2)
        # Compute relative mean squared error, this values will be summed and
        # finally divided by num_examples.
        relative_mse = mean_squared_loss_fn(predictions, labels, axis=-1) / jnp.mean(
            labels**2
        )

        return {
            "mse": mse,
            "rmspe": relative_mse,
        }
