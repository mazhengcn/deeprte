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

import functools
from collections.abc import Callable, Mapping

import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections

from deeprte import dataset
from deeprte.model.base import Model, Solution, SolutionV2
from deeprte.model.integrate import quad
from deeprte.model.mapping import vmap
from deeprte.model.modules import FunctionInputs, GreenFunctionNet


def mean_squared_loss_fn(x, y, axis=None):
    return jnp.mean(jnp.square(x - y), axis=axis)


def make_rte_operator(config: ml_collections.ConfigDict) -> SolutionV2:
    def forward_fn(
        r: jnp.ndarray,
        v: jnp.ndarray,
        sigma: FunctionInputs,
        psi_bc: FunctionInputs,
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
        green_func_module = GreenFunctionNet(self.config.green_function)

        sol = quad(green_func_module, (psi_bc.x, psi_bc.f), argnum=1, use_hk=True)(
            rv, sigma
        )

        return sol

    transformed_solution = hk.transform_with_state(forward_fn)

    return SolutionV2(init=transformed_solution.init, apply=transformed_solution.apply)


class RTEOperator(Solution):
    def forward_fn(
        self,
        r: jnp.ndarray,
        v: jnp.ndarray,
        sigma: FunctionInputs,
        psi_bc: FunctionInputs,
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
        green_func_module = GreenFunctionNet(self.config.green_function)

        sol = quad(green_func_module, (psi_bc.x, psi_bc.f), argnum=1, use_hk=True)(
            rv, sigma
        )

        return sol

    def apply(
        self,
        params: hk.Params,
        state: hk.State,
        rng: jnp.ndarray,
        r: jnp.ndarray,
        v: jnp.ndarray,
        sigma: FunctionInputs,
        psi_bc: FunctionInputs,
        is_training: bool,
    ) -> jnp.ndarray:

        _apply_fn = self._apply

        if not is_training:
            _apply_fn = vmap(
                vmap(_apply_fn, shard_size=128, argnums={3, 4}),
                argnums={5, 6},
                in_axes=(FunctionInputs(), FunctionInputs()),
            )

        return _apply_fn(params, state, rng, r, v, sigma, psi_bc)

    def rho(
        self,
        params: hk.Params,
        r: jnp.ndarray,
        sigma: FunctionInputs,
        psi_bc: FunctionInputs,
        quadratures: tuple[jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        _apply = functools.partial(self.apply, params, None, None, is_training=True)
        _rho_fn = quad(_apply, quadratures, argnum=1)
        _rho_fn = vmap(
            vmap(_rho_fn, shard_size=128, argnums={0}),
            argnums={1, 2},
            in_axes=(FunctionInputs(), FunctionInputs()),
        )

        rho = jax.jit(_rho_fn)(r, sigma, psi_bc)

        return rho


class RTESupervised(Model):
    def loss(
        self, fn: Callable[..., jnp.float32], batch: dataset.Batch
    ) -> tuple[jnp.float32, Mapping[str, jnp.ndarray]]:
        # We vmap the fn since it supposed to be unvmapped before
        # in order to use other logic.
        predict_fn = vmap(
            vmap(fn, argnums={0, 1}),
            excluded={0, 1},
            in_axes=(FunctionInputs(), FunctionInputs()),
        )

        # Predictions and labels
        predictions, _ = predict_fn(*batch["inputs"])
        labels = batch["labels"]

        # Loss
        loss = mean_squared_loss_fn(predictions, labels)

        return loss, {
            "mse": loss,
            "rmspe": jnp.sqrt(loss / jnp.mean(labels**2)),
        }

    def metrics(
        self, fn: Callable[..., jnp.ndarray], batch: dataset.Batch
    ) -> Mapping[str, jnp.ndarray]:

        # Predictions
        predictions, _ = fn(*batch["inputs"])
        # Labels
        labels = batch["labels"]

        mse = mean_squared_loss_fn(predictions, labels, axis=-1)
        label_scale = jnp.mean(labels**2)
        # Compute relative mean squared error, this values will be summed and
        # finally divided by num_examples.
        relative_mse = mean_squared_loss_fn(predictions, labels, axis=-1) / jnp.mean(
            labels**2
        )

        return {
            "mse": mse,
            "rmspe": relative_mse,
        }


class RTEUnsupervised(Model):
    "Unsupervised RTE operator trained by equation."

    def __init__(self, cs, omega, name="radiative_transfer") -> None:
        super().__init__(name=name)

        self.quad_points = (cs, omega)

    @functools.partial(
        vmap, argnums={4, 5}, in_axes=(FunctionInputs(), FunctionInputs())
    )
    @functools.partial(vmap, argnums={2, 3})
    def residual(
        self,
        sol_fn: Callable[..., jnp.ndarray],
        r: jnp.ndarray,
        v: jnp.ndarray,
        sigma: FunctionInputs,
        psi_bc: FunctionInputs,
    ) -> jnp.ndarray:
        """Compute residual of equation for a single point."""

        sol = functools.partial(sol_fn, sigma=sigma, psi_b=psi_bc)

        # Gradients
        df_dr = jax.grad(sol)(r, v)  # [dim]
        # Transport term
        transport = jnp.matmul(v, df_dr)
        # Collision term
        collision = sigma.fx[..., 0] * quad(sol, self.quad_points, argnum=1)(
            r
        ) - sigma.fx[..., 1] * sol(r, v)

        residual = transport - collision

        return residual

    @functools.partial(
        vmap, argnums={3, 4}, in_axes=(FunctionInputs(), FunctionInputs())
    )
    # @partial(vmap, argnums={2, 3})
    def boundary(
        self,
        sol_fn: Callable[..., jnp.ndarray],
        bc_pts,
        sigma: FunctionInputs,
        psi_bc: FunctionInputs,
    ) -> jnp.ndarray:
        sol = functools.partial(sol_fn, sigma=sigma, psi_b=psi_bc)

        res_bc = vmap(sol)(bc_pts[..., :2], bc_pts[..., 2:]) - 1.0

        return res_bc

    def loss(self, sol_fn: Callable[..., jnp.ndarray], batch: dataset.Batch):

        batched_residual = self.residual(sol_fn, *batch["residual"])
        batched_boundary = self.boundary(sol_fn, *batch["boundary"])

        losses = {
            "residual": self._regs["residual"] * self._loss_fn(batched_residual, 0),
            "boundary": self._regs["boundary"] * self._loss_fn(batched_boundary, 0),
        }

        loss = sum(jax.tree_flatten(losses)[0])

        return loss, losses
