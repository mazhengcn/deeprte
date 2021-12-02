import functools
from collections.abc import Callable, Mapping

import haiku as hk
import jax
import jax.numpy as jnp
from deeprte import dataset, integrate
from deeprte.mapping import vmap
from deeprte.models.base_model import BaseModel
from deeprte.modules import GreenFunctionNet
from deeprte.solution import Solution
from deeprte.typing import F


class RTEOperator(Solution):
    def forward_fn(
        self, r: jnp.ndarray, v: jnp.ndarray, sigma: F, psi_bc: F
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

        sol = integrate.quad(
            green_func_module, (psi_bc.x, psi_bc.y), argnum=1, use_hk=True
        )(rv, sigma)

        return 0.15 * sol

    def apply(
        self,
        params: hk.Params,
        state: hk.State,
        rng: jnp.ndarray,
        r: jnp.ndarray,
        v: jnp.ndarray,
        sigma: F,
        psi_bc: F,
        is_training: bool = True,
    ) -> jnp.ndarray:
        _apply_fn = self._apply

        if not is_training:
            _apply_fn = vmap(
                vmap(self._apply, shard_size=128, argnums={3, 4}),
                argnums={5, 6},
                in_axes=(F(), F()),
            )

        return _apply_fn(params, state, rng, r, v, sigma, psi_bc)

    def rho(
        self,
        params: hk.Params,
        r: jnp.ndarray,
        sigma: F,
        psi_bc: F,
        quadratures: tuple[jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        _apply = functools.partial(
            self.apply, params, None, None, is_training=True
        )
        _rho_fn = integrate.quad(_apply, quadratures, argnum=1)
        _rho_fn = vmap(
            vmap(_rho_fn, shard_size=128, argnums={0}),
            argnums={1, 2},
            in_axes=(F(), F()),
        )

        rho = jax.jit(_rho_fn)(r, sigma, psi_bc)

        return rho


def loss_fn(x, y):
    return jnp.mean(jnp.square(x - y))


class RTEModel(BaseModel):
    def loss(
        self, fn: Callable[..., jnp.float32], batch: dataset.Batch
    ) -> tuple[jnp.float32, Mapping[str, jnp.ndarray]]:

        prediction_fn = vmap(
            vmap(fn, argnums={0, 1}),
            excluded={0, 1},
            in_axes=(F(), F()),
        )

        prediction, _ = prediction_fn(*batch["interior"])

        label = batch["labels"]

        loss = {"residual": loss_fn(prediction, label)}

        return loss["residual"], {
            "rmse": jnp.sqrt(loss["residual"] / jnp.mean(batch["labels"] ** 2))
        }


class RTEOpUnsupervised(BaseModel):
    "Unsupervised RTE operator trained by equation."

    def __init__(self, cs, omega, name="radiative_transfer") -> None:
        super().__init__(name=name)

        self.quad_points = (cs, omega)

    @functools.partial(vmap, argnums={4, 5}, in_axes=(F(), F()))
    @functools.partial(vmap, argnums={2, 3})
    def residual(
        self,
        sol_fn: Callable[..., jnp.ndarray],
        r: jnp.ndarray,
        v: jnp.ndarray,
        sigma: F,
        psi_bc: F,
    ) -> jnp.ndarray:
        """Compute residual of equation for a single point."""

        sol = functools.partial(sol_fn, sigma=sigma, psi_b=psi_bc)

        # Gradients
        df_dr = jax.grad(sol)(r, v)  # [dim]
        # Transport term
        transport = jnp.matmul(v, df_dr)
        # Collision term
        collision = sigma.fx[..., 0] * integrate.quad(
            sol, self.quad_points, argnum=1
        )(r) - sigma.fx[..., 1] * sol(r, v)

        residual = transport - collision

        return residual

    @functools.partial(vmap, argnums={3, 4}, in_axes=(F(), F()))
    # @partial(vmap, argnums={2, 3})
    def boundary(
        self,
        sol_fn: Callable[..., jnp.ndarray],
        bc_pts,
        sigma: F,
        psi_bc: F,
    ) -> jnp.ndarray:
        sol = functools.partial(sol_fn, sigma=sigma, psi_b=psi_bc)

        res_bc = vmap(sol)(bc_pts[..., :2], bc_pts[..., 2:]) - 1.0

        return res_bc

    def loss(self, sol_fn: Callable[..., jnp.ndarray], batch: dataset.Batch):

        batched_residual = self.residual(sol_fn, *batch["interior"])
        batched_boundary = self.boundary(sol_fn, *batch["boundary"])

        losses = {
            "residual": self._regs["residual"]
            * self._loss_fn(batched_residual, 0),
            "boundary": self._regs["boundary"]
            * self._loss_fn(batched_boundary, 0),
        }

        loss = sum(jax.tree_flatten(losses)[0])

        return loss, losses
