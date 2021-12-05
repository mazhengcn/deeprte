import functools
from collections.abc import Callable, Mapping

import haiku as hk
import jax
import jax.numpy as jnp

from deeprte import dataset, integrate
from deeprte.base_model import BaseModel
from deeprte.mapping import vmap
from deeprte.typing import F


def loss_fn(x, y):
    return jnp.mean(jnp.square(x - y))


class RTESupervised(BaseModel):
    def loss(
        self, fn: Callable[..., jnp.float32], batch: dataset.Batch
    ) -> tuple[jnp.float32, Mapping[str, jnp.ndarray]]:

        prediction_fn = vmap(
            vmap(fn, argnums={0, 1}),
            excluded={0, 1},
            in_axes=(F(), F()),
        )

        prediction, _ = prediction_fn(*batch["inputs"])

        label = batch["labels"]

        loss = {"inputs": loss_fn(prediction, label)}

        return loss["inputs"], {
            "rmse": jnp.sqrt(loss["inputs"] / jnp.mean(batch["labels"] ** 2))
        }


class RTEUnsupervised(BaseModel):
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

        batched_residual = self.residual(sol_fn, *batch["residual"])
        batched_boundary = self.boundary(sol_fn, *batch["boundary"])

        losses = {
            "residual": self._regs["residual"]
            * self._loss_fn(batched_residual, 0),
            "boundary": self._regs["boundary"]
            * self._loss_fn(batched_boundary, 0),
        }

        loss = sum(jax.tree_flatten(losses)[0])

        return loss, losses
