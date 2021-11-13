import functools
from typing import Callable, Dict

import haiku as hk
import jax
import jax.numpy as jnp
from modnet.integrate import quad
from modnet.mapping import vmap
from modnet.models.base_model import BaseModel
from modnet.modules.green_fn import GreenFunction
from modnet.solution import Solution
from modnet.typing import GraphOfMapping

FeatureDict = Dict[str, jnp.ndarray]
partial = functools.partial


class RTEOperator(Solution):
    def forward_fn(
        self,
        x: jnp.ndarray,
        v: jnp.ndarray,
        sigma: GraphOfMapping,
        bc: GraphOfMapping,
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
        xv = jnp.concatenate([x, v])

        green_func_module = GreenFunction(self.config)

        sol = quad(green_func_module, (bc.x, bc.fx), argnum=1, use_hk=True)(
            xv, sigma
        )

        return 0.15 * sol

    def predict(self, params: hk.Params, rng: jnp.ndarray, x, v, sigma, bc):
        apply_fn = functools.partial(self._apply, params, rng)
        prediction_fn = vmap(
            vmap(apply_fn, shard_size=128, argnums={0, 1}),
            argnums={2, 3},
            in_axes=(GraphOfMapping(None, 0),) * 2,
        )
        result = jax.jit(prediction_fn)(x, v, sigma, bc)

        return result

    def rho(self, params: hk.Params, rng: jnp.ndarray, x, sigma, bc, vw):
        apply_fn = partial(self._apply, params, rng)
        rho_fn = quad(apply_fn, vw, argnum=1)
        vrho_fn = vmap(
            vmap(rho_fn, shard_size=128, argnums={0}),
            argnums={1, 2},
            in_axes=(GraphOfMapping(None, 0),) * 2,
        )
        result = jax.jit(vrho_fn)(x, sigma, bc)

        return result


class RTEModel(BaseModel):
    def loss(self, f, batch):

        prediction_fn = vmap(
            vmap(f, argnums={0, 1}),
            excluded={0, 1},
            in_axes=(GraphOfMapping(None, 0),) * 2,
        )

        prediction = prediction_fn(*batch["interior"])

        label = batch["label"]

        loss = {"residual": self._loss_fn(prediction, label)}

        return loss["residual"], {
            "rmse": jnp.sqrt(loss["residual"] / jnp.mean(batch["label"] ** 2))
        }


class RTEOpUnsupervised(BaseModel):
    "Unsupervised RTE operator trained by equation."

    def __init__(self, cs, omega, name="radiative_transfer") -> None:
        super().__init__(name=name)

        self.quad_points = (cs, omega)

    @partial(vmap, argnums={4, 5}, in_axes=(GraphOfMapping(None, 0),) * 2)
    @partial(vmap, argnums={2, 3})
    def residual(
        self,
        sol_fn: Callable[..., jnp.ndarray],
        x: jnp.ndarray,
        v: jnp.ndarray,
        sigma: GraphOfMapping,
        bc: GraphOfMapping,
    ) -> jnp.ndarray:
        """Compute residual of equation for a single point."""

        sol = partial(sol_fn, sigma=sigma, bc=bc)

        # Gradients
        df_dx = jax.grad(sol)(x, v)  # [dim]
        # Transport term
        transport = jnp.matmul(v, df_dx)
        # Collision term
        collision = sigma.fx[..., 0] * quad(sol, self.quad_points, argnum=1)(
            x
        ) - sigma.fx[..., 1] * sol(x, v)

        residual = transport - collision

        return residual

    @partial(vmap, argnums={3, 4}, in_axes=(GraphOfMapping(None, 0),) * 2)
    # @partial(vmap, argnums={2, 3})
    def boundary(
        self,
        sol_fn: Callable[..., jnp.ndarray],
        bc_pts,
        sigma: GraphOfMapping,
        bc: GraphOfMapping,
    ):
        sol = partial(sol_fn, sigma=sigma, bc=bc)

        res_bc = vmap(sol)(bc_pts[..., :2], bc_pts[..., 2:]) - 1.0

        return res_bc

    # @functools.partial(jnp.vectorize, excluded=(0, 1))
    # def initial(self, fn: Callable[..., jnp.ndarray], x, v) -> jnp.ndarray:
    #     init_f = fn(0.0 * x, x, v)
    #     return init_f

    def loss(self, sol_fn: Callable[..., jnp.ndarray], batch: FeatureDict):

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
