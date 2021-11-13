import functools
from typing import Callable, List

import jax
import jax.numpy as jnp
import numpy as np
from modnet.integrate import quad
from modnet.models.base_model import BaseModel
from modnet.typing import Data, Losses


class MicroMacro1D(BaseModel):
    """Equation configurations container."""

    def __init__(
        self, num_vquads, kn=1.0, random: bool = False, name="linear_transport"
    ):

        self.kn = kn
        self.num_vquads = num_vquads
        self._random = random

    def _get_quadrature_pts(self, rng):
        if self._random:
            v = jax.random.uniform(
                rng, shape=(self.num_vquads, 1), minval=-1, maxval=1
            )
            w = 2.0 / self.num_vquads
        else:
            v, w = np.polynomial.legendre.leggauss(self.num_vquads)
            v = v[:, None]

        self.vquads = (v, w)

    @functools.partial(jnp.vectorize, excludes=(0, 1))
    def residual(
        self,
        fn: List[Callable[..., jnp.ndarray]],
        t: jnp.ndarray,
        x: jnp.ndarray,
        v: jnp.ndarray,
    ) -> List[jnp.ndarray]:
        """Residual of equation.
        f: function of (x, v).
        """

        rho, g = fn

        rho_t, rho_x = jax.grad(rho, argnums=(0, 1))(t, x)
        g_t, g_x = jax.grad(g, argnums=(0, 1))(t, x, v)

        def vg_x(t, x, v):
            return v * jax.grad(g, argnums=1)(t, x, v)

        avg_vg_x = 0.5 * quad(vg_x, self.vquads, argnum=2)(t, x)

        res_rho = rho_t + avg_vg_x

        res_g = (
            self.kn ** 2 * g_t
            + self.kn * (v * g_x - avg_vg_x)
            + (g(t, x, v) + v * rho_x)
        )

        return res_rho, res_g

    @functools.partial(jnp.vectorize, excludes=(0, 1))
    def boundary(
        self, fn: List[Callable[..., jnp.ndarray]], tbc, vbc_l, vbc_r
    ) -> List[jnp.ndarray]:
        rho, g = fn
        # Left
        rho_l, g_l = rho(tbc, 0.0 * tbc), g(tbc, 0.0 * tbc, vbc_l)
        fbc_l = rho_l + self.kn * g_l - 1.0
        # Right
        rho_r, g_r = (
            rho(tbc, jnp.ones_like(tbc)),
            g(tbc, jnp.ones_like(tbc), vbc_r),
        )
        fbc_r = rho_r + self.kn * g_r

        return fbc_l, fbc_r

    @functools.partial(jnp.vectorize, excludes=(0, 1))
    def initial(
        self, fn: List[Callable[..., jnp.ndarray]], x, v
    ) -> jnp.ndarray:
        rho, g = fn
        init_rho, init_g = rho(0.0 * x, x), g(0.0 * x, x, v)
        return init_rho + self.kn * init_g

    def loss(
        self,
        fn: List[Callable[..., jnp.ndarray]],
        rng: jnp.ndarray,
        data: Data,
    ) -> Losses:

        self._get_quadrature_pts(rng)

        rho, g = fn
        g = functools.partial(g, vquads=self.vquads)

        losses = {
            "residual": self.residual((rho, g), *data["interior"]),
            "boundary": self.boundary((rho, g), *data["boundary"]),
            "initial": self.initial((rho, g), *data["initial"]),
        }

        return losses
