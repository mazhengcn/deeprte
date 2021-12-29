import functools
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from deeprte.deeprte_typings import Data, Losses
from deeprte.model.base import Model
from deeprte.model.integrate import quad


class LinearTransport1D(Model):
    """Equation configurations container."""

    def __init__(self, num_vquads, kn=1.0, name="linear_transport"):

        self.kn = kn

        self.num_vquads = num_vquads
        v, self.w = np.polynomial.legendre.leggauss(self.num_vquads)
        self.v = v[:, None]

    @functools.partial(jnp.vectorize, excluded=(0, 1))
    def residual(
        self,
        fn: Callable[..., jnp.ndarray],
        t: jnp.ndarray,
        x: jnp.ndarray,
        v: jnp.ndarray,
    ) -> jnp.ndarray:
        """Residual of equation.
        f: function of (x, v).
        """

        # Gradients
        df_dt, df_dx = jax.grad(fn, argnums=[0, 1])(t, x, v)
        # Transport term
        tran = jnp.squeeze(df_dt + v * df_dx)
        # Collision term
        coll = 0.5 * quad(fn, (self.v, self.w), argnum=2)(t, x) - fn(t, x, v)

        return [tran - coll]

    @functools.partial(jnp.vectorize, excluded=(0, 1))
    def boundary(self, fn: Callable[..., jnp.ndarray], tbc, vbc_l, vbc_r):
        # Left
        fbc_l = fn(tbc, 0.0 * tbc, vbc_l) - 1.0
        # Right
        fbc_r = fn(tbc, jnp.ones_like(tbc), vbc_r)

        return fbc_l, fbc_r

    @functools.partial(jnp.vectorize, excluded=(0, 1))
    def initial(self, fn: Callable[..., jnp.ndarray], x, v) -> jnp.ndarray:
        init_f = fn(0.0 * x, x, v)
        return init_f

    def loss(self, fn: Callable[..., jnp.ndarray], data: Data) -> Losses:

        losses = {
            "residual": self.residual(fn, *data["interior"]),
            "boundary": self.boundary(fn, *data["boundary"]),
            "initial": self.initial(fn, *data["initial"]),
        }

        return losses
