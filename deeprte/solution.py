import abc
import functools
from collections.abc import Callable
from typing import Any, Optional

import haiku as hk
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from deeprte import integrate
from deeprte.mapping import vmap
from deeprte.modules import GreenFunctionNet
from deeprte.typing import F

TemplateFn = Callable[..., Any]


class Solution(object, metaclass=abc.ABCMeta):
    """Solution container used to create Haiku transformed pure function."""

    def __init__(
        self,
        config: ConfigDict,
        name: Optional[str] = "solution",
    ) -> None:
        self.name = name
        self.config = config

        self.init = hk.transform_with_state(self.forward_fn).init
        self._apply = hk.transform_with_state(self.forward_fn).apply

    @abc.abstractmethod
    def forward_fn(self) -> jnp.ndarray:
        pass

    @abc.abstractmethod
    def apply(
        self,
        params: hk.Params,
        state: hk.State,
        rng: jnp.ndarray,
        *,
        is_training: bool
    ) -> jnp.ndarray:
        pass


class MultiSolutions(object, metaclass=abc.ABCMeta):
    """Multi-Solutions container used to create Haiku transformed
    pure functions.
    """

    def __init__(
        self,
        config: tuple[ConfigDict],
        name: Optional[str] = "multi_solutions",
    ) -> None:

        self.name = name
        self.config = config

        self.init = hk.multi_transform(self.forward_fn).init
        self._apply = hk.multi_transform(self.forward_fn).apply

    @abc.abstractmethod
    def forward_fn(self) -> tuple[TemplateFn, Any]:
        pass

    @abc.abstractmethod
    def apply(self, params, state, rng, *, is_training):
        pass


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
