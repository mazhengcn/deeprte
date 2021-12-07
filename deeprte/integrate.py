from collections.abc import Callable
from typing import Optional

import jax.numpy as jnp

from deeprte import mapping


def quad(
    func: Callable[..., jnp.float32],
    quad_points: tuple[jnp.ndarray, jnp.ndarray],
    argnum=0,
    use_hk: Optional[bool] = False,
) -> Callable[..., float]:
    """Compute the integral operator for a scalar function using
    quadratures."""

    nodes, weights = quad_points

    def integral(*args):
        args = list(args)
        args.insert(argnum, nodes)
        values = mapping.vmap(
            func, argnums={argnum}, out_axes=-1, use_hk=use_hk
        )(*args)
        return jnp.matmul(values, weights)

    return integral
