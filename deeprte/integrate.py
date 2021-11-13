from typing import Callable, Optional, Tuple

import jax.numpy as jnp

from deeprte import mapping


def quad(
    func: Callable[..., jnp.float32],
    quad_points: Tuple[jnp.ndarray, jnp.ndarray],
    argnum=0,
    use_hk: Optional[bool] = False,
) -> Callable[..., float]:
    """Compute the integral operator for a scalar function using
    quadratures."""

    points, weights = quad_points

    def integral(*args):
        args = list(args)
        args.insert(argnum, points)
        values = mapping.vmap(
            func, argnums={argnum}, out_axes=-1, use_hk=use_hk
        )(*args)
        return jnp.matmul(values, weights)

    return integral
