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

"""Integration operator."""

from collections.abc import Callable
from typing import Optional

import jax.numpy as jnp

from deeprte.model import mapping


def quad(
    func: Callable[..., jnp.float32],
    quad_points: tuple[jnp.ndarray, jnp.ndarray],
    argnum=0,
    has_aux: Optional[bool] = False,
    use_hk: Optional[bool] = False,
) -> Callable[..., float]:
    """Compute the integral operator for a scalar function using
    quadratures."""

    nodes, weights = quad_points

    def integral(*args):
        args = list(args)
        args.insert(argnum, nodes)
        if not has_aux:
            values = mapping.vmap(func, argnums={argnum}, out_axes=-1, use_hk=use_hk)(
                *args
            )
            return jnp.matmul(values, weights)
        else:
            values, aux = mapping.vmap(
                func, argnums={argnum}, out_axes=-1, use_hk=use_hk
            )(*args)

            return jnp.matmul(values, weights), aux

    return integral
