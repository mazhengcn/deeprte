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

import jax
import jax.numpy as jnp

from deeprte.model.mapping import sharded_map


def quad(
    fun: Callable[..., float],
    quadratures: tuple[jax.Array, jax.Array],
    argnum: int = 0,
    shard_size: int | None = None,
    has_aux: Optional[bool] = False,
) -> Callable[..., float]:
    """Compute the integral operator for a scalar function using
    quadratures.
    """

    points, weights = quadratures

    def integral_fn(*args):
        args = list(args)
        in_axes_ = [None] * len(args)
        args.insert(argnum, points)
        in_axes_.insert(argnum, int(0))
        out = sharded_map(
            fun, shard_size=shard_size, in_axes=tuple(in_axes_), out_axes=-1
        )(*args)
        if has_aux:
            values, aux = out
            result = jnp.dot(values, weights)
            return result, aux

        return jnp.dot(out, weights)

    return integral_fn


def value_and_quad(
    fun: Callable[..., float],
    quadratures: tuple[jax.Array, jax.Array],
    argnum: int = 0,
    shard_size: int | None = None,
    has_aux: Optional[bool] = False,
) -> Callable[..., float]:
    """Compute the integral operator for a scalar function using
    quadratures.
    """

    points, weights = quadratures

    def integral_fn(*args):
        args = list(args)
        in_axes_ = [None] * len(args)
        args.insert(argnum, points)
        in_axes_.insert(argnum, int(0))
        out = sharded_map(fun, shard_size=shard_size, in_axes=in_axes_, out_axes=-1)(
            *args
        )
        if has_aux:
            values, aux = out
            result = jnp.dot(values, weights)
            return result, aux

        return out, jnp.dot(out, weights)

    return integral_fn
