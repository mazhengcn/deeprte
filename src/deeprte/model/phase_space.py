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
"""Module for the PhaseSpace class."""

from collections.abc import Callable
from typing import NamedTuple, Type

import jax
import numpy as np
from jax import numpy as jnp

Array = np.ndarray | jax.Array
Float = float | np.float32


def cartesian_product(*arrays):
    """Compute cartesian product of arrays
    with different shapes in an efficient manner.

    Args:
        arrays: each array shoud be rank 2 with shape (N_i, d_i).
        inds: indices for each array, should be rank 1.

    Returns:
        Cartesian product of arrays with shape (N_1, N_2, ..., N_n, sum(d_i)).
    """
    d = [*map(lambda x: x.shape[-1], arrays)]
    ls = [*map(len, arrays)]
    inds = [*map(np.arange, ls)]

    dtype = np.result_type(*arrays)
    arr = np.empty(ls + [sum(d)], dtype=dtype)

    for i, ind in enumerate(np.ix_(*inds)):
        arr[..., sum(d[:i]) : sum(d[: i + 1])] = arrays[i][ind]
    return arr


def jax_cartesian_product(arrays):
    """Compute cartesian product of arrays
    with different shapes in an efficient manner.

    Args:
        arrays: each array shoud be rank 2 with shape (N_i, d).
        inds: indices for each array, should be rank 1.

    Returns:
        Cartesian product of arrays with shape (N_1, N_2, ..., N_n, sum(d_i)).
    """
    num_arrs = len(arrays)

    def concat_fn(a):
        return jnp.concatenate(a, axis=-1)

    for i in range(num_arrs):
        in_axes = [None] * num_arrs
        in_axes[-i - 1] = int(0)
        concat_fn = jax.vmap(concat_fn, in_axes=(in_axes,))

    return concat_fn(arrays)


def _concat(*arrs):
    if isinstance(arrs[0], np.ndarray):
        return np.concatenate(arrs, axis=-1)

    return jnp.concatenate(arrs, axis=-1)


def _cartesian_product(*arrs):
    if isinstance(arrs[0], np.ndarray):
        return cartesian_product(arrs)

    return jax_cartesian_product(arrs)


class PhaseSpace(NamedTuple):
    """Hold a pair of position and velocity coordinates in the phase space.
    Here we allow position and velocity arrays have different first dimension,
    which by a cartesian product to generate a single state.

    """

    # shape (N, d) or (N, d), (M, d)
    position_coords: Array
    velocity_coords: Array
    # shape (N) or (N), (M)
    position_weights: Array
    velocity_weights: Array

    def single_state(self, cartesian_product: bool = False):
        """Returns the concatenation of position and velocity."""
        if cartesian_product:
            assert self.r.shape[-1] == self.v.shape[-1]
            return _cartesian_product(self.r, self.v)
        else:
            assert self.r.shape == self.v.shape
            return _concat(self.r, self.v)

    @property
    def state_weights(self) -> Array:
        """Returns the total weights of phase space grid points."""
        return self.rw[:, None] * self.vw

    @property
    def ndim(self):
        assert self.r.shape[-1] == self.v.shape[-1]
        return self.r.shape[-1]

    @property
    def r(self):
        """A shorthand for the position element of the phase space."""
        return self.position_coords

    @property
    def v(self):
        """A shorthand for the velocity element of the phase space."""
        return self.velocity_coords

    @property
    def rw(self):
        return self.position_weights

    @property
    def vw(self):
        return self.velocity_weights

    @property
    def position_space(self):
        return self.r, self.rw

    @property
    def velocity_space(self):
        return self.v, self.vw

    @classmethod
    def from_state(cls: Type["PhaseSpace"], state: Array) -> "PhaseSpace":
        r, v = np.split(state, 2, axis=-1)
        return cls(position=r, velocity=v)

    def __str__(self) -> str:
        return f"{type(self).__name__}(r={self.position}, v={self.velocity})"

    def __repr__(self) -> str:
        return self.__str__()


DensityFunction = Callable[[PhaseSpace], jax.Array]
GreenFunction = Callable[[PhaseSpace, PhaseSpace], jax.Array]
