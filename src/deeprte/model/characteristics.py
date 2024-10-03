# Copyright 2022 Zheng Ma
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module describing characteristics of RTE for given geometry."""

import jax
import jax.numpy as jnp


class Characteristics:
    """Characteristic container class."""

    def __init__(self, grid: jax.Array, name: str = "characteristics"):
        self.name = name
        self.grid = grid

    @classmethod
    def from_tensor(cls, grid: jax.Array):
        return cls(grid)

    def apply_to_point(self, phase_coord: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute local coordinates and mask of the grid
        for a given point in phase space.
        """

        x, v = jnp.split(phase_coord, 2, axis=-1)
        theta = v / jnp.sqrt(jnp.sum(v**2, axis=-1) + 1e-16)
        rel_x = x - self.grid
        rel_dist2 = jnp.sqrt(jnp.sum(rel_x**2, axis=-1) + 1e-16)

        local_x = jnp.dot(rel_x, theta)
        local_theta = local_x / (rel_dist2 + 1e-8)

        local_grid = jnp.stack((local_theta, local_x), axis=-1)
        mask = jnp.expand_dims(local_x > 0, axis=-2)

        return local_grid, mask

    # def invert_point(self, transformed_point, extra_dims=0):
    #     pass
