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


import jax
import jax.numpy as jnp


class Characteristics:
    def __init__(self, grid: jax.Array):

        self.grid = grid

    @classmethod
    def from_tensor(cls, tensor):
        return cls(tensor)

    def apply_to_point(self, phase_coord):
        pos, vel = jnp.split(phase_coord, 2, axis=-1)
        angle = vel / jnp.sqrt(jnp.sum(vel**2, axis=-1) + 1e-16)
        rel_pos = pos - self.grid
        rel_dist2 = jnp.sqrt(jnp.sum(rel_pos**2, axis=-1) + 1e-16)

        pos_local = jnp.matmul(rel_pos, angle)
        angle_local = pos_local / (rel_dist2 + 1e-8)
        coord_local = jnp.stack((angle_local, pos_local), axis=-1)
        mask = pos_local > 0

        return coord_local, mask

    def invert_point(self, transformed_point, extra_dims=0):
        pass
