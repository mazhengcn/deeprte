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

"""Core modules including Green's function net and sigma net."""

from typing import NamedTuple, Optional

import haiku as hk
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from deeprte.model.mapping import vmap
from deeprte.model.networks import MLP


class F(NamedTuple):
    """Graph of a function (x, y=f(x)) as a namedtuple."""

    x: jnp.float32 | jnp.ndarray = None
    y: jnp.ndarray = 0


class GreenFunctionNet(hk.Module):
    """Green's function of solution operator."""

    def __init__(
        self,
        config: ConfigDict,
        name: Optional[str] = "green_function",
    ):
        super().__init__(name=name)

        self.config = config

    def __call__(
        self, r: jnp.ndarray, r_prime: jnp.ndarray, coefficient_fn: F
    ) -> jnp.ndarray:
        """Compute Green's function with coefficient net as inputs.

        Args:
            r: Position and velocity variables, both are dimension d.
                Shape (2d,)
            r_prime: Dual position and velocity variables. Shape (2d,)
            coefficient_fn: Coefficient function at as a NamedTuple (x, y)
                where x are the positions and y are corresponding values.
                Shapes are (num_positions, d) for x and
                (num_positions, num_coefficients) for y.
        Returns:
            Green's function at r, r_prime.
        """

        c = self.config

        # Get nn output of coefficient net.
        coefficients = CoefficientNet(c.coefficient_net)(r, coefficient_fn)

        # Green's function inputs.
        inputs = jnp.concatenate([r, r_prime, coefficients])

        # MLP
        outputs = MLP(
            c.green_function_mlp.widths,
            name="green_function_mlp",
        )(inputs)
        # Wrap with exponential function to keep it non-negative.
        outputs = jnp.exp(outputs)

        return outputs


class CoefficientNet(hk.Module):
    """Coefficient functions as inputs of Green's function."""

    def __init__(
        self, config: ConfigDict, name: Optional[str] = "coefficient_net"
    ):
        super().__init__(name=name)

        self.config = config

    def __call__(self, r: jnp.ndarray, coefficient_fn: F) -> jnp.ndarray:
        """Compute coefficients of the equation as the inputs of Green's function.

        Args:
            r: Spatial position with dimention d.shape is (d,).
            coefficient_fn: Coefficient function at as a NamedTuple (x, y)
                where x are the positions and y are corresponding values.
                Shapes are (num_positions, d) for x and
                (num_positions, num_coefficients) for y.
        Returns:
            Coefficient information at spatial position r.
            Shape (num_coefficients,) or ().
        """

        c = self.config
        coeff_positions, coeff_values = coefficient_fn.x, coefficient_fn.y

        x, v = r[:2], r[2:]
        angles_global = v / jnp.sqrt(jnp.sum(v ** 2, axis=-1) + 1e-16)
        rel_vec = x - coeff_positions
        rel_dist2 = jnp.sqrt(jnp.sum(rel_vec ** 2, axis=-1) + 1e-16)

        positions_local = jnp.matmul(rel_vec, angles_global)
        angles_local = positions_local / (rel_dist2 + 1e-8)
        frames_local = jnp.stack((angles_local, positions_local), axis=-1)

        attn_logits_net = MLP(
            c.attention_net.widths,
            name="attention_net",
        )

        def attn_logits_fn(q, k):
            qk = jnp.concatenate([q, k])
            attn_logits_per_example = attn_logits_net(qk)
            return attn_logits_per_example

        attn_logits = vmap(
            attn_logits_fn, argnums={1}, use_hk=True, out_axes=-1
        )(r, frames_local)
        masked_attn_logits = jnp.where(positions_local > 0, attn_logits, -1e30)

        attn_weights = jax.nn.softmax(masked_attn_logits)
        attn = jnp.matmul(attn_weights, coeff_values)
        # Take exponential w.r.t to sigma_a
        attn = jnp.exp(-attn)

        # Point-wise MLP
        coeff_outputs = MLP(c.pointwise_mlp.widths, name="pointwise_mlp")(attn)

        return coeff_outputs
