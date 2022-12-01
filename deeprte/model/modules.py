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

from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections

from deeprte.model.integrate import quad
from deeprte.model.mapping import vmap
from deeprte.model.networks import MLP, Linear
from deeprte.model.tf.rte_dataset import TensorDict
from deeprte.model.tf.rte_features import NUM_DIM


class GreenFunctionResBlock(hk.Module):
    """Green function resnet block with scattering kernel."""

    def __init__(
        self,
        config: ml_collections.ConfigDict,
        name: Optional[str] = "green_function_res_block",
    ):
        super().__init__(name=name)

        self.config = config

    def __call__(
        self,
        coords: jnp.ndarray,
        scattering_kernel: jnp.ndarray,
        batch: TensorDict,
    ) -> jnp.ndarray:

        c = self.config

        green_func_module = GreenFunctionNet(c.green_function)

        green_fn_block_output = green_func_module(
            coords, batch["boundary_coords"], coeff_coords, coeff_values
        )

        if c.green_res_block.depth > 0:

            r_star, weights = scattering_kernel_coords[2:], scattering_kernel
            # weights = (1 - scattering_kernel_coeff) * weights

            for _ in range(c.green_res_block.depth):
                green_fn_kernel_quad = quad(
                    green_func_module, (r_star, weights), argnum=0, use_hk=True
                )(
                    r_prime, coeff_coords, coeff_values
                )  # shape: [N']
                green_fn_block_output += Linear()(
                    green_fn_kernel_quad,
                )

        return green_fn_block_output  # shape: [N']


class GreenFunctionNet(hk.Module):
    """Green function net."""

    def __init__(
        self,
        config: ml_collections.ConfigDict,
        name: Optional[str] = "green_function_res_block",
    ):
        super().__init__(name=name)

        self.config = config

    def __call__(
        self,
        phase_coords: jax.Array,
        boundary_coords: jax.Array,
        # scattering_kernel_coeff: jnp.ndarray,  # [Nv*,]
        coeff_position: jax.Array,
        coeff_values: jax.Array,
        scattering_kernel_coords: jax.Array,  # ((u,u*):[Nv*,4], (1-P(u,u*))*omega:[Nv*,])
        scattering_kernel: jax.Array,
        batch: TensorDict,
    ) -> jax.Array:

        c = self.config

        green_func_module = GreenFunctionBlock(c)

        inputs = (
            phase_coords[:NUM_DIM],
            phase_coords[NUM_DIM:],
            boundary_coords[:NUM_DIM],
            boundary_coords[NUM_DIM:],
            coeff_position,
            coeff_values,
        )

        green_fn_block_output = green_func_module(*inputs)

        if c.green_res_block.depth > 0:

            coords_star, weights = (
                scattering_kernel_coords,
                (1 - scattering_kernel) * batch["velocity_weights"],
            )
            res_block_inputs = (*inputs[:3], coeff_position, coeff_values)

            def _green_res_fn(green_fn):
                def func(*inputs):
                    green_fn_kernel_quad = quad(
                        green_fn,
                        (coords_star, weights),
                        argnum=3,
                        use_hk=True,
                    )(
                        *inputs[:3], coeff_position, coeff_values
                    )  # shape: [N_latent]
                    green_fn_res = MLP(
                        c.green_function_mlp.widths[-1:],
                        activate_final=True,
                    )(green_fn_kernel_quad)
                    return green_fn_res

                return func

            def _green_res_block(block_output, green_func_module):
                func = _green_res_fn(green_func_module)
                block_output += func(
                    *res_block_inputs,
                )
                green_func_module = func

                return block_output

            for _ in range(c.green_res_block.depth):

                func = _green_res_fn(green_func_module)
                green_fn_block_output += func(
                    *res_block_inputs,
                )
                green_func_module = func

        green_fn_block_output = MLP([1])(
            green_fn_block_output,
        )
        return green_fn_block_output  # shape: [1]


class GreenFunctionBlock(hk.Module):
    """Green's function block of solution operator."""

    def __init__(
        self,
        config: ml_collections.ConfigDict,
        name: Optional[str] = "green_function",
    ):
        super().__init__(name=name)

        self.config = config

    def __call__(
        self,
        position: jax.Array,
        velocity: jax.Array,  # pylint:disable=invalid-name
        position_prime: jax.Array,
        velocity_prime: jax.Array,
        coeff_position: jax.Array,
        coeff_values: jax.Array,
    ) -> jax.Array:
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

        c = self.config  # pylint: disable=invalid-name

        # Get nn output of coefficient net.
        coefficients = CoefficientNet(c.coefficient_net)(
            position,
            velocity,
            coeff_position,
            coeff_values,
        )

        # Green's function inputs.
        inputs = jnp.concatenate(
            [
                position,
                velocity,
                position_prime,
                velocity_prime,
                coefficients,
            ]
        )

        # inputs = hk.LayerNorm(axis=[-1],
        # create_scale=True, create_offset=True)(inputs)

        # MLP
        outputs = MLP(
            c.green_function_mlp.widths,
            activate_final=True,
            name="green_function_block_mlp",
        )(inputs)

        # Wrap with exponential function to keep it non-negative.
        outputs = jnp.exp(outputs)

        return outputs


class CoefficientNet(hk.Module):
    """Coefficient functions as inputs of Green's function."""

    def __init__(
        self,
        config: ml_collections.ConfigDict,
        name: Optional[str] = "coefficient_net",
    ):
        super().__init__(name=name)

        self.config = config

    def __call__(
        self,
        position: jax.Array,
        velocity: jax.Array,
        coeff_position: jax.Array,
        coeff_values: jax.Array,
    ) -> jax.Array:
        """Compute coefficients of the equation as the inputs of
        Green's function.

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

        x, v = position, velocity
        coords = jnp.concatenate([x, v])
        angles_global = v / jnp.sqrt(jnp.sum(v**2, axis=-1) + 1e-16)
        rel_vec = x - coeff_position
        rel_dist2 = jnp.sqrt(jnp.sum(rel_vec**2, axis=-1) + 1e-16)

        positions_local = jnp.matmul(rel_vec, angles_global)
        angles_local = positions_local / (rel_dist2 + 1e-8)
        frames_local = jnp.stack((angles_local, positions_local), axis=-1)

        attn_mod = MLP(c.attention_net.widths, name="attention_net")

        def attn_logits_fn(q, k):
            qk = jnp.concatenate([q, k])
            attn_logits_per_example = attn_mod(qk)
            return attn_logits_per_example

        attn_logits = vmap(
            attn_logits_fn, argnums=frozenset([1]), use_hk=True, out_axes=-1
        )(coords, frames_local)

        masked_attn_logits = jnp.where(positions_local > 0, attn_logits, -1e30)

        attn_weights = jax.nn.softmax(masked_attn_logits)
        attn = jnp.matmul(attn_weights, coeff_values)
        # Take exponential w.r.t to sigma_a
        attn = jnp.exp(-attn)

        return attn
