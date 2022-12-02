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
from deeprte.model.layer_stack import layer_stack
from deeprte.model.mapping import vmap
from deeprte.model.networks import MLP, Linear
from deeprte.model.tf.rte_dataset import TensorDict
from deeprte.model.tf.rte_features import NUM_DIM


class GreenFunction(hk.Module):
    def __init__(
        self,
        config,
        name: Optional[str] = "green_function",
    ):
        super().__init__(name=name)

        self.config = config

    def __call__(
        self,
        coords: jax.Array,
        coords_prime: jax.Array,
        scattering_kernel: jax.Array,
        batch: TensorDict,
    ) -> jax.Array:

        c = self.config

        x, v = coords[:NUM_DIM], coords[NUM_DIM:]
        x_prime, v_prime = (
            coords_prime[:NUM_DIM],
            coords_prime[NUM_DIM:],
        )

        green_fn_module = TransportModel(c.scatter_model.transport_model)
        green_fn_output = green_fn_module(
            x,
            v,
            x_prime,
            v_prime,
            batch["position_coords"],
            batch["sigma"],
        )

        if c.scatter_model.res_block_depth > 0:
            scatter_func_module = ScatterModel(c.scatter_model)
            scatter_block_output = scatter_func_module(
                x,
                x_prime,
                v_prime,
                batch,
            )
            weights = (1 - scattering_kernel) * batch["velocity_weights"]
            if c.scatter_model.res_block_depth == 1:
                expr = "j,jk->k"
            else:
                expr = "j,ijk->k"

            green_fn_output += jnp.einsum(
                expr,
                weights,
                scatter_block_output,
            )

        green_fn_output = MLP([1])(green_fn_output)

        return green_fn_output


class ScatterModel(hk.Module):
    def __init__(
        self,
        config,
        name: Optional[str] = "scatter_model",
    ):
        super().__init__(name=name)

        self.config = config

    def __call__(
        self,
        x: jax.Array,
        x_prime: jax.Array,
        v_prime: jax.Array,
        batch: TensorDict,
    ) -> jax.Array:

        c = self.config

        green_func_module = TransportModel(c.transport_model)

        res_weights = (1 - batch["self_scattering_kernel"]) * batch["velocity_weights"]

        _res_block_output = vmap(
            green_func_module,
            argnums=frozenset([1]),
            use_hk=True,
        )(
            x,
            batch["velocity_coords"],
            x_prime,
            v_prime,
            batch["position_coords"],
            batch["sigma"],
        )  # shape: [N_v*, N_latent]
        _output = _res_block_output
        if c.res_block_depth > 1:

            def _res_block(res_block_output):
                _res = jnp.einsum(
                    "ij,jk->ik",
                    res_weights,
                    res_block_output,
                )
                _res = MLP(
                    c.transport_model.transport_block_mlp.widths[-1:],
                    activate_final=True,
                )(_res)

                res_block_output += _res

                return res_block_output, res_block_output

            _res_block_output, zs = layer_stack(
                c.res_block_depth - 1,
                with_state=True,
            )(_res_block)(_res_block_output)
            _output = jnp.concatenate((_output[jnp.newaxis, ...], zs))

        return _output


class TransportModel(hk.Module):
    """Green's function transport block of solution operator."""

    def __init__(
        self,
        config: ml_collections.ConfigDict,
        name: Optional[str] = "transport_model",
    ):
        super().__init__(name=name)

        self.config = config

    def __call__(
        self,
        x: jax.Array,
        v: jax.Array,  # pylint:disable=invalid-name
        x_prime: jax.Array,
        v_prime: jax.Array,
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
            x,
            v,
            coeff_position,
            coeff_values,
        )

        # Green's function inputs.
        inputs = jnp.concatenate(
            [
                x,
                v,
                x_prime,
                v_prime,
                coefficients,
            ]
        )

        # inputs = hk.LayerNorm(axis=[-1],
        # create_scale=True, create_offset=True)(inputs)

        # MLP
        outputs = MLP(
            c.transport_block_mlp.widths,
            activate_final=True,
            name="transport_block_mlp",
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
        x: jax.Array,
        v: jax.Array,
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
