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

from deeprte.data.pipeline import FeatureDict
from deeprte.model.integrate import quad
from deeprte.model.layer_stack import layer_stack
from deeprte.model.mapping import vmap
from deeprte.model.networks import MLP
from deeprte.model.tf.rte_features import (
    _BATCH_FEATURE_NAMES,
    _COLLOCATION_FEATURE_NAMES,
    NUM_DIM,
)


def dropout_wrapper(module, input_act, output_act=None, **kwargs):
    """Applies module + dropout + residual update."""
    if output_act is None:
        output_act = input_act

    residual = module(input_act, **kwargs)

    new_act = output_act + residual

    return new_act


def glorot_uniform():
    return hk.initializers.VarianceScaling(
        scale=1.0, mode="fan_avg", distribution="uniform"
    )


def mean_squared_loss_fn(x, y, axis=None):
    return jnp.mean(jnp.square(x - y), axis=axis)


def make_batch_axes_dict(batch):
    return {k: 0 if k in _BATCH_FEATURE_NAMES else None for k in batch}


def make_collocation_axes_dict(batch):
    return {k: 0 if k in _COLLOCATION_FEATURE_NAMES else None for k in batch}


class DeepRTE(hk.Module):
    def __init__(self, config, name: Optional[str] = "DeepRTE"):
        super().__init__(name)

        self.config = config
        self.batch_axes_dict = None
        self.collocation_axes_dict = None

    def __call__(
        self,
        batch: FeatureDict,
        is_training: bool,
        compute_loss: bool,
        compute_metrics: bool,
    ):
        ret = {}

        # rte_module = RTEModel(self.config)

        def rte_module(batch: FeatureDict) -> jax.Array:
            green_func_module = GreenFunction(
                self.config.green_function,
            )
            sol = quad(
                green_func_module,
                (
                    batch["boundary_coords"],
                    batch["boundary"] * batch["boundary_weights"],
                ),
                argnum=1,
            )(batch["phase_coords"], batch)

            return sol

        # if is_training:
        if not self.batch_axes_dict:
            self.batch_axes_dict = make_batch_axes_dict(batch)
        if not self.collocation_axes_dict:
            self.collocation_axes_dict = make_collocation_axes_dict(batch)

        if is_training:
            rte_module = hk.vmap(
                hk.vmap(
                    rte_module,
                    in_axes=(self.collocation_axes_dict,),
                    split_rng=(not hk.running_init()),
                ),
                in_axes=(self.batch_axes_dict,),
                split_rng=(not hk.running_init()),
            )
        else:
            rte_module = hk.vmap(
                vmap(
                    rte_module,
                    shard_size=128,
                    in_axes=(self.collocation_axes_dict,),
                ),
                in_axes=(self.batch_axes_dict,),
                split_rng=(not hk.running_init()),
            )

        ret["rte_predictions"] = rte_module(batch)

        if compute_metrics:
            metrics = self.metrics(ret, batch)
            ret["metrics"] = metrics

        if compute_loss:
            loss = self.loss(ret, batch)
            ret["loss"] = loss

            # total_loss = loss["mse"]
            # return ret, total_loss

        return ret

    def loss(self, value, batch):
        pred = value["rte_predictions"]
        label = batch["psi_label"]

        mse = mean_squared_loss_fn(pred, label)
        rmspe = jnp.sqrt(mse / jnp.mean(label**2))

        return dict(mse=mse, rmspe=rmspe)

    def metrics(self, value, batch):
        pred = value["rte_predictions"]
        label = batch["psi_label"]

        mse = mean_squared_loss_fn(pred, label, axis=-1)
        rmspe = jnp.sqrt(mse / jnp.mean(label**2))

        return dict(mse=mse, rmspe=rmspe)


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
        batch: FeatureDict,
    ) -> jax.Array:

        c = self.config

        x, v = coords[:NUM_DIM], coords[NUM_DIM:]
        x_prime, v_prime = (
            coords_prime[:NUM_DIM],
            coords_prime[NUM_DIM:],
        )
        width = c.scatter_model.transport_model.transport_block_mlp.widths[-1]
        trans_module = TransportModule(c.scatter_model.transport_model)

        green_fn_output = trans_module(
            x,
            v,
            x_prime,
            v_prime,
            batch["position_coords"],
            batch["sigma"],
        )
        out_layer_weights = hk.get_parameter(
            name="out_layer_weights",
            shape=[
                width,
            ],
            init=glorot_uniform(),
        )

        if c.scatter_model.res_block_depth == 0:
            return jnp.einsum("i,i", green_fn_output, out_layer_weights)

        # prepare inputs
        res_weights_vstar = (1 - batch["self_scattering_kernel"]) * batch[
            "velocity_weights"
        ]
        res_weights_v = (1 - batch["scattering_kernel"]) * batch[
            "velocity_weights"
        ]

        act_v = green_fn_output
        act_vstar = vmap(trans_module, argnums=frozenset([1]), use_hk=True,)(
            x,
            batch["velocity_coords"],
            x_prime,
            v_prime,
            batch["position_coords"],
            batch["sigma"],
        )  # shape: [N_v*, N_latent]

        # stack layer
        if c.scatter_model.res_block_depth > 1:

            def block(x):
                scattering_module = ScatteringModule(c.scatter_model)
                act_v, act_vstar = x

                act_v = dropout_wrapper(
                    module=scattering_module,
                    input_act=act_vstar,
                    res_weights=res_weights_v,
                    output_act=act_v,
                )

                act_vstar = dropout_wrapper(
                    module=scattering_module,
                    input_act=act_vstar,
                    res_weights=res_weights_vstar,
                )

                return act_v, act_vstar

            res_stack = layer_stack(c.scatter_model.res_block_depth - 1)(block)

            act_v, act_vstar = res_stack((act_v, act_vstar))

        scattering_module = ScatteringModule(c.scatter_model)

        green_fn_output = dropout_wrapper(
            module=scattering_module,
            input_act=act_vstar,
            res_weights=res_weights_v,
            output_act=act_v,
        )

        return jnp.einsum("i,i", green_fn_output, out_layer_weights)


class ScatteringModule(hk.Module):
    def __init__(
        self,
        config,
        name: Optional[str] = "scattering_module",
    ):
        super().__init__(name=name)

        self.config = config

    def __call__(
        self,
        act: jax.Array,  # shape [N_v*, N_latent]
        res_weights: jax.Array,  # shape [..., N_v*]
    ) -> jax.Array:

        c = self.config

        width = c.transport_model.transport_block_mlp.widths[-1]
        weights = hk.get_parameter(
            name="weights", shape=[width, width], init=glorot_uniform()
        )
        bias = hk.get_parameter(
            name="bias",
            shape=[
                width,
            ],
            init=glorot_uniform(),
        )

        residue = jnp.einsum("...j,jk->...k", res_weights, act)
        residue = jax.nn.tanh(
            jnp.einsum("ik,...k->...i", weights, residue) + bias
        )

        return residue


class TransportModule(hk.Module):
    """Green's function transport block of solution operator."""

    def __init__(
        self,
        config: ml_collections.ConfigDict,
        name: Optional[str] = "transport_module",
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

        attn_logits = hk.vmap(
            attn_logits_fn,
            in_axes=(None, 0),
            out_axes=-1,
            split_rng=(not hk.running_init()),
        )(coords, frames_local)

        masked_attn_logits = jnp.where(positions_local > 0, attn_logits, -1e30)

        attn_weights = jax.nn.softmax(masked_attn_logits)
        attn = jnp.matmul(attn_weights, coeff_values)
        # Take exponential w.r.t to sigma_a
        attn = jnp.exp(-attn)

        return attn
