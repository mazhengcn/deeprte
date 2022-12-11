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

import functools

import haiku as hk
import jax
import jax.numpy as jnp

from deeprte.data.pipeline import FeatureDict
from deeprte.model import mapping_v2, nets
from deeprte.model.geometry.characteristics import Characteristics
from deeprte.model.tf.rte_features import (
    _BATCH_FEATURE_NAMES,
    _COLLOCATION_FEATURE_NAMES,
)

# from deeprte.


def glorot_uniform():
    return hk.initializers.VarianceScaling(
        scale=1.0, mode="fan_avg", distribution="uniform"
    )


def mean_squared_loss_fn(x, y, axis=None):
    return jnp.mean(jnp.square(x - y), axis=axis)


def make_in_axes(batch_keys, template):
    return [{k: 0 if k in template else None for k in batch_keys}]


def apply_dropout(*, tensor, safe_key, rate, is_training, broadcast_dim=None):
    """Applies dropout to a tensor."""
    if is_training and rate != 0.0:
        shape = list(tensor.shape)
        if broadcast_dim is not None:
            shape[broadcast_dim] = 1
        keep_rate = 1.0 - rate
        keep = jax.random.bernoulli(safe_key.get(), keep_rate, shape=shape)
        return keep * tensor / keep_rate
    else:
        return tensor


def dropout_wrapper(
    module,
    input_act,
    kernel,
    safe_key,
    global_config,
    output_act=None,
    is_training=True,
    **kwargs,
):
    """Applies module + dropout + residual update."""
    if output_act is None:
        output_act = input_act

    gc = global_config
    residual = module(input_act, kernel, is_training=is_training, **kwargs)
    dropout_rate = 0.0 if gc.deterministic else module.config.dropout_rate

    # if module.config.shared_dropout:
    #     if module.config.orientation == "per_row":
    #         broadcast_dim = 0
    #     else:
    #         broadcast_dim = 1
    # else:
    broadcast_dim = None

    residual = apply_dropout(
        tensor=residual,
        safe_key=safe_key,
        rate=dropout_rate,
        is_training=is_training,
        broadcast_dim=broadcast_dim,
    )

    new_act = output_act + residual

    return new_act


class DeepRTE(hk.Module):
    def __init__(self, config, global_config, name="deeprte"):
        super().__init__(name)

        self.config = config
        self.global_config = global_config

    def __call__(self, batch, is_training, compute_loss, compute_metrics):
        c = self.config
        gc = self.global_config
        ret = {}

        def rte_op(batch):
            green_fn = GreenFunction(c.green_function, gc)
            quadratures = (
                batch["boundary_coords"],
                batch["boundary"] * batch["boundary_weights"],
            )
            rte_sol = mapping_v2.quad(
                green_fn,
                quadratures=quadratures,
                argnum=1,
            )(batch["phase_coords"], batch)

            return rte_sol

        collocation_axes = make_in_axes(
            batch.keys(), _COLLOCATION_FEATURE_NAMES
        )
        batch_axes = make_in_axes(batch.keys(), _BATCH_FEATURE_NAMES)

        batch_rte_op = hk.vmap(
            mapping_v2.sharded_map(
                rte_op,
                shard_size=(
                    None
                    if is_training or hk.running_init()
                    else gc.sub_collocation_size
                ),
                in_axes=collocation_axes,
            ),
            in_axes=(batch_axes,),
            split_rng=(not hk.running_init()),
        )
        predictions = batch_rte_op(batch)
        ret["rte_predictions"] = predictions

        if compute_loss:
            labels = batch["psi_label"]
            loss = mean_squared_loss_fn(predictions, labels)
            ret["loss"] = {
                "mse": loss,
                "rmspe": jnp.sqrt(loss / jnp.mean(labels**2)),
            }

        if compute_metrics:
            labels = batch["psi_label"]
            # Compute relative mean squared error, this values will be summed and
            # finally divided by num_examples.
            mse = mean_squared_loss_fn(predictions, labels, axis=-1)
            relative_mse = mse / jnp.mean(labels**2)
            ret["metrics"] = {"mse": mse, "rmspe": relative_mse}

        if compute_loss:
            return loss, ret

        return ret


class GreenFunction(hk.Module):
    def __init__(self, config, global_config, name="green_function"):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    def __call__(self, coords_1, coords_2, batch: FeatureDict):
        c = self.config
        gc = self.global_config
        width = gc.latent_dims

        charc = Characteristics.from_tensor(batch["position_coords"])
        att_mod = AttenuationModule(c.attenuation_module, gc)
        act = att_mod(
            coords_1=coords_1,
            coords_2=coords_2,
            att_coeff=batch["sigma"],
            charc=charc,
        )
        proj_weights = hk.get_parameter(
            name="proj_weights", shape=[width], init=glorot_uniform()
        )

        if c.scattering_module.res_block_depth == 0:
            return jnp.einsum("...i, i->...", act, proj_weights)

        position, _ = jnp.split(coords_1, 2, axis=-1)

        def self_att_fn(velocity):
            coord = jnp.concatenate([position, velocity], axis=-1)
            out = att_mod(
                coords_1=coord,
                coords_2=coords_2,
                att_coeff=batch["sigma"],
                charc=charc,
            )
            return out

        velocity_coords, velocity_weights = (
            batch["velocity_coords"],
            batch["velocity_weights"],
        )
        kernel = (1 - batch["scattering_kernel"]) * velocity_weights
        self_kernel = (1 - batch["self_scattering_kernel"]) * velocity_weights

        self_act = hk.vmap(self_att_fn, split_rng=(not hk.running_init()))(
            velocity_coords
        )  # [N_v*, N_latent]
        act_output, _ = ScatteringModule(c.scattering_module, gc)(
            act=act, self_act=self_act, kernel=kernel, self_kernel=self_kernel
        )

        return jnp.einsum("...i, i->...", act_output, proj_weights)


class ScatteringModule(hk.Module):
    def __init__(self, config, global_config, name="scattering_module"):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    def __call__(
        self,
        act,
        self_act,
        kernel,
        self_kernel,
        is_training=True,
        safe_key=None,
    ):
        c = self.config
        gc = self.global_config

        dropout_wrapper_fn = functools.partial(
            dropout_wrapper,
            is_training=is_training,
            global_config=gc,
        )

        def scattering_fn(x):
            act, self_act = x

            scattering_layer = ScatteringLayer(c, gc)
            act_out = dropout_wrapper_fn(
                module=scattering_layer,
                input_act=self_act,
                output_act=act,
                safe_key=None,
                kernel=kernel,
            )
            self_act_out = dropout_wrapper_fn(
                module=scattering_layer,
                input_act=self_act,
                safe_key=None,
                kernel=self_kernel,
            )
            return act_out, self_act_out

        scattering_stack = hk.experimental.layer_stack(c.res_block_depth)(
            scattering_fn
        )
        act_output, self_act_output = scattering_stack((act, self_act))

        return act_output, self_act_output


class ScatteringLayer(hk.Module):
    def __init__(self, config, global_config, name="scattering_layer"):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    def __call__(self, activations, kernel, is_training=True):
        # shape [N_v*, N_latent]  # shape [..., N_v*]
        c = self.config
        gc = self.global_config

        width = gc.latent_dims

        weights = hk.get_parameter(
            name="scattering_weights",
            shape=[width, width],
            init=glorot_uniform(),
        )
        bias = hk.get_parameter(
            name="scattering_bias",
            shape=[width],
            init=hk.initializers.Constant(0.0),
        )

        out = jnp.einsum("...j,jk->...k", kernel, activations)
        out = jax.nn.tanh(jnp.einsum("ik,...k->...i", weights, out) + bias)

        return out


class AttenuationModule(hk.Module):
    """Attenuation operator module of the tranport equation."""

    def __init__(self, config, global_config, name="attenuation_module"):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    def __call__(self, coords_1, coords_2, att_coeff, charc, is_training=True):
        """
        Args:
            coords_1d: Phase space coordinates that is the concat of
                position and velocity coordinates, shape (2d,)
            coords_2d: Phase space coordinates that is the concat of
                position and velocity coordinates, shape (2d,)
            att_coeff: Attenuation coefficient on a grid, shape (num_positions,d)
                Shapes are (num_positions, d) for x and
        Returns:
            Green's function at r, r_prime.
        """
        c = self.config
        gc = self.global_config

        attention_module = Attention(c.attention, gc)
        local_coords, mask = charc.apply_to_point(coords_1)

        att = attention_module(
            query=coords_1,
            keys=local_coords,
            values=att_coeff,
            mask=mask,
        )
        att = jnp.exp(-att)

        widths = c.attenuation_block.widths
        num_layer = c.attenuation_block.num_layer
        out_widths = gc.latent_dims
        layer_widths = [widths] * num_layer + [out_widths]

        mlp_inputs = jnp.concatenate([coords_1, coords_2, att])
        out = nets.MLP(
            layer_widths,
            name="attenuation_mlp",
        )(mlp_inputs)
        out = jnp.exp(out)

        return out


class Attention(hk.Module):
    """Coefficient functions as inputs of Green's function."""

    def __init__(self, config, global_config, name="attention"):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    def __call__(self, query, keys, values, mask, is_training=True):
        """
        Args:
            q_data: (2d,) or (N, 2d)
            k_data: (M, d)
            v_data: (M, c)
        Returns:
            (c,) or (N, c)
        """
        c = self.config
        key_dim = c.widths

        q_weights = hk.get_parameter(
            "query_w", shape=(query.shape[-1], key_dim), init=glorot_uniform()
        )
        k_weights = hk.get_parameter(
            "key_w", shape=(keys.shape[-1], key_dim), init=glorot_uniform()
        )
        bias = hk.get_parameter(
            "b", shape=(key_dim,), init=hk.initializers.Constant(0.0)
        )
        proj_weights = hk.get_parameter(
            "proj_w", shape=(key_dim,), init=glorot_uniform()
        )

        q = jnp.einsum("...d,dc->...c", query, q_weights)
        k = jnp.einsum("nd,dc->nc", keys, k_weights)
        qk = jax.nn.tanh(q[..., None, :] + k + bias)
        logits = jnp.einsum("...nc,c->...n", qk, proj_weights)

        masked_logits = jnp.where(mask, logits, -1e30)
        weights = jax.nn.softmax(masked_logits)
        weighted_avg = jnp.einsum("...n,nc->...c", weights, values)

        return weighted_avg
