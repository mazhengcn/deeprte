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

"""Core modules including Green's function with Attenuation and Scattering."""

import functools
from typing import Optional

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import config_dict

from deeprte.data import pipeline
from deeprte.model import integrate, mapping
from deeprte.model.characteristics import Characteristics
from deeprte.model.tf import rte_features
from deeprte.model.utils import (
    dropout_wrapper,
    get_initializer_scale,
    mean_squared_loss_fn,
    query_chunk_attention,
)

FEATURES = rte_features.FEATURES
COLLOCATION_AXES = {
    k: 0 if (rte_features.NUM_PHASE_COORDS in FEATURES[k][1]) else None
    for k in FEATURES
}


class DeepRTE(hk.Module):
    """Deep RTE model."""

    def __init__(
        self, config: config_dict.ConfigDict, name: Optional[str] = "deeprte"
    ) -> None:
        super().__init__(name=name)
        self.config = config

    def __call__(
        self,
        batch: pipeline.FeatureDict,
        is_training: bool,
        compute_loss: bool = False,
        compute_metrics: bool = False,
    ) -> (
        dict[str, dict[str, chex.Array]]
        | tuple[float, dict[str, dict[str, chex.Array]]]
    ):
        c = self.config
        gc = self.config.global_config
        ret = {}

        def rte_op(inputs):
            green_fn = GreenFunction(c.green_function, gc)
            quadratures = (
                inputs["boundary_coords"],
                inputs["boundary"] * inputs["boundary_weights"],
            )
            rte_sol = integrate.quad(green_fn, quadratures=quadratures, argnum=1)(
                inputs["phase_coords"], inputs, is_training
            )
            return rte_sol

        low_memory = (
            None if is_training or hk.running_init() else gc.subcollocation_size
        )
        batched_rte_op = hk.vmap(
            mapping.sharded_map(
                rte_op, shard_size=low_memory, in_axes=(COLLOCATION_AXES,)
            ),
            split_rng=(not hk.running_init()),
        )

        rte_inputs = {k: batch[k] for k in FEATURES}
        predictions = batched_rte_op(rte_inputs)
        ret.update({"predicted_psi": predictions})

        if compute_loss:
            interior_labels = batch["psi_label"]
            interior_loss = mean_squared_loss_fn(predictions, interior_labels)
            total_loss = gc.loss_weights * interior_loss
            ret["loss"] = {
                "interior_mse": interior_loss,
                "interior_rmspe": 100
                * jnp.sqrt(interior_loss / jnp.mean(interior_labels**2)),
            }
            if "sampled_boundary_coords" in batch:
                rte_inputs["phase_coords"] = batch["sampled_boundary_coords"]
                rte_inputs["scattering_kernel"] = batch[
                    "sampled_boundary_scattering_kernel"
                ]

                boundary_labels = batch["sampled_boundary"]
                boundary_predictions = batched_rte_op(rte_inputs)
                boundary_loss = mean_squared_loss_fn(
                    boundary_predictions, boundary_labels
                )
                total_loss += gc.bc_loss_weights * boundary_loss
                ret["loss"].update(
                    {
                        "boundary_mse": boundary_loss,
                        "boundary_rmspe": jnp.sqrt(
                            boundary_loss / jnp.mean(boundary_labels**2)
                        ),
                    }
                )
            ret["loss"].update({"total": total_loss})

        if compute_metrics:
            labels = batch["psi_label"]
            mse = mean_squared_loss_fn(predictions, labels, axis=-1)
            relative_mse = mse / jnp.mean(labels**2)
            ret.update({"metrics": {"mse": mse, "rmspe": relative_mse}})

        if compute_loss:
            return total_loss, ret

        return ret


class GreenFunction(hk.Module):
    """Green's function module."""

    def __init__(
        self,
        config: config_dict.ConfigDict,
        global_config: config_dict.ConfigDict,
        name: Optional[str] = "green_function",
    ) -> None:
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    def __call__(
        self,
        coord1: chex.Array,
        coord2: chex.Array,
        batch: pipeline.FeatureDict,
        is_training: bool,
    ) -> chex.Array:
        c = self.config
        gc = self.global_config

        w_init = get_initializer_scale(gc.w_init)
        projection = hk.Linear(
            1, with_bias=False, w_init=w_init, name="output_projection"
        )

        charc = Characteristics.from_tensor(batch["position_coords"])
        attenuation_module = Attenuation(c.attenuation, gc)
        act = attenuation_module(
            coord1=coord1,
            coord2=coord2,
            att_coeff=batch["sigma"],
            charc=charc,
        )

        if c.scattering.num_layer == 0:
            output = jnp.exp(projection(act))
            output = jnp.squeeze(output, axis=-1)
            return output

        position, _ = jnp.split(coord1, 2, axis=-1)

        def self_att_fn(velocity):
            coord = jnp.concatenate([position, velocity], axis=-1)
            out = attenuation_module(
                coord1=coord,
                coord2=coord2,
                att_coeff=batch["sigma"],
                charc=charc,
            )
            return out

        velocity_coords, velocity_weights = (
            batch["velocity_coords"],
            batch["velocity_weights"],
        )
        kernel = (-batch["scattering_kernel"]) * velocity_weights
        self_kernel = (-batch["self_scattering_kernel"]) * velocity_weights

        self_act = hk.vmap(self_att_fn, split_rng=(not hk.running_init()))(
            velocity_coords
        )
        output = Scattering(c.scattering, gc)(
            act=act,
            self_act=self_act,
            kernel=kernel,
            self_kernel=self_kernel,
            is_training=is_training,
        )
        output = jnp.exp(projection(output))
        output = jnp.squeeze(output, axis=-1)

        return output


class ScatteringV1(hk.Module):
    """Scattering module."""

    def __init__(
        self,
        config: config_dict.ConfigDict,
        global_config: config_dict.ConfigDict,
        name: Optional[str] = "scattering_module",
    ) -> None:
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    def __call__(
        self,
        act: chex.Array,
        self_act: chex.Array,
        kernel: chex.Array,
        self_kernel: chex.Array,
        is_training: bool,
    ) -> tuple[chex.Array, chex.Array]:
        c = self.config
        gc = self.global_config
        w_init = get_initializer_scale(gc.w_init)

        dropout_wrapper_fn = functools.partial(
            dropout_wrapper,
            is_training=is_training,
            safe_key=None,
            global_config=gc,
        )

        def scattering_block(x):
            act, self_act = x
            scattering_layer = ScatteringLayer(output_size=c.latent_dim, w_init=w_init)
            layer_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

            act_out = dropout_wrapper_fn(
                module=scattering_layer,
                input_act=self_act,
                kernel=kernel,
                output_act=act,
            )
            act_out = layer_norm(act_out)

            self_act_out = dropout_wrapper_fn(
                module=scattering_layer,
                input_act=self_act,
                kernel=self_kernel,
            )
            self_act_out = layer_norm(self_act_out)

            return (act_out, self_act_out)

        scattering_stack = hk.experimental.layer_stack(c.num_layer)(scattering_block)
        act_output, self_act_output = scattering_stack((act, self_act))

        return act_output, self_act_output


class Scattering(hk.Module):
    """Scattering module."""

    def __init__(
        self,
        config: config_dict.ConfigDict,
        global_config: config_dict.ConfigDict,
        name: Optional[str] = "scattering_module",
    ) -> None:
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    def __call__(
        self,
        act: chex.Array,
        self_act: chex.Array,
        kernel: chex.Array,
        self_kernel: chex.Array,
        is_training: bool,
    ) -> tuple[chex.Array, chex.Array]:
        c = self.config
        gc = self.global_config
        w_init = get_initializer_scale(gc.w_init)

        dropout_wrapper_fn = functools.partial(
            dropout_wrapper,
            is_training=is_training,
            safe_key=None,
            global_config=gc,
        )

        def scattering_block(x):
            self_act, self_act_out = x

            scattering_layer = ScatteringLayer(output_size=c.latent_dim, w_init=w_init)
            layer_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

            out = dropout_wrapper_fn(
                module=scattering_layer,
                input_act=self_act_out,
                kernel=self_kernel,
                output_act=self_act,
            )
            out = layer_norm(out)

            return (self_act, out)

        if c.num_layer == 1:
            self_act_output = self_act
        else:
            scattering_stack = hk.experimental.layer_stack(c.num_layer - 1)(
                scattering_block
            )
            _, self_act_output = scattering_stack((self_act, self_act))

        output_layer = ScatteringLayer(output_size=c.latent_dim, w_init=w_init)
        output_layer_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)

        act_out = dropout_wrapper_fn(
            module=output_layer,
            input_act=self_act_output,
            kernel=kernel,
            output_act=act,
        )
        act_out = output_layer_norm(act_out)

        return act_out


class ScatteringLayer(hk.Module):
    "A single layer describing scattering action."

    def __init__(
        self,
        output_size: int,
        with_bias: bool = True,
        w_init: int | None = None,
        name: Optional[str] = "scattering_layer",
    ) -> None:
        super().__init__(name=name)
        self.output_size = output_size
        self.with_bias = with_bias
        self.w_init = w_init

    def __call__(
        self, act: chex.Array, kernel: chex.Array, is_training: bool | None = None
    ) -> chex.Array:
        output = jnp.einsum("...V,Vd->...d", kernel, act)
        output = hk.Linear(
            output_size=self.output_size,
            with_bias=self.with_bias,
            w_init=self.w_init,
        )(output)
        output = jax.nn.tanh(output)

        return output


class Attenuation(hk.Module):
    """Attenuation operator module of the tranport equation."""

    def __init__(
        self,
        config: config_dict.ConfigDict,
        global_config: config_dict.ConfigDict,
        name: Optional[str] = "attenuation",
    ) -> None:
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    def __call__(
        self,
        coord1: chex.Array,
        coord2: chex.Array,
        att_coeff: chex.Array,
        charc: Characteristics,
    ) -> chex.Array:
        """Module that describes the attenuation part of RTE equation.

        Args:
            coords_1d: Phase space coordinates that is the concat of
                position and velocity coordinates; shape [2*d]
            coords_2d: Phase space coordinates that is the concat of
                position and velocity coordinates; shape [2*d]
            att_coeff: Attenuation coefficient on a grid;
                shape [num_grid_points, D_a]

        Returns:
            Green's function at r, r_prime.
        """
        c = self.config
        gc = self.global_config
        w_init = get_initializer_scale(gc.w_init)

        attention_module = Attention(c.attention, gc)
        local_coords, mask = charc.apply_to_point(coord1)

        att = attention_module(
            query=coord1, key=local_coords, value=att_coeff, mask=mask
        )
        att = jnp.exp(-att)

        act = jnp.concatenate([coord1, coord2, att])
        for _ in range(c.num_layer - 1):
            act = hk.Linear(c.latent_dim, w_init=w_init, name="attenuation_linear")(act)
            act = jax.nn.tanh(act)

        act = jax.nn.tanh(
            hk.Linear(c.output_dim, w_init=w_init, name="output_projection")(act)
        )

        return act


class Attention(hk.Module):
    """Multihead Attention."""

    def __init__(
        self,
        config: config_dict.ConfigDict,
        global_config: config_dict.ConfigDict,
        name: Optional[str] = "attention",
    ) -> None:
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    def __call__(
        self,
        query: chex.Array,
        key: chex.Array,
        value: chex.Array,
        mask: chex.Array | None = None,
    ) -> chex.Array:
        """Computes (optionally masked) MHA with queries, keys & values.

        Args:
          query: Sequence used to compute queries; shape [..., D_q].
          key: Sequence used to compute keys; shape [T, D_k].
          value: Sequence used to compute values; shape [T, D_v].
          mask: Optional mask applied to attention weights; shape [H=1, T', T].

        Returns:
          A new sequence of embeddings, consisting of a projection of the
            attention-weighted value projections; shape [..., D'].
        """
        c = self.config
        gc = self.global_config

        self.num_head = c.num_head
        self.key_dim = c.key_dim
        self.value_dim = c.value_dim or c.key_dim
        self.output_dim = c.output_dim or c.key_dim * c.num_head

        self.w_init = get_initializer_scale(gc.w_init)
        # In shape hints below, we suppress the leading dims [...] for brevity.
        # Hence e.g. [A, B] should be read in every case as [..., A, B].
        *leading_dims, _ = query.shape
        projection = self._linear_projection

        # Compute key/query/values (overload K/Q/V to
        # denote the respective sizes).
        query_heads = projection(query, self.key_dim, "query")  # [T', H, Q=K]
        key_heads = projection(key, self.key_dim, "key")  # [T, H, K]
        value_heads = projection(value, self.value_dim, "value")  # [T, H, V]

        # Compute attention weights.
        attn_logits = jnp.einsum("...hd,Thd->...hT", query_heads, key_heads)
        attn_logits = attn_logits / np.sqrt(self.key_dim).astype(key.dtype)
        if mask is not None:
            if mask.ndim != attn_logits.ndim:
                raise ValueError(
                    f"Mask dimensionality {mask.ndim} must match logits"
                    f"dimensionality {attn_logits.ndim}."
                )
            attn_logits = jnp.where(mask, attn_logits, -1e30)
        attn_weights = jax.nn.softmax(attn_logits)  # [T', H, T]

        # Weight the values by the attention and flatten the head vectors.
        attn = jnp.einsum("...hT,...Thd->...hd", attn_weights, value_heads)
        attn = jnp.reshape(attn, (*leading_dims, -1))  # [T', H*V]

        # Apply another projection to get the final embeddings.
        output_projection = hk.Linear(
            self.output_dim, w_init=self.w_init, name="output_projection"
        )
        return output_projection(attn)  # [T', D']

    @hk.transparent
    def _linear_projection(
        self, x: jax.Array, head_dim: int, name: Optional[str] = None
    ) -> jax.Array:
        y = hk.Linear(self.num_head * head_dim, w_init=self.w_init, name=name)(x)
        *leading_dims, _ = x.shape
        return y.reshape((*leading_dims, self.num_head, head_dim))


class Attention_v2(hk.Module):
    """Multihead Attention."""

    def __init__(
        self,
        config: config_dict.ConfigDict,
        global_config: config_dict.ConfigDict,
        name: Optional[str] = "attention_v2",
    ) -> None:
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    def __call__(self, query, key, value, mask=None):
        """Computes (optionally masked) MHA with queries, keys & values.

        Args:
          query: Sequence used to compute queries; shape [..., D_q].
          key: Sequence used to compute keys; shape [T, D_k].
          value: Sequence used to compute values; shape [T, D_v].
          mask: Optional mask applied to attention weights; shape [H=1, T', T].

        Returns:
          A new sequence of embeddings, consisting of a projection of the
            attention-weighted value projections; shape [..., D'].
        """
        c = self.config
        gc = self.global_config

        self.num_head = c.num_head
        self.key_dim = c.key_dim
        self.value_dim = c.value_dim or c.key_dim
        self.output_dim = c.output_dim or c.key_dim * c.num_head

        self.w_init = get_initializer_scale(gc.w_init)
        # In shape hints below, we suppress the leading dims [...] for brevity.
        # Hence e.g. [A, B] should be read in every case as [..., A, B].
        *leading_dims, _ = query.shape
        projection = self._linear_projection

        # Compute key/query/values (overload K/Q/V to
        # denote the respective sizes).
        query_heads = projection(query, self.key_dim, "query")  # [T', H, Q=K]
        key_heads = projection(key, self.key_dim, "key")  # [T, H, K]
        value_heads = projection(value, self.value_dim, "value")  # [T, H, V]

        attn = query_chunk_attention(
            query_heads,
            key_heads,
            value_heads,
            mask,
            key_chunk_size=None if hk.running_init() else c.key_chunk_size,
            precision=None,
            dtype=jnp.float32,
        )
        attn = jnp.reshape(attn, (*leading_dims, -1))  # [T', H*V]

        # Apply another projection to get the final embeddings.
        output_projection = hk.Linear(
            self.output_dim, w_init=self.w_init, name="output_projection"
        )
        return output_projection(attn)  # [T', D']

    @hk.transparent
    def _linear_projection(
        self, x: jax.Array, head_dim: int, name: Optional[str] = None
    ) -> jax.Array:
        y = hk.Linear(self.num_head * head_dim, w_init=self.w_init, name=name)(x)
        *leading_dims, _ = x.shape
        return y.reshape((*leading_dims, self.num_head, head_dim))
