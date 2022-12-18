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

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from deeprte.model import integrate, mapping
from deeprte.model.characteristics import Characteristics
from deeprte.model.tf.rte_features import (
    BATCH_FEATURE_NAMES,
    COLLOCATION_FEATURE_NAMES,
)
from deeprte.model.utils import (
    dropout_wrapper,
    get_initializer_scale,
    mean_squared_loss_fn,
    query_chunk_attention,
)


def get_vmap_axes(dict_keys: list[str], template: list[str]):
    return ({k: 0 if k in template else None for k in dict_keys},)


class DeepRTE(hk.Module):
    def __init__(self, config, name="deeprte"):
        super().__init__(name)

        self.config = config
        self.global_config = config.global_config

    def __call__(
        self, batch, is_training, compute_loss=False, compute_metrics=False
    ):
        c = self.config
        gc = self.global_config
        ret = {}

        def rte_op(batch):
            green_fn = GreenFunction(c.green_function, gc)
            quadratures = (
                batch["boundary_coords"],
                batch["boundary"] * batch["boundary_weights"],
            )
            rte_sol = integrate.quad(
                green_fn, quadratures=quadratures, argnum=1
            )(batch["phase_coords"], batch, is_training)
            return rte_sol

        low_memory = (
            None
            if is_training or hk.running_init()
            else gc.subcollocation_size
        )
        collocation_axes = get_vmap_axes(
            batch.keys(), COLLOCATION_FEATURE_NAMES
        )
        batch_axes = get_vmap_axes(batch.keys(), BATCH_FEATURE_NAMES)

        batched_rte_op = hk.vmap(
            mapping.sharded_map(
                rte_op, shard_size=low_memory, in_axes=collocation_axes
            ),
            in_axes=batch_axes,
            split_rng=(not hk.running_init()),
        )
        predictions = batched_rte_op(batch)

        ret["predicted_solution"] = predictions
        if compute_loss:
            labels = batch["psi_label"]
            loss = mean_squared_loss_fn(predictions, labels)
            ret["loss"] = {
                "mse": loss,
                "rmspe": jnp.sqrt(loss / jnp.mean(labels**2)),
            }
        if compute_metrics:
            labels = batch["psi_label"]
            # Compute relative mean squared error,
            # this values will be summed and finally divided
            # by num_examples.
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

    def __call__(self, coord1, coord2, batch, is_training):
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
            return jnp.squeeze(output, axis=-1)

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
        kernel = (1 - batch["scattering_kernel"]) * velocity_weights
        self_kernel = (1 - batch["self_scattering_kernel"]) * velocity_weights

        self_act = hk.vmap(self_att_fn, split_rng=(not hk.running_init()))(
            velocity_coords
        )
        act_output, _ = Scattering(c.scattering, gc)(
            act=act,
            self_act=self_act,
            kernel=kernel,
            self_kernel=self_kernel,
            is_training=is_training,
        )
        output = jnp.exp(projection(act_output))
        return jnp.squeeze(output, axis=-1)


class Scattering(hk.Module):
    def __init__(self, config, global_config, name="scattering_module"):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    def __call__(self, act, self_act, kernel, self_kernel, is_training):
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
            scattering_layer = ScatteringLayer(
                output_size=c.latent_dim, w_init=w_init
            )
            act_out = dropout_wrapper_fn(
                module=scattering_layer,
                input_act=self_act,
                kernel=kernel,
                output_act=act,
            )
            self_act_out = dropout_wrapper_fn(
                module=scattering_layer,
                input_act=self_act,
                kernel=self_kernel,
            )
            return (act_out, self_act_out)

        scattering_stack = hk.experimental.layer_stack(c.num_layer)(
            scattering_block
        )
        act_output, self_act_output = scattering_stack((act, self_act))

        return act_output, self_act_output


class ScatteringLayer(hk.Module):
    "A single layer describing scattering action."

    def __init__(
        self, output_size, with_bias=True, w_init=None, name="scattering_layer"
    ):
        super().__init__(name=name)
        self.output_size = output_size
        self.with_bias = with_bias
        self.w_init = w_init

    def __call__(self, act, kernel, is_training=None):
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

    def __init__(self, config, global_config, name="attenuation"):
        super().__init__(name=name)
        self.config = config
        self.global_config = global_config

    def __call__(self, coord1, coord2, att_coeff, charc):
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
            act = hk.Linear(
                c.latent_dim, w_init=w_init, name="attenuation_linear"
            )(act)
            act = jax.nn.tanh(act)

        act = jax.nn.tanh(
            hk.Linear(c.output_dim, w_init=w_init, name="output_projection")(
                act
            )
        )

        return act


class Attention(hk.Module):
    """Multihead Attention."""

    def __init__(self, config, global_config, name="attention"):
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
        y = hk.Linear(self.num_head * head_dim, w_init=self.w_init, name=name)(
            x
        )
        *leading_dims, _ = x.shape
        return y.reshape((*leading_dims, self.num_head, head_dim))


class Attention_v2(hk.Module):
    """Multihead Attention."""

    def __init__(self, config, global_config, name="attention"):
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
            precision=lax.Precision.HIGHEST,
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
        y = hk.Linear(self.num_head * head_dim, w_init=self.w_init, name=name)(
            x
        )
        *leading_dims, _ = x.shape
        return y.reshape((*leading_dims, self.num_head, head_dim))
