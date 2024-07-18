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

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import nnx

from deeprte.configs import default
from deeprte.model import integrate, mapping
from deeprte.model.characteristics import Characteristics
from deeprte.model.tf import rte_features

Shape = tuple[int, ...]
Dtype = Any
Array = jax.Array

kernel_init = nnx.initializers.glorot_uniform()
bias_init = nnx.initializers.zeros_init()


FEATURES = rte_features.FEATURES
COLLOCATION_AXES = {
    k: 0 if rte_features.NUM_PHASE_COORDS in v[1] else None for k, v in FEATURES.items()
}


def constructor(config: DeepRTEConfig, key: jax.Array) -> nnx.Module:
    """A wrapper function to create the DeepRTE."""
    return DeepRTE(config, rngs=nnx.Rngs(params=key))


@dataclasses.dataclass(unsafe_hash=True)
class DeepRTEConfig:
    # Physical position dimensions.
    position_coords_dim: int = 2
    # Physical velocity dimensions.
    velocity_coords_dim: int = 2
    # Dimensions of (scattering) coefficient functions.
    coeffs_fn_dim: int = 2
    # Number of attention heads.
    num_heads: int = 2
    # Attention dimension.
    qkv_dim: int = 64
    # Output dimensions of attention.
    optical_depth_dim: int = 2
    # Number of MLP layers.
    num_mlp_layers: int = 4
    # MLP dimension.
    mlp_dim: int = 128
    # Number of scattering layers.
    num_scattering_layers: int = 2
    # Scattering dimension.
    scattering_dim: int = 16
    # Subcollocation size for evaluation or inference
    subcollocation_size: int = 128
    # Mesh rules
    axis_rules: default.MeshRules = dataclasses.field(default_factory=default.MeshRules)

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


def dot_product_attention_weights(
    query: Array, key: Array, bias: Optional[Array] = None, mask: Optional[Array] = None
):
    """Computes dot-product attention weights given query and key."""
    dtype = query.dtype
    assert query.shape[-2] == key.shape[-2], "q, k num_heads must match."
    assert query.shape[-1] == key.shape[-1], "q, k depths must match."

    # calculate attention matrix
    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)
    # attn weight shape is (batch..., num_heads, q_length, kv_length)
    attn_weights = jnp.einsum("...hd,...khd->...hk", query, key)

    # apply attention bias: masking, dropout, proximity bias, etc.
    if bias is not None:
        attn_weights = attn_weights + bias
    # apply attention mask
    if mask is not None:
        big_neg = jnp.finfo(dtype).min
        attn_weights = jnp.where(mask, attn_weights, big_neg)

    # normalize the attention weights
    attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

    return attn_weights


def dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
):
    """Computes dot-product attention given query, key, and value."""

    assert (
        query.shape[-2] == key.shape[-2] == value.shape[-2]
    ), "q, k, v num_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."

    # compute attention weights
    attn_weights = dot_product_attention_weights(query, key, bias, mask)

    # return weighted sum over values for each query position
    return jnp.einsum("...hk,...khd->...hd", attn_weights, value)


class MultiHeadAttention(nnx.Module):
    """Multi-head attention."""

    def __init__(
        self,
        num_heads: int,
        in_features: int | tuple,
        qkv_features: int | None = None,
        out_features: int | None = None,
        *,
        attention_fn: Callable[..., Array] = dot_product_attention,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        if isinstance(in_features, int):
            self.q_features = self.k_features = self.v_features = in_features
        else:
            self.q_features, self.k_features, self.v_features = in_features
        self.qkv_features = qkv_features if qkv_features is not None else in_features
        self.out_features = out_features if out_features is not None else in_features
        self.attention_fn = attention_fn

        if self.qkv_features % self.num_heads != 0:
            raise ValueError(
                f"Memory dimension ({self.qkv_features}) must be divisible by "
                f"'num_heads' heads ({self.num_heads})."
            )

        self.head_dim = self.qkv_features // self.num_heads

        linear_general = functools.partial(
            nnx.LinearGeneral,
            out_features=(self.num_heads, self.head_dim),
            kernel_init=kernel_init,
            bias_init=bias_init,
        )
        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]
        self.query = linear_general(in_features=self.q_features, rngs=rngs)
        self.key = linear_general(in_features=self.k_features, rngs=rngs)
        self.value = linear_general(in_features=self.v_features, rngs=rngs)

        self.out = nnx.LinearGeneral(
            in_features=(self.num_heads, self.head_dim),
            out_features=self.out_features,
            axis=(-2, -1),
            kernel_init=kernel_init,
            bias_init=bias_init,
            rngs=rngs,
        )

    def __call__(
        self,
        inputs_q: Array,
        inputs_k: Array | None = None,
        inputs_v: Array | None = None,
        mask: Array | None = None,
    ):
        """Applies multi-head dot product attention on the input data."""

        if inputs_k is None:
            if inputs_v is not None:
                raise ValueError(
                    "`inputs_k` cannot be None if `inputs_v` is not None. "
                    "To have both `inputs_k` and `inputs_v` be the same value, pass in the "
                    "value to `inputs_k` and leave `inputs_v` as None."
                )
            inputs_k = inputs_q
        if inputs_v is None:
            inputs_v = inputs_k

        if inputs_q.shape[-1] != self.q_features:
            raise ValueError(
                f"Incompatible input dimension, got {inputs_q.shape[-1]} "
                f"but module expects {self.q_features}."
            )

        query = self.query(inputs_q)
        key = self.key(inputs_k)
        value = self.value(inputs_v)

        # apply attention
        x = self.attention_fn(query, key, value, mask=mask)
        # back to the original inputs dimensions
        out = self.out(x)
        return out


class MlpBlock(nnx.Module):
    """MLP / feed-forward block."""

    def __init__(self, config: DeepRTEConfig, *, rngs: nnx.Rngs):
        self.num_layers = config.num_mlp_layers
        self.in_features = (
            2 * (config.position_coords_dim + config.velocity_coords_dim)
            + config.optical_depth_dim
        )
        self.mlp_dim = config.mlp_dim
        self.out_dim = config.scattering_dim

        linears = []
        for idx in range(self.num_layers):
            in_features = self.in_features if idx == 0 else self.mlp_dim
            out_features = self.out_dim if idx == self.num_layers - 1 else self.mlp_dim
            linears.append(
                nnx.Linear(
                    in_features,
                    out_features,
                    kernel_init=kernel_init,
                    bias_init=bias_init,
                    rngs=rngs,
                )
            )

        self.linears = linears

    def __call__(self, x: Array):
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Incompatible input dimension, got {x.shape[-1]} "
                f"but module expects {self.in_features}."
            )
        for idx, linear in enumerate(self.linears):
            x = linear(x)
            if idx < self.num_layers - 1:
                x = nnx.tanh(x)
        return x


class Attenuation(nnx.Module):
    def __init__(self, config: DeepRTEConfig, rngs: nnx.Rngs):
        self.config = config

        attention_in_dims = [
            config.position_coords_dim + config.velocity_coords_dim,
            config.position_coords_dim,
            config.coeffs_fn_dim,
        ]

        self.attention = MultiHeadAttention(
            config.num_heads,
            attention_in_dims,
            config.qkv_dim,
            config.optical_depth_dim,
            rngs=rngs,
        )
        self.mlp = MlpBlock(config=config, rngs=rngs)

    def __call__(
        self, coord1: Array, coord2: Array, att_coeff: Array, charac: Characteristics
    ):
        local_coords, mask = charac.apply_to_point(coord1)
        optical_depth = self.attention(
            inputs_q=coord1, inputs_k=local_coords, inputs_v=att_coeff, mask=mask
        )
        optical_depth = jnp.exp(-optical_depth)

        attenuation = jnp.concatenate([coord1, coord2, optical_depth])
        attenuation = self.mlp(attenuation)
        return attenuation


class ScatteringLayer(nnx.Module):
    "A single layer describing scattering action."

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nnx.Linear(
            in_features,
            out_features,
            kernel_init=kernel_init,
            bias_init=bias_init,
            rngs=rngs,
        )

    def __call__(self, inputs: Array, kernel: Array):
        x = jnp.einsum("...V,Vd->...d", kernel, inputs)
        x = self.linear(x)
        x = nnx.tanh(x)
        return x


class Scattering(nnx.Module):
    """Scattering block."""

    def __init__(self, config: DeepRTEConfig, rngs: nnx.Rngs):
        self.config = config

        scattering_layers, lns = [], []
        for _ in range(config.num_scattering_layers):
            scattering_layers.append(
                ScatteringLayer(config.scattering_dim, config.scattering_dim, rngs=rngs)
            )
            lns.append(nnx.LayerNorm(config.scattering_dim, rngs=rngs))

        self.scattering_layers = scattering_layers
        self.lns = lns

    def __call__(self, act: Array, self_act: Array, kernel: Array, self_kernel: Array):
        self_act_0 = self_act
        for idx in range(self.config.num_scattering_layers - 1):
            self_act = self.scattering_layers[idx](self_act, self_kernel)
            self_act = self.lns[idx](self_act)
            self_act += self_act_0

        act_res = self.scattering_layers[-1](self_act, kernel)
        act_res = self.lns[-1](act_res)
        act += act_res
        return act


class GreenFunction(nnx.Module):
    """Green's function."""

    def __init__(self, config, rngs: nnx.Rngs):
        self.config = config
        self.attenuation = Attenuation(self.config, rngs=rngs)
        self.scattering = Scattering(self.config, rngs=rngs)
        self.out = nnx.Linear(
            self.config.scattering_dim,
            1,
            kernel_init=kernel_init,
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, coord1: Array, coord2: Array, batch):
        charac = Characteristics.from_tensor(batch["position_coords"])
        act = self.attenuation(
            coord1=coord1, coord2=coord2, att_coeff=batch["sigma"], charac=charac
        )
        if self.config.num_scattering_layers == 0:
            out = jnp.exp(self.out(act))
            out = jnp.squeeze(out, axis=-1)

        position, _ = jnp.split(coord1, 2, axis=-1)

        def self_att_fn(velocity):
            coord = jnp.concatenate([position, velocity], axis=-1)
            out = self.attenuation(
                coord1=coord, coord2=coord2, att_coeff=batch["sigma"], charac=charac
            )
            return out

        velocity_coords, velocity_weights = (
            batch["velocity_coords"],
            batch["velocity_weights"],
        )
        kernel = -batch["scattering_kernel"] * velocity_weights
        self_kernel = -batch["self_scattering_kernel"] * velocity_weights

        self_act = nnx.vmap(self_att_fn)(velocity_coords)
        out = self.scattering(act, self_act, kernel, self_kernel)

        out = jnp.exp(self.out(act))
        out = jnp.squeeze(out, axis=-1)

        return out


class DeepRTE(nnx.Module):
    """Deep RTE model."""

    def __init__(self, config: DeepRTEConfig, *, low_memory=False, rngs: nnx.Rngs):
        self.config = config
        self.low_memory = low_memory

        self.green_fn = GreenFunction(config=self.config, rngs=rngs)

    def __call__(self, batch):
        rte_inputs = {k: batch[k] for k in FEATURES}

        def rte_op(inputs):
            quadratures = (
                inputs["boundary_coords"],
                inputs["boundary"] * inputs["boundary_weights"],
            )
            rte_sol = integrate.quad(self.green_fn, quadratures=quadratures, argnum=1)(
                inputs["phase_coords"], inputs
            )
            return rte_sol

        batched_rte_op = jax.vmap(
            mapping.sharded_map(
                rte_op,
                shard_size=self.config.subcollocation_size if self.low_memory else None,
                in_axes=(COLLOCATION_AXES,),
            )
        )
        output = batched_rte_op(rte_inputs)

        return output
