import dataclasses
import functools
from collections.abc import Callable, Mapping
from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import nnx

from deeprte.model.characteristics import Characteristics
from deeprte.model.tf import rte_features

Shape = tuple[int, ...]
Dtype = Any
Array = jax.Array

kernel_init = nnx.initializers.glorot_uniform()
bias_init = nnx.initializers.zeros_init()


@dataclasses.dataclass(unsafe_hash=True)
class DeepRTEConfig:
    position_coords_dim: int = 2
    velocity_coords_dim: int = 2
    coeffs_fn_dim: int = 2
    num_basis_functions: int = 64
    basis_function_encoder_dim: int = 128
    num_basis_function_encoder_layers: int = 4
    green_function_encoder_dim: int = 128
    num_green_function_encoder_layers: int = 4
    num_scattering_layers: int = 2
    scattering_dim: int = 128
    num_heads: int = 8
    qkv_dim: int = 16
    optical_depth_dim: int = 16
    name: str = "boundary"
    subcollocation_size: int = 128
    # Normalization constant of dataset/model.
    normalization: float = 1.0
    # Where to load the parameters from.
    load_parameters_path: str = ""

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


def constructor(config: DeepRTEConfig, key: jax.Array) -> nnx.Module:
    """A wrapper function to create the DeepRTE."""
    if config.name == "deeprte":
        return DeepRTE(config, rngs=nnx.Rngs(params=key))
    elif config.name == "boundary":
        return BoundaryBasisRepresentation(config, rngs=nnx.Rngs(params=key))
    else:
        raise ValueError(f"Unknown model name: {config.name}")


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
        self.in_features = (
            config.position_coords_dim
            + config.velocity_coords_dim
            + config.optical_depth_dim
        )
        self.mlp_dim = config.green_function_encoder_dim
        self.num_layers = config.num_green_function_encoder_layers

        self.mlp = MlpBlock(
            units=[self.in_features] + [self.mlp_dim] * self.num_layers,
            activation=nnx.tanh,
            rngs=rngs,
        )

    def __call__(self, coord1: Array, att_coeff: Array, charac: Characteristics):
        local_coords, mask = charac.apply_to_point(coord1)
        optical_depth = self.attention(
            inputs_q=coord1, inputs_k=local_coords, inputs_v=att_coeff, mask=mask
        )
        optical_depth = jnp.exp(-optical_depth)

        attenuation = jnp.concatenate([coord1, optical_depth])
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


class MlpBlock(nnx.Module):
    """MLP / feed-forward block."""

    def __init__(
        self, units: list[int], activation: Callable[[Array], Array], rngs: nnx.Rngs
    ):
        linears = []
        for idx in range(len(units) - 1):
            linears.append(
                nnx.Linear(
                    units[idx],
                    units[idx + 1],
                    kernel_init=kernel_init,
                    bias_init=bias_init,
                    rngs=rngs,
                )
            )
        self.linears = linears
        self.activation = activation

    def __call__(self, x: Array):
        for idx, linear in enumerate(self.linears):
            x = linear(x)
            if idx < len(self.linears) - 1:
                x = self.activation(x)
        return x


class BasisFunction(nnx.Module):
    """Encodes the basis functions."""

    def __init__(self, config: DeepRTEConfig, rngs: nnx.Rngs):
        self.in_features = config.position_coords_dim + config.velocity_coords_dim
        self.out_features = config.num_basis_functions
        self.mlp_dim = config.basis_function_encoder_dim
        self.num_layers = config.num_basis_function_encoder_layers

        self.mlp = MlpBlock(
            units=[self.in_features]
            + [self.mlp_dim] * self.num_layers
            + [self.out_features],
            activation=nnx.tanh,
            rngs=rngs,
        )

    def basis_weights(self, quadratures: tuple[Array, Array]):
        points, weights = quadratures
        out = jax.vmap(self, in_axes=(0,), out_axes=-1)(points)
        # print(out.shape, weights.shape)
        return jnp.dot(out, weights)

    def __call__(self, x: Array):
        return self.mlp(x)


class GreenFunctionEncoder(nnx.Module):
    """Encodes the Green function."""

    def __init__(self, config: DeepRTEConfig, rngs: nnx.Rngs):
        self.config = config
        self.attenuation = Attenuation(self.config, rngs=rngs)
        self.scattering = Scattering(self.config, rngs=rngs)
        self.out = nnx.Linear(
            self.config.scattering_dim,
            self.config.num_basis_functions,
            kernel_init=kernel_init,
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, coord1: Array, batch):
        charac = Characteristics.from_tensor(batch["position_coords"])
        act = self.attenuation(coord1=coord1, att_coeff=batch["sigma"], charac=charac)
        if self.config.num_scattering_layers == 0:
            out = jnp.squeeze(jnp.exp(self.out(act)), axis=-1)
            return out

        position, _ = jnp.split(coord1, 2, axis=-1)

        def self_att_fn(velocity):
            coord = jnp.concatenate([position, velocity], axis=-1)
            out = self.attenuation(
                coord1=coord, att_coeff=batch["sigma"], charac=charac
            )
            return out

        velocity_coords, velocity_weights = (
            batch["velocity_coords"],
            batch["velocity_weights"],
        )
        kernel = -batch["scattering_kernel"] * velocity_weights
        self_kernel = -batch["self_scattering_kernel"] * velocity_weights

        self_act = jax.vmap(self_att_fn)(velocity_coords)
        act = self.scattering(act, self_act, kernel, self_kernel)

        # out = jnp.squeeze(jnp.exp(self.out(act)), axis=-1)
        # out = jnp.exp(self.out(act))
        return act


class DeepRTE(nnx.Module):
    """Deep RTE model."""

    def __init__(self, config: DeepRTEConfig, rngs: nnx.Rngs):
        self.config = config
        self.green_function_encoder = GreenFunctionEncoder(config=config, rngs=rngs)
        self.features = rte_features.GREEN_FUNCTION_FEATURES
        self.phase_coords_axes = {
            k: 0 if k in rte_features.PHASE_COORDS_FEATURES else None
            for k in self.features
        }

    def __call__(self, batch: Mapping[str, Array]):
        inputs = {k: batch[k] for k in self.features}

        def green_function_op(inputs):
            return self.green_function_encoder(inputs["phase_coords"], inputs)

        batched_green_function_op = jax.vmap(
            green_function_op, in_axes=(self.phase_coords_axes,)
        )

        def rte_op(inputs):
            act = batched_green_function_op(inputs)

            weights = inputs["basis_weights"]
            inner_product = inputs["basis_inner_product"]

            return jnp.einsum(
                "bj,j->b",
                jnp.einsum("bi,ij->bj", act, inner_product),
                weights,
            )

        batched_rte_op = jax.vmap(rte_op)

        return batched_rte_op(inputs)


class BoundaryBasisRepresentation(nnx.Module):
    """Boundary basis representation."""

    def __init__(self, config: DeepRTEConfig, rngs: nnx.Rngs):
        self.config = config
        self.features = rte_features.BASIS_FEATURES

        self.basis_function = BasisFunction(config=config, rngs=rngs)

    def __call__(self, batch: Mapping[str, Array]):
        inputs = {k: batch[k] for k in self.features}
        # basis_weights = self.basis_function.basis_weights

        def basis_op(inputs):
            act = self.basis_function(inputs["boundary_coords"])
            weights = self.basis_function.basis_weights(
                (
                    inputs["boundary_coords"],
                    inputs["boundary"] * inputs["boundary_weights"],
                )
            )
            # print(act.shape, weights.shape)
            return act @ weights

        batched_basis_op = jax.vmap(basis_op)

        return batched_basis_op(inputs)


class RtePredictor(nnx.Module):
    """RTE predictor."""

    def __init__(self, config: DeepRTEConfig, rngs: nnx.Rngs):
        self.config = config
        self.features = rte_features.FEATURES
        # self.basis_features = rte_features.BASIS_FEATURES
        self.phase_coords_axes = {
            k: 0 if k in rte_features.PHASE_COORDS_FEATURES else None
            for k in self.features
        }
        self.basis_coords_axes = {
            k: 0 if k in rte_features.BASIS_FEATURES else None for k in self.features
        }

        self.basis_function = BasisFunction(config=config, rngs=rngs)
        self.green_function_encoder = GreenFunctionEncoder(config=config, rngs=rngs)

    def __call__(self, batch: Mapping[str, Array]):
        inputs = {k: batch[k] for k in self.features}

        def basis_op(inputs):
            act = self.basis_function(inputs["boundary_coords"])
            # print(act.shape)
            basis_inner_product = jnp.einsum(
                "...ij,...ik->...jk",
                inputs["boundary_weights"][..., None] * act,
                act,
            )
            basis_weights = jnp.einsum(
                "...ij,...i->...j",
                act,
                inputs["boundary_weights"] * inputs["boundary"],
            )
            return basis_inner_product, basis_weights

        def green_function_op(inputs):
            return self.green_function_encoder(inputs["phase_coords"], inputs)

        batched_green_function_op = jax.vmap(
            green_function_op, in_axes=(self.phase_coords_axes,)
        )

        def rte_op(inputs):
            act = batched_green_function_op(inputs)
            inner_product, weights = basis_op(inputs)
            # print(act.shape, inner_product.shape, weights.shape)

            return jnp.einsum(
                "...bj,...j->...b",
                jnp.einsum("bi,ij->bj", act, inner_product),
                weights,
            )

        batched_rte_op = jax.vmap(rte_op)

        return batched_rte_op(inputs)
