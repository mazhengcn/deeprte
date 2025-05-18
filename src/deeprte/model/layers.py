import functools
import typing as tp
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx

kernel_init = nnx.initializers.glorot_uniform()
bias_init = nnx.initializers.zeros_init()


def dot_product_attention_weights(
    query: jax.Array,
    key: jax.Array,
    bias: tp.Optional[jax.Array] = None,
    mask: tp.Optional[jax.Array] = None,
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
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    bias: tp.Optional[jax.Array] = None,
    mask: tp.Optional[jax.Array] = None,
):
    """Computes dot-product attention given query, key, and value."""

    assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
        "q, k, v num_heads must match."
    )
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
        in_features: int | tp.Sequence[int],
        qkv_features: int | None = None,
        out_features: int | None = None,
        attention_fn: Callable[..., jax.Array] = dot_product_attention,
        *,
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
        inputs_q: jax.Array,
        inputs_k: tp.Optional[jax.Array] = None,
        inputs_v: tp.Optional[jax.Array] = None,
        mask: tp.Optional[jax.Array] = None,
    ) -> jax.Array:
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

    def __init__(self, config, *, rngs: nnx.Rngs):
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

    def __call__(self, x: jax.Array) -> jax.Array:
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
