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

"Utilities functions."
import functools

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

# Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
TRUNCATED_NORMAL_STDDEV_FACTOR = np.asarray(
    0.87962566103423978, dtype=np.float32
)


def mean_squared_loss_fn(x, y, axis=None):
    return jnp.mean(jnp.square(x - y), axis=axis)


def apply_dropout(*, tensor, safe_key, rate, is_training):
    """Applies dropout to a tensor."""
    if is_training and rate != 0.0:
        shape = list(tensor.shape)
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

    residual = apply_dropout(
        tensor=residual,
        safe_key=safe_key,
        rate=dropout_rate,
        is_training=is_training,
    )

    new_act = output_act + residual

    return new_act


def get_initializer_scale(initializer_name, input_shape=()):
    """Get Initializer for weights and scale to multiply activations by."""

    if initializer_name == "zeros":
        w_init = hk.initializers.Constant(0.0)
    elif initializer_name == "glorot_uniform":
        w_init = hk.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"
        )
    else:
        # fan-in scaling
        scale = 1.0
        for channel_dim in input_shape:
            scale /= channel_dim
        if initializer_name == "relu":
            scale *= 2

        noise_scale = scale

        stddev = np.sqrt(noise_scale)
        # Adjust stddev for truncation.
        stddev = stddev / TRUNCATED_NORMAL_STDDEV_FACTOR
        w_init = hk.initializers.TruncatedNormal(mean=0.0, stddev=stddev)

    return w_init


def query_chunk_attention(
    query,
    key,
    value,
    mask=None,
    key_chunk_size=None,
    precision=lax.Precision.HIGHEST,
    dtype=jnp.float32,
):
    num_kv, num_heads, k_features = key.shape
    # *leading_dims, num_heads, q_features = query.shape
    v_features = value.shape[-1]
    if key_chunk_size:
        key_chunk_size = min(key_chunk_size, num_kv)
    else:
        key_chunk_size = num_kv
    query = query / jnp.sqrt(k_features).astype(dtype)
    # print(key_chunk_size)

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def summarize_chunk(query, key, value, mask):
        attn_weights = jnp.einsum(
            "...hd,khd->...hk", query, key, precision=precision
        ).astype(dtype)
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -1e30)
        max_score = jnp.max(attn_weights, axis=-1, keepdims=True)
        max_score = jax.lax.stop_gradient(max_score)
        exp_weights = jnp.exp(attn_weights - max_score)
        exp_values = jnp.einsum(
            "vhf,...hv->...hf", value, exp_weights, precision=precision
        ).astype(dtype)
        return (
            exp_values,
            exp_weights.sum(axis=-1),
            max_score.reshape((num_heads,)),
        )

    def chunk_scanner(chunk_idx):
        key_chunk = lax.dynamic_slice(
            key,
            (chunk_idx, 0, 0),
            slice_sizes=(key_chunk_size, num_heads, k_features),
        )
        value_chunk = lax.dynamic_slice(
            value,
            (chunk_idx, 0, 0),
            slice_sizes=(key_chunk_size, num_heads, v_features),
        )
        # print(chunk_idx)
        if mask is not None:
            mask_chunk = lax.dynamic_slice(
                mask,
                (
                    0,
                    chunk_idx,
                ),
                slice_sizes=(
                    1,
                    key_chunk_size,
                ),
            )
        else:
            mask_chunk = None
        return summarize_chunk(query, key_chunk, value_chunk, mask_chunk)

    chunk_values, chunk_weights, chunk_max = lax.map(
        chunk_scanner, xs=jnp.arange(0, num_kv, key_chunk_size)
    )
    global_max = jnp.max(chunk_max, axis=0, keepdims=True)
    max_diffs = jnp.exp(chunk_max - global_max)
    chunk_values *= jnp.expand_dims(max_diffs, axis=-1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(axis=0)
    all_weights = jnp.expand_dims(chunk_weights, -1).sum(axis=0)
    return all_values / all_weights
