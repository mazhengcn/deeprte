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

from deeprte.model.tf import rte_features

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


BATCH_FEATURES = rte_features.BATCH_FEATURE_NAMES + [
    "sampled_boundary",
    "sampled_boundary_scattering_kernel",
]


def split_features(features):
    """Split features into batched and unbatched."""
    batched_feat = {}
    unbatched_feat = {}
    for k, v in features.items():
        if k in BATCH_FEATURES:
            batched_feat.update({k: v})
        else:
            unbatched_feat.update({k: v})

    return batched_feat, unbatched_feat


def accumulate_gradient(grad_fn, params, batch, accum_steps):
    """Accumulate gradient over multiple steps to save on memory."""
    batched_feat, unbatched_feat = split_features(batch)
    batch_size = batched_feat["psi_label"].shape[0]
    print(batch_size)
    if accum_steps and accum_steps > 1:
        assert (
            batch_size % accum_steps == 0
        ), f"Bad accum_steps {accum_steps} for batch size {batch_size}"
        step_size = batch_size // accum_steps

        def dynamic_slice_feat(feat_dict, i, step_size):
            def slice_fn(x):
                return jax.lax.dynamic_slice(
                    x, (i,) + (0,) * (x.ndim - 1), (step_size,) + x.shape[1:]
                )

            return jax.tree_map(slice_fn, feat_dict)

        # loss, (scalars, state) = ret
        l_and_state = grad_fn(
            params,
            {
                **dynamic_slice_feat(batched_feat, 0, step_size),
                **unbatched_feat,
            },
        )

        def acc_grad_and_loss(i, l_and_state):
            sliced_batch = {
                **dynamic_slice_feat(batched_feat, i * step_size, step_size),
                **unbatched_feat,
            }
            grads_i, (scalars_i, state_i) = grad_fn(params, sliced_batch)
            grads, (scalars, state) = l_and_state
            return jax.tree_map(lambda x, y: x + y, grads, grads_i), (
                jax.tree_map(lambda x, y: x + y, scalars, scalars_i),
                state_i,
            )

        grads, (scalars, state) = jax.lax.fori_loop(
            1, accum_steps, acc_grad_and_loss, l_and_state
        )
        return jax.tree_map(
            lambda x: x / accum_steps, (grads, (scalars, state))
        )
    else:
        return grad_fn(params, batch)


def query_chunk_attention(
    query,
    key,
    value,
    mask=None,
    key_chunk_size=None,
    precision=lax.Precision.DEFAULT,
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
                mask, (0, chunk_idx), slice_sizes=(1, key_chunk_size)
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
