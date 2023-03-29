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

"""Data transform for DeepRTE."""

# from collections.abc import Generator, Mapping, Sequence
from typing import Generator, Mapping, Optional, Sequence

import tensorflow as tf


def curry1(f):
    """Supply all arguments but the first."""

    def fc(*args, **kwargs):
        return lambda x: f(x, *args, **kwargs)

    return fc


def sample_phase_points(
    features,
    collocation_features: Mapping[str, int],
    collocation_sizes: int,
    total_grid_sizes: int,
    generator: Generator,
):
    """Sample phase points randomly and take collocation points.

    Args:
        featrues: batch to sample.
        collocation_sizes: number of collocation points.
        seed: random seed.

    Returns:
        sampled data.
    """

    idx = generator.uniform(
        (collocation_sizes,),
        minval=0,
        maxval=total_grid_sizes,
        dtype=tf.int64,
    )
    for k, axis in collocation_features.items():
        if k in features:
            features[k] = tf.gather(features[k], idx, axis=axis)

    return features


def sample_points(
    features,
    collocation_features: Mapping[str, int],
    collocation_sizes: int,
    total_sizes: int,
    generator: Generator,
    is_replacing: bool,
):
    """Sample phase points randomly and take collocation points.

    Args:
        featrues: batch to sample.
        collocation_sizes: number of collocation points.
        seed: random seed.

    Returns:
        sampled data.
    """

    idx = generator.uniform(
        (collocation_sizes,),
        minval=0,
        maxval=total_sizes,
        dtype=tf.int64,
    )
    ret = features.copy()
    for k, axis in collocation_features.items():
        if k in features:
            if is_replacing:
                ret[k] = tf.gather(features[k], idx, axis=axis)
            else:
                key = "sampled_" + k
                ret[key] = tf.gather(features[k], idx, axis=axis)

    return ret


@curry1
def select_feat(feature, feature_name_list):
    return {k: v for k, v in feature.items() if k in feature_name_list}


def repeat_batch(
    batch_sizes: int | Sequence[int],
    ds: tf.data.Dataset,
    repeat: int = 1,
) -> tf.data.Dataset:
    """Tiles the inner most batch dimension."""
    if repeat <= 1:
        return ds
    # Perform regular batching with reduced number of elements.
    for batch_size in reversed(batch_sizes):
        ds = ds.batch(batch_size, drop_remainder=True)

    # Repeat batch.
    fn = lambda x: tf.tile(  # noqa: E731
        x, multiples=[repeat] + [1] * (len(x.shape) - 1)
    )

    def repeat_inner_batch(example):
        return tf.nest.map_structure(fn, example)

    ds = ds.map(repeat_inner_batch, num_parallel_calls=tf.data.AUTOTUNE)
    # Unbatch.
    for _ in batch_sizes:
        ds = ds.unbatch()
    return ds


@curry1
def construct_batch(
    batched_feat,
    collocation_features: Sequence[Mapping[str, int]],
    collocation_sizes: Sequence[int],
    total_grid_sizes: Sequence[int],
    generator: Generator,
    is_replacing: Optional[Sequence[bool]] = None,
):
    for i, col in enumerate(collocation_features):
        batched_feat = sample_points(
            batched_feat,
            collocation_features=col,
            collocation_sizes=collocation_sizes[i],
            total_sizes=total_grid_sizes[i],
            generator=generator,
            is_replacing=is_replacing[i] if is_replacing else True,
        )
    return batched_feat


@curry1
def cat_feat(feat1, feat2):
    feat1.update(feat2)
    return feat1
