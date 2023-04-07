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

from collections.abc import Generator, Sequence

import tensorflow as tf

from deeprte.model.tf import rte_features


def curry1(f):
    """Supply all arguments but the first."""

    def fc(*args, **kwargs):
        return lambda x: f(x, *args, **kwargs)

    return fc


@curry1
def repeat_batch(
    ds: tf.data.Dataset, batch_sizes: int | Sequence[int], repeat: int = 1
) -> tf.data.Dataset:
    """Tiles the inner most batch dimension."""
    if repeat <= 1:
        return ds
    # Perform regular batching with reduced number of elements.
    for batch_size in reversed(batch_sizes):
        ds = ds.batch(batch_size, drop_remainder=True)

    # Repeat batch.
    repeat_fn = lambda x: tf.tile(  # noqa: E731
        x, multiples=[repeat] + [1] * (len(x.shape) - 1)
    )

    def repeat_inner_batch(example):
        return tf.nest.map_structure(repeat_fn, example)

    ds = ds.map(repeat_inner_batch, num_parallel_calls=tf.data.AUTOTUNE)
    # Unbatch.
    for _ in batch_sizes:
        ds = ds.unbatch()
    return ds


@curry1
def sample_collocation_coords(
    batch,
    collocation_sizes: list[int],
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

    num_phase_coords = tf.shape(batch["phase_coords"])[
        rte_features.PHASE_FEATURE_AXIS["phase_coords"]
    ]
    phase_coords_indices = generator.uniform(
        (collocation_sizes[0],),
        minval=0,
        maxval=num_phase_coords,
        dtype=tf.int32,
    )
    for k, axis in rte_features.PHASE_FEATURE_AXIS.items():
        if k in batch:
            batch[k] = tf.gather(batch[k], phase_coords_indices, axis=axis)

    if len(collocation_sizes) > 1:
        num_boundary_coords = tf.shape(batch["boundary_coords"])[
            rte_features.BOUNDARY_FEATURE_AXIS["boundary_coords"]
        ]
        boundary_coords_indices = generator.uniform(
            (collocation_sizes[1],),
            minval=0,
            maxval=num_boundary_coords,
            dtype=tf.int32,
        )
        for k, axis in rte_features.BOUNDARY_FEATURE_AXIS.items():
            if k in batch:
                batch["sampled_" + k] = tf.gather(
                    batch[k], boundary_coords_indices, axis=axis
                )

    return batch
