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

import enum
import os
from collections.abc import Generator, Sequence
from typing import Any, Optional

import jax
import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds
from clu.data import dataset_iterator

DEFAULT_SHUFFLE_BUFFER = 50_000
VALID_KEY = "__valid__"
ArraySpec = dataset_iterator.ArraySpec
ArraySpecDict = dict[str, ArraySpec]
Data = dict[str, Any]
DatasetIterator = dataset_iterator.DatasetIterator
TfDatasetIterator = dataset_iterator.TfDatasetIterator


class Split(enum.Enum):
    TRAIN = 1
    TRAIN_AND_VALID = 2
    VALID = 3
    TEST = 4


def get_data_config(
    name: str,
    split: str,
    process: str,
    batch_size: int,
    shuffle_buffer: Optional[int] = None,
    cache: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> ml_collections.ConfigDict:
    """Returns dataset parameters."""
    config = ml_collections.ConfigDict(type_safe=False)
    config.name = name
    config.split = split
    config.process = process
    config.batch_size = batch_size
    config.prefetch = "autotune"
    config.prefetch_device = 2
    if shuffle_buffer:
        config.shuffle_buffer = shuffle_buffer
    if cache:
        config.cache = cache
    if data_dir:
        config.data_dir = data_dir
    return config


def get_datasets(config: ml_collections.ConfigDict) -> dict[str, DatasetIterator]:
    """Returns a dictionary of datasets to use for different variants."""
    datasets = {}
    for variant, variant_config in config.items():
        if not isinstance(variant_config, ml_collections.ConfigDict):
            raise TypeError(
                f"The config for the {variant!r} variant is not a ConfigDict."
            )
        variant_config = variant_config.to_dict()
        _ = variant_config.pop("prefetch_device", None)
        datasets[variant] = get_dataset(variant=variant, **variant_config)
    return datasets


def get_dataset(
    *,
    variant: str,
    name: str,
    tfds_dir: str | os.PathLike | None = None,
    split: Split,
    split_percentage: str = "",
    batch_size: Optional[int] = None,
    collocation_sizes: Optional[Sequence[int] | None] = None,
    batch_repeat: Optional[int | None] = None
) -> DatasetIterator:
    """Returns a dataset iterator.

    Args:
        variant: Variant (e.g. 'train', 'validation', ...).
        name: Name of the dataset in TFDS.
        split: String with the split to use (e.g. 'train', 'validation[:100]', etc).
        batch_size: (Global) batch size to use. We assume that this batch size is
        evenly split among all devices.
        collocation_sizes: collocation_sizes should be: (total_collocation_size) or (interior_size, boundary_size, quadrature_size).
        batch_repeat: repeat number of inner batch, for training the same batch with
        {repeat} steps of different collocation points.
        **extra_builder_kwargs: Additional kwargs passed to the DatasetBuilder.

    Returns:
        A DatasetIterator.
    """

    is_training = variant == "train"

    # Compute the batch size per process.
    if batch_size % jax.process_count() or batch_size % jax.device_count():
        raise ValueError(
            f"batch_size must divide the process and device count, "
            f"but got {batch_size}, {jax.process_count()}, "
            f"and {jax.device_count()} respectively."
        )
    batch_size_per_process = batch_size // jax.process_count()

    tfds_split = _to_tfds_split(split, split_percentage)
    tfds_jax_split = tfds.split_for_jax_process(tfds_split, drop_remainder=False)

    # Load dataset
    dataset, info = tfds.load(
        name,
        data_dir=tfds_dir,
        split=tfds_jax_split,
        shuffle_files=True,
        with_info=True,
    )

    # tf.data options
    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    options.threading.max_intra_op_parallelism = 1
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    if is_training:
        options.deterministic = False
    dataset = dataset.with_options(options)

    if is_training:
        dataset = dataset.cache()
        dataset = dataset.repeat()
        dataset = dataset.shuffle(
            buffer_size=info.splits[tfds_jax_split].num_examples,
            reshuffle_each_iteration=True,
        )

        if batch_repeat:
            dataset = repeat_batch(batch_size_per_process, batch_repeat)(dataset)

    dataset = dataset.batch(batch_size, drop_remainder=True)

    if is_training and collocation_sizes:
        rng = tf.random.Generator.from_seed(seed=0)
        collocation_axis_dict = (
            info.metadata["phase_feature_axis"],
            info.metadata["boundary_feature_axis"],
        )
        dataset = dataset.map(
            sample_collocation_coords(collocation_sizes, collocation_axis_dict, rng),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return TfDatasetIterator(dataset, checkpoint=False)


def _to_tfds_split(split: Split, split_percentage: str = ""):
    if split in (Split.TRAIN, Split.TRAIN_AND_VALID):
        return f"train[:{split_percentage}]"
    elif split in (Split.VALID, Split.TEST):
        return f"train[{split_percentage}:]"
    else:
        raise ValueError(f"Unknown split {split}")


def curry1(f):
    """Supply all arguments but the first."""

    def fc(*args, **kwargs):
        return lambda x: f(x, *args, **kwargs)

    return fc


@curry1
def sample_collocation_coords(
    batch,
    collocation_sizes: list[int],
    collocation_axis_dicts: list[dict],
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

    phase_feature_axis = collocation_axis_dicts[0]
    num_phase_coords = tf.shape(batch["phase_coords"])[
        phase_feature_axis["phase_coords"]
    ]
    phase_coords_indices = generator.uniform(
        (collocation_sizes[0],),
        minval=0,
        maxval=num_phase_coords,
        dtype=tf.int32,
    )
    for k, axis in phase_feature_axis.items():
        if k in batch:
            batch[k] = tf.gather(batch[k], phase_coords_indices, axis=axis)

    if len(collocation_sizes) > 1:
        boundary_feature_axis = collocation_axis_dicts[1]
        num_boundary_coords = tf.shape(batch["boundary_coords"])[
            boundary_feature_axis["boundary_coords"]
        ]
        boundary_coords_indices = generator.uniform(
            (collocation_sizes[1],),
            minval=0,
            maxval=num_boundary_coords,
            dtype=tf.int32,
        )
        for k, axis in boundary_feature_axis.items():
            if k in batch:
                batch["sampled_" + k] = tf.gather(
                    batch[k], boundary_coords_indices, axis=axis
                )

    return batch


@curry1
def repeat_batch(
    ds: tf.data.Dataset, batch_size: int | Sequence[int], repeat: int = 1
) -> tf.data.Dataset:
    """Tiles the inner most batch dimension."""
    if repeat <= 1:
        return ds
    # Perform regular batching with reduced number of elements.
    ds = ds.batch(batch_size, drop_remainder=True)

    # Repeat batch.
    repeat_fn = lambda x: tf.tile(  # noqa: E731
        x, multiples=[repeat] + [1] * (len(x.shape) - 1)
    )

    def repeat_inner_batch(example):
        return tf.nest.map_structure(repeat_fn, example)

    ds = ds.map(repeat_inner_batch, num_parallel_calls=tf.data.AUTOTUNE)
    # Unbatch.
    ds = ds.unbatch()

    return ds
