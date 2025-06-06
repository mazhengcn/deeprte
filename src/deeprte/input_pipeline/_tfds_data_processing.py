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

"""Input pipeline for a deeprte dataset."""

from collections.abc import Generator, Sequence
from typing import Any, Optional

import jax
import tensorflow as tf
import tensorflow_datasets as tfds
from clu.data import dataset_iterator
from jax.sharding import Mesh, PartitionSpec

from deeprte.input_pipeline import utils
from deeprte.train_lib import multihost_dataloading

DEFAULT_SHUFFLE_BUFFER = 50_000
VALID_KEY = "__valid__"
ArraySpec = dataset_iterator.ArraySpec
ArraySpecDict = dict[str, ArraySpec]
Data = dict[str, Any]
DatasetIterator = dataset_iterator.DatasetIterator
TfDatasetIterator = dataset_iterator.TfDatasetIterator


AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_datasets(
    dataset_name,
    data_dir,
    data_split,
    shuffle_files,
    read_config=None,
    with_info: bool = False,
):
    """Load a TFDS dataset."""
    ds_builder = tfds.builder(dataset_name, data_dir=data_dir)
    ds = ds_builder.as_dataset(
        split=data_split, read_config=read_config, shuffle_files=shuffle_files
    )
    if with_info:
        return ds, ds_builder.info

    return ds


def preprocessing_pipeline(
    dataset,
    dataset_info: tfds.core.DatasetInfo = None,  # ty: ignore
    *,
    global_batch_size: int,
    collocation_size: Optional[int] = None,
    batch_repeat: Optional[int] = None,
    global_mesh: Mesh,
    data_pspec: PartitionSpec,
    shuffle: bool = False,
    data_shuffle_seed=0,
    num_epochs: Optional[int] = 1,
    shuffle_buffer_size: int = 1024,
    drop_remainder: bool = True,
    prefetch_size=tf.data.AUTOTUNE,
):
    """pipeline for preprocessing TFDS dataset."""
    dataset = dataset.shard(num_shards=jax.process_count(), index=jax.process_index())

    # Shuffle and repeat.
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size, seed=data_shuffle_seed)

    dataset = dataset.repeat(num_epochs)

    assert global_batch_size % global_mesh.size == 0, (
        "Batch size should be divisible number of global devices."
    )
    # Batch examples.
    batch_size_per_process = global_batch_size // jax.process_count()

    if batch_repeat:
        dataset = repeat_batch(batch_size_per_process, batch_repeat)(dataset)

    dataset = dataset.batch(batch_size_per_process, drop_remainder=drop_remainder)

    if collocation_size:
        rng = tf.random.Generator.from_seed(seed=0)
        collocation_axes = dataset_info.metadata["phase_feature_axis"]
        dataset = dataset.map(
            sample_collocation_coords(collocation_size, collocation_axes, rng)
        )

    if prefetch_size:
        dataset = dataset.prefetch(prefetch_size)

    multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(
        dataset, global_mesh, data_pspec
    )

    # Return multi-host jax.Array prep iterator
    return multihost_gen


def make_tfds_iterator(config, global_mesh):
    """load dataset, preprocess and return iterators"""
    read_config = tfds.ReadConfig(
        shuffle_seed=config.data_shuffle_seed, skip_prefetch=True
    )

    train_ds, ds_info = get_datasets(
        dataset_name=config.dataset_name,
        data_dir=config.data_dir,
        data_split=config.train_split,
        shuffle_files=config.enable_data_shuffling,
        read_config=read_config,
        with_info=True,
    )
    norm_dict = ds_info.metadata["normalization"]
    config.normalization = utils.get_normalization_ratio(
        norm_dict["psi_range"], norm_dict["boundary_range"]
    )

    train_iter = preprocessing_pipeline(
        dataset=train_ds,
        dataset_info=ds_info,
        global_batch_size=config.global_batch_size,
        collocation_size=config.collocation_size,
        batch_repeat=config.repeat_batch,
        global_mesh=global_mesh,
        data_pspec=PartitionSpec(*config.data_sharding),
        shuffle=config.enable_data_shuffling,
        data_shuffle_seed=config.data_shuffle_seed,
        num_epochs=-1,
    )

    if config.eval_every_steps > 0:
        eval_ds = get_datasets(
            dataset_name=config.dataset_name,
            data_dir=config.data_dir,
            data_split=config.eval_split,
            shuffle_files=False,
        )

        eval_iter = preprocessing_pipeline(
            dataset=eval_ds,
            global_batch_size=config.eval_batch_size,
            global_mesh=global_mesh,
            data_pspec=PartitionSpec(*config.data_sharding),
            shuffle=False,
            data_shuffle_seed=config.data_shuffle_seed,
        )
    else:
        eval_iter = None

    return train_iter, eval_iter


def curry1(f):
    """Supply all arguments but the first."""

    def fc(*args, **kwargs):
        return lambda x: f(x, *args, **kwargs)

    return fc


@curry1
def sample_collocation_coords(
    batch,
    collocation_size: int,
    collocation_axes: dict,
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
    num_phase_coords = tf.shape(batch["phase_coords"])[collocation_axes["phase_coords"]]
    # phase_coords_indices = generator.uniform(
    #     (collocation_size,), minval=0, maxval=num_phase_coords, dtype=tf.int32
    # )
    phase_coords_indices = tf.random.shuffle(tf.range(num_phase_coords))[
        :collocation_size
    ]

    for k, axis in collocation_axes.items():
        if k in batch:
            batch[k] = tf.gather(batch[k], phase_coords_indices, axis=axis)

    if "boundary_scattering_kernel" in batch:
        del batch["boundary_scattering_kernel"]

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
