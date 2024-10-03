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

import dataclasses
from typing import Any, Optional

import grain.python as grain
import jax
import numpy as np
import tensorflow_datasets as tfds
from clu.data import dataset_iterator
from jax.sharding import Mesh, PartitionSpec

from deeprte.input_pipeline import utils
from deeprte.train_lib import multihost_dataloading

ArraySpec = dataset_iterator.ArraySpec
ArraySpecDict = dict[str, ArraySpec]
Data = dict[str, Any]
DatasetIterator = dataset_iterator.DatasetIterator


@dataclasses.dataclass
class SampleCollocationCoords(grain.RandomMapTransform):
    """Sample phase points randomly and take collocation points.

    Args:
        featrues: batch to sample.
        collocation_sizes: number of collocation points.
        seed: random seed.

    Returns:
        sampled data.
    """

    def __init__(self, collocation_size: int, collocation_axes: dict):
        self.collocation_size = collocation_size
        self.collocation_axes = collocation_axes

    def random_map(self, data, rng: np.random.Generator):
        if "boundary_scattering_kernel" in data:
            del data["boundary_scattering_kernel"]

        for k, axis in self.collocation_axes.items():
            data[k] = rng.choice(
                data[k],
                self.collocation_size,
                axis=axis,
                replace=True,
                shuffle=False,
            )

        return data


def get_datasets(dataset_name, data_dir, data_split, with_info: bool = False):
    """Load a dataset as grain datasource."""
    ds_builder = tfds.builder(dataset_name, data_dir=data_dir)
    ds = ds_builder.as_data_source(split=data_split)
    if with_info:
        return ds, ds_builder.info

    return ds


def preprocessing_pipeline(
    dataset,
    dataset_info: tfds.core.DatasetInfo = None,
    *,
    global_batch_size: int,
    collocation_size: int | None = None,
    global_mesh: Mesh,
    data_pspec: PartitionSpec,
    worker_count: int | None = 0,
    worker_buffer_size: int = 1,
    dataloading_host_index,
    dataloading_host_count,
    shuffle: bool = False,
    data_shuffle_seed=0,
    num_epochs: Optional[int] = 1,
    drop_remainder: bool = True,
):
    """Use grain to pre-process the dataset and return iterators"""
    assert (
        global_batch_size % global_mesh.size == 0
    ), "Batch size should be divisible number of global devices."

    # Batch examples.
    batch_size_per_process = global_batch_size // jax.process_count()

    ops = []
    ops.append(grain.Batch(batch_size_per_process, drop_remainder=drop_remainder))

    if collocation_size:
        collocation_axes = dataset_info.metadata["phase_feature_axis"]
        ops.append(
            SampleCollocationCoords(
                collocation_size=collocation_size, collocation_axes=collocation_axes
            )
        )

    index_sampler = grain.IndexSampler(
        num_records=len(dataset),
        num_epochs=num_epochs,
        shard_options=grain.ShardOptions(
            shard_index=dataloading_host_index,
            shard_count=dataloading_host_count,
            drop_remainder=drop_remainder,
        ),
        shuffle=shuffle,
        seed=data_shuffle_seed,
    )
    dataloader = grain.DataLoader(
        data_source=dataset,
        operations=ops,
        sampler=index_sampler,
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
    )
    multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(
        dataloader, global_mesh, data_pspec
    )

    # Return multi-host jax.Array prep iterator
    return multihost_gen


def make_grain_iterator(config, global_mesh, process_indices):
    """load dataset, preprocess and return iterators"""

    train_ds, ds_info = get_datasets(
        dataset_name=config.dataset_name,
        data_dir=config.data_dir,
        data_split=config.train_split,
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
        global_mesh=global_mesh,
        data_pspec=PartitionSpec(*config.data_sharding),
        worker_count=config.grain_worker_count,
        worker_buffer_size=config.grain_worker_buffer_size,
        dataloading_host_index=process_indices.index(jax.process_index()),
        dataloading_host_count=len(process_indices),
        shuffle=config.enable_data_shuffling,
        num_epochs=None,
        data_shuffle_seed=config.data_shuffle_seed,
    )

    if config.eval_every_steps > 0:
        eval_ds = get_datasets(
            dataset_name=config.dataset_name,
            data_dir=config.data_dir,
            data_split=config.eval_split,
        )

        eval_iter = preprocessing_pipeline(
            dataset=eval_ds,
            global_batch_size=config.eval_batch_size,
            global_mesh=global_mesh,
            data_pspec=PartitionSpec(*config.data_sharding),
            worker_count=config.grain_worker_count,
            worker_buffer_size=config.grain_worker_buffer_size,
            dataloading_host_index=process_indices.index(jax.process_index()),
            dataloading_host_count=len(process_indices),
            shuffle=False,
            data_shuffle_seed=config.data_shuffle_seed,
        )
    else:
        eval_iter = None

    return train_iter, eval_iter
