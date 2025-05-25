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
import tensorflow as tf
from clu.data import dataset_iterator
from jax.sharding import Mesh, PartitionSpec
from rte_dataset.builders import pipeline

from deeprte.input_pipeline import splits, utils
from deeprte.model.tf import rte_dataset
from deeprte.model.tf import rte_features as features
from deeprte.train_lib import multihost_dataloading

ArraySpec = dataset_iterator.ArraySpec
ArraySpecDict = dict[str, ArraySpec]
Data = dict[str, Any]
DatasetIterator = dataset_iterator.DatasetIterator


features.register_feature(
    "psi_label",
    tf.float32,
    [features.NUM_PHASE_COORDS],  # ty: ignore
)  # ty: ignore

FEATURES = features.FEATURES
PHASE_FEATURE_AXIS = {
    k: FEATURES[k][1].index(features.NUM_PHASE_COORDS) - len(FEATURES[k][1])
    for k in FEATURES
    if features.NUM_PHASE_COORDS in FEATURES[k][1]
}


class RTEDataset(grain.RandomAccessDataSource):
    def __init__(self, raw_data: Data) -> None:
        self.raw_data = raw_data

    def __len__(self) -> int:
        return self.raw_data["shape"]["num_examples"]

    def __getitem__(self, index: int) -> Data:
        np_example = {
            **jax.tree.map(lambda x: x[index], self.raw_data["functions"]),
            **self.raw_data["grid"],
        }
        tensor_dict = rte_dataset.np_to_tensor_dict(
            np_example,
            self.raw_data["shape"],
            FEATURES.keys(),  # ty: ignore
        )
        return jax.tree.map(lambda x: x.numpy(), tensor_dict)

    def __repr__(self):
        return "RTEDataset: 0.0.2"

    @property
    def metadata(self) -> dict[str, dict[str, int]]:
        return {
            "shapes": self.raw_data["shape"],
            "normalization": jax.tree.map(
                lambda x: str(x), self.raw_data["normalization"]
            ),
        }


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

    collocation_size: int
    collocation_axes: dict

    def random_map(self, data, rng: np.random.Generator):
        if "boundary_scattering_kernel" in data:
            del data["boundary_scattering_kernel"]

        num_phase_coords = (data["phase_coords"].shape)[
            self.collocation_axes["phase_coords"]
        ]
        phase_coords_indices = rng.permutation(num_phase_coords)[
            : self.collocation_size
        ]

        for k, axis in self.collocation_axes.items():
            if k in data:
                data[k] = np.take(data[k], phase_coords_indices, axis=axis)

        return data


def get_datasets(dataset_name, data_dir, data_split: str) -> RTEDataset:
    """Load a dataset as grain datasource."""
    data_pipeline = pipeline.DataPipeline(data_dir, [dataset_name])
    raw_data = data_pipeline.process(normalization=True)

    num_examples = raw_data["shape"]["num_examples"]
    split_instr = splits.get_split_instruction(data_split, num_examples)  # ty: ignore

    raw_data["functions"] = jax.tree.map(
        lambda x: x[split_instr.from_ : split_instr.to], raw_data["functions"]
    )
    raw_data["shape"]["num_examples"] = raw_data["functions"]["psi_label"].shape[0]

    return RTEDataset(raw_data)  # ty: ignore


def preprocessing_pipeline(
    dataset: RTEDataset,
    *,
    global_batch_size: int,
    collocation_size: int | None = None,
    global_mesh: Mesh,
    data_pspec: PartitionSpec,
    worker_count: int | None = 0,
    worker_buffer_size: int = 1,
    shuffle: bool = False,
    data_shuffle_seed=0,
    num_epochs: Optional[int] = 1,
    drop_remainder: bool = True,
):
    """Use grain to pre-process the dataset and return iterators"""
    assert global_batch_size % global_mesh.size == 0, (
        "Batch size should be divisible number of global devices."
    )

    # Batch examples.
    batch_size_per_process = global_batch_size // jax.process_count()

    ops = []
    ops.append(grain.Batch(batch_size_per_process, drop_remainder=drop_remainder))

    if collocation_size:
        ops.append(
            SampleCollocationCoords(
                collocation_size=collocation_size, collocation_axes=PHASE_FEATURE_AXIS
            )
        )

    index_sampler = grain.IndexSampler(
        num_records=len(dataset),
        num_epochs=num_epochs,
        shard_options=grain.ShardByJaxProcess(),
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
        dataloader,  # ty: ignore
        global_mesh,
        data_pspec,
    )

    # Return multi-host jax.Array prep iterator
    return multihost_gen


def make_grain_iterator(config, global_mesh):
    """load dataset, preprocess and return iterators"""

    train_ds = get_datasets(
        dataset_name=config.dataset_name,
        data_dir=config.data_dir,
        data_split=config.train_split,
    )
    norm_dict = train_ds.metadata["normalization"]
    config.normalization = utils.get_normalization_ratio(
        norm_dict["psi_range"], norm_dict["boundary_range"]
    )

    train_iter = preprocessing_pipeline(
        dataset=train_ds,
        global_batch_size=config.global_batch_size,
        collocation_size=config.collocation_size,
        global_mesh=global_mesh,
        data_pspec=PartitionSpec(*config.data_sharding),
        worker_count=config.grain_worker_count,
        worker_buffer_size=config.grain_worker_buffer_size,
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
            shuffle=False,
            data_shuffle_seed=config.data_shuffle_seed,
        )
    else:
        eval_iter = None

    return train_iter, eval_iter
