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
from typing import Optional

import grain.python as grain
import jax
import numpy as np

from deeprte.input_pipeline.dataset import PHASE_FEATURE_AXIS, RTEDataset
from deeprte.train_lib import multihost_dataloading


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

    def random_map(self, element, rng: np.random.Generator):
        if "boundary_scattering_kernel" in element:
            del element["boundary_scattering_kernel"]

        num_phase_coords = (element["phase_coords"].shape)[
            self.collocation_axes["phase_coords"]
        ]
        phase_coords_indices = rng.permutation(num_phase_coords)[
            : self.collocation_size
        ]

        for k, axis in self.collocation_axes.items():
            if k in element:
                element[k] = np.take(element[k], phase_coords_indices, axis=axis)

        return element


def preprocessing_pipeline(
    dataset: RTEDataset,
    *,
    global_batch_size: int,
    collocation_size: int | None = None,
    sharding: jax.P,
    worker_count: int | None = 0,
    worker_buffer_size: int = 1,
    shuffle: bool = False,
    data_shuffle_seed=0,
    num_epochs: Optional[int] = 1,
    drop_remainder: bool = True,
):
    """Use grain to pre-process the dataset and return iterators"""

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
        dataloader, sharding
    )

    # Return multi-host jax.Array prep iterator
    return multihost_gen
