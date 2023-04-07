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

import copy
import enum
from collections.abc import Generator, Sequence
from typing import Optional

import ml_collections
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from deeprte.data.pipeline import FeatureDict
from deeprte.model.tf import data_transforms


class Split(enum.Enum):
    TRAIN = 1
    TRAIN_AND_VALID = 2
    VALID = 3
    TEST = 4


def _to_tfds_split(split: Split, ratio: float = 0.8):
    if split in (Split.TRAIN, Split.TRAIN_AND_VALID):
        return f"train[:{ratio:.0%}]"
    elif split in (Split.VALID, Split.TEST):
        return f"train[{ratio:.0%}:]"
    else:
        raise ValueError(f"Unknown split {split}")


def load(
    split: Split,
    split_ratio: float,
    is_training: bool,
    # batch_sizes should be:
    # [device_count, per_device_outer_batch_size]
    # total_batch_size = device_count * per_device_outer_batch_size
    batch_sizes: Sequence[int],
    # collocation_sizes should be:
    # [total_collocation_size] or
    # [residual_size, boundary_size, quadrature_size]
    collocation_sizes: Optional[Sequence[int]] = None,
    # repeat number of inner batch, for training the same batch with
    # {repeat} steps of different collocation points
    repeat: Optional[int] = 1,
    # shuffle buffer size
    name: str = "rte",
    data_dir: str = "/workspaces/deeprte/data/tfds",
) -> Generator[FeatureDict, None, None]:
    tfds_split = _to_tfds_split(split, split_ratio)
    total_batch_size = np.prod(batch_sizes)

    ds = tfds.load(
        name, data_dir=data_dir, split=tfds_split, shuffle_files=True
    )
    # tf.data options
    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    options.threading.max_intra_op_parallelism = 1
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    if is_training:
        options.deterministic = False
    ds = ds.with_options(options)

    if is_training:
        ds = ds.cache()
        ds = ds.repeat()
        ds = ds.shuffle(buffer_size=10 * total_batch_size)
        ds = data_transforms.repeat_batch(batch_sizes, repeat)(ds)

    for batch_size in reversed(batch_sizes):
        ds = ds.batch(batch_size, drop_remainder=True)

    if is_training and collocation_sizes:
        rng = tf.random.Generator.from_seed(seed=0)
        ds = ds.map(
            data_transforms.sample_collocation_coords(collocation_sizes, rng),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.prefetch(tf.data.AUTOTUNE)

    yield from tfds.as_numpy(ds)


def make_data_config(
    config: ml_collections.ConfigDict,
) -> ml_collections.ConfigDict:
    """Makes a data config for the input pipeline."""
    cfg = copy.deepcopy(config.data)

    return cfg
