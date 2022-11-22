import ml_collections
import numpy as np
import copy
import jax
import tensorflow as tf
from typing import Generator, Sequence
from deeprte.data.pipeline import FeatureDict
from deeprte.model.tf.rte_dataset import _shard, np_to_tensor_dict


def make_data_config(
    config: ml_collections.ConfigDict,
) -> ml_collections.ConfigDict:
    """Makes a data config for the input pipeline."""
    cfg = copy.deepcopy(config.data)

    return cfg


def process_features(
    features: FeatureDict,
    split: ml_collections.config_dict,
    is_training: bool,
    # batch_sizes should be:
    # [device_count, per_device_outer_batch_size]
    # total_batch_size = device_count * per_device_outer_batch_size
    batch_sizes: Sequence[int],
    # collocation_sizes should be:
    # total_collocation_size or
    # [residual_size, boundary_size, quadrature_size]
    collocation_sizes: int | Sequence[int] | None,
    # repeat number of inner batch, for training the same batch with
    # {repeat} steps of different collocation points
    repeat: int | None = 1,
    # shuffle buffer size
    buffer_size: int = 5_000,
    # Dataset options
    threadpool_size: int = 48,
    max_intra_op_parallelism: int = 1,
) -> Generator[FeatureDict, None, None]:

    if is_training:
        if not collocation_sizes and not repeat:
            raise ValueError(
                "`collocation_sizes` and `repeat` should not be None"
                "when `is_training=True`"
            )

    start, end = _shard(
        split, jax.process_index(), jax.process_count(), is_training=is_training
    )

    tensor_features = np_to_tensor_dict(features)

    ds = tf.data.Dataset.from_tensor_slices(
        tf.nest.map_structure(
            lambda arr: arr[start, end],
        )
    )
    return
