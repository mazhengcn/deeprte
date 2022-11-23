import ml_collections
import numpy as np
import copy
import jax
import tensorflow as tf
from typing import Generator, Sequence
from deeprte.data.pipeline import FeatureDict
from deeprte.model.tf.rte_dataset import (
    _shard,
    np_to_tensor_dict,
    TensorDict,
    divide_batch_feat,
    make_collocation_axis,
)
import tensorflow_datasets as tfds
from deeprte.model.tf import feature_transform

AUTOTUNE = tf.data.AUTOTUNE


def make_data_config(
    config: ml_collections.ConfigDict,
) -> ml_collections.ConfigDict:
    """Makes a data config for the input pipeline."""
    cfg = copy.deepcopy(config.data)

    return cfg


def process_features(
    features: TensorDict,
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
    seed: int = 0,
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
    batched_feat, unbatched_feat = divide_batch_feat(features)

    total_grid_sizes = tf.cast(
        tf.shape(unbatched_feat["phase_coords"])[-2], dtype=tf.int64
    )

    ds = tf.data.Dataset.from_tensor_slices(batched_feat)

    options = tf.data.Options()
    options.threading.max_intra_op_parallelism = max_intra_op_parallelism
    options.threading.private_threadpool_size = threadpool_size
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    if is_training:
        options.deterministic = False

    if is_training:
        if jax.process_count() > 1:
            # Only cache if we are reading a subset of the dataset.
            ds = ds.cache()
        ds = ds.repeat()
        ds = ds.shuffle(buffer_size=buffer_size)
        ds = feature_transform.repeat_batch(batch_sizes, ds, repeat)

    # batch per_device outer first,
    # since they share the same random grid points
    ds = ds.batch(batch_sizes[-1], drop_remainder=True)
    # construct the inputs structure
    ds = ds.map(feature_transform.construct_batch(unbatched_feat))
    # batch device dim
    ds = ds.batch(batch_sizes[0], drop_remainder=True)

    if is_training:
        assert collocation_sizes <= total_grid_sizes
        g = tf.random.Generator.from_seed(seed=seed)
        ds = ds.map(
            feature_transform.sample_phase_points(
                collocation_sizes=collocation_sizes,
                collocation_features=make_collocation_axis(),
                total_grid_sizes=total_grid_sizes,
                generator=g,
            )
        )

    ds = ds.prefetch(AUTOTUNE)
    ds = ds.with_options(options)

    # convert to a numpy generator
    yield from tfds.as_numpy(ds)
