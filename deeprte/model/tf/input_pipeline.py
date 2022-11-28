import ml_collections
import copy
import jax
import tensorflow as tf
import numpy as np
from typing import Generator, Sequence, Optional
from deeprte.data.pipeline import FeatureDict
from deeprte.model.tf.rte_dataset import (
    divide_batch_feat,
    TensorDict,
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


def make_device_batch(
    global_batch_size: int,
    num_devices: int,
):
    per_device_batch_size, ragged = divmod(global_batch_size, num_devices)
    # Raise error if not divisible
    if ragged:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by "
            f"number of devices {num_devices}"
        )
    return [num_devices, per_device_batch_size]


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
    collocation_sizes: Optional[int] = None,
    # repeat number of inner batch, for training the same batch with
    # {repeat} steps of different collocation points
    seed: int = jax.process_index(),
    repeat: int | None = 1,
    # shuffle buffer size
    buffer_size: int = 5_000,
    # Dataset options
    threadpool_size: int = 48,
    max_intra_op_parallelism: int = 1,
) -> Generator[FeatureDict, None, None]:

    batched_feat, unbatched_feat = divide_batch_feat(features)
    ds = tf.data.Dataset.from_tensor_slices(batched_feat)

    if is_training:
        if not collocation_sizes and not repeat:
            raise ValueError(
                "`collocation_sizes` and `repeat` should not be None"
                "when `is_training=True`"
            )
    total_grid_sizes = tf.cast(
        tf.shape(unbatched_feat["phase_coords"])[-2], dtype=tf.int64
    )
    options = tf.data.Options()
    options.threading.max_intra_op_parallelism = max_intra_op_parallelism
    options.threading.private_threadpool_size = threadpool_size
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    if is_training:
        options.deterministic = False

        assert collocation_sizes <= total_grid_sizes

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
    g = tf.random.Generator.from_seed(seed=seed)
    ds = ds.map(
        feature_transform.construct_batch(
            unbatched_feat=unbatched_feat,
            collocation_sizes=collocation_sizes,
            collocation_features=make_collocation_axis(),
            total_grid_sizes=total_grid_sizes,
            generator=g,
            is_training=is_training,
        )
    )
    # batch device dim
    ds = ds.batch(batch_sizes[0], drop_remainder=True)

    ds = ds.prefetch(AUTOTUNE)
    ds = ds.with_options(options)

    # convert to a numpy generator
    yield from tfds.as_numpy(ds)


def shard(
    shard_index: int,
    num_shards: int,
    num_val: int,
    num_train: Optional[int] = None,
) -> tuple[int, int]:
    """Returns [start, end) for the given shard index."""
    assert shard_index < num_shards

    def _get_endpoint(num):
        arange = np.arange(num)
        shard_range = np.split(arange, num_shards)[shard_index]
        start, end = shard_range[0], (shard_range[-1] + 1)
        return start, end

    if num_train:
        start, end = _get_endpoint(num_train)
        offset = num_val
        start += offset
        end += offset
    else:
        start, end = _get_endpoint(num_val)

    return start, end


# def get_split_num(
#     config: ml_collections.config_dict,
# ):
#     split_config = config.data_split
#     num_test = split_config.num_test_samples
#     num_train_and_val = split_config.num_samples - num_test
#     num_train = int(num_train_and_val * split_config.train_validation_split_rate)
#     num_val = num_train_and_val - num_train

#     return num_test, num_train, num_val


def _split(
    features: TensorDict | FeatureDict,
    split_rate: float,
):
    num_train_and_val = list(features.values())[0].shape[0]
    num_train = int(num_train_and_val * split_rate)
    num_val = num_train_and_val - num_train

    def _make_ds(
        num_train: Optional[int] = None,
    ) -> tf.data.Dataset:
        _start, _end = shard(
            jax.process_index(),
            jax.process_count(),
            num_val=num_val,
            num_train=num_train,
        )
        return tf.nest.map_structure(
            lambda arr: arr[_start:_end],
            features,
        )

    train_ds = _make_ds(num_train=num_train)
    val_ds = _make_ds()

    return (train_ds, val_ds)


def split_feat(
    features: TensorDict | FeatureDict,
    split_rate: float,
):
    batched_data, unbatched_data = divide_batch_feat(features)
    assert batched_data

    train_ds, val_ds = _split(
        features=batched_data,
        split_rate=split_rate,
    )

    return {**train_ds, **unbatched_data}, {**val_ds, **unbatched_data}
