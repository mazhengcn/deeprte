import ml_collections
import copy
import jax
import tensorflow as tf
import numpy as np
from typing import Generator, Sequence, Optional
from deeprte.data.pipeline import FeatureDict
from deeprte.model.tf.rte_dataset import (
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
    ds: tf.data.Dataset,
    unbatched_feat: TensorDict,
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


def _get_split_num(
    config: ml_collections.config_dict,
):
    split_config = config.data_split
    num_test = split_config.num_test_samples
    num_train_and_val = split_config.num_samples - num_test
    num_train = int(num_train_and_val * split_config.train_validation_split_rate)
    num_val = num_train_and_val - num_train

    return num_test, num_train, num_val


def load_and_split_data(
    features: TensorDict,
    config: Optional[ml_collections.ConfigDict] = None,
    pre_shuffle: bool = False,
    seed: int = 0,
):
    batched_feat, unbatched_feat = divide_batch_feat(features)

    if pre_shuffle:
        arange = tf.range(batched_feat["psi_label"].shape[0])
        shuffled_indices = tf.random.shuffle(arange, seed=seed)
        batched_feat = tf.nest.map_structure(
            lambda x: tf.gather(x, shuffled_indices.numpy(), axis=0), batched_feat
        )

    if config and config.data_split.is_split_datasets:

        num_test, num_train, num_val = _get_split_num(config)

        def _make_ds(
            num_train: Optional[int] = None,
        ) -> tf.data.Dataset:
            _start, _end = shard(
                jax.process_index(),
                jax.process_count(),
                num_val=num_val,
                num_train=num_train,
            )
            return tf.data.Dataset.from_tensor_slices(
                tf.nest.map_structure(
                    lambda arr: arr[(num_test + _start) : (num_test + _end)],
                    batched_feat,
                )
            )

        test_ds = tf.data.Dataset.from_tensor_slices(
            tf.nest.map_structure(lambda arr: arr[:num_test], batched_feat)
        )
        train_ds = _make_ds(num_train=num_train)
        val_ds = _make_ds()

        return (test_ds, train_ds, val_ds), unbatched_feat

    else:
        ds = tf.data.Dataset.from_tensor_slices(batched_feat)
        return ds, unbatched_feat


def np_example_to_features(
    np_features: FeatureDict,
    batch_sizes: Sequence[int],
    data_config: ml_collections.ConfigDict,
    features_names: Optional[Sequence[str]] = None,
):
    tf_data = np_to_tensor_dict(np_features, features_names)

    ds, unbatched_feat = load_and_split_data(
        features=tf_data,
        config=data_config,
        pre_shuffle=True,
        seed=data_config.seed,
    )

    if data_config and data_config.is_split_datasets:
        test_ds, train_ds, val_ds = ds
        train_input = process_features(
            ds=train_ds,
            unbatched_feat=unbatched_feat,
            is_training=True,
            batch_sizes=batch_sizes,
            collocation_sizes=data_config.train.collocation_sizes,
            seed=data_config.seed,
            repeat=data_config.train.repeat,
            buffer_size=data_config.buffer_size,
            threadpool_size=data_config.threadpool_size,
            max_intra_op_parallelism=data_config.max_intra_op_parallelism,
        )
        val_input = process_features(
            ds=val_ds,
            unbatched_feat=unbatched_feat,
            is_training=False,
            batch_sizes=batch_sizes,
            seed=data_config.seed,
            buffer_size=data_config.buffer_size,
            threadpool_size=data_config.threadpool_size,
            max_intra_op_parallelism=data_config.max_intra_op_parallelism,
        )
        return train_input, val_input
    else:
        input = process_features(
            ds=ds,
            unbatched_feat=unbatched_feat,
            is_training=False,
            batch_sizes=batch_sizes,
            seed=data_config.seed,
            buffer_size=data_config.buffer_size,
            threadpool_size=data_config.threadpool_size,
            max_intra_op_parallelism=data_config.max_intra_op_parallelism,
        )
        return input
