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

"""Dataset pipline."""

from __future__ import annotations

from typing import Optional, Sequence, Mapping, Dict, Generator
from ml_collections import config_dict

import numpy as np
import tensorflow as tf
import jax

from deeprte.model.tf import rte_features
from deeprte.data.pipeline import FeatureDict

TensorDict = Dict[str, tf.Tensor]
AUTOTUNE = tf.data.AUTOTUNE


def _make_features_metadata(
    feature_names: Sequence[str],
) -> rte_features.FeaturesMetadata:
    """Makes a feature name to type and shape mapping from a list of names."""

    features_metadata = {name: rte_features.FEATURES[name] for name in feature_names}
    return features_metadata


def np_to_tensor_dict(
    np_example: Mapping[str, np.ndarray],
    placeholder_shape_config: config_dict,
    features: Optional[Sequence[str]] = None,
) -> TensorDict:
    """Creates dict of tensors from a dict of NumPy arrays.

    Args:
        np_example: A dict of NumPy feature arrays.
        features: A list of strings of feature names to be returned in the  dataset.

    Returns:
        A dictionary of features mapping feature names to features. Only the given
        features are returned, all other ones are filtered out.
    """
    features = features or rte_features._FEATURE_NAMES
    features_metadata = _make_features_metadata(features)
    tensor_dict = {
        k: tf.constant(v) for k, v in np_example.items() if k in features_metadata
    }

    # Ensures shapes are as expected. Needed for setting size of empty features
    # e.g. when no template hits were found.
    tensor_dict = parse_reshape_logic(
        tensor_dict, placeholder_shape_config, features_metadata
    )
    return tensor_dict


def parse_reshape_logic(
    parsed_features: TensorDict,
    placeholder_shape_config: config_dict,
    features: rte_features.FeaturesMetadata,
) -> TensorDict:

    for k, v in parsed_features.items():
        new_shape = rte_features.shape(
            feature_name=k,
            num_batch=placeholder_shape_config.num_batch,
            num_position_points=placeholder_shape_config.num_position_points,
            num_velocity=placeholder_shape_config.num_velocity,
            num_pahse_points=placeholder_shape_config.num_phase_points,
            num_boundary_points=placeholder_shape_config.num_boundary_points,
            features=features,
        )
        new_shape_size = tf.constant(1, dtype=tf.int32)
        for dim in new_shape:
            new_shape_size *= tf.cast(dim, tf.int32)

        assert_equal = tf.assert_equal(
            tf.size(v),
            new_shape_size,
            name="assert_%s_shape_correct" % k,
            message="The size of feature %s (%s) could not be reshaped "
            "into %s" % (k, tf.size(v), new_shape),
        )
        with tf.control_dependencies([assert_equal]):
            parsed_features[k] = tf.reshape(v, new_shape, name="reshape_%s" % k)

    return parsed_features


def _shard(
    split: config_dict,
    shard_index: int,
    num_shards: int,
    is_training: bool,
) -> tuple[int, int]:
    """Returns [start, end) for the given shard index."""
    assert shard_index < num_shards

    def _get_endpoint(num):
        arange = np.arange(num)
        shard_range = np.array_split(arange, num_shards)[shard_index]
        start, end = shard_range[0], (shard_range[-1] + 1)
        return start, end

    if is_training:
        start, end = _get_endpoint(split.num_train_examples)
        offset = split.num_eval_examples
        start += offset
        end += offset
    else:
        start, end = _get_endpoint(split.num_eval_examples)

    return start, end


def process_features(
    features: FeatureDict,
    split: config_dict,
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
