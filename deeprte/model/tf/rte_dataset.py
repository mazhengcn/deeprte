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

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import tensorflow as tf

from deeprte.model.tf import rte_features

TensorDict = Dict[str, tf.Tensor]
AUTOTUNE = tf.data.AUTOTUNE


def make_collocation_axis():
    axis_dict = {
        k: rte_features.FEATURES[k][1].index(rte_features.NUM_PHASE_COORDS)
        - len(rte_features.FEATURES[k][1])
        for k in rte_features.COLLOCATION_FEATURE_NAMES
    }
    return axis_dict


def make_boudary_sample_axis():
    axis_dict = {
        k: rte_features.FEATURES[k][1].index(rte_features.NUM_BOUNDARY_COORDS)
        - len(rte_features.FEATURES[k][1])
        for k in rte_features.BOUNDARY_FEATURE_NAMES
    }
    return axis_dict


def _make_features_metadata(
    feature_names: Sequence[str],
) -> rte_features.FeaturesMetadata:
    """Makes a feature name to type and shape mapping from a list of names."""

    features_metadata = {
        name: rte_features.FEATURES[name] for name in feature_names
    }
    return features_metadata


def parse_reshape_logic(
    parsed_features: TensorDict,
    shape_dict,
    features: rte_features.FeaturesMetadata,
) -> TensorDict:

    for k, v in parsed_features.items():
        new_shape = rte_features.shape(
            feature_name=k,
            **shape_dict,
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
            parsed_features[k] = tf.reshape(
                v, new_shape, name="reshape_%s" % k
            )

    return parsed_features


def np_to_tensor_dict(
    np_examples: Mapping[str, np.ndarray],
    features_names: Optional[Sequence[str]] = None,
) -> TensorDict:
    """Creates dict of tensors from a dict of NumPy arrays.

    Args:
        np_examples: A dict of NumPy feature arrays.
        features: A list of strings of feature names to be returned in the
            dataset.

    Returns:
        A dictionary of features mapping feature names to features.
            Only the given features are returned, all other ones are
            filtered out.
    """
    features_names = features_names or rte_features.FEATURE_NAMES
    features_metadata = _make_features_metadata(features_names)
    tensor_dict = {
        k: tf.constant(v)
        for k, v in np_examples.items()
        if k in features_metadata
    }
    shape_dict = np_examples["shapes"]
    # Ensures shapes are as expected. Needed for setting size of empty features
    # e.g. when no template hits were found.
    tensor_dict = parse_reshape_logic(
        tensor_dict, shape_dict, features_metadata
    )
    return tensor_dict
