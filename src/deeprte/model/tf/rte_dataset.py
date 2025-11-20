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

from collections.abc import Mapping, Sequence

import numpy as np

from deeprte.model.tf import rte_features

FeatureDict = dict[str, np.ndarray]


def _make_features_metadata(
    feature_names: Sequence[str],
) -> rte_features.FeaturesMetadata:
    """Makes a feature name to type and shape mapping from a list of names."""
    features_metadata = {name: rte_features.FEATURES[name] for name in feature_names}
    return features_metadata


def parse_reshape_logic(
    parsed_features: FeatureDict,
    placeholder_shapes: Mapping[str, int],
    features: rte_features.FeaturesMetadata,
) -> FeatureDict:
    for k, v in parsed_features.items():
        new_shape = rte_features.shape(
            feature_name=k,
            num_position_coords=placeholder_shapes["num_position_coords"],
            num_velocity_coords=placeholder_shapes["num_velocity_coords"],
            num_phase_coords=placeholder_shapes["num_phase_coords"],
            num_boundary_coords=placeholder_shapes["num_boundary_coords"],
            features=features,
        )
        parsed_features[k] = np.reshape(v, new_shape)

    return parsed_features


def np_to_tensor_dict(
    np_example: Mapping[str, np.ndarray],
    placeholder_shapes: Mapping[str, int],
    features_names: Sequence[str] | None = None,
) -> FeatureDict:
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
    features_metadata = _make_features_metadata(features_names)  # ty: ignore
    tensor_dict = {
        k: np.asarray(v) if not isinstance(v, np.ndarray) else v
        for k, v in np_example.items()
        if k in features_metadata
    }
    # Ensures shapes are as expected. Needed for setting size of empty features
    # e.g. when no template hits were found.
    return parse_reshape_logic(tensor_dict, placeholder_shapes, features_metadata)
