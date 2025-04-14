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

import enum
from collections.abc import Callable, Sequence
from typing import Optional

import numpy as np
import tensorflow as tf

# Type aliases.
FeaturesDict = dict[str, np.float32]
FeaturesMetadata = dict[str, tuple[np.dtype, Sequence[str | int]]]


class FeatureType(enum.Enum):
    ZERO_DIM = 0
    ONE_DIM = 1
    TWO_DIM = 2


NUM_DIM = 2
NUM_MOMENTS_DIM = "num basis function placeholder"
# Placeholder values that will be replaced with their true value at runtime.
NUM_POSITION_COORDS = "num position coordinates placeholder"
NUM_VELOCITY_COORDS = "num velocity coordinates placeholder"
NUM_PHASE_COORDS = "num phase coordinates placeholder"
NUM_BOUNDARY_COORDS = "num boundary coordinates placeholder"
NUM_SOURCE_COORDS = "num source coordinates placeholder"

FEATURES = {
    # Static features of RTE #
    "phase_coords": (np.float32, [NUM_PHASE_COORDS, 2 * NUM_DIM]),
    "boundary_coords": (np.float32, [NUM_BOUNDARY_COORDS, 2 * NUM_DIM]),
    "boundary_weights": (np.float32, [NUM_BOUNDARY_COORDS]),
    "position_coords": (np.float32, [NUM_POSITION_COORDS, NUM_DIM]),
    "velocity_coords": (np.float32, [NUM_VELOCITY_COORDS, NUM_DIM]),
    "velocity_weights": (np.float32, [NUM_VELOCITY_COORDS]),
    "boundary": (np.float32, [NUM_BOUNDARY_COORDS]),
    "sigma": (np.float32, [NUM_POSITION_COORDS, 2]),
    "scattering_kernel": (np.float32, [NUM_PHASE_COORDS, NUM_VELOCITY_COORDS]),
    "self_scattering_kernel": (
        np.float32,
        [NUM_VELOCITY_COORDS, NUM_VELOCITY_COORDS],
    ),
    "source_coords": (tf.float32, [NUM_SOURCE_COORDS, 2 * NUM_DIM]),
    "source_weights": (tf.float32, [NUM_SOURCE_COORDS]),
    "source": (tf.float32, [NUM_SOURCE_COORDS]),
}
AUTOENCODER_FEATURES = {
    "phase_coords": (tf.float32, [NUM_PHASE_COORDS, 2 * NUM_DIM]),
    "source_coords": (tf.float32, [NUM_SOURCE_COORDS, 2 * NUM_DIM]),
    "source_weights": (tf.float32, [NUM_SOURCE_COORDS]),
    "source": (tf.float32, [NUM_SOURCE_COORDS]),
}
# register_feature("source_label", tf.float32, [NUM_PHASE_COORDS])

GREEN_FUNCTION_FEATURES = {
    "phase_coords": (tf.float32, [NUM_PHASE_COORDS, 2 * NUM_DIM]),
    "moments": (tf.float32, [NUM_MOMENTS_DIM]),
    "basis_inner_product": (tf.float32, [NUM_MOMENTS_DIM, NUM_MOMENTS_DIM]),
    "position_coords": (tf.float32, [NUM_POSITION_COORDS, NUM_DIM]),
    "velocity_coords": (tf.float32, [NUM_VELOCITY_COORDS, NUM_DIM]),
    "velocity_weights": (tf.float32, [NUM_VELOCITY_COORDS]),
    "sigma": (tf.float32, [NUM_POSITION_COORDS, 2]),
    "scattering_kernel": (tf.float32, [NUM_PHASE_COORDS, NUM_VELOCITY_COORDS]),
    "self_scattering_kernel": (
        tf.float32,
        [NUM_VELOCITY_COORDS, NUM_VELOCITY_COORDS],
    ),
}
FEATURE_TYPES = {k: v[0] for k, v in FEATURES.items()}
FEATURE_SIZES = {k: v[1] for k, v in FEATURES.items()}


def register_feature(name: str, type_: np.dtype, shape_: tuple[str | int]):
    """Register extra features used in custom datasets."""
    FEATURES[name] = (type_, shape_)
    FEATURE_TYPES[name] = type_
    FEATURE_SIZES[name] = shape_


# Extra features for training
# register_feature("psi_label", np.float32, [features.NUM_PHASE_COORDS])


def get_features_metadata(feature_names: list[str] = FEATURES) -> FeaturesMetadata:
    """Makes a feature name to type and shape mapping from a list of names."""
    features_metadata = {name: FEATURES[name] for name in feature_names}
    return features_metadata


def get_phase_coords_features() -> list[str]:
    return [k for k in FEATURES if NUM_PHASE_COORDS in FEATURES[k][1]]


def get_phase_feature_axis() -> dict[str, int]:
    return {
        k: FEATURES[k][1].index(NUM_PHASE_COORDS) - len(FEATURES[k][1])
        for k in FEATURES
        if NUM_PHASE_COORDS in FEATURES[k][1]
    }


def shape(
    feature_name: str,
    num_position_coords: int,
    num_velocity_coords: int,
    num_phase_coords: int,
    num_boundary_coords: int,
    num_source_coords: int = 0,
    num_moments_dim: int = 0,
    features: Optional[FeaturesMetadata] = None,
    *args,
    **kwargs,
):
    """Get the shape for the given feature name.

    Args:
      feature_name: String identifier for the feature.
      features: A feature_name to (np.dtype, shape) lookup;
        defaults to FEATURES.

    Returns:
      List of ints representation the tensor size.

    Raises:
      ValueError: If a feature is requested but no concrete
        placeholder value is given.
    """
    features = features or FEATURES

    unused_dtype, raw_sizes = features[feature_name]

    replacements = {
        NUM_POSITION_COORDS: num_position_coords,
        NUM_VELOCITY_COORDS: num_velocity_coords,
        NUM_PHASE_COORDS: num_phase_coords,
        NUM_BOUNDARY_COORDS: num_boundary_coords,
        NUM_SOURCE_COORDS: num_source_coords,
        NUM_MOMENTS_DIM: num_moments_dim,
    }

    sizes = [replacements.get(dimension, dimension) for dimension in raw_sizes]
    for dimension in sizes:
        if isinstance(dimension, str):
            raise ValueError(
                "Could not parse %s (shape: %s) with values: %s"
                % (feature_name, raw_sizes, replacements)
            )
    return sizes


def split_feature(
    features: FeaturesDict,
    filter: Callable[..., bool] = lambda key: key in get_phase_coords_features(),
):
    feat1, feat2 = {}, {}
    for k, v in features.items():
        if filter(k):
            feat1[k] = v
        else:
            feat2[k] = v

    return feat1, feat2
