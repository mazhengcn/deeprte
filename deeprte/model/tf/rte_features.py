import tensorflow as tf
from typing import Dict, Optional, Sequence, Tuple, Union

# Type aliases.
FeaturesMetadata = Dict[str, Tuple[tf.dtypes.DType, Sequence[Union[str, int]]]]

# Placeholder values that will be replaced with their true value at runtime.
NUM_SAMPLES = "num batch placeholder"
NUM_POSITION_COORDS = "num position coords placeholder"
NUM_VELOCITY_COORDS = "num velocity placeholder"
# NUM_BOUNDARY_VELOCITY = "num boundary velocity placeholder"
NUM_PHASE_COORDS = "num phase placeholder"
NUM_BOUNDARY_COORDS = "num boundary placeholder"

NUM_DIM = 2

FEATURES = {
    #### Static features of rte ####
    "sigma": (tf.float32, [NUM_SAMPLES, NUM_POSITION_COORDS, 2]),
    "boundary": (tf.float32, [NUM_SAMPLES, NUM_BOUNDARY_COORDS]),
    "position_coords": (tf.float32, [NUM_POSITION_COORDS, NUM_DIM]),
    "velocity_coords": (tf.float32, [NUM_VELOCITY_COORDS, NUM_DIM]),
    "phase_coords": (tf.float32, [NUM_PHASE_COORDS, 2 * NUM_DIM]),
    "scattering_kernel": (
        tf.float32,
        [NUM_SAMPLES, NUM_PHASE_COORDS, NUM_VELOCITY_COORDS],
    ),
    "boundary_coords": (
        tf.float32,
        [NUM_BOUNDARY_COORDS, 2 * NUM_DIM],
    ),
    "boundary_weights": (tf.float32, [NUM_BOUNDARY_COORDS]),
    "velocity_weights": (tf.float32, [NUM_VELOCITY_COORDS]),
    "psi_label": (tf.float32, [NUM_SAMPLES, NUM_PHASE_COORDS]),
}

_FEATURE_NAMES = [k for k in FEATURES.keys()]
_COLLOCATION_FEATURE_NAMES = [
    k for k in FEATURES.keys() if NUM_PHASE_COORDS in FEATURES[k][1]
]
_BATCH_FEATURE_NAMES = [k for k in FEATURES.keys() if NUM_SAMPLES in FEATURES[k][1]]


def register_feature(name: str, type_: tf.dtypes.DType, shape_: Tuple[Union[str, int]]):
    """Register extra features used in custom datasets."""
    FEATURES[name] = (type_, shape_)


def shape(
    feature_name: str,
    num_samples: int,
    num_position_coords: int,
    num_velocity_coords: int,
    num_phase_coords: int,
    num_boundary_coords: int,
    features: Optional[FeaturesMetadata] = None,
):
    """Get the shape for the given feature name.

    Args:
      feature_name: String identifier for the feature.
      features: A feature_name to (tf_dtype, shape) lookup; defaults to FEATURES.

    Returns:
      List of ints representation the tensor size.

    Raises:
      ValueError: If a feature is requested but no concrete placeholder value is given.
    """
    features = features or FEATURES

    unused_dtype, raw_sizes = features[feature_name]

    replacements = {
        NUM_SAMPLES: num_samples,
        NUM_POSITION_COORDS: num_position_coords,
        NUM_VELOCITY_COORDS: num_velocity_coords,
        NUM_PHASE_COORDS: num_phase_coords,
        NUM_BOUNDARY_COORDS: num_boundary_coords,
    }

    sizes = [replacements.get(dimension, dimension) for dimension in raw_sizes]
    for dimension in sizes:
        if isinstance(dimension, str):
            raise ValueError(
                "Could not parse %s (shape: %s) with values: %s"
                % (feature_name, raw_sizes, replacements)
            )
    return sizes
