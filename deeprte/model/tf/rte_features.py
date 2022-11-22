import tensorflow as tf
from typing import Dict, Optional, Sequence, Tuple, Union

# Type aliases.
FeaturesMetadata = Dict[str, Tuple[tf.dtypes.DType, Sequence[Union[str, int]]]]

# Placeholder values that will be replaced with their true value at runtime.
NUM_SAMPLE = "num batch placeholder"
NUM_POSITION_COORDS = "num position coords placeholder"
NUM_VELOCITY_COORDS = "num velocity placeholder"
# NUM_BOUNDARY_VELOCITY = "num boundary velocity placeholder"
NUM_PHASE_COORDS = "num phase placeholder"
NUM_BOUNDARY_COORDS = "num boundary placeholder"

NUM_DIMENSION = 2

FEATURES = {
    #### Static features of rte ####
    "sigma": (tf.float32, [NUM_SAMPLE, NUM_POSITION_COORDS, 2]),
    "boundary": (tf.float32, [NUM_SAMPLE, NUM_BOUNDARY_COORDS]),
    "position_coords": (tf.float32, [NUM_POSITION_COORDS, NUM_DIMENSION]),
    "velocity_coords": (tf.float32, [NUM_VELOCITY_COORDS, NUM_DIMENSION]),
    "phase_coords": (tf.float32, [NUM_PHASE_COORDS, 2 * NUM_DIMENSION]),
    "scattering_kernel": (
        tf.float32,
        [NUM_SAMPLE, NUM_PHASE_COORDS, NUM_VELOCITY_COORDS],
    ),
    "boundary_coords": (
        tf.float32,
        [NUM_BOUNDARY_COORDS, 2 * NUM_DIMENSION],
    ),
    "boundary_weights": (tf.float32, [NUM_BOUNDARY_COORDS]),
    "velocity_weights": (tf.float32, [NUM_VELOCITY_COORDS]),
}

_FEATURE_NAMES = [k for k in FEATURES.keys()]
_COLLOCATION_FEATURE_NAMES = [
    k for k in FEATURES.keys() if NUM_POSITION_COORDS in FEATURES[k][1]
]
_BATCH_FEATURE_NAMES = [k for k in FEATURES.keys() if NUM_SAMPLE in FEATURES[k][1]]


def register_feature(name: str, type_: tf.dtypes.DType, shape_: Tuple[Union[str, int]]):
    """Register extra features used in custom datasets."""
    FEATURES[name] = (type_, shape_)


def shape(
    feature_name: str,
    num_batch: int,
    num_position_points: int,
    num_velocity: int,
    num_pahse_points: int,
    num_boundary_points: int,
    num_boundary_velocity: Optional[int] = None,
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
        NUM_BATCH: num_batch,
        NUM_POSITION_POINTS: num_position_points,
        NUM_VELOCITY: num_velocity,
        NUM_PHASE_POINTS: num_pahse_points,
        NUM_BOUNDARY_POINTS: num_boundary_points,
    }

    if num_boundary_velocity is not None:
        replacements[NUM_BOUNDARY_VELOCITY] = num_boundary_velocity
    else:
        assert num_velocity % 2 == 0
        replacements[NUM_BOUNDARY_VELOCITY] = int(num_velocity / 2)

    sizes = [replacements.get(dimension, dimension) for dimension in raw_sizes]
    for dimension in sizes:
        if isinstance(dimension, str):
            raise ValueError(
                "Could not parse %s (shape: %s) with values: %s"
                % (feature_name, raw_sizes, replacements)
            )
    return sizes
