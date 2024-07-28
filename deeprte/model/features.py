"""Code to generate processed features."""

from collections.abc import Mapping

import numpy as np
import tensorflow as tf

from deeprte.model.tf import rte_dataset, rte_features

FeatureDict = Mapping[str, Mapping[str, np.ndarray]]

PHASE_FEATURES = {
    k for k, v in rte_features.FEATURES.items() if rte_features.NUM_PHASE_COORDS in v[1]
}


def np_data_to_features(raw_data: FeatureDict) -> FeatureDict:
    """Preprocesses NumPy feature dict using TF pipeline."""

    num_examples = raw_data["functions"]["boundary"].shape[0]

    def to_features(x):
        raw_example = {**x, **raw_data["grid"]}
        tensor_dict = rte_dataset.np_to_tensor_dict(
            raw_example, raw_data["shape"], rte_features.FEATURES.keys()
        )
        return tensor_dict

    dataset = (
        tf.data.Dataset.from_tensor_slices(raw_data["functions"])
        .map(to_features, tf.data.AUTOTUNE)
        .batch(num_examples)
    )
    processed_features = dataset.get_single_element()

    return tf.nest.map_structure(lambda x: x.numpy(), processed_features)
