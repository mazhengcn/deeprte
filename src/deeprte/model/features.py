"""Code to generate processed features."""

from collections.abc import Callable, Mapping

import numpy as np
import tensorflow as tf

from deeprte.model.tf import rte_dataset, rte_features

FeatureDict = Mapping[str, Mapping[str, np.ndarray]]


def np_data_to_features(raw_data: FeatureDict) -> FeatureDict:
    """Preprocesses NumPy feature dict using TF pipeline."""

    num_examples = raw_data["functions"]["boundary"].shape[0]

    def to_features(x):
        raw_example = {**x, **raw_data["grid"]}
        tensor_dict = rte_dataset.np_to_tensor_dict(
            raw_example,
            raw_data["shape"],  # ty: ignore
            rte_features.FEATURES.keys(),  # ty: ignore
        )
        return tensor_dict

    dataset = (
        tf.data.Dataset.from_tensor_slices(raw_data["functions"])
        .map(to_features, tf.data.AUTOTUNE)
        .batch(num_examples)
    )
    processed_features = dataset.get_single_element()

    return tf.nest.map_structure(lambda x: x.numpy(), processed_features)


def split_feature(
    features,
    filter: Callable[..., bool] = lambda key: key in rte_features.PHASE_COORDS_FEATURES,
):
    feat1, feat2 = {}, {}
    for k, v in features.items():
        if filter(k):
            feat1[k] = v
        else:
            feat2[k] = v

    return feat1, feat2
