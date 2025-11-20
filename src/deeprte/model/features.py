"""Code to generate processed features."""

from collections.abc import Callable, Mapping

import grain
import jax
import numpy as np

from deeprte.model.tf import rte_dataset, rte_features

FeatureDict = Mapping[str, Mapping[str, np.ndarray]]


def np_data_to_features(raw_data: FeatureDict) -> FeatureDict | None:
    """Preprocesses NumPy feature dict using TF pipeline."""
    num_examples = raw_data["functions"]["boundary"].shape[0]

    def function_features(idx: int) -> Mapping[str, np.ndarray]:
        return jax.tree.map(lambda x: x[idx], raw_data["functions"])

    def to_features(idx: int) -> Mapping[str, np.ndarray]:
        raw_example = {**function_features(idx), **raw_data["grid"]}
        return rte_dataset.np_to_tensor_dict(
            raw_example,
            raw_data["shape"],  # ty: ignore
            rte_features.FEATURES.keys(),  # ty: ignore
        )

    dataset = grain.MapDataset.range(num_examples).map(to_features).batch(num_examples)

    return dataset[0]


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
