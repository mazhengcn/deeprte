"""Data transform for DeepRTE."""

import numpy as np
import tensorflow as tf

from typing import Optional, Sequence
from deeprte.model.tf.rte_features import _COLLOCATION_FEATURE_NAMES, _FEATURE_NAMES
from deeprte.model.tf.rte_dataset import TensorDict


def curry1(f):
    """Supply all arguments but the first."""

    def fc(*args, **kwargs):
        return lambda x: f(x, *args, **kwargs)

    return fc


@curry1
def sample_phase_points(
    features,
    collocation_sizes,
    seed,
):
    """Sample phase points randomly and take collocation points.

    Args:
        featrues: batch to sample.
        collocation_sizes: number of collocation points.
        seed: random seed.

    Returns:
        sampled data.
    """
    total_grid_sizes = tf.cast(tf.shape(features["phase_coords"])[-2], dtype=tf.int64)

    if total_grid_sizes >= collocation_sizes:
        g = tf.random.Generator.from_seed(seed)
        idx = g.uniform(
            (collocation_sizes,),
            minval=0,
            maxval=total_grid_sizes,
            dtype=tf.int64,
        )
    else:
        raise ValueError("total_grid_sizes < collocation_sizes")

    # for k in _COLLOCATION_FEATURE_NAMES:
    #     if k in features:
    #         features[k] = tf.gather(features[k], idx, axis=-2)

    features = sample_features(
        idx=idx,
        axis=-2,
        sampled_features_names=_COLLOCATION_FEATURE_NAMES,
    )(features)

    return features


@curry1
def select_feat(feature, feature_name_list):
    return {k: v for k, v in feature.items() if k in feature_name_list}


@curry1
def sample_features(
    features: TensorDict,
    idx: int,
    axis: int,
    sampled_features_names: Optional[Sequence[str]] = None,
):
    sampled_features_names = sampled_features_names or _FEATURE_NAMES
    for k in sampled_features_names:
        if k in features:
            features[k] = tf.gather(features[k], idx, axis=axis)

    return features
