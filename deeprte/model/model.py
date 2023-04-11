# Copyright 2022 Zheng Ma
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from collections.abc import Mapping
from typing import Any, Optional

import haiku as hk
import jax
import ml_collections

from deeprte.model import features, modules


class RunModel:
    """Container for JAX model."""

    def __init__(
        self,
        config: ml_collections.ConfigDict,
        params: Optional[Mapping[str, Mapping[str, jax.Array]]] = None,
    ):
        self.config = config
        self.params = params

        def _forward_fn(batch):
            model = modules.DeepRTE(self.config)
            return model(
                batch,
                is_training=False,
                compute_loss=False,
                compute_metrics=False,
            )

        self.apply = jax.jit(hk.transform(_forward_fn).apply)
        self.init = jax.jit(hk.transform(_forward_fn).init)

    def init_params(self, feat: features.FeatureDict, random_seed: int = 0):
        """Initializes the model parameters."""

        if not self.params:
            # Init params randomly.
            rng = jax.random.PRNGKey(random_seed)
            self.params = hk.data_structures.to_mutable_dict(
                self.init(rng, feat)
            )
            logging.warning("Initialized parameters randomly")

    def process_features(
        self, raw_features: features.FeatureDict
    ) -> features.FeatureDict:
        """Processes features to prepare for feeding them into the model."""

        return features.np_data_to_features(raw_features)

    def eval_shape(self, feat: features.FeatureDict) -> jax.ShapeDtypeStruct:
        self.init_params(feat)
        logging.info(
            "Running eval_shape with shape(feat) = %s",
            jax.tree_map(lambda x: x.shape, feat),
        )
        shape = jax.eval_shape(
            self.apply, self.params, jax.random.PRNGKey(0), feat
        )
        logging.info("Output shape was %s", shape)
        return shape

    def predict(
        self, feat: features.FeatureDict, random_seed: int
    ) -> Mapping[str, Any]:
        """Makes a prediction by inferencing the model on the provided
        features.
        """
        self.init_params(feat)
        logging.info(
            "Running predict with shape(feat) = %s",
            jax.tree_map(lambda x: x.shape, feat),
        )
        result = self.apply(self.params, jax.random.PRNGKey(random_seed), feat)

        logging.info(
            "Output shape was %s",
            jax.tree_map(lambda x: x.shape, result),
        )
        return result
