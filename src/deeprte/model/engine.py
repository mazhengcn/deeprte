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


import jax
from absl import logging
from flax import nnx
from jax.sharding import AxisType

from deeprte.configs import config
from deeprte.model import features
from deeprte.model.model import DeepRTE
from deeprte.model.tf import rte_features
from deeprte.train_lib import utils


class RteEngine:
    """Container for DeepRTE model."""

    def __init__(self, config: config.Config, low_memory: bool = True):
        self.config = config
        self.key = jax.random.key(0)
        self.low_memory = low_memory

        # Mesh definition, currently for single process only.
        self.mesh = jax.make_mesh(
            config.mesh_shape,
            config.mesh_axis_names,
            len(config.mesh_shape) * (AxisType.Explicit,),
        )
        jax.set_mesh(self.mesh)

        self.feature_sharding = {
            k: jax.P(None, *config.data_sharding)
            if k in rte_features.PHASE_COORDS_FEATURES
            else jax.P()
            for k in rte_features.FEATURES
        }

        self.model = utils.setup_infer_state(
            model_class=DeepRTE, config=self.config, rng=self.key, mesh=self.mesh
        )
        num_params = utils.calculate_num_params_from_pytree(nnx.state(self.model))
        logging.info(f"Number of model params={num_params}")

        def predict_fn(x):
            return self.model(x) * config.normalization

        self.predict_fn = jax.jit(predict_fn)

    def process_features(
        self, raw_features: features.FeatureDict
    ) -> features.FeatureDict:
        """Processes features to prepare for feeding them into the model."""
        return features.np_data_to_features(raw_features)

    def predict(self, feat: features.FeatureDict) -> jax.Array:
        """Makes a prediction by inferencing the model on the provided
        features.
        """
        logging.info(
            "Running predict with shape(feat) = %s",
            jax.tree.map(lambda x: x.shape, feat),
        )
        predictions = self.predict_fn(jax.device_put(feat, self.feature_sharding))
        jax.tree.map(lambda x: x.block_until_ready(), predictions)
        logging.info(
            "Output shape was %s",
            jax.tree.map(lambda x: x.shape, predictions),
        )
        return predictions
