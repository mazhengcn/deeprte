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
from jax.sharding import PartitionSpec as P

from deeprte.configs import default
from deeprte.model import features
from deeprte.model.mapping import inference_subbatch
from deeprte.model.model import DeepRTE
from deeprte.model.tf import rte_features
from deeprte.train_lib import utils


class RteEngine:
    """Container for DeepRTE model."""

    def __init__(self, config: default.Config, low_memory: bool = True):
        self.config = config
        self.key = jax.random.key(0)
        self.low_memory = low_memory

        # Mesh definition, currently for single process only.
        devices_array = utils.create_device_mesh(config, devices=jax.local_devices())
        self.mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

        replicated_sharding = jax.sharding.NamedSharding(self.mesh, P(None))
        data_sharding = jax.sharding.NamedSharding(
            self.mesh, P(None, *config.data_sharding)
        )
        feature_sharding = {
            k: data_sharding
            if k in rte_features.PHASE_COORDS_FEATURES
            else replicated_sharding
            for k in rte_features.FEATURES
        }

        self.model = utils.setup_infer_state(
            model_class=DeepRTE, config=self.config, rng=self.key, mesh=self.mesh
        )
        num_params = utils.calculate_num_params_from_pytree(nnx.state(self.model))
        logging.info(f"Number of model params={num_params}")

        @jax.jit
        def predict_fn(x):
            return self.model(x) * config.normalization

        self.predict_fn = lambda phase_feat, other_feat: inference_subbatch(
            module=lambda x: predict_fn(jax.device_put(x, feature_sharding)),
            subbatch_size=config.subcollocation_size,
            batched_args=phase_feat,
            nonbatched_args=other_feat,
            low_memory=True,
            input_subbatch_dim=1,
        )

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
        phase_feat, other_feat = features.split_feature(feat)
        predictions = self.predict_fn(phase_feat, other_feat)
        jax.tree.map(lambda x: x.block_until_ready(), predictions)
        logging.info(
            "Output shape was %s",
            jax.tree.map(lambda x: x.shape, predictions),
        )
        return predictions
