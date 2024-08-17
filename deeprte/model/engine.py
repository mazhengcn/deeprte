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

from collections.abc import Mapping
from typing import Any

import jax
import jax.experimental
from absl import logging
from flax import nnx
from jax.sharding import PartitionSpec as P

from deeprte.configs import default
from deeprte.model import features
from deeprte.model.mapping import inference_subbatch
from deeprte.model.modules import constructor as model_constructor
from deeprte.model.tf import rte_features
from deeprte.train_lib import utils


class RteEngine:
    """Container for DeepRTE model."""

    def __init__(self, config: default.Config):
        self.config = config
        self.key = jax.random.key(0)

        # Mesh definition, currently for single process only.
        devices_array = utils.create_device_mesh(config, devices=jax.local_devices())
        self.mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

        replicated_sharding = jax.sharding.NamedSharding(self.mesh, P(None))
        self.data_sharding = jax.sharding.NamedSharding(
            self.mesh, P(None, *config.data_sharding)
        )
        self.feature_sharding = {
            k: self.data_sharding
            if k in rte_features.PHASE_COORDS_FEATURES
            else replicated_sharding
            for k in rte_features.FEATURES
        }

        self.params = self.load_params()

        self.graphdef, _ = nnx.split(
            nnx.eval_shape(lambda: model_constructor(config, self.key))
        )

        def predict_fn(params, features, graphdef):
            return nnx.merge(graphdef, params)(features)

        self.jit_predict_fn = jax.jit(
            predict_fn,
            in_shardings=(self.state_sharding.params, self.feature_sharding),
            out_shardings=self.data_sharding,
            static_argnums=2,
        )

    def load_params(self):
        state, self.state_sharding = utils.setup_infer_state(
            constructor=model_constructor,
            config=self.config,
            rng=self.key,
            mesh=self.mesh,
            checkpoint_manager=None,
        )
        params = state.params
        num_params = utils.calculate_num_params_from_pytree(params)
        logging.info(f"Number of model params={num_params}")
        return params

    def process_features(
        self, raw_features: features.FeatureDict
    ) -> features.FeatureDict:
        """Processes features to prepare for feeding them into the model."""
        return features.np_data_to_features(raw_features)

    def predict(self, feat: features.FeatureDict) -> Mapping[str, Any]:
        """Makes a prediction by inferencing the model on the provided
        features.
        """
        logging.info(
            "Running predict with shape(feat) = %s",
            jax.tree_map(lambda x: x.shape, feat),
        )
        phase_feat, other_feat = features.split_feature(feat)
        result = inference_subbatch(
            module=lambda x: self.jit_predict_fn(self.params, x, self.graphdef)
            * float(self.config.normalization),
            subbatch_size=self.config.subcollocation_size,
            batched_args=phase_feat,
            nonbatched_args=other_feat,
            low_memory=True,
            input_subbatch_dim=1,
        )
        # result = self.jit_predict_fn(self.params, feat)
        jax.tree_map(lambda x: x.block_until_ready(), result)
        logging.info(
            "Output shape was %s",
            jax.tree_map(lambda x: x.shape, result),
        )
        return result
