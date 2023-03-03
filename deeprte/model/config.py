# Copyright 2022 Zheng Ma
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model config."""

import copy

import ml_collections


def model_config() -> ml_collections.ConfigDict:
    cfg = copy.deepcopy(CONFIG)
    cg = cfg.green_function
    cg.attenuation.output_dim = cg.scattering.latent_dim
    return cfg


CONFIG = ml_collections.ConfigDict(
    {
        "data": {"is_normalization": True},
        "global_config": {
            "deterministic": True,
            "subcollocation_size": 128,
            "w_init": "glorot_uniform",
        },
        "green_function": {
            "scattering": {
                "num_layer": 2,
                "latent_dim": 16,
            },
            "attenuation": {
                "num_layer": 4,
                "latent_dim": 128,
                "output_dim": 16,
                "attention": {
                    "num_head": 2,
                    "key_dim": 32,
                    "value_dim": None,
                    "output_dim": 2,
                    "key_chunk_size": 128,
                },
            },
        },
    }
)
