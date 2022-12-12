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

import ml_collections

CONFIG = ml_collections.ConfigDict(
    {
        "global_config": {
            "deterministic": True,
            "subcollocation_size": 128,
            "w_init": "glorot_uniform",
        },
        "green_function": {
            "scattering": {
                "num_layer": 2,
                "latent_dim": 64,
            },
            "attenuation": {
                "num_layer": 3,
                "latent_dim": 128,
                "attention": {
                    "num_head": 2,
                    "key_dim": 16,
                    "value_dim": None,
                    "output_dim": None,
                },
            },
        },
    }
)
