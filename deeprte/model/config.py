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


def model_config(name: str = "supervised") -> ml_collections.ConfigDict:
    """Get the ConfigDict of DeepRTE model."""

    if name not in CONFIG_DIFFS:
        raise ValueError(f"Invalid model name {name}.")

    cfg = copy.deepcopy(CONFIG)
    cfg.update_from_flattened_dict(CONFIG_DIFFS[name])

    return cfg


CONFIG_DIFFS = {"supervised": {"model": {}}, "unsupervised": {"model": {}}}

CONFIG = ml_collections.ConfigDict(
    {
        "data": {},
        "rte_operator": {
            "green_function": {
                "green_function_mlp": {"widths": [128, 128, 128, 128, 1]},
                "coefficient_net": {
                    "attention_net": {"widths": [64, 1]},
                    "pointwise_mlp": {"widths": [64, 2]},
                },
            },
            "activation": "gelu",
        },
        "model": {},
    }
)
