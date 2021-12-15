"""Model config."""

import copy

import ml_collections


def model_config(name: str) -> ml_collections.ConfigDict:
    cfg = copy.deepcopy(CONFIG)
    return cfg


CONFIG = ml_collections.ConfigDict(
    {
        "rte_operator": {
            "green_function": {
                "green_function_mlp": {"widths": [128, 128, 128, 128, 1]},
                "coefficient_net": {
                    "attention_net": {"widths": [64, 1]},
                    "pointwise_mlp": {"widths": [64, 2]},
                },
            }
        },
        "model": {},
    }
)
