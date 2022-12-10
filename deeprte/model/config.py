"""model config"""
import ml_collections

CONFIG_MODEL = ml_collections.ConfigDict(
    {
        "global_config": {
            "deterministic": True,
            "sub_collocation_size": 128,
            "latent_dims": 64,
        },
        "model_structure": {
            "green_function": {
                "scattering_module": {
                    "res_block_depth": 1,
                },
                "attenuation_module": {
                    "attenuation_block": {
                        "widths": 128,
                        "num_layer": 3,
                    },
                    "attention": {
                        "widths": 64,
                        "num_layer": 1,
                    },
                },
            },
        },
    }
)
