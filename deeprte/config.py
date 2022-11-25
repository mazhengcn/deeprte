import ml_collections
from jaxline import base_config


def make_config() -> ml_collections.ConfigDict:
    config = base_config.get_base_config()

    config.dataset = ml_collections.ConfigDict(
        {
            "train": {
                "batch_size": 6,
                "collocation_sizes": 500,
                "repeat": 1,
            },
            "validation": {
                "batch_size": 2,
            },
            "data_split": {
                "num_samples": 10,
                "num_test_samples": 2,
                "train_validation_split_rate": 0.8,
                "is_split_datasets": False,
            },
            "seed": 42,
            "buffer_size": 5_000,
            "threadpool_size": 48,
            "max_intra_op_parallelism": 1,
            "pre_shuffle": True,
        },
    )

    return config
