import ml_collections
from jaxline import base_config


CONFIG_DATASET = ml_collections.ConfigDict(
    {
        "num_samples": 10,
        "train": {
            "batch_size": 6,
            "collocation_sizes": 500,
            "repeat": 1,
        },
        "validation": {
            "batch_size": 2,
        },
        "data_split": {
            "num_test_samples": 2,
            "train_validation_split_rate": 0.8,
            "is_split_datasets": True,
        },
        "seed": 42,
        "buffer_size": 5_000,
        "threadpool_size": 48,
        "max_intra_op_parallelism": 1,
        "pre_shuffle": True,
    }
)
CONFIG_TRAINING = ml_collections.ConfigDict(
    {
        "num_epochsh": 5000,
        "optimizer": {
            "base_lr": 1e-3,
            "scale_by_batch": True,
            "schedule_type": "exponential",
            "exponential_decay_kwargs": {
                "transition_steps": 200,
                "decay_rate": 0.96,
            },
        },
    }
)


def get_steps_from_epochs(batch_size, num_epochs, n_train_examples, repeat=1):
    """Get global steps from given epoch."""
    return max(int(repeat * num_epochs * n_train_examples // batch_size), 1)


def make_config() -> ml_collections.ConfigDict:
    config = base_config.get_base_config()
    config.dataset = CONFIG_DATASET
    config.training = CONFIG_TRAINING

    return config
