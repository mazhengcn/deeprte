import ml_collections
from jaxline import base_config


CONFIG_DATASET = ml_collections.ConfigDict(
    {
        "data_path": "/workspaces/deeprte/rte_data/rte_data/matlab/eval-data/test_shape.mat",
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
            "save_path": "/workspaces/deeprte/rte_data/rte_data/matlab/eval-data/test_ds.npz",
            "train_validation_split_rate": 0.8,
            "is_split_datasets": True,
        },
        "pre_shuffle_seed": 42,
        "buffer_size": 5_000,
        "threadpool_size": 48,
        "max_intra_op_parallelism": 1,
        "pre_shuffle": True,
    }
)
CONFIG_TRAINING = ml_collections.ConfigDict(
    {
        "num_epoch": 5000,
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

    make_split_num(config)

    return config


def make_split_num(
    config: ml_collections.config_dict,
):
    split_config = config.dataset.data_split
    num_test = split_config.num_test_samples
    num_train_and_val = config.dataset.num_samples - num_test
    num_train = int(num_train_and_val * split_config.train_validation_split_rate)
    num_val = num_train_and_val - num_train

    split_config["num_train_samples"] = num_train
    split_config["num_val_samples"] = num_val
