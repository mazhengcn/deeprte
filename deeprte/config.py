import functools

import ml_collections
from jaxline import base_config

from deeprte.model.config import CONFIG_MODEL

CONFIG_DATASET = ml_collections.ConfigDict(
    {
        "data_path": "",
        "num_samples": 10,
        "train": {
            "batch_size": 2,
            "collocation_sizes": 150,
            "repeat": 1,
        },
        "validation": {
            "batch_size": 2,
        },
        "data_split": {
            "num_test_samples": 2,
            "save_path": None,
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
        "num_epochs": 5000,
        "optimizer": {
            "base_lr": 1e-3,
            "scale_by_batch": True,
            "schedule_type": "exponential",
            "decay_kwargs": {
                "transition_steps": 200,
                "decay_rate": 0.96,
            },
            "optimizer": "adam",
            "adam_kwargs": {},
        },
    }
)

CONFIG_GLOBAL = ml_collections.ConfigDict(
    {
        # N means N epochs
        "save_checkpoint_interval": 40,
        "log_tensors_interval": 1,
        "log_train_data_interval": 2,
        # When True, the eval job immediately loads a checkpoint runs evaluate()
        # once, then terminates.
        "one_off_evaluate": False,
        # Seed for the RNGs (default is 42).
        "random_seed": 42,
        "checkpoint_dir": "",
        "restore_dir": "",
    }
)


def get_steps_from_epochs(num_epochs, batch_size, n_train_examples, repeat=1):
    """Get global steps from given epoch."""
    # print(n_train_examples.type)
    return max(int(repeat * num_epochs * n_train_examples // batch_size), 1)


def get_config() -> ml_collections.ConfigDict:
    jl_config = base_config.get_base_config()

    make_split_num(CONFIG_DATASET)

    steps_from_epochs = functools.partial(
        get_steps_from_epochs,
        n_train_examples=CONFIG_DATASET.data_split.num_train_samples,
        batch_size=CONFIG_DATASET.train.batch_size,
        repeat=CONFIG_DATASET.train.repeat,
    )

    if "transition_steps" in CONFIG_TRAINING.optimizer.decay_kwargs:
        num = CONFIG_TRAINING.optimizer.decay_kwargs.transition_steps
        CONFIG_TRAINING.optimizer.decay_kwargs.transition_steps = (
            steps_from_epochs(num)
        )

    CONFIG_GLOBAL.save_checkpoint_interval = steps_from_epochs(
        CONFIG_GLOBAL.save_checkpoint_interval
    )
    CONFIG_GLOBAL.log_tensors_interval = steps_from_epochs(
        CONFIG_GLOBAL.log_tensors_interval
    )
    CONFIG_GLOBAL.log_train_data_interval = steps_from_epochs(
        CONFIG_GLOBAL.log_train_data_interval
    )

    expr_config = ml_collections.config_dict.ConfigDict(
        dict(
            config=dict(
                dataset=CONFIG_DATASET,
                training=CONFIG_TRAINING,
                model=CONFIG_MODEL,
            )
        )
    )

    jl_config.experiment_kwargs = expr_config

    jl_config.update(**CONFIG_GLOBAL)

    return jl_config


def make_split_num(
    config: ml_collections.config_dict,
):
    split_config = config.data_split
    num_test = split_config.num_test_samples
    num_train_and_val = config.num_samples - num_test
    num_train = int(
        num_train_and_val * split_config.train_validation_split_rate
    )
    num_val = num_train_and_val - num_train

    split_config["num_train_samples"] = num_train
    split_config["num_val_samples"] = num_val
