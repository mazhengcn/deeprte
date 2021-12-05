import functools
import pathlib

import ml_collections
from jaxline import base_config

from deeprte import dataset

N_TRAIN_EXAMPLES = dataset.Split.TRAIN_AND_VALID.num_examples


def get_steps_from_epochs(batch_size, num_epochs):
    return max(int(num_epochs * N_TRAIN_EXAMPLES // batch_size), 1)


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


def get_config() -> ml_collections.ConfigDict:
    """Return config object for solver."""
    config = base_config.get_base_config()

    # Batch size, training steps and data.
    num_epochs = 1000
    train_batch_size = 10

    steps_from_epochs = functools.partial(
        get_steps_from_epochs, train_batch_size
    )
    # Steps and test batch size.
    num_steps = steps_from_epochs(num_epochs)
    test_batch_size = train_batch_size

    # Solution config
    config.solution_kwargs = ml_collections.ConfigDict(
        dict(config=CONFIG.rte_operator)
    )

    # Model config
    config.model_kwargs = ml_collections.ConfigDict(dict(config=CONFIG.model))

    # Solver config
    config.experiment_kwargs = ml_collections.ConfigDict(
        dict(
            config=dict(
                save_final_checkpoint_as_npy=False,
                optimizer=dict(
                    base_lr=1e-3,
                    scale_by_batch=False,
                    schedule_type="constant",
                    exponential_decay_kwargs=dict(
                        transition_steps=100, decay_rate=0.96
                    ),
                    optimizer="adam",
                    adam_kwargs={},
                ),
                training=dict(
                    num_train_examples=N_TRAIN_EXAMPLES,
                    batch_size=train_batch_size,
                    num_epochs=num_epochs,
                ),
                evaluation=dict(
                    batch_size=test_batch_size,
                    # If `interval` is positive, synchronously evaluate at
                    # regular intervals. Setting it to zero will not evaluate
                    # while training, unless `--jaxline_mode` is set to
                    # `train_eval_multithreaded`, which asynchronously
                    # evaluates checkpoints.
                    # interval=steps_from_epochs(20),
                    interval=0,
                ),
            )
        )
    )

    # Global config
    config.training_steps = num_steps
    config.interval_type = "steps"
    config.save_checkpoint_interval = steps_from_epochs(40)
    config.log_tensors_interval = steps_from_epochs(1)
    config.log_train_data_interval = steps_from_epochs(1)

    # Directory config
    config.checkpoint_dir = pathlib.Path("./data/ckpt")

    config.lock()

    return config
