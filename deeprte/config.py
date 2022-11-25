import ml_collections
from jaxline import base_config
from ml_collections import config_dict

from deeprte.model.tf import dataset
from deeprte.model import rte
from deeprte.model.config import model_config

N_TRAIN_EXAMPLES = dataset.Split.TRAIN_AND_VALID.num_examples


def get_steps_from_epochs(batch_size, num_epochs, repeat=1):
    """Get global steps from given epoch."""
    return max(int(repeat * num_epochs * N_TRAIN_EXAMPLES // batch_size), 1)


def _get_date_label(prefix):
    data_str = datetime.datetime.now().isoformat().split(".")[0]
    return f"{prefix}_ckpt_{data_str}"


def get_config() -> ml_collections.ConfigDict:
    """Return config object for solver."""
    config = base_config.get_base_config()

    train_batch_size = 8
    eval_batch_size = 40

    num_epochs = 10_000
    repeat = 1
    steps_from_epochs = functools.partial(
        get_steps_from_epochs, train_batch_size, repeat=repeat
    )
    # Steps and test batch size.
    num_steps = steps_from_epochs(num_epochs)

    # Datasetconfig.
    dataset_config = dict(
        data_path=config_dict.placeholder(str),
        buffer_size=5_000,
        threadpool_size=48,
        max_intra_op_parallelism=1,
    )

    model_cfg = model_config()
    # Solution config
    solution_ctor = rte.RTEOp
    solution_config = model_cfg.rte_operator

    # Model config
    model_ctor = rte.RTESupervised

    # Solver config
    config.experiment_kwargs = ml_collections.ConfigDict(
        dict(
            config=dict(
                dataset=dataset_config,
                solution=dict(
                    constructor=solution_ctor,
                    kwargs=dict(config=solution_config),
                ),
                model=dict(constructor=model_ctor, kwargs={"name": "rte"}),
                training=dict(
                    batch_size=train_batch_size,
                    collocation_sizes=500,
                    num_epochs=num_epochs,
                    num_train_examples=N_TRAIN_EXAMPLES,
                    repeat=repeat,
                ),
                optimizer=dict(
                    base_lr=1e-3,
                    scale_by_batch=True,
                    schedule_type="exponential",
                    exponential_decay_kwargs=dict(
                        transition_steps=steps_from_epochs(200),
                        decay_rate=0.96,
                    ),
                    cosine_decay_kwargs=dict(
                        warmup_epochs=None,
                        init_value=None,
                        end_value=None,
                        warmup_steps=None,
                    ),
                    optimizer="adam",
                    adam_kwargs={},
                ),
                evaluation=dict(batch_size=eval_batch_size),
            )
        )
    )

    # Global config
    config.training_steps = num_steps
    config.interval_type = "steps"
    config.save_checkpoint_interval = steps_from_epochs(10)
    config.log_tensors_interval = steps_from_epochs(1)
    config.log_train_data_interval = steps_from_epochs(2)

    # When True, the eval job immediately loads a checkpoint runs evaluate()
    # once, then terminates.
    config.one_off_evaluate = False

    # Seed for the RNGs (default is 42).
    config.random_seed = 25

    # Directory config
    # Should be set in the shell scripts
    config.checkpoint_dir = ""
    config.restore_path = ""

    return config


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
