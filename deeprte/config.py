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

import ml_collections
import tensorflow_datasets as tfds
from jaxline import base_config

from deeprte.model.config import model_config


def get_config(arg_string: str = "8, 5000"):
    args = arg_string.split(",")
    if len(args) != 2:
        raise ValueError(
            "You must provide exactly two arguments separated by a "
            "comma - train_batch_size,num_epochs"
        )
    train_batch_size, num_epochs = args
    train_batch_size = int(train_batch_size)
    num_epochs = int(num_epochs)

    config = base_config.get_base_config()

    dataset_config = ml_collections.ConfigDict(
        dict(name="rte", tfds_dir="data/tfds", split_percentage="80%")
    )
    dataset_builder = tfds.builder(
        dataset_config.name, data_dir=dataset_config.tfds_dir
    )
    dataset_config.num_train_examples = dataset_builder.info.splits[
        f"train[:{dataset_config.split_percentage}]"
    ].num_examples

    model = model_config()
    model.data.normalization_dict = dataset_builder.info.metadata["normalization"]

    config.experiment_kwargs = ml_collections.ConfigDict(
        dict(
            config=dict(
                dataset=dataset_config,
                training=dict(
                    num_epochs=num_epochs,
                    batch_size=train_batch_size,
                    collocation_sizes=[140],
                    batch_repeat=1,
                    accum_grads_steps=1,
                ),
                optimizer=dict(
                    base_lr=1e-3,
                    scale_by_batch=False,
                    schedule_type="exponential",
                    decay_kwargs=dict(transition_steps=10000, decay_rate=0.96),
                    optimizer="adam",
                    adam_kwargs=dict(),
                ),
                evaluation=dict(batch_size=4),
                model=model,
            )
        )
    )

    def steps_from_epochs(num_epochs):
        return max(
            int(
                config.experiment_kwargs.config.training.batch_repeat
                * num_epochs
                * dataset_config.num_train_examples
                // train_batch_size
            ),
            1,
        )

    config.training_steps = steps_from_epochs(num_epochs)

    config.interval_type = "steps"
    config.legacy_random_seed_behavior = True
    config.save_checkpoint_interval = steps_from_epochs(10)
    config.log_tensors_interval = steps_from_epochs(1)
    config.log_train_data_interval = steps_from_epochs(1)
    # When True, the eval job immediately loads a checkpoint
    # runs evaluate() once, then terminates.
    config.one_off_evaluate = False
    config.max_checkpoints_to_keep = 2
    # Seed for the RNGs (default is 42).
    config.random_seed = 42
    config.best_model_eval_metric = "eval_rmspe"
    config.best_model_eval_metric_higher_is_better = False
    config.checkpoint_dir = ""
    config.restore_dir = ""

    return config
