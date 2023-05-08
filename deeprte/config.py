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


def get_config():
    config = base_config.get_base_config()

    num_epochs = 5000
    train_batch_size = 16
    batch_repeat = 1
    eval_batch_size = 4

    dataset = ml_collections.ConfigDict(
        dict(name="rte", data_dir="./data/tfds", split_percentage="60%")
    )

    dataset_builder = tfds.builder(dataset.name, data_dir=dataset.data_dir)
    dataset.num_train_examples = dataset_builder.info.splits[
        f"train[:{dataset.split_percentage}]"
    ].num_examples

    model = model_config()
    model.data.normalization_dict = dataset_builder.info.metadata[
        "normalization"
    ]

    def steps_from_epochs(num_epochs):
        return max(
            int(
                batch_repeat
                * num_epochs
                * dataset.num_train_examples
                // train_batch_size
            ),
            1,
        )

    config.training_steps = steps_from_epochs(num_epochs)

    config.experiment_kwargs = ml_collections.ConfigDict(
        dict(
            config=dict(
                dataset=dataset,
                training=dict(
                    num_epochs=num_epochs,
                    batch_size=train_batch_size,
                    collocation_sizes=[135],
                    batch_repeat=batch_repeat,
                    accum_grads_steps=4,
                ),
                optimizer=dict(
                    base_lr=1e-3,
                    scale_by_batch=False,
                    schedule_type="exponential",
                    decay_kwargs=dict(
                        transition_steps=steps_from_epochs(100),
                        decay_rate=0.96,
                    ),
                    optimizer="adam",
                    adam_kwargs=dict(),
                ),
                evaluation=dict(batch_size=eval_batch_size),
                model=model,
            )
        )
    )

    config.interval_type = "steps"
    config.save_checkpoint_interval = steps_from_epochs(10)
    config.log_tensors_interval = steps_from_epochs(1)
    config.log_train_data_interval = steps_from_epochs(1)
    # When True, the eval job immediately loads a checkpoint
    # runs evaluate() once, then terminates.
    config.one_off_evaluate = False
    # Seed for the RNGs (default is 42).
    config.random_seed = 42
    config.checkpoint_dir = ""
    config.restore_dir = ""

    return config
