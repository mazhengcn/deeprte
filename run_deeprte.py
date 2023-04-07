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

"""Train."""

import functools
import os
import pathlib

from absl import app, flags, logging
from jaxline import platform

from deeprte.checkpoint import (
    restore_state_to_in_memory_checkpointer,
    save_state_from_in_memory_checkpointer,
    setup_signals,
)
from deeprte.train import Trainer

FLAGS = flags.FLAGS

flags.DEFINE_string("tfds_dir", None, "source dir path")


def main(experiment_class, argv):
    write_data_path(FLAGS.config.experiment_kwargs.config.dataset)

    # Maybe restore a model.
    restore_dir = FLAGS.config.restore_dir

    if restore_dir:
        restore_state_to_in_memory_checkpointer(restore_dir)

    # Maybe save a model.
    save_dir = os.path.join(FLAGS.config.checkpoint_dir, "models")

    if FLAGS.config.one_off_evaluate:
        save_model_fn = (
            lambda: None
        )  # noqa: E731  # No need to save checkpoint in this case.
    else:
        save_model_fn = functools.partial(
            save_state_from_in_memory_checkpointer, save_dir, experiment_class
        )
    setup_signals(save_model_fn)  # Save on Ctrl+C (continue) or Ctrl+\ (exit).

    if FLAGS.jaxline_mode.startswith("train"):
        if not pathlib.Path(FLAGS.config.checkpoint_dir).exists():
            pathlib.Path(FLAGS.config.checkpoint_dir).mkdir()
        logging.get_absl_handler().use_absl_log_file(
            "train", FLAGS.config.checkpoint_dir
        )

    try:
        platform.main(experiment_class, argv)
    finally:
        save_model_fn()  # Save at the end of training or in case of exception.


def write_data_path(config):
    # config.data_name_list = FLAGS.data_name_list
    config.tfds_dir = FLAGS.tfds_dir
    # config.data_split.save_path = FLAGS.save_path


if __name__ == "__main__":
    flags.mark_flag_as_required("config")
    app.run(functools.partial(main, Trainer))
