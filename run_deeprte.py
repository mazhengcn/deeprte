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

from deeprte.checkpoint import (
    restore_state_to_in_memory_checkpointer,
    save_state_from_in_memory_checkpointer,
    setup_signals,
)
from deeprte.experiment import Experiment
from deeprte.jaxline import platform

FLAGS = flags.FLAGS


def main(experiment_class, argv):

    # Maybe restore a model.
    restore_path = FLAGS.config.restore_path
    print(f"{restore_path}")
    print(f"{FLAGS.config.experiment_kwargs.config.training.batch_size}")

    if restore_path:
        restore_state_to_in_memory_checkpointer(restore_path)

    # Maybe save a model.
    save_dir = os.path.join(FLAGS.config.checkpoint_dir, "models")
    print(f"{save_dir}")

    if FLAGS.config.one_off_evaluate:
        save_model_fn = lambda: None  # No need to save checkpoint in this case.
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


if __name__ == "__main__":
    flags.mark_flag_as_required("config")
    app.run(functools.partial(main, Experiment))
