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
from deeprte.train import Experiment

FLAGS = flags.FLAGS


def main(experiment_class, argv):
    # Maybe restore a model.
    restore_dir = FLAGS.config.restore_dir

    if restore_dir:
        restore_state_to_in_memory_checkpointer(restore_dir, FLAGS.config)

    # Maybe save a model.
    save_dir = os.path.join(FLAGS.config.checkpoint_dir, "models")

    if FLAGS.config.one_off_evaluate:

        def save_model_fn():
            return None  # noqa: E731  # No need to save checkpoint in this case.

    else:
        save_model_fn = functools.partial(
            save_state_from_in_memory_checkpointer,
            save_dir,
            experiment_class,
            FLAGS.config,
        )
    setup_signals(save_model_fn)  # Save on Ctrl+C (continue) or Ctrl+\ (exit).

    if FLAGS.jaxline_mode.startswith("train"):
        if not pathlib.Path(FLAGS.config.checkpoint_dir).exists():
            pathlib.Path(FLAGS.config.checkpoint_dir).mkdir(parents=True)
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
