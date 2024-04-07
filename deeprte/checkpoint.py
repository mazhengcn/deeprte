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


import datetime
import os
import pathlib
import signal
import threading

import dill
import numpy as np
from absl import logging
from jaxline import utils as jl_utils

from deeprte.utils import to_flat_dict


def _get_step_date_label(global_step):
    # Date removing microseconds.
    date_str = datetime.datetime.now().isoformat().split(".")[0]
    return f"step_{global_step}_{date_str}"


def restore_state_to_in_memory_checkpointer(restore_path, config):
    """Initializes experiment state from a checkpoint."""
    if not isinstance(restore_path, pathlib.Path):
        restore_path = pathlib.Path(restore_path)

    # Load pretrained experiment state.
    python_state_path = restore_path / "checkpoint.pickle"
    with open(python_state_path, "rb") as f:
        pickle_nest = dill.load(f)
    logging.info("Restored checkpoint from %s", python_state_path)

    snapshot = jl_utils.SnapshotNT(0, pickle_nest)

    # Finally, seed the jaxline `utils.InMemoryCheckpointer` global dict.
    jl_utils.GLOBAL_CHECKPOINT_DICT["latest"] = jl_utils.CheckpointNT(
        threading.local(), [snapshot]
    )


def save_state_from_in_memory_checkpointer(save_path, config):
    """Saves experiment state to a checkpoint."""
    if not isinstance(save_path, pathlib.Path):
        save_path = pathlib.Path(save_path)

    # Serialize config as json
    logging.info("Saving config.")
    config_path = save_path.parent / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config.to_json_best_effort(indent=4))

    logging.info("Saving model.")
    for checkpoint_name, checkpoint in jl_utils.GLOBAL_CHECKPOINT_DICT.items():
        if not checkpoint.history:
            logging.info('Nothing to save in "%s"', checkpoint_name)
            continue

        pickle_nest = checkpoint.history[-1].pickle_nest
        global_step = pickle_nest["global_step"]

        # Saving directory
        save_dir = save_path / checkpoint_name / _get_step_date_label(global_step)

        # Save params and states in a dill file
        python_state_path = save_dir / "checkpoint.pickle"
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(python_state_path, "wb") as f:
            dill.dump(pickle_nest, f)

        # Save flat params separately
        numpy_params_path = save_dir / "params.npz"
        flat_np_params = to_flat_dict(
            pickle_nest["experiment_module"]["params"].to_dict()
        )
        np.savez(numpy_params_path, **flat_np_params)

        # Save model config under the same directory of params
        model_config_path = save_dir / "model.json"
        model_config = config.experiment_kwargs.config.model
        with open(model_config_path, "w", encoding="utf-8") as f:
            f.write(model_config.to_json_best_effort(indent=4))

        logging.info(
            'Saved "%s" checkpoint and flat numpy params under %s',
            checkpoint_name,
            save_dir,
        )


def setup_signals(save_model_fn):
    """Sets up a signal for model saving."""

    # Save a model on Ctrl+C.
    def sigint_handler(unused_sig, unused_frame):
        # Ideally, rather than saving immediately, we would then "wait" for
        # a good time to save. In practice this reads from an in-memory
        # checkpoint that only saves every 30 seconds or so, so chances of
        # race conditions are very small.
        save_model_fn()
        logging.info(r"Use `Ctrl+\` to save and exit.")

    # Exit on `Ctrl+\`, saving a model.
    prev_sigquit_handler = signal.getsignal(signal.SIGQUIT)

    def sigquit_handler(unused_sig, unused_frame):
        # Restore previous handler early, just in case something goes wrong
        # in the next lines, so it is possible to press again and exit.
        signal.signal(signal.SIGQUIT, prev_sigquit_handler)
        save_model_fn()
        logging.info(r"Exiting on `Ctrl+\`")

        # Re-raise for clean exit.
        os.kill(os.getpid(), signal.SIGQUIT)

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGQUIT, sigquit_handler)
