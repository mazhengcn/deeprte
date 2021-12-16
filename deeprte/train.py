import functools
import os

from absl import flags
from jaxline import platform

from deeprte.checkpoint import (
    restore_state_to_in_memory_checkpointer,
    save_state_from_in_memory_checkpointer,
    setup_signals,
)

FLAGS = flags.FLAGS


def main(experiment_class, argv):

    # Maybe restore a model.
    restore_path = FLAGS.config.restore_path
    if restore_path:
        restore_state_to_in_memory_checkpointer(restore_path)

    # Maybe save a model.
    save_dir = os.path.join(FLAGS.config.checkpoint_dir, "models")
    if FLAGS.config.one_off_evaluate:
        save_model_fn = (
            lambda: None
        )  # No need to save checkpoint in this case.
    else:
        save_model_fn = functools.partial(
            save_state_from_in_memory_checkpointer, save_dir, experiment_class
        )
    setup_signals(save_model_fn)  # Save on Ctrl+C (continue) or Ctrl+\ (exit).

    try:
        platform.main(experiment_class, argv)
    finally:
        save_model_fn()  # Save at the end of training or in case of exception.
