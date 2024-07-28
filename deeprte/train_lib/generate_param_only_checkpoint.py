# pylint: disable=g-bad-todo, abstract-method, consider-using-with, ungrouped-imports
"""Transforms a "full state" including optimizer state to a bfloat16 "parameter state" without optimizer state.
This typically used for turning a state output by training.py into a state than can be consumed by decode.py.

The input "fullstate" is passed in via:
  load_full_state_path.
The output "parameter state" is output to the checkpoint directory. Additionally it is cast down to bf16.
"""

import checkpointing
import jax
from absl import app, flags, logging
from etils import epath
from jax.sharding import Mesh
from ml_collections import config_flags
from train import save_checkpoint

from deeprte.model.modules import constructor
from deeprte.train_lib import optimizers
from deeprte.train_lib import utils as train_utils

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_dir", None, "Directory to store model params.")
config_flags.DEFINE_config_file(
    "config",
    "configs/default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)
flags.mark_flags_as_required(["checkpoint_dir"])


def _read_train_checkpoint(config, checkpoint_manager, mesh):
    """Read training checkpoint at path defined by load_full_state_path."""
    # Model and Optimizer definition
    key = jax.random.key(0)
    learning_rate_dict = {
        "schedule": config.schedule,
        "init_value": config.learning_rate,
        "decay_rate": config.decay_rate,
        "transition_steps": config.transition_steps,
    }
    _, tx = optimizers.create_optimizer(
        name=config.optimizer,
        total_steps=config.num_train_steps,
        learning_rate=learning_rate_dict,
    )
    state, state_sharding, _ = train_utils.setup_training_state(
        constructor, None, tx, config, key, mesh, checkpoint_manager
    )
    num_params = train_utils.calculate_num_params_from_pytree(state.params)
    logging.info(f"In input checkpoint Number of model params={num_params}.")
    return state, state_sharding


def _save_infer_checkpoint(state, checkpoint_manager):
    """Generate checkpoint for inference from the training_state."""
    with jax.spmd_mode("allow_all"):
        infer_state = train_utils.init_infer_state(None, state.params)
    if checkpoint_manager is not None:
        if save_checkpoint(checkpoint_manager, 0, infer_state):
            logging.info("Saved an inference checkpoint at.")
    checkpoint_manager.wait_until_finished()


def generate_infer_checkpoint(config, checkpoint_dir):
    """
    Generate an inference checkpoint and params checkpoint from a given training checkpoint.
    - Training checkpoint is loaded from config.load_full_state_path.
    - Inference checkpoint will be saved at the checkpoint directory under 0/train_state folder.
    - Params checkpoint will be saved at the checkpoint driectory under params folder.
    """

    devices_array = train_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    assert checkpoint_dir, "checkpoint_dir not configured"
    # Remove any old checkpoint
    path = epath.Path(checkpoint_dir)
    if path.exists():
        if jax.process_index() == 0:
            path.rmtree()

    # Create a checkpoint manager to save infer checkpoint at config.checkpoint_dir
    checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
        checkpoint_dir,
        config.save_checkpoints,
        config.async_checkpointing,
        config.checkpoint_every_steps,
    )
    # Read training state from config.load_paramaters_path
    logging.info(f"Read training checkpoint from: {config.load_full_state_path}")
    training_state, _ = _read_train_checkpoint(config, checkpoint_manager, mesh)
    assert training_state.opt_state != {}, "missing opt_state in training checkpoint"

    # Save infer state to checkpoint directory at step 0
    logging.info(f"Save decode checkpoint at: {checkpoint_dir}")
    _save_infer_checkpoint(training_state, checkpoint_manager)
    logging.info(
        f"Successfully generated inference checkpoint at: {checkpoint_dir}/0/train_state"
    )
    # Save params to checkpoint directory under params folder
    checkpointing.save_params_to_path(f"{checkpoint_dir}/params", training_state.params)
    logging.info(
        f"Successfully generated params checkpoint at: {checkpoint_dir}/params"
    )

    return True


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    generate_infer_checkpoint(FLAGS.config, FLAGS.checkpoint_dir)


if __name__ == "__main__":
    app.run(main)
