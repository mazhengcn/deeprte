import dataclasses
import json
import pathlib

import jax
import optax
from absl import app, flags, logging
from etils import epath
from flax import nnx
from jax.sharding import Mesh

from deeprte.configs import default
from deeprte.model.model import DeepRTE, DeepRTEConfig
from deeprte.train_lib import checkpointing, optimizers
from deeprte.train_lib import utils as train_utils

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "config", None, "File path to the training hyperparameter configuration."
)
flags.DEFINE_string("checkpoint_dir", None, "Directory to store model params.")
flags.mark_flags_as_required(["config", "checkpoint_dir"])


def _read_train_checkpoint(config, mesh):
    """Read training checkpoint at path defined by load_full_state_path."""
    # Model and Optimizer definition
    rng = jax.random.key(0)
    lr_schedule = optimizers.create_learning_rate_schedule(config)
    tx = optimizers.create_optimizer(config, lr_schedule)
    tx = optax.MultiSteps(tx, every_k_schedule=config.micro_steps)

    model, optimizer, _ = train_utils.setup_training_state(
        model_class=DeepRTE,
        config=config,
        rng=rng,
        tx=tx,
        mesh=mesh,
        data_iterator=None,
        checkpoint_manager=None,
    )
    num_params = train_utils.calculate_num_params_from_pytree(nnx.state(model))
    logging.info(f"In input checkpoint Number of model params={num_params}.")
    return nnx.state(model), nnx.state(optimizer)


def generate_infer_checkpoint(config, checkpoint_dir):
    """Generate an inference checkpoint and params checkpoint from a given training checkpoint.
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

    # Read training state from config.load_full_state_path
    logging.info(f"Read training checkpoint from: {config.load_full_state_path}")
    model_state, train_state = _read_train_checkpoint(config, mesh)
    assert train_state.opt_state != {}, "missing opt_state in training checkpoint"

    # Save params to checkpoint directory under params folder
    logging.info(f"Save infer checkpoint at: {checkpoint_dir}")
    checkpointing.save_params_to_path(f"{checkpoint_dir}/params", model_state)
    logging.info(
        f"Successfully generated params checkpoint at: {checkpoint_dir}/params"
    )

    # Save config file to checkpoint directory
    model_config = DeepRTEConfig(
        position_coords_dim=config.position_coords_dim,
        velocity_coords_dim=config.velocity_coords_dim,
        coeffs_fn_dim=config.coeffs_fn_dim,
        num_heads=config.num_heads,
        qkv_dim=config.qkv_dim,
        optical_depth_dim=config.optical_depth_dim,
        num_mlp_layers=config.num_mlp_layers,
        mlp_dim=config.mlp_dim,
        num_scattering_layers=config.num_scattering_layers,
        scattering_dim=config.scattering_dim,
        subcollocation_size=config.subcollocation_size,
        normalization=config.normalization,
        load_parameters_path=f"{checkpoint_dir}/params",
    )
    config_dict = dataclasses.asdict(model_config)
    config_dict["load_full_state_path"] = config.load_full_state_path
    with pathlib.Path(f"{checkpoint_dir}/config.json").open("w") as f:
        json.dump(config_dict, f, indent=2)
    logging.info(
        f"Successfully save model config file at: {checkpoint_dir}/config.json"
    )

    return True


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    config = default.get_config(FLAGS.config)
    generate_infer_checkpoint(config, FLAGS.checkpoint_dir)


if __name__ == "__main__":
    app.run(main)
