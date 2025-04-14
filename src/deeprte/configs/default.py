"""Default Hyperparameter configuration."""

import dataclasses
import json
import pathlib

import jax
import yaml


@dataclasses.dataclass(unsafe_hash=True)
class Config:
    # Integer for PRNG random seed.
    seed: int = 42
    # Dataset type.
    dataset_type: str = "grain"
    # Number of child processes launched to parallelize the transformations among.
    # Zero means processing runs in the same process. None lets the python backend choose the value.
    grain_worker_count: int | None = 4
    # Count of output batches to produce in advance per worker.
    # This ensures batches are ready when the consumer requests them.
    grain_worker_buffer_size: int | None = 2
    # Name of TFDS dataset to use.
    dataset_name: str = "g0.5-sigma_a3-sigma_t6.mat"
    # Path to directory where TFDS data is stored.
    data_dir: str = "/workspaces/deeprte/data/raw_data/train/g0.5-sigma_a3-sigma_t6"
    # TFDS split for training dataset.
    train_split: str = "train[:80%]"
    # TFDS split for evaluation dataset.
    eval_split: str = "train[80%:]"
    # Whether to enable data shuffling.
    enable_data_shuffling: bool = True
    # Seed for data shuffling.
    data_shuffle_seed: int = 42
    # per_device_batch_size for training.
    per_device_batch_size: int = 1
    # Global batch size for training.
    global_batch_size: int = 8
    # Number of collocation points to sample from phase space for training.
    collocation_size: int | None = 128
    # Number of same batch with different collocation points (in order to
    # increase collocation sizes for training).
    repeat_batch: int = 1
    # Global batch size for evaluation.
    eval_batch_size: int = 4
    # Number of steps to train for.
    num_train_steps: int = 1_000_001

    train_state_path: str = None
    # Number of micro steps for grads accumulation, None for no accumulation.
    micro_steps: int = int(
        global_batch_size / per_device_batch_size / jax.device_count()
    )
    # Frequency of logging metrics during training, e.g. every 1_000 steps.
    log_every_steps: int = 1_000
    # Frequency of eval during training, e.g. every 1_000 steps.
    eval_every_steps: int = 50_000
    # Optimizer
    optimizer: str = "adam"
    # Initial learning rate.
    learning_rate: float = 0.001
    # Learning rate schedule.
    schedule: str = "cosine_decay"
    # Decay steps for cosine decay scheduler.
    decay_steps: int = 10_000
    # Decay steps for exponential decay scheduler.
    transition_steps: int = 10_000
    # Decay rate for exponential decay scheduler.
    decay_rate: float = 0.96
    # Whether to save model checkpoints.
    save_checkpoints: bool = True
    # Save a checkpoint every these number of steps.
    checkpoint_every_steps: int = 10_000
    # Whether to enable async checkpointing.
    async_checkpointing: bool = True
    # Whether to enable standard logger for checkpointing.
    enable_checkpoint_standard_logger: bool = True
    # If there is no checkpoint in the checkpoint manager,
    # load parameters from a parameter only checkpoint at this path.
    load_parameters_path: str = ""
    # If there is no checkpoint in the checkpoint manager,
    # load full state from a full state checkpoint at this path.
    load_full_state_path: str = ""

    # Physical position dimensions.
    position_coords_dim: int = 2
    # Physical velocity dimensions.
    velocity_coords_dim: int = 2
    # Dimensions of (scattering) coefficient functions.
    coeffs_fn_dim: int = 2
    # Number of attention heads.
    num_heads: int = 2
    # Attention dimension.
    qkv_dim: int = 64
    # Output dimensions of attention.
    optical_depth_dim: int = 17
    # Number of MLP layers.
    num_mlp_layers: int = 4
    # MLP dimension.
    mlp_dim: int = 128
    # Number of scattering layers.
    num_scattering_layers: int = 2
    # Scattering dimension.
    scattering_dim: int = 16
    # Subcollocation size for evaluation or inference
    subcollocation_size: int = 128
    # Normalization constant of dataset.
    normalization: float = 1.0

    position_coords_dim: int = 2
    velocity_coords_dim: int = 2
    coeffs_fn_dim: int = 2
    num_basis_functions: int = 64
    basis_function_encoder_dim: int = 128
    num_basis_function_encoder_layers: int = 4
    green_function_encoder_dim: int = 128
    num_green_function_encoder_layers: int = 4
    num_scattering_layers: int = 2
    scattering_dim: int = 128
    num_heads: int = 8
    qkv_dim: int = 16
    optical_depth_dim: int = 16
    name: str = "boundary"

    # Parallelism
    mesh_axes: tuple[str, ...] = ("data", "fsdp", "tensor")
    data_sharding: tuple[str, ...] = (("data", "fsdp", "tensor"),)
    # One axis for each parallelism type may hold a placeholder (-1)
    # value to auto-shard based on available slices and devices.
    # By default, product of the DCN axes should equal number of slices
    # and product of the ICI axes should equal number of devices per slice.
    # ICI (Inter-Chip Interconnection): A high-speed connection between
    # sets of TPU chips, which form the TPU network.
    # DCN (Data Center Network): A connection between the TPU networks;
    # not as fast as ICI.
    # ICI has around 100x the bandwidth of DCN, but it is not a general
    # purpose connection, which is why DCN is necessary for scaling to
    # extremely large ML models.
    dcn_data_parallelism: int = -1
    dcn_fsdp_parallelism: int = 1
    dcn_tensor_parallelism: int = 1
    ici_data_parallelism: int = 1
    ici_fsdp_parallelism: int = -1
    ici_tensor_parallelism: int = 1

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


def get_config(cfg_path: str | None = None) -> Config:
    """Get the default hyperparameter configuration."""
    config = Config()
    if cfg_path:
        suffix = pathlib.Path(cfg_path).suffix
        if suffix in [".yaml", ".yml"]:
            file_loader = yaml.full_load
        elif suffix == ".json":
            file_loader = json.load
        else:
            raise ValueError(f"Unsupported configuration file format: {suffix}")

        with open(cfg_path, "r") as f:
            cfg = file_loader(f)

        return config.replace(**cfg)

    return config
