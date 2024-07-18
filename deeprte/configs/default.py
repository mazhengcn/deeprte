"""Default Hyperparameter configuration."""

from __future__ import annotations

import dataclasses

from flax import nnx


@dataclasses.dataclass(unsafe_hash=True)
class MeshRules:
    embed: str | None = None
    mlp: str | None = None
    kv: str | None = None
    vocab: str | None = None

    def __call__(self, *keys: str) -> tuple[str, ...]:
        return tuple(getattr(self, key) for key in keys)


@dataclasses.dataclass(unsafe_hash=True)
class Config:
    # Integer for PRNG random seed.
    seed: int = 42

    # Dataset config
    dataset_type: str = "tfds"
    dataset_name: str = "rte"
    data_dir: str = "/workspaces/deeprte/data/tfds"
    train_split: str = "train[:80%]"
    eval_split: str = "train[80%:]"
    enable_data_shuffling: bool = True
    data_shuffle_seed: int = 42
    prefetch_to_device: bool = True

    # Physical dimensions
    position_coords_dim: int = 2
    velocity_coords_dim: int = 2
    # Attention / Optical depths
    coeffs_fn_dim: int = 2
    num_heads: int = 2
    qkv_dim: int = 64
    optical_depth_dim: int = 2
    # Mlp
    num_mlp_layers: int = 4
    mlp_dim: int = 128
    # Scattering
    num_scattering_layers: int = 2
    scattering_dim: int = 16
    kernel_init: nnx.Initializer = nnx.initializers.glorot_uniform()
    bias_init: nnx.Initializer = nnx.initializers.zeros_init()
    # Subcollocation size
    subcollocation_size: int = 128

    # Training config
    num_train_steps: int = 500_000
    # Number of steps to take during training.
    global_batch_size: int = 8
    # Train
    global_batch_size_to_load: int = global_batch_size
    # Train
    global_batch_size_to_train_on: int = global_batch_size_to_load
    # Train
    collocation_sizes: tuple[int] = (140,)
    # Train
    repeat_batch: int = 1
    # expansion_factor_real_data
    expansion_factor_real_data: int = -1
    # Evaluation
    eval_batch_size: int = 8
    # Frequency of eval during training, e.g. every 1_000 steps.
    eval_every_steps: int = 1_000
    # Number of steps to take during evaluation.
    micro_steps: int = -1
    # Base learning rate.
    learning_rate: float = 0.001
    lr_schedule: str = "exponential_decay"
    decay_rate: float = 0.96
    transition_steps: int = 10_000
    # Linear learning rate warmup.
    warmup_steps: int = 1000
    # Decay factor for AdamW style weight decay.
    weight_decay: float = 0.1

    # Prompt for language model sampling,
    # taken from MaxText (https://github.com/google/maxtext/blob/main/MaxText/configs/base.yml).
    # Parallelism
    mesh_axes: tuple[str, ...] = ("data", "fsdp", "tensor")
    axis_rules: MeshRules = MeshRules(
        embed="fsdp",
        mlp="tensor",
        kv="tensor",
        vocab="tensor",
    )
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
    # Whether restoring checkpoitn with SingleReplicaArrayHandler
    enable_single_replica_ckpt_restoring: bool = False
    # Whether to enable emergency checkpointing.
    enable_emergency_checkpoint: bool = False
    local_checkpoint_dir: str = ""
    local_checkpoint_every_steps: int = 10_000

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


def get_config():
    """Get the default hyperparameter configuration."""
    config = Config()
    return config
