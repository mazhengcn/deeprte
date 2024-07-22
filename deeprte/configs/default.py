"""Default Hyperparameter configuration."""

import dataclasses


@dataclasses.dataclass(unsafe_hash=True)
class MeshRules:
    mlp: str | None = None
    kv: str | None = None

    def __call__(self, *keys: str) -> tuple[str, ...]:
        return tuple(getattr(self, key) for key in keys)


@dataclasses.dataclass(unsafe_hash=True)
class Config:
    # Integer for PRNG random seed.
    seed: int = 42
    # Dataset type.
    dataset_type: str = "tfds"
    # Name of TFDS dataset to use.
    dataset_name: str = "rte"
    # Path to directory where TFDS data is stored.
    data_dir: str = "/workspaces/deeprte/data/tfds"
    # TFDS split for training dataset.
    train_split: str = "train[:80%]"
    # TFDS split for evaluation dataset.
    eval_split: str = "train[80%:]"
    # Whether to enable data shuffling.
    enable_data_shuffling: bool = True
    # Seed for data shuffling.
    data_shuffle_seed: int = 42
    # Global batch size for training.
    global_batch_size: int = 8
    # Number of collocation points to sample from phase space for training.
    collocation_sizes: tuple[int] = (140,)
    # Number of same batch with different collocation points (in order to
    # increase collocation sizes for training).
    repeat_batch: int = 1
    # Global batch size for evaluation.
    eval_batch_size: int = 8
    # Number of steps to train for.
    num_train_steps: int = 500_000
    # Number of micro steps for grads accumulation, None for no accumulation.
    microsteps: int | None = None
    # Frequency of logging metrics during training, e.g. every 1_000 steps.
    log_every_steps: int = 1_000
    # Frequency of eval during training, e.g. every 1_000 steps.
    eval_every_steps: int = 1_000
    # Initial learning rate.
    learning_rate: float = 0.001
    # Learning rate schedule.
    schedule: str = "exponential_decay"
    # Decay rate of learning rate scheduler.
    decay_rate: float = 0.96
    # After how many steps to start annealing.
    transition_steps: int = 10_000
    # Linear learning rate warmup.
    warmup_steps: int = 1000
    # Decay factor for AdamW style weight decay.
    weight_decay: float = 0.1
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
    # Local checkpoint directory for emergency checkpointing.
    local_checkpoint_dir: str = ""
    # Local checkpoint every these number of steps.
    local_checkpoint_every_steps: int = 10_000

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
    optical_depth_dim: int = 2
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

    # Parallelism
    mesh_axes: tuple[str, ...] = ("data", "fsdp", "tensor")
    axis_rules: MeshRules = MeshRules(mlp="fsdp", kv="fsdp")
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


def get_config():
    """Get the default hyperparameter configuration."""
    config = Config()
    return config
