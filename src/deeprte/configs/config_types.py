"""Pydantic-based configuration system for DeepRTE, organized into modular classes."""

# pylint: disable=too-many-lines

import datetime
import logging
import math
import os
from enum import Enum
from typing import Any, Literal, NewType

import jax
from pydantic.config import ConfigDict
from pydantic.fields import Field
from pydantic.functional_validators import field_validator, model_validator
from pydantic.main import BaseModel
from pydantic.types import NonNegativeFloat, PositiveInt

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Reusable Enums and Type Aliases
# ----------------------------------------------------------------------------

PathStr = str
AxisNames = NewType("AxisNames", str)


class DType(str, Enum):
    """Supported data types for weights and activations."""

    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    FLOAT64 = "float64"


class OptimizerType(str, Enum):
    """Supported optimizer algorithms."""

    ADAMW = "adamw"
    ADAM_PAX = "adam_pax"
    SGD = "sgd"


class ProfilerType(str, Enum):
    """Supported performance profilers."""

    NONE = ""
    XPLANE = "xplane"
    NSYS = "nsys"


# ----------------------------------------------------------------------------
# Pydantic models for configuration
# ----------------------------------------------------------------------------

type ModelName = Literal["effcient", "default"]


class RunInfo(BaseModel):
    """Configuration for the overall run, model identity, and logging."""

    base_config: None | str = Field(
        None,
        description="Base config to inherit from. This is a meta-field and is consumed by the config loading system.",
    )
    run_name: str = Field(
        "",
        description="The name of the run. Checkpoints will be stored under this name.",
    )
    model_name: ModelName = Field(
        "default", description="The name of the model configuration to use."
    )
    override_model_config: bool = Field(
        False, description="If True, allows overriding model parameters via CLI."
    )
    log_config: bool = Field(
        True,
        description="If True, prints the final configuration after initialization.",
    )
    base_output_directory: PathStr = Field(
        "", description="Base directory for all outputs, typically a GCS path."
    )


class Checkpointing(BaseModel):
    """Core configuration for checkpointing and run restoration."""

    load_parameters_path: PathStr = Field(
        "", description="Loads only model parameters from a specific checkpoint path."
    )
    load_full_state_path: PathStr = Field(
        "", description="Loads the complete training state from a checkpoint path."
    )
    enable_checkpointing: bool = Field(
        True, description="If True, enables saving checkpoints during training."
    )
    async_checkpointing: bool = Field(
        True, description="If True, uses an asynchronous checkpointer for performance."
    )
    checkpoint_period: int = Field(
        10_000, description="The frequency (in steps) at which to save checkpoints."
    )
    max_num_checkpoints_to_keep: int | None = Field(
        None, description="Maximum number of checkpoints to keep."
    )
    checkpoint_conversion_fn: None | str = Field(
        None, description="Function for processing loaded checkpoint dict."
    )
    source_checkpoint_layout: Literal["orbax", "safetensors"] = Field(
        "orbax", description="The layout of the source checkpoint to load."
    )
    save_checkpoint_on_completion: bool = Field(
        True, description="If True, saves a final checkpoint upon training completion."
    )


class OrbaxStorage(BaseModel):
    """Configuration for Orbax checkpoint storage options."""

    checkpoint_storage_use_ocdbt: bool = Field(
        True, description="Whether to use the OCDbT storage format for checkpoints."
    )
    checkpoint_storage_use_zarr3: bool = Field(
        True, description="Whether to use Zarr3 with OCDbT. Requires use_ocdbt=True."
    )


class DataTypes(BaseModel):
    """Configuration for data types and precision."""

    dtype: DType = Field(DType.FLOAT32, description="The data type for activations.")
    grad_dtype: DType = Field(DType.FLOAT32, description="The data type for gradients.")
    weight_dtype: DType = Field(
        DType.FLOAT32, description="The data type for model weights."
    )


class ModelArchitecture(BaseModel):
    """Core model architecture parameters."""

    position_coords_dim: int = Field(2, description="Physical position dimensions.")
    velocity_coords_dim: int = Field(2, description="Physical velocity dimensions.")
    coeffs_fn_dim: int = Field(
        2, description="Dimensions of (scattering) coefficient functions."
    )
    num_heads: int = Field(2, description="Number of attention heads.")
    qkv_dim: int = Field(64, description="Attention dimension.")
    optical_depth_dim: int = Field(2, description="Output dimensions of attention.")
    num_mlp_layers: int = Field(4, description="Number of MLP layers.")
    mlp_dim: int = Field(128, description="MLP dimension.")
    num_scattering_layers: int = Field(2, description="Number of scattering layers.")
    scattering_dim: int = Field(16, description="Scattering dimension.")
    subcollocation_size: int = Field(
        128, description="Subcollocation size for evaluation or inference."
    )
    normalization: float = Field(1.0, description="Normalization constant of dataset.")


class DatasetGeneral(BaseModel):
    """General configuration for dataset and data loading."""

    dataset_type: str = Field(
        "grain", description="The type of the data loading pipeline."
    )
    per_device_batch_size: int | float = Field(
        1, description="The batch size per device."
    )
    eval_per_device_batch_size: int | float = Field(
        0.0,
        description="The batch size per device for evaluation. Defaults to per_device_batch_size.",
    )
    num_epoch: int = Field(1, description="Number of epochs to train for.")


class GrainDataset(BaseModel):
    """Configuration specific to Grain datasets."""

    grain_file_type: str = Field("arrayrecord", description="File type for Grain data.")
    grain_worker_count: int = Field(
        1, description="Number of workers for Grain data loading."
    )
    grain_per_worker_buffer_size: int = Field(
        1,
        description="Buffer size for each worker for Grain data loading during training.",
    )
    grain_worker_count_eval: int = Field(
        1, description="Number of workers for Grain eval data loading."
    )
    grain_per_worker_buffer_size_eval: int = Field(
        1,
        description="Buffer size for each worker for Grain data loading during evaluation.",
    )
    grain_ram_budget_mb: int = Field(
        1024, description="RAM budget (MB) for auto-tuning worker count."
    )
    grain_num_threads: int = Field(
        16, description="Number of threads for Grain ReadOptions during training."
    )
    grain_prefetch_buffer_size: int = Field(
        500, description="Prefetch buffer size for Grain ReadOptions during training."
    )
    grain_num_threads_eval: int = Field(
        16, description="Number of threads for Grain ReadOptions during evaluation."
    )
    grain_prefetch_buffer_size_eval: int = Field(
        500, description="Prefetch buffer size for Grain ReadOptions during evaluation."
    )
    grain_data_source_max_workers: int = Field(
        16,
        description="Max workers for ThreadPoolExecutor when mixing multiple Grain data sources.",
    )


class TrainingLoop(BaseModel):
    """Configuration for the main training loop, evaluation, and reproducibility."""

    steps: int = Field(
        150_001,
        ge=-1,
        description="Total number of training steps. -1 defaults to learning_rate_schedule_steps.",
    )
    log_period: int = Field(
        100, description="Frequency (in steps) to log metrics and flush Tensorboard."
    )
    eval_interval: int = Field(
        -1,
        description="Run evaluation every N training steps. -1 disables interval-based evaluation.",
    )
    eval_steps: int = Field(
        -1,
        description="Number of steps to run for each evaluation. -1 runs on entire eval split.",
    )
    target_eval_loss: float = Field(
        0.0,
        description="If set, training will stop early when this evaluation loss is reached.",
    )
    enable_data_shuffling: bool = Field(
        True, description="Enables shuffling of the training data."
    )
    data_shuffle_seed: int = Field(0, description="Seed for data shuffling.")
    init_weights_seed: int = Field(
        0, description="Seed for model weight initialization."
    )


class Optimizer(BaseModel):
    """Configuration for the optimizer and learning rate schedule."""

    opt_type: OptimizerType = Field(
        OptimizerType.ADAMW, description="The type of optimizer to use."
    )
    gradient_accumulation_steps: PositiveInt = Field(
        1, description="Number of steps to accumulate gradients before updating."
    )
    gradient_clipping_threshold: NonNegativeFloat = Field(
        1.0, description="The threshold for gradient clipping. 0 disables clipping."
    )
    learning_rate: NonNegativeFloat = Field(
        3.0e-5, description="The peak learning rate."
    )
    cosine_learning_rate_final_fraction: float = Field(
        0.1, description="Final LR as a fraction of peak LR in cosine decay."
    )
    warmup_steps_fraction: float = Field(
        0.1, ge=0.0, le=1.0, description="Fraction of total steps for LR warmup."
    )
    learning_rate_schedule_steps: int = Field(
        -1,
        ge=-1,
        description="Total steps for the LR schedule. -1 defaults to `steps`.",
    )


class AdamW(BaseModel):
    """Configuration specific to the AdamW optimizer."""

    adam_b1: float = Field(
        0.9,
        description="Exponential decay rate for the first moment of past gradients (beta1).",
    )
    adam_b2: float = Field(
        0.95,
        description="Exponential decay rate for the second moment of past gradients (beta2).",
    )
    adam_eps: float = Field(
        1.0e-8,
        description="A small constant for numerical stability (epsilon), applied outside of the square root.",
    )
    adam_eps_root: float = Field(
        0.0,
        description="A small constant for numerical stability (epsilon), applied inside of the square root.",
    )
    adam_weight_decay: float = Field(0.0, description="Weight decay regularization.")
    mu_dtype: str = Field(
        "",
        description="Data type for 'mu' (first moment) in AdamW. Inherits from weight_dtype if empty.",
    )


class DevelopmentAndDebugging(BaseModel):
    """General settings for development and debugging."""

    constant_bound_config: list = Field(
        [], description="Legacy configuration for constant bounds."
    )
    jax_cache_dir: PathStr = Field(
        os.path.join(os.path.expanduser("~"), "jax_cache"),
        description="Directory for JAX compilation cache.",
    )
    jax_debug_log_modules: str = Field(
        "", description="Set to 'jax' for verbose JAX logging."
    )

    @classmethod
    def _clean_empty_string_for_list(cls, v: Any) -> Any:
        """Coerces an empty string from YAML into an empty list before validation."""
        if v == "":
            return []
        elif isinstance(v, str):
            return list(map(float, v.split(",")))
        return v

    # Manually apply the field_validator decorator outside of the class definition to avoid pytype issues
    _validate_config = field_validator("constant_bound_config", mode="before")(
        _clean_empty_string_for_list
    )


class Profiling(BaseModel):
    """Configuration for performance profiling."""

    profiler: ProfilerType = Field(
        ProfilerType.NONE, description="Profiler to use ('xplane', 'nsys')."
    )
    upload_all_profiler_results: bool = Field(
        False, description="Upload profiler results from all hosts."
    )
    skip_first_n_steps_for_profiler: int = Field(
        1, description="Number of initial steps to skip for profiling."
    )
    profiler_steps: int = Field(5, description="Number of steps to profile.")
    profile_cleanly: bool = Field(
        True, description="Add block_until_ready to align profile for each step."
    )
    profile_periodically_period: int = Field(
        -1, description="If positive, profile every N steps."
    )
    hide_profiler_step_metric: bool = Field(
        False, description="Whether to enable profiler step metric."
    )
    enable_jax_profiler: bool = Field(
        False, description="Enable the JAX live profiler."
    )
    jax_profiler_port: int = Field(9999, description="Port for the JAX profiler.")


class Metrics(BaseModel):
    """General configuration for metrics and monitoring."""

    metrics_file: None | PathStr = Field(
        None, description="Local file to store scalar metrics for testing."
    )
    gcs_metrics: bool = Field(False, description="If True, save metrics to GCS.")
    save_config_to_gcs: bool = Field(False, description="If True, save config to GCS.")
    record_internal_nn_metrics: int = Field(
        0, description="Record internal neural network metrics."
    )
    prometheus_port: int = Field(
        0, description="Port for Prometheus metrics server. 0 disables it."
    )
    enable_checkpoint_cloud_logger: bool = Field(
        False, description="Enables structured logging for checkpointing."
    )
    enable_tunix_perf_metrics: bool = Field(
        False,
        description="Whether to enable Tunix-managed metrics measurement. The metrics will be uploaded to tensorboard.",
    )


class Tensorboard(BaseModel):
    """Configuration for Tensorboard logging."""

    enable_tensorboard: bool = Field(True, description="Enable Tensorboard logging.")
    use_vertex_tensorboard: bool = Field(
        False, description="Set to True for GCE, False if running via XPK."
    )
    vertex_tensorboard_project: str = Field(
        "", description="GCP project for Vertex AI Tensorboard."
    )
    vertex_tensorboard_region: str = Field(
        "", description="Region for Vertex AI Tensorboard."
    )


class Debug(BaseModel):
    """Configuration for debugging options."""

    rl: bool = Field(False, description="RL-specific debugging")


class DerivedValues(BaseModel):
    """Holds all fields that are derived from other config values for perfect legacy compatibility."""

    emb_dim: None | int = Field(
        None,
        description="Effective embedding dimension, scaled by `global_parameter_scale`.",
    )
    mlp_dim: None | int = Field(
        None, description="Effective MLP dimension, scaled by `global_parameter_scale`."
    )
    moe_mlp_dim: None | int = Field(
        None,
        description="Effective MLP dimension for MoE layers, scaled by `global_parameter_scale`.",
    )
    num_decoder_layers: None | int = Field(
        None,
        description="Effective number of decoder layers, scaled by `global_parameter_scale`.",
    )
    num_kv_heads: None | int = Field(
        None,
        description="Effective number of key/value heads, scaled by `global_parameter_scale`.",
    )
    num_query_heads: None | int = Field(
        None,
        description="Effective number of query heads, scaled by `global_parameter_scale`.",
    )

    ici_parallelism: None | list[int] = Field(
        None,
        description="Aggregated list of all ICI parallelism values for legacy compatibility.",
    )
    dcn_parallelism: None | list[int] = Field(
        None,
        description="Aggregated list of all DCN parallelism values for legacy compatibility.",
    )

    using_pipeline_parallelism: None | bool = Field(
        None,
        description="Boolean flag indicating if pipeline parallelism is active across ICI or DCN.",
    )
    model_fsdp_ag_once: bool = Field(
        False,
        description="An alias for `pipeline_fsdp_ag_once` for backward compatibility.",
    )

    context_parallel_size: None | int = Field(
        None,
        description="The total size of context parallelism, derived from ICI and DCN values.",
    )

    num_target_devices: None | int = Field(
        None,
        description="The number of devices computed from topology in train_compile or jax.devices() in train",
    )

    global_batch_size_to_train_on: None | int = Field(
        None,
        description="The total batch size for training across all devices. Derived from `per_device_batch_size` and data"
        "parallelism.",
    )
    global_batch_size_to_eval_on: None | int = Field(
        None,
        description="The total batch size for evaluation across all devices. Derived from `eval_per_device_batch_size` and"
        " data parallelism.",
    )
    global_batch_size_to_load: None | int = Field(
        None,
        description="The global batch size for the training dataloader, potentially scaled by `expansion_factor_real_data`.",
    )
    global_batch_size_to_load_eval: None | int = Field(
        None,
        description="The global batch size for the evaluation dataloader, potentially scaled by `expansion_factor_real_data`.",
    )
    micro_batch_size_to_train_on: None | int = Field(
        None,
        description="The size of each micro-batch for training, used in pipeline parallelism. Derived from "
        "`global_batch_size_to_train_on`.",
    )
    micro_batch_size_to_eval_on: None | int = Field(
        None,
        description="The size of each micro-batch for evaluation, used in pipeline parallelism. Derived from "
        "`global_batch_size_to_eval_on`.",
    )

    checkpoint_dir: None | str = Field(
        None,
        description="The full path to the checkpoint directory, derived from `run_name`.",
    )
    metrics_dir: None | str = Field(
        None,
        description="The full path to the metrics directory, derived from `run_name`.",
    )
    tensorboard_dir: None | str = Field(
        None,
        description="The full path to the tensorboard directory, derived from `run_name`.",
    )
    managed_mldiagnostics_dir: None | str = Field(
        None,
        description="The full path to the managed mldiagnostics directory, derived from `run_name`.",
    )

    rampup_end_step: None | int = Field(
        None, description="The step at which the batch size ramp-up phase concludes."
    )
    tensors_on_device: None | list[str] = Field(
        None, description="List of tensors to keep on device memory for custom remat."
    )
    tensors_to_offload: None | list[str] = Field(
        None, description="List of tensors to offload to host memory for custom remat."
    )
    global_batch_size_to_load_start: None | int = Field(
        None, description="Starting global batch size for rampup."
    )
    global_batch_size_to_load_increment: None | int = Field(
        None, description="Increment for global batch size during rampup."
    )
    rampup_samples_per_increment_to_load: None | float = Field(
        None, description="Samples per increment for rampup."
    )


class HardwareAndMesh(BaseModel):
    """Configuration for hardware and parallelism mesh."""

    hardware: Literal["tpu", "gpu", "gpu_multiprocess", "cpu"] = Field(
        "tpu", description="The type of hardware to run on."
    )
    num_slices: int = Field(
        -1, description="Number of TPU slices. Automatically determined."
    )
    mesh_axes: list[str] = Field(
        [
            "data",
            "stage",
            "fsdp",
            "fsdp_transpose",
            "sequence",
            "context",
            "context_autoregressive",
            "tensor",
            "tensor_transpose",
            "tensor_sequence",
            "expert",
            "autoregressive",
        ],
        description="The names of the axes in the logical device mesh.",
    )
    shard_mode = Field("auto", description="can be either auto or explicit")
    inhomogeneous_layer_cycle_interval: int = Field(
        1, description="The interval of repeated inhomogeneous layer patterns."
    )
    scan_layers: bool = Field(
        True, description="Whether to use jax.lax.scan over layers."
    )
    param_scan_axis: int = Field(1, description="Axis to scan over for parameters.")
    context_parallel_load_balance: bool = Field(
        True, description="Whether to use load balancing for context parallelism."
    )
    context_parallel_strategy: str = Field(
        "all_gather",
        description="Strategy for context parallelism ('all_gather' or 'ring').",
    )
    custom_mesh: str = Field(
        "", description="Available options: ['hybrid_ring_64x4', 'hybrid_ring_32x8']"
    )
    allow_split_physical_axes: bool = Field(
        False, description="Allow splitting physical axes for device mesh creation."
    )
    enable_nnx: bool = Field(
        False, description="Whether to use NNX for model definition."
    )
    optimize_mesh_for_tpu_v6e: bool = Field(
        False, description="Apply transformations to the mesh for TPU v6e."
    )
    shardy: bool = Field(True, description="Whether to use shardy XLA backend.")


class LayoutAndSharding(BaseModel):
    """Configuration for data and model sharding rules."""

    logical_axis_rules: Any = Field(
        [], description="Rules for mapping logical axes to physical mesh axes."
    )
    data_sharding: Any = Field([], description="Sharding for input data.")
    input_data_sharding_logical_axes: list[str] = Field(
        ["activation_embed_and_logits_batch", "activation_norm_length"],
        description="Logical axes for sharding input data.",
    )
    sharding_tolerance: float = Field(
        0.02,
        ge=0.0,
        le=1.0,
        description="Allowed percentage of non-sharded parameters.",
    )
    shard_optimizer_over_data: bool = Field(
        False, description="Enable ZeRO-1 optimizer sharding over the data axis."
    )


# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------
def get_individual_scales(scale: int) -> tuple[int, int, int, int]:
    """Choose appropriate scales for individual dimensions based on global scale."""
    if scale == 0:
        raise ValueError("Global parameter scale cannot be zero.")
    log_2_scale = math.floor(math.log2(scale))
    if 2**log_2_scale != scale:
        raise ValueError(
            "Global parameter scale should be a power of two. If you want finer grained control of the model sizes "
            "then you can explicitly set base_embed_dim, base_num_query_heads, base_num_kv_heads, "
            "base_mlp_dim, base_num_decoder_layers and/or head_dim."
        )
    base_scale, rem = divmod(log_2_scale, 3)
    num_head_scale = base_scale + int(rem > 0)
    mlp_dim_scale = num_head_scale
    emb_scale = base_scale + int(rem > 1)
    layer_scale = base_scale
    return emb_scale, num_head_scale, mlp_dim_scale, layer_scale


# ----------------------------------------------------------------------------
# Main Config Class
# ----------------------------------------------------------------------------


class DeepRTEConfig(
    # Run and Checkpointing
    RunInfo,
    Checkpointing,
    OrbaxStorage,
    # Data Types and Quantization
    DataTypes,
    # Core Model Architecture
    ModelArchitecture,
    # Training, Optimization, and Fine-Tuning
    TrainingLoop,
    Optimizer,
    AdamW,
    # Dataset Loading and Tokenization
    DatasetGeneral,
    GrainDataset,
    # Development and Debugging
    DevelopmentAndDebugging,
    Profiling,
    # Metrics and Monitoring
    Metrics,
    Tensorboard,
    # Derived
    DerivedValues,
):
    """
    The main configuration object for MaxText.

    This class aggregates all configuration options from modular `BaseModel` classes
    into a single, validated object. It is populated by the `initialize` function.
    Every field is explicitly defined to prevent misconfigurations (`extra='forbid'`).
    """

    debug: Debug = Field(
        default_factory=Debug, description="Configuration for debugging options."
    )
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def load_model_specific_defaults(cls, values: dict[str, Any]) -> dict[str, Any]:
        """This method is a no-op because `pyconfig` handles model-specific config loading."""
        return values

    @model_validator(mode="after")
    def set_derived_and_validate_values(self) -> "DeepRTEConfig":
        """
        Computes all derived values and runs all cross-field validations after initial parsing.
        This logic is ported from the legacy pyconfig_deprecated.py system and adapted for Pydantic.
        """
        # A. SET RUN NAME AND PATHS
        # If run_name is not set, generate one from the JOBSET_NAME environment variable (if available)
        # or create one from the model name and a timestamp.
        if not self.run_name:
            if os.environ.get("JOBSET_NAME"):
                self.run_name = os.environ.get("JOBSET_NAME")
            else:
                self.run_name = f"{self.model_name}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"

        # Construct full paths for output directories based on the base directory and run name.
        if self.run_name and self.base_output_directory:
            output_dir = os.path.join(self.base_output_directory, self.run_name)
            self.checkpoint_dir = os.path.join(output_dir, "checkpoints", "")
            self.metrics_dir = os.path.join(output_dir, "metrics", "")
            self.tensorboard_dir = os.path.join(output_dir, "tensorboard", "")
        else:
            self.checkpoint_dir, self.metrics_dir, self.tensorboard_dir = (
                None,
                None,
                None,
            )

        # C. SET PRIMARY DEPENDENCIES & DEFAULTS
        # If learning_rate_schedule_steps is -1, it defaults to the total number of training steps.
        if self.learning_rate_schedule_steps == -1:
            self.learning_rate_schedule_steps = self.steps
        # If steps is -1, it defaults to the length of the learning rate schedule.
        if self.steps == -1:
            self.steps = self.learning_rate_schedule_steps
        # If eval_per_device_batch_size is not set, it defaults to the training per_device_batch_size.
        if getattr(self, "eval_per_device_batch_size", 0.0) == 0.0:
            self.eval_per_device_batch_size = self.per_device_batch_size
        # The mu_dtype for the AdamW optimizer defaults to the weight_dtype if not specified.
        if not self.mu_dtype:
            self.mu_dtype = self.weight_dtype

        # D. CALCULATE MODEL DIMENSIONS from global_parameter_scale
        # This allows scaling the model size up or down easily with a single power-of-two factor.

        # E. HARDWARE-DEPENDENT CALCULATIONS
        def get_num_target_devices():
            return len(jax.devices())

        self.num_target_devices = (
            1  # Default for validation when JAX is not initialized
        )
        try:
            self.num_target_devices = get_num_target_devices()
        except (RuntimeError, IndexError):
            logger.warning(
                "JAX device system not available for config validation. Assuming 1 device."
            )

        # F. CALCULATE BATCH SIZES
        def calculate_global_batch_sizes(
            per_device_batch_size, num_devices, grad_accum_steps
        ):
            """Helper to calculate global and micro batch sizes for training and loading."""
            micro_batch_to_train = int(num_devices * per_device_batch_size)
            global_batch_to_train = int(micro_batch_to_train * grad_accum_steps)
            return global_batch_to_train, micro_batch_to_train

        # Calculate final training batch sizes.
        self.global_batch_size_to_train_on, self.micro_batch_size_to_train_on = (
            calculate_global_batch_sizes(
                self.per_device_batch_size,
                self.num_target_devices,
                self.gradient_accumulation_steps,
            )
        )

        # Calculate final evaluation batch sizes.
        self.global_batch_size_to_eval_on, self.micro_batch_size_to_eval_on = (
            calculate_global_batch_sizes(
                self.eval_per_device_batch_size,
                self.num_target_devices,
                1,
            )
        )

        # H. RUN ALL CROSS-FIELD VALIDATIONS
        if self.load_parameters_path and self.load_full_state_path:
            raise ValueError(
                "At most one of `load_parameters_path` or `load_full_state_path` should be set."
            )
        if (
            self.load_parameters_path or self.load_full_state_path
        ) and not self.enable_checkpointing:
            raise ValueError(
                "You must set enable_checkpointing=True to load a checkpoint."
            )

        # I. FINAL TYPE CONVERSIONS AND DERIVED LISTS
        # Create the ici_parallelism and dcn_parallelism lists for legacy compatibility.
        # Final string-to-enum conversions if they haven't been coerced by pydantic yet.

        constant_bound_config = getattr(self, "constant_bound_config", None)
        if isinstance(constant_bound_config, str):
            if constant_bound_config:
                self.constant_bound_config = [
                    float(v.strip()) for v in constant_bound_config.split(",")
                ]
            else:
                self.constant_bound_config = []

        return self
