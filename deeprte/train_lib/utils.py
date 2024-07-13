import functools
import json
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from flax import nnx
from jax.experimental import mesh_utils
from typing_extensions import Protocol, runtime_checkable

from deeprte.train_lib import checkpointing

Mesh = jax.sharding.Mesh
PyTree = Any

Dtype = Any
Shape = tuple[int, ...]


TrainState = nnx.TrainState


@runtime_checkable
class HasCache(Protocol):
    def init_cache(self, input_shape: Shape, dtype: Dtype = jnp.float32): ...


def close_summary_writer(summary_writer):
    if jax.process_index() == 0:
        summary_writer.close()


def _prepare_metrics_for_json(metrics, step, run_name):
    """Converts metric dictionary into json supported types (e.g. float)"""
    metrics_dict = {}
    for val in metrics["scalar"]:
        metrics_dict[val] = float(metrics["scalar"][val])
    metrics_dict["step"] = float(step)
    metrics_dict["run_name"] = run_name
    return metrics_dict


def write_metrics_locally(metrics, step, config, file):
    """Writes metrics locally for testing"""
    if step == 0:
        file.truncate(0)

    metrics_dict = _prepare_metrics_for_json(metrics, step, config.run_name)
    file.write(str(json.dumps(metrics_dict)) + "\n")

    if step == config.steps - 1:
        file.close()


# Mesh utils.
# -----------------------------------------------------------------------------


def create_device_mesh(config, devices=None):
    """Creates a device mesh with each slice in its own data parallel group. If there is only one slice, uses two replicas."""
    if devices is None:
        devices = jax.devices()
    num_devices = len(devices)
    try:
        num_slices = 1 + max([d.slice_index for d in devices])
    except:  # noqa: E722
        num_slices = 1
    num_devices_per_slice = num_devices // num_slices
    logging.info(f"Devices: {devices}")
    logging.info(f"Number of devices: {num_devices}")

    multi_slice_env = hasattr(jax.devices()[0], "slice_index")

    dcn_parallelism = [
        config.dcn_data_parallelism,
        config.dcn_fsdp_parallelism,
        config.dcn_tensor_parallelism,
    ]
    ici_parallelism = [
        config.ici_data_parallelism,
        config.ici_fsdp_parallelism,
        config.ici_tensor_parallelism,
    ]

    # Find possible unspecified parallelisms
    dcn_parallelism = fill_unspecified_mesh_axes(dcn_parallelism, num_slices, "DCN")
    ici_parallelism = fill_unspecified_mesh_axes(
        ici_parallelism, num_devices_per_slice, "ICI"
    )

    if multi_slice_env:
        mesh = mesh_utils.create_hybrid_device_mesh(ici_parallelism, dcn_parallelism)
    else:
        mesh = mesh_utils.create_device_mesh(ici_parallelism)

    logging.info(f"Decided on mesh: {mesh}")
    logging.info(f"Mesh shape: {mesh.shape}")

    return mesh


def fill_unspecified_mesh_axes(parallelism_vals, target_product, parallelism_type):
    """Evaluates unspecified DCN/ICI parallelism values"""
    if -1 in parallelism_vals:
        assert parallelism_vals.count(-1) == 1, (
            f"Found unspecified values (-1) for more than one {parallelism_type}   "
            "   parallelism axis. At most one axis can be unspecified."
        )

        determined_val = target_product / np.prod(parallelism_vals) * -1

        assert determined_val >= 1 and determined_val.is_integer, (
            "Unspecified value unable to be determined with the given     "
            f" {parallelism_type} parallelism values"
        )

        parallelism_vals[parallelism_vals.index(-1)] = int(determined_val)

    target_type = "slices" if parallelism_type == "DCN" else "devices per slice"

    assert np.prod(parallelism_vals) == target_product, (
        f"Number of {target_type} {target_product} does not match    the product"
        f" of the {parallelism_type} parallelism {np.prod(parallelism_vals)}"
    )

    return parallelism_vals


# State initialization utils.
# -----------------------------------------------------------------------------


def _to_array(x):
    if not isinstance(x, jax.Array):
        x = jnp.asarray(x)
    return x


def init_train_state(constructor: Callable, tx, config, rng):
    """We initialize the model and optimizer state, and optionally load from a
    checkpoint as necessary.

    Args:
      constructor: the model constructor
      tx: the optax.GradientTransformation
      config: config object
      rng: jax.prng key
      mesh: jax.devices() mesh

    Returns:
      state: the initialized train state
      state_mesh_annotations: the mesh annotations for the train state
    """

    # Initialization
    model = constructor(config, rng)
    graphdef, params = nnx.split(model, nnx.Param)
    state = TrainState.create(graphdef=graphdef, params=params, tx=tx)
    state = jax.tree.map(_to_array, state)

    return state


def setup_initial_state(
    constructor, data_iterator, tx, config, rng, mesh, checkpoint_manager
):
    """We initialize the model and optimizer state, and optionally load from a
    checkpoint as necessary.

    Args:
      model: the flax model to initialize
      tx: the optax.GradientTransformation
      config: config object
      rng: jax.prng key
      mesh: jax.devices() mesh
      checkpoint_manager: an Orbax checkpointing.CheckpointManager object
      is_training: True to initialize training state, False for decode state

    Returns:
      state: the initialized train state
      state_mesh_annotations: the mesh annotations for the train state
    """

    abstract_sharded_state, state_shardings = get_abstract_state(
        constructor, tx, config, rng, mesh
    )

    # Initialization
    restored, raw_params = checkpointing.load_state_if_possible(
        checkpoint_manager,
        data_iterator,
        config.load_parameters_path,
        config.load_full_state_path,
        abstract_sharded_state,
        config.enable_single_replica_ckpt_restoring,
        config.dataset_type,
    )

    if restored:
        if isinstance(
            checkpoint_manager,
            checkpointing.emergency_checkpoint_manager.CheckpointManager,
        ):
            state = restored
        else:
            if "iter" in restored and restored["iter"] is not None:
                data_iterator.local_iterator = restored["iter"]
            state = restored["items"]
    else:
        init_train_state_partial = functools.partial(
            init_train_state, constructor, tx, config
        )
        state = jax.jit(init_train_state_partial, out_shardings=state_shardings)(rng)
        if raw_params:  # If we loaded a partial state, we need to merge it.
            state = state.replace(params=raw_params)

    return state, data_iterator


def get_abstract_state(constructor, tx, config, rng, mesh):
    """Get a shaped abstraction of the state (including optimizer)"""

    def init_fn():
        return init_train_state(constructor, tx, config, rng)

    abstract_state = jax.eval_shape(init_fn)
    state_shardings = nnx.get_named_sharding(abstract_state, mesh)
    abstract_sharded_state = jax.jit(
        init_fn, out_shardings=state_shardings
    ).eval_shape()

    return abstract_sharded_state, state_shardings
