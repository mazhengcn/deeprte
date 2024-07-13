"""Create an Orbax CheckpointManager with specified (Async or not) Checkpointer."""

from typing import Any, Optional

import grain.python as grain
import jax
import numpy as np
import orbax.checkpoint as ocp
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager
from absl import flags, logging
from etils import epath
from flax import nnx
from orbax.checkpoint.logging import (
    abstract_logger,
    composite_logger,
    standard_logger,
)

from deeprte.train_lib.multihost_dataloading import MultiHostDataLoadIterator

PyTree = Any
PyTreeCheckpointHandler = ocp.PyTreeCheckpointHandler
LocalCheckpointOptions = emergency_checkpoint_manager.LocalCheckpointOptions
PersistentCheckpointOptions = emergency_checkpoint_manager.PersistentCheckpointOptions


def create_orbax_checkpoint_manager(
    checkpoint_dir: str,
    enable_checkpointing: bool,
    use_async: bool,
    save_interval_steps: int,
    dataset_type: Optional[str] = "tfds",
    orbax_logger: Optional[abstract_logger.AbstractLogger] = None,
):
    """Returns specified Orbax (async or not) CheckpointManager or None if checkpointing is disabled."""
    if not enable_checkpointing:
        logging.info("Checkpointing disabled, not creating checkpoint manager.")
        return None
    logging.info("Creating checkpoint manager...")
    p = epath.Path(checkpoint_dir)

    if dataset_type == "grain":
        item_names = ("train_state", "data_iter")
    else:
        item_names = ("train_state",)

    options = ocp.CheckpointManagerOptions(
        save_interval_steps=save_interval_steps, enable_async_checkpointing=use_async
    )
    mngr = ocp.CheckpointManager(
        p, item_names=item_names, options=options, logger=orbax_logger
    )
    logging.info("Checkpoint manager created!")

    return mngr


def create_orbax_emergency_checkpoint_manager(
    local_checkpoint_dir: str,
    persistent_checkpoint_dir: str,
    global_mesh: jax.sharding.Mesh,
    abstract_state: PyTree,
    local_save_interval_steps: int,
    persistent_save_interval_steps: int,
):
    """Returns an emergency checkpoint."""
    flags.FLAGS.experimental_orbax_use_distributed_process_id = True
    logging.info("Creating emergency checkpoint manager...")

    local_registry = ocp.type_handlers.create_type_handler_registry(
        (
            jax.Array,
            ocp.type_handlers.ArrayHandler(primary_host=None, replica_id=None),
        ),
    )

    local_checkpoint_handler = PyTreeCheckpointHandler(
        use_ocdbt=True,
        use_zarr3=True,
        primary_host=None,
        type_handler_registry=local_registry,
    )

    options = emergency_checkpoint_manager.CheckpointManagerOptions(
        local=LocalCheckpointOptions(save_interval_steps=local_save_interval_steps),
        persistent=PersistentCheckpointOptions(
            save_interval_steps=persistent_save_interval_steps
        ),
    )

    emergency_mngr = emergency_checkpoint_manager.CheckpointManager(
        local_checkpoint_dir,
        epath.Path(persistent_checkpoint_dir),
        global_mesh=global_mesh,
        abstract_state=abstract_state,
        options=options,
        local_state_handler=local_checkpoint_handler,
    )

    logging.info("Emergency checkpoint manager created!")
    return emergency_mngr


def _find_idx(array: np.ndarray, replica_axis_idx: int):
    """Returns the index along given dimension that the current host belongs to."""
    idx = None
    for idx, val in np.ndenumerate(array):
        if val.process_index == jax.process_index():
            break
    return idx[replica_axis_idx]


def _replica_devices(device_array: np.ndarray, replica_axis_idx: int):
    """Returns the devices from the replica that current host belongs to.

    Replicas are assumed to be restricted to the first axis.

    Args:
      device_array: devices of the mesh that can be obtained by mesh.devices()
      replica_axis_idx: axis dimension along which replica is taken

    Returns:
      devices inside the replica that current host is in
    """
    idx = _find_idx(device_array, replica_axis_idx)
    replica_result = np.take(device_array, idx, axis=replica_axis_idx)
    return np.expand_dims(replica_result, axis=replica_axis_idx)


def load_state_if_possible(
    checkpoint_manager: ocp.CheckpointManager | None,
    data_iterator: MultiHostDataLoadIterator | None,
    load_parameters_from_path: str,
    load_full_state_from_path: str,
    abstract_pre_state: nnx.TrainState,
    enable_single_replica_ckpt_restoring: Optional[bool] = False,
    dataset_type: Optional[str] = "tfds",
):
    """Loads TrainState as possible from the inputs.

    Args:
      checkpoint_manager: if the checkpoint_manager has a valid checkpoint, return
        that TrainState. This enables a full reload of a run in progress.
      load_parameters_from_path: if there is no checkpoint in the checkpoint manager,
        load parameters from a parameter only checkpoint at this path.
      load_full_state_from_path: if there is no checkpoint in the checkpoint manager,
        load full state from a full state checkpoint at this path.
      abstract_unboxed_pre_state: an unboxed, abstract TrainState that Orbax
        matches type against.
      enable_single_replica_ckpt_restoring: bool flag for restoring checkpoitn
        with SingleReplicaArrayHandler

    Returns:
      A tuple of (train_state, train_state_params) where full_train_state captures
       a full reload and train_state_params just the params for a partial reload.
       At most one will be non-None. Both can be None if neither checkpoint is
       set.
    """

    if checkpoint_manager is not None:
        logging.info(
            "checkpoint manager exists so trying to load this run's existing checkpoint"
        )

        latest_step = checkpoint_manager.latest_step()
        if latest_step is not None:
            logging.info(
                f"restoring from this run's directory latest step \
          {latest_step}"
            )

            def map_to_pspec(data):
                pspec = data.sharding.spec
                mesh = data.sharding.mesh
                if not enable_single_replica_ckpt_restoring:
                    return ocp.type_handlers.ArrayRestoreArgs(
                        mesh=mesh, mesh_axes=pspec
                    )
                replica_axis_index = 0
                replica_devices = _replica_devices(mesh.devices, replica_axis_index)
                replica_mesh = jax.sharding.Mesh(replica_devices, mesh.axis_names)
                single_replica_sharding = jax.sharding.NamedSharding(
                    replica_mesh, pspec
                )

                array_handler = ocp.type_handlers.SingleReplicaArrayHandler(
                    replica_axis_index=0,
                    broadcast_memory_limit_bytes=1024 * 1024 * 1000,  # 1000 MB limit
                )
                ocp.type_handlers.register_type_handler(
                    jax.Array, array_handler, override=True
                )

                return ocp.type_handlers.SingleReplicaArrayRestoreArgs(
                    sharding=jax.sharding.NamedSharding(mesh, pspec),
                    single_replica_sharding=single_replica_sharding,
                    global_shape=data.shape,
                    dtype=data.dtype,
                )

            restore_args = jax.tree.map(map_to_pspec, abstract_pre_state)

            if isinstance(
                checkpoint_manager, emergency_checkpoint_manager.CheckpointManager
            ):
                return (
                    checkpoint_manager.restore(
                        latest_step,
                        args=ocp.args.PyTreeRestore(
                            train_state=abstract_pre_state, restore_args=restore_args
                        ),
                    ),
                    None,
                )

            if dataset_type == "grain" and data_iterator is not None:
                return (
                    checkpoint_manager.restore(
                        latest_step,
                        args=ocp.args.Composite(
                            train_state=ocp.args.PyTreeRestore(
                                item=abstract_pre_state,
                                restore_args=restore_args,
                            ),
                            data_iter=grain.PyGrainCheckpointRestore(
                                data_iterator.local_iterator
                            ),
                        ),
                    ),
                    None,
                )
            else:
                return (
                    checkpoint_manager.restore(
                        latest_step,
                        args=ocp.args.Composite(
                            train_state=ocp.args.PyTreeRestore(
                                item=abstract_pre_state,
                                restore_args=restore_args,
                            )
                        ),
                    ),
                    None,
                )

    if load_parameters_from_path != "":
        restored_params = load_params_from_path(
            load_parameters_from_path, abstract_pre_state.params
        )
        return None, restored_params
    elif load_full_state_from_path != "":
        logging.info(f"restoring full state from {load_full_state_from_path=}")
        p = epath.Path(load_full_state_from_path)
        ckptr = ocp.StandardCheckpointer()
        restored = ckptr.restore(p, args=ocp.args.StandardRestore(abstract_pre_state))
        return {"train_state": restored}, None

    else:
        logging.info("No existing checkpoints found, not restoring checkpoint.")
        return None, None


def setup_checkpoint_logger(config) -> composite_logger.CompositeLogger | None:
    """Setup checkpoint logger.
    Args:
      config
    Returns:
      CompositeLogger
    """
    orbax_cloud_logger = None
    orbax_standard_logger = None
    logging.info("Setting up checkpoint logger...")

    if config.enable_checkpoint_standard_logger:
        orbax_standard_logger = standard_logger.StandardLogger()
        logging.info("Successfully set up checkpoint standard logger.")

    orbax_logger = None
    if orbax_cloud_logger is not None and orbax_standard_logger is not None:
        orbax_logger = composite_logger.CompositeLogger(
            orbax_cloud_logger, orbax_standard_logger
        )
        logging.info("Successfully set up checkpoint composite logger.")

    return orbax_logger


def load_params_from_path(load_parameters_from_path, abstract_params):
    """Load decode params from checkpoint at specified path."""
    assert load_parameters_from_path, "load_parameters_from_path is not defined."
    logging.info(f"restoring params from {load_parameters_from_path}")
    ckpt = epath.Path(load_parameters_from_path)
    ckptr = ocp.StandardCheckpointer()
    restored = ckptr.restore(
        ckpt, args=ocp.args.StandardRestore({"params": abstract_params})
    )
    return restored["params"]


def save_params_to_path(checkpoint_dir, params):
    """Save decode params in checkpoint at specified path."""
    assert checkpoint_dir, "checkpoint_dir is not defined."
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(
        checkpoint_dir, args=ocp.args.StandardSave({"params": params}), force=True
    )
    print(f"Params checkpoint saved at: {checkpoint_dir}")
