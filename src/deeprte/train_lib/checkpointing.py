"""Create an Orbax CheckpointManager with specified (Async or not) Checkpointer."""

from typing import Any, Optional

import grain.python as grain
import orbax.checkpoint as ocp
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager
from absl import logging
from etils import epath
from flax import nnx
from orbax.checkpoint.logging import abstract_logger

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


def save_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    step,
    state,
    dataset_type="tfds",
    data_iterator=None,
):
    if dataset_type == "grain":
        return checkpoint_manager.save(
            step,
            args=ocp.args.Composite(
                train_state=ocp.args.StandardSave(state),
                data_iter=grain.PyGrainCheckpointSave(data_iterator.local_iterator),
            ),
        )
    else:
        return checkpoint_manager.save(
            step,
            args=ocp.args.Composite(train_state=ocp.args.StandardSave(state)),
        )


def load_state_if_possible(
    checkpoint_manager: ocp.CheckpointManager | None,
    data_iterator: MultiHostDataLoadIterator | None,
    load_parameters_from_path: str,
    load_full_state_from_path: str,
    abstract_state: nnx.State,
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
                f"restoring from this run's directory latest step {latest_step}"
            )

            if dataset_type == "grain" and data_iterator is not None:
                return (
                    checkpoint_manager.restore(
                        latest_step,
                        args=ocp.args.Composite(
                            train_state=ocp.args.StandardRestore(abstract_state),
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
                            train_state=ocp.args.StandardRestore(abstract_state)
                        ),
                    ),
                    None,
                )

    if load_parameters_from_path != "":
        restored_params = load_params_from_path(
            load_parameters_from_path, abstract_state.model
        )
        return None, restored_params
    elif load_full_state_from_path != "":
        logging.info(f"restoring full state from {load_full_state_from_path=}")
        p = epath.Path(load_full_state_from_path)
        ckptr = ocp.StandardCheckpointer()
        restored = ckptr.restore(p, abstract_state)
        return restored, None

    else:
        logging.info("No existing checkpoints found, not restoring checkpoint.")
        return None, None


def load_params_from_path(load_parameters_from_path, abstract_params):
    """Load inference params from checkpoint at specified path."""
    assert load_parameters_from_path, "load_parameters_from_path is not defined."
    logging.info(f"restoring params from {load_parameters_from_path}")
    ckpt = epath.Path(load_parameters_from_path)
    ckptr = ocp.StandardCheckpointer()
    restored = ckptr.restore(ckpt, target=abstract_params)
    return restored


def save_params_to_path(checkpoint_dir, params):
    """Save params in checkpoint at specified path."""
    assert checkpoint_dir, "checkpoint_dir is not defined."
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(checkpoint_dir, params)
    print(f"Params checkpoint saved at: {checkpoint_dir}")
