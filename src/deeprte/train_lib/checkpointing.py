"""Create an Orbax CheckpointManager with specified (Async or not) Checkpointer."""

import pathlib
import time

import grain.python as grain
import jax
import orbax.checkpoint as ocp
from etils import epath
from flax import nnx

from deeprte.train_lib import logging
from deeprte.train_lib import utils as train_utils
from deeprte.train_lib.multihost_dataloading import MultiHostDataLoadIterator


def create_orbax_checkpoint_manager(
    checkpoint_dir: str | pathlib.Path,
    enable_checkpointing: bool,
    use_async: bool,
    save_interval_steps: int,
):
    """Returns specified Orbax (async or not) CheckpointManager or None if checkpointing is disabled."""
    if not enable_checkpointing:
        logging.log("Checkpointing disabled, not creating checkpoint manager.")
        return None
    logging.log("Creating checkpoint manager...")

    p = epath.Path(checkpoint_dir)
    p.mkdir(exist_ok=True, parents=True)
    manager = ocp.CheckpointManager(
        p,
        item_names=("train_state", "data_iter"),
        options=ocp.CheckpointManagerOptions(
            create=True,
            save_interval_steps=save_interval_steps,  # type: ignore
            enable_async_checkpointing=use_async,  # type: ignore
        ),
    )
    logging.log("Checkpoint manager created!")

    return manager


def load_state_if_possible(
    checkpoint_manager: ocp.CheckpointManager | None,
    data_iterator: MultiHostDataLoadIterator | None,
    load_parameters_from_path: str,
    load_full_state_from_path: str,
    abstract_unboxed_pre_state: nnx.State,
    step: int = -1,
):
    if checkpoint_manager is not None:
        logging.log(
            "checkpoint manager exists so trying to load this run's existing checkpoint"
        )

        if step < 0:
            latest_step = checkpoint_manager.latest_step()
            if latest_step is not None:
                step = latest_step
        if step is not None:
            logging.log(f"restoring from this run's directory latest step {step}")
            return (
                checkpoint_manager.restore(
                    step,
                    args=ocp.args.Composite(
                        train_state=ocp.args.StandardRestore(
                            abstract_unboxed_pre_state
                        ),  # type: ignore
                        data_iter=grain.PyGrainCheckpointRestore(  # type: ignore
                            data_iterator.local_iterator  # type: ignore
                        ),
                    ),
                ),
                None,
            )

    if load_parameters_from_path != "":
        restored_params = load_params_from_path(
            load_parameters_from_path, abstract_unboxed_pre_state
        )
        return None, restored_params
    elif load_full_state_from_path != "":
        logging.log(f"restoring full state from {load_full_state_from_path=}")
        p = epath.Path(load_full_state_from_path)
        ckptr = ocp.StandardCheckpointer()
        restored_train_state = ckptr.restore(p, abstract_unboxed_pre_state)
        return {"train_state": restored_train_state}, None
    else:
        logging.log("No existing checkpoints found, not restoring checkpoint.")
    return None, None


def load_params_from_path(load_parameters_from_path, abstract_model_state):
    """Load inference params from checkpoint at specified path."""
    assert load_parameters_from_path, "load_parameters_from_path is not defined."
    logging.log(f"restoring params from {load_parameters_from_path}")
    ckpt = epath.Path(load_parameters_from_path)
    ckptr = ocp.StandardCheckpointer()
    restored_model_state = ckptr.restore(ckpt, target=abstract_model_state)
    return restored_model_state


def save_params_to_path(checkpoint_dir, model_state):
    """Save params in checkpoint at specified path."""
    assert checkpoint_dir, "checkpoint_dir is not defined."
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(checkpoint_dir, model_state)
    ckptr.wait_until_finished()


def print_save_message(step, async_checkpointing):
    if async_checkpointing:
        logging.log(f"Started an asynchronous checkpoint save for step {step}")
    else:
        logging.log(f"Saved a checkpoint at step {step}.")


def maybe_save_checkpoint(checkpoint_manager, state, config, data_iterator, step=None):
    """Save checkpoint if checkpointing is enabled."""
    if checkpoint_manager is None:
        return

    # Determine the effective step for saving a checkpoint.
    # If 'step' is not provided, this call is for a potential final checkpoint
    # and use the last completed step from the state.
    actual_step = (int(state.step) - 1) if step is None else int(step)

    # Determine if a checkpoint save should be forced, overriding the usual `config.checkpoint_period` logic.
    # This occurs if this function was called:
    # without an explicit 'step' (implying it's a checkpoint save for final step),
    # AND the 'actual_step' is a valid step,
    # AND it's not a step that would normally trigger a checkpoint save.
    force_ckpt_save = (
        step is None
        and actual_step != -1
        and (actual_step % config.checkpoint_period != 0)
    )

    try:
        checkpoint_saved = save_checkpoint(
            checkpoint_manager,
            actual_step,
            state,
            config,
            data_iterator,
            force_ckpt_save,
        )
        if checkpoint_saved:
            print_save_message(actual_step, config.async_checkpointing)
    except Exception as e:
        raise train_utils.StopTraining(f"Checkpointing failed. {str(e)}") from e

    # Wait for any pending checkpoint save to finish during preemption or final step save
    if force_ckpt_save or checkpoint_manager.reached_preemption(actual_step):
        checkpoint_manager.wait_until_finished()

    # Raise exception upon preemption
    if checkpoint_manager.reached_preemption(actual_step):
        raise train_utils.StopTraining("Job is preempted.")


def save_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    step,
    state,
    config,
    data_iterator=None,
    force=False,
):
    if step % config.checkpoint_period == 0:
        blocking_until_ready_start = time.time()
        logging.log(f"Waiting for step {step} to finish before checkpoint...")
        # We block here on the step finishing so that our checkpointing metrics
        # measure only checkpointing time, not training time.
        jax.block_until_ready(state)
        logging.log(
            f"Waited {time.time() - blocking_until_ready_start} seconds for step "
            f"{step} to finish before starting checkpointing."
        )

    return checkpoint_manager.save(
        step,
        args=ocp.args.Composite(
            train_state=ocp.args.StandardSave(state),  # type: ignore
            data_iter=grain.PyGrainCheckpointSave(data_iterator.local_iterator),  # type: ignore
        ),
        force=force,
    )
