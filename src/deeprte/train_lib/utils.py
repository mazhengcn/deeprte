from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tensorboardX import writer

from deeprte.input_pipeline import input_pipeline_interface
from deeprte.train_lib import checkpointing, logging

Mesh = jax.sharding.Mesh
PyTree = Any
PartitionSpec = jax.sharding.PartitionSpec
Sharding = jax.sharding.Sharding

Dtype = Any
Shape = tuple[int, ...]
PROXY = object()


# Tree utils.
def _expand_axes(axes, values, name="collect_pytrees"):
    values_tree_def = jax.tree.flatten(values)[1]
    flat_axes = jax.api_util.flatten_axes(name, values_tree_def, axes)
    # Replace None's with PROXY
    flat_axes = [PROXY if x is None else x for x in flat_axes]
    return jax.tree.unflatten(values_tree_def, flat_axes)


def collect_pytrees(
    pytrees: Sequence[PyTree],
    axes: PyTree | int = 0,
    collective_fn: Callable[[Sequence, int], PyTree] | None = None,
):
    axes_ = _expand_axes(axes, pytrees[0])
    if collective_fn:

        def collect_args(*args):
            return collective_fn(args[:-1], args[-1])
    else:

        def collect_args(*args):
            return list(args[:-1])

    return jax.tree.map(collect_args, *pytrees, axes_)


# State initialization utils.
# -----------------------------------------------------------------------------
def calculate_num_params_from_pytree(params):
    params_sizes = jax.tree.map(jax.numpy.size, params)
    total_parameters = jax.tree.reduce(lambda x, y: x + y, params_sizes)
    assert total_parameters >= 0
    return total_parameters


def init_fn(model_class, config, key, tx):
    # Initialization
    model = model_class(config, rngs=nnx.Rngs(key))
    if tx:
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        return model, optimizer
    return model


def setup_training_state(
    model_class, config, rng, tx, data_iterator, checkpoint_manager
) -> tuple[nnx.Module, nnx.Optimizer, input_pipeline_interface.DataIterator]:
    abs_model_and_optimizer = jax.eval_shape(
        lambda: nnx.as_immutable_vars(init_fn(model_class, config, rng, tx))
    )
    graphdef, abs_train_state = nnx.split(abs_model_and_optimizer)

    # Initialization
    restored_train_state, restored_model_state = checkpointing.load_state_if_possible(
        checkpoint_manager,
        data_iterator,
        config.load_parameters_path,
        config.load_full_state_path,
        abs_train_state,
    )

    if restored_train_state:
        if (
            "data_iter" in restored_train_state
            and restored_train_state["data_iter"] is not None
        ):
            data_iterator.local_iterator = restored_train_state["data_iter"]
        model, optimizer = nnx.merge(graphdef, restored_train_state["model_state"])
    elif restored_model_state:
        model = nnx.merge(graphdef, restored_model_state)
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    else:
        model, optimizer = init_fn(model_class, config, rng, tx)

    return model, optimizer, data_iterator


def setup_infer_state(model_class, config, rng, mesh):
    if not config.load_parameters_path:
        # generate random params
        logging.log("No infer checkpoint specified - generating random weights.")
        model = init_fn(model_class, config, rng, None)
    else:
        # Load params from checkpoint
        logging.log(f"Loading decode params from {config.load_parameters_path}")
        abs_model = jax.eval_shape(
            lambda: nnx.as_immutable_vars(init_fn(model_class, config, rng, None))
        )
        graphdef, abs_state = nnx.split(abs_model)
        model_state = checkpointing.load_params_from_path(
            config.load_parameters_path, abs_state
        )
        model = nnx.merge(graphdef, model_state)
    return model


def _bytes_of(x):
    """Return the number of bytes used by a single leaf in a pytree.
    Handles concrete arrays (NumPy/JAX), abstract shapes, scalars, and None.
    Unknown types default to 0.
    """
    # Abstract JAX values: compute bytes from shape × dtype size.
    if isinstance(x, jax.ShapeDtypeStruct):
        # jnp.dtype() normalizes to a consistent dtype object (e.g., handles bfloat16)
        return int(np.prod(x.shape)) * int(jnp.dtype(x.dtype).itemsize)

    # Concrete arrays (NumPy, JAX): rely on their native nbytes property.
    if hasattr(x, "nbytes"):
        return int(x.nbytes)

    # Python scalars (int, float, bool): convert to a NumPy array to measure size.
    if isinstance(x, (int, float, bool)):
        return int(np.array(x).nbytes)

    # None or unsupported leaf types: count as zero bytes.
    if x is not None:
        logging.log(f"Unsupported leaf type in calculate_bytes_from_pytree: {type(x)}")

    return 0


def calculate_bytes_from_pytree(params):
    """Return the total memory footprint (in bytes) of all leaves in a pytree.

    Each leaf is measured using `_bytes_of`. Non-array or unsupported types
    contribute 0 unless they are scalars.
    """
    return sum(map(_bytes_of, jax.tree_util.tree_leaves(params)))


def summarize_size_from_pytree(params):
    num_params = calculate_num_params_from_pytree(params)
    num_bytes = calculate_bytes_from_pytree(params)
    return num_params, num_bytes, num_bytes / num_params


def initialize_summary_writer(tensorboard_dir, run_name):
    summary_writer_path = os.path.join(tensorboard_dir, run_name)
    return (
        writer.SummaryWriter(summary_writer_path) if jax.process_index() == 0 else None
    )


def close_summary_writer(summary_writer):
    if jax.process_index() == 0:
        summary_writer.close()


def add_text_to_summary_writer(key, value, summary_writer):
    """Writes given key-value pair to tensorboard as text/summary."""
    if jax.process_index() == 0:
        summary_writer.add_text(key, value)


def add_config_to_summary_writer(config, summary_writer):
    """Writes config params to tensorboard"""
    if jax.process_index() == 0:
        for key, value in dataclasses.asdict(config).items():
            add_text_to_summary_writer(key, str(value), summary_writer)


class StopTraining(Exception):
    """Custom exception to halt a training process."""

    def __init__(self, reason):
        super().__init__(reason)
