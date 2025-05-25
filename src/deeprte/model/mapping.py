"""Specialized mapping functions."""

import functools
from collections.abc import Callable, Sequence
from typing import Any, Optional

import jax
import jax.numpy as jnp

PYTREE = Any
PYTREE_JAX_ARRAY = Any

partial = functools.partial
PROXY = object()


def collect_pytrees(
    pytrees: Sequence[PYTREE],
    axes: PYTREE | int = 0,
    collective_fn: Callable[[Sequence, int], PYTREE] | None = None,
):
    axes_ = _expand_axes(axes, pytrees[0])

    if collective_fn:

        def collect_args(*args):
            return collective_fn(args[:-1], args[-1])  # ty: ignore
    else:

        def collect_args(*args):
            return list(args[:-1])

    return jax.tree.map(collect_args, *pytrees, axes_)


def _maybe_slice(array, i, slice_size, axis):
    if axis is PROXY:
        return array
    else:
        return jax.lax.dynamic_slice_in_dim(array, i, slice_size=slice_size, axis=axis)


def _maybe_get_size(array, axis):
    if axis == PROXY:
        return -1
    else:
        return array.shape[axis]


def _expand_axes(axes, values, name="sharded_apply"):
    values_tree_def = jax.tree.flatten(values)[1]
    flat_axes = jax.api_util.flatten_axes(name, values_tree_def, axes)
    # Replace None's with PROXY
    flat_axes = [PROXY if x is None else x for x in flat_axes]
    return jax.tree.unflatten(values_tree_def, flat_axes)


def _concat_or_stack_arrays(arrays, axis):
    if arrays[0].ndim == 0:
        return jnp.stack(arrays)
    else:
        return jnp.concatenate(arrays, axis=axis)


def sharded_apply(
    fun: Callable[..., PYTREE_JAX_ARRAY],  # pylint: disable=g-bare-generic
    shard_size: int | None = 1,
    in_axes: int | PYTREE = 0,
    out_axes: int | PYTREE = 0,
) -> Callable[..., PYTREE]:
    docstr = (
        "Mapped version of {fun}. Takes similar arguments to {fun} "
        "but with additional array axes over which {fun} is mapped."
    )

    @jax.util.wraps(fun, docstr=docstr)
    def mapped_fn(*args):
        # Expand in axes and Determine Loop range
        in_axes_ = _expand_axes(in_axes, args)
        in_sizes = jax.tree.map(_maybe_get_size, args, in_axes_)
        flat_sizes = jax.tree.flatten(in_sizes)[0]
        in_size = max(flat_sizes)
        assert all(i in {in_size, -1} for i in flat_sizes)

        num_shards = in_size // shard_size
        # Fix Up if necessary
        last_shard_size = in_size % shard_size

        def compute_shard(slice_start, slice_size):
            input_slice = jax.tree.map(
                lambda array, axis: _maybe_slice(array, slice_start, slice_size, axis),
                args,
                in_axes_,
            )
            return fun(*input_slice)

        outputs = []
        for i in range(num_shards):
            sliced_outputs = compute_shard(i * shard_size, shard_size)  # ty: ignore
            outputs.append(sliced_outputs)

        if last_shard_size != 0:
            remainder_start = in_size - last_shard_size
            sliced_outputs = compute_shard(remainder_start, last_shard_size)
            outputs.append(sliced_outputs)

        out_axes_ = _expand_axes(out_axes, outputs[0])
        outputs = collect_pytrees(outputs, out_axes_, _concat_or_stack_arrays)
        return outputs

    return mapped_fn


def sharded_apply_with_scan(
    fun: Callable[..., PYTREE_JAX_ARRAY],  # pylint: disable=g-bare-generic
    shard_size: int | None = 1,
    in_axes: int | PYTREE = 0,
    out_axes: int | PYTREE = 0,
    new_out_axes: bool = False,
) -> Callable[..., PYTREE_JAX_ARRAY]:
    docstr = (
        "Mapped version of {fun}. Takes similar arguments to {fun} "
        "but with additional array axes over which {fun} is mapped."
    )
    if new_out_axes:
        raise NotImplementedError("New output axes not yet implemented.")

    # shard size None denotes no sharding
    if shard_size is None:
        return fun

    @jax.util.wraps(fun, docstr=docstr)
    def mapped_fn(*args):
        # Expand in axes and Determine Loop range
        in_axes_ = _expand_axes(in_axes, args)

        in_sizes = jax.tree.map(_maybe_get_size, args, in_axes_)
        flat_sizes = jax.tree.flatten(in_sizes)[0]
        in_size = max(flat_sizes)
        assert all(i in {in_size, -1} for i in flat_sizes)

        num_extra_shards = (in_size - 1) // shard_size

        # Fix Up if necessary
        last_shard_size = in_size % shard_size
        last_shard_size = shard_size if last_shard_size == 0 else last_shard_size

        def apply_fun_to_slice(slice_start, slice_size):
            input_slice = jax.tree.map(
                lambda array, axis: _maybe_slice(array, slice_start, slice_size, axis),
                args,
                in_axes_,
            )
            return fun(*input_slice)

        remainder_shape_dtype = jax.eval_shape(
            partial(apply_fun_to_slice, 0, last_shard_size)
        )
        out_dtypes = jax.tree.map(lambda x: x.dtype, remainder_shape_dtype)
        out_shapes = jax.tree.map(lambda x: x.shape, remainder_shape_dtype)
        out_axes_ = _expand_axes(out_axes, remainder_shape_dtype)

        if num_extra_shards > 0:
            regular_shard_shape_dtype = jax.eval_shape(
                partial(apply_fun_to_slice, 0, shard_size)
            )
            shard_shapes = jax.tree.map(lambda x: x.shape, regular_shard_shape_dtype)

            def make_output_shape(axis, shard_shape, remainder_shape):
                return (
                    shard_shape[:axis]
                    + (shard_shape[axis] * num_extra_shards + remainder_shape[axis],)
                    + shard_shape[axis + 1 :]
                )

            out_shapes = jax.tree.map(
                make_output_shape, out_axes_, shard_shapes, out_shapes
            )

        # Calls dynamic Update slice with different argument order
        # This is here since tree_map only works with positional arguments
        def dynamic_update_slice_in_dim(full_array, update, axis, i):
            return jax.lax.dynamic_update_slice_in_dim(full_array, update, i, axis)

        def compute_shard(outputs, slice_start, slice_size):
            slice_out = apply_fun_to_slice(slice_start, slice_size)
            update_slice = partial(dynamic_update_slice_in_dim, i=slice_start)
            return jax.tree.map(update_slice, outputs, slice_out, out_axes_)

        def scan_iteration(outputs, i):
            new_outputs = compute_shard(outputs, i, shard_size)
            return new_outputs, ()

        slice_starts = jnp.arange(0, in_size - shard_size + 1, shard_size)

        def allocate_buffer(dtype, shape):
            return jnp.zeros(shape, dtype=dtype)

        outputs = jax.tree.map(allocate_buffer, out_dtypes, out_shapes)

        if slice_starts.shape[0] > 0:
            outputs, _ = jax.lax.scan(scan_iteration, outputs, slice_starts)

        if last_shard_size != shard_size:
            remainder_start = in_size - last_shard_size
            outputs = compute_shard(outputs, remainder_start, last_shard_size)

        return outputs

    return mapped_fn


def inference_subbatch(
    module: Callable[..., PYTREE_JAX_ARRAY],
    subbatch_size: int,
    batched_args: dict[PYTREE_JAX_ARRAY],  # ty: ignore
    nonbatched_args: dict[PYTREE_JAX_ARRAY],  # ty: ignore
    low_memory: bool = True,
    input_subbatch_dim: int = 0,
    output_subbatch_dim: Optional[int] = None,
    in_jit: bool = False,
) -> PYTREE_JAX_ARRAY:
    """Run through subbatches (like batch apply but with split and concat)."""
    assert len(batched_args) > 0

    if not low_memory:
        args = batched_args | nonbatched_args
        return module(args)

    if output_subbatch_dim is None:
        output_subbatch_dim = input_subbatch_dim

    def run_module(batched_args):
        args = batched_args | nonbatched_args
        return module(args)

    if in_jit:
        apply = sharded_apply_with_scan
    else:
        apply = sharded_apply

    sharded_module = apply(
        run_module,
        shard_size=subbatch_size,
        in_axes=input_subbatch_dim,
        out_axes=output_subbatch_dim,
    )

    return sharded_module(batched_args)
