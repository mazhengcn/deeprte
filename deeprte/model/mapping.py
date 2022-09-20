# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Specialized mapping functions."""

import functools
from collections.abc import Callable, Sequence
from typing import Any, Optional

import haiku as hk
import jax
import jax.numpy as jnp

PYTREE = Any
PYTREE_JAX_ARRAY = Any

partial = functools.partial
PROXY = object()


def _apply_excluded(func, excluded, args):
    """Partially apply positional arguments in `excluded` to a function."""
    if not excluded:
        return func, args

    if max(excluded) >= len(args):
        raise ValueError(
            f"excluded={excluded:!r} is invalid for {len(args):!r} argument(s)"
        )

    dynamic_args = [arg for i, arg in enumerate(args) if i not in excluded]
    static_args = [(i, args[i]) for i in sorted(excluded)]

    def new_fun(*args):
        args = list(args)
        for i, arg in static_args:
            args.insert(i, arg)
        return func(*args)

    return new_fun, dynamic_args


def _maybe_slice(array, i, slice_size, axis):
    if axis is PROXY:
        return array
    return jax.lax.dynamic_slice_in_dim(array, i, slice_size=slice_size, axis=axis)


def _maybe_get_size(array, axis):
    if axis == PROXY:
        return -1
    return array.shape[axis]


def _expand_axes(axes, values, name="sharded_apply"):
    values_tree_def = jax.tree_util.tree_flatten(values)[1]
    flat_axes = jax.api_util.flatten_axes(name, values_tree_def, axes)
    # Replace None's with PROXY
    flat_axes = [PROXY if x is None else x for x in flat_axes]
    return jax.tree_util.tree_unflatten(values_tree_def, flat_axes)


def vmap(
    func: Callable,
    shard_size: int | None = None,
    argnums: frozenset = None,
    excluded: frozenset = None,
    in_axes: Optional[int | tuple] = 0,
    out_axes: Optional[int | tuple] = 0,
    use_hk: Optional[bool] = False,
) -> Callable[..., Any]:

    if argnums and excluded:
        raise RuntimeError(
            "Can't use `argnums` and `excluded` arguments at the same time."
        )

    if shard_size:
        vmap_ = partial(sharded_map, shard_size=shard_size, use_hk=use_hk)
    else:
        vmap_ = (
            partial(hk.vmap, split_rng=(not hk.running_init())) if use_hk else jax.vmap
        )

    @functools.wraps(func)
    def wrapped(*args):
        if argnums:
            excluded_argnums = [i for i, _ in enumerate(args) if i not in argnums]
        else:
            excluded_argnums = excluded
        excluded_fun, args = _apply_excluded(func, excluded_argnums, args)

        return vmap_(excluded_fun, in_axes=in_axes, out_axes=out_axes)(*args)

    return wrapped


def sharded_map(
    func: Callable[..., PYTREE_JAX_ARRAY],
    shard_size: int | None = 1,
    in_axes: int | PYTREE = 0,
    out_axes: int | PYTREE = 0,
    use_hk: bool = False,
) -> Callable[..., PYTREE_JAX_ARRAY]:
    """Sharded vmap.

    Maps `fun` over axes, in a way similar to vmap, but does so in shards of
    `shard_size`. This allows a smooth trade-off between memory usage
    (as in a plain map) vs higher throughput (as in a vmap).

    Args:
        fun: Function to apply smap transform to.
        shard_size: Integer denoting shard size.
        in_axes: Either integer or pytree describing which axis to map over for
            each input to `fun`, None denotes broadcasting.
        out_axes: integer or pytree denoting to what axis in the output the
            mapped over axis maps.

    Returns:
        function with smap applied.
    """
    vmap_ = partial(hk.vmap, split_rng=(not hk.running_init())) if use_hk else jax.vmap
    vmapped_func = vmap_(func, in_axes, out_axes)
    return sharded_apply(vmapped_func, shard_size, in_axes, out_axes, use_hk)


def sharded_apply(
    fun: Callable[..., PYTREE_JAX_ARRAY],
    shard_size: int | None = 1,
    in_axes: int | PYTREE = 0,
    out_axes: int | PYTREE = 0,
    new_out_axes: bool = False,
    use_hk: bool = False,
) -> Callable[..., PYTREE_JAX_ARRAY]:
    """Sharded apply.

    Applies `fun` over shards to axes, in a way similar to vmap,
    but does so in shards of `shard_size`. Shards are stacked after.
    This allows a smooth trade-off between
    memory usage (as in a plain map) vs higher throughput (as in a vmap).

    Args:
        fun: Function to apply smap transform to.
        shard_size: Integer denoting shard size.
        in_axes: Either integer or pytree describing which axis to map over for
            each input to `fun`, None denotes broadcasting.
        out_axes: integer or pytree denoting to what axis in the output the
            mapped over axis maps.
        new_out_axes: whether to stack outputs on new axes. This assumes that
            the output sizes for each shard (including the possible remainder
            shard) are the same.
        use_hk: whether to use jax.vmap or haiku's wrapped vmap which does not
            map module parameters/state.

    Returns:
        function with smap applied.
    """
    docstr = (
        "Mapped version of {fun}. Takes similar arguments to {fun} "
        "but with additional array axes over which {fun} is mapped."
    )
    if use_hk:
        eval_shape_ = hk.eval_shape
        scan_ = hk.scan
    else:
        eval_shape_ = jax.eval_shape
        scan_ = jax.lax.scan

    if new_out_axes:
        raise NotImplementedError("New output axes not yet implemented.")

    # shard size None denotes no sharding
    if shard_size is None:
        return fun

    @jax.util.wraps(fun, docstr=docstr)  # pylint: disable=no-value-for-parameter
    def mapped_fn(*args):
        # Expand in axes and Determine Loop range
        in_axes_ = _expand_axes(in_axes, args)

        in_sizes = jax.tree_util.tree_map(_maybe_get_size, args, in_axes_)
        flat_sizes = jax.tree_util.tree_flatten(in_sizes)[0]
        in_size = max(flat_sizes)
        assert all(i in {in_size, -1} for i in flat_sizes)

        num_extra_shards = (in_size - 1) // shard_size
        # Fix Up if necessary
        last_shard_size = in_size % shard_size
        last_shard_size = shard_size if last_shard_size == 0 else last_shard_size

        def apply_fun_to_slice(slice_start, slice_size):
            input_slice = jax.tree_util.tree_map(
                lambda array, axis: _maybe_slice(array, slice_start, slice_size, axis),
                args,
                in_axes_,
            )
            return fun(*input_slice)

        remainder_shape_dtype = eval_shape_(
            partial(apply_fun_to_slice, 0, last_shard_size)
        )
        out_dtypes = jax.tree_util.tree_map(lambda x: x.dtype, remainder_shape_dtype)
        out_shapes = jax.tree_util.tree_map(lambda x: x.shape, remainder_shape_dtype)
        out_axes_ = _expand_axes(out_axes, remainder_shape_dtype)

        if num_extra_shards > 0:
            regular_shard_shape_dtype = eval_shape_(
                partial(apply_fun_to_slice, 0, shard_size)
            )
            shard_shapes = jax.tree_util.tree_map(
                lambda x: x.shape, regular_shard_shape_dtype
            )

            def make_output_shape(axis, shard_shape, remainder_shape):
                return (
                    shard_shape[:axis]
                    + (shard_shape[axis] * num_extra_shards + remainder_shape[axis],)
                    + shard_shape[axis + 1 :]
                )

            out_shapes = jax.tree_util.tree_map(
                make_output_shape, out_axes_, shard_shapes, out_shapes
            )

        # Calls dynamic Update slice with different argument order
        # This is here since tree_multimap only works with positional arguments
        def dynamic_update_slice_in_dim(full_array, update, axis, i):
            return jax.lax.dynamic_update_slice_in_dim(full_array, update, i, axis)

        def compute_shard(outputs, slice_start, slice_size):
            slice_out = apply_fun_to_slice(slice_start, slice_size)
            update_slice = partial(dynamic_update_slice_in_dim, i=slice_start)
            return jax.tree_util.tree_map(update_slice, outputs, slice_out, out_axes_)

        def scan_iteration(outputs, i):
            new_outputs = compute_shard(outputs, i, shard_size)
            return new_outputs, ()

        slice_starts = jnp.arange(0, in_size - shard_size + 1, shard_size)

        def allocate_buffer(dtype, shape):
            return jnp.zeros(shape, dtype=dtype)

        outputs = jax.tree_util.tree_map(allocate_buffer, out_dtypes, out_shapes)

        if slice_starts.shape[0] > 0:
            outputs, _ = scan_(scan_iteration, outputs, slice_starts)

        if last_shard_size != shard_size:
            remainder_start = in_size - last_shard_size
            outputs = compute_shard(outputs, remainder_start, last_shard_size)

        return outputs

    return mapped_fn


def inference_subbatch(
    module: Callable[..., PYTREE_JAX_ARRAY],
    subbatch_size: int,
    batched_args: Sequence[PYTREE_JAX_ARRAY],
    nonbatched_args: Sequence[PYTREE_JAX_ARRAY],
    low_memory: bool = True,
    input_subbatch_dim: int = 0,
    output_subbatch_dim: Optional[int] = None,
) -> PYTREE_JAX_ARRAY:
    """Run through subbatches (like batch apply but with split and concat)."""
    assert len(batched_args) > 0
    if not low_memory:
        args = list(batched_args) + list(nonbatched_args)
        return module(*args)

    if output_subbatch_dim is None:
        output_subbatch_dim = input_subbatch_dim

    def run_module(*batched_args):
        args = list(batched_args) + list(nonbatched_args)
        return module(*args)

    sharded_module = sharded_apply(
        run_module,
        shard_size=subbatch_size,
        in_axes=input_subbatch_dim,
        out_axes=output_subbatch_dim,
    )
    return sharded_module(*batched_args)
