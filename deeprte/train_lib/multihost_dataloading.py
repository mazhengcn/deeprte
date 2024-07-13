import collections
import itertools
import time
from collections.abc import Iterable, Iterator
from functools import partial  # pylint: disable=g-importing-member
from typing import Any

import jax
import jax.tree_util as jtu
import numpy as np
import tensorflow as tf
from absl import logging
from clu.data import dataset_iterator
from jax.sharding import Mesh, NamedSharding, PartitionSpec

PyTree = Any

Dtype = Any
Shape = tuple[int, ...]


def get_dataset_shape_dtype_struct(
    iterator: tf.data.Dataset | dataset_iterator.DatasetIterator,
    global_mesh: Mesh,
    data_pspec: PartitionSpec,
) -> PyTree:
    """Returns the jax.ShapeDtypeStruct."""

    sharding = NamedSharding(global_mesh, data_pspec)

    def fn(x):
        # Dtype and local shape (of this particular process) of the given array x.
        shape, dtype = x.shape, x.dtype
        dtype = dtype.as_numpy_dtype if hasattr(dtype, "as_numpy_dtype") else dtype
        # Global shape
        shape = (shape[0] * jax.process_count(),) + shape[1:]
        # Return a ShapeDtypeStruct with the global shape and sharding.
        return jax.ShapeDtypeStruct(shape=shape, dtype=dtype, sharding=sharding)

    return jax.tree.map(fn, iterator.element_spec)


def _build_global_shape_and_sharding(
    local_shape: tuple[int, ...], global_mesh: Mesh, data_pspec: PartitionSpec
) -> tuple[tuple[int, ...], NamedSharding]:
    sharding = NamedSharding(global_mesh, data_pspec)
    global_shape = (jax.process_count() * local_shape[0],) + local_shape[1:]

    return global_shape, sharding


def _form_global_array(
    path, array: np.ndarray, global_mesh: Mesh, data_pspec: PartitionSpec
) -> jax.Array:
    """Put local sharded array into local devices"""
    global_shape, sharding = _build_global_shape_and_sharding(
        np.shape(array), global_mesh, data_pspec
    )

    try:
        local_device_arrays = np.split(array, len(global_mesh.local_devices), axis=0)
    except ValueError as array_split_error:
        raise ValueError(
            f"Unable to put to devices shape {array.shape} with "
            f"local device count {len(global_mesh.local_devices)} "
            f"at {jtu.keystr(path)}"
        ) from array_split_error

    local_device_buffers = jax.device_put(
        local_device_arrays, global_mesh.local_devices
    )
    return jax.make_array_from_single_device_arrays(
        global_shape, sharding, local_device_buffers
    )


def get_next_batch_sharded(
    local_iterator: Iterator, global_mesh: Mesh, data_pspec: PartitionSpec
) -> jax.Array:
    """Splits the host loaded data equally over all devices."""

    SLEEP_TIME = 10
    MAX_DATA_LOAD_ATTEMPTS = 30

    data_load_attempts = 0
    loaded_data_success = False
    while not loaded_data_success and data_load_attempts < MAX_DATA_LOAD_ATTEMPTS:
        data_load_attempts += 1
        try:
            local_data = next(local_iterator)
            loaded_data_success = True
        except tf.errors.FailedPreconditionError:
            logging.log("Failed to get next data batch, retrying")
            time.sleep(SLEEP_TIME)

    # Try one last time, if this fails we will see the full stack trace.
    if not loaded_data_success:
        local_data = next(local_iterator)

    input_gdas = jtu.tree_map_with_path(
        partial(_form_global_array, global_mesh=global_mesh, data_pspec=data_pspec),
        local_data,
    )

    return input_gdas


class MultiHostDataLoadIterator:
    """fold get_next_batch_sharded into a iterator class"""

    def __init__(
        self,
        dataloader: tf.data.Dataset,
        global_mesh: Mesh,
        data_pspec: PartitionSpec | None = None,
    ):
        self.global_mesh = global_mesh
        self.dataloader = dataloader
        if data_pspec:
            self.data_pspec = data_pspec
        else:
            self.data_pspec = PartitionSpec(global_mesh.axis_names)
        if isinstance(self.dataloader, tf.data.Dataset):
            self.local_iterator = self.dataloader.as_numpy_iterator()
        elif isinstance(self.dataloader, Iterable):
            self.local_iterator = iter(self.dataloader)
        else:
            raise ValueError(
                "Type error: dataloader should be either tf.data.Dataset or Iterable."
            )

    def reset(self):
        if isinstance(self.dataloader, tf.data.Dataset):
            self.local_iterator = self.dataloader.as_numpy_iterator()
        elif isinstance(self.dataloader, Iterable):
            self.local_iterator = iter(self.dataloader)
        else:
            raise ValueError(
                "Type error: dataloader should be either tf.data.Dataset or grain.DataLoader."
            )

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        return get_next_batch_sharded(
            self.local_iterator, self.global_mesh, self.data_pspec
        )


def prefetch_to_device(iterator: MultiHostDataLoadIterator, size: int):
    """Prefetches data to the devices in the global mesh."""

    if size and size > 0:
        # We fill items to this queue, and pop from it when a new item is yielded.
        queue = collections.deque()

        def enqueue(n):
            for data in itertools.islice(iterator, n):
                queue.append(data)

        enqueue(size)
        while queue:
            yield queue.popleft()
            enqueue(1)
    else:
        # If size is None, 0 or negative, simply create jax.Arrays without
        # prefetching.
        for data in iterator:
            yield data
