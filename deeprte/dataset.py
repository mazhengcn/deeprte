# Copyright 2022 Zheng Ma
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import enum
import pathlib
from collections.abc import Generator, Mapping, Sequence

import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging

from deeprte.model.modules import F
from deeprte.utils import flat_dict_to_rte_data, to_flat_dict

Batch = Mapping[str, np.ndarray]
AUTOTUNE = tf.data.AUTOTUNE


def log_shapes(d: dict, name: str):
    logs = ""
    for k, v in get_nest_dict_shape(d).items():
        logs += f", {k:s}: {v}"

    logging.info(f"{name} shapes" + logs)


def get_nest_dict_shape(d):
    return tf.nest.map_structure(lambda x: x.shape, d)


class Split(enum.Enum):
    """Datset split."""

    TRAIN = 1
    TRAIN_AND_VALID = 2
    VALID = 3
    TEST = 4

    @classmethod
    def from_string(cls, name: str) -> "Split":
        return {
            "TRAIN": Split.TRAIN,
            "TRAIN_AND_VALID": Split.TRAIN_AND_VALID,
            "VALID": Split.VALID,
            "VALIDATION": Split.VALID,
            "TEST": Split.TEST,
        }[name.upper()]

    @property
    def num_examples(self):
        return {
            Split.TRAIN: 300,
            Split.TRAIN_AND_VALID: 400,
            Split.VALID: 100,
            Split.TEST: 100,
        }[self]


def load(
    data_path: str | pathlib.Path,
    split: Split,
    is_training: bool,
    # batch_sizes should be:
    # [device_count, per_device_outer_batch_size]
    # total_batch_size = device_count * per_device_outer_batch_size
    batch_sizes: Sequence[int],
    # collocation_sizes should be:
    # total_collocation_size or
    # [residual_size, boundary_size, quadrature_size]
    collocation_sizes: int | Sequence[int] | None,
    # repeat number of inner batch, for training the same batch with
    # {repeat} steps of different collocation points
    repeat: int | None = 1,
    # shuffle buffer size
    buffer_size: int = 5_000,
    # Dataset options
    threadpool_size: int = 48,
    max_intra_op_parallelism: int = 1,
) -> Generator[Batch, None, None]:

    if is_training:
        if not collocation_sizes and not repeat:
            raise ValueError(
                "`collocation_sizes` and `repeat` should not be None"
                "when `is_training=True`"
            )

    start, end = _shard(split, jax.process_index(), jax.process_count())

    # total_batch_size = np.prod(batch_dims)

    (ds, (grid, total_grid_sizes)), _ = _load_and_split_dataset(
        data_path, split, from_=start, end=end
    )

    options = tf.data.Options()
    options.threading.max_intra_op_parallelism = max_intra_op_parallelism
    options.threading.private_threadpool_size = threadpool_size
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    if is_training:
        options.deterministic = False

    if is_training:
        if jax.process_count() > 1:
            # Only cache if we are reading a subset of the dataset.
            ds = ds.cache()
        ds = ds.repeat()
        ds = ds.shuffle(buffer_size=buffer_size)
        ds = _repeat_batch(batch_sizes, ds, repeat)

    # batch per_device outer first,
    # since they share the same random grid points
    ds = ds.batch(batch_sizes[-1], drop_remainder=True)
    # construct the inputs structure
    ds = process_inputs(ds, grid)
    # batch device dim
    ds = ds.batch(batch_sizes[0], drop_remainder=True)

    if is_training:
        ds = sample_from_dataset(ds, collocation_sizes, total_grid_sizes)

    ds = ds.prefetch(AUTOTUNE)
    ds = ds.with_options(options)

    # convert to a numpy generator
    yield from tfds.as_numpy(ds)


def sample_from_dataset(
    dataset: tf.data.Dataset,
    collocation_sizes: int | Sequence[int],
    total_grid_sizes: int | Sequence[int],
    sampler: str = "uniform",
    seed: int = jax.process_index(),
):

    g = tf.random.Generator.from_seed(seed)

    if sampler == "uniform":

        def _sample_fn(_):
            idx = g.uniform(
                (collocation_sizes,),
                minval=0,
                maxval=total_grid_sizes,
                dtype=tf.int64,
            )
            return idx

    else:
        raise ValueError(
            f"Sample from {sampler} distribution is not implemented."
        )

    # generate random sample indices
    indices_ds = tf.data.Dataset.range(1).repeat()
    indices_ds = indices_ds.map(_sample_fn, num_parallel_calls=AUTOTUNE)
    ds = slice_inputs(indices_ds, dataset)

    return ds


def process_inputs(data: tf.data.Dataset, grid: Mapping[str, np.ndarray]):

    ds = tf.data.Dataset.zip(
        (data, tf.data.Dataset.from_tensors(grid).repeat())
    )

    def _construct_batch(data, grid):

        sigma = tf.stack([data["sigma_t"], data["sigma_a"]], axis=-1)
        psi_bc = data["psi_bc"]
        psi_label = data["psi_label"]

        r, rv = grid["r"], grid["rv"]
        rv_prime, w_prime = grid["rv_prime"], grid["w_prime"]
        rv_r, rv_v = tf.split(rv, num_or_size_splits=2, axis=-1)

        return {
            "inputs": (
                rv_r,
                rv_v,
                F(x=r, y=sigma),
                F(x=rv_prime, y=psi_bc * w_prime),
            ),
            "labels": psi_label,
        }

    ds = ds.map(_construct_batch, num_parallel_calls=AUTOTUNE)

    return ds


def slice_inputs(indices_dataset: tf.data.Dataset, inputs: tf.data.Dataset):

    dataset = tf.data.Dataset.zip((indices_dataset, inputs))

    def grab_batch(i, data):
        batch = dict(**data)
        rv_r, rv_v = tf.nest.map_structure(
            lambda x: tf.gather(x, i, axis=-2), data["inputs"][:2]
        )
        batch["inputs"] = (rv_r, rv_v) + data["inputs"][2:]
        batch["labels"] = tf.gather(data["labels"], i, axis=-1)
        return batch

    dataset = dataset.map(grab_batch, num_parallel_calls=AUTOTUNE)

    return dataset


def _repeat_batch(
    batch_sizes: int | Sequence[int],
    ds: tf.data.Dataset,
    repeat: int = 1,
) -> tf.data.Dataset:
    """Tiles the inner most batch dimension."""
    if repeat <= 1:
        return ds
    # Perform regular batching with reduced number of elements.
    for batch_size in reversed(batch_sizes):
        ds = ds.batch(batch_size, drop_remainder=True)

    # Repeat batch.
    fn = lambda x: tf.tile(x, multiples=[repeat] + [1] * (len(x.shape) - 1))

    def repeat_inner_batch(example):
        return tf.nest.map_structure(fn, example)

    ds = ds.map(repeat_inner_batch, num_parallel_calls=tf.data.AUTOTUNE)
    # Unbatch.
    for _ in batch_sizes:
        ds = ds.unbatch()
    return ds


def _load_and_split_dataset(
    path_npz: str | pathlib.Path, split: Split, end: int, from_: int = 0
) -> tuple[tuple[tf.data.Dataset, Mapping[str, np.ndarray]], Split]:

    if not isinstance(path_npz, pathlib.Path):
        path_npz = pathlib.Path(path_npz)

    with tf.io.gfile.GFile(path_npz, "rb") as fp:
        npzfile = np.load(fp, allow_pickle=False)
        rte_data = flat_dict_to_rte_data(npzfile)
        data, grid = rte_data["data"], rte_data["grid"]

    # log_shapes(data, "Data")

    def _flatten_fn(example):
        return tf.nest.map_structure(lambda x: tf.reshape(x, [-1]), example)

    ds = tf.data.Dataset.from_tensor_slices(
        tf.nest.map_structure(lambda arr: arr[from_:end], data)
    ).map(_flatten_fn, num_parallel_calls=AUTOTUNE)

    # log_shapes(grid, "Grid")

    grid = preprocess_grid(grid)

    return (ds, grid), split


def _shard(split: Split, shard_index: int, num_shards: int) -> tuple[int, int]:
    """Returns [start, end) for the given shard index."""
    assert shard_index < num_shards
    arange = np.arange(split.num_examples)
    shard_range = np.array_split(arange, num_shards)[shard_index]
    start, end = shard_range[0], (shard_range[-1] + 1)
    if split == Split.TRAIN_AND_VALID:
        offset = Split.TEST.num_examples
        start += offset
        end += offset
    return start, end


def preprocess_grid(
    grid: Mapping[str, np.ndarray], is_training: bool = True
) -> Mapping[str, np.ndarray]:
    r, v = grid["r"], grid["v"]
    r = r.reshape(-1, r.shape[-1])
    grid["r"] = r

    rv = np.concatenate((r[:, None] + 0.0 * v, v + 0.0 * r[:, None]), axis=-1)
    rv = rv.reshape(-1, rv.shape[-1])
    total_grid_size = rv.shape[0]
    grid["rv"] = rv

    rv_prime, w_prime = grid["rv_prime"], grid["w_prime"]
    grid["rv_prime"] = rv_prime.reshape(-1, rv_prime.shape[-1])
    grid["w_prime"] = w_prime.flatten()

    if is_training:
        del grid["w_angle"], grid["v"]

    return grid, total_grid_size


def unstack_np_array(arr, axis=0):
    return list(np.swapaxes(arr, axis, 0))


def convert_dataset(
    data_path: str, save_npz: bool = True
) -> Mapping[str, Mapping[str, np.ndarray]]:
    """Convert matlab dataset to numpy for use."""

    if not isinstance(data_path, pathlib.Path):
        data_path = pathlib.Path(data_path)

    data = np.load(data_path, allow_pickle=True)

    if data_path.suffix == ".npy":
        data = data.item()
    # Load reference solutions, boundary functions and sigmas
    phi = data["list_Phi"]  # [B, I, J]
    # print(data["list_Psi"].shape)
    psi = data["list_Psi"]  # [B, I, J, M]
    # print(data["list_psiR"].shape)
    psi_bc = np.swapaxes(data["list_psiL"], -2, -1)  # [B, I'*J', M']

    sigma_t = data["list_sigma_T"]  # [B, I, J]
    sigma_a = data["list_sigma_a"]  # [B, I, J]

    # theta = data["theta"].T  # [M, 1]
    w_angle = data["omega"][0]  # [M]
    vx = data["ct"].T  # [M, 1]
    vy = data["st"].T  # [M, 1]
    v = np.concatenate([vx, vy], axis=-1)

    converted_data = {"data": {}, "grid": {}}

    # sigma = np.stack([sigma_t, sigma_a], axis=-1)  # [B, I, J, 2]
    converted_data["data"].update(
        {
            "sigma_t": sigma_t,
            "sigma_a": sigma_a,
            "psi_bc": psi_bc,
            "psi_label": psi,
            "phi": phi,
        }
    )
    converted_data["grid"].update({"w_angle": w_angle, "v": v})

    # Construct grid points
    # interior
    _, nx, ny, _ = psi.shape  # [B, I, J, M]
    dx, dy = 1.0 / nx, 1.0 / ny
    x = np.arange(0.0 + 0.5 * dx, 1.0, dx, dtype=np.float32)
    y = np.arange(0.0 + 0.5 * dy, 1.0, dy, dtype=np.float32)
    r = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)  # [I, J, 2]
    converted_data["grid"].update({"r": r})

    # on left the boundary
    vx_prime, vy_prime = vx[vx > 0], vy[vx > 0]  # [M', 1]
    rvx_prime = np.stack(
        np.meshgrid(y, vx_prime, indexing="ij"), axis=-1
    )  # [I'*J', M', 2]
    rvy_prime = np.stack(
        np.meshgrid(y, vy_prime, indexing="ij"), axis=-1
    )  # [I'*J', M', 2]
    rv_prime = np.concatenate(
        (
            np.zeros_like(rvx_prime[..., 0:1]),
            rvx_prime[..., 0:1],
            rvx_prime[..., 1:2],
            rvy_prime[..., 1:2],
        ),
        axis=-1,
    )  # [I'*J', M', 4]
    w_prime = (
        2 * (1 / (nx + 1)) * w_angle[vx[:, 0] > 0] * np.ones_like(y)[:, None]
    )  # [I'*J', M']

    converted_data["grid"].update({"rv_prime": rv_prime, "w_prime": w_prime})

    # Save converted dataset
    if save_npz:
        npz_path = data_path.with_name(data_path.stem + "_converted")
        np.savez(
            npz_path,
            **to_flat_dict(converted_data, sep="/"),
        )
        logging.info(f"Saved converted dataset {npz_path}.")

    return converted_data
