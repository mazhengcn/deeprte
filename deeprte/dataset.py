from __future__ import annotations

import enum
import pathlib
from collections.abc import Generator, Mapping, Sequence

import jax
import numpy as np
import tensorflow as tf

from deeprte.typing import F
from deeprte.utils import flat_dict_to_rte_data, to_flat_dict

# from typing import Union


# import tensorflow_datasets as tfds
# from numpy.lib.npyio import NpzFile


Batch = Mapping[str, np.ndarray]
AUTOTUNE = tf.data.AUTOTUNE


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
            Split.TRAIN_AND_VALID: 400,
            Split.TRAIN: 300,
            Split.VALID: 100,
            Split.TEST: 100,
        }[self]


def load(
    split: Split,
    *,
    is_training: bool,
    # batch_dims should be:
    # [device_count, per_device_outer_batch_size, perdevice_inner_batch_size]
    # or [total_batch_size]
    batch_dims: Sequence[int],
    data_dir: str,
    threadpool_size: int = 48,
    max_intra_op_parallelism: int = 1,
) -> Generator[Batch, None, None]:

    start, end = _shard(split, jax.process_index(), jax.process_count())

    # total_batch_size = np.prod(batch_dims)

    (ds, (grid, num_grid_points)), _ = _load_and_split_dataset(
        data_dir, split, from_=start, end=end
    )

    # options = ds.options()
    # options.experimental_threading.private_threadpool_size = 48
    # options.experimental_threading.max_intra_op_parallelism = 1
    # options.experimental_optimization.map_parallelization = True
    # options.experimental_optimization.parallel_batch = True
    # options.experimental_optimization.hoist_random_uniform = True

    # if is_training:
    #     options.experimental_deterministic = False

    ds = ds.cache()
    if is_training:
        ds = ds.repeat()
        ds = ds.shuffle(buffer_size=500, seed=jax.process_index())

        # batch per_device outer first,
        # since they share the same random grid points
        ds = ds.batch(batch_dims[-2])
        # zip with grid ds
        ds = tf.data.Dataset.zip(
            (ds, tf.data.Dataset.from_tensors(grid).repeat())
        )
        # batch device dim
        ds = ds.batch(batch_dims[0])

        # generate random sample indices
        gen = tf.random.Generator.from_seed(seed=jax.process_index())
        indices_ds = tf.data.Dataset.range(1).repeat()
        indices_ds = indices_ds.map(
            lambda _: gen.uniform(
                (batch_dims[-1],),
                minval=0,
                maxval=num_grid_points,
                dtype=tf.int32,
            ),
            num_parallel_calls=AUTOTUNE,
            deterministic=False,
        )

        ds = slice_inputs(indices_ds, ds)

    return ds.prefetch(AUTOTUNE)


def slice_inputs(indices_dataset, inputs):
    """Slice inputs into a Dataset of batches.
    Given a Dataset of batch indices and the unsliced inputs,
    this step slices the inputs in a parallelized fashion
    and produces a dataset of input batches.
    Args:
      indices_dataset: A Dataset of batched indices
      inputs: A python data structure or a single element Dataset
        that contains the inputs, targets, and possibly sample weights.
    Returns:
      A Dataset of input batches matching the batch indices.
    """

    dataset = tf.data.Dataset.zip((indices_dataset, inputs))

    def grab_batch(i, data):
        batch, grid = data

        batch_sigma = tf.stack([batch["sigma_t"], batch["sigma_a"]], axis=-1)
        batch_psi_bc = batch["psi_bc"]
        batch_psi_label = tf.gather(batch["psi_label"], i, axis=-1)

        r, rv = grid["r"], grid["rv"]
        rv_prime, w_prime = grid["rv_prime"], grid["w_prime"]
        batch_rv = tf.gather(rv, i, axis=-2)
        batch_r, batch_v = tf.split(batch_rv, num_or_size_splits=2, axis=-1)

        inputs = {
            "interior": (
                batch_r,
                batch_v,
                F(x=r, y=batch_sigma),
                F(x=rv_prime, y=batch_psi_bc * w_prime[:, None]),
            ),
            "label": batch_psi_label,
        }

        return inputs

    dataset = dataset.map(grab_batch, num_parallel_calls=AUTOTUNE)

    # Default optimizations are disabled to avoid the overhead of (unnecessary)
    # input pipeline graph serialization and deserialization
    options = tf.data.Options()
    options.experimental_optimization.apply_default_optimizations = False
    # if self._shuffle:
    #     # See b/141490660 for more details.
    #     options.experimental_external_state_policy = (
    #         tf.data.experimental.ExternalStatePolicy.IGNORE
    #     )
    dataset = dataset.with_options(options)

    return dataset


def _load_and_split_dataset(
    path_npz: str, split: Split, end: int, from_: int = 0
) -> tuple[tuple[tf.data.Dataset, tf.data.Dataset], Split]:

    if not isinstance(path_npz, pathlib.Path):
        path_npz = pathlib.Path(path_npz)

    with tf.io.gfile.GFile(path_npz, "rb") as fp:
        npzfile = np.load(fp, allow_pickle=False)
        rte_data = flat_dict_to_rte_data(npzfile)
        data, grid = rte_data["data"], rte_data["grid"]

    print(f"Processing data, shapes are: {get_nest_dict_shape(data)}")

    def _flatten_fn(example):
        return tf.nest.map_structure(lambda x: tf.reshape(x, [-1]), example)

    ds = tf.data.Dataset.from_tensor_slices(
        tf.nest.map_structure(lambda arr: arr[from_:end], data)
    ).map(_flatten_fn, num_parallel_calls=AUTOTUNE, deterministic=False)

    print(f"Processing grid, shapes are: {get_nest_dict_shape(grid)}")

    grid = _preprocess_grid(grid)

    return (ds, grid), split


def _shard(split: Split, shard_index: int, num_shards: int) -> tuple[int, int]:
    """Returns [start, end) for the given shard index."""
    assert shard_index < num_shards
    arange = np.arange(split.num_examples)
    shard_range = np.array_split(arange, num_shards)[shard_index]
    start, end = shard_range[0], (shard_range[-1] + 1)
    if split == Split.TRAIN:
        # Note that our TRAIN=TFDS_TRAIN[10000:] and VALID=TFDS_TRAIN[:10000].
        offset = Split.VALID.num_examples
        start += offset
        end += offset
    return start, end


def _preprocess_grid(
    grid: Mapping[str, np.ndarray]
) -> Mapping[str, np.ndarray]:
    r, v = grid["r"], grid["v"]
    r = r.reshape(-1, r.shape[-1])
    grid["r"] = r

    rv = np.concatenate((r[:, None] + 0.0 * v, v + 0.0 * r[:, None]), axis=-1)
    rv = rv.reshape(-1, rv.shape[-1])
    num_grid_points = rv.shape[0]
    grid["rv"] = rv

    rv_prime, w_prime = grid["rv_prime"], grid["w_prime"]
    grid["rv_prime"] = rv_prime.reshape(-1, rv_prime.shape[-1])
    grid["w_prime"] = w_prime.flatten()

    del grid["w_angle"], grid["v"]

    return grid, num_grid_points


def unstack_np_array(arr, axis=0):
    return list(np.swapaxes(arr, axis, 0))


def convert_dataset(data_path: str) -> dict[str, np.ndarray]:
    """Convert matlab dataset to numpy for use."""

    if not isinstance(data_path, pathlib.Path):
        data_path = pathlib.Path(data_path)

    data = np.load(data_path, allow_pickle=True).item()
    # Load reference solutions, boundary functions and sigmas
    phi = data["list_Phi"]  # [B, I, J]
    # print(data["list_Psi"].shape)
    psi = data["list_Psi"]  # [B, I, J, M]
    # print(data["list_psiR"].shape)
    psi_bc = np.swapaxes(data["list_psiR"], -2, -1)  # [B, I'*J', M']

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

    # on right the boundary
    vx_prime, vy_prime = vx[vx < 0], vy[vx < 0]  # [M', 1]
    rvx_prime = np.stack(
        np.meshgrid(y, vx_prime, indexing="ij"), axis=-1
    )  # [I'*J', M', 2]
    rvy_prime = np.stack(
        np.meshgrid(y, vy_prime, indexing="ij"), axis=-1
    )  # [I'*J', M', 2]
    rv_prime = np.concatenate(
        (
            np.ones_like(rvx_prime[..., 0:1]),
            rvx_prime[..., 0:1],
            rvx_prime[..., 1:2],
            rvy_prime[..., 1:2],
        ),
        axis=-1,
    )  # [I'*J', M', 4]
    w_prime = (
        2 * (1 / (nx + 1)) * w_angle[vx[:, 0] < 0] * np.ones_like(y)[:, None]
    )  # [I'*J', M']

    converted_data["grid"].update({"rv_prime": rv_prime, "w_prime": w_prime})

    # Save converted dataset
    np.savez(
        data_path.with_name(data_path.stem + "_converted"),
        **to_flat_dict(converted_data, sep="/"),
    )

    return converted_data
