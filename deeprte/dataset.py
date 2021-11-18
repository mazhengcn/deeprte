from __future__ import annotations

import enum
import pathlib
from collections.abc import Generator, Mapping, Sequence
from typing import Union

import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from numpy.lib.npyio import NpzFile

from deeprte.typing import GraphOfMapping

Batch = Mapping[str, np.ndarray]
AUTOTUNE = tf.data.experimental.AUTOTUNE


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

    total_batch_size = np.prod(batch_dims)

    (ds, grid), _ = _load_and_split_dataset(
        data_dir, split, from_=start, ent=end
    )

    num_total_samples, num_total_pts = np_dataset["psi_label"].shape

    xy, quads, w = (grid["xy"], grid["bc_pts"], grid["bc_w"])

    g = tf.random.Generator.from_seed(0)

    def test_map_fn(batch):

        xycs = mesh["xycs"]

        return {
            "interior": (
                xycs[..., 0:2],
                xycs[..., 2:],
                GraphOfMapping(xy, batch["sigma"]),
                GraphOfMapping(quads, w * batch["bc_psi"]),
            ),
            "label": batch["psi_label"],
        }

    train_ds = tf.data.Dataset.from_tensor_slices(np_train_ds)
    train_ds = (
        train_ds.shuffle(split, reshuffle_each_iteration=True)
        .batch(
            train_batch_size,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        .map(
            train_map_fn,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices(
            tf.nest.map_structure(
                lambda arr: np.take(arr, test_indices, axis=0), np_dataset
            )
        )
        .batch(test_batch_size)
        .map(test_map_fn, num_parallel_calls=AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    init_ds = (
        mesh["xycs"][0, 0:2],
        mesh["xycs"][0, 2:],
        GraphOfMapping(xy, np_dataset["sigma"][0]),
        GraphOfMapping(quads, w * np_dataset["bc_psi"][0]),
    )

    yield from tfds.as_numpy(train_ds)


def residual_points_sampler(batch, g: tf.random.Generator):

    collocation_indices = g.uniform(
        (num_collocation_pts,), minval=0, maxval=num_total_pts, dtype=tf.int32
    )
    xycs = tf.gather(grid["xycs"], collocation_indices, axis=0)

    return {
        "interior": (
            xycs[..., 0:2],
            xycs[..., 2:],
            GraphOfMapping(xy, batch["sigma"]),
            GraphOfMapping(quads, w * batch["bc_psi"]),
        ),
        "label": tf.gather(batch["psi_label"], collocation_indices, axis=1),
    }


def _load_and_split_dataset(
    path_npz: str, split: Split, end: int, from_: int = 0
) -> tuple[tf.data.Dataset, dict[str, np.ndarray]]:

    if not isinstance(path_npz, pathlib.Path):
        path_npz = pathlib.Path(path_npz)

    with tf.io.gfile.GFile(path_npz, "rb") as fp:
        npzfile = np.load(fp, allow_pickle=False)
        data, grid = _preprocess_dataset(npzfile)

    ds = tf.data.Dataset.from_tensor_slices(
        tf.nest.map_structure(lambda arr: arr[from_:end], data)
    )

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


def _convert_dataset(data_path: str) -> dict[str, np.ndarray]:
    """Convert matlab dataset to numpy for use."""

    converted_data = {}
    # Load data, notice that "data_path" should not contain ".npz"
    with np.load(data_path + ".npy", allow_pickle=True) as data:
        # Load reference solutions, boundary functions and sigmas
        phi = data["list_Phi"]  # [B, I, J]
        # print(data["list_Psi"].shape)
        psi = data["list_Psi"]  # [B, I, J, M]
        # print(data["list_psiR"].shape)
        psi_bc = np.swapaxes(data["list_psiR"], -2, -1)  # [B, I'*J', M']

        sigma_T = data["list_sigma_T"]  # [B, I, J]
        sigma_a = data["list_sigma_a"]  # [B, I, J]

        theta = data["theta"].T  # [M, 1]
        omega = data["omega"][0]  # [M]
        c = data["ct"].T  # [M, 1]
        s = data["st"].T  # [M, 1]

    sigma = np.stack([sigma_T, sigma_a], axis=-1)  # [B, I, J, 2]
    converted_data.update(
        {
            "sigma": sigma,
            "psi_bc": psi_bc,
            "psi_label": psi,
            "phi": phi,
            "theta": theta,
            "omega": omega,
            "c": c,
            "s": s,
        }
    )

    # Construct mesh points
    # interior
    _, nx, ny, _ = psi.shape  # [B, I, J, M]
    dx, dy = 1.0 / nx, 1.0 / ny
    x = np.arange(0.0 + 0.5 * dx, 1.0, dx, dtype=np.float32)
    y = np.arange(0.0 + 0.5 * dy, 1.0, dy, dtype=np.float32)
    xy = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)  # [I, J, 2]
    converted_data.update({"xy": xy})

    # on right the boundary
    c_bc, s_bc = c[c < 0], s[c < 0]  # [M', 1]
    ytheta_cbc = np.stack(
        np.meshgrid(y, c_bc, indexing="ij"), axis=-1
    )  # [I'*J', M', 2]
    ytheta_sbc = np.stack(
        np.meshgrid(y, s_bc, indexing="ij"), axis=-1
    )  # [I'*J', M', 2]
    xytheta_bc = np.concatenate(
        (
            np.ones_like(ytheta_cbc[..., 0:1]),
            ytheta_cbc[..., 0:1],
            ytheta_cbc[..., 1:2],
            ytheta_sbc[..., 1:2],
        ),
        axis=-1,
    )  # [I'*J', M', 4]
    weights_bc = (
        2 * (1 / (nx + 1)) * omega[c[:, 0] < 0] * np.ones_like(y)[:, None]
    )  # [I'*J', M']

    converted_data.update({"bc_points": xytheta_bc, "bc_weights": weights_bc})

    # Save converted dataset
    np.savez(data_path + "_converted.npz", **converted_data)

    return converted_data


def _preprocess_dataset(
    data_dict: Union[dict[str, np.ndarray], NpzFile]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:

    print("Prepare numpy arrays for feeding into neural network.")
    print(
        f"Origin shapes: {tf.nest.map_structure(lambda x: x.shape, data_dict)}"
    )

    ndim, nsigma = 2, 2
    nsample = data_dict["psi_label"].shape[0]

    dataset = {}
    # Data
    dataset["psi_label"] = data_dict["psi_label"].reshape(nsample, -1)
    dataset["sigma"] = data_dict["sigma"].reshape(nsample, -1, nsigma)
    dataset["phi"] = data_dict["phi"].reshape(nsample, -1)

    # Grid
    grid = {}
    xy, cs = data_dict["xy"], np.concatenate(
        [data_dict["c"], data_dict["s"]], axis=-1
    )
    print(f"cs shape: {cs.shape}")
    grid["xy"] = np.reshape(xy, [-1, ndim])
    grid["xycs"] = np.concatenate(
        (xy[:, :, None] + 0.0 * cs[:, 0:1], cs + 0.0 * xy[:, :, None, 0:1]),
        axis=-1,
    ).reshape(-1, ndim * 2)
    grid["omega"] = data_dict["omega"]

    # Boundary
    data["bc_psi"] = data_dict["psi_bc"].reshape(nsample, -1)
    grid["bc_pts"] = data_dict["bc_points"].reshape(-1, ndim * 2)
    grid["bc_w"] = data_dict["bc_weights"].flatten()

    print(
        "Dataset is ready, "
        f"shapes are: {tf.nest.map_structure(lambda x: x.shape, dataset)}"
    )

    return (dataset, grid)
