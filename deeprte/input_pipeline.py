from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import tensorflow as tf

from deeprte.typing import GraphOfMapping

FeatureDict = Mapping[str, np.ndarray]


def get_numpy_dataset(data_path: str) -> dict[str, np.ndarray]:
    """Convert matlab dataset to numpy for use."""

    # Load data, notice that "data_path" should not contain ".npz"
    data = np.load(data_path + ".npy", allow_pickle=True).item()
    converted_data = {}

    # Load reference solutions, boundary functions and sigmas
    phi = data["list_Phi"]  # [B, I, J]
    # print(data["list_Psi"].shape)
    psi = data["list_Psi"]  # [B, I, J, M]
    # print(data["list_psiR"].shape)
    psi_bc = np.swapaxes(data["list_psiR"], -2, -1)  # [B, I'*J', M']

    sigma_T = data["list_sigma_T"]  # [B, I, J]
    sigma_a = data["list_sigma_a"]  # [B, I, J]
    sigma = np.stack([sigma_T, sigma_a], axis=-1)  # [B, I, J, 2]

    converted_data.update(
        {"sigma": sigma, "psi_bc": psi_bc, "psi_label": psi, "phi": phi}
    )

    # Construct mesh points
    # interior
    _, nx, ny, _ = psi.shape  # [B, I, J, M]
    dx, dy = 1.0 / nx, 1.0 / ny
    x = np.arange(0.0 + 0.5 * dx, 1.0, dx, dtype=np.float32)
    y = np.arange(0.0 + 0.5 * dy, 1.0, dy, dtype=np.float32)
    xy = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)  # [I, J, 2]

    theta = data["theta"].T  # [M, 1]
    omega = data["omega"][0]  # [M]
    c = data["ct"].T  # [M, 1]
    s = data["st"].T  # [M, 1]

    converted_data.update(
        {"xy": xy, "theta": theta, "omega": omega, "c": c, "s": s}
    )

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


def prepare_np_dataset(
    data_dict: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:

    if not isinstance(data_dict, dict):
        data_dict = dict(data_dict)
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

    # Mesh
    mesh = {}
    xy, cs = data_dict["xy"], np.concatenate(
        [data_dict["c"], data_dict["s"]], axis=-1
    )
    print(f"cs shape: {cs.shape}")
    mesh["xy"] = np.reshape(xy, [-1, ndim])
    mesh["xycs"] = np.concatenate(
        (xy[:, :, None] + 0.0 * cs[:, 0:1], cs + 0.0 * xy[:, :, None, 0:1]),
        axis=-1,
    ).reshape(-1, ndim * 2)
    mesh["omega"] = data_dict["omega"]

    # Boundary
    dataset["bc_psi"] = data_dict["psi_bc"].reshape(nsample, -1)
    mesh["bc_pts"] = data_dict["bc_points"].reshape(-1, ndim * 2)
    mesh["bc_w"] = data_dict["bc_weights"].flatten()

    print(
        "Dataset is ready, "
        f"shapes are: {tf.nest.map_structure(lambda x: x.shape, dataset)}"
    )

    return dataset, mesh


def create_tf_dataset(
    data_path: str,
    train_batch_size: int,
    num_collocation_pts: int = 500,
    test_batch_size=1,
):

    with np.load(data_path + ".npz") as np_data:
        np_dataset, mesh = prepare_np_dataset(np_data)

    num_total_samples, num_total_pts = np_dataset["psi_label"].shape

    xy, quads, w = (
        mesh["xy"],
        mesh["bc_pts"],
        mesh["bc_w"],
    )

    # shuffle and split
    split = int(num_total_samples * 0.8)
    np_rng = np.random.default_rng(12345)
    dataset_indices = np_rng.permutation(num_total_samples)
    train_indices, test_indices = (
        dataset_indices[:split],
        dataset_indices[split:],
    )

    np_train_ds = tf.nest.map_structure(
        lambda arr: np.take(arr, train_indices, axis=0), np_dataset
    )

    g = tf.random.Generator.from_seed(0)

    def train_map_fn(batch):

        collocation_indices = g.uniform(
            (num_collocation_pts,),
            minval=0,
            maxval=num_total_pts,
            dtype=tf.int32,
        )
        xycs = tf.gather(mesh["xycs"], collocation_indices, axis=0)

        return {
            "interior": (
                xycs[..., 0:2],
                xycs[..., 2:],
                GraphOfMapping(xy, batch["sigma"]),
                GraphOfMapping(quads, w * batch["bc_psi"]),
            ),
            "label": tf.gather(
                batch["psi_label"], collocation_indices, axis=1
            ),
        }

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
        .map(test_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    init_ds = (
        mesh["xycs"][0, 0:2],
        mesh["xycs"][0, 2:],
        GraphOfMapping(xy, np_dataset["sigma"][0]),
        GraphOfMapping(quads, w * np_dataset["bc_psi"][0]),
    )

    return train_ds, test_ds, init_ds


if __name__ == "__main__":

    ROOT_PATH = "/workspace/modnet/"
    _ = get_numpy_dataset(ROOT_PATH + "data/rte/rte_2d")

    data = np.load(ROOT_PATH + "data/rte/rte_2d_converted.npz")
    print(tf.nest.map_structure(lambda x: x.shape, dict(data.items())))
