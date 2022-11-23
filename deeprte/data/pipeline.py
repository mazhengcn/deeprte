"""Convert Matlab dataset to numpy dataset."""

from collections.abc import Mapping, MutableMapping

import numpy as np
import scipy.io as sio
import pathlib

from deeprte.data.utils import cartesian_product_nd

Float = float | np.float32

DIMENSIONS = 2
FeatureDict = MutableMapping[str, np.ndarray]


def make_data_features(np_data: Mapping[str, np.ndarray]) -> FeatureDict:
    """Convert numpy data dict to unified numpy data dict for rectangle domain."""
    # Load reference solutions and sigmas
    phi = np_data["phi"]  # [B, I, J]
    psi = np_data["psi_label"]  # [B, I, J, M]
    sigma_t = np_data["sigma_t"]  # [B, I, J]
    sigma_a = np_data["sigma_a"]  # [B, I, J]
    psi_bc = np_data["psi_bc"]  # [B, 2*(I+J), 4]

    sigma = np.stack([sigma_t, sigma_a], axis=-1)

    features = {
        "sigma": sigma,
        "psi_label": psi,
        "phi": phi,
        "boundary": psi_bc,
    }
    return features


def make_grid_features(np_data: Mapping[str, np.ndarray]) -> FeatureDict:
    """Convert numpy grid dict and preprocess grid points."""

    features = {}

    vx, vy = np_data["ct"], np_data["st"]  # [M, 1]
    v_coords = np.concatenate([vx, vy], axis=-1)
    x, y = np.squeeze(np_data["x"]), np.squeeze(np_data["y"])

    features["position_coords"] = cartesian_product_nd(
        np.expand_dims(x, axis=-1),
        np.expand_dims(y, axis=-1),
    )
    features["velocity_coords"] = v_coords

    rv = cartesian_product_nd(
        np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1), v_coords
    )
    features["phase_coords"] = rv

    # v_star = np.concatenate([np_data["ct"], np_data["st"]], axis=-1)

    vv_star = cartesian_product_nd(
        np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1), v_coords, v_coords
    )
    # features["scattering_velocity_coords"] = vv_star[..., 2:]

    scattering_kernel_value = np.tile(
        np_data["scattering_kernel"], (1, rv.shape[0] * rv.shape[1], 1)
    )
    features["scattering_kernel"] = scattering_kernel_value.reshape(
        *(scattering_kernel_value.shape[0:1] + vv_star.shape[:-1])
    )

    rv_prime, w_prime = np_data["rv_prime"], np_data["omega_prime"]
    features["boundary_coords"] = rv_prime
    features["boundary_weights"] = w_prime

    features["velocity_weights"] = np.squeeze(np_data["w_angle"])

    return features


def make_shape_dict(np_data: Mapping[str, np.ndarray]) -> Mapping[str, int]:
    shape_dict = {}
    shape_dict["num_x"] = np.shape(np.squeeze(np_data["x"]))[0]
    shape_dict["num_y"] = np.shape(np.squeeze(np_data["y"]))[0]
    shape_dict["num_v"] = np.shape(np.squeeze(np_data["ct"]))[0]
    shape_dict["num_samples"] = np.shape(np_data["psi_label"])[0]

    return shape_dict


class DataPipeline:
    def __init__(
        self,
    ):
        pass

    def load_data(
        self,
        data_path: str,
    ):
        if not isinstance(data_path, pathlib.Path):
            data_path = pathlib.Path(data_path)

        if data_path.suffix == ".mat":
            data = sio.loadmat(data_path)

        elif data_path.suffix == ".npy":
            data = np.load(data_path, allow_pickle=True)
            data = data.item()
        else:
            data = np.load(data_path, allow_pickle=True)

        return data

    def process(self, data_path: str) -> FeatureDict:

        data = self.load_data(data_path)

        data_feature = make_data_features(data)
        grid_feature = make_grid_features(data)
        shape_dict = make_shape_dict(data)

        return {**data_feature, **grid_feature, **shape_dict}
