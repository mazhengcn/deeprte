"""Convert Matlab dataset to numpy dataset."""

from collections.abc import Mapping, MutableMapping
from typing import Optional

import numpy as np

from deeprte.data import utils
from deeprte.data.tool import matlab

FeatureDict = MutableMapping[str, np.ndarray]


def make_data_features(np_data: Mapping[str, np.ndarray]) -> FeatureDict:
    """Convert numpy data dict to unified numpy data dict
    for rectangle domain.
    """
    # Load reference solutions and sigmas
    psi = np_data["psi_label"]  # [B, I, J, M]
    sigma_t = np_data["sigma_t"]  # [B, I, J]
    sigma_a = np_data["sigma_a"]  # [B, I, J]
    psi_bc = np_data["psi_bc"]  # [B, 2*(I+J), 4]

    scattering_kernel_value = np.tile(
        np_data["scattering_kernel"],
        (1, sigma_t.shape[1] * sigma_t.shape[2], 1),
    )
    scattering_kernel = scattering_kernel_value.reshape(
        *(psi.shape + psi.shape[-1:])
    )
    nv = int(psi.shape[-1] / 4)
    boundary_scattering_kernel_L = np.concatenate(
        [
            scattering_kernel[:, 0, :, -nv:, :],
            scattering_kernel[:, 0, :, :nv, :],
        ],
        axis=-2,
    )
    boundary_scattering_kernel = np.concatenate(
        [
            boundary_scattering_kernel_L,
            scattering_kernel[:, -1, :, nv : 3 * nv, :],
            scattering_kernel[:, :, 0, : 2 * nv, :],
            scattering_kernel[:, :, -1, 2 * nv :, :],
        ],
        axis=1,
    )

    sigma = np.stack([sigma_t, sigma_a], axis=-1)

    features = {
        "sigma": sigma,
        "psi_label": psi,
        "scattering_kernel": scattering_kernel,
        "boundary_scattering_kernel": boundary_scattering_kernel,
        "self_scattering_kernel": np_data["scattering_kernel"],
        "boundary": psi_bc,
    }

    return features


def make_grid_features(np_data: Mapping[str, np.ndarray]) -> FeatureDict:
    """Convert numpy grid dict and preprocess grid points."""

    features = {}

    vx, vy = np_data["ct"], np_data["st"]  # [M, 1]
    v_coords = np.concatenate([vx, vy], axis=-1)
    x, y = np.squeeze(np_data["x"]), np.squeeze(np_data["y"])

    features["position_coords"] = utils.cartesian_product(
        np.expand_dims(x, axis=-1),
        np.expand_dims(y, axis=-1),
    )
    features["velocity_coords"] = v_coords

    rv = utils.cartesian_product(
        np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1), v_coords
    )
    features["phase_coords"] = rv

    rv_prime, w_prime = np_data["rv_prime"], np_data["omega_prime"]
    features["boundary_coords"] = rv_prime
    features["boundary_weights"] = w_prime

    features["velocity_weights"] = np.squeeze(np_data["w_angle"])

    return features


def make_shape_dict(np_data: Mapping[str, np.ndarray]) -> Mapping[str, int]:
    num_x = np.shape(np.squeeze(np_data["x"]))[0]
    num_y = np.shape(np.squeeze(np_data["y"]))[0]
    num_v = np.shape(np.squeeze(np_data["ct"]))[0]

    shape_dict = {}
    shape_dict["num_examples"] = np.shape(np_data["psi_label"])[0]
    shape_dict["num_position_coords"] = num_x * num_y
    shape_dict["num_velocity_coords"] = num_v
    shape_dict["num_phase_coords"] = num_x * num_y * num_v
    shape_dict["num_boundary_coords"] = (num_x + num_y) * num_v

    return shape_dict


class DataPipeline:
    def __init__(self, source_dir: str, data_name_list: list[str]):
        self.source_dir = source_dir
        self.data_name_list = data_name_list
        self.data = self.load_data()

    def load_data(self):
        return matlab.mat_loader(self.source_dir, self.data_name_list)

    def process(self, normalization: Optional[bool] = False) -> FeatureDict:

        data_feature = make_data_features(self.data)
        grid_feature = make_grid_features(self.data)
        shape_dict = make_shape_dict(self.data)

        raw_data = {
            "functions": data_feature,
            "grid": grid_feature,
            "shape": shape_dict,
        }

        if normalization:
            (
                data_feature["psi_label"],
                psi_min,
                psi_range,
            ) = utils.normalize(data_feature["psi_label"])
            (
                data_feature["boundary"],
                boundary_min,
                boundary_range,
            ) = utils.normalize(data_feature["boundary"])

            normalization_dict = {}
            normalization_dict["psi_min"] = psi_min
            normalization_dict["psi_range"] = psi_range
            normalization_dict["boundary_min"] = boundary_min
            normalization_dict["boundary_range"] = boundary_range

            raw_data.update({"normalization": normalization_dict})

        return raw_data
