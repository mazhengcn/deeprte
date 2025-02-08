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

"""Convert Matlab dataset to numpy dataset."""

import os
from collections.abc import Mapping, MutableMapping
from typing import Optional

import numpy as np
from rte_dataset.builders import utils

FeatureDict = MutableMapping[str, np.ndarray]


def make_data_features(np_data: Mapping[str, np.ndarray]) -> FeatureDict:
    """Convert numpy data dict to unified numpy data dict
    for rectangle domain.
    """
    # Load reference solutions and sigmas
    # print("Available keys in np_data:", np_data.keys())
    psi = np_data["psi"]  # [B, I, J, M]
    sigma = np_data["sigma"]  # [B, I', J']
    # psi_bc = np_data["boundary"]  # [B, 2*(I+J), 4]
    psi_bc = np.tile(np_data["q"][..., None], (1, 1, 24))

    scattering_kernel_value = np.tile(
        np_data["scattering_kernel"],
        (1, psi.shape[1] * psi.shape[2], 1),
    )
    scattering_kernel = scattering_kernel_value.reshape(*(psi.shape + psi.shape[-1:]))
    nv = int(psi.shape[-1] / 4)
    # boundary_scattering_kernel_L = np.concatenate(
    #     [
    #         scattering_kernel[:, 0, :, -nv:, :],
    #         scattering_kernel[:, 0, :, :nv, :],
    #     ],
    #     axis=-2,
    # )
    # boundary_scattering_kernel = np.concatenate(
    #     [
    #         boundary_scattering_kernel_L,
    #         scattering_kernel[:, -1, :, nv : 3 * nv, :],
    #         scattering_kernel[:, :, 0, : 2 * nv, :],
    #         scattering_kernel[:, :, -1, 2 * nv :, :],
    #     ],
    #     axis=1,
    # )
    boundary_scattering_kernel = scattering_kernel.reshape(
        (
            scattering_kernel.shape[0],
            -1,
            scattering_kernel.shape[-2],
            scattering_kernel.shape[-1],
        )
    )

    # source is the same as boundary
    source = psi_bc.copy()
    source_label = psi_bc.copy()

    features = {
        "sigma": sigma,
        "psi_label": psi,
        "scattering_kernel": scattering_kernel,
        "boundary_scattering_kernel": boundary_scattering_kernel,
        "self_scattering_kernel": np_data["scattering_kernel"],
        "boundary": psi_bc,
        "source": source,
        "source_label": source_label,
    }
    if "moments" in np_data:
        features["moments"] = np_data["moments"]
        features["basis_inner_product"] = np_data["basis_inner_product"]

    return features


def make_grid_features(np_data: Mapping[str, np.ndarray]) -> FeatureDict:
    """Convert numpy grid dict and preprocess grid points."""

    features = {}

    v_coords = np_data["v"]
    x, y = np.squeeze(np_data["x"]), np.squeeze(np_data["y"])
    x_coef, y_coef = np.squeeze(np_data["x_coef"]), np.squeeze(np_data["y_coef"])

    features["position_coords"] = utils.cartesian_product(
        np.expand_dims(x_coef, axis=-1),
        np.expand_dims(y_coef, axis=-1),
    )
    features["velocity_coords"] = v_coords

    rv = utils.cartesian_product(
        np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1), v_coords
    )
    features["phase_coords"] = rv
    _, w_prime = np_data["bc_coords"], np_data["bc_weights"]
    # features["boundary_coords"] = rv_prime
    features["boundary_coords"] = rv

    features["boundary_weights"] = np.tile(
        np_data["velocity_weights"][None, None, :], (*rv.shape[:2], 1)
    ) / np.prod(rv.shape[:2])
    features["velocity_weights"] = np.squeeze(np_data["velocity_weights"])

    # source is the same as boundary
    features["source_coords"] = rv.copy()
    features["source_weights"] = features["boundary_weights"].copy()

    return features


def make_shape_dict(np_data: Mapping[str, np.ndarray]) -> Mapping[str, int]:
    num_x_coef = np.shape(np.squeeze(np_data["x_coef"]))[0]
    num_y_coef = np.shape(np.squeeze(np_data["y_coef"]))[0]
    num_x = np.shape(np_data["psi"])[1]
    num_y = np.shape(np_data["psi"])[2]
    num_v = np.shape(np.squeeze(np_data["v"]))[0]

    shape_dict = {}
    shape_dict["num_examples"] = np.shape(np_data["psi"])[0]
    shape_dict["num_position_coords"] = num_x_coef * num_y_coef
    shape_dict["num_velocity_coords"] = num_v
    shape_dict["num_phase_coords"] = num_x * num_y * num_v
    # shape_dict["num_boundary_coords"] = np.multiply(
    #     *np.shape(np_data["bc_coords"])[:-1]
    # )
    shape_dict["num_boundary_coords"] = num_x * num_y * num_v
    shape_dict["num_source_coords"] = num_x * num_y * num_v

    return shape_dict


class DataPipeline:
    def __init__(self, source_dir: str, data_name_list: list[str]):
        self.source_dir = source_dir
        self.data_name_list = data_name_list
        self.data = self.load_data()

    def load_data(self):
        return np.load(os.path.join(self.source_dir, self.data_name_list[0]))

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
