"""Convert Matlab dataset to numpy dataset."""

import pathlib
from typing import Mapping, MutableMapping, Optional

import numpy as np
import tree

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

    sigma = np.stack([sigma_t, sigma_a], axis=-1)

    features = {
        "sigma": sigma,
        "psi_label": psi,
        "scattering_kernel": scattering_kernel,
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
    shape_dict = {}
    shape_dict["num_x"] = np.shape(np.squeeze(np_data["x"]))[0]
    shape_dict["num_y"] = np.shape(np.squeeze(np_data["y"]))[0]
    shape_dict["num_v"] = np.shape(np.squeeze(np_data["ct"]))[0]
    shape_dict["num_examples"] = np.shape(np_data["psi_label"])[0]

    return shape_dict


class DataPipeline:
    def __init__(
        self,
        source_dir: str,
        data_name_list: list[str],
    ):
        self.source_dir = source_dir
        self.data_name_list = data_name_list
        self.data = self.load_data()

    def load_data(
        self,
    ):

        return matlab.mat_loader(self.source_dir, self.data_name_list)

    def process(
        self,
        pre_shuffle: bool = False,
        pre_shuffle_seed: int = 0,
        is_split_test_samples: bool = False,
        num_test_samples: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> FeatureDict:

        data_feature = make_data_features(self.data)
        grid_feature = make_grid_features(self.data)
        shape_dict = make_shape_dict(self.data)

        if pre_shuffle:
            rng = np.random.default_rng(seed=pre_shuffle_seed)
            indices = np.arange(shape_dict["num_examples"])

            _ = rng.shuffle(indices)

            data_feature = tree.map_structure(
                lambda x: np.take(x, indices, axis=0), data_feature
            )

        if is_split_test_samples:

            test_ds = tree.map_structure(
                lambda x: x[:num_test_samples],
                data_feature,
            )
            train_ds = tree.map_structure(
                lambda x: x[num_test_samples:],
                data_feature,
            )

            if save_path:
                if not isinstance(save_path, pathlib.Path):
                    save_path = pathlib.Path(save_path)
            else:
                path = pathlib.Path(self.source_dir)
                save_path = path / (self.data_name_list[0] + "_test_ds.npz")
            np.savez(save_path, **test_ds, **grid_feature, **shape_dict)

            shape_dict["num_train_and_val"] = (
                shape_dict["num_examples"] - num_test_samples
            )

            return {**train_ds, **grid_feature, **shape_dict}

        return {**data_feature, **grid_feature, **shape_dict}
