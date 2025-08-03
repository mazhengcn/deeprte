"""A Python wrapper for matlab files"""

from pathlib import Path
from typing import List, Dict
import numpy as np
import scipy.io as sio
import tree

BATCH_FEAT_LIST = [
    "sigma_a",
    "sigma_t",
    "psi_label",
    "psi_bc",
    "scattering_kernel",
    "phi",
]

UNUSED_KEY_LIST = ["rand_params", "config"]


def interpolate(data):
    list_omega = [data["omega_prime"][41 * i : 41 * (i + 1), :] for i in range(4)]
    list_omega = [(omega[1:] + omega[:-1]) / 2 for omega in list_omega]
    data["omega_prime"] = np.concatenate(list_omega, axis=0) / 40

    phi = data["phi"]
    phi = (phi[:, :-1, :] + phi[:, 1:, :]) / 2
    data["phi"] = (phi[:, :, :-1] + phi[:, :, 1:]) / 2

    list_psi_bc = [data["psi_bc"][:, 41 * i : 41 * (i + 1), :] for i in range(4)]
    list_psi_bc = [(psi_bc[:, 1:, :] + psi_bc[:, :-1, :]) / 2 for psi_bc in list_psi_bc]
    data["psi_bc"] = np.concatenate(list_psi_bc, axis=-2)

    psi = data["psi_label"]
    psi = (psi[:, :-1, :, :] + psi[:, 1:, :, :]) / 2
    data["psi_label"] = (psi[:, :, :-1, :] + psi[:, :, 1:, :]) / 2

    list_rv_prime = [data["rv_prime"][41 * i : 41 * (i + 1), :, :] for i in range(4)]
    list_rv_prime = [
        (rv_prime[1:, :, :] + rv_prime[:-1, :, :]) / 2 for rv_prime in list_rv_prime
    ]
    data["rv_prime"] = np.concatenate(list_rv_prime, axis=0)

    sigma_a = data["sigma_a"]
    sigma_a = (sigma_a[:, :-1, :] + sigma_a[:, 1:, :]) / 2
    data["sigma_a"] = (sigma_a[:, :, :-1] + sigma_a[:, :, 1:]) / 2

    sigma_t = data["sigma_t"]
    sigma_t = (sigma_t[:, :-1, :] + sigma_t[:, 1:, :]) / 2
    data["sigma_t"] = (sigma_t[:, :, :-1] + sigma_t[:, :, 1:]) / 2

    x = np.squeeze(data["x"])
    x = (x[1:] + x[:-1]) / 2
    data["x"] = x

    y = np.squeeze(data["y"])
    y = (y[1:] + y[:-1]) / 2
    data["y"] = y

    return data


def mat_loader(source_dir: str, data_name_list: List[str]) -> Dict[str, np.ndarray]:
    np_data = {}

    dir_path = Path(source_dir)
    for filename in data_name_list:
        assert filename.endswith(".mat")
        data_path = dir_path / filename
        mat_dict = sio.loadmat(data_path)
        if np_data:
            if mat_dict.keys() != np_data.keys():
                raise ValueError("The keys of the data are not the same")
            for k in mat_dict.keys():
                if k not in BATCH_FEAT_LIST:
                    if isinstance(mat_dict[k], np.ndarray):
                        assert np.allclose(
                            mat_dict[k], np_data[k]
                        ), f"The value of key: {k} are not the same!"
                else:
                    np_data[k] = np.concatenate([mat_dict[k], np_data[k]], axis=0)
        else:
            np_data = mat_dict.copy()

    unused_keys = [k for k in UNUSED_KEY_LIST if k in np_data.keys()]
    unused_keys += [k for k in np_data.keys() if k.startswith("__")]
    # for k in unused_keys
    for k in unused_keys:
        del np_data[k]
    np_data = tree.map_structure(lambda x: np.array(x, dtype=np.float32), np_data)
    np_data = interpolate(np_data)

    return np_data
