"""A Python wrapper for matlab files"""

from pathlib import Path

import numpy as np
import scipy.io as sio
import tree

BATCH_FEAT_LIST = [
    "sigma_a",
    "sigma_t",
    "psi_label",
    "psi_bc",
    "scattering_kernel",
]


def interpolate(data):
    list_omega = [
        data["omega_prime"][41 * i : 41 * (i + 1), :] for i in range(4)
    ]
    list_omega = [(omega[1:] + omega[:-1]) / 2 for omega in list_omega]
    data["omega_prime"] = np.concatenate(list_omega, axis=0) / 40

    phi = data["phi"]
    phi = (phi[:, :-1, :] + phi[:, 1:, :]) / 2
    data["phi"] = (phi[:, :, :-1] + phi[:, :, 1:]) / 2

    list_psi_bc = [
        data["psi_bc"][:, 41 * i : 41 * (i + 1), :] for i in range(4)
    ]
    list_psi_bc = [
        (psi_bc[:, 1:, :] + psi_bc[:, :-1, :]) / 2 for psi_bc in list_psi_bc
    ]
    data["psi_bc"] = np.concatenate(list_psi_bc, axis=-2)

    psi = data["psi_label"]
    psi = (psi[:, :-1, :, :] + psi[:, 1:, :, :]) / 2
    data["psi_label"] = (psi[:, :, :-1, :] + psi[:, :, 1:, :]) / 2

    list_rv_prime = [
        data["rv_prime"][41 * i : 41 * (i + 1), :, :] for i in range(4)
    ]
    list_rv_prime = [
        (rv_prime[1:, :, :] + rv_prime[:-1, :, :]) / 2
        for rv_prime in list_rv_prime
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


def mat_loader(
    source_dir: str, data_name_list: list[str], seed: int = 12345
) -> dict[str, np.ndarray]:

    dir_path = Path(source_dir)
    data_list = []
    for filename in data_name_list:
        data_path = dir_path / filename
        mat_dict = sio.loadmat(data_path)
        if "scattering_kernel" not in mat_dict.keys():
            print("Expact Scattering Kernel.")
            # rng = np.random.default_rng(seed)
            # num_sample = mat_dict["sigma_a"].shape[0]
            # num_vec = mat_dict["ct"].shape[0]
            # mat_dict["scattering_kernel"] = rng.uniform(
            #     0, 1, (num_sample, num_vec, num_vec)
            # )
        data_list.append(mat_dict)
    data = data_list[0]

    for k in BATCH_FEAT_LIST:
        data[k] = np.concatenate([d[k] for d in data_list], axis=0)

    unused_keys = [k for k in data.keys() if k.startswith("__")]
    # for k in unused_keys
    for k in unused_keys:
        del data[k]
    # print(mat_dict.keys())
    data = tree.map_structure(lambda x: np.array(x, dtype=np.float32), data)

    data = interpolate(data)
    return data
