"""A Python wrapper for matlab files"""

from pathlib import Path

import numpy as np
import scipy.io as sio

BATCH_FEAT_LIST = [
    "sigma_a",
    "sigma_t",
    "psi_label",
    "psi_bc",
    "scattering_kernel",
]


def mat_loader(
    source_dir: str, data_name_list: list[str], seed: int = 12345
) -> dict[str, np.ndarray]:

    dir_path = Path(source_dir)
    data_list = []
    for filename in data_name_list:
        data_path = dir_path / filename
        mat_dict = sio.loadmat(data_path)
        if "scattering_kernel" not in mat_dict.keys():
            rng = np.random.default_rng(seed)
            num_sample = mat_dict["sigma_a"].shape[0]
            num_vec = mat_dict["ct"].shape[0]
            mat_dict["scattering_kernel"] = rng.uniform(
                0, 1, (num_sample, num_vec, num_vec)
            )
        data_list.append(mat_dict)
    data = data_list[0]

    for k in BATCH_FEAT_LIST:
        data[k] = np.concatenate([d[k] for d in data_list], axis=0)

    unused_keys = [k for k in data.keys() if k.startswith("__")]
    # for k in unused_keys
    for k in unused_keys:
        del data[k]

    return data
