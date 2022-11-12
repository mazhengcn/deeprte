"""Convert Matlab dataset to numpy dataset."""

import pathlib
from collections.abc import Mapping

import numpy as np
import scipy.io as sio
from absl import app, flags, logging

from deeprte.utils import to_flat_dict

FLAGS = flags.FLAGS

Float = float | np.float32

flags.DEFINE_string("source_dir", None, "Directory of dataset to be converted.")

flags.DEFINE_list("datafiles", None, "List of data file names to be converted.")

flags.DEFINE_string("save_path", None, "Directory to save converted numpy dataset.")


def mat_to_np_dict(
    mat_data: Mapping[str, np.ndarray], update_grid: bool = False
) -> Mapping[str, Mapping[str, np.ndarray]]:
    """Convert Matlab data dict to unified numpy data dict for rectangle domain.

    Args:
        data: originial data dict generated using Matlab.

    Returns:
        A nested dict containing numpy arrays. {"data": {}, "grid": {}}
    """
    # Load reference solutions and sigmas
    phi = mat_data["phi"]  # [B, I, J]
    psi = mat_data["psi_label"]  # [B, I, J, M]
    sigma_t = mat_data["sigma_t"]  # [B, I, J]
    sigma_a = mat_data["sigma_a"]  # [B, I, J]
    psi_bc = mat_data["psi_bc"]  # [B, 2*(I+J), 4]

    data_dict = {
        "sigma_t": sigma_t,
        "sigma_a": sigma_a,
        "psi_label": psi,
        "phi": phi,
        "psi_bc": psi_bc,
    }

    if update_grid:
        vx = mat_data["ct"]  # [M, 1]
        vy = mat_data["st"]  # [M, 1]
        v_coords = np.concatenate([vx, vy], axis=-1)
        grid_dict = {
            "r": mat_data["r"],
            "v": v_coords,
            "w_angle": np.squeeze(mat_data["w_angle"]),
        }

        rv_primes, w_primes = mat_data["rv_prime"], mat_data["omega_prime"]

        grid_dict.update(
            {
                "rv_prime": rv_primes,
                "w_prime": w_primes,
            }
        )

        return data_dict, grid_dict

    return data_dict


def main(argv):
    # unused
    del argv

    source_dir = FLAGS.source_dir
    if not isinstance(source_dir, pathlib.Path):
        source_dir = pathlib.Path(source_dir)

    data_dicts = []
    for i, filename in enumerate(FLAGS.datafiles):

        data_path = source_dir / filename
        logging.info("Processing dataset %d from path %s.", i, data_path)

        if data_path.suffix == ".mat":
            data = sio.loadmat(data_path)

        elif data_path.suffix == ".npy":
            data = np.load(data, allow_pickle=True)
            data = data.item()
        else:
            data = np.load(data_path, allow_pickle=True)

        if i == 0:
            data_dict, grid = mat_to_np_dict(data, update_grid=True)
        else:
            data_dict = mat_to_np_dict(data, update_grid=False)

        data_dicts.append(data_dict)

    converted_data = {"data": {}, "grid": grid}
    for k in data_dict:
        converted_data["data"][k] = np.concatenate([d[k] for d in data_dicts], axis=0)

    save_path = FLAGS.save_path
    # Save converted dataset
    if save_path:
        if not isinstance(save_path, pathlib.Path):
            save_path = pathlib.Path(save_path)
    else:
        save_path = source_dir / "converted.npz"

    np.savez(
        save_path,
        **to_flat_dict(converted_data, sep="/"),
    )
    logging.info("Saved converted dataset %s.", save_path)


if __name__ == "__main__":
    app.run(main)
