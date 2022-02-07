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
import enum
import pathlib
from collections.abc import Mapping

import numpy as np
import scipy.io as sio
from absl import app, flags, logging

from deeprte.model.geometry.phase_space import PhaseSpace
from deeprte.model.utils import cartesian_product
from deeprte.utils import to_flat_dict

FLAGS = flags.FLAGS

Float = float | np.float32

flags.DEFINE_string("source_dir", None, "Directory of dataset to be converted.")

flags.DEFINE_list("datafiles", None, "List of data file names to be converted.")

flags.DEFINE_string(
    "save_path", None, "Directory to save converted numpy dataset."
)


class BoundaryPosition(enum.Enum):
    """Boundary mappings."""

    LEFT = 1
    RIGHT = 2
    BOTTOM = 3
    TOP = 4

    @classmethod
    def from_string(cls, name: str) -> "BoundaryPosition":
        return {
            "LEFT": BoundaryPosition.LEFT,
            "RIGHT": BoundaryPosition.RIGHT,
            "BOTTOM": BoundaryPosition.BOTTOM,
            "TOP": BoundaryPosition.TOP,
        }[name.upper()]

    @property
    def inner_normal(self):
        return {
            BoundaryPosition.LEFT: np.asarray([1, 0]),
            BoundaryPosition.RIGHT: np.asanyarray([-1, 0]),
            BoundaryPosition.BOTTOM: np.asarray([0, 1]),
            BoundaryPosition.TOP: np.asarray([0, -1]),
        }[self]


BOUNDARY_KEYS = {
    "list_psiL": (BoundaryPosition.LEFT, 0.0),
    "list_psiR": (BoundaryPosition.RIGHT, 1.0),
    "list_psiB": (BoundaryPosition.BOTTOM, 0.0),
    "list_psiT": (BoundaryPosition.TOP, 1.0),
}


def get_boundary_grid(
    boundary_position: BoundaryPosition,
    boundary_value: Float,
    boundary_coords: np.ndarray,
    v_coords: np.ndarray,
    v_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct boundary points  of a rectangle

    Args:
        on_which_boundary: a string indicating which boundary
            to the rectangle domain.
        boundary_value: value of boundary position, e.g., xmin, ymax, etc.
        boundary_coords: coordinates of boundary in lower dimension (d - 1).
        v_coords: coordinates of velocity.
        v_weights: weights of velocity.
    Returns:
        Tuple of boundary grid points and weights.
    """
    num_coords, delta = (
        boundary_coords.shape[0],
        boundary_coords[1] - boundary_coords[0],
    )
    if boundary_coords.ndim == 1:
        boundary_coords = np.expand_dims(boundary_coords, axis=-1)

    inflow_vel_indices = v_coords @ boundary_position.inner_normal > 0.0

    pos_coords = np.pad(
        boundary_coords,
        [[0, 0], np.abs(boundary_position.inner_normal)],
        constant_values=boundary_value,
    )

    pos_weights = delta * np.ones(num_coords, dtype=np.float32)
    vel_coords = v_coords[inflow_vel_indices]
    vel_weights = v_weights[inflow_vel_indices]

    phase_space = PhaseSpace(
        position_coords=pos_coords,
        velocity_coords=vel_coords,
        position_weights=pos_weights,
        velocity_weights=vel_weights,
    )

    return (
        phase_space.single_state(cartesian_product=True),
        phase_space.state_weights,
    )


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
    phi = mat_data["list_Phi"]  # [B, I, J]
    psi = np.swapaxes(mat_data["list_Psi"], 1, -1)  # [B, I, J, M]
    sigma_t = mat_data["list_sigma_T"]  # [B, I, J]
    sigma_a = mat_data["list_sigma_a"]  # [B, I, J]

    data_dict = {
        "sigma_t": sigma_t,
        "sigma_a": sigma_a,
        "psi_label": psi,
        "phi": phi,
    }

    psi_bc_list = []
    for key in BOUNDARY_KEYS.keys():
        psi_bc_list.append(np.swapaxes(mat_data[key], -1, -2).copy())

    data_dict.update({"psi_bc": np.concatenate(psi_bc_list, axis=1)})

    if update_grid:

        # Load velocity/direction coordinates and weights
        vx = np.transpose(mat_data["ct"])  # [M, 1]
        vy = np.transpose(mat_data["st"])  # [M, 1]
        v_coords = np.concatenate([vx, vy], axis=-1)
        v_weights = mat_data["omega"][0]  # [M]

        # Construct grid points
        _, nx, ny, _ = psi.shape  # [B, I, J, M]
        dx, dy = 1.0 / nx, 1.0 / ny
        x = np.arange(0.0 + 0.5 * dx, 1.0, dx, dtype=np.float32)
        y = np.arange(0.0 + 0.5 * dy, 1.0, dy, dtype=np.float32)
        r_coords = cartesian_product(x[:, None], y[:, None])
        grid_dict = {"r": r_coords, "v": v_coords, "w_angle": v_weights}

        rv_primes, w_primes = [], []
        for key in mat_data.keys() & BOUNDARY_KEYS.keys():
            boundary_position, boundary_value = BOUNDARY_KEYS[key]
            rv, w = get_boundary_grid(
                boundary_position, boundary_value, x, v_coords, v_weights
            )
            rv_primes.append(rv.copy())
            w_primes.append(w.copy())

        grid_dict.update(
            {
                "rv_prime": np.concatenate(rv_primes, axis=0),
                "w_prime": np.concatenate(w_primes, axis=0),
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
        logging.info(f"Processing dataset {i} from path {data_path}.")

        if data_path.suffix == ".mat":
            data = sio.loadmat(data_path)
            for k, v in data.items():
                if isinstance(v, np.ndarray) and v.ndim > 1:
                    data[k] = np.moveaxis(v, -1, 0).astype(np.float32)
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
    for k in data_dict.keys():
        converted_data["data"][k] = np.concatenate(
            [d[k] for d in data_dicts], axis=0
        )

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
    logging.info(f"Saved converted dataset {save_path}.")


if __name__ == "__main__":
    app.run(main)
