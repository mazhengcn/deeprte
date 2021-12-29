import os

import matplotlib.pyplot as plt
import ml_collections
import numpy as np
from absl import app

from deeprte.deeprte_typings import GraphOfMapping
from deeprte.model.rte_op import RTEOperator
from deeprte.modules.green_fn import GreenFunction
from deeprte.utils import get_model_haiku_params

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

ROOT_PATH = "/workspace/modnet/"


def main(_):
    config = ml_collections.ConfigDict(
        {
            "green_net": [128, 128, 128, 128, 1],
            "coeffs_net": {"weights": [64, 1], "coeffs": [64, 2]},
        }
    )

    sol = RTEOperator(config, GreenFunction)

    data_dict = np.load(ROOT_PATH + "data/rte/rte_2d_converted.npz")

    idx = np.random.randint(500)

    xy = data_dict["xy"].reshape(-1, 2)
    sigma = data_dict["sigma"][idx : idx + 1].reshape(1, -1, 2)
    bc_pts = data_dict["bc_points"].reshape(-1, 4)
    bc_weights = (
        data_dict["psi_bc"][idx : idx + 1].reshape(1, -1)
        * data_dict["bc_weights"].flatten()
    )
    c, s, omega = data_dict["c"], data_dict["s"], data_dict["omega"]

    phi = data_dict["phi"][idx]

    params = get_model_haiku_params("rte_2d_1", ROOT_PATH + "data/rte/")

    pred_rho = sol.rho(
        params,
        None,
        xy,
        GraphOfMapping(xy, sigma),
        GraphOfMapping(bc_pts, bc_weights),
        (np.concatenate([c, s], axis=-1), omega),
    )

    pred_rho = pred_rho.reshape(-1, 38, 38).squeeze()

    fig, _axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    fig.subplots_adjust(hspace=0.3)
    axs = _axs.flatten()

    cs = axs[0].contour(
        data_dict["xy"][..., 0],
        data_dict["xy"][..., 1],
        np.abs(pred_rho - phi),
    )
    axs[0].clabel(cs, inline=True, fontsize=10)
    fig.colorbar(cs, ax=axs[0])

    axs[1].plot(
        data_dict["xy"][:, 0, 0],
        phi[:, 0],
        data_dict["xy"][:, 0, 0],
        pred_rho[:, 0],
        "*",
    )

    plt.show()

    plt.savefig(ROOT_PATH + "data/rte/params/rte_2d_1.pdf")


if __name__ == "__main__":
    app.run(main)
