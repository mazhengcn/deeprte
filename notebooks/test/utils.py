# import plotly.express as px
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from deeprte.data.pipeline import DataPipeline, make_shape_dict
from deeprte.model.tf.rte_features import BATCH_FEATURE_NAMES


def plot_phi(r, phi_label, phi_pre):
    viridis = mpl.colormaps["viridis"](np.linspace(0, 1.2, 128))

    fig, _axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
    fig.subplots_adjust(hspace=0.3)
    axs = _axs.flatten()

    # fig = px.density_contour(phi_label)
    # fig.show()

    cs_1 = axs[0].contourf(
        r[..., 0], r[..., 1], phi_label, cmap=ListedColormap(viridis)
    )
    axs[0].set_title(r"Exact $f(x,v)$", fontsize=20)
    axs[0].tick_params(axis="both", labelsize=15)
    cbar = fig.colorbar(cs_1)
    cbar.ax.tick_params(labelsize=16)

    # fig = px.density_contour(phi_pre)
    cs_2 = axs[1].contourf(
        r[..., 0], r[..., 1], phi_pre, cmap=ListedColormap(viridis)
    )
    axs[1].set_title(r"Predict $f(x,v)$", fontsize=20)
    axs[1].tick_params(axis="both", labelsize=15)
    cbar = fig.colorbar(cs_2)
    cbar.ax.tick_params(labelsize=16)

    cs_3 = axs[2].contourf(
        r[..., 0],
        r[..., 1],
        abs(phi_pre - phi_label),
        cmap=ListedColormap(viridis),
    )
    axs[2].set_title(r"Absolute error", fontsize=20)
    axs[2].tick_params(axis="both", labelsize=15)
    cbar = fig.colorbar(cs_3)
    cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()

    plt.show()
    print(
        "rmse:",
        np.sqrt(np.mean((phi_label - phi_pre) ** 2) / np.mean(phi_label**2)),
    )


def slice_batch(i: int, feat: dict):
    return {
        k: feat[k][i : i + 1] if k in BATCH_FEATURE_NAMES else feat[k]
        for k in feat
    }


def get_psi_shape(source_dir, data_name_list):
    data_pipeline = DataPipeline(source_dir, data_name_list)
    shape_dict = make_shape_dict(data_pipeline.data)
    return (shape_dict["num_x"], shape_dict["num_y"], shape_dict["num_v"])


def get_normalized_rate(config):
    psi_range = float(
        (config.model.data.normalization_dict.psi_range).split(" ")[-1]
    )
    boundary_range = float(
        (config.model.data.normalization_dict.boundary_range).split(" ")[-1]
    )
    return psi_range / boundary_range
