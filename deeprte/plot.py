import os

import dill
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from absl import logging
from matplotlib.ticker import MultipleLocator

result_path = "/nfs/my/projects/deeprte/results/2023-06-15T16:21:40"


def plot_phi(
    X,
    Y,
    Z,
    save_path=None,
    cmap="RdYlBu_r",
    norm=mcolors.PowerNorm(0.7),
    levels=25,
    figsize=(6, 6),
):
    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={"projection": "3d"})

    ax.xaxis._axinfo["grid"].update({"linewidth": 0.3, "linestyle": "--", "color": "k"})
    ax.yaxis._axinfo["grid"].update({"linewidth": 0.3, "linestyle": "--", "color": "k"})
    ax.zaxis._axinfo["grid"].update({"linewidth": 0.3, "linestyle": "--", "color": "k"})

    # 设置网格背景色
    ax.xaxis.set_pane_color((0, 0, 0, 0))
    ax.yaxis.set_pane_color((0, 0, 0, 0))
    ax.zaxis.set_pane_color((0, 0, 0, 0))

    ax.zaxis._axinfo["juggled"] = (1, 2, 7)

    ax.xaxis._axinfo["tick"]["outward_factor"] = 0
    ax.xaxis._axinfo["tick"]["inward_factor"] = 0.4
    ax.yaxis._axinfo["tick"]["outward_factor"] = 0
    ax.yaxis._axinfo["tick"]["inward_factor"] = 0.4
    ax.zaxis._axinfo["tick"]["outward_factor"] = 0.4
    ax.zaxis._axinfo["tick"]["inward_factor"] = 0.0

    x0 = MultipleLocator(0.2)  # x轴每10一个刻度
    y0 = MultipleLocator(0.25)
    z0 = MultipleLocator(0.9)

    ax.xaxis.set_major_locator(x0)
    ax.yaxis.set_major_locator(y0)
    ax.zaxis.set_major_locator(z0)

    ax.tick_params(axis="z", labelsize=10, pad=-3, direction="in")
    ax.tick_params(axis="x", labelsize=10, pad=-5, direction="in")
    ax.tick_params(axis="y", labelsize=10, pad=-1, direction="in")
    font = {"size": 12, "weight": "normal"}
    ax.set_xlabel("x", labelpad=-10, rotation=-25, fontdict=font)
    ax.set_ylabel("y", labelpad=0, rotation=30, fontdict=font)
    ax.set_zlabel(r"Predict $f(r,\Omega)$", labelpad=-5, rotation=90, fontdict=font)

    # # Plot the 3D surface
    # surf = ax.plot_surface(
    #     X, Y, Z, cmap=cmap, linewidth=0.2, alpha=0.7, lw=0.5, norm=norm
    # )

    z_min, z_max = np.min(Z), np.max(Z)
    z_range = z_max - z_min
    offset = z_min - 0.2 * z_range
    ax.contour(
        X,
        Y,
        Z,
        zdir="z",
        offset=offset,
        cmap=cmap,
        norm=norm,
        levels=levels,
        linewidths=0.7,
    )

    # cbar = fig.colorbar(
    #     surf,
    #     ax=ax,
    #     location="right",
    #     anchor=(-0.5, 0.4),
    #     shrink=0.6,
    #     format=matplotlib.ticker.ScalarFormatter(),
    # )
    # cbar.minorticks_off()

    ax.set(zlim=(offset, z_max * 1.1))
    # ax.set_title(r"Predict $f(r,\Omega)$", fontsize=15, loc='center', pad=0)

    ax.view_init(elev=17, azim=280)
    # plt.show()
    if save_path is not None:
        plt.savefig(save_path)


class FormatterNoInfo(logging.Formatter):
    def __init__(self, fmt="%(levelname)s: %(message)s"):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        if record.levelno == logging.INFO:
            return str(record.getMessage())
        return logging.Formatter.format(self, record)


def setup_default_logging(default_level=logging.INFO, log_path=""):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(FormatterNoInfo())
    logging.root.addHandler(console_handler)
    logging.root.setLevel(default_level)
    if log_path:
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=(1024**2 * 2), backupCount=3
        )
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)20s: [%(levelname)8s] - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)


def rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2) / np.mean(target**2))


def get_normalization_ratio(psi_range, boundary_range):
    psi_range = float(psi_range.split(" ")[-1])
    boundary_range = float(boundary_range.split(" ")[-1])
    return psi_range / boundary_range


def plot_contour(X, Y, phi, title, fontdict=None, save_path=None, figsize=(20, 5)):
    def _plt_contour(ax, X, Y, Z, cmap="RdBu_r", levels=25):
        ax.tick_params(axis="x", labelsize=15, pad=2, direction="in")
        ax.tick_params(axis="y", labelsize=15, pad=2, direction="in")
        font = {"size": 17, "weight": "normal"}
        ax.set_xlabel("x", labelpad=5, rotation=0, fontdict=font)
        ax.set_ylabel("y", labelpad=0, rotation=0, fontdict=font)

        cs = ax.contourf(X, Y, Z, cmap=cmap, levels=levels)

        return cs

    fig, ax = plt.subplots(figsize=figsize)

    cs1 = _plt_contour(ax, X, Y, phi, cmap="RdYlBu_r")
    ax.set_title(title, **fontdict)
    cbar = fig.colorbar(cs1)
    cbar.ax.tick_params(labelsize=16)
    # axs[0].set_position([0.0, 0.23, 0.25, 0.45])

    if save_path is not None:
        plt.savefig(save_path)


def main():
    setup_default_logging(log_path=os.path.join(result_path, "plot.log"))
    logging.info("Plotting results from %s", result_path)

    # plot contour params
    n = 40
    h = 1 / n
    X, Y = np.meshgrid(
        np.linspace(0 + 0.5 * h, 1 - 0.5 * h, 40),
        np.linspace(0 + 0.5 * h, 1 - 0.5 * h, 40),
    )
    fontdict = {"fontsize": 17, "fontweight": "normal"}

    for item in os.listdir(result_path):
        item_path = os.path.join(result_path, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "result.dill")
            logging.info("Plotting %s", file_path)
            with open(file_path, "rb") as f:
                result = dill.load(f)

            # Plot phi
            phi_pre = np.squeeze(result["predicted_phi"])
            phi_label = np.squeeze(result["phi_label"])

            # print(phi_pre.shape, phi_label.shape)

            plot_contour(
                X,
                Y,
                phi_label,
                title=r"Label $\Phi_{label}(r)$",
                figsize=(6, 5),
                fontdict=fontdict,
                save_path=os.path.join(item_path, "phi_label.pdf"),
            )

            plot_contour(
                X,
                Y,
                phi_pre,
                title=r"Predict $\Phi(r)$",
                figsize=(6, 5),
                fontdict=fontdict,
                save_path=os.path.join(item_path, "phi_pre.pdf"),
            )

            plot_contour(
                X,
                Y,
                abs(phi_pre - phi_label),
                title=r"Error $|\Phi(r) - \Phi_{label}(r)|$",
                figsize=(6, 5),
                fontdict=fontdict,
                save_path=os.path.join(item_path, "phi_error.pdf"),
            )

            plot_phi(
                X,
                Y,
                phi_label,
                figsize=(6, 5),
                save_path=os.path.join(item_path, "phi_label_3d.pdf"),
            )
    logging.info("Done!")


if __name__ == "__main__":
    main()
