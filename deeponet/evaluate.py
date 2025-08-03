import json
import logging
import os
from typing import Any

import dill
import matplotlib
import matplotlib.colors as colors
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import ml_collections
import numpy as np
import torch
import torch.nn as nn
import tree
import utils
import yaml
from matplotlib import cm
from matplotlib.collections import PolyCollection
from matplotlib.colors import LightSource, ListedColormap
from matplotlib.ticker import MultipleLocator
from modules import DeepONet, FullyConnected, ModifiedMlp, ResNet
from mpl_toolkits.mplot3d import art3d, axes3d
from utils import Key

from data.dataset import preprocess

_logger = logging.getLogger("evaluate")

DATA_PATH = "/root/projects/deeponet/data/g0.1-sigma_a3-sigma_t6_test_normalized.npz"
result_path = "/root/projects/deeponet/deeponet/output/train/20231010-065451-deeponet"

# 获取文件名和扩展名
filename, ext = os.path.splitext(os.path.basename(DATA_PATH))
file_name = filename[: -len(ext)]

config_path = os.path.join(result_path, "config.yaml")
model_ckpt_path = os.path.join(result_path, "model_best.pth.tar")
outdir = os.path.join(result_path, "results", filename)
os.makedirs(outdir, exist_ok=True)


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


def rmse(pre, label):
    return torch.sqrt(torch.sum((label - pre) ** 2) / torch.sum(label**2))


def main():
    utils.setup_default_logging(log_path=os.path.join(outdir, "log.INFO"))

    _logger.info("Loading config from %s..." % config_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        cfg = ml_collections.ConfigDict(cfg)

    _logger.info("Loading data from %s..." % DATA_PATH)
    np_data = dict(np.load(DATA_PATH, allow_pickle=True))
    weights = np_data["weights"]

    _logger.info("Preprocessing data...")

    np_data = preprocess(cfg, np_data)
    branch_dr, trunk_dr, label_dr, input_shape_dict = np_data

    branch_keys = list(input_shape_dict["branch"].keys())
    trunk_keys = list(input_shape_dict["trunk"].keys())
    # input_keys = branch_keys + trunk_keys

    def slice_batch(batch, i):
        branch_input = {k: batch[k][i] for k in branch_keys}
        trunk_input = {k: batch[k] for k in trunk_keys}
        label = {"psi_label": batch["psi_label"][i]}
        return {**branch_input, **trunk_input, **label}

    latent_size = cfg.model.latent_size

    def _get_activation():
        act = cfg.model.get("activation", "relu")
        if act == "tanh":
            return nn.Tanh
        elif act == "relu":
            return nn.ReLU

    def create_model(model_cfg, shape_dict):
        input_name = model_cfg.get("input_key")
        model_type = model_cfg.get("type")
        if model_type == "mlp":
            net = FullyConnected(
                [Key(input_name, size=shape_dict[input_name])],
                [Key(model_cfg.get("output_key"), latent_size)],
                model_cfg.hidden_units,
                activation=_get_activation(),
            )
        elif model_type == "modified_mlp":
            net = ModifiedMlp(
                [Key(input_name, size=shape_dict[input_name])],
                [Key(model_cfg.get("output_key"), latent_size)],
                model_cfg.hidden_units,
                activation=_get_activation(),
            )
        elif model_type == "resnet":
            net = ResNet(
                [Key(input_name, size=shape_dict[input_name])],
                [Key(model_cfg.get("output_key"), latent_size)],
                model_cfg.hidden_units,
                activation=_get_activation(),
            )
        return net

    branch_net_list = []
    for k, d in cfg.model.items():
        if "branch" in k:
            branch_net_list.append(create_model(d, input_shape_dict["branch"]))
    trunk_net = create_model(cfg.model.trunk_net, input_shape_dict["trunk"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepONet(branch_net_list, trunk_net, output_keys=[Key("psi", 1)])

    model.to(device)
    loss_fn = nn.MSELoss().to(device)

    _logger.info("Loading model from %s..." % model_ckpt_path)
    ckpt = torch.load(model_ckpt_path, map_location="cuda")
    model.load_state_dict(ckpt["state_dict"])

    model.eval()

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info("Model created, param count: %d" % param_count)

    length = len(label_dr[list(label_dr.keys())[0]])
    _logger.info("Evaluating on %d samples..." % length)

    losses_m = utils.AverageMeter()
    rmse_m = utils.AverageMeter()

    # plot contour params
    n = 40
    h = 1 / n
    X, Y = np.meshgrid(
        np.linspace(0 + 0.5 * h, 1 - 0.5 * h, 40),
        np.linspace(0 + 0.5 * h, 1 - 0.5 * h, 40),
    )
    fontdict = {"fontsize": 17, "fontweight": "normal"}

    with torch.no_grad():
        for i in range(length):
            batch = slice_batch({**branch_dr, **trunk_dr, **label_dr}, i)
            input = {k: v.to(device) for k, v in batch.items()}
            label = input["psi_label"]

            output = model(input)["psi"]
            loss = loss_fn(output, label)
            rmse_loss = rmse(output, label)

            losses_m.update(loss.item(), label.size(0))
            rmse_m.update(rmse_loss.item(), label.size(0))

            _logger.info(
                f"Sample {i}/{length} "
                f"loss:{losses_m.val:#.3g}({losses_m.avg:#.3g})  "
                f"rmse:{rmse_m.val:#.3g}({rmse_m.avg:#.3g})"
            )

            pre_psi = output.cpu().numpy().reshape(n, n, -1)
            label_psi = label.cpu().numpy().reshape(n, n, -1)

            pre_phi = np.sum(pre_psi * weights, axis=-1)
            label_phi = np.sum(label_psi * weights, axis=-1)

            np_result = {
                "pre_phi": pre_phi,
                "label_phi": label_phi,
                "pre_psi": pre_psi,
                "label_psi": label_psi,
            }

            metrics = {
                "loss": losses_m.val,
                "rmse": rmse_m.val,
            }

            _logger.info("Plotting...")
            path = os.path.join(outdir, f"{i:03d}")
            os.makedirs(path, exist_ok=True)

            result_output_path = os.path.join(path, "result.dill")
            with open(result_output_path, "wb") as f:
                dill.dump(np_result, f)

            metrics_output_path = os.path.join(path, "metrics.json")
            with open(metrics_output_path, "w") as f:
                f.write(json.dumps(metrics, indent=4))

            plot_contour(
                X,
                Y,
                pre_phi,
                title=r"Predict $\Phi(r)$",
                fontdict=fontdict,
                figsize=(6, 5),
                save_path=os.path.join(path, "pred.pdf"),
            )

            plot_contour(
                X,
                Y,
                label_phi,
                title=r"Label $\Phi_{label}(r)$",
                fontdict=fontdict,
                figsize=(6, 5),
                save_path=os.path.join(path, "label.pdf"),
            )

            plot_contour(
                X,
                Y,
                abs(pre_phi - label_phi),
                title=r"Error $|\Phi(r) - \Phi_{label}(r)|$",
                fontdict=fontdict,
                figsize=(6, 5),
                save_path=os.path.join(path, "error.pdf"),
            )
    _logger.info("Done!")


if __name__ == "__main__":
    main()
