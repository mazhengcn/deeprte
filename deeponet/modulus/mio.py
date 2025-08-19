import os
import sys
import warnings

import modulus.sym
import numpy as np
import torch
from modules import MioBranchNet
from modulus.sym.dataset.discrete import DictGridDataset
from modulus.sym.domain.constraint.continuous import DeepONetConstraint
from modulus.sym.domain.validator.discrete import GridValidator
from modulus.sym.key import Key
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.hydra import (
    ModulusConfig,
    instantiate_arch,
    to_absolute_path,
    to_yaml,
)
from physicsnemo.sym.models.fourier_net import FourierNetArch
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.solver import Solver
from physicsnemosym.models.deeponet import DeepONetArch
from preprocess import preprocess


@modulus.sym.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    # [init-model]

    arch_1 = FullyConnectedArch(
        [Key("sigma", size=3200)], [Key("b1", size=128)], layer_size=128, nr_layers=2
    )
    arch_2 = FullyConnectedArch(
        [Key("boundary", size=1920)], [Key("b2", size=128)], layer_size=128, nr_layers=2
    )
    branch_net = MioBranchNet([arch_1, arch_2], [Key("branch", size=128)])

    trunk_net = FourierNetArch(
        input_keys=[Key("phase_coords", 4)],
        output_keys=[Key("trunk", 128)],
        nr_layers=4,
        layer_size=128,
        frequencies=("axis", [i for i in range(5)]),
    )

    deeponet = DeepONetArch(
        output_keys=[Key("psi")],
        branch_net=branch_net,
        trunk_net=trunk_net,
    )

    nodes = [deeponet.make_node("deepo")]
    # [init-model]

    # [datasets]
    # load training data
    DATA_PATH = "/root/projects/deeponet/data/g0.1-sigma_a3-sigma_t6.npz"
    BRANCH_KEYS = ["sigma", "boundary"]
    TRUNK_KEYS = ["phase_coords"]
    LABEL_KEY = "psi_label"

    np_data = dict(np.load(DATA_PATH, allow_pickle=True))

    np_data = preprocess(np_data, BRANCH_KEYS, TRUNK_KEYS, LABEL_KEY, repeat=2)

    # [datasets]

    # [constraint]
    # make domain
    domain = Domain()

    data = DeepONetConstraint.from_numpy(
        nodes=nodes,
        invar={k: np_data[k] for k in BRANCH_KEYS + TRUNK_KEYS},
        outvar={"psi": np_data[LABEL_KEY]},
        batch_size=cfg.batch_size.train,
    )
    domain.add_constraint(data, "data")
    # [constraint]

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
