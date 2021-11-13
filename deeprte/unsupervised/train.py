import os
import random
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app, logging

# get_numpy_dataset
from deeprte.models.rte_op import RTEOperator, RTEOpUnsupervised
from deeprte.modules.green_fn import GreenFunction
from deeprte.solver import Solver
from deeprte.utils import to_flat_dict
from ml_collections import ConfigDict

from input_pipeline import create_tf_dataset

# RTEModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "4.0"

ROOT_PATH = "/workspace/modnet/"

logging.set_verbosity("info")


def main(_):

    # original_ds = get_numpy_dataset(ROOT_PATH + "data/rte/rte_2d"

    data_dict = np.load(ROOT_PATH + "data/rte/rte_2d_converted.npz")
    print(list(data_dict.keys()))
    c, s, omega = data_dict["c"], data_dict["s"], data_dict["omega"]

    train_ds, test_ds, init_ds = create_tf_dataset(
        ROOT_PATH + "data/rte/rte_2d_converted",
        2,  # 10,
        num_collocation_pts=200,  # 500,
        num_boundary_pts=40,
        test_batch_size=100,
    )

    for data in test_ds.take(1).as_numpy_iterator():
        val_data = data

    print(jax.tree_map(lambda x: x.shape, val_data))

    config = ConfigDict(
        {
            "green_net": [128, 128, 128, 128, 1],
            "coeffs_net": {"weights": [64, 1], "coeffs": [64, 2]},
        }
    )
    sol = RTEOperator(config, GreenFunction)
    # eqn = RTEModel(name="rte_2d")
    eqn = RTEOpUnsupervised(
        cs=jnp.concatenate([c, s], axis=-1),
        omega=omega,
        name="rte_2d_unsupervised",
    )

    solver = Solver(sol, eqn)

    # opt = optax.adam(1e-4)

    schedule_fn = optax.exponential_decay(-1e-4, 400, 0.96)
    opt = optax.chain(
        optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn)
    )

    def loss_fn(x, y):
        return jnp.mean(jnp.square(x - y))

    solver.compile(loss_fn, opt, {"residual": 1, "boundary": 10.0})

    logging.info("Begin training process.")

    solver.solve(
        dataset=train_ds,
        init_data=init_ds,
        num_epochs=1000,
        val_data=(val_data["interior"], val_data["label"]),
        val_freq=5,
        seed=random.randrange(sys.maxsize),
    )

    flat_params = to_flat_dict(solver.params)

    np.savez(
        ROOT_PATH + "/data/rte/params/rte_2d_unsupervised_1.npz", **flat_params
    )


if __name__ == "__main__":
    app.run(main)
