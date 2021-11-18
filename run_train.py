import os
import random
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
from absl import app, logging
from ml_collections import ConfigDict

from deeprte.input_pipeline import create_tf_dataset

# get_numpy_dataset
from deeprte.models.rte_op import RTEModel, RTEOperator
from deeprte.modules.green_fn import GreenFunction
from deeprte.solver import Solver
from deeprte.utils import to_flat_dict

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "4.0"

ROOT_PATH = "/workspace/deeprte/"


def main(_):

    # original_ds = get_numpy_dataset(ROOT_PATH + "data/rte/rte_2d")

    train_ds, test_ds, init_ds = create_tf_dataset(
        ROOT_PATH + "data/rte/rte_2d_converted",
        10,
        num_collocation_pts=500,
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
    eqn = RTEModel(name="lte")

    solver = Solver(sol, eqn)

    opt = optax.adam(1e-3)

    # schedule_fn = optax.exponential_decay(-1e-3, 500, 0.96)
    # opt = optax.chain(
    #     optax.scale_by_adam(eps=1e-7), optax.scale_by_schedule(schedule_fn)
    # )

    def loss_fn(x, y):
        return jnp.mean(jnp.square(x - y))

    solver.compile(loss_fn, opt, {"residual": 1})

    def train_load():
        yield from tfds.as_numpy(train_ds)

    logging.info("Begin training process.")

    solver.solve(
        dataset=train_load,
        init_data=init_ds,
        num_epochs=1000,
        steps_per_epoch=40,
        val_data=(val_data["interior"], val_data["label"]),
        val_freq=20,
        seed=random.randrange(sys.maxsize),
    )

    flat_params = to_flat_dict(solver.params)

    np.savez(ROOT_PATH + "/data/rte/params/rte_2d_1.npz", **flat_params)


if __name__ == "__main__":
    app.run(main)
