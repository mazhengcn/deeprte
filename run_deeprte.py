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


import json
import pathlib
import time
from typing import Any

import dill
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags, logging
from matplotlib.colors import ListedColormap
from rte_dataset.builders import pipeline

from deeprte.configs import default
from deeprte.model.engine import RteEngine

logging.set_verbosity(logging.INFO)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_dir", None, "Model directory containing the params and config."
)
flags.DEFINE_string("data_path", None, "Path to directory containing the data.")
flags.DEFINE_string(
    "output_dir",
    None,
    "Path to output directory. If not specified, a directory will be created "
    "in the system's temporary directory.",
)
flags.DEFINE_bool("benchmark", True, "If True, benchmark the model.")
flags.DEFINE_integer("num_eval", None, "Number of examples to evaluate.")
flags.mark_flags_as_required(["model_dir", "data_path", "output_dir"])


def rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2) / np.mean(target**2))


def mse(pred, target):
    return np.mean((pred - target) ** 2)


def get_normalization_ratio(psi_range, boundary_range):
    psi_range = float(psi_range.split(" ")[-1])
    boundary_range = float(boundary_range.split(" ")[-1])
    return psi_range / boundary_range


def _jnp_to_np(output: dict[str, Any]) -> dict[str, Any]:
    """Recursively changes jax arrays to numpy arrays."""
    for k, v in output.items():
        if isinstance(v, dict):
            output[k] = _jnp_to_np(v)
        elif isinstance(v, jnp.ndarray):
            output[k] = np.array(v)
    return output


def plot_phi(r, phi_pre, phi_label, save_path):
    fig, _axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
    fig.subplots_adjust(hspace=0.3)
    axs = _axs.flatten()

    viridis = matplotlib.colormaps["viridis"](np.linspace(0, 1.2, 128))
    cs_1 = axs[0].contourf(
        r[..., 0], r[..., 1], phi_label, cmap=ListedColormap(viridis)
    )
    axs[0].set_title(r"Exact $f(r,\Omega)$", fontsize=20)
    axs[0].tick_params(axis="both", labelsize=15)
    cbar = fig.colorbar(cs_1)
    cbar.ax.tick_params(labelsize=16)

    cs_2 = axs[1].contourf(
        r[..., 0],
        r[..., 1],
        phi_pre,
        cmap=ListedColormap(viridis),
    )
    axs[1].set_title(r"Predict $f(r,\Omega)$", fontsize=20)
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
    plt.savefig(save_path)


def predict_radiative_transfer(
    output_dir_base: str,
    data_pipeline: pipeline.DataPipeline,
    engine: RteEngine,
    benchmark: bool,
    num_eval: int = None,
):
    # Get features.
    raw_feature_dict = data_pipeline.process()

    del data_pipeline
    num_examples = raw_feature_dict["shape"]["num_examples"]

    logging.info("Predicting %d examples sequentially", num_examples)

    if not num_eval:
        num_eval = num_examples

    for i in range(num_examples - num_eval, num_examples):
        timings = {}

        logging.info("Predicting example %d/%d", i + 1, num_examples)

        output_dir = output_dir_base / f"example_{i}"
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        feature_dict = {
            "functions": jax.tree.map(
                lambda x: x[i : i + 1], raw_feature_dict["functions"]
            ),
            "grid": raw_feature_dict["grid"],
            "shape": raw_feature_dict["shape"],
        }

        # Write out features as a pickled dictionary.
        features_output_path = output_dir / "features.dill"
        with open(features_output_path, "wb") as f:
            dill.dump(feature_dict, f)

        # Run the model.
        logging.info("Running rte engine...")
        t_0 = time.time()
        processed_feature_dict = engine.process_features(feature_dict)
        timings["process_features"] = time.time() - t_0

        t_0 = time.time()
        prediction = engine.predict(processed_feature_dict)
        t_diff = time.time() - t_0
        timings["predict_and_compile"] = t_diff

        if i == 0:
            logging.info(
                "Total JAX model predict time "
                "(includes compilation time, see --benchmark): %.6fs",
                t_diff,
            )
            if benchmark:
                t_0 = time.time()
                engine.predict(processed_feature_dict)
                t_diff = time.time() - t_0
                timings["predict_benchmark"] = t_diff
                logging.info(
                    "Total JAX model predict time (excludes compilation time): %.6fs",
                    t_diff,
                )
        else:
            logging.info("Total JAX model predict time: %.6fs", t_diff)

        psi_shape = feature_dict["functions"]["psi_label"].shape
        t_0 = time.time()
        predicted_psi = prediction.reshape(psi_shape)

        predicted_phi = jnp.sum(
            predicted_psi * feature_dict["grid"]["velocity_weights"],
            axis=-1,
        )
        t_diff = time.time() - t_0
        timings["compute_psi_and_phi"] = t_diff

        prediction_result = {
            "predicted_psi": predicted_psi,
            "predicted_phi": predicted_phi,
        }

        # Compute metrics.
        metrics = {}
        psi_label = feature_dict["functions"]["psi_label"]
        phi_label = np.sum(
            psi_label * feature_dict["grid"]["velocity_weights"], axis=-1
        )
        psi_rmse = rmse(predicted_psi, psi_label)
        phi_rmse = rmse(predicted_phi, phi_label)
        psi_mse = mse(predicted_psi, psi_label)
        phi_mse = mse(predicted_phi, phi_label)

        metrics.update(
            {
                "psi_rmse": str(psi_rmse),
                "phi_rmse": str(phi_rmse),
                "psi_mse": str(psi_mse),
                "phi_mse": str(phi_mse),
            }
        )
        logging.info("RMSE of psi: %f, RMSE of phi: %f\n", psi_rmse, phi_rmse)

        # Remove jax dependency from results.
        np_prediction_result = _jnp_to_np(dict(prediction_result))
        # Save the model outputs.
        np_result = {
            **np_prediction_result,
            "psi_label": psi_label,
            "phi_label": phi_label,
        }
        result_output_path = output_dir / "result.dill"
        with open(result_output_path, "wb") as f:
            dill.dump(np_result, f)

        metrics_output_path = output_dir / "metrics.json"
        with open(metrics_output_path, "w") as f:
            f.write(json.dumps(metrics, indent=4))

        timings_output_path = output_dir / "timings.json"
        with open(timings_output_path, "w") as f:
            f.write(json.dumps(timings, indent=4))

        figure_save_path = output_dir / "plot.png"
        plot_phi(
            feature_dict["grid"]["position_coords"].reshape(*psi_shape[1:-1], -1),
            predicted_phi[0],
            phi_label[0],
            figure_save_path,
        )


@app.run
def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    model_dir = pathlib.Path(FLAGS.model_dir)
    output_dir = pathlib.Path(FLAGS.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    logging.info("Writing config file...")
    record_config_path = output_dir / "config.json"
    model_config_path = model_dir / "config.json"
    record_config = {
        "config": str(model_config_path),
        "data_dir": FLAGS.data_path,
    }
    with record_config_path.open("w") as f:
        json.dump(record_config, f, indent=2)

    data_path = pathlib.Path(FLAGS.data_path)
    data_pipeline = pipeline.DataPipeline(data_path.parent, [data_path.name])
    logging.info("Data pipeline created from %s", FLAGS.data_path)

    model_config = default.get_config(model_config_path)
    model_config = model_config.replace(
        load_parameters_path=model_dir.resolve() / "params"
    )
    rte_engine = RteEngine(model_config)

    logging.info("Running prediction...")
    predict_radiative_transfer(
        output_dir, data_pipeline, rte_engine, FLAGS.benchmark, FLAGS.num_eval
    )


if __name__ == "__main__":
    main()
