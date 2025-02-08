import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as orbax
from absl import flags, logging
from flax import nnx
from jax.sharding import PartitionSpec as P
from rte_dataset.builders import pipeline

from deeprte.configs import default
from deeprte.model import features
from deeprte.model.autoencoder import AutoEncoder
from deeprte.model.mapping import inference_subbatch
from deeprte.model.tf import rte_features
from deeprte.train_lib import utils

# JAX_TRACEBACK_FILTERING = off
logging.set_verbosity(logging.INFO)

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", None, "Path to the configuration file.")
flags.DEFINE_string("data_path", None, "Path to directory containing the data.")
flags.DEFINE_string(
    "output_dir",
    None,
    "Path to output directory. If not specified, a directory will be created "
    "in the system's temporary directory.",
)
flags.mark_flags_as_required(["config", "data_path", "output_dir"])


def postprocess(model_path, data_path, output_dir):
    # 配置初始化

    config = default.get_config(model_path + "/config.yaml")
    config.load_parameters_path = (model_path + "/infer/params",)

    # 设备和分片设置
    devices_array = utils.create_device_mesh(config, devices=jax.local_devices())
    mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

    replicated_sharding = jax.sharding.NamedSharding(mesh, P(None))
    data_sharding = jax.sharding.NamedSharding(mesh, P(None, *config.data_sharding))

    feature_sharding = {
        k: data_sharding
        if k in rte_features.PHASE_COORDS_FEATURES
        else replicated_sharding
        for k in rte_features.FEATURES
    }

    # 模型初始化
    rngs = jax.random.key(0)

    state, state_sharding = utils.get_abstract_state(
        constructor=lambda config, rngs: AutoEncoder(
            config, rngs=nnx.Rngs(params=rngs)
        ),
        tx=None,
        config=config,
        rng=rngs,
        mesh=mesh,
        is_training=False,
    )

    checkpointer = orbax.PyTreeCheckpointer()
    _params = {}
    for path in config.load_parameters_path:
        _res_params = checkpointer.restore(path)
        _params = _params | _res_params["params"]

    graphdef, state = utils.module_from_variables_dict(
        lambda: nnx.eval_shape(lambda: AutoEncoder(config, rngs=nnx.Rngs(params=rngs))),
        _params,
        lambda path: path[:-1] if path[-1] == "value" else path,
    )

    params = utils.init_infer_state(None, state).params

    def forward_fn(params, features, graphdef):
        model = nnx.merge(graphdef, params)

        def func(features):
            moments = model.encoder(
                features["source"],
                [features["source_coords"], features["source_weights"]],
            )
            # print(moments.shape)
            basis = model.mlp(features["source_coords"])
            # print(basis.shape)
            return basis, moments

        return jax.vmap(func)(features)

    jit_predict_fn = jax.jit(
        forward_fn,
        in_shardings=(state_sharding.params, feature_sharding),
        out_shardings=data_sharding,
        static_argnums=2,
    )

    # 数据加载和处理
    data_pipeline = pipeline.DataPipeline(data_path.parent, [data_path.name])
    raw_feature_dict = data_pipeline.process()
    del data_pipeline
    # 获取总样本数
    num_samples = raw_feature_dict["shape"]["num_examples"]
    results = []

    # 使用tqdm添加进度条
    for i in range(num_samples):
        # i = 0
        feature_dict = {
            "functions": jax.tree.map(
                lambda x: x[i : i + 1], raw_feature_dict["functions"]
            ),
            "grid": raw_feature_dict["grid"],
            "shape": raw_feature_dict["shape"],
        }

        processed_feature_dict = features.np_data_to_features(feature_dict)
        phase_feat, other_feat = features.split_feature(
            processed_feature_dict,
            filter=lambda x: x in rte_features.SOURCE_COORDS_FEATURES,
        )
        # 推理
        out = inference_subbatch(
            module=lambda x: jit_predict_fn(params, x, graphdef),
            subbatch_size=128,
            batched_args=phase_feat,
            nonbatched_args=other_feat,
            low_memory=False,
        )

        result = {}
        result["moments"] = out[1]
        result["basis_inner_product"] = jnp.einsum(
            "...ij,...ik->...jk",
            phase_feat["source_weights"][..., None] * out[0],
            out[0],
        )
        # result["basis_weights"] = jnp.sum(out[1], axis=-2)

        results.append(jax.tree.map(lambda x: np.array(x), result))

        logging.info(f"Processed sample {i + 1}/{num_samples}")

    del raw_feature_dict

    all_results = jax.tree.map(lambda *xs: np.concatenate(xs, axis=0), *results)

    # 保存结果
    np.savez(output_dir / "result.npz", **all_results)
    logging.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    output_dir = pathlib.Path(FLAGS.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    data_path = pathlib.Path(FLAGS.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path {data_path} does not exist")

    postprocess(FLAGS.model_path, data_path, output_dir)
