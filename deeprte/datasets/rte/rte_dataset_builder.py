"""rte_dataset dataset."""
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from deeprte.data import pipeline
from deeprte.model.tf import rte_dataset, rte_features

os.environ["NO_GCE_CHECK"] = "true"

tfds.core.utils.gcs_utils._is_gcs_disabled = True


rte_features.register_feature(
    "psi_label", tf.float32, [rte_features.NUM_PHASE_COORDS]
)
rte_features.register_feature(
    "boundary_scattering_kernel",
    tf.float32,
    [rte_features.NUM_BOUNDARY_COORDS, rte_features.NUM_VELOCITY_COORDS],
)
FEATURES = rte_features.FEATURES
PHASE_FEATURE_AXIS = {
    k: FEATURES[k][1].index(rte_features.NUM_PHASE_COORDS)
    - len(FEATURES[k][1])
    for k in FEATURES
    if rte_features.NUM_PHASE_COORDS in FEATURES[k][1]
}
BOUNDARY_FEATURE_AXIS = {
    k: FEATURES[k][1].index(rte_features.NUM_BOUNDARY_COORDS)
    - len(FEATURES[k][1])
    for k in FEATURES
    if rte_features.NUM_BOUNDARY_COORDS in FEATURES[k][1]
}


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for rte_dataset dataset."""

    VERSION = tfds.core.Version("0.0.1")
    RELEASE_NOTES = {
        "0.0.1": "Initial release.",
    }
    BUILDER_CONFIGS = [
        # `name` (and optionally `description`) are required for each config
        tfds.core.BuilderConfig(
            name="g0.0-0.2", description="g0.0-0.2 dataset."
        )
    ]
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
        Please download the raw dataset to project_root/data/raw_data
    """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        feature_info_dict = {}
        for k, v in FEATURES.items():
            _, shape = v
            new_shape = [None if isinstance(k, str) else k for k in shape]
            feature_info_dict[k] = tfds.features.Tensor(
                shape=new_shape, dtype=np.float32, encoding="zlib"
            )
        self.metadata_dict = tfds.core.MetadataDict(
            {
                "phase_feature_axis": PHASE_FEATURE_AXIS,
                "boundary_feature_axis": BOUNDARY_FEATURE_AXIS,
            }
        )
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(feature_info_dict),
            metadata=self.metadata_dict,
            homepage="https://github.com/mazhengcn/deeprte",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        datasets_dir = dl_manager.manual_dir
        filenames = [file.name for file in datasets_dir.iterdir()]

        data_pipeline = pipeline.DataPipeline(datasets_dir, filenames)
        raw_data = data_pipeline.process(normalization=True)

        self.metadata_dict.update(
            {
                "normalization": tf.nest.map_structure(
                    lambda x: str(x), raw_data["normalization"]
                )
            }
        )

        return {
            "train": self._generate_examples(raw_data),
        }

    def _generate_examples(self, raw_data):
        """Yields examples."""

        num_examples = raw_data["shape"]["num_examples"]

        for i in range(num_examples):
            np_example = {
                **tf.nest.map_structure(lambda x: x[i], raw_data["functions"]),
                **raw_data["grid"],
            }
            tensor_dict = rte_dataset.np_to_tensor_dict(
                np_example, raw_data["shape"], FEATURES.keys()
            )
            yield i, tf.nest.map_structure(lambda x: x.numpy(), tensor_dict)
