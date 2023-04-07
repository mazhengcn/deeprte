"""rte_dataset dataset."""
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from deeprte.data import pipeline
from deeprte.model.tf import rte_dataset, rte_features

os.environ["NO_GCE_CHECK"] = "true"

tfds.core.utils.gcs_utils._is_gcs_disabled = True


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for rte_dataset dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
        test
    """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        feature_info_dict = {}
        for k, v in rte_features.FEATURES.items():
            dtype, shape = v
            new_shape = [None if isinstance(k, str) else k for k in shape]
            feature_info_dict[k] = tfds.features.Tensor(
                shape=new_shape, dtype=np.float32, encoding="zlib"
            )

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(feature_info_dict),
            homepage="https://github.com/mazhengcn/deeprte",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        datasets_dir = dl_manager.manual_dir
        filenames = [file.name for file in datasets_dir.iterdir()]

        data_pipeline = pipeline.DataPipeline(datasets_dir, filenames)
        raw_data = data_pipeline.process()

        return {
            "train": self._generate_examples(raw_data),
        }

    def _generate_examples(self, raw_data):
        """Yields examples."""

        num_examples = raw_data["shape"].pop("num_examples")

        for i in range(num_examples):
            np_example = {
                **tf.nest.map_structure(lambda x: x[i], raw_data["functions"]),
                **raw_data["grid"],
            }
            tensor_dict = rte_dataset.np_to_tensor_dict(
                np_example, raw_data["shape"]
            )
            yield i, tf.nest.map_structure(lambda x: x.numpy(), tensor_dict)
