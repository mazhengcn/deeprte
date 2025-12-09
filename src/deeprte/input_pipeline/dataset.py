import grain.python as grain
import jax
import numpy as np
from rte_dataset.builders import pipeline

from deeprte.input_pipeline import splits
from deeprte.model.tf import rte_dataset
from deeprte.model.tf import rte_features as features

features.register_feature(
    "psi_label",
    np.dtype(np.float32),
    [features.NUM_PHASE_COORDS],  # ty: ignore
)  # ty: ignore

FEATURES = features.FEATURES
PHASE_FEATURE_AXIS = {
    k: FEATURES[k][1].index(features.NUM_PHASE_COORDS) - len(FEATURES[k][1])
    for k in FEATURES
    if features.NUM_PHASE_COORDS in FEATURES[k][1]
}


class RTEDataset(grain.RandomAccessDataSource):
    def __init__(self, raw_data) -> None:
        self.raw_data = raw_data

    def __len__(self) -> int:
        return self.raw_data["shape"]["num_examples"]

    def __getitem__(self, record_key):
        np_example = {
            **jax.tree.map(lambda x: x[record_key], self.raw_data["functions"]),
            **self.raw_data["grid"],
        }
        return rte_dataset.np_to_tensor_dict(
            np_example,
            self.raw_data["shape"],
            FEATURES.keys(),  # ty: ignore
        )

    def __repr__(self):
        return "RTEDataset: 0.0.2"

    @property
    def metadata(self) -> dict[str, dict[str, int]]:
        return {
            "shapes": self.raw_data["shape"],
            "normalization": jax.tree.map(
                lambda x: str(x), self.raw_data["normalization"]
            ),
        }


def get_datasets(dataset_name, data_dir, data_split: str) -> RTEDataset:
    """Load a dataset as grain datasource."""
    data_pipeline = pipeline.DataPipeline(data_dir, [dataset_name])
    raw_data = data_pipeline.process(normalization=True)

    num_examples = raw_data["shape"]["num_examples"]
    split_instr = splits.get_split_instruction(data_split, num_examples)  # ty: ignore

    raw_data["functions"] = jax.tree.map(
        lambda x: x[split_instr.from_ : split_instr.to], raw_data["functions"]
    )
    raw_data["shape"]["num_examples"] = raw_data["functions"]["psi_label"].shape[0]

    return RTEDataset(raw_data)  # ty: ignore
