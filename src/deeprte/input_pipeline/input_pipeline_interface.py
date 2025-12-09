import jax

from deeprte.input_pipeline import utils
from deeprte.input_pipeline.data_processing import preprocessing_pipeline
from deeprte.input_pipeline.dataset import get_datasets


def create_data_iterator(config):
    """load dataset, preprocess and return iterators"""

    train_ds = get_datasets(
        dataset_name=config.dataset_name,
        data_dir=config.data_dir,
        data_split=config.train_split,
    )
    norm_dict = train_ds.metadata["normalization"]
    config.normalization = utils.get_normalization_ratio(
        norm_dict["psi_range"], norm_dict["boundary_range"]
    )

    train_iter = preprocessing_pipeline(
        dataset=train_ds,
        global_batch_size=config.global_batch_size,
        collocation_size=config.collocation_size,
        sharding=jax.P(config.mesh_axis_names),
        worker_count=config.grain_worker_count,
        worker_buffer_size=config.grain_worker_buffer_size,
        shuffle=config.enable_data_shuffling,
        num_epochs=None,
        data_shuffle_seed=config.data_shuffle_seed,
    )

    if config.eval_every_steps > 0:
        eval_ds = get_datasets(
            dataset_name=config.dataset_name,
            data_dir=config.data_dir,
            data_split=config.eval_split,
        )

        eval_iter = preprocessing_pipeline(
            dataset=eval_ds,
            global_batch_size=config.eval_batch_size,
            sharding=jax.P(config.mesh_axis_names),
            worker_count=config.grain_worker_count,
            worker_buffer_size=config.grain_worker_buffer_size,
            shuffle=False,
            data_shuffle_seed=config.data_shuffle_seed,
        )
    else:
        eval_iter = None

    return train_iter, eval_iter
