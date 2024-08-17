"""Input pipeline"""

import jax
import numpy as np
import tensorflow as tf
from jax.sharding import PartitionSpec as P

from deeprte.input_pipeline._grain_data_processing import make_grain_iterator
from deeprte.input_pipeline._tfds_data_processing import make_tfds_iterator
from deeprte.train_lib import multihost_dataloading

NUM_DIM = 2
# Placeholder values that will be replaced with their true value at runtime.
NUM_POSITION_COORDS = 1600
NUM_VELOCITY_COORDS = 24
NUM_PHASE_COORDS = 128
NUM_BOUNDARY_COORDS = 1920


class SyntheticDataIterator:
    """Creates a synthetic data iterator for performance testing work"""

    def __init__(self, config, mesh):
        self.mesh = mesh
        self.config = config
        data_pspec = P(*config.data_sharding)
        data_pspec_shardings = jax.tree_util.tree_map(
            lambda p: jax.sharding.NamedSharding(mesh, p), data_pspec
        )
        self.data_generator = jax.jit(
            SyntheticDataIterator.raw_generate_synthetic_data,
            out_shardings=data_pspec_shardings,
            static_argnums=0,
        )

    def __iter__(self):
        return self

    def __next__(self):
        with self.mesh:
            return self.data_generator(self.config)

    @staticmethod
    def raw_generate_synthetic_data(config):
        """Generates a single batch of synthetic data"""
        output = {}
        output["boundary"] = jax.numpy.zeros(
            (config.global_batch_size, NUM_BOUNDARY_COORDS)
        )
        output["boundary_coords"] = jax.numpy.zeros(
            (config.global_batch_size, NUM_BOUNDARY_COORDS, 2 * NUM_DIM)
        )
        output["boundary_weights"] = jax.numpy.ones(
            (config.global_batch_size, NUM_BOUNDARY_COORDS)
        )
        output["phase_coords"] = jax.numpy.zeros(
            (config.global_batch_size, NUM_PHASE_COORDS, 2 * NUM_DIM)
        )
        output["position_coords"] = jax.numpy.zeros(
            (config.global_batch_size, NUM_POSITION_COORDS, NUM_DIM),
        )
        output["psi_label"] = jax.numpy.ones(
            (config.global_batch_size, NUM_PHASE_COORDS)
        )
        output["scattering_kernel"] = jax.numpy.ones(
            (config.global_batch_size, NUM_PHASE_COORDS, NUM_VELOCITY_COORDS)
        )
        output["self_scattering_kernel"] = jax.numpy.ones(
            (config.global_batch_size, NUM_VELOCITY_COORDS, NUM_VELOCITY_COORDS)
        )
        output["sigma"] = jax.numpy.ones(
            (config.global_batch_size, NUM_POSITION_COORDS, NUM_DIM)
        )
        output["velocity_coords"] = jax.numpy.ones(
            (config.global_batch_size, NUM_VELOCITY_COORDS, NUM_DIM)
        )
        output["velocity_weights"] = jax.numpy.ones(
            (config.global_batch_size, NUM_VELOCITY_COORDS)
        )
        return output


class BadSyntheticDataIterator:
    """Creates a Bad synthetic data iterator for loading on subset of hosts"""

    def __init__(self, config, mesh):
        self.mesh = mesh
        dataset = BadSyntheticDataIterator.get_bad_synthetic_data(config)
        self.data_generator = multihost_dataloading.MultiHostDataLoadIterator(
            dataset, self.mesh
        )

    def __iter__(self):
        return self.data_generator

    def __next__(self):
        return next(self.data_generator)

    @staticmethod
    def get_bad_synthetic_data(config):
        """fill negative value in synthetic data"""
        output = {}
        output["inputs"] = tf.data.Dataset.from_tensor_slices(
            np.full((1, config.max_target_length), -1, dtype=jax.numpy.int32)
        )
        output = {}
        output["boundary"] = tf.data.Dataset.from_tensor_slices(
            np.full((1, NUM_BOUNDARY_COORDS), -1)
        )
        output["boundary_coords"] = tf.data.Dataset.from_tensor_slices(
            np.full((1, NUM_BOUNDARY_COORDS, 2 * NUM_DIM), -1)
        )
        output["boundary_weights"] = tf.data.Dataset.from_tensor_slices(
            np.full((1, NUM_BOUNDARY_COORDS), -1)
        )
        output["phase_coords"] = tf.data.Dataset.from_tensor_slices(
            np.full((1, NUM_PHASE_COORDS, 2 * NUM_DIM), -1)
        )
        output["position_coords"] = tf.data.Dataset.from_tensor_slices(
            np.full((1, NUM_POSITION_COORDS, NUM_DIM), -1)
        )
        output["psi_label"] = tf.data.Dataset.from_tensor_slices(
            np.full((1, NUM_PHASE_COORDS), -1)
        )
        output["scattering_kernel"] = tf.data.Dataset.from_tensor_slices(
            (np.full((1, NUM_PHASE_COORDS, NUM_VELOCITY_COORDS), -1),)
        )
        output["self_scattering_kernel"] = tf.data.Dataset.from_tensor_slices(
            (np.full((1, NUM_VELOCITY_COORDS, NUM_VELOCITY_COORDS), -1),)
        )
        output["sigma"] = tf.data.Dataset.from_tensor_slices(
            np.full((1, NUM_POSITION_COORDS, NUM_DIM), -1)
        )
        output["velocity_coords"] = tf.data.Dataset.from_tensor_slices(
            np.full((1, NUM_VELOCITY_COORDS, NUM_DIM), -1)
        )
        output["velocity_weights"] = tf.data.Dataset.from_tensor_slices(
            np.full((1, NUM_VELOCITY_COORDS), -1)
        )
        dataset = tf.data.Dataset.zip((output))  # pytype: disable=wrong-arg-types
        dataset = dataset.repeat()
        dataset = dataset.batch(config.global_batch_size // jax.process_count())
        return dataset


def get_process_loading_real_data(config, mesh):
    """Get list of processes loading data from GCS when expansion_factor_real_data != -1"""
    sharding = jax.sharding.NamedSharding(mesh, P(*config.data_sharding))
    devices_indices_map = sharding.devices_indices_map((config.global_batch_size,))
    batch_cutoff = config.global_batch_size
    process_loading_real_data = set()
    for p, indices in devices_indices_map.items():
        if indices[0].stop <= batch_cutoff:
            process_loading_real_data.add(p.process_index)
    return list(process_loading_real_data), sharding


def make_mixed_train_iterator(config, mesh):
    """Return iterators according to dataset_type"""
    process_indices, sharding = get_process_loading_real_data(config, mesh)
    if jax.process_index() in process_indices:
        if config.dataset_type == "tfds":
            return make_tfds_iterator(config, mesh, process_indices), sharding
        elif config.dataset_type == "grain":
            return make_grain_iterator(config, mesh, process_indices), sharding
    else:
        return BadSyntheticDataIterator(config, mesh), None


def create_data_iterator(config, mesh):
    if config.dataset_type == "synthetic":
        return SyntheticDataIterator(config, mesh), None
    elif config.dataset_type in ("tfds", "grain"):
        return make_mixed_train_iterator(config, mesh)
    else:
        assert False, f"Unknown dataset_type {config.dataset_type}, dataset_type must be synthetic, tfds"
