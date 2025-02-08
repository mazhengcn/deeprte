from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from deeprte.model.autoencoder import AutoEncoder
from deeprte.model.config import DeepRTEConfig
from deeprte.model.deeprte import GreenFunction
from deeprte.model.tf import rte_features

Shape = tuple[int, ...]
Dtype = Any
Array = jax.Array


class RtePredictor(nnx.Module):
    """RTE predictor."""

    def __init__(self, config: DeepRTEConfig, rngs: nnx.Rngs):
        self.config = config
        self.features = rte_features.FEATURES
        # self.basis_features = rte_features.BASIS_FEATURES
        self.phase_coords_axes = {
            k: 0 if k in rte_features.PHASE_COORDS_FEATURES else None
            for k in self.features
        }
        self.basis_coords_axes = {
            k: 0 if k in rte_features.AUTOENCODER_FEATURES else None
            for k in self.features
        }
        self.autoencoder = AutoEncoder(config=config, rngs=rngs)

        # self.basis_function = self.autoencoder.mlp
        self.green_function = GreenFunction(config=config, rngs=rngs)

    def __call__(self, batch: Mapping[str, Array]):
        inputs = {k: batch[k] for k in self.features}

        def basis_op(features):
            moments = self.autoencoder.encoder(
                features["source"],
                [features["source_coords"], features["source_weights"]],
            )
            basis = self.autoencoder.mlp(features["source_coords"])
            basis_inner_product = jnp.einsum(
                "...ij,...ik->...jk",
                features["source_weights"][..., None] * basis,
                basis,
            )
            # print(moments.shape)
            # print(basis.shape)
            return basis_inner_product, moments

        def green_function_op(inputs):
            return self.green_function(inputs["phase_coords"], inputs)

        batched_green_function_op = jax.vmap(
            green_function_op, in_axes=(self.phase_coords_axes,)
        )

        def rte_op(inputs):
            act = batched_green_function_op(inputs)
            inner_product, weights = basis_op(inputs)
            # print(act.shape, inner_product.shape, weights.shape)

            return jnp.einsum(
                "...bj,...j->...b",
                jnp.einsum("bi,ij->bj", act, inner_product),
                weights,
            )

        batched_rte_op = jax.vmap(rte_op)

        return batched_rte_op(inputs)
