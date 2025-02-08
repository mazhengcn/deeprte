import dataclasses
import functools
from collections.abc import Callable, Mapping
from typing import Any, Optional

import jax
import jax.numpy as jnp
from flax import nnx

from deeprte.model.config import DeepRTEConfig
from deeprte.model.tf import rte_features as features

Shape = tuple[int, ...]
Dtype = Any
Array = jax.Array

kernel_init = nnx.initializers.glorot_uniform()
bias_init = nnx.initializers.zeros_init()


class MlpBlock(nnx.Module):
    """MLP / feed-forward block."""

    def __init__(
        self, units: list[int], activation: Callable[[Array], Array], rngs: nnx.Rngs
    ):
        self.activation = activation
        layers = []
        for idx in range(len(units) - 1):
            layers.append(
                nnx.Linear(
                    units[idx],
                    units[idx + 1],
                    kernel_init=kernel_init,
                    bias_init=bias_init,
                    rngs=rngs,
                )
            )
            if idx < len(units) - 2:
                layers.append(self.activation)
        self.layers = nnx.Sequential(*layers)

    def __call__(self, x: Array):
        x = self.layers(x)
        return x


class AutoEncoder(nnx.Module):
    """Boundary basis representation."""

    def __init__(self, config: DeepRTEConfig, rngs: nnx.Rngs):
        self.config = config
        self.features = features.AUTOENCODER_FEATURES

        self.in_features = config.position_coords_dim + config.velocity_coords_dim
        self.out_features = config.num_basis_functions

        self.mlp_dim = config.basis_function_encoder_dim
        self.num_layers = config.num_basis_function_encoder_layers

        self.mlp = MlpBlock(
            units=[self.in_features]
            + [self.mlp_dim] * self.num_layers
            + [self.out_features],
            activation=nnx.tanh,
            rngs=rngs,
        )

    def encoder(self, f: jax.Array, quadratures) -> jax.Array:
        points, weights = quadratures
        out = jax.vmap(self.mlp, in_axes=(0,), out_axes=-1)(points)
        return jnp.dot(out, f * weights)

    def decoder(self, m: jax.Array, points: jax.Array) -> jax.Array:
        basis = self.mlp(points)
        return basis @ m

    def __call__(self, batch: Mapping[str, Array]):
        inputs = {k: batch[k] for k in self.features}

        @jax.vmap
        def autoencoder_op(inputs):
            moments = self.encoder(
                inputs["source"], [inputs["source_coords"], inputs["source_weights"]]
            )
            out = self.decoder(moments, inputs["phase_coords"])
            return out

        return autoencoder_op(inputs)
