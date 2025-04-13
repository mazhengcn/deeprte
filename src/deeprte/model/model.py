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

"""Core modules including Green's function with Attenuation and Scattering."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from flax import nnx

from deeprte.model import integrate
from deeprte.model.characteristics import Characteristics
from deeprte.model import features
from deeprte.model.modules import Attenuation, Scattering

kernel_init = nnx.initializers.glorot_uniform()
bias_init = nnx.initializers.zeros_init()


@dataclasses.dataclass(unsafe_hash=True)
class DeepRTEConfig:
    # Physical position dimensions.
    position_coords_dim: int = 2
    # Physical velocity dimensions.
    velocity_coords_dim: int = 2
    # Dimensions of (scattering) coefficient functions.
    coeffs_fn_dim: int = 2
    # Number of attention heads.
    num_heads: int = 2
    # Attention dimension.
    qkv_dim: int = 64
    # Number of MLP layers.
    num_mlp_layers: int = 4
    # MLP dimension.
    mlp_dim: int = 128
    # Number of scattering layers.
    num_scattering_layers: int = 2
    # Scattering dimension.
    scattering_dim: int = 16
    # Output dimensions of attention.
    optical_depth_dim: int = 1 + scattering_dim
    # Subcollocation size for evaluation or inference
    subcollocation_size: int = 128
    # Normalization constant of dataset/model.
    normalization: float = 1.0
    # Where to load the parameters from.
    load_parameters_path: str = ""

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


class GreenFunction(nnx.Module):
    """Green's function."""

    def __init__(self, config, *, rngs: nnx.Rngs):
        self.config = config
        self.attenuation = Attenuation(self.config, rngs=rngs)
        self.scattering = Scattering(self.config, rngs=rngs)
        self.out = nnx.Linear(
            self.config.scattering_dim,
            1,
            kernel_init=kernel_init,
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, coord1: jax.Array, coord2: jax.Array, batch):
        charac = Characteristics.from_tensor(batch["position_coords"])
        act, tau_s = self.attenuation(
            coord1=coord1, coord2=coord2, att_coeff=batch["sigma"], charac=charac
        )
        if self.config.num_scattering_layers == 0:
            out = jnp.squeeze(jnp.exp(self.out(act)), axis=-1)
            return out

        position, _ = jnp.split(coord1, 2, axis=-1)

        def self_att_fn(velocity):
            coord = jnp.concatenate([position, velocity], axis=-1)
            out, _ = self.attenuation(
                coord1=coord, coord2=coord2, att_coeff=batch["sigma"], charac=charac
            )
            return out

        velocity_coords, velocity_weights = (
            batch["velocity_coords"],
            batch["velocity_weights"],
        )
        kernel = batch["scattering_kernel"] * velocity_weights
        self_kernel = batch["self_scattering_kernel"] * velocity_weights

        self_act = jax.vmap(self_att_fn)(velocity_coords)
        act = self.scattering(act, self_act, kernel, self_kernel, tau_s)

        out = jnp.squeeze(jnp.exp(self.out(act)), axis=-1)
        return out


class DeepRTE(nnx.Module):
    """Deep RTE model."""

    def __init__(self, config: DeepRTEConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.features = features.FEATURES
        self.phase_coords_axes = {
            k: 0 if k in features.get_phase_coords_features() else None
            for k in self.features
        }
        self.green_fn = GreenFunction(config=self.config, rngs=rngs)

    def __call__(self, batch):
        rte_inputs = {k: batch[k] for k in self.features}

        def rte_op(inputs):
            quadratures = (
                inputs["boundary_coords"],
                inputs["boundary"] * inputs["boundary_weights"],
            )
            rte_sol = integrate.quad(self.green_fn, quadratures=quadratures, argnum=1)(
                inputs["phase_coords"], inputs
            )
            return rte_sol

        batched_rte_op = jax.vmap(jax.vmap(rte_op, in_axes=(self.phase_coords_axes,)))
        return batched_rte_op(rte_inputs)
