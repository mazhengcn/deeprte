import jax
import jax.numpy as jnp
from flax import nnx

from deeprte.model.characteristics import Characteristics
from deeprte.model.layers import MlpBlock, MultiHeadAttention

kernel_init = nnx.initializers.glorot_uniform()
bias_init = nnx.initializers.zeros_init()


class Attenuation(nnx.Module):
    def __init__(self, config, *, rngs: nnx.Rngs):
        self.config = config

        attention_in_dims = [
            config.position_coords_dim + config.velocity_coords_dim,
            config.position_coords_dim,
            config.coeffs_fn_dim,
        ]

        self.attention = MultiHeadAttention(
            config.num_heads,
            attention_in_dims,
            config.qkv_dim,
            config.optical_depth_dim,
            rngs=rngs,
        )
        self.mlp = MlpBlock(config=config, rngs=rngs)

    def __call__(
        self,
        coord1: jax.Array,
        coord2: jax.Array,
        att_coeff: jax.Array,
        charac: Characteristics,
    ):
        local_coords, mask = charac.apply_to_point(coord1)
        optical_depth = self.attention(
            inputs_q=coord1, inputs_k=local_coords, inputs_v=att_coeff, mask=mask
        )
        optical_depth = jnp.exp(-optical_depth)

        attenuation = jnp.concatenate([coord1, coord2, optical_depth])
        attenuation = self.mlp(attenuation)
        return attenuation


class ScatteringLayer(nnx.Module):
    "A single layer describing scattering action."

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nnx.Linear(
            in_features,
            out_features,
            kernel_init=kernel_init,
            bias_init=bias_init,
            rngs=rngs,
        )

    def __call__(self, inputs: jax.Array, kernel: jax.Array) -> jax.Array:
        x = jnp.einsum("...V,Vd->...d", kernel, inputs)
        x = self.linear(x)
        x = nnx.tanh(x)
        return x


class Scattering(nnx.Module):
    """Scattering block."""

    def __init__(self, config, *, rngs: nnx.Rngs):
        self.config = config

        scattering_layers, lns = [], []
        for _ in range(config.num_scattering_layers):
            scattering_layers.append(
                ScatteringLayer(config.scattering_dim, config.scattering_dim, rngs=rngs)
            )
            lns.append(nnx.LayerNorm(config.scattering_dim, rngs=rngs))
        self.scattering_layers = scattering_layers
        self.lns = lns

    def __call__(
        self,
        act: jax.Array,
        self_act: jax.Array,
        kernel: jax.Array,
        self_kernel: jax.Array,
    ) -> jax.Array:
        self_act_0 = self_act
        for idx in range(self.config.num_scattering_layers - 1):
            self_act = self.scattering_layers[idx](self_act, self_kernel)
            self_act = self.lns[idx](self_act)
            self_act += self_act_0
        act_res = self.scattering_layers[-1](self_act, kernel)
        act_res = self.lns[-1](act_res)
        act += act_res
        return act
