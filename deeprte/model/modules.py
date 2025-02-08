import jax
from flax import nnx

from deeprte.model.autoencoder import AutoEncoder
from deeprte.model.config import DeepRTEConfig
from deeprte.model.deeprte import DeepRTE


def constructor(config: DeepRTEConfig, key: jax.Array) -> nnx.Module:
    if config.name == "deeprte":
        return DeepRTE(config, rngs=nnx.Rngs(params=key))
    elif config.name == "source" or config.name == "boundary":
        return AutoEncoder(config, rngs=nnx.Rngs(params=key))
    else:
        raise ValueError(f"Unknown model name: {config.name}")
