from typing import Any, Mapping, NamedTuple

import jax.numpy as jnp

Batch = Mapping[str, jnp.ndarray]
Data = Mapping[str, jnp.ndarray]
Logs = Mapping[str, jnp.float32]
State = Mapping[str, Any]
Metric = Mapping[str, Any]
Losses = Mapping[str, jnp.ndarray]


class GraphOfMapping(NamedTuple):
    x: Any = None
    fx: jnp.ndarray = 0
