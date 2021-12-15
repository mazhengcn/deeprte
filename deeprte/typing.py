from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

Batch = Mapping[str, jnp.ndarray]
Data = Mapping[str, jnp.ndarray]
Logs = Mapping[str, jnp.float32]
State = Mapping[str, Any]
Metric = Mapping[str, Any]
Losses = Mapping[str, jnp.ndarray]
