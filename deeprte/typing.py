from collections.abc import Mapping
from typing import Any, NamedTuple, Union

import jax.numpy as jnp

Batch = Mapping[str, jnp.ndarray]
Data = Mapping[str, jnp.ndarray]
Logs = Mapping[str, jnp.float32]
State = Mapping[str, Any]
Metric = Mapping[str, Any]
Losses = Mapping[str, jnp.ndarray]


class F(NamedTuple):
    """Graph of a function (x, f(x)) as a namedtuple."""

    x: Union[jnp.float32, jnp.ndarray] = None
    y: jnp.ndarray = 0
