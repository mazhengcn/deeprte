import abc
from collections.abc import Callable
from typing import Any, Optional

import haiku as hk
import jax.numpy as jnp
from ml_collections import ConfigDict

TemplateFn = Callable[..., Any]


class Solution(object, metaclass=abc.ABCMeta):
    """Solution container used to create Haiku transformed pure function."""

    def __init__(
        self,
        config: ConfigDict,
        name: Optional[str] = "solution",
    ) -> None:
        self.name = name
        self.config = config

        self.init = hk.transform(self.forward_fn).init
        self._apply = hk.transform(self.forward_fn).apply

    @abc.abstractmethod
    def forward_fn(self) -> jnp.ndarray:
        pass

    @abc.abstractmethod
    def apply(
        self,
        params: hk.Params,
        state: hk.State,
        rng: jnp.ndarray,
        *,
        is_training: bool
    ) -> jnp.ndarray:
        pass


class MultiSolutions(object, metaclass=abc.ABCMeta):
    """Multi-Solutions container used to create Haiku transformed
    pure functions.
    """

    def __init__(
        self,
        config: tuple[ConfigDict],
        name: Optional[str] = "multi_solutions",
    ) -> None:

        self.name = name
        self.config = config

        self.init = hk.multi_transform(self.forward_fn).init
        self._apply = hk.multi_transform(self.forward_fn).apply

    @abc.abstractmethod
    def forward_fn(self) -> tuple[TemplateFn, Any]:
        pass

    @abc.abstractmethod
    def apply(self, params, state, rng, *, is_training):
        pass
