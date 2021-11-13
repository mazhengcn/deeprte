import abc
from functools import partial
from typing import Any, Callable, Optional, Tuple

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

    def apply(self, partial_args=None):

        if partial_args:
            _apply_fn = partial(self._apply, *partial_args)
        else:
            _apply_fn = self._apply

        return _apply_fn


class MultiSolutions(object, metaclass=abc.ABCMeta):
    """Multi-Solutions container used to create Haiku transformed
    pure functions.
    """

    def __init__(
        self,
        config: Tuple[ConfigDict],
        name: Optional[str] = "multi_solutions",
    ) -> None:

        self.name = name
        self.config = config

        self.init = hk.multi_transform(self.forward_fn).init
        self._apply = hk.multi_transform(self.forward_fn).apply

    @abc.abstractmethod
    def forward_fn(self) -> Tuple[TemplateFn, Any]:
        pass

    def apply(self, partial_args=None):

        if partial_args:
            _apply_fn = tuple(
                map(lambda x: partial(x, *partial_args), self._apply)
            )
        else:
            _apply_fn = self._apply

        return _apply_fn
