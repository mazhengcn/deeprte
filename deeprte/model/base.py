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


import abc
import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, Optional

import haiku as hk
import jax.numpy as jnp
from ml_collections import ConfigDict

TemplateFn = Callable[..., Any]


class Model(object, metaclass=abc.ABCMeta):
    """Container class for the model that describes the equations, etc.
    At least the `self.loss` method should be implemented.
    """

    def __init__(self, name: Optional[str]):
        self._loss_fn = None
        self._regs = None

    @abc.abstractmethod
    def loss(
        self,
        fun,
        batch: Mapping[str, jnp.ndarray],
        rng: Optional[jnp.ndarray] = None,
    ) -> tuple[jnp.float32, Mapping[str, jnp.ndarray]]:
        pass

    @abc.abstractmethod
    def metrics(self, op_fn: Callable[..., Any], inputs):
        pass


class Solution(object, metaclass=abc.ABCMeta):
    """Solution container used to create Haiku transformed pure function."""

    def __init__(
        self,
        config: ConfigDict,
        name: Optional[str] = "solution",
    ) -> None:
        self.name = name
        self.config = config

        self._init = hk.transform_with_state(self.forward_fn).init
        self._apply = hk.transform_with_state(self.forward_fn).apply

    @abc.abstractmethod
    def forward_fn(self) -> jnp.ndarray:
        """The forward pass of solution."""

    @abc.abstractmethod
    def apply(
        self, params: hk.Params, state: hk.State, rng: jnp.ndarray, *, is_training: bool
    ) -> jnp.ndarray:
        """Apply function of solution."""

    @property
    def init(self):
        return self._init


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


@dataclasses.dataclass
class SolutionV2:
    """Holds a pair of pure functions defining a solution operator.

    Attributes:
        init: A pure function: ``params = init(rng, *a, **k)``
        apply: A pure function: ``out = apply(params, rng, *a, **k)``
    """

    init: Callable[..., hk.Params]
    apply: Callable[..., jnp.ndarray]
