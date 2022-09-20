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
from collections.abc import Callable, Mapping
from typing import Any, Optional

import haiku as hk
import jax.numpy as jnp
from ml_collections import ConfigDict

from deeprte import dataset

TemplateFn = Callable[..., Any]


class Model(abc.ABC):
    """Container class for the model that describes the equations, etc.
    At least the `self.loss` method should be implemented.
    """

    def __init__(self, name: Optional[str] = "model"):
        self._name = name
        self._loss_fn = None
        self._regs = None

    @abc.abstractmethod
    def loss(
        self,
        func,
        batch: Mapping[str, jnp.ndarray],
        rng: Optional[jnp.ndarray] = None,
    ) -> tuple[jnp.float32, Mapping[str, jnp.ndarray]]:
        pass

    @abc.abstractmethod
    def metrics(self, func: Callable[..., Any], batch: dataset.Batch):
        pass


class Solution(abc.ABC):
    """Solution container used to create Haiku transformed pure function."""

    def __init__(self, config: ConfigDict, name: Optional[str] = "solution", **kwargs):
        self.name = name
        self.config = config

        self._init = hk.transform_with_state(self.forward).init
        self._apply = hk.transform_with_state(self.forward).apply

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> jnp.ndarray:
        """The forward pass of solution."""

    @property
    def init(self):
        return self._init

    @property
    def apply(self):
        return self._apply


class MultiSolutions(abc.ABC):
    """Multi-Solutions container used to create Haiku transformed
    pure functions.
    """

    def __init__(
        self,
        config: tuple[ConfigDict],
        name: Optional[str] = "multi_solutions",
        **kwargs
    ):
        self.config = config
        self.name = name

        self._init = hk.multi_transform(self.forward_fn).init
        self._apply = hk.multi_transform(self.forward_fn).apply

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> tuple[TemplateFn, Any]:
        """The forward pass of solution."""

    @abc.abstractmethod
    def apply(
        self, params: hk.Params, rng: jnp.ndarray, *, is_training: bool
    ) -> tuple[TemplateFn, Any]:
        """Apply function of the solution."""

    @property
    def init(self):
        return self._init
