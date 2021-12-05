import abc
from typing import Callable, Mapping, Optional, Tuple

import jax.numpy as jnp
from deeprte.typing import Data, Logs


class BaseModel(object, metaclass=abc.ABCMeta):
    """Container class for the model that describes the equations, etc.
    At least the `self.loss` method should be implemented.
    """

    def __init__(self, name: Optional[str]) -> None:
        self._loss_fn = None
        self._regs = None

    def compile(
        self,
        loss_fn: Callable[..., jnp.float32],
        regularizers: Mapping[str, jnp.float32],
    ) -> None:
        if not self._loss_fn:
            self._loss_fn = loss_fn
            self._regs = regularizers

    @abc.abstractmethod
    def loss(
        self, fun, batch: Data, rng: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.float32, Logs]:
        pass

    def residual(self, fun) -> jnp.ndarray:
        pass

    def boundary(self, fun) -> jnp.ndarray:
        pass

    def initial(self, fun) -> jnp.ndarray:
        pass
