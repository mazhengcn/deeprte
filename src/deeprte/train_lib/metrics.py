import jax
import jax.numpy as jnp
from flax import nnx


class RelativeError(nnx.Metric):
    def __init__(self, argname_1="loss", argname_2="true_value"):
        self.argname_1, self.argname_2 = argname_1, argname_2
        self.error = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))
        self.true = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))

    def reset(self):
        self.error.value = jnp.array(0, dtype=jnp.float32)
        self.true.value = jnp.array(0, dtype=jnp.float32)

    def update(self, **kwargs) -> None:
        if self.argname_1 not in kwargs:
            raise TypeError(f"Expected keyword argument '{self.argname_1}'")
        if self.argname_2 not in kwargs:
            raise TypeError(f"Expected keyword argument '{self.argname_2}'")

        error: int | float | jax.Array = kwargs[self.argname_1]
        self.error.value += error if isinstance(error, (int, float)) else error.mean()

        true_value: int | float | jax.Array = kwargs[self.argname_2]
        self.true.value += (
            true_value if isinstance(true_value, (int, float)) else true_value.mean()
        )

    def compute(self) -> jax.Array:
        return jnp.sqrt(self.error.value / self.true.value)
