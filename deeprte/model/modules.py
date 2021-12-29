from typing import NamedTuple, Optional

import haiku as hk
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from deeprte.model.mapping import vmap
from deeprte.model.networks import MLP


class F(NamedTuple):
    """Graph of a function (x, y=f(x)) as a namedtuple."""

    x: jnp.float32 | jnp.ndarray = None
    y: jnp.ndarray = 0


class GreenFunctionNet(hk.Module):
    """Green's function of solution operator."""

    def __init__(
        self, config: ConfigDict, name: Optional[str] = "green_function"
    ):
        super().__init__(name=name)

        self.config = config

    def __call__(
        self, r: jnp.ndarray, r_prime: jnp.ndarray, coefficient_fn: F
    ) -> jnp.ndarray:
        """Compute Green's function with coefficient net as inputs.

        Args:
            r: Position and velocity variables, both are dimension d.
                Shape (2d,)
            r_prime: Dual position and velocity variables. Shape (2d,)
            coefficient_fn: Coefficient function at as a NamedTuple (x, y)
                where x are the positions and y are corresponding values.
                Shapes are (num_positions, d) for x and
                (num_positions, num_coefficients) for y.
        Returns:
            Green's function at r, r_prime.
        """

        c = self.config

        # Get nn output of coefficient net.
        coefficients = CoefficientNet(c.coefficient_net)(r[:2], coefficient_fn)

        # Green's function inputs.
        inputs = jnp.concatenate([r, r_prime, coefficients])

        # MLP
        outputs = MLP(c.green_function_mlp.widths, name="green_function_mlp")(
            inputs
        )
        # Wrap with exponential function to keep it non-negative.
        outputs = jnp.exp(outputs)

        return outputs


class CoefficientNet(hk.Module):
    """Coefficient functions as inputs of Green's function."""

    def __init__(
        self, config: ConfigDict, name: Optional[str] = "coefficient_net"
    ):
        super().__init__(name=name)

        self.config = config

    def __call__(self, r: jnp.ndarray, coefficient_fn: F) -> jnp.ndarray:
        """Compute coefficients of the equation as the inputs of Green's function.

        Args:
            r: Spatial position with dimention d.shape is (d,).
            coefficient_fn: Coefficient function at as a NamedTuple (x, y)
                where x are the positions and y are corresponding values.
                Shapes are (num_positions, d) for x and
                (num_positions, num_coefficients) for y.
        Returns:
            Coefficient information at spatial position r.
            Shape (num_coefficients,) or ().
        """

        c = self.config

        attn_net = MLP(
            c.attention_net.widths, use_bias=False, name="attention_net"
        )

        def attn_fn(q, k):
            # [2 * d]
            qk = jnp.concatenate([q, k])
            # scalar
            attn_qk_scalar = attn_net(qk)
            return attn_qk_scalar

        attn_logits = vmap(attn_fn, argnums={1}, use_hk=True, out_axes=-1)(
            r, coefficient_fn.x
        )
        attn = jax.nn.softmax(attn_logits)
        attn_outputs = jnp.matmul(attn, coefficient_fn.y)

        # Point-wise MLP
        outputs = MLP(c.pointwise_mlp.widths, name="pointwise_mlp")(
            attn_outputs
        )

        return outputs
