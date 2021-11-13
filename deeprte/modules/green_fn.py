from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
from deeprte.mapping import vmap
from deeprte.modules import nets
from deeprte.typing import GraphOfMapping
from ml_collections import ConfigDict


class GreenFunction(hk.Module):
    """Green's function of solution operator."""

    def __init__(
        self,
        config: ConfigDict,
        name: Optional[str] = "greens_func",
    ):
        super().__init__(name=name)
        self.config = config

    def __call__(
        self, x: jnp.ndarray, s: jnp.ndarray, coeffs: GraphOfMapping
    ) -> jnp.ndarray:
        """Compute Green's function using coefficient net as inputs.

        Args:
            x: [dim] primal variables.
            s: [dim] or [dim - 1] dual variables.
            coeffs: [num_coeffs_values, dim] and
                [num_coeffs_values, num_coeffs] tuple of
                coeffs_points and coeffs_values.

        Returns:
            Green's function.
        """
        y = coeffs.x
        # [num_coeffs]
        coeff = Coeffs(self.config.coeffs_net)(
            x[:2], GraphOfMapping(y, coeffs.fx)
        )
        # [xdim + xdim - 1 + num_coeffs]
        inputs = jnp.concatenate([x, s, coeff])
        # [num_sols]
        output = nets.MLP(self.config.green_net, name="green_net")(inputs)

        return jnp.exp(output)


class Coeffs(hk.Module):
    """Coefficient functions as inputs of Green's function."""

    def __init__(
        self,
        config: ConfigDict,
        name: Optional[str] = "coeff_func",
    ):
        super().__init__(name=name)
        self.config = config

    def __call__(self, x: jnp.ndarray, coeffs: GraphOfMapping) -> jnp.ndarray:
        """Compute coefficients of the equation as the inputs of Green's function
        using attention-like structure.

        Args:
            x: (N, d)
            coeffs: (M, d) and
                (M, sigma_y) tuple of
                coeffs_points and coeffs_values.

        Returns:
            neural network representations of coefficient functions.
        """

        attn_net = nets.MLP(
            self.config.weights, use_bias=False, name="attension_net"
        )

        def attn_fn(q, k):
            # [2 * d]
            qk = jnp.concatenate([q, k])
            # scalar
            attn_qk_scalar = attn_net(qk)
            return attn_qk_scalar

        attn_logits = vmap(attn_fn, argnums=(1,), use_hk=True, out_axes=-1)(
            x, coeffs.x
        )
        attn = jax.nn.softmax(attn_logits)
        output = jnp.matmul(attn, coeffs.fx)

        output = nets.MLP(self.config.coeffs, name="pointwise_mlp")(output)

        return output
