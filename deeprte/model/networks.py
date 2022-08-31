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


import numbers
from collections.abc import Callable, Iterable, Sequence

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

# Constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
TRUNCATED_NORMAL_STDDEV_FACTOR = np.asarray(0.87962566103423978, dtype=np.float32)


# TODO(dropout): Add residual connection and dropout function for future use.
def apply_dropout(*, tensor, safe_key, rate, is_training, broadcast_dim=None):
    """Applies dropout to a tensor."""
    if is_training and rate != 0.0:
        shape = list(tensor.shape)
        if broadcast_dim is not None:
            shape[broadcast_dim] = 1
        keep_rate = 1.0 - rate
        keep = jax.random.bernoulli(safe_key.get(), keep_rate, shape=shape)
        return keep * tensor / keep_rate
    else:
        return tensor


def dropout_wrapper(
    module,
    input_act,
    mask,
    safe_key,
    global_config,
    output_act=None,
    is_training=True,
    **kwargs,
):
    """Applies module + dropout + residual update."""
    if output_act is None:
        output_act = input_act

    gc = global_config  # pylint: disable=invalid-name
    residual = module(input_act, mask, is_training=is_training, **kwargs)
    dropout_rate = 0.0 if gc.deterministic else module.config.dropout_rate

    if module.config.shared_dropout:
        if module.config.orientation == "per_row":
            broadcast_dim = 0
        else:
            broadcast_dim = 1
    else:
        broadcast_dim = None

    residual = apply_dropout(
        tensor=residual,
        safe_key=safe_key,
        rate=dropout_rate,
        is_training=is_training,
        broadcast_dim=broadcast_dim,
    )

    new_act = output_act + residual

    return new_act


def get_initializer_scale(initializer_name, input_shape):
    """Get Initializer for weights and scale to multiply activations by."""

    if initializer_name == "zeros":
        w_init = hk.initializers.Constant(0.0)
    elif initializer_name == "glorot_uniform":
        w_init = hk.initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform"
        )
    else:
        # fan-in scaling
        scale = 1.0
        for channel_dim in input_shape:
            scale /= channel_dim
        if initializer_name == "relu":
            scale *= 2

        noise_scale = scale

        stddev = np.sqrt(noise_scale)
        # Adjust stddev for truncation.
        stddev = stddev / TRUNCATED_NORMAL_STDDEV_FACTOR
        w_init = hk.initializers.TruncatedNormal(mean=0.0, stddev=stddev)

    return w_init


class Linear(hk.Module):
    """RTE specific Linear module.

    This differs from the standard Haiku Linear in a few ways:
        * It supports inputs and outputs of arbitrary rank
        * Initializers are specified by strings
    """

    def __init__(
        self,
        num_output: int | Sequence[int],
        initializer: str = "linear",
        num_input_dims: int = 1,
        use_bias: bool = True,
        bias_init: float = 0.0,
        precision=None,
        name: str = "linear",
    ):
        """Constructs Linear Module.

        Args:
          num_output: Number of output channels. Can be tuple when outputting
                multiple dimensions.
          initializer: What initializer to use, should be one of
                {'linear', 'relu','zeros'}
          num_input_dims: Number of dimensions from the end to project.
          use_bias: Whether to include trainable bias
          bias_init: Value used to initialize bias.
          precision: What precision to use for matrix multiplication, defaults
                to None.
          name: Name of module, used for name scopes.
        """
        super().__init__(name=name)
        if isinstance(num_output, numbers.Integral):
            self.output_shape = (num_output,)
        else:
            self.output_shape = tuple(num_output)
        self.initializer = initializer
        self.use_bias = use_bias
        self.bias_init = bias_init
        self.num_input_dims = num_input_dims
        self.num_output_dims = len(self.output_shape)
        self.precision = precision

    def __call__(self, inputs):
        """Connects Module.

        Args:
          inputs: Tensor with at least num_input_dims dimensions.

        Returns:
          output of shape [...] + num_output.
        """

        # num_input_dims = self.num_input_dims

        if self.num_input_dims > 0:
            in_shape = inputs.shape[-self.num_input_dims :]
        else:
            in_shape = ()

        weight_init = get_initializer_scale(self.initializer, in_shape)

        in_letters = "abcde"[: self.num_input_dims]
        out_letters = "hijkl"[: self.num_output_dims]

        weight_shape = in_shape + self.output_shape
        weights = hk.get_parameter("weights", weight_shape, inputs.dtype, weight_init)

        equation = f"...{in_letters}, {in_letters}{out_letters}->...{out_letters}"

        output = jnp.einsum(equation, inputs, weights, precision=self.precision)

        if self.use_bias:
            bias = hk.get_parameter(
                "bias",
                self.output_shape,
                inputs.dtype,
                hk.initializers.Constant(self.bias_init),
            )
            output += bias

        return output


class MLP(hk.Module):
    """RTE MLP using specific Linear module."""

    def __init__(
        self,
        output_sizes: Iterable[int] | Iterable[Sequence[int]],
        initializer: str = "linear",
        squeeze_output: bool = True,
        num_input_dims: int = 1,
        use_bias: bool = True,
        bias_init: float = 0.0,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.tanh,
        activate_final: bool = False,
        precision=None,
        name: str = "mlp",
    ):
        super().__init__(name=name)

        self.num_input_dims = num_input_dims
        self.initializer = initializer
        self.use_bias = use_bias
        self.bias_init = bias_init
        self.activation = activation
        self.activate_final = activate_final
        self.squeeze_output = squeeze_output
        self.precision = precision

        self.output_sizes = output_sizes
        self.num_layers = len(self.output_sizes)
        self.output_size = output_sizes[-1] if output_sizes else None

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        out = inputs
        for i, output_size in enumerate(self.output_sizes):
            out = Linear(
                output_size,
                self.initializer,
                self.num_input_dims,
                self.use_bias,
                self.bias_init,
                self.precision,
                name="linear",
            )(out)
            if i < (self.num_layers - 1) or self.activate_final:
                out = self.activation(out)

        if self.squeeze_output:
            out = jnp.squeeze(out)

        return out
