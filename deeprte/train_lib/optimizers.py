"""Optimizers."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

import jax.numpy as jnp
import optax

DType = type(jnp.float32)
PyTree = Any
MaskFn = Callable[[optax.Params], PyTree]
WeightDecay = float | Sequence[tuple[str, float]]


def warmup_polynomial_decay_schedule(
    peak_value: float,
    end_value: float,
    power: float,
    warmup_steps: int,
    decay_steps: int,
) -> optax.Schedule:
    """Linear warmup followed by polynomial decay."""
    return optax.schedules.join_schedules(
        [
            optax.schedules.linear_schedule(
                init_value=0.0, end_value=peak_value, transition_steps=warmup_steps
            ),
            optax.schedules.polynomial_schedule(
                init_value=peak_value,
                end_value=end_value,
                power=power,
                transition_steps=decay_steps - warmup_steps,
            ),
        ],
        [warmup_steps],
    )


def create_learning_rate_schedule(
    *, schedule: str, total_steps: int, **kwargs
) -> optax.Schedule:
    """Creates a optax learning rate schedule."""
    if schedule == "constant":
        return optax.schedules.constant_schedule(**kwargs)
    if schedule == "exponential_decay":
        return optax.schedules.exponential_decay(**kwargs)
    if schedule == "warmup_exponential_decay":
        return optax.schedules.warmup_exponential_decay_schedule(
            init_value=0.0, **kwargs
        )
    # The following schedules support decay_steps, set its default value.
    kwargs["decay_steps"] = kwargs.pop("decay_steps", total_steps)
    if schedule == "warmup_cosine_decay":
        return optax.schedules.warmup_cosine_decay_schedule(init_value=0.0, **kwargs)
    if schedule == "warmup_linear_decay":
        return warmup_polynomial_decay_schedule(power=1.0, **kwargs)
    if schedule == "warmup_polynomial_decay":
        return warmup_polynomial_decay_schedule(**kwargs)
    # Unknown learning rate schedule.
    raise ValueError(f"Unknown learning rate schedule: {schedule!r}")


def create_optimizer(
    *,
    name: str,
    total_steps: int,
    learning_rate: float | Mapping[str, Any],
    **optim_kwargs,
) -> optax.GradientTransformation:
    """Creates an optax optimizer."""
    ops = []
    # Optionally, apply a scale factor to some gradients.
    # WARNING: Use this with caution. Notice that this is NOT equivalent to having
    # a specific learning rate per parameter, since the scale that you use here
    # will affect the state of the optimizers like momentum.
    # ops.append(gradient_scaling(gradient_scale))
    # Optionally, add gradient clipping.
    # ops.append(gradient_clipping(**(gradient_clip or {})))
    # Optimizer-dependant scaling of gradients.
    # Note: we don't use optax aliases (e.g. optax.adam, optax.sgd, ...) because
    # we want to control explicitly how to add weight decay.
    if name == "adam":
        ops.append(optax.scale_by_adam(**optim_kwargs))
    elif name == "sgd":
        # Optionally, add momentum with SGD.
        ops.append(trace_momentum(**optim_kwargs))
    else:
        raise ValueError(f"Unknown optimizer: {name}")
    # # Optionally, add weight decay to the gradients.
    # ops.append(add_decayed_weights(weight_decay))
    # Scale gradients by learning rate.
    if isinstance(learning_rate, (float, int)):
        learning_rate = {"schedule": "constant", "value": learning_rate}
    lr_schedule = create_learning_rate_schedule(
        **learning_rate, total_steps=total_steps
    )

    # Wrap scale with inject_hyperparams to keep the last learning rate in the
    # optimizer state.
    @optax.inject_hyperparams
    def _scale_by_learning_rate(learning_rate):
        return optax.scale(-learning_rate)

    ops.append(_scale_by_learning_rate(lr_schedule))

    # Chain all operations on the gradients.
    return optax.chain(*ops), lr_schedule


def trace_momentum(momentum: Optional[float] = None, **kwargs):
    return optax.trace(decay=momentum, **kwargs) if momentum else optax.identity()
