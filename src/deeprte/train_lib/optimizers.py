"""Optimizers."""

import optax


def create_learning_rate_schedule(config):
    """Creates a optax learning rate schedule."""

    schedule = config.schedule
    lr = config.learning_rate

    if schedule == "constant":
        return optax.schedules.constant_schedule(lr)
    elif schedule == "exponential_decay":
        return optax.schedules.exponential_decay(
            init_value=lr,
            transition_steps=config.transition_steps,
            decay_rate=config.decay_rate,
        )
    elif schedule == "warmup_exponential_decay":
        return optax.schedules.warmup_exponential_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=config.warmup_steps,
            transition_steps=config.transition_steps,
            decay_rate=config.decay_rate,
        )
    elif schedule == "cosine_decay":
        return optax.schedules.cosine_decay_schedule(
            init_value=lr, decay_steps=config.decay_steps
        )
    elif schedule == "warmup_cosine_decay":
        return optax.schedules.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=config.warmup_steps,
            decay_steps=config.decay_steps,
        )
    # Unknown learning rate schedule.
    raise ValueError(f"Unknown learning rate schedule: {schedule!r}")


def create_optimizer(
    config,
    learning_rate_schedule: optax.Schedule,
) -> optax.GradientTransformation:
    """Creates an optax optimizer."""
    if config.optimizer == "adam":
        return optax.adam(learning_rate_schedule)
    elif config.optimizer == "adamw":
        return optax.adamw(learning_rate_schedule, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        return optax.sgd(learning_rate_schedule)
    else:
        raise ValueError(f"{config.optimizer} is not a supported.")
