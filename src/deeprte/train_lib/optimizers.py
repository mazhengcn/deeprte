"""Optimizers."""

import optax


def create_learning_rate_schedule(config) -> optax.Schedule:
    """Creates a optax learning rate schedule."""

    lr = config.learning_rate

    match config.schedule:
        case "constant":
            return optax.schedules.constant_schedule(lr)
        case "exponential_decay":
            return optax.schedules.exponential_decay(
                init_value=lr,
                transition_steps=config.transition_steps,
                decay_rate=config.decay_rate,
            )
        case "warmup_exponential_decay":
            return optax.schedules.warmup_exponential_decay_schedule(
                init_value=0.0,
                peak_value=lr,
                warmup_steps=config.warmup_steps,
                transition_steps=config.transition_steps,
                decay_rate=config.decay_rate,
            )
        case "cosine_decay":
            return optax.schedules.cosine_decay_schedule(
                init_value=lr, decay_steps=config.num_train_steps
            )
        case "warmup_cosine_decay":
            return optax.schedules.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=lr,
                warmup_steps=config.warmup_steps,
                decay_steps=config.decay_steps,
            )
        case _:
            # Unknown learning rate schedule.
            raise ValueError(f"Unknown learning rate schedule: {config.schedule!r}")


def create_optimizer(
    config,
    learning_rate_schedule: optax.Schedule,
) -> optax.GradientTransformation:
    """Creates an optax optimizer."""
    match config.optimizer:
        case "adam":
            return optax.adam(learning_rate_schedule)
        case "adamw":
            return optax.adamw(learning_rate_schedule, weight_decay=config.weight_decay)
        case "sgd":
            return optax.sgd(learning_rate_schedule)
        case _:
            raise ValueError(f"{config.optimizer} is not a supported.")
