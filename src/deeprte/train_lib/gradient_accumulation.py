"""Functions for gradient accumulation (GA)"""

import jax
import jax.numpy as jnp


def gradient_accumulation_loss_and_grad(
    loss_fn, gradient_accumulation_steps: int, params, batch
):
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def accumulate_gradient(acc_grad_and_loss, data):
        (loss, aux), cur_batch_gradient = grad_fn(params, data)
        acc_grad_and_loss["loss"] += loss
        acc_grad_and_loss["mean_squared_labels"] += aux["mean_squared_labels"]
        acc_grad_and_loss["grad"] = jax.tree.map(
            lambda x, y: x + y, cur_batch_gradient, acc_grad_and_loss["grad"]
        )
        return acc_grad_and_loss, aux

    def reshape_to_microbatch_accumulations(batch_arr):
        """Reshape global batch to microbatches, assuming batch axis is leading."""
        num_microbatches = gradient_accumulation_steps
        microbatch_shape = (
            batch_arr.shape[0] // num_microbatches,
            num_microbatches,
        ) + batch_arr.shape[1:]
        reshaped_batch_arr = jnp.reshape(batch_arr, microbatch_shape)
        return jnp.swapaxes(reshaped_batch_arr, 0, 1)

    batch = jax.tree.map(reshape_to_microbatch_accumulations, batch)
    init_grad = jax.tree.map(jnp.zeros_like, params)
    init_grad_and_loss = {"loss": 0.0, "mean_squared_labels": 0.0, "grad": init_grad}

    grad_and_loss, _ = jax.lax.scan(
        accumulate_gradient,
        init_grad_and_loss,
        batch,
        length=gradient_accumulation_steps,
    )
    grad_and_loss = jax.tree.map(
        lambda x: x / gradient_accumulation_steps, grad_and_loss
    )
    loss = grad_and_loss["loss"]
    aux = {"mean_squared_labels": grad_and_loss["mean_squared_labels"]}
    raw_grads = grad_and_loss["grad"]

    return (loss, aux), raw_grads
