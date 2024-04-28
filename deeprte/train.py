# Copyright 2024 The Flax Authors.
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

"""Machine Translation example.

This script trains a Transformer on a WMT dataset.
"""

import collections
import functools
import os
from typing import Optional

import bleu
import decode
import haiku as hk
import input_pipeline
import jax
import jax.numpy as jnp
import ml_collections
import models
import numpy as np
import optax
import orbax
import tensorflow as tf
from absl import logging
from clu import metric_writers, periodic_actions
from flax import jax_utils
from flax import linen as nn
from flax.training import checkpoints, common_utils, train_state
from flax.training import dynamic_scale as dynamic_scale_lib

from deeprte.model.modules import DeepRTE

BASE_BATCH_SIZE = 8


class TrainState(train_state.TrainState):
    dynamic_scale: dynamic_scale_lib.DynamicScale


def _get_batch_scaled_lr(total_batch_size, lr, scale_by_batch=True):
    # This is the linear scaling rule in Section 5.1 of
    # https://arxiv.org/pdf/1706.02677.pdf.

    if scale_by_batch:
        lr = (lr * total_batch_size) / BASE_BATCH_SIZE

    return lr


def create_learning_rate_schedule(
    total_batch_size, steps_per_epoch, total_steps, opt_config
):
    """Build the learning rate schedule function."""
    base_lr = _get_batch_scaled_lr(
        total_batch_size,
        opt_config.base_lr,
        opt_config.scale_by_batch,
    )

    schedule_type = opt_config.schedule_type
    if schedule_type == "steps":
        boundaries = opt_config.decay_kwargs.decay_boundaries
        boundaries.sort()

        decay_rate = opt_config.decay_kwargs.decay_rate
        boundaries_and_scales = {
            int(boundary * total_steps): decay_rate for boundary in boundaries
        }
        schedule_fn = optax.piecewise_constant_schedule(
            init_value=base_lr, boundaries_and_scales=boundaries_and_scales
        )
    elif schedule_type == "exponential":
        transition_steps = opt_config.decay_kwargs.transition_steps
        decay_rate = opt_config.decay_kwargs.decay_rate
        schedule_fn = optax.exponential_decay(
            init_value=base_lr,
            transition_steps=transition_steps,
            decay_rate=decay_rate,
        )
    elif schedule_type == "cosine":
        warmup_steps = opt_config.decay_kwargs.warmup_epochs * steps_per_epoch
        # Batch scale the other lr values as well:
        init_value = _get_batch_scaled_lr(
            total_batch_size,
            opt_config.decay_kwargs.init_value,
            opt_config.scale_by_batch,
        )
        end_value = _get_batch_scaled_lr(
            total_batch_size,
            opt_config.decay_kwargs.end_value,
            opt_config.scale_by_batch,
        )

        schedule_fn = optax.warmup_cosine_decay_schedule(
            init_value=init_value,
            peak_value=base_lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=end_value,
        )
    elif schedule_type == "constant_cosine":
        # Convert end_value to alpha, used by cosine_decay_schedule.
        alpha = opt_config.decay_kwargs.end_value / base_lr

        # Number of steps spent in constant phase.
        constant_steps = int(opt_config.decay_kwargs.constant_fraction * total_steps)
        decay_steps = total_steps - constant_steps

        constant_phase = optax.constant_schedule(value=base_lr)
        decay_phase = optax.cosine_decay_schedule(
            init_value=base_lr, decay_steps=decay_steps, alpha=alpha
        )
        schedule_fn = optax.join_schedules(
            schedules=[constant_phase, decay_phase],
            boundaries=[constant_steps],
        )
    elif schedule_type == "constant":
        schedule_fn = optax.constant_schedule(value=base_lr)
    else:
        raise ValueError(f"Unknown learning rate schedule: {schedule_type}")

    return schedule_fn


def create_optimizer(lr_schedule, opt_config):
    """Construct the optax optimizer with given LR schedule."""
    optax_chain = []

    if opt_config.optimizer == "adam":
        # See: https://arxiv.org/abs/1412.6980
        optax_chain.extend([optax.scale_by_adam(**opt_config.adam_kwargs)])
    elif opt_config.optimizer == "lamb":
        # See: https://arxiv.org/abs/1904.00962
        optax_chain.extend(
            [
                optax.scale_by_adam(**opt_config.lamb_kwargs),
                optax.scale_by_trust_ratio(),
            ]
        )
    else:
        raise ValueError(f"Undefined optimizer {opt_config.optimizer}")

    # Scale by the (negative) learning rate.
    optax_chain.extend(
        [
            optax.scale_by_schedule(lr_schedule),
            optax.scale(-1),
        ]
    )

    return optax.chain(*optax_chain)


def compute_mean_squared_error(predictions, targets, axis=None):
    return jnp.mean(jnp.square(predictions - targets), axis=axis)


def compute_metrics(predictions, labels):
    mse = compute_mean_squared_error(predictions, labels, axis=-1)
    relative_mse = mse / jnp.mean(labels**2)
    metrics = {
        "mse": mse,
        "rmse": relative_mse,
    }
    return metrics


def dynamic_slice_feat(feat_dict, i, step_size):
    def slice_fn(x):
        return jax.lax.dynamic_slice(
            x, (i,) + (0,) * (x.ndim - 1), (step_size,) + x.shape[1:]
        )

    return jax.tree_map(slice_fn, feat_dict)


def accumulate_gradients_and_metrics(grad_and_metrics_fn, microsteps: int):
    """Wraps a function that computes gradients and metrics to use microsteps.

    The new function will split the input `batch' into `microsteps`
    chunks, and will compute the gradients and metrics on each chunk, and average
    all gradients and metrics. A new PRNGKey is passed to each microstep.

    Args:
      grad_and_metrics_fn: A function that takes (params, images, labels, rngs)
        and returns gradients (same shape as params), and metrics (a dictionary
        with array leaves).
      microsteps: Positive integer, controlling the number of microsteps.

    Returns:
      A function with the same signature as grad_and_metric_fn.
    """
    if not microsteps or microsteps < 0:
        return grad_and_metrics_fn

    def new_grad_and_metrics_fn(params, batch):
        batch_size = batch["psi_label"].shape[0]
        assert (
            batch_size % microsteps == 0
        ), f"Bad accum_steps {microsteps} for batch size {batch_size}"
        step_size = batch_size // microsteps

        def accum_fn(i, state):
            grad, metrics = state
            sliced_batch = dynamic_slice_feat(batch, i * step_size, step_size)
            grad_i, metrics_i = grad_and_metrics_fn(params, sliced_batch)

            new_grad, new_metrics = jax.tree_map(
                lambda x, y: x + y, (grad, metrics), (grad_i, metrics_i)
            )
            return new_grad, new_metrics

        state_shape_dtype = jax.eval_shape(
            grad_and_metrics_fn, params, dynamic_slice_feat(batch, 0, step_size)
        )
        state_0 = jax.tree_map(
            lambda sd: jnp.zeros(sd.shape, sd.dtype), state_shape_dtype
        )
        grads, metrics = jax.lax.fori_loop(0, microsteps, accum_fn, state_0)

        grads, metrics = jax.tree_map(lambda x: x / microsteps, (grads, metrics))

        return grads, metrics

    return new_grad_and_metrics_fn


def create_checkpoint_manager(
    *,
    workdir: str,
    every_steps: int,
    keep_last: Optional[int] = None,
    keep_steps_multiple_of: Optional[int] = None,
    wait_seconds: int = 300,
) -> orbax.checkpoint.CheckpointManager:
    """Creates an Orbax checkpoint manager."""
    directory = os.path.join(workdir, "ckpt")
    if jax.process_index() == 0 and not tf.io.gfile.exists(directory):
        tf.io.gfile.makedirs(directory)
    multihost_utils.sync_devices("create-ckpt-dir")
    ckpt_options = orbax.checkpoint.CheckpointManagerOptions(
        save_interval_steps=every_steps,
        max_to_keep=keep_last,
        keep_period=keep_steps_multiple_of,
    )
    ckpt_manager = orbax.checkpoint.CheckpointManager(
        directory,
        {
            "state": orbax.checkpoint.AsyncCheckpointer(
                orbax.checkpoint.PyTreeCheckpointHandler(
                    write_tree_metadata=True,
                ),
                timeout_secs=wait_seconds,
            ),
            "dataset_iterator": orbax.checkpoint.Checkpointer(
                orbax.checkpoint.JsonCheckpointHandler()
            ),
        },
        options=ckpt_options,
    )
    return ckpt_manager


# Primary training / eval / inference step functions.
# -----------------------------------------------------------------------------


# Temporary using old Haiku model transformation API.
def create_model(config):
    def _forward_fn(*args, **kwargs):
        model = DeepRTE(config.model)
        return model(*args, **kwargs)

    return hk.transform(_forward_fn)


def train_step(
    state, batch, learning_rate_fn, rng=None, microsteps: Optional[int] = None
):
    """Perform a single training step."""

    rng = jax.random.fold_in(rng, state.step)

    step = state.step

    def loss_fn(params, batch):
        predictions = state.apply_fn(params, rng, batch, is_training=True)
        labels = batch["psi_label"]
        metrics = compute_metrics(predictions, labels)
        loss = metrics["mse"]
        return loss, metrics

    grad_and_metrics_fn = jax.grad(loss_fn, has_aux=True)
    compute_grads_and_metrics = accumulate_gradients_and_metrics(
        grad_and_metrics_fn, microsteps
    )
    grads, (next_rngs, metrics) = compute_grads_and_metrics(state.params, batch)
    grads = jax.lax.pmean(grads, axis_name="batch")
    new_state = state.apply_gradients(grads=grads)

    metrics = jax.lax.pmean(metrics, axis_name="batch")
    metrics["learning_rate"] = learning_rate_fn(step)

    return new_state, metrics


def eval_step(params, batch, config, label_smoothing=0.0):
    """Calculate evaluation metrics on a batch."""
    inputs, targets = batch["inputs"], batch["targets"]
    weights = jnp.where(targets > 0, 1.0, 0.0)
    logits = models.Transformer(config).apply({"params": params}, inputs, targets)

    return compute_metrics(logits, targets, weights, label_smoothing)


def initialize_cache(inputs, max_decode_len, config):
    """Initialize a cache for a given input shape and max decode length."""
    target_shape = (inputs.shape[0], max_decode_len) + inputs.shape[2:]
    initial_variables = models.Transformer(config).init(
        jax.random.key(0),
        jnp.ones(inputs.shape, config.dtype),
        jnp.ones(target_shape, config.dtype),
    )
    return initial_variables["cache"]


def predict_step(inputs, params, cache, eos_id, max_decode_len, config, beam_size=4):
    """Predict translation with fast decoding beam search on a batch."""
    # Prepare transformer fast-decoder call for beam search: for beam search, we
    # need to set up our decoder model to handle a batch size equal to
    # batch_size * beam_size, where each batch item"s data is expanded in-place
    # rather than tiled.
    # i.e. if we denote each batch element subtensor as el[n]:
    # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
    encoded_inputs = decode.flat_batch_beam_expand(
        models.Transformer(config).apply(
            {"params": params}, inputs, method=models.Transformer.encode
        ),
        beam_size,
    )
    raw_inputs = decode.flat_batch_beam_expand(inputs, beam_size)

    def tokens_ids_to_logits(flat_ids, flat_cache):
        """Token slice to logits from decoder model."""
        # --> [batch * beam, 1, vocab]
        flat_logits, new_vars = models.Transformer(config).apply(
            {"params": params, "cache": flat_cache},
            encoded_inputs,
            raw_inputs,  # only needed for input padding mask
            flat_ids,
            mutable=["cache"],
            method=models.Transformer.decode,
        )
        new_flat_cache = new_vars["cache"]
        # Remove singleton sequence-length dimension:
        # [batch * beam, 1, vocab] --> [batch * beam, vocab]
        flat_logits = flat_logits.squeeze(axis=1)
        return flat_logits, new_flat_cache

    # Using the above-defined single-step decoder function, run a
    # beam search over possible sequences given input encoding.
    beam_seqs, _ = decode.beam_search(
        inputs,
        cache,
        tokens_ids_to_logits,
        beam_size=beam_size,
        alpha=0.6,
        eos_id=eos_id,
        max_decode_len=max_decode_len,
    )

    # Beam search returns [n_batch, n_beam, n_length + 1] with beam dimension
    # sorted in increasing order of log-probability.
    # Return the highest scoring beam sequence, drop first dummy 0 token.
    return beam_seqs[:, -1, 1:]


# Utils for prediction and BLEU calculation
# -----------------------------------------------------------------------------


def pad_examples(x, desired_batch_size):
    """Expand batch to desired size by repeating last slice."""
    batch_pad = desired_batch_size - x.shape[0]
    return np.concatenate([x, np.tile(x[-1], (batch_pad, 1))], axis=0)


def per_host_sum_pmap(in_tree):
    """Execute psum on in_tree"s leaves over one device per host."""
    host2devices = collections.defaultdict(list)
    for d in jax.devices():
        host2devices[d.process_index].append(d)
    devices = [host2devices[k][0] for k in host2devices]
    host_psum = jax.pmap(lambda x: jax.lax.psum(x, "i"), "i", devices=devices)

    def pre_pmap(xs):
        return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (1,) + x.shape), xs)

    def post_pmap(xs):
        return jax.tree_util.tree_map(lambda x: x[0], xs)

    return post_pmap(host_psum(pre_pmap(in_tree)))


def tohost(x):
    """Collect batches from all devices to host and flatten batch dimensions."""
    n_device, n_batch, *remaining_dims = x.shape
    return np.array(x).reshape((n_device * n_batch,) + tuple(remaining_dims))


def evaluate(*, p_eval_step, params, eval_ds: tf.data.Dataset, num_eval_steps: int):
    """Evaluate the params an return a dictionary with the metrics."""
    logging.info("Gathering evaluation metrics.")
    eval_metrics = []
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
    for _, eval_batch in zip(range(num_eval_steps), eval_iter):
        eval_batch = jax.tree_util.tree_map(lambda x: x._numpy(), eval_batch)  # pylint: disable=protected-access
        eval_batch = common_utils.shard(eval_batch)
        metrics = p_eval_step(params, eval_batch)
        eval_metrics.append(metrics)
    eval_metrics = common_utils.get_metrics(eval_metrics)
    eval_metrics_sums = jax.tree_util.tree_map(jnp.sum, eval_metrics)
    eval_denominator = eval_metrics_sums.pop("denominator")
    eval_summary = jax.tree_util.tree_map(
        lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
        eval_metrics_sums,
    )
    return eval_summary


def translate_and_calculate_bleu(
    *,
    p_pred_step,
    p_init_cache,
    params,
    predict_ds: tf.data.Dataset,
    decode_tokens,
    max_predict_length: int,
):
    """Translates the `predict_ds` and calculates the BLEU score."""
    n_devices = jax.local_device_count()
    logging.info("Translating evaluation dataset.")
    sources, references, predictions = [], [], []
    for pred_batch in predict_ds:
        pred_batch = jax.tree_util.tree_map(lambda x: x._numpy(), pred_batch)  # pylint: disable=protected-access
        # Handle final odd-sized batch by padding instead of dropping it.
        cur_pred_batch_size = pred_batch["inputs"].shape[0]
        if cur_pred_batch_size % n_devices:
            padded_size = int(np.ceil(cur_pred_batch_size / n_devices) * n_devices)
            pred_batch = jax.tree_util.tree_map(
                lambda x: pad_examples(x, padded_size),  # pylint: disable=cell-var-from-loop
                pred_batch,
            )
        pred_batch = common_utils.shard(pred_batch)
        cache = p_init_cache(pred_batch["inputs"])
        predicted = p_pred_step(
            pred_batch["inputs"], params, cache, decode.EOS_ID, max_predict_length
        )
        predicted = tohost(predicted)
        inputs = tohost(pred_batch["inputs"])
        targets = tohost(pred_batch["targets"])
        # Iterate through non-padding examples of batch.
        for i, s in enumerate(predicted[:cur_pred_batch_size]):
            sources.append(decode_tokens(inputs[i]))
            references.append(decode_tokens(targets[i]))
            predictions.append(decode_tokens(s))
    logging.info(
        "Translation: %d predictions %d references %d sources.",
        len(predictions),
        len(references),
        len(sources),
    )

    # Calculate BLEU score for translated eval corpus against reference.
    bleu_matches = bleu.bleu_partial(references, predictions)
    all_bleu_matches = per_host_sum_pmap(bleu_matches)
    bleu_score = bleu.complete_bleu(*all_bleu_matches)
    # Save translation samples for tensorboard.
    exemplars = ""
    for n in np.random.choice(np.arange(len(predictions)), 8):
        exemplars += f"{sources[n]}\n\n{references[n]}\n\n{predictions[n]}\n\n"
    return exemplars, bleu_score


def preferred_dtype(config):
    platform = jax.local_devices()[0].platform
    if config.use_mixed_precision:
        if platform == "tpu":
            return jnp.bfloat16
        elif platform == "gpu":
            return jnp.float16
    return jnp.float32


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    """Runs a training and evaluation loop.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """
    tf.io.gfile.makedirs(workdir)

    vocab_path = config.vocab_path
    if vocab_path is None:
        vocab_path = os.path.join(workdir, "sentencepiece_model")
        config.vocab_path = vocab_path
    tf.io.gfile.makedirs(os.path.split(vocab_path)[0])

    # Load Dataset
    # ---------------------------------------------------------------------------
    logging.info("Initializing dataset.")
    train_ds, eval_ds, predict_ds, encoder = input_pipeline.get_wmt_datasets(
        n_devices=jax.local_device_count(),
        config=config,
        reverse_translation=config.reverse_translation,
        vocab_path=vocab_path,
    )

    train_iter = iter(train_ds)
    vocab_size = int(encoder.vocab_size())
    eos_id = decode.EOS_ID  # Default Sentencepiece EOS token.

    def decode_tokens(toks):
        valid_toks = toks[: np.argmax(toks == eos_id) + 1].astype(np.int32)
        return encoder.detokenize(valid_toks).numpy().decode("utf-8")

    if config.num_predict_steps > 0:
        predict_ds = predict_ds.take(config.num_predict_steps)

    logging.info("Initializing model, optimizer, and step functions.")

    dtype = preferred_dtype(config)

    # Build Model and Optimizer
    # ---------------------------------------------------------------------------
    train_config = models.TransformerConfig(
        vocab_size=vocab_size,
        output_vocab_size=vocab_size,
        share_embeddings=config.share_embeddings,
        logits_via_embedding=config.logits_via_embedding,
        dtype=dtype,
        emb_dim=config.emb_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        qkv_dim=config.qkv_dim,
        mlp_dim=config.mlp_dim,
        max_len=max(config.max_target_length, config.max_eval_target_length),
        dropout_rate=config.dropout_rate,
        attention_dropout_rate=config.attention_dropout_rate,
        deterministic=False,
        decode=False,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
    )
    eval_config = train_config.replace(deterministic=True)
    predict_config = train_config.replace(deterministic=True, decode=True)

    start_step = 0
    rng = jax.random.key(config.seed)
    rng, init_rng = jax.random.split(rng)
    input_shape = (config.per_device_batch_size, config.max_target_length)
    target_shape = (config.per_device_batch_size, config.max_target_length)

    m = models.Transformer(eval_config)
    initial_variables = jax.jit(m.init)(
        init_rng,
        jnp.ones(input_shape, jnp.float32),
        jnp.ones(target_shape, jnp.float32),
    )

    # Create train state with Adam optimizer and weight decay.
    learning_rate_fn = create_learning_rate_schedule(
        learning_rate=config.learning_rate, warmup_steps=config.warmup_steps
    )
    dynamic_scale = None
    if dtype == jnp.float16:
        dynamic_scale = dynamic_scale_lib.DynamicScale()
    state = TrainState.create(
        apply_fn=m.apply,
        params=initial_variables["params"],
        tx=optax.adamw(
            learning_rate=learning_rate_fn,
            b1=0.9,
            b2=0.98,
            eps=1e-9,
            weight_decay=config.weight_decay,
        ),
        dynamic_scale=dynamic_scale,
    )

    # We access model params only via state.params
    del initial_variables

    if config.restore_checkpoints:
        # Restore unreplicated optimizer + model state from last checkpoint.
        state = checkpoints.restore_checkpoint(workdir, state)
        # Grab last step.
        start_step = int(state.step)

    writer = metric_writers.create_default_writer(
        workdir, just_logging=jax.process_index() > 0
    )
    if start_step == 0:
        writer.write_hparams(dict(config))

    # Replicate state.
    state = jax_utils.replicate(state)

    # compile multidevice versions of train/eval/predict step and cache init fn.
    p_train_step = jax.pmap(
        functools.partial(
            train_step,
            config=train_config,
            learning_rate_fn=learning_rate_fn,
            label_smoothing=config.label_smoothing,
        ),
        axis_name="batch",
        donate_argnums=(0,),
    )  # pytype: disable=wrong-arg-types
    p_eval_step = jax.pmap(
        functools.partial(eval_step, config=eval_config), axis_name="batch"
    )
    p_init_cache = jax.pmap(
        functools.partial(
            initialize_cache,
            max_decode_len=config.max_predict_length,
            config=predict_config,
        ),
        axis_name="batch",
    )
    p_pred_step = jax.pmap(
        functools.partial(
            predict_step, config=predict_config, beam_size=config.beam_size
        ),
        axis_name="batch",
        static_broadcasted_argnums=(3, 4),
    )  # eos token, max_length are constant

    # Main Train Loop
    # ---------------------------------------------------------------------------

    # We init the first set of dropout PRNG keys, but update it afterwards inside
    # the main pmap"d training update for performance.
    dropout_rngs = jax.random.split(rng, jax.local_device_count())
    del rng

    logging.info("Starting training loop.")
    hooks = []
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=config.num_train_steps, writer=writer
    )
    if jax.process_index() == 0:
        hooks += [
            report_progress,
            periodic_actions.Profile(logdir=workdir, num_profile_steps=5),
        ]
    train_metrics = []
    with metric_writers.ensure_flushes(writer):
        for step in range(start_step, config.num_train_steps):
            is_last_step = step == config.num_train_steps - 1

            # Shard data to devices and do a training step.
            with jax.profiler.StepTraceAnnotation("train", step_num=step):
                batch = common_utils.shard(
                    jax.tree_util.tree_map(np.asarray, next(train_iter))
                )
                state, metrics = p_train_step(state, batch, dropout_rng=dropout_rngs)
                train_metrics.append(metrics)

            # Quick indication that training is happening.
            logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
            for h in hooks:
                h(step)

            # Periodic metric handling.
            if step % config.eval_every_steps == 0 or is_last_step:
                with report_progress.timed("training_metrics"):
                    logging.info("Gathering training metrics.")
                    train_metrics = common_utils.get_metrics(train_metrics)
                    lr = train_metrics.pop("learning_rate").mean()
                    metrics_sums = jax.tree_util.tree_map(jnp.sum, train_metrics)
                    denominator = metrics_sums.pop("denominator")
                    summary = jax.tree_util.tree_map(
                        lambda x: x / denominator, metrics_sums
                    )  # pylint: disable=cell-var-from-loop
                    summary["learning_rate"] = lr
                    summary = {"train_" + k: v for k, v in summary.items()}
                    writer.write_scalars(step, summary)
                    train_metrics = []

                with report_progress.timed("eval"):
                    eval_results = evaluate(
                        p_eval_step=p_eval_step,
                        params=state.params,
                        eval_ds=eval_ds,
                        num_eval_steps=config.num_eval_steps,
                    )
                    writer.write_scalars(
                        step, {"eval_" + k: v for k, v in eval_results.items()}
                    )

                with report_progress.timed("translate_and_bleu"):
                    exemplars, bleu_score = translate_and_calculate_bleu(
                        p_pred_step=p_pred_step,
                        p_init_cache=p_init_cache,
                        params=state.params,
                        predict_ds=predict_ds,
                        decode_tokens=decode_tokens,
                        max_predict_length=config.max_predict_length,
                    )
                    writer.write_scalars(step, {"bleu": bleu_score})
                    writer.write_texts(step, {"samples": exemplars})

            # Save a checkpoint on one host after every checkpoint_freq steps.
            save_checkpoint = step % config.checkpoint_every_steps == 0 or is_last_step
            if config.save_checkpoints and save_checkpoint:
                logging.info("Saving checkpoint step %d.", step)
                with report_progress.timed("checkpoint"):
                    checkpoints.save_checkpoint_multiprocess(
                        workdir, jax_utils.unreplicate(state), step
                    )
