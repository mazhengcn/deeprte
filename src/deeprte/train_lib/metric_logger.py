"""Logger that saves metrics to a local file, GCS and TensorBoard."""

import enum
import json
from collections import defaultdict

import jax
import numpy as np
from absl import logging

import deeprte.train_lib.utils as train_utils

EPS = 1e-8

# Mapping MaxText metrics to managed profiler metrics
_METRICS_TO_MANAGED = {
    "learning/current_learning_rate": "learning_rate",
    "learning/loss": "loss",
    "learning/grad_norm": "gradient_norm",
    "learning/total_weights": "total_weights",
    "perf/step_time_seconds": "step_time",
    "perf/per_device_tokens_per_sec": "throughput",
    "perf/per_device_tflops_per_sec": "tflops",
    # There are no mappings to the following metrics yet:
    # "latency", "mfu"
}


def _prepare_metrics_for_json(metrics, step, run_name):
    """Converts metric dictionary into json supported types (e.g. float)"""
    metrics_dict = {val: float(metrics["scalar"][val]) for val in metrics["scalar"]}
    metrics_dict["step"] = float(step)
    metrics_dict["run_name"] = run_name
    return metrics_dict


class MetadataKey(enum.Enum):
    PER_DEVICE_TFLOPS = "per_device_tflops"
    PER_DEVICE_TOKENS = "per_device_tokens"


class MetricLogger:
    """
    Logger for saving metrics to a local file and TensorBoard.
    """

    def __init__(self, config, learning_rate_schedule):
        self.writer = train_utils.initialize_summary_writer(
            config.tensorboard_dir, config.run_name
        )
        self.config = config
        self.metadata = {}
        self.learning_rate_schedule = learning_rate_schedule
        self.cumulative_eval_metrics = {"scalar": defaultdict(float)}
        self.buffered_train_metrics = None

    def reset_eval_metrics(self):
        """Resets the cumulative metrics dictionary for a new evaluation run."""
        self.cumulative_eval_metrics = {"scalar": defaultdict(float)}

    def write_metrics(self, metrics, step, is_training=True):
        """Entry point for all metrics writing in Train's Main."""
        if metrics:
            self.log_metrics(metrics, step, is_training)

            # if self.config.enable_tensorboard:
            #     self.write_metrics_to_tensorboard(metrics, step, is_training)

            # if self.config.metrics_file:
            #     self.write_metrics_locally(metrics, step)

    def log_metrics(self, metrics, step, is_training):
        """Logs metrics via max_logging."""
        if is_training:
            self._log_training_metrics(metrics, step)
        else:
            self._log_eval_metrics(metrics, step)

    def _log_training_metrics(self, metrics, step):
        """Handles training-specific metric logging."""
        # Skip logging if in profiler activation/deactivation steps
        # TODO(b/456828037): Switch to subprocess profiling to avoid timing artifacts at boundary steps.
        scalars = metrics["scalar"]
        loss = scalars["learning/loss"]
        # is_metric_hidden_step = (
        #     self.config.hide_profiler_step_metric
        #     and self._is_profiler_boundary_step(step)
        # )

        # Start building the log parts
        log_parts = []

        # if is_metric_hidden_step:
        #     log_parts.append(
        #         f"completed profiler activation/deactivation step: {step}",
        #     )
        # else:
        log_parts.extend(
            [
                f"completed step: {step}",
                f"seconds: {scalars['perf/step_time_seconds']:.3f}",
            ]
        )

        # Add performance metrics only if strictly NOT in rampup phase
        # TODO(b/452468482): Enable performance metric (TFLOPs, Tokens/s) tracking during batch size rampup.
        # if not is_metric_hidden_step:
        #     log_parts.extend(
        #         [
        #             f"tflop/s/device: {scalars['perf/per_device_tflops_per_sec']:.3f}",
        #             f"tokens/s/device: {scalars['perf/per_device_tokens_per_sec']:.3f}",
        #         ]
        #     )

        log_parts.extend(
            [
                f"relative_loss: {scalars['learning/relative_loss']:.3f}",
                f"loss: {loss}",
            ]
        )

        logging.info(", ".join(log_parts))

    def _log_eval_metrics(self, metrics, step):
        """Handles evaluation-specific metric logging."""
        scalars = metrics["scalar"]
        log_parts = [
            f"eval metrics after step: {step}",
            f"loss={scalars['eval/avg_loss']:.3f}",
            f"total_weights={scalars['eval/total_weights']}",
        ]

        if self.config.mtp_num_layers > 0:
            log_parts.extend(
                [
                    f"avg_mtp_loss={scalars['eval/avg_mtp_loss']:.3f}",
                    f"avg_mtp_acceptance_rate={scalars['eval/avg_mtp_acceptance_rate_percent']:.2f}%",
                ]
            )

        logging.info(", ".join(log_parts))

    def _is_profiler_boundary_step(self, step):
        """Determines if the current step is a profiler start/stop boundary that should be hidden."""
        if len(self.config.profiler) == 0:
            return False
        skip_steps = self.config.skip_first_n_steps_for_profiler
        profiler_steps = self.config.profiler_steps
        # Steps immediately before/at start, and at/immediately after end of profiling
        boundary_steps = {
            skip_steps,
            skip_steps + 1,
            skip_steps + profiler_steps,
            skip_steps + profiler_steps + 1,
        }
        return step in boundary_steps

    def write_metrics_locally(self, metrics, step):
        """Writes metrics locally for testing."""
        with open(self.config.metrics_file, "a", encoding="utf8") as local_metrics_file:
            if step == 0:
                local_metrics_file.truncate(0)

            metrics_dict = _prepare_metrics_for_json(
                metrics, step, self.config.run_name
            )
            local_metrics_file.write(str(json.dumps(metrics_dict)) + "\n")

    def write_metrics_to_tensorboard(self, metrics, step, is_training):
        """Writes metrics to TensorBoard."""
        if jax.process_index() == 0:
            for metric_name in metrics.get("scalar", []):
                self.writer.add_scalar(
                    metric_name, np.array(metrics["scalar"][metric_name]), step
                )
            for metric_name in metrics.get("scalars", []):
                self.writer.add_scalars(
                    metric_name, metrics["scalars"][metric_name], step
                )

        if is_training:
            full_log = step % self.config.log_period == 0

            if full_log and jax.process_index() == 0:
                logging.info(
                    f"To see full metrics 'tensorboard --logdir={self.config.tensorboard_dir}'"
                )
                self.writer.flush()

    def write_setup_info_to_tensorboard(self, params):
        """Writes setup information like train config params, num model params, and XLA flags to TensorBoard."""
        num_model_parameters = train_utils.calculate_num_params_from_pytree(params)
        # self.metadata[MetadataKey.PER_DEVICE_TFLOPS], _, _ = (
        #     train_utils.calculate_tflops_training_per_device(self.config)
        # )
        # self.metadata[MetadataKey.PER_DEVICE_TOKENS] = (
        #     train_utils.calculate_tokens_training_per_device(self.config)
        # )
        logging.info(f"number parameters: {num_model_parameters}")
        train_utils.add_text_to_summary_writer(
            "num_model_parameters", str(num_model_parameters), self.writer
        )
        train_utils.add_config_to_summary_writer(self.config, self.writer)

    def buffer_and_write_train_metrics(self, metrics, step, step_time_delta):
        """
        Buffers metrics for the current training step and simultaneously writes the training metrics
        for the previous step to GCS and/or TensorBoard. This buffering strategy allows for back-to-back
        execution of training steps, by overlapping data loading for step n with the execution of step n−1.
        This significantly boosts training efficiency.
        """
        if self.buffered_train_metrics is not None:
            (step_to_write, metrics_to_write) = self.buffered_train_metrics
            self.write_metrics(metrics_to_write, step_to_write)

        self.record_train_metrics(metrics, step, step_time_delta.total_seconds())
        self.buffered_train_metrics = (step, metrics)

    def record_train_metrics(self, metrics, step, step_time):
        """Records training metrics for the current step."""
        metrics["scalar"].update({"perf/step_time_seconds": step_time})
        metrics["scalar"].update(
            {"learning/current_learning_rate": self.learning_rate_schedule(step)}
        )
        # if step >= self.config.rampup_end_step:
        #     metrics["scalar"].update(
        #         {"perf/per_device_tflops": self.metadata[MetadataKey.PER_DEVICE_TFLOPS]}
        #     )
        #     metrics["scalar"].update(
        #         {
        #             "perf/per_device_tflops_per_sec": (
        #                 self.metadata[MetadataKey.PER_DEVICE_TFLOPS] / step_time
        #             )
        #         }
        #     )
        #     metrics["scalar"].update(
        #         {"perf/per_device_tokens": self.metadata[MetadataKey.PER_DEVICE_TOKENS]}
        #     )
        #     metrics["scalar"].update(
        #         {
        #             "perf/per_device_tokens_per_sec": (
        #                 self.metadata[MetadataKey.PER_DEVICE_TOKENS] / step_time
        #             )
        #         }
        #     )

    def record_eval_metrics(self, step, metrics=None, eval_step_count=None):
        """Records eval metrics and writes the metrics to GCS and/or to TensorBoard."""
        if metrics:
            self.cumulative_eval_metrics["scalar"]["eval/total_loss"] += float(
                metrics["scalar"].get("evaluation/total_loss", 0.0)
            )
            self.cumulative_eval_metrics["scalar"]["eval/total_weights"] += float(
                metrics["scalar"].get("evaluation/total_weights", 0.0)
            )
            self.cumulative_eval_metrics["scalar"]["eval/moe_lb_loss"] += float(
                metrics["scalar"].get("evaluation/moe_lb_loss", 0.0)
            )
            self.cumulative_eval_metrics["scalar"]["eval/mtp_loss"] += float(
                metrics["scalar"].get("evaluation/mtp_loss", 0.0)
            )
            self.cumulative_eval_metrics["scalar"][
                "eval/mtp_acceptance_rate_percent"
            ] += float(
                metrics["scalar"].get("evaluation/mtp_acceptance_rate_percent", 0.0)
            )
            if self.config.use_dpo:
                self.cumulative_eval_metrics["scalar"]["eval/dpo_reward_accuracy"] += (
                    float(metrics["scalar"].get("evaluation/dpo_reward_accuracy", 0.0))
                )

        if eval_step_count:
            eval_loss = self.cumulative_eval_metrics["scalar"]["eval/total_loss"] / (
                self.cumulative_eval_metrics["scalar"]["eval/total_weights"] + EPS
            )
            self.cumulative_eval_metrics["scalar"]["eval/avg_loss"] = eval_loss
            self.cumulative_eval_metrics["scalar"]["eval/avg_moe_lb_loss"] = (
                self.cumulative_eval_metrics["scalar"]["eval/moe_lb_loss"]
                / eval_step_count
            )
            self.cumulative_eval_metrics["scalar"]["eval/avg_mtp_loss"] = (
                self.cumulative_eval_metrics["scalar"]["eval/mtp_loss"]
                / eval_step_count
            )
            self.cumulative_eval_metrics["scalar"][
                "eval/avg_mtp_acceptance_rate_percent"
            ] = (
                self.cumulative_eval_metrics["scalar"][
                    "eval/mtp_acceptance_rate_percent"
                ]
                / eval_step_count
            )
            if self.config.use_dpo:
                self.cumulative_eval_metrics["scalar"]["eval/dpo_reward_accuracy"] = (
                    self.cumulative_eval_metrics["scalar"]["eval/dpo_reward_accuracy"]
                    / eval_step_count
                )

            self.write_metrics(self.cumulative_eval_metrics, step, is_training=False)

    def flush_metrics_and_cleanup(self):
        """
        This is a terminal operation that uploads any buffered metrics to GCS
        and/or TensorBoard before closing the writer objects. Once called, the
        logger instance should not be used to add or write more metrics as the
        underlying writer objects (e.g., TensorBoard SummaryWriter) will be closed.
        """
        if self.buffered_train_metrics is not None:
            (step_to_write, metrics_to_write) = self.buffered_train_metrics
            self.write_metrics(metrics_to_write, step_to_write)

        train_utils.close_summary_writer(self.writer)
