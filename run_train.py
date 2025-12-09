import jax
from absl import app, flags, logging

from deeprte.configs import config
from deeprte.train_lib import train

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data.")
flags.DEFINE_string(
    "config", None, "File path to the training hyperparameter configuration."
)
flags.mark_flags_as_required(["config", "workdir"])


def main(argv) -> None:  # noqa: ANN001, D103
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")  # noqa: EM101, TRY003

    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    # Load the configuration.
    cfg = config.get_config(FLAGS.config)
    # Train and evaluate
    train.train_loop(cfg, FLAGS.workdir)


if __name__ == "__main__":
    jax.config.config_with_absl()
    app.run(main)
