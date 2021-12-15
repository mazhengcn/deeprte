import functools

from absl import app, flags

from deeprte.experiment import RTExperiment, main

if __name__ == "__main__":
    flags.mark_flag_as_required("config")
    app.run(functools.partial(main, RTExperiment))
