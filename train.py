import functools

from absl import app, flags

from deeprte import solver

if __name__ == "__main__":
    flags.mark_flag_as_required("config")
    app.run(functools.partial(solver.main, solver.Solver))
