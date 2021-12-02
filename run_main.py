import functools

from absl import app, flags
from jaxline import platform

from deeprte import config
from deeprte.models import rte_op
from deeprte.solver import Solver

MODEL_CONFIG = config.CONFIG


sol = rte_op.RTEOperator(MODEL_CONFIG.rte_operator)
eqn = rte_op.RTEModel(name="rte")

RTESolver = Solver.from_solution_and_model(sol, eqn)


if __name__ == "__main__":
    flags.mark_flag_as_required("config")
    app.run(functools.partial(platform.main, RTESolver))
