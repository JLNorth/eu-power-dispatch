"""Solver factory so models stay independent of the chosen MILP/LP backend."""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo
from pyomo.opt import SolverResults, TerminationCondition

from eu_power_dispatch.config import SolverConfig

# Friendly config names -> Pyomo solver plugin names.
_SOLVER_ALIASES = {
    "highs": "appsi_highs",
    "gurobi": "gurobi_direct",
}

# Generic option names -> solver-specific option names.
_OPTION_NAMES = {
    "appsi_highs": {"mip_gap": "mip_rel_gap", "time_limit_s": "time_limit", "threads": "threads"},
    "gurobi_direct": {"mip_gap": "MIPGap", "time_limit_s": "TimeLimit", "threads": "Threads"},
    "gurobi": {"mip_gap": "MIPGap", "time_limit_s": "TimeLimit", "threads": "Threads"},
}


class SolveError(RuntimeError):
    """Raised when the solver does not return an optimal solution."""


def make_solver(cfg: SolverConfig | None = None) -> Any:
    """Return a configured Pyomo solver for ``cfg`` (defaults to HiGHS)."""
    cfg = cfg or SolverConfig()
    plugin = _SOLVER_ALIASES.get(cfg.name.lower(), cfg.name)
    solver = pyo.SolverFactory(plugin)
    if solver is None or not solver.available(exception_flag=False):
        raise SolveError(f"Solver '{cfg.name}' (Pyomo plugin '{plugin}') is not available")

    names = _OPTION_NAMES.get(plugin, {})
    for generic in ("mip_gap", "time_limit_s", "threads"):
        value = getattr(cfg, generic)
        if value is not None and generic in names:
            solver.options[names[generic]] = value
    for key, value in (cfg.options or {}).items():
        solver.options[key] = value
    return solver


def solve(model: pyo.ConcreteModel, solver: Any) -> SolverResults:
    """Solve ``model`` and raise :class:`SolveError` unless the result is optimal."""
    results = solver.solve(model, load_solutions=False)
    tc = results.solver.termination_condition
    if tc != TerminationCondition.optimal:
        raise SolveError(f"Solver terminated with condition '{tc}'")
    model.solutions.load_from(results)
    return results
