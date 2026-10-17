"""Price-taking battery arbitrage as a linear programme.

For hours :math:`t = 1..T` with step :math:`\\Delta t = 1` h, prices :math:`\\pi_t`::

    max   sum_t pi_t (d_t - c_t) - k (c_t + d_t)
    s.t.  0 <= c_t, d_t <= P                                  (power limits)
          S_min <= s_t <= S_max                               (SoC bounds)
          s_t = s_{t-1} + eta c_t - d_t / eta                 (SoC dynamics)
          s_0 = s_T = S_init                                  (cyclic day)
          sum_t d_t <= N_cyc * E                              (cycle limit)

where ``eta = sqrt(round-trip efficiency)`` and ``k`` is an optional
throughput degradation cost. The constraint set is built by
:func:`add_battery_block` so the market model can reuse it unchanged.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyomo.environ as pyo

from eu_power_dispatch.config import BatteryParams, SolverConfig
from eu_power_dispatch.solver import make_solver, solve


def add_battery_block(model: pyo.ConcreteModel, time_set: pyo.Set, params: BatteryParams, name: str = "battery") -> pyo.Block:
    """Attach a battery block with variables ``charge``, ``discharge``, ``soc``.

    The block exposes ``net_injection[t]`` (discharge minus charge, MW) and
    ``throughput_cost`` for use in a parent model's balance and objective.
    ``time_set`` must be an ordered set of consecutive hourly periods.
    """
    blk = pyo.Block()
    model.add_component(name, blk)
    eta = params.one_way_efficiency

    blk.charge = pyo.Var(time_set, bounds=(0, params.power_mw))
    blk.discharge = pyo.Var(time_set, bounds=(0, params.power_mw))
    blk.soc = pyo.Var(time_set, bounds=(params.soc_min_mwh, params.soc_max_mwh))

    first, last = time_set.first(), time_set.last()

    @blk.Constraint(time_set)
    def soc_dynamics(b, t):
        prev = params.soc_initial_mwh if t == first else b.soc[time_set.prev(t)]
        return b.soc[t] == prev + eta * b.charge[t] - b.discharge[t] / eta

    blk.cyclic_soc = pyo.Constraint(expr=blk.soc[last] == params.soc_initial_mwh)
    blk.cycle_limit = pyo.Constraint(
        expr=sum(blk.discharge[t] for t in time_set) <= params.max_cycles_per_day * params.energy_mwh
    )

    blk.net_injection = pyo.Expression(time_set, rule=lambda b, t: b.discharge[t] - b.charge[t])
    blk.throughput_cost = pyo.Expression(
        expr=params.degradation_cost_per_mwh * sum(blk.charge[t] + blk.discharge[t] for t in time_set)
    )
    return blk


def build_battery_model(prices: pd.Series, params: BatteryParams) -> pyo.ConcreteModel:
    if prices.isna().any():
        raise ValueError("prices contain NaN; fill or drop them before optimising")
    m = pyo.ConcreteModel(name="battery_arbitrage")
    m.T = pyo.Set(initialize=range(len(prices)), ordered=True)
    m.price = pyo.Param(m.T, initialize=dict(enumerate(prices.to_numpy(dtype=float))))
    bat = add_battery_block(m, m.T, params)
    m.revenue = pyo.Objective(
        expr=sum(m.price[t] * bat.net_injection[t] for t in m.T) - bat.throughput_cost,
        sense=pyo.maximize,
    )
    return m


def solve_battery_day(
    prices: pd.Series, params: BatteryParams, solver_cfg: SolverConfig | None = None, solver: Any = None
) -> pd.DataFrame:
    """Optimise one day (or any horizon) of hourly battery arbitrage.

    Parameters
    ----------
    prices:
        Hourly prices (currency/MWh) indexed by timestamp.
    params:
        Battery technical parameters.
    solver_cfg, solver:
        Either a solver configuration or a pre-built Pyomo solver (reused across days).

    Returns
    -------
    DataFrame indexed like ``prices`` with columns ``price``, ``charge_mw``,
    ``discharge_mw``, ``net_mw``, ``soc_mwh`` (end of hour) and ``revenue``
    (currency, gross of degradation cost).
    """
    model = build_battery_model(prices, params)
    solve(model, solver or make_solver(solver_cfg))
    bat = model.battery
    out = pd.DataFrame(
        {
            "price": prices.to_numpy(dtype=float),
            "charge_mw": [pyo.value(bat.charge[t]) for t in model.T],
            "discharge_mw": [pyo.value(bat.discharge[t]) for t in model.T],
            "soc_mwh": [pyo.value(bat.soc[t]) for t in model.T],
        },
        index=prices.index,
    )
    out["net_mw"] = out["discharge_mw"] - out["charge_mw"]
    out["revenue"] = out["price"] * out["net_mw"]
    return out[["price", "charge_mw", "discharge_mw", "net_mw", "soc_mwh", "revenue"]]
