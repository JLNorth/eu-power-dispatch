"""Unit commitment and economic dispatch MIP with dual-based price formation.

For units :math:`u` and hours :math:`t`::

    min   sum_{u,t} MC_{u,t} p_{u,t} + SC_u y_{u,t}  + k_curt q_t  [+ storage throughput cost]
    s.t.  sum_u p_{u,t} + (d_t - c_t) - q_t = D_t                  (demand balance, dual = price)
          Pmin_u x_{u,t} <= p_{u,t} <= Pmax_u x_{u,t}              (committable units)
          0 <= p_{u,t} <= Pmax_u                                   (non-committable units)
          y_{u,t} >= x_{u,t} - x_{u,t-1}                           (start-up detection)
          x_{u,t}, y_{u,t} in {0, 1}
          battery constraints (see battery_lp.add_battery_block)

with marginal cost ``MC = fuel_price * heat_rate + carbon_price * emissions_factor + VOM``,
``D_t`` the residual demand and ``q_t >= 0`` curtailment of surplus must-run output.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from eu_power_dispatch.config import BatteryParams, SolverConfig, Unit
from eu_power_dispatch.models.battery_lp import add_battery_block
from eu_power_dispatch.solver import make_solver, solve

PriceLike = float | pd.Series


def _as_series(value: PriceLike, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        s = value.reindex(index)
        if s.isna().any():
            raise ValueError(f"Price series '{value.name}' does not cover the full horizon")
        return s.astype(float)
    return pd.Series(float(value), index=index)


def marginal_costs(
    units: Iterable[Unit], fuel_prices: Mapping[str, PriceLike], carbon_price: PriceLike, index: pd.Index
) -> pd.DataFrame:
    """Hourly short-run marginal cost (currency/MWh_e) for each unit.

    ``MC = fuel_price * heat_rate + carbon_price * emissions_factor + VOM``; units
    with ``fixed_marginal_cost`` use that value instead (imports, VoLL slack).
    """
    carbon = _as_series(carbon_price, index)
    cols = {}
    for u in units:
        if u.fixed_marginal_cost is not None:
            cols[u.name] = pd.Series(u.fixed_marginal_cost + u.vom, index=index)
            continue
        if u.fuel not in fuel_prices:
            raise KeyError(f"No price supplied for fuel '{u.fuel}' (unit '{u.name}')")
        fuel = _as_series(fuel_prices[u.fuel], index)
        cols[u.name] = fuel * u.heat_rate + carbon * u.emissions_factor + u.vom
    return pd.DataFrame(cols, index=index)


def build_market_model(
    residual_demand: pd.Series,
    units: Iterable[Unit],
    mc: pd.DataFrame,
    storage: BatteryParams | None = None,
    initial_on: Mapping[str, bool] | None = None,
    curtailment_cost: float = 0.0,
) -> pyo.ConcreteModel:
    units = list(units)
    unit_map = {u.name: u for u in units}
    committable = [u.name for u in units if u.committable]
    initial_on = initial_on or {}

    m = pyo.ConcreteModel(name="merit_order_uc")
    m.T = pyo.Set(initialize=range(len(residual_demand)), ordered=True)
    m.U = pyo.Set(initialize=[u.name for u in units], ordered=True)
    m.UC = pyo.Set(within=m.U, initialize=committable, ordered=True)

    demand = residual_demand.to_numpy(dtype=float)
    m.demand = pyo.Param(m.T, initialize=dict(enumerate(demand)))
    m.mc = pyo.Param(m.U, m.T, initialize={(u, t): float(mc[u].iat[t]) for u in m.U for t in m.T})

    m.p = pyo.Var(m.U, m.T, bounds=lambda _, u, t: (0, unit_map[u].capacity_mw))
    m.on = pyo.Var(m.UC, m.T, domain=pyo.Binary)
    m.start = pyo.Var(m.UC, m.T, domain=pyo.Binary)
    m.curtail = pyo.Var(m.T, bounds=(0, None))

    m.max_output = pyo.Constraint(m.UC, m.T, rule=lambda m, u, t: m.p[u, t] <= unit_map[u].capacity_mw * m.on[u, t])
    m.min_stable = pyo.Constraint(m.UC, m.T, rule=lambda m, u, t: m.p[u, t] >= unit_map[u].min_stable_mw * m.on[u, t])

    first = m.T.first()

    @m.Constraint(m.UC, m.T)
    def startup_def(m, u, t):
        prev = (1.0 if initial_on.get(u, True) else 0.0) if t == first else m.on[u, m.T.prev(t)]
        return m.start[u, t] >= m.on[u, t] - prev

    storage_cost = 0.0
    if storage is not None:
        bat = add_battery_block(m, m.T, storage, name="storage")
        storage_cost = bat.throughput_cost

    @m.Constraint(m.T)
    def balance(m, t):
        supply = sum(m.p[u, t] for u in m.U)
        if storage is not None:
            supply += m.storage.net_injection[t]
        return supply - m.curtail[t] == m.demand[t]

    m.cost = pyo.Objective(
        expr=sum(m.mc[u, t] * m.p[u, t] for u in m.U for t in m.T)
        + sum(unit_map[u].start_cost * m.start[u, t] for u in m.UC for t in m.T)
        + curtailment_cost * sum(m.curtail[t] for t in m.T)
        + storage_cost,
        sense=pyo.minimize,
    )
    return m


def clearing_prices_from_fixed_commitment(model: pyo.ConcreteModel, solver: Any) -> np.ndarray:
    """Recover hourly clearing prices from a solved unit-commitment MIP.

    A MIP has no meaningful dual values: integrality makes the feasible region
    non-convex, so there is no shadow price on the demand-balance constraint.
    The standard workaround (O'Neill et al., 2005) is to

    1. solve the MIP to find the optimal commitment (on/off and start-up decisions),
    2. fix every integer variable at its optimal value, relax its integrality,
    3. re-solve the remaining continuous problem as an LP.

    The LP has the same optimal dispatch, and the dual of each hour's balance
    constraint is the marginal cost of serving one more MWh given the committed
    fleet, i.e. the short-run marginal price. Start-up and no-load costs are
    sunk once commitment is fixed, so they do not appear in the price; in a real
    market they would be recovered through uplift or bid mark-ups.

    The model is modified in place (integer variables are fixed and relaxed),
    and the LP solution is loaded so dispatch values remain readable.
    """
    for var in model.component_data_objects(pyo.Var, descend_into=True):
        if var.is_integer():
            value = round(pyo.value(var))
            var.domain = pyo.Reals
            var.fix(value)

    if not hasattr(model, "dual"):
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    solve(model, solver)
    return np.array([model.dual[model.balance[t]] for t in model.T], dtype=float)


def solve_market_day(
    residual_demand: pd.Series,
    units: Iterable[Unit],
    fuel_prices: Mapping[str, PriceLike],
    carbon_price: PriceLike,
    storage: BatteryParams | None = None,
    initial_on: Mapping[str, bool] | None = None,
    curtailment_cost: float = 0.0,
    solver_cfg: SolverConfig | None = None,
    solver: Any = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Commit and dispatch the fleet against one day of residual demand.

    Parameters
    ----------
    residual_demand:
        Hourly load minus wind, solar and nuclear (MW). May be negative.
    units:
        Fleet definition (see ``config/fleet_gb.yaml``).
    fuel_prices:
        Fuel key -> price (currency/MWh_th), scalar or hourly series.
    carbon_price:
        Currency/tCO2, scalar or hourly series.
    storage:
        Optional aggregated battery participating in the market.
    initial_on:
        Commitment state of each unit in the hour before the horizon
        (defaults to online, i.e. no start-up cost in the first hour).
    curtailment_cost:
        Penalty on surplus curtailment; sets the price floor (= -cost).

    Returns
    -------
    dispatch:
        Hourly MW per unit plus ``curtailment_mw``, ``residual_demand_mw`` and,
        if storage is modelled, ``storage_charge_mw``, ``storage_discharge_mw``,
        ``storage_soc_mwh``. ``dispatch.attrs["commitment"]`` holds the 0/1
        on-status of committable units.
    prices:
        Hourly clearing price (currency/MWh), the demand-balance dual of the
        fixed-commitment LP.
    """
    units = list(units)
    index = residual_demand.index
    mc = marginal_costs(units, fuel_prices, carbon_price, index)
    model = build_market_model(residual_demand, units, mc, storage, initial_on, curtailment_cost)

    solver = solver or make_solver(solver_cfg)
    solve(model, solver)
    prices = clearing_prices_from_fixed_commitment(model, solver)

    T = list(model.T)
    dispatch = pd.DataFrame({u: [pyo.value(model.p[u, t]) for t in T] for u in model.U}, index=index)
    dispatch["curtailment_mw"] = [pyo.value(model.curtail[t]) for t in T]
    if storage is not None:
        dispatch["storage_charge_mw"] = [pyo.value(model.storage.charge[t]) for t in T]
        dispatch["storage_discharge_mw"] = [pyo.value(model.storage.discharge[t]) for t in T]
        dispatch["storage_soc_mwh"] = [pyo.value(model.storage.soc[t]) for t in T]
    dispatch["residual_demand_mw"] = residual_demand.to_numpy(dtype=float)
    dispatch.attrs["commitment"] = pd.DataFrame(
        {u: [round(pyo.value(model.on[u, t])) for t in T] for u in model.UC}, index=index
    )
    return dispatch, pd.Series(prices, index=index, name="modelled_price")
