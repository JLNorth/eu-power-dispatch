import numpy as np
import pandas as pd
import pytest

from eu_power_dispatch.config import BatteryParams, Unit
from eu_power_dispatch.models.merit_order_mip import marginal_costs, solve_market_day

TOL = 1e-4


def hourly(values) -> pd.Series:
    idx = pd.date_range("2024-03-01", periods=len(values), freq="h", tz="UTC")
    return pd.Series(np.asarray(values, dtype=float), index=idx)


GAS = 30.0
CARBON = 50.0


@pytest.fixture
def toy_units() -> list[Unit]:
    # Marginal costs at gas 30, carbon 50:
    #   cheap = 30*1.8 + 50*0.35 + 2 = 73.5 ; mid = 30*2.0 + 50*0.40 + 3 = 83 ; peaker = 30*3 + 50*0.55 + 5 = 122.5
    return [
        Unit("peaker", capacity_mw=500, heat_rate=3.0, emissions_factor=0.55, vom=5.0, fuel="gas", committable=True),
        Unit("cheap", capacity_mw=1000, heat_rate=1.8, emissions_factor=0.35, vom=2.0, fuel="gas", committable=True),
        Unit("mid", capacity_mw=1000, heat_rate=2.0, emissions_factor=0.40, vom=3.0, fuel="gas", committable=True),
        Unit("voll", capacity_mw=10_000, fixed_marginal_cost=3000.0, committable=False),
    ]


def solve(demand, units, **kwargs):
    return solve_market_day(hourly(demand), units, {"gas": GAS}, CARBON, **kwargs)


def test_marginal_cost_formula(toy_units):
    mc = marginal_costs(toy_units, {"gas": GAS}, CARBON, hourly([0]).index)
    assert mc["cheap"].iat[0] == pytest.approx(73.5)
    assert mc["mid"].iat[0] == pytest.approx(83.0)
    assert mc["peaker"].iat[0] == pytest.approx(122.5)
    assert mc["voll"].iat[0] == pytest.approx(3000.0)


def test_demand_balance_every_hour(toy_units):
    demand = [400, 900, 1500, 2200, 2600, 1800, 700, -100]
    dispatch, _ = solve(demand, toy_units)
    supply = dispatch[[u.name for u in toy_units]].sum(axis=1) - dispatch["curtailment_mw"]
    np.testing.assert_allclose(supply, demand, atol=TOL)


def test_demand_balance_with_storage(toy_units):
    storage = BatteryParams(power_mw=300, energy_mwh=600, round_trip_efficiency=0.9)
    demand = [500, 500, 2300, 2300, 500, 500]
    dispatch, _ = solve(demand, toy_units, storage=storage)
    supply = (
        dispatch[[u.name for u in toy_units]].sum(axis=1)
        + dispatch["storage_discharge_mw"]
        - dispatch["storage_charge_mw"]
        - dispatch["curtailment_mw"]
    )
    np.testing.assert_allclose(supply, demand, atol=TOL)


def test_cheapest_units_dispatch_first(toy_units):
    dispatch, _ = solve([800, 1500, 2300], toy_units)
    # 800 MW: only the cheapest unit runs.
    assert dispatch["cheap"].iat[0] == pytest.approx(800, abs=TOL)
    assert dispatch[["mid", "peaker", "voll"]].iloc[0].sum() == pytest.approx(0, abs=TOL)
    # 1500 MW: cheap at full output, mid takes the rest.
    assert dispatch["cheap"].iat[1] == pytest.approx(1000, abs=TOL)
    assert dispatch["mid"].iat[1] == pytest.approx(500, abs=TOL)
    # 2300 MW: peaker only covers what cheap + mid cannot.
    assert dispatch["peaker"].iat[2] == pytest.approx(300, abs=TOL)
    assert dispatch["voll"].sum() == pytest.approx(0, abs=TOL)


def test_price_equals_marginal_unit_cost(toy_units):
    _, prices = solve([800, 1500, 2300], toy_units)
    np.testing.assert_allclose(prices.to_numpy(), [73.5, 83.0, 122.5], atol=TOL)


def test_scarcity_priced_at_voll(toy_units):
    dispatch, prices = solve([3000], toy_units)
    assert dispatch["voll"].iat[0] == pytest.approx(500, abs=TOL)
    assert prices.iat[0] == pytest.approx(3000.0, abs=TOL)


def test_min_stable_generation_respected():
    units = [
        Unit("baseload", capacity_mw=1000, min_stable_mw=600, heat_rate=1.8, fuel="gas", start_cost=10_000),
        Unit("flex", capacity_mw=1000, heat_rate=2.5, fuel="gas"),
    ]
    dispatch, _ = solve([300, 900, 300], units, initial_on={"baseload": False, "flex": False})
    on = dispatch["baseload"] > TOL
    assert (dispatch.loc[on, "baseload"] >= 600 - TOL).all()


def test_negative_residual_demand_curtails(toy_units):
    dispatch, prices = solve([-200], toy_units, curtailment_cost=5.0)
    assert dispatch["curtailment_mw"].iat[0] == pytest.approx(200, abs=TOL)
    assert prices.iat[0] == pytest.approx(-5.0, abs=TOL)
