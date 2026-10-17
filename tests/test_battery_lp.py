import numpy as np
import pandas as pd
import pytest

from eu_power_dispatch.config import BatteryParams
from eu_power_dispatch.models.battery_lp import solve_battery_day

TOL = 1e-6


def hourly(values) -> pd.Series:
    idx = pd.date_range("2024-03-01", periods=len(values), freq="h", tz="UTC")
    return pd.Series(np.asarray(values, dtype=float), index=idx)


@pytest.fixture
def params() -> BatteryParams:
    return BatteryParams(
        power_mw=10.0,
        energy_mwh=20.0,
        round_trip_efficiency=0.81,
        max_cycles_per_day=1.0,
        soc_min_frac=0.1,
        soc_max_frac=0.9,
        soc_initial_frac=0.5,
    )


@pytest.fixture
def volatile_prices() -> pd.Series:
    rng = np.random.default_rng(42)
    return hourly(60 + 40 * np.sin(np.arange(24) / 24 * 4 * np.pi) + rng.normal(0, 10, 24))


def test_soc_within_bounds(params, volatile_prices):
    out = solve_battery_day(volatile_prices, params)
    assert (out["soc_mwh"] >= params.soc_min_mwh - TOL).all()
    assert (out["soc_mwh"] <= params.soc_max_mwh + TOL).all()
    assert (out["charge_mw"] <= params.power_mw + TOL).all()
    assert (out["discharge_mw"] <= params.power_mw + TOL).all()


def test_no_simultaneous_charge_and_discharge(params, volatile_prices):
    out = solve_battery_day(volatile_prices, params)
    simultaneous = np.minimum(out["charge_mw"], out["discharge_mw"])
    assert simultaneous.max() < 1e-3


def test_positive_revenue_low_then_high(params):
    prices = hourly([20.0] * 12 + [100.0] * 12)
    out = solve_battery_day(prices, params)
    assert out["revenue"].sum() > 0
    assert out.loc[out.index[:12], "charge_mw"].sum() > 0
    assert out.loc[out.index[12:], "discharge_mw"].sum() > 0


def test_energy_balance(params, volatile_prices):
    out = solve_battery_day(volatile_prices, params)
    eta = params.one_way_efficiency
    soc_prev = np.r_[params.soc_initial_mwh, out["soc_mwh"].to_numpy()[:-1]]
    expected = soc_prev + eta * out["charge_mw"] - out["discharge_mw"] / eta
    np.testing.assert_allclose(out["soc_mwh"], expected, atol=1e-6)
    # Cyclic: ends where it started, so losses equal (1 - RTE) of energy charged.
    assert out["soc_mwh"].iloc[-1] == pytest.approx(params.soc_initial_mwh, abs=1e-6)
    assert out["discharge_mw"].sum() == pytest.approx(params.round_trip_efficiency * out["charge_mw"].sum(), rel=1e-6)


def test_cycle_limit(params):
    # Alternating prices would reward many cycles; the cap must bind.
    prices = hourly([10.0, 200.0] * 12)
    out = solve_battery_day(prices, params)
    assert out["discharge_mw"].sum() <= params.max_cycles_per_day * params.energy_mwh + TOL


def test_flat_prices_do_nothing(params):
    out = solve_battery_day(hourly([50.0] * 24), params)
    assert out["charge_mw"].sum() == pytest.approx(0.0, abs=1e-6)
    assert out["revenue"].sum() == pytest.approx(0.0, abs=1e-6)
