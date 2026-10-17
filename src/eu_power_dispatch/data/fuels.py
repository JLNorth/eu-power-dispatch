"""Daily gas and carbon prices, forward-filled to hourly resolution.

Expected CSV (``paths.fuels_csv``)::

    date,gas_price,carbon_price
    2024-01-02,31.45,77.10
    2024-01-03,30.90,76.25
    ...

* ``date``          trading day (ISO date). Weekends/holidays may be omitted.
* ``gas_price``     front-month or day-ahead gas, currency/MWh_th (e.g. TTF in EUR/MWh).
* ``carbon_price``  allowance price, currency/tCO2 (e.g. EUA or UKA).

Prices are multiplied by ``fuels.gas_fx`` / ``fuels.carbon_fx`` to reach the
model currency, and ``fuels.carbon_price_adder`` (e.g. GB Carbon Price Support)
is added to carbon. Each day's price applies to all 24 UTC hours of that day;
non-trading days carry the last available price forward.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from eu_power_dispatch.config import FuelSettings

log = logging.getLogger(__name__)

FUEL_COLUMNS = ("gas_price", "carbon_price")


def read_fuel_csv(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Fuel price CSV not found at {path}. Provide columns date,{','.join(FUEL_COLUMNS)}."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    missing = [c for c in ("date", *FUEL_COLUMNS) if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df = df.dropna(subset=list(FUEL_COLUMNS), how="all")
    return df.set_index("date").sort_index()[list(FUEL_COLUMNS)].astype(float)


def daily_to_hourly(daily: pd.DataFrame, hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Map daily prices onto a UTC hourly index, forward-filling non-trading days."""
    if hourly_index.tz is None:
        raise ValueError("hourly_index must be tz-aware (UTC)")
    daily = daily[~daily.index.duplicated(keep="last")]
    daily.index = pd.DatetimeIndex(daily.index).tz_localize("UTC") if daily.index.tz is None else daily.index
    days = pd.date_range(hourly_index.min().floor("D"), hourly_index.max().floor("D"), freq="D")
    full = daily.reindex(daily.index.union(days)).sort_index().ffill()
    if full.loc[days].isna().any().any():
        first = daily.index.min()
        log.warning("Fuel prices start at %s, after the model start; back-filling the gap.", first.date())
        full = full.bfill()
    hourly = full.reindex(hourly_index.floor("D"))
    hourly.index = hourly_index
    return hourly


def load_fuel_prices(path: Path | str, hourly_index: pd.DatetimeIndex, settings: FuelSettings) -> pd.DataFrame:
    """Read the fuel CSV and return hourly ``gas_price`` and ``carbon_price`` in model currency."""
    hourly = daily_to_hourly(read_fuel_csv(path), hourly_index)
    hourly["gas_price"] *= settings.gas_fx
    hourly["carbon_price"] = hourly["carbon_price"] * settings.carbon_fx + settings.carbon_price_adder
    return hourly
