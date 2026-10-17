"""CLI: solve standalone battery arbitrage day by day for the configured period.

The battery is a price-taker with perfect foresight of each day's prices. Days
are solved independently with start and end state of charge fixed to the
same level, so annual revenue is the sum of daily optima.

Price source:

* ``actual``   - historic day-ahead prices from the prepared data (default)
* ``modelled`` - prices from a previous ``run_market`` run (``--run-name``)

Outputs (``<outputs_dir>/<run_name>/``): ``battery_schedule.parquet`` and
``battery_daily.parquet``.

Usage::

    python -m eu_power_dispatch.run_battery --prices actual
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from eu_power_dispatch.config import BatteryParams, Config, load_config
from eu_power_dispatch.data.prepare import load_prepared
from eu_power_dispatch.models.battery_lp import solve_battery_day
from eu_power_dispatch.run_market import iter_days
from eu_power_dispatch.solver import make_solver

log = logging.getLogger(__name__)


def load_prices(cfg: Config, source: str, run_name: str) -> pd.Series:
    if source == "actual":
        return load_prepared(cfg)["da_price"]
    if source == "modelled":
        path = cfg.paths.outputs_dir / run_name / "modelled_prices.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `epd-market --run-name {run_name}` first")
        return pd.read_parquet(path)["modelled_price"]
    raise ValueError(f"Unknown price source '{source}'")


def run_battery(cfg: Config, prices: pd.Series, params: BatteryParams | None = None, max_days: int | None = None) -> pd.DataFrame:
    """Solve each day of ``prices``; skips days with missing prices."""
    params = params or cfg.battery_asset
    solver = make_solver(cfg.solver)
    frames = []
    for day, frame in iter_days(prices.to_frame("price"), max_days):
        day_prices = frame["price"]
        if day_prices.isna().any():
            log.warning("Skipping %s: %d hours without a price", day.date(), int(day_prices.isna().sum()))
            continue
        frames.append(solve_battery_day(day_prices, params, solver=solver))
    if not frames:
        raise ValueError("No complete days of prices to optimise")
    return pd.concat(frames)


def daily_summary(schedule: pd.DataFrame, params: BatteryParams) -> pd.DataFrame:
    daily = schedule.groupby(schedule.index.floor("D")).agg(
        revenue=("revenue", "sum"),
        discharged_mwh=("discharge_mw", "sum"),
        charged_mwh=("charge_mw", "sum"),
        price_spread=("price", lambda p: p.max() - p.min()),
    )
    daily["cycles"] = daily["discharged_mwh"] / params.energy_mwh
    return daily


def write_battery_outputs(schedule: pd.DataFrame, daily: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    schedule.to_parquet(out_dir / "battery_schedule.parquet")
    daily.to_parquet(out_dir / "battery_daily.parquet")
    log.info("Wrote battery outputs to %s", out_dir)


def summarise(daily: pd.DataFrame, params: BatteryParams, currency: str) -> str:
    total = daily["revenue"].sum()
    n_days = len(daily)
    annualised = total * 365 / n_days if n_days else float("nan")
    return "\n".join(
        [
            f"Battery: {params.power_mw:g} MW / {params.energy_mwh:g} MWh, RTE {params.round_trip_efficiency:.0%}",
            f"Days solved:            {n_days}",
            f"Total revenue:          {total:,.0f} {currency}",
            f"Annualised revenue:     {annualised:,.0f} {currency}/yr "
            f"({annualised / params.power_mw / 1000:,.1f} k{currency}/MW/yr)",
            f"Mean cycles per day:    {daily['cycles'].mean():.2f}",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Solve battery arbitrage day by day.")
    parser.add_argument("--config", default=None, help="Path to base.yaml")
    parser.add_argument("--prices", choices=["actual", "modelled"], default="actual")
    parser.add_argument("--run-name", default="base", help="Output (and modelled price) sub-directory")
    parser.add_argument("--days", type=int, default=None, help="Only solve the first N days")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("pyomo").setLevel(logging.WARNING)

    cfg = load_config(args.config) if args.config else load_config()
    schedule = run_battery(cfg, load_prices(cfg, args.prices, args.run_name), max_days=args.days)
    daily = daily_summary(schedule, cfg.battery_asset)
    write_battery_outputs(schedule, daily, cfg.paths.outputs_dir / args.run_name / f"battery_{args.prices}")
    print(summarise(daily, cfg.battery_asset, cfg.currency))


if __name__ == "__main__":
    main()
