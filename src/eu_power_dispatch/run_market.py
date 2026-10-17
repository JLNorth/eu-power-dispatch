"""CLI: solve the unit-commitment market model day by day and write modelled prices.

Each UTC day is solved independently with perfect foresight of residual demand
and fuel prices. Commitment status at the end of one day is carried into the
next so start-up costs are not double counted at midnight.

Outputs (``<outputs_dir>/<run_name>/``):

* ``modelled_prices.parquet`` - ``modelled_price``, ``da_price``, ``residual_demand_mw``
* ``dispatch.parquet``        - hourly MW per unit, storage and curtailment

Usage::

    python -m eu_power_dispatch.run_market --config config/base.yaml [--days 7]
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from eu_power_dispatch.config import Config, load_config
from eu_power_dispatch.data.prepare import load_prepared
from eu_power_dispatch.models.merit_order_mip import solve_market_day
from eu_power_dispatch.solver import make_solver

log = logging.getLogger(__name__)


def iter_days(df: pd.DataFrame, max_days: int | None = None):
    """Yield ``(day, frame)`` for each UTC calendar day in ``df``."""
    for i, (day, frame) in enumerate(df.groupby(df.index.floor("D"))):
        if max_days is not None and i >= max_days:
            break
        yield day, frame


def run_market(cfg: Config, data: pd.DataFrame, max_days: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Solve every day in ``data``; return ``(prices, dispatch)`` frames."""
    solver = make_solver(cfg.solver)
    fleet = cfg.fleet
    initial_on: dict[str, bool] | None = None
    price_frames, dispatch_frames = [], []
    t0 = time.perf_counter()

    for day, frame in iter_days(data, max_days):
        fuel_prices = {"gas": frame["gas_price"], **fleet.static_fuel_prices}
        dispatch, prices = solve_market_day(
            frame["residual_demand_mw"],
            fleet.units,
            fuel_prices,
            frame["carbon_price"],
            storage=fleet.storage,
            initial_on=initial_on,
            curtailment_cost=fleet.curtailment_cost,
            solver=solver,
        )
        commitment = dispatch.attrs.pop("commitment")
        initial_on = {u: bool(v) for u, v in commitment.iloc[-1].items()}
        price_frames.append(
            pd.DataFrame(
                {
                    "modelled_price": prices,
                    "da_price": frame["da_price"],
                    "residual_demand_mw": frame["residual_demand_mw"],
                }
            )
        )
        dispatch_frames.append(dispatch)
        log.debug("Solved %s: mean price %.2f", day.date(), prices.mean())

    if not price_frames:
        raise ValueError("No data to solve: check the configured period and prepared data")
    log.info("Solved %d days in %.1fs", len(price_frames), time.perf_counter() - t0)
    return pd.concat(price_frames), pd.concat(dispatch_frames)


def write_market_outputs(prices: pd.DataFrame, dispatch: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(out_dir / "modelled_prices.parquet")
    dispatch.to_parquet(out_dir / "dispatch.parquet")
    log.info("Wrote market outputs to %s", out_dir)


def summarise(prices: pd.DataFrame, dispatch: pd.DataFrame, currency: str) -> str:
    gen_cols = [c for c in dispatch.columns if not c.startswith(("storage_", "curtailment", "residual"))]
    energy_twh = dispatch[gen_cols].sum() / 1e6
    lines = [
        f"Hours solved:          {len(prices)}",
        f"Mean modelled price:   {prices['modelled_price'].mean():.2f} {currency}/MWh",
        f"Mean actual price:     {prices['da_price'].mean():.2f} {currency}/MWh",
        "Energy by unit (TWh):",
        *[f"  {name:<18}{val:8.2f}" for name, val in energy_twh.items()],
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Solve the market model day by day.")
    parser.add_argument("--config", default=None, help="Path to base.yaml")
    parser.add_argument("--days", type=int, default=None, help="Only solve the first N days")
    parser.add_argument("--run-name", default="base", help="Output sub-directory name")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("pyomo").setLevel(logging.WARNING)

    cfg = load_config(args.config) if args.config else load_config()
    prices, dispatch = run_market(cfg, load_prepared(cfg), max_days=args.days)
    write_market_outputs(prices, dispatch, cfg.paths.outputs_dir / args.run_name)
    print(summarise(prices, dispatch, cfg.currency))


if __name__ == "__main__":
    main()
