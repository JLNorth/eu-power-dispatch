"""Apply a scenario YAML on top of the base configuration and data, then re-run.

Supported scenario keys:

``profile_scaling``
    ``{column: {installed_mw, add_mw}}`` - scale a historic generation profile
    by ``(installed + add) / installed`` (constant capacity-factor shape), then
    recompute residual demand.
``storage_add``
    ``{power_mw, energy_mwh}`` - add to the base market storage (enabling it if absent).
``fleet_overrides``
    Arbitrary mapping deep-merged into the fleet file.
``base_overrides``
    Arbitrary mapping deep-merged into ``base.yaml`` (e.g. solver settings).

Usage::

    python -m eu_power_dispatch.scenarios config/scenarios/high_wind.yaml [--days 7]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from eu_power_dispatch.config import Config, build_config, deep_merge, load_config, read_yaml
from eu_power_dispatch.data.prepare import load_prepared
from eu_power_dispatch.run_market import run_market, write_market_outputs

log = logging.getLogger(__name__)

_PROFILE_COLUMNS = ("wind_mw", "solar_mw", "nuclear_mw", "load_mw")


def apply_scenario_config(cfg: Config, scenario: Mapping[str, Any]) -> Config:
    base = deep_merge(cfg.raw["base"], scenario.get("base_overrides") or {})
    fleet = deep_merge(cfg.raw["fleet"], scenario.get("fleet_overrides") or {})

    add = scenario.get("storage_add")
    if add:
        storage = dict(fleet.get("storage") or {})
        if not storage.get("enabled", False):
            storage.update(enabled=True, power_mw=0.0, energy_mwh=0.0)
        storage["power_mw"] = float(storage.get("power_mw", 0.0)) + float(add.get("power_mw", 0.0))
        storage["energy_mwh"] = float(storage.get("energy_mwh", 0.0)) + float(add.get("energy_mwh", 0.0))
        fleet["storage"] = storage

    return build_config(base, fleet, root=cfg.root)


def apply_scenario_data(data: pd.DataFrame, scenario: Mapping[str, Any]) -> pd.DataFrame:
    out = data.copy()
    for column, spec in (scenario.get("profile_scaling") or {}).items():
        if column not in _PROFILE_COLUMNS:
            raise ValueError(f"Cannot scale '{column}'; choose from {_PROFILE_COLUMNS}")
        installed = float(spec["installed_mw"])
        if installed <= 0:
            raise ValueError(f"{column}: installed_mw must be positive")
        factor = (installed + float(spec.get("add_mw", 0.0))) / installed
        out[column] = out[column] * factor
        log.info("Scaled %s by %.3f", column, factor)
    out["residual_demand_mw"] = out["load_mw"] - out["wind_mw"] - out["solar_mw"] - out["nuclear_mw"]
    return out


def compare_runs(base: pd.DataFrame, scen: pd.DataFrame, currency: str) -> pd.DataFrame:
    """Side-by-side headline statistics for base and scenario modelled prices."""

    def stats(p: pd.Series) -> dict[str, float]:
        return {
            f"mean ({currency}/MWh)": p.mean(),
            f"std ({currency}/MWh)": p.std(),
            "hours <= 0": float((p <= 0).sum()),
            f"p95 ({currency}/MWh)": p.quantile(0.95),
            f"mean daily spread ({currency}/MWh)": p.groupby(p.index.floor("D")).agg(lambda x: x.max() - x.min()).mean(),
        }

    idx = base.index.intersection(scen.index)
    table = pd.DataFrame(
        {"base": stats(base.loc[idx, "modelled_price"]), "scenario": stats(scen.loc[idx, "modelled_price"])}
    )
    table["delta"] = table["scenario"] - table["base"]
    return table


def run_scenario(cfg: Config, scenario_path: Path, max_days: int | None = None) -> pd.DataFrame:
    scenario = read_yaml(scenario_path)
    name = scenario.get("name", scenario_path.stem)
    scen_cfg = apply_scenario_config(cfg, scenario)
    data = apply_scenario_data(load_prepared(cfg), scenario)
    prices, dispatch = run_market(scen_cfg, data, max_days=max_days)
    write_market_outputs(prices, dispatch, cfg.paths.outputs_dir / "scenarios" / name)
    return prices


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the market model under a scenario.")
    parser.add_argument("scenario", type=Path, help="Path to a scenario YAML")
    parser.add_argument("--config", default=None, help="Path to base.yaml")
    parser.add_argument("--days", type=int, default=None, help="Only solve the first N days")
    parser.add_argument("--base-run", default="base", help="Base run to compare against")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("pyomo").setLevel(logging.WARNING)

    cfg = load_config(args.config) if args.config else load_config()
    prices = run_scenario(cfg, args.scenario, max_days=args.days)

    base_path = cfg.paths.outputs_dir / args.base_run / "modelled_prices.parquet"
    if base_path.exists():
        print(compare_runs(pd.read_parquet(base_path), prices, cfg.currency).round(2).to_string())
    else:
        log.info("No base run at %s; skipping comparison", base_path)
        print(prices["modelled_price"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
