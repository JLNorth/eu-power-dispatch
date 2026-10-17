"""Load and validate YAML configuration into typed dataclasses.

The run configuration is split across two files:

* ``config/base.yaml``   - dates, zone, paths, solver and the standalone battery asset.
* ``config/fleet_gb.yaml`` - the aggregated dispatchable fleet and market storage.

Scenario files (``config/scenarios/*.yaml``) are applied on top by
:mod:`eu_power_dispatch.scenarios`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_CONFIG = REPO_ROOT / "config" / "base.yaml"


class ConfigError(ValueError):
    """Raised when a configuration file is missing keys or has invalid values."""


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing required key '{key}' in {context}")
    return mapping[key]


@dataclass(frozen=True)
class BatteryParams:
    """Technical parameters of a (possibly aggregated) battery."""

    power_mw: float
    energy_mwh: float
    round_trip_efficiency: float = 0.88
    max_cycles_per_day: float = 1.5
    soc_min_frac: float = 0.0
    soc_max_frac: float = 1.0
    soc_initial_frac: float = 0.5
    degradation_cost_per_mwh: float = 0.0

    def __post_init__(self) -> None:
        if self.power_mw <= 0 or self.energy_mwh <= 0:
            raise ConfigError("Battery power_mw and energy_mwh must be positive")
        if not 0 < self.round_trip_efficiency <= 1:
            raise ConfigError("round_trip_efficiency must be in (0, 1]")
        if self.max_cycles_per_day <= 0:
            raise ConfigError("max_cycles_per_day must be positive")
        if not 0 <= self.soc_min_frac <= self.soc_initial_frac <= self.soc_max_frac <= 1:
            raise ConfigError("Require 0 <= soc_min_frac <= soc_initial_frac <= soc_max_frac <= 1")
        if self.degradation_cost_per_mwh < 0:
            raise ConfigError("degradation_cost_per_mwh must be non-negative")

    @property
    def one_way_efficiency(self) -> float:
        """Charge and discharge efficiency, each the square root of round-trip."""
        return self.round_trip_efficiency**0.5

    @property
    def soc_min_mwh(self) -> float:
        return self.soc_min_frac * self.energy_mwh

    @property
    def soc_max_mwh(self) -> float:
        return self.soc_max_frac * self.energy_mwh

    @property
    def soc_initial_mwh(self) -> float:
        return self.soc_initial_frac * self.energy_mwh

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BatteryParams":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass(frozen=True)
class Unit:
    """An aggregated generating unit (or price-taking block) in the merit order."""

    name: str
    capacity_mw: float
    min_stable_mw: float = 0.0
    heat_rate: float = 0.0
    emissions_factor: float = 0.0
    start_cost: float = 0.0
    vom: float = 0.0
    fuel: str | None = None
    fixed_marginal_cost: float | None = None
    committable: bool = True

    def __post_init__(self) -> None:
        if self.capacity_mw <= 0:
            raise ConfigError(f"Unit '{self.name}': capacity_mw must be positive")
        if not 0 <= self.min_stable_mw <= self.capacity_mw:
            raise ConfigError(f"Unit '{self.name}': require 0 <= min_stable_mw <= capacity_mw")
        if self.fuel is None and self.fixed_marginal_cost is None:
            raise ConfigError(f"Unit '{self.name}': set either 'fuel' or 'fixed_marginal_cost'")
        if self.heat_rate < 0 or self.emissions_factor < 0 or self.start_cost < 0:
            raise ConfigError(f"Unit '{self.name}': heat_rate, emissions_factor, start_cost must be >= 0")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Unit":
        _require(d, "name", "unit definition")
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass(frozen=True)
class Fleet:
    units: tuple[Unit, ...]
    static_fuel_prices: Mapping[str, float] = field(default_factory=dict)
    curtailment_cost: float = 0.0
    storage: BatteryParams | None = None

    def __post_init__(self) -> None:
        names = [u.name for u in self.units]
        if len(names) != len(set(names)):
            raise ConfigError(f"Duplicate unit names in fleet: {names}")
        if not self.units:
            raise ConfigError("Fleet must contain at least one unit")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Fleet":
        units = tuple(Unit.from_dict(u) for u in _require(d, "units", "fleet file"))
        storage_cfg = d.get("storage") or {}
        storage = BatteryParams.from_dict(storage_cfg) if storage_cfg.get("enabled", False) else None
        return cls(
            units=units,
            static_fuel_prices=dict(d.get("fuel_prices_static") or {}),
            curtailment_cost=float(d.get("curtailment_cost", 0.0)),
            storage=storage,
        )


@dataclass(frozen=True)
class SolverConfig:
    name: str = "highs"
    mip_gap: float = 1e-3
    time_limit_s: float | None = 120
    threads: int | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SolverConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    processed_dir: Path
    fuels_csv: Path
    outputs_dir: Path
    fleet_file: Path

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], root: Path) -> "Paths":
        def resolve(key: str) -> Path:
            p = Path(_require(d, key, "paths"))
            return p if p.is_absolute() else root / p

        return cls(**{k: resolve(k) for k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class FuelSettings:
    gas_fx: float = 1.0
    carbon_fx: float = 1.0
    carbon_price_adder: float = 0.0

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FuelSettings":
        known = {k: float(v) for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass(frozen=True)
class Config:
    zone: str
    currency: str
    start: pd.Timestamp
    end: pd.Timestamp
    paths: Paths
    solver: SolverConfig
    fuels: FuelSettings
    battery_asset: BatteryParams
    fleet: Fleet
    max_interpolation_hours: int = 3
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ConfigError("project.end must be after project.start")

    @property
    def period_tag(self) -> str:
        return f"{self.zone}_{self.start:%Y%m%d}_{self.end:%Y%m%d}"


def _to_utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def read_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {path} must be a mapping")
    return data


def build_config(base: Mapping[str, Any], fleet: Mapping[str, Any], root: Path = REPO_ROOT) -> Config:
    """Construct a validated :class:`Config` from already-parsed dictionaries."""
    project = _require(base, "project", "base config")
    return Config(
        zone=str(_require(project, "zone", "project")),
        currency=str(project.get("currency", "EUR")),
        start=_to_utc(_require(project, "start", "project")),
        end=_to_utc(_require(project, "end", "project")),
        paths=Paths.from_dict(_require(base, "paths", "base config"), root),
        solver=SolverConfig.from_dict(base.get("solver") or {}),
        fuels=FuelSettings.from_dict(base.get("fuels") or {}),
        battery_asset=BatteryParams.from_dict(_require(base, "battery_asset", "base config")),
        fleet=Fleet.from_dict(fleet),
        max_interpolation_hours=int((base.get("data") or {}).get("max_interpolation_hours", 3)),
        raw={"base": copy.deepcopy(dict(base)), "fleet": copy.deepcopy(dict(fleet))},
    )


def load_config(base_path: Path | str = DEFAULT_BASE_CONFIG, fleet_path: Path | str | None = None) -> Config:
    """Load ``base.yaml`` and the fleet file it references (or ``fleet_path``)."""
    base_path = Path(base_path)
    base = read_yaml(base_path)
    root = REPO_ROOT if base_path.resolve().is_relative_to(REPO_ROOT) else base_path.resolve().parents[1]
    if fleet_path is None:
        fleet_path = Path(_require(_require(base, "paths", "base config"), "fleet_file", "paths"))
        fleet_path = fleet_path if fleet_path.is_absolute() else root / fleet_path
    return build_config(base, read_yaml(fleet_path), root=root)
