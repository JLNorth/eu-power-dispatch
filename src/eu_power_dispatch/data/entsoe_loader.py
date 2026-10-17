"""Fetch hourly system data from the ENTSO-E Transparency Platform.

Output schema (written to ``<raw_dir>/<ZONE>_<start>_<end>.parquet``):

==============  =========================================================
index           hourly ``DatetimeIndex``, tz-aware UTC, hour-beginning
``load_mw``     actual total load, MW (hourly mean)
``wind_mw``     wind onshore + offshore generation, MW
``solar_mw``    solar generation, MW
``nuclear_mw``  nuclear generation, MW
``da_price``    day-ahead clearing price, currency/MWh (label set in config)
==============  =========================================================

Sub-hourly series (15/30-minute resolution) are averaged to hourly.

GB note: GB data stopped flowing to ENTSO-E after the UK left the EU internal
energy market (mid-2021). For later periods, assemble the same schema from
Elexon BMRS / NESO data and save it as a parquet at the raw path instead;
:func:`fetch_entsoe` raises :class:`DataUnavailableError` with that instruction.

Usage::

    export ENTSOE_API_KEY=...
    python -m eu_power_dispatch.data.entsoe_loader --config config/base.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from eu_power_dispatch.config import Config, load_config
from eu_power_dispatch.data import RAW_COLUMNS

log = logging.getLogger(__name__)

API_KEY_ENV = "ENTSOE_API_KEY"

_WIND_TYPES = ("Wind Onshore", "Wind Offshore")
_SOLAR_TYPES = ("Solar",)
_NUCLEAR_TYPES = ("Nuclear",)


class DataUnavailableError(RuntimeError):
    """Raised when ENTSO-E cannot supply a required series for the period."""


def _schema_hint(zone: str, path: Path) -> str:
    return (
        f"ENTSO-E data for zone '{zone}' is unavailable for the requested period. "
        f"Supply a parquet at {path} instead, with an hourly tz-aware UTC DatetimeIndex and "
        f"columns {list(RAW_COLUMNS)} (MW for volumes, currency/MWh for da_price). "
        "For GB after 2021, Elexon BMRS / NESO data can be used to build this file."
    )


def raw_path(cfg: Config) -> Path:
    return cfg.paths.raw_dir / f"{cfg.period_tag}.parquet"


def get_api_key() -> str:
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(
            f"Environment variable {API_KEY_ENV} is not set. Copy .env.example and export your key."
        )
    return key


def _to_hourly_utc(obj: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    obj = obj.copy()
    obj.index = pd.DatetimeIndex(obj.index).tz_convert("UTC")
    return obj.resample("1h").mean()


def _actual_generation(gen: pd.DataFrame) -> pd.DataFrame:
    """Reduce entsoe-py generation output to one column per production type."""
    if isinstance(gen.columns, pd.MultiIndex):
        level1 = gen.columns.get_level_values(1)
        if "Actual Aggregated" in set(level1):
            gen = gen.xs("Actual Aggregated", axis=1, level=1)
        else:
            gen = gen.T.groupby(level=0).sum().T
    return gen


def _sum_types(gen: pd.DataFrame, types: tuple[str, ...], label: str, zone: str, path: Path) -> pd.Series:
    present = [t for t in types if t in gen.columns]
    if not present:
        if label == "nuclear_mw":
            log.warning("No nuclear generation reported for %s; assuming zero.", zone)
            return pd.Series(0.0, index=gen.index, name=label)
        raise DataUnavailableError(f"No '{label}' series ({types}) returned. " + _schema_hint(zone, path))
    return gen[present].sum(axis=1, min_count=1).rename(label)


def fetch_entsoe(zone: str, start: pd.Timestamp, end: pd.Timestamp, out_path: Path, api_key: str) -> pd.DataFrame:
    """Download load, wind, solar, nuclear and day-ahead price for ``zone``.

    Parameters
    ----------
    zone:
        ENTSO-E area code understood by entsoe-py (``GB``, ``DE_LU``, ``FR``...).
    start, end:
        UTC timestamps; ``end`` is exclusive.
    out_path:
        Destination parquet. Parent directories are created.
    api_key:
        ENTSO-E RESTful API security token.

    Returns
    -------
    DataFrame following the module-level schema.
    """
    from entsoe import EntsoePandasClient
    from entsoe.exceptions import NoMatchingDataError

    client = EntsoePandasClient(api_key=api_key)
    # entsoe-py expects tz-aware timestamps; querying in UTC avoids DST ambiguity.
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")

    try:
        log.info("Fetching load for %s", zone)
        load = client.query_load(zone, start=start, end=end)
        log.info("Fetching generation by type for %s", zone)
        gen = client.query_generation(zone, start=start, end=end)
        log.info("Fetching day-ahead prices for %s", zone)
        price = client.query_day_ahead_prices(zone, start=start, end=end)
    except NoMatchingDataError as exc:
        raise DataUnavailableError(_schema_hint(zone, out_path)) from exc

    load_s = load.iloc[:, 0] if isinstance(load, pd.DataFrame) else load
    gen = _to_hourly_utc(_actual_generation(gen))

    df = pd.concat(
        [
            _to_hourly_utc(load_s).rename("load_mw"),
            _sum_types(gen, _WIND_TYPES, "wind_mw", zone, out_path),
            _sum_types(gen, _SOLAR_TYPES, "solar_mw", zone, out_path),
            _sum_types(gen, _NUCLEAR_TYPES, "nuclear_mw", zone, out_path),
            _to_hourly_utc(price).rename("da_price"),
        ],
        axis=1,
    )
    df = df.loc[(df.index >= start) & (df.index < end), list(RAW_COLUMNS)]
    df.index.name = "timestamp_utc"

    empty = [c for c in RAW_COLUMNS if df[c].isna().all()]
    if df.empty or empty:
        raise DataUnavailableError(f"Empty series {empty or list(RAW_COLUMNS)}. " + _schema_hint(zone, out_path))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    log.info("Wrote %d hourly rows to %s", len(df), out_path)
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch hourly ENTSO-E data to parquet.")
    parser.add_argument("--config", default=None, help="Path to base.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config) if args.config else load_config()
    fetch_entsoe(cfg.zone, cfg.start, cfg.end, raw_path(cfg), get_api_key())


if __name__ == "__main__":
    main()
