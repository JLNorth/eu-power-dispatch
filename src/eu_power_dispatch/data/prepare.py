"""Merge raw system data and fuel prices into one clean hourly UTC frame.

Steps
-----
1. Normalise the index to tz-aware UTC, hour-beginning. Naive timestamps are
   assumed to be UTC. Working in UTC sidesteps DST: there are no repeated or
   missing hours, every day has 24 rows.
2. Drop duplicate timestamps (keep last) and reindex onto a complete hourly
   range ``[start, end)`` so gaps become explicit NaNs.
3. Linearly interpolate gaps up to ``data.max_interpolation_hours``; longer gaps
   in volumes are filled with the same hour from the previous day; hours still
   missing volumes are dropped. Long price gaps stay NaN (prices are only used
   for validation and battery revenue).
4. Compute ``residual_demand_mw = load - wind - solar - nuclear``.
5. Join hourly ``gas_price`` and ``carbon_price``.

Output: ``<processed_dir>/<ZONE>_<start>_<end>.parquet``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from eu_power_dispatch.config import Config, load_config
from eu_power_dispatch.data import RAW_COLUMNS
from eu_power_dispatch.data.fuels import load_fuel_prices

log = logging.getLogger(__name__)

VOLUME_COLUMNS = ("load_mw", "wind_mw", "solar_mw", "nuclear_mw")


def processed_path(cfg: Config) -> Path:
    return cfg.paths.processed_dir / f"{cfg.period_tag}.parquet"


def normalise_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with a sorted, de-duplicated, tz-aware UTC hourly index."""
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out = df.copy()
    out.index = idx.floor("h")
    n_dup = out.index.duplicated(keep="last").sum()
    if n_dup:
        log.warning("Dropping %d duplicate timestamps", n_dup)
        out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def fill_gaps(df: pd.DataFrame, max_interp_hours: int) -> pd.DataFrame:
    """Interpolate short gaps; fill long volume gaps with the previous day's profile."""
    out = df.interpolate(method="time", limit=max_interp_hours, limit_area="inside")
    vols = [c for c in VOLUME_COLUMNS if c in out.columns]
    out[vols] = out[vols].fillna(out[vols].shift(24))
    remaining = out[vols].isna().any(axis=1)
    if remaining.any():
        log.warning("Dropping %d hours with unrecoverable missing volumes", int(remaining.sum()))
        out = out[~remaining]
    if "da_price" in out.columns and out["da_price"].isna().any():
        # Prices are only used for validation and battery revenue; leave gaps as NaN.
        log.warning("%d hours have no day-ahead price", int(out["da_price"].isna().sum()))
    return out


def clean_system_data(
    raw: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, max_interp_hours: int = 3
) -> pd.DataFrame:
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Raw data is missing columns {missing}; expected {list(RAW_COLUMNS)}")
    df = normalise_index(raw[list(RAW_COLUMNS)].astype(float))
    full_index = pd.date_range(start, end, freq="1h", inclusive="left", tz="UTC", name="timestamp_utc")
    n_missing = len(full_index.difference(df.index))
    if n_missing:
        log.info("%d hours missing from raw data before gap filling", n_missing)
    df = df.reindex(full_index)
    df[list(VOLUME_COLUMNS)] = df[list(VOLUME_COLUMNS)].clip(lower=0)
    df = fill_gaps(df, max_interp_hours)
    df["residual_demand_mw"] = df["load_mw"] - df["wind_mw"] - df["solar_mw"] - df["nuclear_mw"]
    return df


def prepare(cfg: Config, raw: pd.DataFrame | None = None, write: bool = True) -> pd.DataFrame:
    """Build the model-ready hourly frame for ``cfg``'s zone and period."""
    if raw is None:
        from eu_power_dispatch.data.entsoe_loader import raw_path

        path = raw_path(cfg)
        if not path.exists():
            raise FileNotFoundError(f"Raw data not found at {path}. Run the ENTSO-E loader first.")
        raw = pd.read_parquet(path)

    df = clean_system_data(raw, cfg.start, cfg.end, cfg.max_interpolation_hours)
    fuels = load_fuel_prices(cfg.paths.fuels_csv, df.index, cfg.fuels)
    df = df.join(fuels)

    if write:
        out = processed_path(cfg)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
        log.info("Wrote %d rows to %s", len(df), out)
    return df


def load_prepared(cfg: Config) -> pd.DataFrame:
    path = processed_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"Prepared data not found at {path}. Run `epd-prepare` first.")
    return pd.read_parquet(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Clean raw data and join fuel prices.")
    parser.add_argument("--config", default=None, help="Path to base.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config) if args.config else load_config()
    df = prepare(cfg)
    print(df.describe().T.round(1).to_string())


if __name__ == "__main__":
    main()
