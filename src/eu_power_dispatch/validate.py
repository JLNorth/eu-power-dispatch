"""Compare modelled clearing prices with actual day-ahead prices.

Metrics: MAE, RMSE, Pearson correlation, mean bias (modelled - actual), plus
the same on the sorted price-duration curves.

Plots (PNG, written next to the input file):

* ``price_duration_curve.png`` - sorted modelled vs actual prices
* ``price_scatter.png``        - hourly modelled vs actual with a 45-degree line
* ``price_overlay.png``        - two-week time-series overlay

Usage::

    python -m eu_power_dispatch.validate --run-name base [--overlay-start 2024-01-15]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eu_power_dispatch.config import load_config  # noqa: E402

log = logging.getLogger(__name__)

# Plot styling: two categorical series on a light surface, recessive axes.
ACTUAL_COLOR = "#2a78d6"
MODELLED_COLOR = "#eb6834"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"


def _style() -> dict:
    return {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_MUTED,
        "axes.titlecolor": TEXT,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "font.size": 10,
    }


def price_metrics(actual: pd.Series, modelled: pd.Series) -> dict[str, float]:
    """Hourly error metrics on aligned, non-missing pairs."""
    pair = pd.concat([actual.rename("actual"), modelled.rename("modelled")], axis=1).dropna()
    if pair.empty:
        raise ValueError("No overlapping non-missing hours between actual and modelled prices")
    err = pair["modelled"] - pair["actual"]
    a_sorted = np.sort(pair["actual"].to_numpy())[::-1]
    m_sorted = np.sort(pair["modelled"].to_numpy())[::-1]
    return {
        "hours": int(len(pair)),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "correlation": float(pair["actual"].corr(pair["modelled"])),
        "mean_bias": float(err.mean()),
        "mean_actual": float(pair["actual"].mean()),
        "mean_modelled": float(pair["modelled"].mean()),
        "pdc_mae": float(np.abs(m_sorted - a_sorted).mean()),
    }


def plot_price_duration(actual: pd.Series, modelled: pd.Series, currency: str, path: Path) -> None:
    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for series, label, color in ((actual, "Actual", ACTUAL_COLOR), (modelled, "Modelled", MODELLED_COLOR)):
            values = np.sort(series.dropna().to_numpy())[::-1]
            ax.plot(np.linspace(0, 100, len(values)), values, label=label, color=color)
        ax.set_title("Price-duration curve")
        ax.set_xlabel("% of hours")
        ax.set_ylabel(f"{currency}/MWh")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_scatter(actual: pd.Series, modelled: pd.Series, currency: str, path: Path) -> None:
    pair = pd.concat([actual.rename("a"), modelled.rename("m")], axis=1).dropna()
    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(pair["a"], pair["m"], s=8, alpha=0.35, color=ACTUAL_COLOR, linewidths=0)
        lo, hi = np.nanpercentile(pair.to_numpy(), [0.5, 99.5])
        ax.plot([lo, hi], [lo, hi], color=TEXT_MUTED, linewidth=1, linestyle="--", label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title("Hourly modelled vs actual price")
        ax.set_xlabel(f"Actual ({currency}/MWh)")
        ax.set_ylabel(f"Modelled ({currency}/MWh)")
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_overlay(
    actual: pd.Series, modelled: pd.Series, currency: str, path: Path, start: pd.Timestamp | None = None, days: int = 14
) -> None:
    start = start or actual.index.min()
    end = start + pd.Timedelta(days=days)
    window = slice(start, end - pd.Timedelta(hours=1))
    with plt.rc_context(_style()):
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(actual.loc[window], label="Actual", color=ACTUAL_COLOR)
        ax.plot(modelled.loc[window], label="Modelled", color=MODELLED_COLOR)
        ax.set_title(f"Hourly prices, {start:%d %b %Y} to {end - pd.Timedelta(days=1):%d %b %Y}")
        ax.set_ylabel(f"{currency}/MWh")
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=2)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def validate(prices: pd.DataFrame, out_dir: Path, currency: str, overlay_start: pd.Timestamp | None = None) -> dict:
    actual, modelled = prices["da_price"], prices["modelled_price"]
    metrics = price_metrics(actual, modelled)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_price_duration(actual, modelled, currency, out_dir / "price_duration_curve.png")
    plot_scatter(actual, modelled, currency, out_dir / "price_scatter.png")
    plot_overlay(actual, modelled, currency, out_dir / "price_overlay.png", start=overlay_start)
    (out_dir / "validation_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate modelled prices against actuals.")
    parser.add_argument("--config", default=None, help="Path to base.yaml")
    parser.add_argument("--run-name", default="base")
    parser.add_argument("--overlay-start", default=None, help="Start date (UTC) of the two-week overlay")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config) if args.config else load_config()
    run_dir = cfg.paths.outputs_dir / args.run_name
    path = run_dir / "modelled_prices.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `epd-market --run-name {args.run_name}` first")
    start = pd.Timestamp(args.overlay_start, tz="UTC") if args.overlay_start else None
    metrics = validate(pd.read_parquet(path), run_dir, cfg.currency, start)
    for key, value in metrics.items():
        print(f"{key:<15}{value:>12.3f}" if isinstance(value, float) else f"{key:<15}{value:>12}")


if __name__ == "__main__":
    main()
