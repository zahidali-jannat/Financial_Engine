"""Static PNG charts for the report: calibration (the key check for a probabilistic model) and the forecast.

Colours are the dataviz reference palette's validated categorical slots 1-2
(engine = blue, GBM baseline = orange); reference lines and tolerance bands are
neutral grey so they recede behind the data. Every chart has a title that says
what "good" looks like, since calibration plots are unfamiliar to most readers.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#8a8984"
BAND = "#d9d8d3"
ENGINE_COLOUR = "#2a78d6"
BASELINE_COLOUR = "#eb6834"


def _style(axis, title: str, xlabel: str, ylabel: str):
    axis.set_facecolor(SURFACE)
    axis.set_title(title, loc="left", fontsize=10.5, color=TEXT_PRIMARY, pad=10)
    axis.set_xlabel(xlabel, fontsize=9, color=TEXT_SECONDARY)
    axis.set_ylabel(ylabel, fontsize=9, color=TEXT_SECONDARY)
    axis.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    axis.grid(color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)


def plot_calibration(
    engine_intervals: pd.DataFrame,
    baseline_intervals: pd.DataFrame,
    reliability: pd.DataFrame,
    pit_shares: pd.Series,
    forecasts: int,
    path: Path,
) -> Path:
    figure, (left, middle, right) = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)

    nominal = engine_intervals.index.to_numpy(float)
    left.fill_between(nominal, nominal - engine_intervals["tolerance"], nominal + engine_intervals["tolerance"], color=BAND, alpha=0.6, linewidth=0, label="range expected from chance")
    left.plot([0, 1], [0, 1], color=REFERENCE, linewidth=1, linestyle="--")
    left.plot(nominal, baseline_intervals["observed"], color=BASELINE_COLOUR, linewidth=2, marker="o", markersize=5, label="GBM baseline")
    left.plot(nominal, engine_intervals["observed"], color=ENGINE_COLOUR, linewidth=2, marker="o", markersize=5, label="Engine")
    left.annotate("Engine", (nominal[-1], engine_intervals["observed"].iloc[-1]), xytext=(6, -10), textcoords="offset points", fontsize=8, color=TEXT_PRIMARY)
    left.annotate("GBM", (nominal[4], baseline_intervals["observed"].iloc[4]), xytext=(-30, 8), textcoords="offset points", fontsize=8, color=TEXT_PRIMARY)
    left.set_xlim(0, 1)
    left.set_ylim(0, 1)
    left.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=TEXT_SECONDARY)
    _style(left, "Interval calibration\n(on the dashed line = honest probabilities)", "Stated probability of the interval", "Share of outcomes inside it")

    middle.plot([0.3, 0.7], [0.3, 0.7], color=REFERENCE, linewidth=1, linestyle="--")
    # Points, not a connected line: bins are separate groups, and a line would make noise look like a pattern.
    middle.errorbar(
        reliability["mean_forecast"], reliability["observed_frequency"], yerr=reliability["tolerance"],
        color=ENGINE_COLOUR, ecolor=BAND, elinewidth=4, capsize=0, marker="o", markersize=9, linestyle="none", zorder=3,
    )
    middle.set_xlim(0.3, 0.7)
    middle.set_ylim(0.3, 0.7)
    _style(middle, f"Engine: stated P(up) vs how often it rose\n({len(reliability)} bins of ~{int(reliability['forecasts'].mean())} forecasts; grey = chance range)", "Forecast probability of an up move", "Observed share of up moves")

    bins = np.arange(len(pit_shares))
    expected = 1 / len(pit_shares)
    tolerance = 2 * np.sqrt(expected * (1 - expected) / forecasts)
    right.axhspan(expected - tolerance, expected + tolerance, color=BAND, alpha=0.6, linewidth=0)
    right.bar(bins, pit_shares.to_numpy(), width=0.8, color=ENGINE_COLOUR)
    right.axhline(expected, color=REFERENCE, linewidth=1, linestyle="--")
    right.set_xticks(bins, [label.split("-")[0] for label in pit_shares.index])
    _style(right, "Engine: where outcomes fell in the forecast\n(flat = calibrated; U-shape = overconfident)", "Forecast percentile of the actual outcome", "Share of forecasts")

    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return path


def plot_forecast_distribution(prices: np.ndarray, last_close: float, intervals: dict, title: str, path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    low, high = np.quantile(prices, [0.002, 0.998])
    axis.hist(prices[(prices >= low) & (prices <= high)], bins=80, color=ENGINE_COLOUR, zorder=2)
    # Interval edges as dashed lines behind the bars; their values go in the subtitle, so no label can collide.
    for interval_low, interval_high in intervals.values():
        for edge in (interval_low, interval_high):
            axis.axvline(edge, color=REFERENCE, linewidth=1, linestyle="--", zorder=1)
    subtitle = " · ".join(f"{level:.0%} interval {low:,.0f}–{high:,.0f}" for level, (low, high) in sorted(intervals.items()))
    axis.axvline(last_close, color=TEXT_PRIMARY, linewidth=1.5, zorder=3)
    axis.annotate(
        f"last close {last_close:,.2f}", (last_close, 0.93), xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points",
        fontsize=8, color=TEXT_PRIMARY, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5}, zorder=4,
    )
    axis.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    axis.set_yticks([])
    _style(axis, f"{title}\nDashed lines: {subtitle}", "Price at the end of the forecast window", "")
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return path
