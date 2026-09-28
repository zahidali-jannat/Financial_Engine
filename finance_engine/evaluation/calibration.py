"""Calibration checks: when the model says X%, does it happen X% of the time?

Two views, because they test different claims:
    reliability_table     — directional: bins forecasts by P(up) and compares the
                            average forecast with how often the price actually rose.
                            Quantile bins, since P(up) clusters near 50%.
    interval_calibration  — distributional: for central intervals from 10% to 95%,
                            the share of outcomes that landed inside. A calibrated
                            model sits on the diagonal at every level.
"""

import numpy as np
import pandas as pd

from finance_engine.evaluation.model_comparison import coverage_tolerance


def reliability_table(probability_up: pd.Series, realized: pd.Series, bins: int) -> pd.DataFrame:
    went_up = (realized > 0).astype(float)
    bin_labels = pd.qcut(probability_up.rank(method="first"), q=bins, labels=False)
    table = pd.DataFrame({"p": probability_up, "up": went_up, "bin": bin_labels}).groupby("bin").agg(
        mean_forecast=("p", "mean"), observed_frequency=("up", "mean"), forecasts=("up", "size")
    )
    table["tolerance"] = [2 * np.sqrt(max(p * (1 - p), 1e-6) / n) for p, n in zip(table["mean_forecast"], table["forecasts"])]
    return table


def interval_calibration(pit: pd.Series) -> pd.DataFrame:
    """Coverage of central intervals at many levels, derived from PIT values: inside ⇔ |PIT − ½| ≤ level/2."""
    levels = np.round(np.arange(0.1, 1.0, 0.1).tolist() + [0.95], 2)
    distance_from_centre = (pit - 0.5).abs()
    rows = [
        {"nominal": level, "observed": float((distance_from_centre <= level / 2).mean()), "tolerance": coverage_tolerance(level, len(pit))}
        for level in levels
    ]
    return pd.DataFrame(rows).set_index("nominal")


def pit_histogram(pit: pd.Series, bins: int) -> pd.Series:
    """Share of outcomes per PIT decile; each should be ~1/bins. U-shape = overconfident, hump = underconfident."""
    counts, _ = np.histogram(pit, bins=bins, range=(0, 1))
    return pd.Series(counts / len(pit), index=[f"{i / bins:.1f}-{(i + 1) / bins:.1f}" for i in range(bins)])
