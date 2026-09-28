"""Causal building blocks for mapping raw indicator values onto [-1, +1].

Every helper uses trailing windows only, so a normalized value at day t depends
on data through t — the same guarantee the raw indicators give. Each indicator
chooses which helper fits its meaning (see the indicator's `normalize`):

    bounded_to_unit        — for indicators with a natural centre and range (RSI: 0–100, centre 50)
    scale_by_trailing_std  — for unbounded, sign-meaningful values (MACD histogram):
                             tanh(x / trailing std) keeps the sign and squashes outliers
    trailing_percentile_to_unit — for level-type values with no natural zero (ATR %):
                             "how high is this compared with the past year?"
"""

import numpy as np
import pandas as pd


def bounded_to_unit(values: pd.Series, centre: float, half_range: float) -> pd.Series:
    """Linear map of [centre − half_range, centre + half_range] to [-1, +1], clipped."""
    return ((values - centre) / half_range).clip(-1.0, 1.0)


def scale_by_trailing_std(values: pd.Series, window: int) -> pd.Series:
    """tanh(x / σ_trailing): 1σ → ±0.76, 2σ → ±0.96. Zero stays zero, so the sign keeps its meaning."""
    trailing_std = values.rolling(window, min_periods=window // 2).std()
    return np.tanh(values / trailing_std.replace(0.0, np.nan))


def tanh_scaled(values: pd.Series, scale: float) -> pd.Series:
    """tanh(x / scale) for indicators whose conventional reading scale is known (e.g. CCI ±100)."""
    return np.tanh(values / scale)


def trailing_percentile_to_unit(values: pd.Series, window: int) -> pd.Series:
    """Rank of today's value within the trailing window, mapped to [-1, +1] (−1 = lowest, +1 = highest)."""
    def percentile_of_last(window_values: np.ndarray) -> float:
        last = window_values[-1]
        return (np.sum(window_values < last) + 0.5 * (np.sum(window_values == last) - 1)) / (len(window_values) - 1)

    percentile = values.rolling(window, min_periods=window // 2).apply(percentile_of_last, raw=True)
    return 2 * percentile - 1
