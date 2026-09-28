"""Volatility-state indicators — how turbulent is the market for this stock right now?

These are NOT directional (is_directional = False): +1 means "volatility is at
the top of its past-year range", not "bullish". How they differ:
    atr_percent          — average true range / price: includes overnight gaps (true range).
    bollinger_bandwidth  — 20-day std of closes / mean: close-to-close only, reacts to
                           trend legs as well as noise ("squeeze" when low).
    realized_volatility  — std of daily log returns: the statistical definition, the same
                           quantity the stochastic models forecast.
"""

import numpy as np

from finance_engine.indicators.base import Indicator, true_range, wilder_average
from finance_engine.normalization.normalize import trailing_percentile_to_unit


class _PercentileNormalized(Indicator):
    category, is_directional = "volatility", False

    def normalize(self, raw):
        # No natural zero for a volatility level, so rank it against its own trailing year.
        return trailing_percentile_to_unit(raw, self.normalization["percentile_window"])


class AtrPercent(_PercentileNormalized):
    name = "atr_percent"

    def compute(self, bars):
        return wilder_average(true_range(bars), self.parameters["period"]) / bars["close"]


class BollingerBandwidth(_PercentileNormalized):
    name = "bollinger_bandwidth"

    def compute(self, bars):
        period, width = self.parameters["period"], self.parameters["width_in_std"]
        middle = bars["close"].rolling(period).mean()
        return 2 * width * bars["close"].rolling(period).std(ddof=0) / middle


class RealizedVolatility(_PercentileNormalized):
    name = "realized_volatility"

    def compute(self, bars):
        return np.log(bars["close"]).diff().rolling(self.parameters["window"]).std()
