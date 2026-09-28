"""Momentum (oscillator) indicators — how stretched is the recent move?

How they differ:
    rsi               — balance of average up-closes vs down-closes (close-to-close only).
    stochastic        — where the close sits inside the recent high–low range. Uses
                        highs and lows, but in practice tracks RSI closely: expect the
                        redundancy check to flag the pair (the double-counting risk).
    cci               — distance of the typical price from its average, in units of
                        mean absolute deviation: unbounded, so it keeps extreme readings.
    bollinger_position — distance of the close from its 20-day mean in standard
                        deviations (%B centred). The Bollinger *position* is a momentum
                        / mean-reversion reading; the band *width* is a volatility state
                        and lives in volatility.py.

Convention: +1 = strong upward momentum ("overbought"). Whether that predicts
continuation or reversal is an empirical question the Bayesian layer answers.
"""

import numpy as np
import pandas as pd

from finance_engine.indicators.base import Indicator, wilder_average
from finance_engine.normalization.normalize import bounded_to_unit, tanh_scaled


class Rsi(Indicator):
    name, category = "rsi", "momentum"

    def compute(self, bars):
        change = bars["close"].diff()
        average_gain = wilder_average(change.clip(lower=0), self.parameters["period"])
        average_loss = wilder_average((-change).clip(lower=0), self.parameters["period"])
        return 100 - 100 / (1 + average_gain / average_loss)  # all-gain window → loss 0 → RSI 100

    def normalize(self, raw):
        # RSI lives on 0–100 with 50 = balanced: 50 → 0, 70 (classic overbought) → +0.4, 100 → +1.
        return bounded_to_unit(raw, centre=50.0, half_range=50.0)


class Stochastic(Indicator):
    name, category = "stochastic", "momentum"

    def compute(self, bars):
        period = self.parameters["period"]
        lowest = bars["low"].rolling(period).min()
        highest = bars["high"].rolling(period).max()
        percent_k = 100 * (bars["close"] - lowest) / (highest - lowest).replace(0.0, np.nan)
        return percent_k.rolling(self.parameters["smoothing"]).mean()  # %D, the smoothed line

    def normalize(self, raw):
        # 0–100 with 50 = close mid-range: same linear map as RSI.
        return bounded_to_unit(raw, centre=50.0, half_range=50.0)


class Cci(Indicator):
    name, category = "cci", "momentum"

    def compute(self, bars):
        period = self.parameters["period"]
        typical = (bars["high"] + bars["low"] + bars["close"]) / 3
        average = typical.rolling(period).mean()
        mean_deviation = typical.rolling(period).apply(lambda w: np.mean(np.abs(w - w.mean())), raw=True)
        return (typical - average) / (0.015 * mean_deviation)

    def normalize(self, raw):
        # Lambert's constant 0.015 makes ~±100 the usual range; tanh keeps rare ±200+ readings
        # distinguishable without letting them dominate: ±100 → ±0.76, ±200 → ±0.96.
        return tanh_scaled(raw, self.normalization["cci_scale"])


class BollingerPosition(Indicator):
    name, category = "bollinger_position", "momentum"

    def compute(self, bars):
        period, width = self.parameters["period"], self.parameters["width_in_std"]
        middle = bars["close"].rolling(period).mean()
        std = bars["close"].rolling(period).std(ddof=0)
        return (bars["close"] - middle) / (width * std)  # 0 at the middle band, ±1 at the bands

    def normalize(self, raw):
        # Already expressed in band-widths; clipped because closes outside the bands are rare but possible.
        return raw.clip(-1.0, 1.0)
