"""Trend indicators — is price persistently moving one way?

How they differ (why all three are kept before redundancy pruning decides):
    moving_average_spread — WHERE the fast average sits relative to the slow one:
                            the level of the medium-term trend.
    macd                  — the CHANGE in that spread (histogram = MACD − signal):
                            trend acceleration/deceleration, so it turns before the spread does.
    directional_movement  — WHICH side has been making the bigger daily extensions
                            (+DI vs −DI): built from highs/lows, not closes.
    adx                   — HOW STRONG the trend is, regardless of direction. Not a
                            directional signal: used to label trending vs ranging regimes.
"""

import numpy as np
import pandas as pd

from finance_engine.indicators.base import Indicator, true_range, wilder_average
from finance_engine.normalization.normalize import bounded_to_unit, scale_by_trailing_std


class MovingAverageSpread(Indicator):
    name, category = "moving_average_spread", "trend"

    def compute(self, bars):
        fast = bars["close"].ewm(span=self.parameters["fast_ema"], adjust=False, min_periods=self.parameters["fast_ema"]).mean()
        slow = bars["close"].rolling(self.parameters["slow_sma"]).mean()
        return fast / slow - 1  # fractional gap: +2% = fast average 2% above slow

    def normalize(self, raw):
        # Zero = averages crossed; sign = which side. Scaled by the gap's own past
        # variability so a volatile stock's normal wobble doesn't read as a strong trend.
        return scale_by_trailing_std(raw, self.normalization["scale_window"])


class Macd(Indicator):
    name, category = "macd", "trend"

    def compute(self, bars):
        close = bars["close"]
        macd_line = close.ewm(span=self.parameters["fast"], adjust=False).mean() - close.ewm(span=self.parameters["slow"], adjust=False).mean()
        signal_line = macd_line.ewm(span=self.parameters["signal"], adjust=False).mean()
        histogram = (macd_line - signal_line) / close  # divided by price so it's comparable across price levels
        histogram.iloc[: self.parameters["slow"] + self.parameters["signal"]] = np.nan  # EMA warm-up
        return histogram

    def normalize(self, raw):
        # Histogram > 0 = trend accelerating upward. Unbounded, so scaled by its trailing std.
        return scale_by_trailing_std(raw, self.normalization["scale_window"])


def directional_indicators(bars: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder's +DI, −DI and ADX."""
    up_move = bars["high"].diff()
    down_move = -bars["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = wilder_average(true_range(bars), period)
    plus_di = 100 * wilder_average(plus_dm, period) / atr
    minus_di = 100 * wilder_average(minus_dm, period) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = wilder_average(dx, period)
    return plus_di, minus_di, adx


class DirectionalMovement(Indicator):
    name, category = "directional_movement", "trend"

    def compute(self, bars):
        plus_di, minus_di, _ = directional_indicators(bars, self.parameters["period"])
        return (plus_di - minus_di) / (plus_di + minus_di)  # already in [-1, +1]

    def normalize(self, raw):
        # Naturally bounded: +1 = all directional movement upward, −1 = all downward.
        return raw.clip(-1.0, 1.0)


class Adx(Indicator):
    name, category, is_directional = "adx", "trend", False

    def compute(self, bars):
        _, _, adx = directional_indicators(bars, self.parameters["period"])
        return adx

    def normalize(self, raw):
        # Strength, not direction: ADX 0 → −1 (no trend), 25 → 0 (Wilder's trend threshold), 50+ → +1.
        return bounded_to_unit(raw, centre=25.0, half_range=25.0)
