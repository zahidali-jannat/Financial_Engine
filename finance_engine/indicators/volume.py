"""Volume indicators — is money flowing into or out of the stock?

How they differ:
    obv_slope          — On-Balance Volume: adds the whole day's volume on up-closes and
                         subtracts it on down-closes. Crude (a +0.1% day counts fully) but
                         captures persistent accumulation. Its slope, not level, carries meaning.
    vwap_distance      — close vs the rolling volume-weighted average price: are recent
                         buyers, on average, in profit (+) or at a loss (−)?
    chaikin_money_flow — WHERE each day closed inside its high–low range, weighted by volume:
                         distinguishes strong closes from weak ones on the same net change.

Data caveat: all three need real volume. Indices on Yahoo report zero volume
on many days, so these indicators are only meaningful for individual stocks.
"""

import numpy as np

from finance_engine.indicators.base import Indicator
from finance_engine.normalization.normalize import scale_by_trailing_std


class ObvSlope(Indicator):
    name, category = "obv_slope", "volume"

    def compute(self, bars):
        window = self.parameters["slope_window"]
        signed_volume = np.sign(bars["close"].diff()).fillna(0.0) * bars["volume"]
        obv = signed_volume.cumsum()
        average_volume = bars["volume"].rolling(window).mean()
        # OBV change over the window, in units of "days of average volume".
        return obv.diff(window) / (window * average_volume.replace(0.0, np.nan))

    def normalize(self, raw):
        # Sign = net accumulation (+) or distribution (−); scaled by its own trailing variability.
        return scale_by_trailing_std(raw, self.normalization["scale_window"])


class VwapDistance(Indicator):
    name, category = "vwap_distance", "volume"

    def compute(self, bars):
        window = self.parameters["window"]
        typical = (bars["high"] + bars["low"] + bars["close"]) / 3
        vwap = (typical * bars["volume"]).rolling(window).sum() / bars["volume"].rolling(window).sum().replace(0.0, np.nan)
        return bars["close"] / vwap - 1

    def normalize(self, raw):
        # 0 = price at the average cost of recent volume. Scaled by trailing std, since the
        # typical distance depends on the stock's volatility.
        return scale_by_trailing_std(raw, self.normalization["scale_window"])


class ChaikinMoneyFlow(Indicator):
    name, category = "chaikin_money_flow", "volume"

    def compute(self, bars):
        period = self.parameters["period"]
        day_range = (bars["high"] - bars["low"]).replace(0.0, np.nan)
        close_location = ((bars["close"] - bars["low"]) - (bars["high"] - bars["close"])) / day_range
        money_flow_volume = close_location.fillna(0.0) * bars["volume"]
        return money_flow_volume.rolling(period).sum() / bars["volume"].rolling(period).sum().replace(0.0, np.nan)

    def normalize(self, raw):
        # CMF is bounded in [-1, +1] but in practice sits within ±0.3, so a linear map would
        # waste most of the scale. tanh(CMF / trailing std) makes "unusual for this stock" read as large.
        return scale_by_trailing_std(raw, self.normalization["scale_window"])

