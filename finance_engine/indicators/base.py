"""Common interface for every technical indicator.

Contract:
    compute(bars)        -> pd.Series aligned to bars.index, raw indicator values (NaN during warm-up)
    normalize(raw)       -> pd.Series in [-1, +1] (NaN during warm-up)
    name, category       -> identifiers used in settings.yaml, redundancy pruning and reports
    is_directional       -> True if +1 means the conventional bullish reading;
                            False for state measures (volatility level), where +1 means "high"

Both methods are causal: the value at day t uses bars up to and including t.
`bars` follows the data-stage contract (open/high/low/close/volume, float64).
"""

from abc import ABC, abstractmethod

import pandas as pd


class Indicator(ABC):
    name: str
    category: str  # "trend" | "momentum" | "volatility" | "volume"
    is_directional: bool = True

    def __init__(self, parameters: dict, normalization: dict):
        self.parameters = parameters
        self.normalization = normalization

    @abstractmethod
    def compute(self, bars: pd.DataFrame) -> pd.Series: ...

    @abstractmethod
    def normalize(self, raw: pd.Series) -> pd.Series: ...

    def compute_normalized(self, bars: pd.DataFrame) -> pd.Series:
        return self.normalize(self.compute(bars)).rename(self.name)


def wilder_average(values: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (an EMA with α = 1/period), used by RSI, ATR and ADX."""
    return values.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def true_range(bars: pd.DataFrame) -> pd.Series:
    previous_close = bars["close"].shift(1)
    return pd.concat([bars["high"] - bars["low"], (bars["high"] - previous_close).abs(), (bars["low"] - previous_close).abs()], axis=1).max(axis=1)
