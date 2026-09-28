"""Build the configured indicator set and compute every normalized indicator in one frame.

Output contract of compute_normalized_indicators:
    pd.DataFrame indexed like `bars`, one column per indicator name, values in
    [-1, +1] (NaN during each indicator's warm-up). Causal row by row.
"""

import pandas as pd

from finance_engine.indicators.base import Indicator
from finance_engine.indicators.momentum import BollingerPosition, Cci, Rsi, Stochastic
from finance_engine.indicators.trend import Adx, DirectionalMovement, Macd, MovingAverageSpread
from finance_engine.indicators.volatility import AtrPercent, BollingerBandwidth, RealizedVolatility
from finance_engine.indicators.volume import ChaikinMoneyFlow, ObvSlope, VwapDistance

# Maps each indicator class to the settings.yaml `indicators:` entry holding its parameters.
INDICATOR_PARAMETER_KEYS = {
    MovingAverageSpread: "moving_average_spread",
    Macd: "macd",
    DirectionalMovement: "directional_movement",
    Adx: "adx",
    Rsi: "rsi",
    Stochastic: "stochastic",
    Cci: "cci",
    BollingerPosition: "bollinger",
    AtrPercent: "atr",
    BollingerBandwidth: "bollinger",
    RealizedVolatility: "realized_volatility",
    ObvSlope: "obv",
    VwapDistance: "vwap",
    ChaikinMoneyFlow: "chaikin_money_flow",
}


def build_indicators(indicator_settings: dict, normalization_settings: dict) -> list[Indicator]:
    return [cls(indicator_settings[key], normalization_settings) for cls, key in INDICATOR_PARAMETER_KEYS.items()]


def compute_normalized_indicators(bars: pd.DataFrame, indicators: list[Indicator]) -> pd.DataFrame:
    return pd.concat([indicator.compute_normalized(bars) for indicator in indicators], axis=1)


def indicator_categories(indicators: list[Indicator]) -> dict[str, str]:
    return {indicator.name: indicator.category for indicator in indicators}
