"""Assemble the evidence matrix the Bayesian drift model learns from.

Columns (all causal, one row per day):
    trend, momentum, volatility, volume   — Layer 1 category composites
    kalman_trend                          — Kalman slope ÷ its standard error (a t-statistic for "is there a trend")
    trend_when_trending, momentum_when_trending
                                          — the composite × 1[ADX regime = trending], so the model can
                                            weight trend and oscillator signals differently in trending
                                            vs ranging markets (regime-conditional weighting)
"""

import numpy as np
import pandas as pd

from finance_engine.aggregation.composite import category_composites


def build_evidence(
    normalized_indicators: pd.DataFrame,
    kept_indicators: list[str],
    indicator_categories: dict[str, str],
    kalman_trend_z: np.ndarray,
    trending: pd.Series,
) -> pd.DataFrame:
    evidence = category_composites(normalized_indicators, kept_indicators, indicator_categories)
    evidence["kalman_trend"] = kalman_trend_z
    evidence["trend_when_trending"] = evidence["trend"] * trending
    evidence["momentum_when_trending"] = evidence["momentum"] * trending
    return evidence


def forward_log_returns(log_prices: np.ndarray, horizon: int, index: pd.Index) -> pd.Series:
    """y_t = log P_{t+h} − log P_t, NaN where t + h is beyond the data. Uses the FUTURE by definition —
    only ever read at rows whose t + h is on or before the forecast date."""
    values = np.full(len(log_prices), np.nan)
    values[:-horizon] = log_prices[horizon:] - log_prices[:-horizon]
    return pd.Series(values, index=index, name="forward_log_return")
