"""Trend regime labels from ADX: trending vs ranging.

ADX measures trend strength, not direction. Above the threshold (Wilder's 25)
the market is "trending", below it "ranging". Layer 4 lets signal weights
differ between the two (trend-following signals may only help in trends).
The volatility regime (calm vs turbulent) comes from the HMM, not from here.
"""

import pandas as pd


def is_trending(adx: pd.Series, threshold: float) -> pd.Series:
    """1.0 where ADX > threshold, 0.0 where not, NaN during ADX warm-up."""
    return (adx > threshold).astype(float).where(adx.notna())
