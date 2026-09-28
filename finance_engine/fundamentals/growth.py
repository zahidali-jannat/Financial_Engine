"""Growth and earnings quality.

Growth — compound annual growth of revenue and net income over the annual
statements Yahoo provides (~4 years), plus the year-on-year change in the latest
quarter's revenue when 5 quarters exist. Only ~5 quarters of statements are
available for free, so the 8-12 quarter margin trend the design called for
cannot be computed; the latest quarterly margins are reported instead.
    score = mean of tanh(growth / growth_scale) over the available growth rates

Earnings surprise — over the last 12 reported quarters with an analyst estimate:
    score = ½·(2·beat_rate − 1) + ½·tanh(mean surprise % / surprise_scale)
Consistent, sizeable beats → positive; habitual misses → negative.
"""

import math

import numpy as np
import pandas as pd

from finance_engine.fundamentals.valuation import ComponentScore

SURPRISE_LOOKBACK_QUARTERS = 12


def _row(statement: pd.DataFrame, names: tuple[str, ...]) -> pd.Series | None:
    for name in names:
        if name in statement.index:
            values = statement.loc[name].dropna().astype(float)
            return values.sort_index() if len(values) else None
    return None


def compound_growth(values: pd.Series) -> float | None:
    """CAGR between the oldest and newest positive values."""
    if values is None or len(values) < 2 or values.iloc[0] <= 0 or values.iloc[-1] <= 0:
        return None
    years = (values.index[-1] - values.index[0]).days / 365.25
    return (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1 if years > 0 else None


def growth_score(annual_income: pd.DataFrame, quarterly_income: pd.DataFrame, growth_scale: float) -> ComponentScore:
    rates, parts = [], []
    revenue = _row(annual_income, ("Total Revenue", "Operating Revenue"))
    net_income = _row(annual_income, ("Net Income", "Net Income Common Stockholders"))
    for label, series in (("revenue", revenue), ("net income", net_income)):
        rate = compound_growth(series)
        if rate is not None:
            rates.append(rate)
            parts.append(f"{label} {rate:+.1%}/yr over {len(series) - 1} years")

    quarterly_revenue = _row(quarterly_income, ("Total Revenue", "Operating Revenue"))
    if quarterly_revenue is not None and len(quarterly_revenue) >= 5 and quarterly_revenue.iloc[-5] > 0:
        year_on_year = quarterly_revenue.iloc[-1] / quarterly_revenue.iloc[-5] - 1
        rates.append(year_on_year)
        parts.append(f"latest quarter revenue {year_on_year:+.1%} year on year")

    quarterly_net = _row(quarterly_income, ("Net Income", "Net Income Common Stockholders"))
    if quarterly_revenue is not None and quarterly_net is not None:
        margins = (quarterly_net / quarterly_revenue).dropna()
        if len(margins) >= 2:
            parts.append(f"net margin {margins.iloc[0]:.1%} → {margins.iloc[-1]:.1%} over the last {len(margins)} quarters")

    if not rates:
        return ComponentScore(None, "income statements unavailable")
    return ComponentScore(float(np.mean([math.tanh(rate / growth_scale) for rate in rates])), "; ".join(parts))


def earnings_surprise_score(earnings: pd.DataFrame, surprise_scale_pct: float) -> ComponentScore:
    history = earnings.dropna(subset=["reported_eps", "eps_estimate", "surprise_pct"]).sort_index().tail(SURPRISE_LOOKBACK_QUARTERS)
    if len(history) < 4:
        return ComponentScore(None, "fewer than 4 quarters with analyst estimates")
    beat_rate = float((history["surprise_pct"] > 0).mean())
    # Median, not mean: one near-zero estimate can produce a ±1000% "surprise".
    typical_surprise = float(history["surprise_pct"].median())
    score = 0.5 * (2 * beat_rate - 1) + 0.5 * math.tanh(typical_surprise / surprise_scale_pct)
    return ComponentScore(
        score, f"beat estimates in {beat_rate:.0%} of the last {len(history)} quarters; median surprise {typical_surprise:+.1f}%"
    )
