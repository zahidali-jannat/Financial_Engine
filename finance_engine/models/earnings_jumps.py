"""Earnings announcements as scheduled jumps in variance.

Most genuine "jumps" in a single stock are results days. Unlike Merton's random
jumps, their dates are known in advance, so the forecast can widen exactly on
the day the market reacts instead of spreading jump risk over every day.

    k = E[ε² on reaction days] / E[ε² on other days]      (ε = demeaned log return)

On a reaction day inside the forecast window, that day's shock is scaled by √k.
k is estimated only from announcements before the forecast date, and only when
at least `min_events` exist; otherwise no adjustment is made.
"""

import numpy as np
import pandas as pd


def earnings_reaction_days(
    announcement_times: pd.DatetimeIndex,
    trading_days: pd.DatetimeIndex,
    *,
    exchange_timezone: str,
    market_close_time: str,
) -> pd.DatetimeIndex:
    """Map each announcement to the first session whose price can react to it.

    Announced before the close → that session (if it is one); at/after the close
    or on a holiday → the next session. Timestamps without a timezone are taken
    as exchange-local.
    """
    if len(announcement_times) == 0:
        return pd.DatetimeIndex([])
    times = pd.DatetimeIndex(announcement_times)
    times = times.tz_localize(exchange_timezone) if times.tz is None else times.tz_convert(exchange_timezone)
    close_hour, close_minute = (int(part) for part in market_close_time.split(":"))
    after_close = (times.hour * 60 + times.minute) >= close_hour * 60 + close_minute
    dates = times.tz_localize(None).normalize()
    earliest_reaction = dates + pd.to_timedelta(after_close.astype(int), unit="D")

    positions = trading_days.searchsorted(earliest_reaction)
    valid = positions < len(trading_days)
    return pd.DatetimeIndex(trading_days[positions[valid]]).unique()


def estimate_earnings_variance_multiplier(
    log_returns: pd.Series,
    reaction_days: pd.DatetimeIndex,
    *,
    min_events: int,
) -> float | None:
    """k = mean squared shock on reaction days / on other days, floored at 1; None if too few events."""
    demeaned = log_returns - log_returns.mean()
    is_reaction = demeaned.index.isin(reaction_days)
    if is_reaction.sum() < min_events:
        return None
    ratio = float(np.mean(demeaned[is_reaction] ** 2) / np.mean(demeaned[~is_reaction] ** 2))
    return max(ratio, 1.0)


def horizon_shock_multipliers(forecast_days: pd.DatetimeIndex, reaction_days: pd.DatetimeIndex, variance_multiplier: float | None) -> np.ndarray:
    """Per-day shock scale for the forecast window: √k on reaction days, 1 elsewhere."""
    multipliers = np.ones(len(forecast_days))
    if variance_multiplier is not None:
        multipliers[np.asarray(forecast_days.isin(reaction_days))] = np.sqrt(variance_multiplier)
    return multipliers
