"""Valuation: is the stock cheap or expensive versus its own history and versus its peers?

Own history: Yahoo gives only today's P/E, so the history is rebuilt point-in-time
from reported quarterly EPS: trailing-twelve-month EPS = sum of the last four
reported quarters, effective from each announcement date. P/E = close / TTM EPS
(undefined when TTM EPS ≤ 0). Today's percentile within that history maps to
    score = −(2·percentile − 1)   → +1 = cheapest ever seen, −1 = most expensive.
Caveat: Yahoo's "reported EPS" can mix standalone and consolidated figures for
some Indian companies; the report prints the rebuilt P/E next to Yahoo's own
trailing P/E so a mismatch is visible.

Peers: log(own multiple / peer median) averaged over P/E, P/B and EV/EBITDA,
    score = −tanh(mean log ratio / scale)   → cheaper than peers = positive.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

MULTIPLES = ("trailingPE", "priceToBook", "enterpriseToEbitda")
MIN_HISTORY_DAYS = 250


@dataclass(frozen=True)
class ComponentScore:
    score: float | None  # in [-1, +1]; + = supports undervalued; None = not available
    detail: str


def trailing_pe_history(close: pd.Series, earnings: pd.DataFrame, exchange_timezone: str) -> pd.Series:
    reported = earnings["reported_eps"].dropna().sort_index()
    if len(reported) < 4:
        return pd.Series(dtype=float)
    ttm_eps = reported.rolling(4).sum().dropna()
    effective_dates = ttm_eps.index.tz_convert(exchange_timezone).tz_localize(None).normalize()
    ttm_eps = pd.Series(ttm_eps.to_numpy(), index=effective_dates).groupby(level=0).last()
    ttm_on_trading_days = ttm_eps.reindex(close.index.union(ttm_eps.index)).ffill().reindex(close.index)
    return (close / ttm_on_trading_days.where(ttm_on_trading_days > 0)).dropna()


def valuation_vs_history(pe_history: pd.Series) -> ComponentScore:
    if len(pe_history) < MIN_HISTORY_DAYS:
        return ComponentScore(None, "not enough P/E history (needs ~1 year of positive trailing EPS)")
    current = pe_history.iloc[-1]
    percentile = float((pe_history < current).mean())
    score = -(2 * percentile - 1)
    years = (pe_history.index[-1] - pe_history.index[0]).days / 365.25
    return ComponentScore(
        score,
        f"P/E {current:.1f} is at the {percentile:.0%} percentile of its {years:.0f}-year range "
        f"({pe_history.min():.1f}–{pe_history.max():.1f}, median {pe_history.median():.1f})",
    )


def _plausible(value, upper_bound: float) -> bool:
    return isinstance(value, (int, float)) and 0 < value <= upper_bound


def valuation_vs_peers(own_info: dict, peer_infos: dict[str, dict], scale: float, plausible_max: dict) -> ComponentScore:
    if not peer_infos:
        return ComponentScore(None, "no peers configured for this ticker (settings.yaml → fundamentals.peers)")
    log_ratios, parts, excluded = [], [], []
    for multiple in MULTIPLES:
        upper = plausible_max[multiple]
        own = own_info.get(multiple)
        peers = []
        for peer, info in peer_infos.items():
            value = info.get(multiple)
            if _plausible(value, upper):
                peers.append(value)
            elif isinstance(value, (int, float)) and value > upper:
                excluded.append(f"{peer} {_label(multiple)} {value:.0f}")
        if not _plausible(own, upper) or len(peers) < 2:
            continue
        median = float(np.median(peers))
        log_ratios.append(math.log(own / median))
        parts.append(f"{_label(multiple)} {own:.1f} vs peer median {median:.1f}")
    if not log_ratios:
        return ComponentScore(None, "peer multiples unavailable")
    detail = "; ".join(parts) + f" (peers: {', '.join(peer_infos)})"
    if excluded:
        detail += f"; excluded as implausible data: {', '.join(excluded)}"
    return ComponentScore(-math.tanh(float(np.mean(log_ratios)) / scale), detail)


def _label(multiple: str) -> str:
    return {"trailingPE": "P/E", "priceToBook": "P/B", "enterpriseToEbitda": "EV/EBITDA"}[multiple]
