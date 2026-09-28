"""Market and sector context: how the stock is moving relative to the Nifty 50 and its sector index.

    relative strength (3 months) = stock return − index return over 63 sessions
    beta (1 year)                = cov(stock, Nifty) / var(Nifty) on daily log returns

Interest-rate sensitivity is not assessed: there is no reliable free daily
series for Indian government bond yields on Yahoo, and a proxy (e.g. US yields)
would measure the wrong thing. The report states this rather than inventing one.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

RELATIVE_STRENGTH_DAYS = 63
BETA_DAYS = 252


@dataclass(frozen=True)
class MarketContext:
    benchmark: str
    sector_index: str | None
    relative_strength_vs_benchmark: float | None
    relative_strength_vs_sector: float | None
    sector_vs_benchmark: float | None
    beta: float | None
    correlation: float | None


def _return_over(close: pd.Series, days: int) -> float | None:
    return float(close.iloc[-1] / close.iloc[-days - 1] - 1) if len(close) > days else None


def build_market_context(stock_close: pd.Series, benchmark_close: pd.Series | None, sector_close: pd.Series | None, benchmark: str, sector_index: str | None) -> MarketContext:
    stock_return = _return_over(stock_close, RELATIVE_STRENGTH_DAYS)

    def relative(index_close):
        if index_close is None or stock_return is None:
            return None
        index_return = _return_over(index_close.loc[: stock_close.index[-1]], RELATIVE_STRENGTH_DAYS)
        return None if index_return is None else stock_return - index_return

    beta = correlation = None
    if benchmark_close is not None:
        joined = pd.concat([np.log(stock_close).diff(), np.log(benchmark_close).diff()], axis=1, join="inner").dropna().tail(BETA_DAYS)
        if len(joined) > 60:
            covariance = np.cov(joined.iloc[:, 0], joined.iloc[:, 1])
            beta = float(covariance[0, 1] / covariance[1, 1])
            correlation = float(np.corrcoef(joined.iloc[:, 0], joined.iloc[:, 1])[0, 1])

    sector_vs_benchmark = None
    if sector_close is not None and benchmark_close is not None:
        sector_return, benchmark_return = _return_over(sector_close, RELATIVE_STRENGTH_DAYS), _return_over(benchmark_close, RELATIVE_STRENGTH_DAYS)
        if sector_return is not None and benchmark_return is not None:
            sector_vs_benchmark = sector_return - benchmark_return

    return MarketContext(benchmark, sector_index, relative(benchmark_close), relative(sector_close), sector_vs_benchmark, beta, correlation)
