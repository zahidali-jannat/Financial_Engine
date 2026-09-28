"""Fetch and cache fundamental data from Yahoo Finance.

What Yahoo provides for NSE stocks (checked, not assumed):
    info                 — CURRENT ratios only (trailing P/E, P/B, EV/EBITDA, sector); no history
    earnings history     — ~15-20 years of quarterly EPS: estimate, reported, surprise %,
                           with announcement timestamps (delivered in New York time)
    income statements    — only the last ~5 quarters and ~4-5 years
Anything missing comes back empty; Layer 2 then marks that component unavailable
rather than guessing.

Output contract (FundamentalData):
    info              dict of the INFO_FIELDS below (values may be None)
    earnings          DataFrame indexed by tz-aware announcement time, columns
                      eps_estimate, reported_eps, surprise_pct (NaN for upcoming dates)
    quarterly_income  DataFrame, rows = line items, columns = period-end dates (newest first)
    annual_income     same, annual
"""

import json
import logging
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pandas as pd
import yfinance as yf

from finance_engine.data.cache import cache_file_paths

logger = logging.getLogger(__name__)

INFO_FIELDS = (
    "trailingPE", "forwardPE", "priceToBook", "enterpriseToEbitda", "trailingEps",
    "sector", "industry", "marketCap", "currency", "longName",
)
EARNINGS_COLUMNS = {"EPS Estimate": "eps_estimate", "Reported EPS": "reported_eps", "Surprise(%)": "surprise_pct"}


@dataclass(frozen=True)
class FundamentalData:
    info: dict
    earnings: pd.DataFrame
    quarterly_income: pd.DataFrame
    annual_income: pd.DataFrame


def fetch_fundamentals(ticker: str, earnings_limit: int = 100) -> FundamentalData:
    handle = yf.Ticker(ticker)
    info = _attempt(lambda: {key: handle.info.get(key) for key in INFO_FIELDS}, {}, f"{ticker} info")
    earnings = _attempt(lambda: _clean_earnings(handle.get_earnings_dates(limit=earnings_limit)), _empty_earnings(), f"{ticker} earnings history")
    quarterly = _attempt(lambda: handle.quarterly_income_stmt, pd.DataFrame(), f"{ticker} quarterly income statement")
    annual = _attempt(lambda: handle.income_stmt, pd.DataFrame(), f"{ticker} annual income statement")
    return FundamentalData(info, earnings, _standardize_statement(quarterly), _standardize_statement(annual))


def _standardize_statement(statement: pd.DataFrame | None) -> pd.DataFrame:
    """Period-end dates as datetime64[ns] columns, newest first — identical whether fetched or read from cache."""
    if statement is None or statement.empty:
        return pd.DataFrame()
    statement = statement.copy()
    statement.columns = pd.DatetimeIndex(pd.to_datetime(statement.columns)).as_unit("ns")
    return statement.sort_index(axis=1, ascending=False)


def load_fundamentals(ticker: str, cache_directory: Path, *, max_age_hours: float, now: pd.Timestamp | None = None) -> FundamentalData:
    """Cached wrapper around fetch_fundamentals (one JSON file per ticker)."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    path, _ = cache_file_paths(cache_directory, ticker, "fundamentals")
    path = path.with_suffix(".json")
    if path.exists():
        stored = json.loads(path.read_text())
        age_hours = (now - pd.Timestamp(stored["fetched_at_utc"])).total_seconds() / 3600
        if age_hours <= max_age_hours:
            return _from_json(stored)

    data = fetch_fundamentals(ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_json(data, now)))
    return data


def _attempt(getter, fallback, description):
    try:
        value = getter()
        return fallback if value is None else value
    except Exception as error:  # Yahoo endpoints fail independently; one gap shouldn't stop the rest
        logger.warning("Could not fetch %s: %s", description, error)
        return fallback


def _empty_earnings() -> pd.DataFrame:
    return pd.DataFrame(columns=list(EARNINGS_COLUMNS.values()), index=pd.DatetimeIndex([], tz="UTC"), dtype=float)


def _clean_earnings(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return _empty_earnings()
    earnings = raw.rename(columns=EARNINGS_COLUMNS)[list(EARNINGS_COLUMNS.values())].astype(float)
    earnings.index = pd.DatetimeIndex(earnings.index).tz_convert("UTC").as_unit("us")
    earnings.index.name = "announced_at"
    return earnings[~earnings.index.duplicated()].sort_index()


def _to_json(data: FundamentalData, now: pd.Timestamp) -> dict:
    def frame(df):
        return df.to_json(orient="split", date_format="iso") if not df.empty else None

    return {
        "fetched_at_utc": now.isoformat(),
        "info": data.info,
        "earnings": frame(data.earnings),
        "quarterly_income": frame(data.quarterly_income),
        "annual_income": frame(data.annual_income),
    }


def _from_json(stored: dict) -> FundamentalData:
    def statement(text):
        if not text:
            return pd.DataFrame()
        return _standardize_statement(pd.read_json(StringIO(text), orient="split").astype(float))

    earnings = _empty_earnings()
    if stored["earnings"]:
        earnings = pd.read_json(StringIO(stored["earnings"]), orient="split").astype(float)  # JSON drops the ".0"
        earnings.index = pd.to_datetime(earnings.index, utc=True).as_unit("us")
        earnings.index.name = "announced_at"
    return FundamentalData(stored["info"], earnings, statement(stored["quarterly_income"]), statement(stored["annual_income"]))
