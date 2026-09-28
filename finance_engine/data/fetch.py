"""Fetch raw OHLCV bars from an external source.

This module only talks to the outside world. It does not cache, clean or
validate — that is loader.py's job.

Output of every fetch_* function:
    pd.DataFrame indexed by a DatetimeIndex (may be tz-aware), containing at
    least Open/High/Low/Close/Volume columns in any letter case. Extra columns
    are allowed. Rows are NOT guaranteed sorted, unique or NaN-free.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

DATE_COLUMN_CANDIDATES = ("date", "datetime", "timestamp")


class DataFetchError(RuntimeError):
    """The external source could not provide any bars for the requested ticker."""


def fetch_ohlcv_from_yfinance(
    ticker: str,
    start: pd.Timestamp,
    *,
    interval: str,
    auto_adjust: bool,
    drop_todays_bar: bool,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Download bars from Yahoo Finance from `start` up to the latest available bar."""
    try:
        bars = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=auto_adjust,
            actions=False,
        )
    except Exception as error:
        raise DataFetchError(f"yfinance request for {ticker!r} failed: {error}") from error

    if bars is None or bars.empty:
        raise DataFetchError(f"yfinance returned no bars for {ticker!r} (unknown ticker or no data since {start.date()})")

    if drop_todays_bar:
        bars = drop_incomplete_today_bar(bars, now=now)
    return bars


def drop_incomplete_today_bar(bars: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Remove any bar dated today or later in the exchange's own timezone.

    During market hours that bar is still forming; letting it through would give
    indicators a partial day that later changes — a subtle form of lookahead.
    """
    exchange_tz = bars.index.tz
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    if exchange_tz is not None:
        now = now.tz_localize("UTC") if now.tzinfo is None else now
        today = now.tz_convert(exchange_tz).normalize()
    else:
        today = (now.tz_convert(None) if now.tzinfo is not None else now).normalize()
    return bars[bars.index < today]


def fetch_ohlcv_from_csv(ticker: str, csv_directory: Path) -> pd.DataFrame:
    """Read {csv_directory}/{ticker}.csv, using its date column as the index."""
    csv_path = Path(csv_directory) / f"{ticker}.csv"
    if not csv_path.exists():
        raise DataFetchError(f"No CSV file for {ticker!r} at {csv_path}")

    bars = pd.read_csv(csv_path)
    date_column = next((col for col in bars.columns if col.strip().lower() in DATE_COLUMN_CANDIDATES), None)
    if date_column is None:
        raise DataFetchError(f"{csv_path} has no date column (expected one of {DATE_COLUMN_CANDIDATES})")

    bars[date_column] = pd.to_datetime(bars[date_column])
    return bars.set_index(date_column)
