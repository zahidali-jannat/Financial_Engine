import pandas as pd
import pytest

from finance_engine.data import fetch
from finance_engine.data.fetch import DataFetchError, drop_incomplete_today_bar, fetch_ohlcv_from_csv, fetch_ohlcv_from_yfinance
from tests.conftest import make_ohlcv_bars


def test_drop_incomplete_today_bar_uses_exchange_timezone():
    bars = make_ohlcv_bars(start="2025-03-03", periods=5, tz="Asia/Kolkata")  # Mon 3rd .. Fri 7th
    # 20:00 UTC on the 6th is already 01:30 on the 7th in India, so the 7th's bar is "today" there.
    trimmed = drop_incomplete_today_bar(bars, now=pd.Timestamp("2025-03-06 20:00", tz="UTC"))
    assert trimmed.index[-1].date() == pd.Timestamp("2025-03-06").date()


def test_drop_incomplete_today_bar_keeps_completed_history():
    bars = make_ohlcv_bars(start="2025-03-03", periods=5, tz="America/New_York")
    trimmed = drop_incomplete_today_bar(bars, now=pd.Timestamp("2025-03-20 15:00", tz="UTC"))
    assert len(trimmed) == 5


def test_yfinance_empty_response_raises(monkeypatch):
    class EmptyTicker:
        def __init__(self, ticker):
            pass

        def history(self, **kwargs):
            return pd.DataFrame()

    monkeypatch.setattr(fetch.yf, "Ticker", EmptyTicker)
    with pytest.raises(DataFetchError, match="no bars"):
        fetch_ohlcv_from_yfinance("NOPE", pd.Timestamp("2020-01-01"), interval="1d", auto_adjust=True, drop_todays_bar=True)


def test_csv_fetch_uses_date_column_as_index(tmp_path):
    make_ohlcv_bars(periods=10).to_csv(tmp_path / "TCS.csv")
    bars = fetch_ohlcv_from_csv("TCS", tmp_path)
    assert isinstance(bars.index, pd.DatetimeIndex)
    assert len(bars) == 10


def test_csv_fetch_missing_file_raises(tmp_path):
    with pytest.raises(DataFetchError, match="No CSV"):
        fetch_ohlcv_from_csv("TCS", tmp_path)
