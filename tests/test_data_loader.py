import pandas as pd
import pytest

from finance_engine.data import loader
from finance_engine.data.loader import (
    OHLCV_COLUMNS,
    InsufficientHistoryError,
    InvalidOHLCVError,
    drop_holiday_placeholder_bars,
    ensure_minimum_history,
    load_ohlcv,
    standardize_ohlcv,
    validate_ohlcv,
)
from tests.conftest import make_ohlcv_bars

NOW = pd.Timestamp("2025-03-03 12:00", tz="UTC")


# --- standardize_ohlcv -------------------------------------------------------

def test_standardize_produces_contract_shape_from_yfinance_style_input():
    raw = make_ohlcv_bars(periods=50, tz="America/New_York").assign(Dividends=0.0)
    bars = standardize_ohlcv(raw.iloc[::-1], ohlc_repair_tolerance=0.005)

    assert list(bars.columns) == OHLCV_COLUMNS
    assert bars.index.name == "date"
    assert bars.index.tz is None
    assert bars.index.is_monotonic_increasing
    assert bars.index[0] == pd.Timestamp("2020-01-01")  # local calendar date preserved, not shifted by UTC conversion
    validate_ohlcv(bars)


def test_standardize_is_idempotent():
    once = standardize_ohlcv(make_ohlcv_bars(periods=50), 0.005)
    pd.testing.assert_frame_equal(once, standardize_ohlcv(once, 0.005))


def test_standardize_keeps_last_duplicate_and_drops_rows_with_gaps():
    raw = make_ohlcv_bars(periods=10)
    duplicate = raw.iloc[[3]].assign(Close=raw["Close"].iloc[3] * 1.001, High=raw["High"].iloc[3] * 1.01)
    raw = pd.concat([raw, duplicate])
    raw.iloc[5, raw.columns.get_loc("Volume")] = float("nan")

    bars = standardize_ohlcv(raw, 0.005)

    assert len(bars) == 9  # 11 rows - 1 duplicate date - 1 row with a gap
    assert bars.loc[raw.index[3], "close"] == pytest.approx(duplicate["Close"].iloc[0])


def test_standardize_clips_rounding_level_high_low_violation():
    raw = make_ohlcv_bars(periods=10)
    body_top = max(raw["Open"].iloc[2], raw["Close"].iloc[2])
    raw.iloc[2, raw.columns.get_loc("High")] = body_top * 0.999  # 0.1% below the candle body

    bars = standardize_ohlcv(raw, ohlc_repair_tolerance=0.005)

    assert bars["high"].iloc[2] == pytest.approx(body_top)


def test_standardize_rejects_large_high_low_violation():
    raw = make_ohlcv_bars(periods=10)
    raw.iloc[2, raw.columns.get_loc("Low")] = raw["Close"].iloc[2] * 1.05

    with pytest.raises(InvalidOHLCVError, match="outside open/close"):
        standardize_ohlcv(raw, ohlc_repair_tolerance=0.005)


def test_standardize_rejects_missing_columns():
    with pytest.raises(InvalidOHLCVError, match="volume"):
        standardize_ohlcv(make_ohlcv_bars(periods=5).drop(columns="Volume"), 0.005)


def test_standardize_rejects_non_positive_prices():
    raw = make_ohlcv_bars(periods=5)
    raw.iloc[1, raw.columns.get_loc("Close")] = 0.0
    with pytest.raises(InvalidOHLCVError, match="non-positive"):
        standardize_ohlcv(raw, 0.005)


# --- validate_ohlcv / ensure_minimum_history --------------------------------

def test_validate_rejects_unsorted_index():
    bars = standardize_ohlcv(make_ohlcv_bars(periods=5), 0.005).iloc[::-1]
    with pytest.raises(InvalidOHLCVError, match="increasing"):
        validate_ohlcv(bars)


def test_ensure_minimum_history():
    bars = standardize_ohlcv(make_ohlcv_bars(periods=520), 0.005)  # ~2 years of weekdays
    ensure_minimum_history(bars, min_history_years=1.5)
    with pytest.raises(InsufficientHistoryError):
        ensure_minimum_history(bars, min_history_years=3)


# --- load_ohlcv --------------------------------------------------------------

@pytest.fixture
def fake_yfinance(monkeypatch):
    """Replace the network fetch with synthetic bars and count how often it's called."""
    calls = []

    def fake_fetch(ticker, start, **options):
        calls.append((ticker, start, options))
        return make_ohlcv_bars(start="2019-06-03", periods=1500, tz="America/New_York")

    monkeypatch.setattr(loader, "fetch_ohlcv_from_yfinance", fake_fetch)
    return calls


def test_load_ohlcv_fetches_once_then_serves_from_cache(fake_yfinance, data_settings):
    first = load_ohlcv("aapl", data_settings, now=NOW)
    second = load_ohlcv("AAPL", data_settings, now=NOW + pd.Timedelta(hours=1))

    assert len(fake_yfinance) == 1
    pd.testing.assert_frame_equal(first, second)
    assert first.index[0] >= pd.Timestamp("2020-03-03")  # trimmed to history_years before NOW
    validate_ohlcv(first)


def test_load_ohlcv_refetches_when_cache_is_stale_or_refresh_forced(fake_yfinance, data_settings):
    load_ohlcv("AAPL", data_settings, now=NOW)
    load_ohlcv("AAPL", data_settings, now=NOW + pd.Timedelta(hours=13))
    load_ohlcv("AAPL", data_settings, now=NOW + pd.Timedelta(hours=14), force_refresh=True)
    assert len(fake_yfinance) == 3


def test_load_ohlcv_refetches_when_fetch_options_change(fake_yfinance, data_settings):
    load_ohlcv("AAPL", data_settings, now=NOW)
    data_settings["yfinance"]["auto_adjust"] = False
    load_ohlcv("AAPL", data_settings, now=NOW)
    assert len(fake_yfinance) == 2


def test_load_ohlcv_reads_user_csv(data_settings, tmp_path):
    csv_directory = tmp_path / "csv"
    csv_directory.mkdir()
    make_ohlcv_bars(start="2020-06-01", periods=1200).to_csv(csv_directory / "INFY.csv")
    data_settings["source"] = "csv"

    bars = load_ohlcv("INFY", data_settings, now=NOW)

    assert list(bars.columns) == OHLCV_COLUMNS
    assert bars.index[0] == pd.Timestamp("2020-06-01")


def test_load_ohlcv_rejects_short_history(monkeypatch, data_settings):
    monkeypatch.setattr(loader, "fetch_ohlcv_from_yfinance", lambda *a, **k: make_ohlcv_bars(start="2023-01-02", periods=500))
    with pytest.raises(InsufficientHistoryError):
        load_ohlcv("NEWIPO", data_settings, now=NOW)


def test_drop_holiday_placeholder_bars_removes_only_flat_zero_volume_bars():
    bars = standardize_ohlcv(make_ohlcv_bars(periods=6), 0.005)
    flat_price = bars["close"].iloc[1]
    bars.iloc[1] = [flat_price, flat_price, flat_price, flat_price, 0.0]  # holiday placeholder
    bars.iloc[3, bars.columns.get_loc("volume")] = 0.0  # index-style: real price move, zero volume

    cleaned = drop_holiday_placeholder_bars(bars)

    assert bars.index[1] not in cleaned.index
    assert bars.index[3] in cleaned.index
    assert len(cleaned) == 5
