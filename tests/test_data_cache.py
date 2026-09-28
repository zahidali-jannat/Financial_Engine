import pandas as pd

from finance_engine.data.cache import cache_file_paths, is_cache_usable, read_cached_ohlcv, write_cached_ohlcv
from finance_engine.data.loader import standardize_ohlcv
from tests.conftest import make_ohlcv_bars

NOW = pd.Timestamp("2025-03-03 12:00", tz="UTC")
OPTIONS = {"source": "yfinance", "interval": "1d", "auto_adjust": True, "drop_todays_bar": True}
START = pd.Timestamp("2020-03-03")


def metadata(fetched_hours_ago=1.0, requested_start=START, options=OPTIONS):
    return {
        "fetched_at_utc": (NOW - pd.Timedelta(hours=fetched_hours_ago)).isoformat(),
        "requested_start": requested_start.isoformat(),
        "fetch_options": options,
    }


def usable(meta, requested_start=START, options=OPTIONS):
    return is_cache_usable(meta, requested_start=requested_start, fetch_options=options, max_age_hours=12, now=NOW)


def test_fresh_matching_cache_is_usable():
    assert usable(metadata())


def test_stale_cache_is_not_usable():
    assert not usable(metadata(fetched_hours_ago=13))


def test_cache_not_usable_when_it_starts_after_the_requested_start():
    assert not usable(metadata(requested_start=START + pd.Timedelta(days=30)))
    assert usable(metadata(requested_start=START - pd.Timedelta(days=30)))  # longer cached history is fine


def test_cache_not_usable_when_fetch_options_differ():
    assert not usable(metadata(), options={**OPTIONS, "auto_adjust": False})


def test_write_then_read_round_trips_bars(tmp_path):
    bars = standardize_ohlcv(make_ohlcv_bars(periods=100), 0.005)
    write_cached_ohlcv(bars, tmp_path, "AAPL", requested_start=START, fetch_options=OPTIONS, now=NOW)

    cached = read_cached_ohlcv(tmp_path, "AAPL", requested_start=START, fetch_options=OPTIONS, max_age_hours=12, now=NOW)

    pd.testing.assert_frame_equal(standardize_ohlcv(cached, 0.005), bars)


def test_read_returns_none_when_nothing_cached(tmp_path):
    assert read_cached_ohlcv(tmp_path, "AAPL", requested_start=START, fetch_options=OPTIONS, max_age_hours=12, now=NOW) is None


def test_cache_file_names_are_filesystem_safe(tmp_path):
    bars_path, metadata_path = cache_file_paths(tmp_path, "BRK/B", "1d")
    assert bars_path.name == "BRK_B_1d.csv"
    assert metadata_path.name == "BRK_B_1d.meta.json"
