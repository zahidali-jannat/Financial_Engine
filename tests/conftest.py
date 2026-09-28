"""Shared fixtures: synthetic OHLCV bars with known properties, so no test touches the network."""

import numpy as np
import pandas as pd
import pytest


def make_ohlcv_bars(start: str = "2020-01-01", periods: int = 1300, seed: int = 7, tz: str | None = None) -> pd.DataFrame:
    """Random-walk daily bars in yfinance's shape (capitalised columns, optional exchange tz)."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=periods, tz=tz, name="Date")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, periods)))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.002, periods))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, periods))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, periods))
    volume = rng.integers(1_000_000, 5_000_000, periods).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=index)


@pytest.fixture
def data_settings(tmp_path) -> dict:
    """The data section of settings.yaml, pointed at a temporary cache/CSV directory."""
    return {
        "source": "yfinance",
        "history_years": 5,
        "min_history_years": 3,
        "ohlc_repair_tolerance": 0.005,
        "drop_holiday_placeholder_bars": True,
        "cache": {"directory": str(tmp_path / "cache"), "max_age_hours": 12},
        "yfinance": {"interval": "1d", "auto_adjust": True, "drop_todays_bar": True},
        "csv": {"directory": str(tmp_path / "csv")},
    }
