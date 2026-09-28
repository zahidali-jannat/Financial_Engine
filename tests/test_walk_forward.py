import copy

import numpy as np
import pandas as pd
import pytest

from finance_engine.backtest.walk_forward import run_walk_forward
from finance_engine.config.settings import load_settings
from finance_engine.data.loader import standardize_ohlcv
from tests.conftest import make_ohlcv_bars

FORECAST_FIELDS = ["drift_mean", "drift_std", "engine_volatility_model", "turbulent_probability"]


@pytest.fixture(scope="module")
def small_settings():
    settings = copy.deepcopy(load_settings())
    settings["forecast"]["monte_carlo_paths"] = 1000
    settings["walk_forward"].update({"min_training_days": 400, "refit_every_days": 42, "min_scored_forecasts_for_model_selection": 5})
    settings["models"]["hmm"]["random_restarts"] = 2
    return settings


@pytest.fixture(scope="module")
def bars():
    return standardize_ohlcv(make_ohlcv_bars(periods=640, seed=31), 0.005)


def test_walk_forward_produces_scored_out_of_sample_forecasts(bars, small_settings):
    result = run_walk_forward(bars, pd.DatetimeIndex([]), small_settings)
    records = result.records
    assert len(records) == len(range(400, 640 - 5, 5))
    assert records.index[0] == bars.index[400]
    assert {"gbm", "garch_t", "merton_jump", "regime_switching", "engine"} <= set(result.model_names)
    assert records["engine_crps"].gt(0).all()
    assert records["engine_pit"].between(0, 1).all()
    np.testing.assert_allclose(records["realized"].iloc[0], np.log(bars["close"].iloc[405] / bars["close"].iloc[400]))


def test_forecasts_never_depend_on_future_prices(bars, small_settings):
    """End-to-end lookahead guard: scrambling every bar from day 520 on must not change any forecast made before day 520."""
    cutoff = 520
    scrambled = bars.copy()
    factors = np.random.default_rng(99).uniform(0.5, 1.5, len(bars) - cutoff)
    for column in ("open", "high", "low", "close"):
        scrambled.iloc[cutoff:, scrambled.columns.get_loc(column)] *= factors
    scrambled.iloc[cutoff:, scrambled.columns.get_loc("volume")] *= factors[::-1]

    original = run_walk_forward(bars, pd.DatetimeIndex([]), small_settings).records
    altered = run_walk_forward(scrambled, pd.DatetimeIndex([]), small_settings).records

    before_cutoff = original["position"] < cutoff
    probability_columns = [c for c in original.columns if c.endswith("_p_up")]
    pd.testing.assert_frame_equal(original.loc[before_cutoff, FORECAST_FIELDS + probability_columns], altered.loc[before_cutoff, FORECAST_FIELDS + probability_columns])
    assert not np.allclose(original.loc[~before_cutoff, "drift_mean"], altered.loc[~before_cutoff, "drift_mean"])
