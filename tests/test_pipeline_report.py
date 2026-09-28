"""Offline end-to-end run: synthetic prices and fundamentals through every layer to the written report."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from finance_engine.config.settings import load_settings
from finance_engine.data.fundamentals import FundamentalData
from finance_engine.data.loader import standardize_ohlcv
from finance_engine.output.report import build_verdicts, render_report, write_report
from finance_engine.pipeline import analyze
from tests.conftest import make_ohlcv_bars


@pytest.fixture(scope="module")
def analysis_and_settings(tmp_path_factory):
    settings = copy.deepcopy(load_settings())
    settings["forecast"]["monte_carlo_paths"] = 2000
    settings["walk_forward"].update({"min_training_days": 500, "refit_every_days": 63})
    settings["output"]["directory"] = str(tmp_path_factory.mktemp("reports"))
    bars = standardize_ohlcv(make_ohlcv_bars(periods=900, seed=41), 0.005)
    announcements = pd.date_range(bars.index[20], periods=12, freq="63B", tz="Asia/Kolkata") + pd.Timedelta(hours=17)
    earnings = pd.DataFrame(
        {"eps_estimate": 10.0, "reported_eps": np.linspace(10, 12, 12), "surprise_pct": 2.0},
        index=announcements.tz_convert("UTC").rename("announced_at"),
    )
    fundamentals = FundamentalData({"trailingPE": 20.0, "currency": "INR", "longName": "Test Ltd", "sector": "Technology"}, earnings, pd.DataFrame(), pd.DataFrame())
    benchmark = bars["close"] * np.exp(np.random.default_rng(0).normal(0, 0.005, len(bars)).cumsum())
    return analyze("TEST.NS", bars, fundamentals, {}, benchmark, None, settings), settings


def test_report_names_forecast_trust_and_drivers(analysis_and_settings):
    analysis, settings = analysis_and_settings
    text = render_report(analysis, settings)
    for heading in ("WHAT THE ENGINE EXPECTS", "HOW MUCH TO TRUST THIS", "WHY — WHAT SHAPED THIS FORECAST", "FUNDAMENTALS", "MODEL CHECKS", "WHERE IT DOES WORST"):
        assert heading in text
    assert "68% probability the price ends between ₹" in text
    verdicts = build_verdicts(analysis, settings)
    assert verdicts["forecasts_tested"] == len(analysis.backtest.records)


def test_live_forecast_is_a_sensible_distribution(analysis_and_settings):
    forecast = analysis_and_settings[0].forecast
    low68, high68 = forecast.price_interval(0.68)
    low95, high95 = forecast.price_interval(0.95)
    assert low95 < low68 < forecast.last_close * 1.05 and forecast.last_close * 0.95 < high68 < high95
    assert 0.2 < forecast.probability_of_return_above(0.0) < 0.8
    assert forecast.forecast_days[0] > forecast.as_of


def test_write_report_creates_every_artifact(analysis_and_settings):
    analysis, settings = analysis_and_settings
    paths = write_report(analysis, settings)
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0
    summary = json.loads(paths["summary"].read_text())
    assert summary["ticker"] == "TEST.NS" and "verdicts" in summary
