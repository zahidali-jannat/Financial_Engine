import numpy as np
import pandas as pd
import pytest

from finance_engine.config.settings import load_settings
from finance_engine.fundamentals.fair_value import assess_fair_value
from finance_engine.fundamentals.growth import compound_growth, earnings_surprise_score, growth_score
from finance_engine.fundamentals.market_context import build_market_context
from finance_engine.fundamentals.valuation import ComponentScore, trailing_pe_history, valuation_vs_history, valuation_vs_peers

FUNDAMENTALS = load_settings()["fundamentals"]


def quarterly_earnings(eps_values, estimates=None, start="2016-01-15"):
    index = pd.date_range(start, periods=len(eps_values), freq="3MS", tz="UTC") + pd.Timedelta(hours=10)
    estimates = estimates if estimates is not None else [np.nan] * len(eps_values)
    surprise = [(r / e - 1) * 100 if e == e else np.nan for r, e in zip(eps_values, estimates)]
    return pd.DataFrame({"eps_estimate": estimates, "reported_eps": eps_values, "surprise_pct": surprise}, index=index)


def test_trailing_pe_is_rebuilt_point_in_time_from_four_quarters():
    days = pd.bdate_range("2016-01-01", "2020-12-31")
    close = pd.Series(100.0, index=days)
    earnings = quarterly_earnings([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0] + [2.0] * 12)
    pe = trailing_pe_history(close, earnings, "Asia/Kolkata")
    first_ttm_date = earnings.index[3].tz_convert("Asia/Kolkata").tz_localize(None).normalize()
    assert pe.index[0] >= first_ttm_date  # no P/E before four quarters were public
    assert pe.loc[first_ttm_date] == pytest.approx(100 / 4.0)
    assert pe.iloc[-1] == pytest.approx(100 / 8.0)


def test_valuation_vs_history_scores_cheap_positive_and_expensive_negative():
    days = pd.bdate_range("2015-01-01", periods=1000)
    rising = pd.Series(np.linspace(10, 40, 1000), index=days)
    assert valuation_vs_history(rising).score == pytest.approx(-1.0, abs=0.01)  # at its most expensive
    assert valuation_vs_history(rising[::-1].set_axis(days)).score == pytest.approx(1.0, abs=0.01)
    assert valuation_vs_history(rising.iloc[:100]).score is None


def test_peer_valuation_excludes_implausible_multiples_and_says_so():
    own = {"trailingPE": 15.0, "priceToBook": 5.0, "enterpriseToEbitda": 10.0}
    peers = {
        "A.NS": {"trailingPE": 20.0, "priceToBook": 5.0, "enterpriseToEbitda": 900.0},  # currency-mismatch style error
        "B.NS": {"trailingPE": 20.0, "priceToBook": 5.0, "enterpriseToEbitda": 12.0},
        "C.NS": {"trailingPE": 20.0, "priceToBook": 5.0, "enterpriseToEbitda": 12.0},
    }
    result = valuation_vs_peers(own, peers, FUNDAMENTALS["peer_log_ratio_scale"], FUNDAMENTALS["plausible_multiple_max"])
    assert result.score > 0  # cheaper than peers on P/E and EV/EBITDA, level on P/B
    assert "A.NS EV/EBITDA 900" in result.detail
    assert valuation_vs_peers(own, {}, 0.5, FUNDAMENTALS["plausible_multiple_max"]).score is None


def test_growth_and_surprise_scores():
    years = pd.to_datetime(["2022-03-31", "2023-03-31", "2024-03-31", "2025-03-31"])
    annual = pd.DataFrame([[100.0, 110.0, 121.0, 133.1], [10.0, 11.0, 12.1, 13.31]], index=["Total Revenue", "Net Income"], columns=years)
    assert compound_growth(annual.loc["Total Revenue"]) == pytest.approx(0.10, abs=0.002)
    assert growth_score(annual, pd.DataFrame(), growth_scale=0.15).score == pytest.approx(np.tanh(0.10 / 0.15), abs=0.01)

    beats = quarterly_earnings([11.0] * 8, estimates=[10.0] * 8)
    misses = quarterly_earnings([9.0] * 8, estimates=[10.0] * 8)
    assert earnings_surprise_score(beats, 5.0).score > 0.9
    assert earnings_surprise_score(misses, 5.0).score < -0.9


def test_fair_value_combines_available_components_and_scales_the_tilt_by_confidence():
    components = {
        "valuation_vs_history": ComponentScore(0.8, ""),
        "valuation_vs_peers": ComponentScore(0.6, ""),
        "growth": ComponentScore(None, "missing"),
        "earnings_surprise": ComponentScore(0.7, ""),
    }
    result = assess_fair_value(components, FUNDAMENTALS)
    assert result.label == "undervalued"
    assert result.bias == pytest.approx((0.3 * 0.8 + 0.3 * 0.6 + 0.2 * 0.7) / 0.8)
    assert 0 < result.annual_drift_tilt < FUNDAMENTALS["max_annual_tilt"] * result.bias

    conflicting = assess_fair_value({"valuation_vs_history": ComponentScore(1.0, ""), "valuation_vs_peers": ComponentScore(-1.0, "")}, FUNDAMENTALS)
    assert conflicting.label == "fairly valued" and conflicting.confidence == 0.0


def test_market_context_beta_and_relative_strength():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2023-01-01", periods=400)
    index_returns = rng.normal(0, 0.01, 400)
    stock_returns = 1.5 * index_returns + rng.normal(0, 0.002, 400)
    index_close = pd.Series(100 * np.exp(np.cumsum(index_returns)), index=days)
    stock_close = pd.Series(100 * np.exp(np.cumsum(stock_returns)), index=days)
    context = build_market_context(stock_close, index_close, None, "^NSEI", None)
    assert context.beta == pytest.approx(1.5, abs=0.05)
    expected = stock_close.iloc[-1] / stock_close.iloc[-64] - index_close.iloc[-1] / index_close.iloc[-64]
    assert context.relative_strength_vs_benchmark == pytest.approx(expected)
