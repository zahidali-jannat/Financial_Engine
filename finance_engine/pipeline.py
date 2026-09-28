"""Run the whole engine for one ticker: data -> backtest -> fundamentals -> live forecast.

    analyze(...)       — pure: takes already-loaded data, returns an Analysis (testable offline)
    run_analysis(...)  — loads everything (prices, fundamentals, peers, indices), then analyze()
"""

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from finance_engine.backtest.walk_forward import WalkForwardResult, prepare_series, run_walk_forward
from finance_engine.config.settings import resolve_project_path
from finance_engine.data.fundamentals import FundamentalData, load_fundamentals
from finance_engine.data.loader import load_ohlcv
from finance_engine.evaluation.calibration import interval_calibration, pit_histogram, reliability_table
from finance_engine.evaluation.failure_modes import segment_performance, worst_misses
from finance_engine.evaluation.model_comparison import base_rate_brier, compare_models
from finance_engine.evaluation.strategy_check import evaluate_long_only_rule
from finance_engine.fundamentals.fair_value import FairValueAssessment, assess_fair_value
from finance_engine.fundamentals.growth import earnings_surprise_score, growth_score
from finance_engine.fundamentals.market_context import MarketContext, build_market_context
from finance_engine.fundamentals.valuation import trailing_pe_history, valuation_vs_history, valuation_vs_peers
from finance_engine.models.gbm import gbm_assumption_tests
from finance_engine.synthesis.live_forecast import LiveForecast, best_volatility_model, make_live_forecast

logger = logging.getLogger(__name__)
# With fewer forecasts per bin, the chance range of each bin's hit rate is too wide to read (±10 points at n≈40).
MIN_FORECASTS_PER_RELIABILITY_BIN = 75


@dataclass(frozen=True)
class Analysis:
    ticker: str
    bars: pd.DataFrame
    backtest: WalkForwardResult
    model_comparison: pd.DataFrame
    base_rate_brier: float
    engine_interval_calibration: pd.DataFrame
    baseline_interval_calibration: pd.DataFrame
    engine_reliability: pd.DataFrame
    engine_pit_histogram: pd.Series
    segments: pd.DataFrame
    worst_misses: pd.DataFrame
    strategy: dict
    gbm_assumptions: dict
    fair_value: FairValueAssessment
    market_context: MarketContext | None
    fundamentals_info: dict
    rebuilt_pe: float | None
    forecast: LiveForecast


def analyze(
    ticker: str,
    bars: pd.DataFrame,
    fundamentals: FundamentalData,
    peer_infos: dict[str, dict],
    benchmark_close: pd.Series | None,
    sector_close: pd.Series | None,
    settings: dict,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> Analysis:
    announcements = fundamentals.earnings.index
    backtest = run_walk_forward(bars, announcements, settings, ticker=ticker, progress=progress)
    records = backtest.records
    comparison = compare_models(records, backtest.model_names, backtest.interval_levels)

    fundamental_settings, market = settings["fundamentals"], settings["market"]
    pe_history = trailing_pe_history(bars["close"], fundamentals.earnings, market["exchange_timezone"])
    components = {
        "valuation_vs_history": valuation_vs_history(pe_history),
        "valuation_vs_peers": valuation_vs_peers(
            fundamentals.info, peer_infos, fundamental_settings["peer_log_ratio_scale"], fundamental_settings["plausible_multiple_max"]
        ),
        "growth": growth_score(fundamentals.annual_income, fundamentals.quarterly_income, fundamental_settings["growth_scale"]),
        "earnings_surprise": earnings_surprise_score(fundamentals.earnings, fundamental_settings["surprise_scale_pct"]),
    }
    fair_value = assess_fair_value(components, fundamental_settings)
    context = (
        build_market_context(bars["close"], benchmark_close, sector_close, market["benchmark_index"], _sector_index(fundamentals.info, settings))
        if benchmark_close is not None
        else None
    )

    prepared = prepare_series(bars, announcements, settings)
    volatility_model = best_volatility_model(comparison, set(comparison.index), settings["walk_forward"]["default_volatility_model"])
    forecast = make_live_forecast(
        bars, prepared, announcements, settings, volatility_model=volatility_model, fundamental_tilt=fair_value.annual_drift_tilt, ticker=ticker
    )

    horizon = settings["forecast"]["horizon_days"]
    return Analysis(
        ticker=ticker,
        bars=bars,
        backtest=backtest,
        model_comparison=comparison,
        base_rate_brier=base_rate_brier(records),
        engine_interval_calibration=interval_calibration(records["engine_pit"]),
        baseline_interval_calibration=interval_calibration(records["gbm_pit"]),
        engine_reliability=reliability_table(records["engine_p_up"], records["realized"], min(settings["evaluation"]["reliability_bins"], max(2, len(records) // MIN_FORECASTS_PER_RELIABILITY_BIN))),
        engine_pit_histogram=pit_histogram(records["engine_pit"], settings["evaluation"]["pit_bins"]),
        segments=segment_performance(records),
        worst_misses=worst_misses(records),
        strategy=evaluate_long_only_rule(
            records,
            probability_column="engine_p_up",
            enter_probability=settings["strategy_check"]["enter_probability"],
            horizon=horizon,
            trading_days_per_year=market["trading_days_per_year"],
            risk_free_rate_annual=market["risk_free_rate_annual"],
            round_trip_cost=market["round_trip_cost"],
        ),
        gbm_assumptions=gbm_assumption_tests(np.log(bars["close"].to_numpy())),
        fair_value=fair_value,
        market_context=context,
        fundamentals_info=fundamentals.info,
        rebuilt_pe=float(pe_history.iloc[-1]) if len(pe_history) else None,
        forecast=forecast,
    )


def _sector_index(info: dict, settings: dict) -> str | None:
    return settings["market"]["sector_indices"].get(info.get("sector"))


def run_analysis(ticker: str, settings: dict, *, refresh: bool = False, progress: Callable[[int, int], None] | None = None) -> Analysis:
    ticker = ticker.upper()
    data_settings = settings["data"]
    bars = load_ohlcv(ticker, data_settings, force_refresh=refresh)

    fundamentals_cache = resolve_project_path(data_settings["cache"]["directory"]).parent / "fundamentals"
    max_age = 0 if refresh else settings["fundamentals"]["cache_max_age_hours"]
    fundamentals = load_fundamentals(ticker, fundamentals_cache, max_age_hours=max_age)
    peer_infos = {
        peer: load_fundamentals(peer, fundamentals_cache, max_age_hours=settings["fundamentals"]["cache_max_age_hours"]).info
        for peer in settings["fundamentals"]["peers"].get(ticker, [])
    }
    benchmark_close = _optional_close(settings["market"]["benchmark_index"], data_settings)
    sector = _sector_index(fundamentals.info, settings)
    sector_close = _optional_close(sector, data_settings) if sector else None
    return analyze(ticker, bars, fundamentals, peer_infos, benchmark_close, sector_close, settings, progress=progress)


def _optional_close(index_ticker: str, data_settings: dict) -> pd.Series | None:
    """Index context is a nice-to-have: a failed download shouldn't stop the forecast."""
    try:
        return load_ohlcv(index_ticker, data_settings)["close"]
    except Exception as error:
        logger.warning("Could not load %s for market context: %s", index_ticker, error)
        return None
