"""Walk-forward backtest: every forecast is made with data available at the time, then scored on what happened.

At each forecast date t (every `forecast_every_days`, starting after `min_training_days`):
    1. Refit all models on bars up to t if a refit is due; otherwise keep the
       parameters and only filter their states forward to t.
    2. Re-decide redundant indicators on bars up to t.
    3. Estimate the Bayesian drift from outcomes that had finished by t.
    4. Pick the engine's volatility model by CRPS of forecasts already scored by t.
    5. Simulate every candidate model and the engine; score them against the
       realized h-day return (known only now, after the fact).

Lookahead guards (each also covered by tests):
    • models are fitted/filtered on log_prices[start : t+1] only;
    • indicators are causal (tested by truncation); pruning uses rows ≤ t;
    • forward returns for the last h rows before t are masked — not yet realized;
    • model selection uses only forecasts whose outcome window ended by t;
    • future bars are used for exactly two things: the realized outcome being scored,
      and the trading calendar (which dates are sessions; published in advance).
Known limitation: earnings announcement dates are treated as known at t. NSE
listing rules only require a few working days' notice, so an announcement
falling late in a 5-day window may not always have been public yet.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from finance_engine.aggregation.redundancy import prune_redundant_indicators
from finance_engine.aggregation.regime import is_trending
from finance_engine.evaluation.scoring import score_forecast
from finance_engine.indicators.registry import build_indicators, compute_normalized_indicators, indicator_categories
from finance_engine.models.earnings_jumps import earnings_reaction_days, horizon_shock_multipliers
from finance_engine.synthesis.bayesian_drift import drift_prior, estimate_drift_posterior
from finance_engine.synthesis.evidence import build_evidence, forward_log_returns
from finance_engine.synthesis.model_suite import (
    VOLATILITY_MODEL_NAMES,
    ModelSuite,
    fit_model_suite,
    simulate_candidates,
    simulate_engine,
)

ENGINE = "engine"


@dataclass
class WalkForwardResult:
    records: pd.DataFrame  # one row per forecast date
    model_names: list[str]  # candidate models + "engine"
    interval_levels: list[float]
    refits: list[dict] = field(default_factory=list)
    final_suite: ModelSuite | None = None


@dataclass(frozen=True)
class PreparedSeries:
    """Everything derived once from the full bar history. Each series is causal row by row."""

    dates: pd.DatetimeIndex
    log_prices: np.ndarray
    log_returns: pd.Series
    normalized_indicators: pd.DataFrame
    indicator_categories: dict
    trending: pd.Series
    reaction_days: pd.DatetimeIndex
    forward_returns: pd.Series  # uses the future by definition — read only through masking


def prepare_series(bars: pd.DataFrame, announcement_times: pd.DatetimeIndex, settings: dict) -> PreparedSeries:
    indicators = build_indicators(settings["indicators"], settings["normalization"])
    normalized = compute_normalized_indicators(bars, indicators)
    adx = next(indicator for indicator in indicators if indicator.name == "adx").compute(bars)
    log_prices = np.log(bars["close"].to_numpy())
    market = settings["market"]
    return PreparedSeries(
        dates=bars.index,
        log_prices=log_prices,
        log_returns=pd.Series(np.diff(log_prices), index=bars.index[1:]),
        normalized_indicators=normalized,
        indicator_categories=indicator_categories(indicators),
        trending=is_trending(adx, settings["regimes"]["trend_adx_threshold"]),
        reaction_days=earnings_reaction_days(
            announcement_times, bars.index, exchange_timezone=market["exchange_timezone"], market_close_time=market["market_close_time"]
        ),
        forward_returns=forward_log_returns(log_prices, settings["forecast"]["horizon_days"], bars.index),
    )


def select_volatility_model(records: list[dict], position: int, horizon: int, available: set, min_scored: int, default: str) -> str:
    """Lowest mean CRPS over forecasts whose outcomes were known by `position`; `default` until enough exist."""
    scored = [record for record in records if record["position"] + horizon <= position]
    best_name, best_crps = None, np.inf
    for name in VOLATILITY_MODEL_NAMES:
        if name not in available:
            continue
        crps = [record[f"{name}_crps"] for record in scored if f"{name}_crps" in record]
        if len(crps) >= min_scored and np.mean(crps) < best_crps:
            best_name, best_crps = name, float(np.mean(crps))
    if best_name is not None:
        return best_name
    return default if default in available else "gbm"


def training_start(position: int, max_training_days: int | None) -> int:
    return 0 if max_training_days is None else max(0, position + 1 - max_training_days)


def evidence_and_drift(prepared: PreparedSeries, suite: ModelSuite, kept: list[str], start: int, position: int, settings: dict, fundamental_tilt: float = 0.0):
    """Evidence matrix over [start, position] and the drift posterior at `position` (no outcome after `position` is readable)."""
    horizon = settings["forecast"]["horizon_days"]
    window_prices = prepared.log_prices[start : position + 1]
    evidence = build_evidence(
        prepared.normalized_indicators.iloc[start : position + 1],
        kept,
        prepared.indicator_categories,
        suite.kalman.slope_z_scores(window_prices),
        prepared.trending.iloc[start : position + 1],
    )
    known_outcomes = prepared.forward_returns.iloc[start : position + 1].copy()
    known_outcomes.iloc[-horizon:] = np.nan  # these windows end after `position`: not yet realized
    prior_mean, prior_std = drift_prior(
        settings["bayes"], horizon=horizon, trading_days_per_year=settings["market"]["trading_days_per_year"], fundamental_annual_tilt=fundamental_tilt
    )
    posterior = estimate_drift_posterior(
        evidence,
        known_outcomes,
        position - start,
        horizon=horizon,
        prior_mean=prior_mean,
        prior_std=prior_std,
        signal_effect_prior_sd=settings["bayes"]["signal_effect_prior_sd"],
    )
    return evidence, posterior


def run_walk_forward(
    bars: pd.DataFrame,
    announcement_times: pd.DatetimeIndex,
    settings: dict,
    *,
    ticker: str = "",
    progress: Callable[[int, int], None] | None = None,
) -> WalkForwardResult:
    forecast, walk, market = settings["forecast"], settings["walk_forward"], settings["market"]
    horizon, levels = forecast["horizon_days"], forecast["interval_levels"]
    price_band = market["price_bands"].get(ticker)
    rng = np.random.default_rng(forecast["random_seed"])
    prepared = prepare_series(bars, announcement_times, settings)
    priority = settings["redundancy"]["priority"]

    origins = list(range(walk["min_training_days"], len(bars) - horizon, walk["forecast_every_days"]))
    suite, last_refit, kept = None, None, []
    records, refits, model_names = [], [], set()

    for step, position in enumerate(origins):
        start = training_start(position, walk["max_training_days"])
        window_prices = prepared.log_prices[start : position + 1]

        if suite is None or position - last_refit >= walk["refit_every_days"]:
            suite = fit_model_suite(
                window_prices,
                prepared.log_returns.iloc[start:position],
                prepared.reaction_days[prepared.reaction_days <= prepared.dates[position]],
                settings,
                rng,
                previous=suite,
            )
            training_indicators = prepared.normalized_indicators.iloc[start : position + 1][priority].dropna()
            redundancy = prune_redundant_indicators(training_indicators, priority, settings["redundancy"]["max_abs_correlation"])
            kept, last_refit = redundancy.kept, position
            refits.append(
                {
                    "date": prepared.dates[position],
                    "kept_indicators": kept,
                    "dropped_indicators": redundancy.dropped,
                    "adf_statistic": suite.adf.statistic,
                    "adf_rejects_unit_root": suite.adf.is_stationary,
                    "earnings_variance_multiplier": suite.earnings_variance_multiplier,
                    "garch_persistence": suite.garch.persistence,
                    "garch_tail_dof": suite.garch.degrees_of_freedom,
                }
            )
        else:
            suite = suite.updated_to(window_prices)

        _, posterior = evidence_and_drift(prepared, suite, kept, start, position, settings)
        forecast_days = prepared.dates[position + 1 : position + horizon + 1]
        multipliers = horizon_shock_multipliers(forecast_days, prepared.reaction_days, suite.earnings_variance_multiplier)

        samples = simulate_candidates(suite, horizon=horizon, n_paths=forecast["monte_carlo_paths"], rng=rng, earnings_multipliers=multipliers, price_band=price_band)
        engine_model = select_volatility_model(
            records, position, horizon, set(samples), walk["min_scored_forecasts_for_model_selection"], walk["default_volatility_model"]
        )
        samples[ENGINE] = simulate_engine(
            suite, engine_model, posterior.mean, posterior.std,
            horizon=horizon, n_paths=forecast["monte_carlo_paths"], rng=rng, earnings_multipliers=multipliers, price_band=price_band,
        )

        realized = float(prepared.log_prices[position + horizon] - prepared.log_prices[position])
        record = {
            "date": prepared.dates[position],
            "position": position,
            "realized": realized,
            "engine_volatility_model": engine_model,
            "drift_mean": posterior.mean,
            "drift_std": posterior.std,
            "turbulent_probability": suite.hmm.turbulent_probability,
            "trending": prepared.trending.iloc[position],
            "earnings_in_window": bool(forecast_days.isin(prepared.reaction_days).any()),
            "ou_active": suite.ornstein_uhlenbeck is not None,
        }
        for name, simulated in samples.items():
            model_names.add(name)
            score = score_forecast(simulated, realized, levels)
            record.update({f"{name}_crps": score.crps, f"{name}_pit": score.pit, f"{name}_p_up": score.probability_up, f"{name}_brier": score.brier})
            record.update({f"{name}_hit_{level}": hit for level, hit in score.interval_hits.items()})
        records.append(record)
        if progress:
            progress(step + 1, len(origins))

    ordered_names = [name for name in (*VOLATILITY_MODEL_NAMES, "ornstein_uhlenbeck", ENGINE) if name in model_names]
    return WalkForwardResult(pd.DataFrame(records).set_index("date"), ordered_names, list(levels), refits, suite)
