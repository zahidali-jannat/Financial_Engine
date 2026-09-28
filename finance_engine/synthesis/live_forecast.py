"""Today's forecast: fit everything on all available history and simulate the next h trading days.

Uses exactly the same machinery as each walk-forward step (so what was validated
is what runs), with two differences only a live forecast has:
    • the volatility model is the one with the best out-of-sample CRPS over the
      whole backtest (the backtest is, by construction, all in the past now);
    • the fundamental fair-value tilt is added to the drift prior.

Output contract (LiveForecast): samples of the h-day log return plus summaries —
price intervals, probabilities of moves, the drift posterior with each evidence
contribution, the regime reading and the models' fitted parameters.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from finance_engine.aggregation.redundancy import RedundancyDecision, prune_redundant_indicators
from finance_engine.backtest.walk_forward import PreparedSeries, evidence_and_drift, training_start
from finance_engine.models.earnings_jumps import earnings_reaction_days, horizon_shock_multipliers
from finance_engine.synthesis.bayesian_drift import DriftPosterior
from finance_engine.synthesis.model_suite import VOLATILITY_MODEL_NAMES, ModelSuite, fit_model_suite, simulate_engine


@dataclass(frozen=True)
class LiveForecast:
    as_of: pd.Timestamp
    last_close: float
    horizon: int
    forecast_days: pd.DatetimeIndex
    samples: np.ndarray  # h-day log returns
    volatility_model: str
    posterior: DriftPosterior
    suite: ModelSuite
    redundancy: RedundancyDecision
    evidence_today: dict
    trending: float
    upcoming_reaction_days: list
    fundamental_tilt: float

    def price_interval(self, level: float) -> tuple[float, float]:
        tail = (1 - level) / 2
        low, high = np.quantile(self.samples, [tail, 1 - tail])
        return self.last_close * float(np.exp(low)), self.last_close * float(np.exp(high))

    def probability_of_return_above(self, threshold: float) -> float:
        """P(simple return > threshold)."""
        return float(np.mean(np.expm1(self.samples) > threshold))

    @property
    def median_price(self) -> float:
        return self.last_close * float(np.exp(np.median(self.samples)))


def best_volatility_model(model_comparison: pd.DataFrame, available: set, default: str) -> str:
    eligible = model_comparison.loc[[name for name in VOLATILITY_MODEL_NAMES if name in model_comparison.index and name in available]]
    return str(eligible["mean_crps"].idxmin()) if len(eligible) else default


def next_trading_days(last_day: pd.Timestamp, count: int) -> pd.DatetimeIndex:
    """Next `count` weekdays. NSE holidays aren't in this calendar, so a holiday inside the
    window makes the forecast cover one extra calendar day — a small, conservative error."""
    return pd.bdate_range(last_day + pd.Timedelta(days=1), periods=count)


def make_live_forecast(
    bars: pd.DataFrame,
    prepared: PreparedSeries,
    all_announcement_times: pd.DatetimeIndex,
    settings: dict,
    *,
    volatility_model: str,
    fundamental_tilt: float,
    ticker: str,
) -> LiveForecast:
    forecast, market = settings["forecast"], settings["market"]
    horizon = forecast["horizon_days"]
    rng = np.random.default_rng(forecast["random_seed"])
    position = len(bars) - 1
    start = training_start(position, settings["walk_forward"]["max_training_days"])
    window_prices = prepared.log_prices[start:]

    suite = fit_model_suite(window_prices, prepared.log_returns.iloc[start:], prepared.reaction_days, settings, rng)
    priority = settings["redundancy"]["priority"]
    redundancy = prune_redundant_indicators(
        prepared.normalized_indicators.iloc[start:][priority].dropna(), priority, settings["redundancy"]["max_abs_correlation"]
    )
    evidence, posterior = evidence_and_drift(prepared, suite, redundancy.kept, start, position, settings, fundamental_tilt)

    forecast_days = next_trading_days(bars.index[-1], horizon)
    future_reaction_days = earnings_reaction_days(
        all_announcement_times[all_announcement_times > bars.index[-1].tz_localize(market["exchange_timezone"])],
        forecast_days,
        exchange_timezone=market["exchange_timezone"],
        market_close_time=market["market_close_time"],
    )
    multipliers = horizon_shock_multipliers(forecast_days, future_reaction_days, suite.earnings_variance_multiplier)
    model_name = volatility_model if volatility_model in suite.candidate_models() else "garch_t"
    samples = simulate_engine(
        suite, model_name, posterior.mean, posterior.std,
        horizon=horizon, n_paths=forecast["monte_carlo_paths"], rng=rng,
        earnings_multipliers=multipliers, price_band=market["price_bands"].get(ticker),
    )
    return LiveForecast(
        as_of=bars.index[-1],
        last_close=float(bars["close"].iloc[-1]),
        horizon=horizon,
        forecast_days=forecast_days,
        samples=samples,
        volatility_model=model_name,
        posterior=posterior,
        suite=suite,
        redundancy=redundancy,
        evidence_today=evidence.iloc[-1].to_dict(),
        trending=float(prepared.trending.iloc[-1]),
        upcoming_reaction_days=list(future_reaction_days),
        fundamental_tilt=fundamental_tilt,
    )
