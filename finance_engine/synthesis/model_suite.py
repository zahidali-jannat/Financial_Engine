"""Fit, update and simulate the full set of Layer 3 models at one point in time.

Contract:
    fit_model_suite(log_prices, ...)   — estimate every model on log_prices (training data only)
    suite.updated_to(log_prices)       — same parameters, states filtered forward to the new last day
    suite.candidate_models()           — {name: model} competing to describe the return distribution
    simulate_candidates(...)           — {name: samples of h-day log return}
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from finance_engine.models.base import FittedReturnModel, simulate_cumulative_log_returns
from finance_engine.models.earnings_jumps import estimate_earnings_variance_multiplier
from finance_engine.models.garch import FittedGarch, fit_garch
from finance_engine.models.gbm import FittedGbm, fit_gbm
from finance_engine.models.hmm import FittedHmm, fit_hmm
from finance_engine.models.jump_diffusion import FittedMerton, fit_merton
from finance_engine.models.kalman import KalmanTrendModel, fit_kalman_trend
from finance_engine.models.mean_reversion import FittedOrnsteinUhlenbeck, detect_mean_reversion, fit_ornstein_uhlenbeck
from finance_engine.stats.hypothesis_tests import AdfResult

EARNINGS_VARIANT = "garch_t_earnings"
# Models whose shocks can carry the Bayesian drift. OU is excluded: its drift is part
# of its dynamics (pull toward θ), so replacing it would no longer be an OU forecast.
VOLATILITY_MODEL_NAMES = ("gbm", "garch_t", EARNINGS_VARIANT, "merton_jump", "regime_switching")


@dataclass(frozen=True)
class ModelSuite:
    gbm: FittedGbm
    garch: FittedGarch
    merton: FittedMerton
    hmm: FittedHmm
    kalman: KalmanTrendModel
    adf: AdfResult
    ornstein_uhlenbeck: FittedOrnsteinUhlenbeck | None
    earnings_variance_multiplier: float | None

    def updated_to(self, log_prices: np.ndarray) -> "ModelSuite":
        return replace(
            self,
            garch=self.garch.with_data(log_prices),
            hmm=self.hmm.with_data(log_prices),
            ornstein_uhlenbeck=self.ornstein_uhlenbeck.with_data(log_prices) if self.ornstein_uhlenbeck else None,
        )

    def candidate_models(self) -> dict[str, FittedReturnModel]:
        candidates = {"gbm": self.gbm, "garch_t": self.garch, "merton_jump": self.merton, "regime_switching": self.hmm}
        if self.earnings_variance_multiplier is not None:
            candidates[EARNINGS_VARIANT] = self.garch
        if self.ornstein_uhlenbeck is not None:
            candidates["ornstein_uhlenbeck"] = self.ornstein_uhlenbeck
        return candidates


def fit_model_suite(
    log_prices: np.ndarray,
    log_returns: pd.Series,
    reaction_days: pd.DatetimeIndex,
    settings: dict,
    rng: np.random.Generator,
    previous: ModelSuite | None = None,
) -> ModelSuite:
    """Estimate every model on `log_prices`. `previous` warm-starts the iterative fits at walk-forward refits.

    `log_returns` (dated, same span) and `reaction_days` (only those within the span
    are used) feed the earnings-variance estimate.
    """
    models, horizon = settings["models"], settings["forecast"]["horizon_days"]
    adf = detect_mean_reversion(log_prices, models["mean_reversion"]["adf_significance"])
    return ModelSuite(
        gbm=fit_gbm(log_prices),
        garch=fit_garch(
            log_prices,
            asymmetric=models["garch"]["asymmetric"],
            max_iterations=models["garch"]["max_iterations"],
            initial=previous.garch if previous else None,
        ),
        merton=fit_merton(
            log_prices,
            max_jumps_per_day=models["merton"]["max_jumps_per_day"],
            initial_jump_threshold_sd=models["merton"]["initial_jump_threshold_sd"],
            max_iterations=models["merton"]["max_iterations"],
        ),
        hmm=fit_hmm(
            log_prices,
            n_states=models["hmm"]["states"],
            max_iterations=models["hmm"]["max_iterations"],
            tolerance=models["hmm"]["tolerance"],
            random_restarts=models["hmm"]["random_restarts"],
            horizon=horizon,
            rng=rng,
            initial=previous.hmm if previous else None,
        ),
        kalman=fit_kalman_trend(log_prices, max_iterations=models["kalman"]["max_iterations"], initial=previous.kalman if previous else None),
        adf=adf,
        ornstein_uhlenbeck=fit_ornstein_uhlenbeck(log_prices, horizon=horizon) if adf.is_stationary else None,
        earnings_variance_multiplier=estimate_earnings_variance_multiplier(
            log_returns, reaction_days, min_events=models["earnings"]["min_events"]
        ),
    )


def simulate_candidates(
    suite: ModelSuite,
    *,
    horizon: int,
    n_paths: int,
    rng: np.random.Generator,
    earnings_multipliers: np.ndarray,
    price_band: float | None,
) -> dict[str, np.ndarray]:
    samples = {}
    for name, model in suite.candidate_models().items():
        multipliers = earnings_multipliers if name == EARNINGS_VARIANT else None
        samples[name] = simulate_cumulative_log_returns(model, horizon, n_paths, rng, shock_multipliers=multipliers, price_band=price_band)
    return samples


def simulate_engine(
    suite: ModelSuite,
    volatility_model_name: str,
    drift_mean: float,
    drift_std: float,
    *,
    horizon: int,
    n_paths: int,
    rng: np.random.Generator,
    earnings_multipliers: np.ndarray,
    price_band: float | None,
) -> np.ndarray:
    """The engine's forecast: Bayesian drift (one draw per path, integrating its uncertainty) + chosen model's shocks."""
    model = suite.candidate_models()[volatility_model_name]
    daily_drift = rng.normal(drift_mean, drift_std, n_paths) / horizon
    multipliers = earnings_multipliers if volatility_model_name == EARNINGS_VARIANT else None
    return simulate_cumulative_log_returns(model, horizon, n_paths, rng, daily_drift=daily_drift, shock_multipliers=multipliers, price_band=price_band)
