"""Ornstein–Uhlenbeck mean reversion — applied only when the data shows it.

    dx = κ(θ − x)dt + σ dW,   x = log price

Sampled daily this is exactly an AR(1): x_{t+1} = a + b·x_t + e_t with
b = e^{−κ}, θ = a/(1 − b), so κ = −ln b and the half-life is ln 2 / κ.

Gatekeeping: OU is fitted only if the Augmented Dickey–Fuller test rejects a
unit root in log price at the configured level. Stock prices almost never
pass — a stationary price would be an arbitrage — so on most stocks this
model is (correctly) absent. Note the multiple-testing effect: re-testing at
every walk-forward refit means ~5% of windows reject by chance even for a pure
random walk; the report shows how often the test rejected, not just the latest result.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

from finance_engine.models.base import FittedReturnModel
from finance_engine.stats.hypothesis_tests import AdfResult, augmented_dickey_fuller


@dataclass(frozen=True)
class FittedOrnsteinUhlenbeck(FittedReturnModel):
    intercept: float  # a
    persistence: float  # b
    residual_volatility: float
    current_log_price: float
    horizon_for_drift: int = 5
    name: str = "ornstein_uhlenbeck"

    @property
    def reversion_speed(self) -> float:
        return -math.log(self.persistence)

    @property
    def long_run_log_price(self) -> float:
        return self.intercept / (1 - self.persistence)

    @property
    def daily_drift(self) -> float:
        h, b = self.horizon_for_drift, self.persistence
        expected_gap_closed = (self.long_run_log_price - self.current_log_price) * (1 - b**h)
        return expected_gap_closed / h

    def simulate_daily_shocks(self, horizon, n_paths, rng):
        x = np.full(n_paths, self.current_log_price)
        daily = np.empty((n_paths, horizon))
        for day in range(horizon):
            next_x = self.intercept + self.persistence * x + self.residual_volatility * rng.normal(size=n_paths)
            daily[:, day] = next_x - x
            x = next_x
        return daily - self.daily_drift

    def with_data(self, log_prices):
        return replace(self, current_log_price=float(log_prices[-1]))

    def describe(self):
        return {
            "half_life_days": math.log(2) / self.reversion_speed,
            "long_run_price": math.exp(self.long_run_log_price),
            "residual_volatility": self.residual_volatility,
        }


def detect_mean_reversion(log_prices: np.ndarray, significance: float) -> AdfResult:
    return augmented_dickey_fuller(np.asarray(log_prices, float), significance=significance)


def fit_ornstein_uhlenbeck(log_prices: np.ndarray, *, horizon: int) -> FittedOrnsteinUhlenbeck | None:
    """OLS AR(1) on log price; None if the fitted process isn't mean-reverting (b outside (0, 1))."""
    x = np.asarray(log_prices, float)
    design = np.column_stack([np.ones(len(x) - 1), x[:-1]])
    (intercept, persistence), *_ = np.linalg.lstsq(design, x[1:], rcond=None)
    if not 0 < persistence < 1:
        return None
    residuals = x[1:] - design @ np.array([intercept, persistence])
    return FittedOrnsteinUhlenbeck(
        intercept=float(intercept),
        persistence=float(persistence),
        residual_volatility=float(residuals.std(ddof=2)),
        current_log_price=float(x[-1]),
        horizon_for_drift=horizon,
    )
