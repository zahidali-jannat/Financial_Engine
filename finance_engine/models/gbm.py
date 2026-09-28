"""Geometric Brownian Motion — the null model every other model must beat.

Assumption: daily log returns are independent draws from one normal
distribution N(μ, σ²). Equivalently the price is dP = (μ + σ²/2)·P·dt + σ·P·dW.
The assumption is testable — assumption_tests() reports whether returns look
normal (Jarque–Bera) and whether volatility clusters (ARCH-LM). When either
fails, GBM's intervals will be miscalibrated in a predictable way: too narrow
in the tails, and too wide in calm periods / too narrow in turbulent ones.
"""

from dataclasses import dataclass

import numpy as np

from finance_engine.models.base import FittedReturnModel
from finance_engine.stats.hypothesis_tests import arch_lm_test, jarque_bera


@dataclass(frozen=True)
class FittedGbm(FittedReturnModel):
    daily_drift: float
    daily_volatility: float
    name: str = "gbm"

    def simulate_daily_shocks(self, horizon, n_paths, rng):
        return rng.normal(0.0, self.daily_volatility, size=(n_paths, horizon))

    def with_data(self, log_prices):
        return self  # no state: parameters are fixed between refits

    def describe(self):
        return {"daily_drift": self.daily_drift, "daily_volatility": self.daily_volatility}


def fit_gbm(log_prices: np.ndarray) -> FittedGbm:
    """Maximum-likelihood μ and σ of daily log returns."""
    log_returns = np.diff(np.asarray(log_prices, float))
    return FittedGbm(daily_drift=float(log_returns.mean()), daily_volatility=float(log_returns.std(ddof=1)))


def gbm_assumption_tests(log_prices: np.ndarray, arch_lags: int = 5) -> dict:
    """Test GBM's two core assumptions on the data rather than assuming them."""
    log_returns = np.diff(np.asarray(log_prices, float))
    normality = jarque_bera(log_returns)
    clustering = arch_lm_test(log_returns - log_returns.mean(), lags=arch_lags)
    z = (log_returns - log_returns.mean()) / log_returns.std()
    return {
        "excess_kurtosis": float(np.mean(z**4) - 3),
        "skewness": float(np.mean(z**3)),
        "jarque_bera_p_value": normality.p_value,
        "returns_are_normal": normality.p_value >= 0.05,
        "arch_lm_p_value": clustering.p_value,
        "volatility_clusters": clustering.p_value < 0.05,
        "moves_beyond_4_sd": int(np.sum(np.abs(z) > 4)),
        "moves_beyond_4_sd_expected_if_normal": float(len(z) * 6.334e-5),
    }
