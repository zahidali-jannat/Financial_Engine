"""Merton jump-diffusion: normal diffusion plus Poisson-arriving normal jumps.

    r_t = μ + σ·Z_t + Σ_{i=1}^{N_t} Y_i,    N_t ~ Poisson(λ),  Y_i ~ N(μ_J, σ_J²)

Given N_t = n the return is N(μ + n·μ_J, σ² + n·σ_J²), so the exact density is
the Poisson-weighted mixture Σ_n P(N=n)·φ(r; μ + nμ_J, σ² + nσ_J²), truncated
at `max_jumps_per_day` terms. Parameters are fitted by maximum likelihood,
starting from the historical moves beyond `initial_jump_threshold_sd`.

Known weakness (reported, not hidden): λ and σ_J trade off against each other
— many small jumps look like a slightly fatter normal — so individual jump
parameters are less reliable than the overall return distribution they imply.
Merton also has no volatility clustering; the out-of-sample comparison with
GARCH shows which failing matters more for a given stock.
"""

import math
from dataclasses import dataclass

import numpy as np

from finance_engine.models.base import FittedReturnModel
from finance_engine.stats.optimize import minimize_nelder_mead

PERCENT = 100.0
MAX_DAILY_JUMP_INTENSITY = 0.5


@dataclass(frozen=True)
class FittedMerton(FittedReturnModel):
    diffusion_drift: float  # μ, daily log-return units
    diffusion_volatility: float  # σ
    jump_intensity: float  # λ, expected jumps per day
    jump_mean: float  # μ_J
    jump_volatility: float  # σ_J
    log_likelihood: float
    converged: bool
    name: str = "merton_jump"

    @property
    def daily_drift(self) -> float:
        return self.diffusion_drift + self.jump_intensity * self.jump_mean

    def simulate_daily_shocks(self, horizon, n_paths, rng):
        diffusion = rng.normal(0.0, self.diffusion_volatility, size=(n_paths, horizon))
        jump_counts = rng.poisson(self.jump_intensity, size=(n_paths, horizon))
        # The sum of n iid N(μ_J, σ_J²) jumps is N(n·μ_J, n·σ_J²).
        jumps = jump_counts * self.jump_mean + np.sqrt(jump_counts) * self.jump_volatility * rng.normal(size=(n_paths, horizon))
        return diffusion + jumps - self.jump_intensity * self.jump_mean

    def with_data(self, log_prices):
        return self  # returns are independent across days: no state to update

    def describe(self):
        return {
            "diffusion_volatility": self.diffusion_volatility,
            "jumps_per_year_at_248_days": self.jump_intensity * 248,
            "jump_mean": self.jump_mean,
            "jump_volatility": self.jump_volatility,
            "converged": self.converged,
        }


def _unpack(theta):
    mu, log_sigma, intensity_logit, jump_mean, log_jump_sigma = theta
    intensity = MAX_DAILY_JUMP_INTENSITY / (1 + math.exp(-intensity_logit))
    return mu, math.exp(log_sigma), intensity, jump_mean, math.exp(log_jump_sigma)


def merton_log_density(returns_pct: np.ndarray, theta, max_jumps: int) -> np.ndarray:
    mu, sigma, intensity, jump_mean, jump_sigma = _unpack(theta)
    n = np.arange(max_jumps + 1)[:, None]
    log_poisson = -intensity + n * math.log(intensity) - np.array([math.lgamma(k + 1) for k in range(max_jumps + 1)])[:, None]
    variance = sigma**2 + n * jump_sigma**2
    log_normal = -0.5 * (returns_pct[None, :] - mu - n * jump_mean) ** 2 / variance - 0.5 * np.log(2 * math.pi * variance)
    terms = log_poisson + log_normal
    largest = terms.max(axis=0)
    return largest + np.log(np.exp(terms - largest).sum(axis=0))


def fit_merton(log_prices: np.ndarray, *, max_jumps_per_day: int, initial_jump_threshold_sd: float, max_iterations: int) -> FittedMerton:
    returns_pct = PERCENT * np.diff(np.asarray(log_prices, float))
    z = (returns_pct - returns_pct.mean()) / returns_pct.std()
    is_extreme = np.abs(z) > initial_jump_threshold_sd
    normal_days = returns_pct[~is_extreme]
    extreme_days = returns_pct[is_extreme]
    start_intensity = min(max(is_extreme.mean(), 0.005), MAX_DAILY_JUMP_INTENSITY * 0.9)
    start_jump_sigma = extreme_days.std() if len(extreme_days) > 2 else 3 * normal_days.std()
    theta0 = np.array(
        [
            normal_days.mean(),
            math.log(normal_days.std()),
            math.log(start_intensity / (MAX_DAILY_JUMP_INTENSITY - start_intensity)),
            extreme_days.mean() if len(extreme_days) else 0.0,
            math.log(max(start_jump_sigma, 1e-3)),
        ]
    )
    result = minimize_nelder_mead(
        lambda theta: -np.sum(merton_log_density(returns_pct, theta, max_jumps_per_day)),
        theta0,
        max_iterations=max_iterations,
        tolerance=1e-9,
    )
    mu, sigma, intensity, jump_mean, jump_sigma = _unpack(result.parameters)
    return FittedMerton(
        diffusion_drift=mu / PERCENT,
        diffusion_volatility=sigma / PERCENT,
        jump_intensity=intensity,
        jump_mean=jump_mean / PERCENT,
        jump_volatility=jump_sigma / PERCENT,
        log_likelihood=-result.objective_value,
        converged=result.converged,
    )
