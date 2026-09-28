"""Kalman filter with a local linear trend: a denoised trend (slope) estimate that carries its own uncertainty.

    y_t     = ℓ_t + e_t,          e_t ~ N(0, σ²·ψ_obs)     observed log price = level + noise
    ℓ_{t+1} = ℓ_t + b_t + u_t,    u_t ~ N(0, σ²)           level moves by the slope
    b_{t+1} = b_t + v_t,          v_t ~ N(0, σ²·ψ_slope)   slope (daily drift) wanders slowly

The filtered slope b_t is the trend estimate; its variance P_bb,t says how sure
the filter is. Both use data through t only. Noise ratios ψ are fitted by
maximum likelihood with σ² concentrated out (it has a closed form given ψ).

Honest caveat: when the observation noise ψ_obs is near zero — typical for
daily prices, which are close to a random walk — the steady-state filter is
equivalent to an exponentially weighted average of past returns. What it adds
over a plain EMA is a principled smoothing weight (from the likelihood) and the
slope's variance, which the Bayesian step uses. Its predictive value is not
assumed: Layer 4 learns the weight on it from out-of-sample-safe data.
"""

import math
from dataclasses import dataclass

import numpy as np

from finance_engine.stats.optimize import minimize_nelder_mead

# Initial state variance, in units of the level noise σ²: large enough to mean "unknown" (a diffuse prior).
DIFFUSE_SCALE = 1e4
BURN_IN = 2  # observations needed before the level and slope are both identified


@dataclass(frozen=True)
class KalmanTrendModel:
    level_variance: float  # σ²
    observation_ratio: float  # ψ_obs
    slope_ratio: float  # ψ_slope
    converged: bool

    def filter(self, log_prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Filtered slope b_t and its variance for each t (causal)."""
        sigma_squared = self.level_variance
        slopes, slope_variances, _, _ = _run_filter(
            np.asarray(log_prices, float), sigma_squared, sigma_squared * self.observation_ratio, sigma_squared * self.slope_ratio
        )
        return slopes, slope_variances

    def slope_z_scores(self, log_prices: np.ndarray) -> np.ndarray:
        slopes, variances = self.filter(log_prices)
        z = slopes / np.sqrt(variances)
        z[:BURN_IN] = np.nan
        return z

    def describe(self):
        return {
            "signal_to_noise_level": 1 / self.observation_ratio if self.observation_ratio > 0 else math.inf,
            "slope_to_level_noise_ratio": self.slope_ratio,
            "converged": self.converged,
        }


def _run_filter(y, level_noise, observation_noise, slope_noise):
    # Scaling the diffuse prior by the level noise keeps the fit (σ² = 1) and the filter (σ² estimated) identical.
    level, slope = y[0], 0.0
    p_ll, p_lb, p_bb = DIFFUSE_SCALE * level_noise, 0.0, DIFFUSE_SCALE * level_noise
    n = len(y)
    slopes, slope_variances = np.empty(n), np.empty(n)
    sum_log_f = sum_scaled_sq = 0.0
    for t, observation in enumerate(y.tolist()):
        if t > 0:  # predict one step ahead
            level = level + slope
            p_ll, p_lb = p_ll + 2 * p_lb + p_bb + level_noise, p_lb + p_bb
            p_bb = p_bb + slope_noise
        innovation = observation - level
        f = p_ll + observation_noise
        gain_level, gain_slope = p_ll / f, p_lb / f
        level += gain_level * innovation
        slope += gain_slope * innovation
        p_ll, p_lb, p_bb = p_ll - gain_level * p_ll, p_lb - gain_level * p_lb, p_bb - gain_slope * p_lb
        if t >= BURN_IN:
            sum_log_f += math.log(f)
            sum_scaled_sq += innovation * innovation / f
        slopes[t], slope_variances[t] = slope, p_bb
    return slopes, slope_variances, sum_log_f, sum_scaled_sq


def fit_kalman_trend(log_prices: np.ndarray, *, max_iterations: int, initial: "KalmanTrendModel | None" = None) -> KalmanTrendModel:
    y = np.asarray(log_prices, float)
    n_effective = len(y) - BURN_IN

    def negative_concentrated_log_likelihood(theta):
        observation_ratio, slope_ratio = math.exp(theta[0]), math.exp(theta[1])
        _, _, sum_log_f, sum_scaled_sq = _run_filter(y, 1.0, observation_ratio, slope_ratio)
        sigma_squared = sum_scaled_sq / n_effective
        return 0.5 * (n_effective * math.log(sigma_squared) + sum_log_f)

    theta0 = (
        np.array([math.log(initial.observation_ratio), math.log(initial.slope_ratio)])
        if initial is not None
        else np.array([math.log(0.1), math.log(1e-4)])
    )
    result = minimize_nelder_mead(negative_concentrated_log_likelihood, theta0, initial_step=1.0, max_iterations=max_iterations, tolerance=1e-8)
    observation_ratio, slope_ratio = math.exp(result.parameters[0]), math.exp(result.parameters[1])
    _, _, _, sum_scaled_sq = _run_filter(y, 1.0, observation_ratio, slope_ratio)
    return KalmanTrendModel(
        level_variance=sum_scaled_sq / n_effective,
        observation_ratio=observation_ratio,
        slope_ratio=slope_ratio,
        converged=result.converged,
    )
