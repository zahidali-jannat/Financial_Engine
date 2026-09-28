"""GJR-GARCH(1,1) with standardized Student-t innovations.

    r_t  = μ + ε_t,         ε_t = σ_t · z_t,     z_t ~ t_ν scaled to unit variance
    σ²_t = ω + (α + γ·1[ε_{t−1} < 0]) · ε²_{t−1} + β · σ²_{t−1}

Why this form:
  • volatility clustering  — β carries yesterday's variance forward, α reacts to
    yesterday's shock; high-vol days follow high-vol days (ARCH-LM tests for it).
  • leverage effect        — γ > 0 lets falls raise volatility more than rises.
  • fat tails              — ν (degrees of freedom) is estimated; small ν = fat tails.
  EGARCH would also capture asymmetry; GJR is kept because its parameters map
  directly onto variance and its stationarity condition is simple.

Stationarity requires α + γ/2 + β < 1 (z is symmetric, so a shock is negative
half the time). The optimiser works on unconstrained numbers that are mapped
into this region, so every candidate it tries is a valid, stationary model.

μ is fixed at the sample mean (the drift is Layer 4's job). Returns are scaled
to percent inside the fit because the optimiser is better conditioned there.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

from finance_engine.models.base import FittedReturnModel
from finance_engine.stats.distributions import sample_standardized_t, standardized_t_logpdf
from finance_engine.stats.optimize import minimize_nelder_mead

PERCENT = 100.0
MAX_PERSISTENCE = 0.9995
MIN_DEGREES_OF_FREEDOM = 2.05


@dataclass(frozen=True)
class FittedGarch(FittedReturnModel):
    daily_drift: float
    omega: float  # in percent² units
    alpha: float
    gamma: float
    beta: float
    degrees_of_freedom: float
    next_variance: float  # σ²_{T+1} in percent², the variance of the first forecast day
    last_shock: float  # ε_T in percent
    last_variance: float  # σ²_T in percent²
    log_likelihood: float
    converged: bool
    name: str = "garch_t"

    @property
    def persistence(self) -> float:
        return self.alpha + self.gamma / 2 + self.beta

    def simulate_daily_shocks(self, horizon, n_paths, rng):
        variance = np.full(n_paths, self.next_variance)
        shocks = np.empty((n_paths, horizon))
        for day in range(horizon):
            shock = np.sqrt(variance) * sample_standardized_t(rng, self.degrees_of_freedom, n_paths)
            shocks[:, day] = shock
            variance = self.omega + (self.alpha + self.gamma * (shock < 0)) * shock**2 + self.beta * variance
        return shocks / PERCENT

    def with_data(self, log_prices):
        shocks = PERCENT * (np.diff(np.asarray(log_prices, float)) - self.daily_drift)
        variances = garch_variance_path(shocks, self.omega, self.alpha, self.gamma, self.beta, self.unconditional_variance)
        return replace(self, next_variance=float(variances[-1]), last_shock=float(shocks[-1]), last_variance=float(variances[-2]))

    @property
    def unconditional_variance(self) -> float:
        return self.omega / (1 - self.persistence)

    def describe(self):
        return {
            "alpha": self.alpha,
            "gamma_leverage": self.gamma,
            "beta": self.beta,
            "persistence": self.persistence,
            "volatility_half_life_days": math.log(0.5) / math.log(self.persistence) if self.persistence > 0 else 0.0,
            "tail_degrees_of_freedom": self.degrees_of_freedom,
            "long_run_daily_volatility": math.sqrt(self.unconditional_variance) / PERCENT,
            "next_day_volatility": math.sqrt(self.next_variance) / PERCENT,
            "converged": self.converged,
        }


def garch_variance_path(shocks, omega, alpha, gamma, beta, initial_variance) -> np.ndarray:
    """σ²_1..σ²_{T+1} for shocks ε_1..ε_T (percent units). Plain-float loop: the recursion is inherently sequential."""
    variances = [initial_variance]
    previous_variance = initial_variance
    for shock in shocks.tolist():
        weight = alpha + gamma if shock < 0 else alpha
        previous_variance = omega + weight * shock * shock + beta * previous_variance
        variances.append(previous_variance)
    return np.asarray(variances)


def _unpack(theta, asymmetric: bool):
    omega = math.exp(theta[0])
    persistence = MAX_PERSISTENCE / (1 + math.exp(-theta[1]))
    logits = [theta[2], theta[3], 0.0] if asymmetric else [theta[2], 0.0]
    largest = max(logits)
    weights = [math.exp(v - largest) for v in logits]
    total = sum(weights)
    shares = [persistence * w / total for w in weights]
    if asymmetric:
        alpha, half_gamma, beta = shares
        gamma = 2 * half_gamma
    else:
        (alpha, beta), gamma = shares, 0.0
    degrees_of_freedom = MIN_DEGREES_OF_FREEDOM + math.exp(theta[-1])
    return omega, alpha, gamma, beta, degrees_of_freedom


def _negative_log_likelihood(theta, shocks, initial_variance, asymmetric):
    omega, alpha, gamma, beta, nu = _unpack(theta, asymmetric)
    if nu > 200:
        return np.inf
    variances = garch_variance_path(shocks, omega, alpha, gamma, beta, initial_variance)[:-1]
    log_likelihood = np.sum(standardized_t_logpdf(shocks / np.sqrt(variances), nu) - 0.5 * np.log(variances))
    return -log_likelihood


def fit_garch(log_prices: np.ndarray, *, asymmetric: bool, max_iterations: int, initial: "FittedGarch | None" = None) -> FittedGarch:
    """Maximum-likelihood GJR-GARCH-t. `initial` warm-starts from a previous fit (used at walk-forward refits)."""
    log_returns = np.diff(np.asarray(log_prices, float))
    daily_drift = float(log_returns.mean())
    shocks = PERCENT * (log_returns - daily_drift)
    sample_variance = float(np.var(shocks))

    if initial is not None:
        theta0 = _pack(initial, asymmetric)
    else:
        start_persistence, start_alpha, start_half_gamma = 0.97, 0.05, 0.03
        start_beta = start_persistence - start_alpha - (start_half_gamma if asymmetric else 0)
        logits = [math.log(start_alpha / start_beta)] + ([math.log(start_half_gamma / start_beta)] if asymmetric else [])
        theta0 = np.array(
            [math.log(sample_variance * (1 - start_persistence)), math.log(start_persistence / (MAX_PERSISTENCE - start_persistence))]
            + logits
            + [math.log(8.0 - MIN_DEGREES_OF_FREEDOM)]
        )

    result = minimize_nelder_mead(
        lambda theta: _negative_log_likelihood(theta, shocks, sample_variance, asymmetric),
        theta0,
        max_iterations=max_iterations,
        tolerance=1e-9,
    )
    omega, alpha, gamma, beta, nu = _unpack(result.parameters, asymmetric)
    variances = garch_variance_path(shocks, omega, alpha, gamma, beta, sample_variance)
    return FittedGarch(
        daily_drift=daily_drift,
        omega=omega,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        degrees_of_freedom=nu,
        next_variance=float(variances[-1]),
        last_shock=float(shocks[-1]),
        last_variance=float(variances[-2]),
        log_likelihood=-result.objective_value,
        converged=result.converged,
    )


def _pack(model: FittedGarch, asymmetric: bool) -> np.ndarray:
    persistence = min(model.persistence, MAX_PERSISTENCE * 0.999)
    beta = max(model.beta, 1e-6)
    logits = [math.log(max(model.alpha, 1e-6) / beta)]
    if asymmetric:
        logits.append(math.log(max(model.gamma / 2, 1e-6) / beta))
    return np.array(
        [math.log(model.omega), math.log(persistence / (MAX_PERSISTENCE - persistence))]
        + logits
        + [math.log(model.degrees_of_freedom - MIN_DEGREES_OF_FREEDOM)]
    )

