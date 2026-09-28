"""Gaussian hidden Markov model of daily log returns (latent market regimes).

    s_t ∈ {calm, turbulent},  P(s_t = j | s_{t−1} = i) = A_ij,   r_t | s_t = k ~ N(m_k, v_k)

Fitted by Baum–Welch (EM). States are ordered by volatility after fitting so
"state 0 = calm, state 1 = turbulent" means the same thing at every refit
(EM itself returns states in arbitrary order — the label-switching problem).

What a returns-only HMM can and cannot see: it separates periods by the
*distribution* of returns, which in practice means volatility level. It does
not identify "trending vs ranging" — that is measured separately with ADX.

Lookahead: the current regime uses FILTERED probabilities P(s_T | r_1..r_T).
Smoothed probabilities P(s_t | r_1..r_T) for t < T use later data and are only
used inside EM, never as a regime reading.
"""

from dataclasses import dataclass, replace

import numpy as np

from finance_engine.models.base import FittedReturnModel

# A state's variance may not fall below this fraction of the overall return variance.
# Without a floor, EM can shrink one state onto a handful of identical days (a
# degenerate likelihood spike) — a known EM failure mode for Gaussian mixtures.
RELATIVE_VARIANCE_FLOOR = 1e-2


@dataclass(frozen=True)
class FittedHmm(FittedReturnModel):
    initial_probabilities: np.ndarray
    transition_matrix: np.ndarray
    state_means: np.ndarray
    state_variances: np.ndarray
    filtered_probabilities: np.ndarray  # P(s_T | data through T)
    log_likelihood: float
    iterations: int
    horizon_for_drift: int = 5
    name: str = "regime_switching"

    @property
    def next_day_state_probabilities(self) -> np.ndarray:
        return self.filtered_probabilities @ self.transition_matrix

    @property
    def daily_drift(self) -> float:
        # Average expected return over the forecast horizon, as regime probabilities evolve.
        probabilities, total = self.filtered_probabilities, 0.0
        for _ in range(self.horizon_for_drift):
            probabilities = probabilities @ self.transition_matrix
            total += probabilities @ self.state_means
        return float(total / self.horizon_for_drift)

    def simulate_daily_shocks(self, horizon, n_paths, rng):
        n_states = len(self.state_means)
        cumulative_transition = np.cumsum(self.transition_matrix, axis=1)
        states = _sample_categorical(rng, self.next_day_state_probabilities, n_paths)
        returns = np.empty((n_paths, horizon))
        for day in range(horizon):
            if day > 0:
                draws = rng.uniform(size=n_paths)[:, None]
                states = np.minimum((draws > cumulative_transition[states]).sum(axis=1), n_states - 1)
            returns[:, day] = self.state_means[states] + np.sqrt(self.state_variances[states]) * rng.normal(size=n_paths)
        return returns - self.daily_drift

    def with_data(self, log_prices):
        log_returns = np.diff(np.asarray(log_prices, float))
        filtered, _, log_likelihood = forward_filter(log_returns, self.initial_probabilities, self.transition_matrix, self.state_means, self.state_variances)
        return replace(self, filtered_probabilities=filtered[-1], log_likelihood=log_likelihood)

    @property
    def turbulent_probability(self) -> float:
        return float(self.filtered_probabilities[-1])

    def describe(self):
        stay = np.diag(self.transition_matrix)
        return {
            "state_daily_volatility": np.sqrt(self.state_variances).tolist(),
            "state_daily_mean": self.state_means.tolist(),
            "expected_regime_duration_days": (1 / np.maximum(1 - stay, 1e-12)).tolist(),
            "current_turbulent_probability": self.turbulent_probability,
        }


def _sample_categorical(rng, probabilities, size):
    return np.minimum(np.searchsorted(np.cumsum(probabilities), rng.uniform(size=size)), len(probabilities) - 1)


def _scaled_emissions(log_returns, means, variances):
    """Emission densities divided by each day's largest one (computed in log space, so they can't underflow).

    Returns (scaled emissions T×K, per-day log of the divisor). The divisor cancels
    in every filtered/smoothed probability and is added back in the log-likelihood.
    """
    log_emissions = -0.5 * (log_returns[:, None] - means) ** 2 / variances - 0.5 * np.log(2 * np.pi * variances)
    log_divisor = log_emissions.max(axis=1)
    return np.exp(log_emissions - log_divisor[:, None]), log_divisor


def forward_filter(log_returns, initial, transition, means, variances):
    """Scaled forward pass. Returns filtered probabilities (T×K), scaling constants, log-likelihood."""
    emissions, log_divisor = _scaled_emissions(log_returns, means, variances)
    n, k = emissions.shape
    filtered = np.empty((n, k))
    scale = np.empty(n)
    predicted = initial
    for t in range(n):
        joint = predicted * emissions[t]
        scale[t] = joint.sum()
        filtered[t] = joint / scale[t]
        predicted = filtered[t] @ transition
    return filtered, scale, float(np.sum(np.log(scale)) + np.sum(log_divisor))


def _baum_welch(log_returns, initial, transition, means, variances, max_iterations, tolerance):
    variance_floor = RELATIVE_VARIANCE_FLOOR * log_returns.var()
    previous_log_likelihood = -np.inf
    for iteration in range(1, max_iterations + 1):
        emissions, _ = _scaled_emissions(log_returns, means, variances)
        filtered, scale, log_likelihood = forward_filter(log_returns, initial, transition, means, variances)

        n, k = emissions.shape
        backward = np.ones((n, k))
        for t in range(n - 2, -1, -1):
            backward[t] = transition @ (emissions[t + 1] * backward[t + 1]) / scale[t + 1]
        smoothed = filtered * backward
        smoothed /= smoothed.sum(axis=1, keepdims=True)
        # Expected transition counts ξ summed over time.
        pair_weights = (filtered[:-1, :, None] * transition[None] * (emissions[1:] * backward[1:])[:, None, :]) / scale[1:, None, None]
        transition_counts = pair_weights.sum(axis=0)

        initial = smoothed[0]
        transition = transition_counts / transition_counts.sum(axis=1, keepdims=True)
        weights = smoothed.sum(axis=0)
        means = (smoothed * log_returns[:, None]).sum(axis=0) / weights
        variances = np.maximum((smoothed * (log_returns[:, None] - means) ** 2).sum(axis=0) / weights, variance_floor)

        if abs(log_likelihood - previous_log_likelihood) < tolerance * (1 + abs(log_likelihood)):
            break
        previous_log_likelihood = log_likelihood
    return initial, transition, means, variances, iteration


def fit_hmm(
    log_prices: np.ndarray,
    *,
    n_states: int,
    max_iterations: int,
    tolerance: float,
    random_restarts: int,
    horizon: int,
    rng: np.random.Generator,
    initial: "FittedHmm | None" = None,
) -> FittedHmm:
    """EM fit. With `initial` (a previous walk-forward fit) it warm-starts once; otherwise it tries several starts."""
    log_returns = np.diff(np.asarray(log_prices, float))
    starts = []
    if initial is not None:
        starts.append((initial.initial_probabilities, initial.transition_matrix, initial.state_means, initial.state_variances))
    else:
        overall_variance = log_returns.var()
        for restart in range(random_restarts):
            spread = np.sort(rng.uniform(0.3, 3.0, n_states)) if restart else np.geomspace(0.5, 2.5, n_states)
            stay = rng.uniform(0.9, 0.99) if restart else 0.97
            transition = np.full((n_states, n_states), (1 - stay) / max(n_states - 1, 1))
            np.fill_diagonal(transition, stay)
            starts.append((np.full(n_states, 1 / n_states), transition, np.full(n_states, log_returns.mean()), overall_variance * spread))

    best = None
    for start in starts:
        initial_p, transition, means, variances, iterations = _baum_welch(log_returns, *start, max_iterations, tolerance)
        _, _, log_likelihood = forward_filter(log_returns, initial_p, transition, means, variances)
        if not np.isfinite(log_likelihood):
            continue  # numerically failed start: discard rather than let it win
        if best is None or log_likelihood > best[-1]:
            best = (initial_p, transition, means, variances, iterations, log_likelihood)

    initial_p, transition, means, variances, iterations, log_likelihood = best
    order = np.argsort(variances)  # calm first, turbulent last
    initial_p, means, variances = initial_p[order], means[order], variances[order]
    transition = transition[np.ix_(order, order)]
    filtered, _, log_likelihood = forward_filter(log_returns, initial_p, transition, means, variances)
    return FittedHmm(
        initial_probabilities=initial_p,
        transition_matrix=transition,
        state_means=means,
        state_variances=variances,
        filtered_probabilities=filtered[-1],
        log_likelihood=log_likelihood,
        iterations=iterations,
        horizon_for_drift=horizon,
    )


def filtered_turbulent_probability_series(model: FittedHmm, log_prices: np.ndarray) -> np.ndarray:
    """P(turbulent | data through t) for every t — causal, safe to use as a historical regime label."""
    log_returns = np.diff(np.asarray(log_prices, float))
    filtered, _, _ = forward_filter(log_returns, model.initial_probabilities, model.transition_matrix, model.state_means, model.state_variances)
    return np.concatenate([[np.nan], filtered[:, -1]])
