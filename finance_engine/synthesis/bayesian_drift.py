"""Bayesian estimate of the expected h-day log return (the drift), given today's evidence.

Model (conjugate Bayesian linear regression on standardized evidence x):

    y_i = β₀ + x_iᵀβ + e_i,       e_i ~ N(0, s²)
    β₀  ~ N(m₀, v₀)               prior: long-run drift (+ fundamental tilt), wide
    β_j ~ N(0, τ²)                prior: each signal has NO effect unless data shows one

    Posterior:  Σ = (Λ₀ + XᵀX / s²)⁻¹,    m = Σ (Λ₀ m_prior + Xᵀy / s²)
    Today:      E[y | x_T] ~ N(x̃_Tᵀ m, x̃_Tᵀ Σ x̃_T)      with x̃ = [1, x]

Why this and not plain regression: τ shrinks signal effects toward zero, so a
signal only moves the forecast when the data supports it strongly enough to
overcome the prior — the principled defence against overfitting noise. With
typical technical signals the posterior stays close to the prior; that is the
correct answer when evidence is weak, not a malfunction.

Lookahead guards:
  • training rows are origin − h, origin − 2h, …: each outcome y_i ended on or before
    the forecast date, and outcome windows don't overlap (overlap would make s² and
    the effective sample size wrong);
  • standardization uses training rows only.
s² is the training variance of y — a plug-in (empirical Bayes) choice; with
hundreds of rows its own uncertainty is negligible next to the drift's.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_TRAINING_ROWS = 20
MIN_ROWS_PER_FEATURE = 30


@dataclass(frozen=True)
class DriftPosterior:
    mean: float  # expected h-day log return
    std: float  # uncertainty about that expectation (not outcome noise)
    prior_mean: float
    prior_std: float
    contributions: dict  # "baseline" and each feature -> its share of `mean`
    coefficients: dict  # feature -> (posterior mean, posterior std), per 1 training std of the feature
    training_rows: int


def estimate_drift_posterior(
    evidence: pd.DataFrame,
    forward_returns: pd.Series,
    origin_position: int,
    *,
    horizon: int,
    prior_mean: float,
    prior_std: float,
    signal_effect_prior_sd: float,
) -> DriftPosterior:
    training_positions = np.arange(origin_position - horizon, -1, -horizon)[::-1]
    y = forward_returns.iloc[training_positions]
    X = evidence.iloc[training_positions]

    usable_features = [
        column
        for column in evidence.columns
        if X[column].notna().sum() >= MIN_ROWS_PER_FEATURE and X[column].std(skipna=True) > 1e-12
    ]
    rows = y.notna() & X[usable_features].notna().all(axis=1)
    y, X = y[rows].to_numpy(), X.loc[rows, usable_features]

    if len(y) < MIN_TRAINING_ROWS:
        return DriftPosterior(prior_mean, prior_std, prior_mean, prior_std, {"baseline": prior_mean}, {}, len(y))

    feature_means, feature_stds = X.mean(), X.std()
    standardized = ((X - feature_means) / feature_stds).to_numpy()
    current = ((evidence.iloc[origin_position][usable_features] - feature_means) / feature_stds).fillna(0.0).to_numpy()

    design = np.column_stack([np.ones(len(y)), standardized])
    noise_variance = float(np.var(y, ddof=1))
    prior_precision = np.diag([1 / prior_std**2] + [1 / signal_effect_prior_sd**2] * len(usable_features))
    prior_location = np.array([prior_mean] + [0.0] * len(usable_features))

    posterior_covariance = np.linalg.inv(prior_precision + design.T @ design / noise_variance)
    posterior_mean = posterior_covariance @ (prior_precision @ prior_location + design.T @ y / noise_variance)

    today = np.concatenate([[1.0], current])
    contributions = {"baseline": float(posterior_mean[0])}
    contributions.update({name: float(coef * value) for name, coef, value in zip(usable_features, posterior_mean[1:], current)})
    coefficient_std = np.sqrt(np.diag(posterior_covariance))
    return DriftPosterior(
        mean=float(today @ posterior_mean),
        std=float(np.sqrt(today @ posterior_covariance @ today)),
        prior_mean=prior_mean,
        prior_std=prior_std,
        contributions=contributions,
        coefficients={name: (float(m), float(s)) for name, m, s in zip(usable_features, posterior_mean[1:], coefficient_std[1:])},
        training_rows=len(y),
    )


def drift_prior(bayes_settings: dict, *, horizon: int, trading_days_per_year: int, fundamental_annual_tilt: float = 0.0) -> tuple[float, float]:
    """Prior mean and std of the h-day expected log return, from annual settings."""
    scale = horizon / trading_days_per_year
    return (bayes_settings["prior_annual_drift"] + fundamental_annual_tilt) * scale, bayes_settings["prior_annual_drift_sd"] * scale
