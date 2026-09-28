"""Parameter-recovery tests: simulate from known parameters, fit, and check the fit finds them."""

import math

import numpy as np
import pandas as pd
import pytest

from finance_engine.models.base import simulate_cumulative_log_returns
from finance_engine.models.earnings_jumps import (
    earnings_reaction_days,
    estimate_earnings_variance_multiplier,
    horizon_shock_multipliers,
)
from finance_engine.models.garch import fit_garch
from finance_engine.models.gbm import fit_gbm, gbm_assumption_tests
from finance_engine.models.hmm import filtered_turbulent_probability_series, fit_hmm
from finance_engine.models.jump_diffusion import fit_merton
from finance_engine.models.kalman import fit_kalman_trend
from finance_engine.models.mean_reversion import detect_mean_reversion, fit_ornstein_uhlenbeck
from finance_engine.stats.distributions import sample_standardized_t


def to_log_prices(log_returns, start=math.log(100)):
    return np.concatenate([[start], start + np.cumsum(log_returns)])


def test_gbm_recovers_drift_and_volatility_and_flags_normal_data_as_normal():
    returns = np.random.default_rng(1).normal(0.0005, 0.015, 5000)
    model = fit_gbm(to_log_prices(returns))
    assert model.daily_drift == pytest.approx(0.0005, abs=0.0005)
    assert model.daily_volatility == pytest.approx(0.015, rel=0.03)
    diagnostics = gbm_assumption_tests(to_log_prices(returns))
    assert diagnostics["returns_are_normal"] and not diagnostics["volatility_clusters"]


def simulate_gjr_garch_t(n, omega, alpha, gamma, beta, nu, seed):
    rng = np.random.default_rng(seed)
    variance = omega / (1 - alpha - gamma / 2 - beta)
    shocks = np.empty(n)
    for t in range(n):
        shocks[t] = math.sqrt(variance) * sample_standardized_t(rng, nu, 1)[0]
        variance = omega + (alpha + gamma * (shocks[t] < 0)) * shocks[t] ** 2 + beta * variance
    return shocks / 100  # percent -> log-return units


def test_garch_recovers_known_parameters():
    returns = simulate_gjr_garch_t(5000, omega=0.03, alpha=0.04, gamma=0.08, beta=0.89, nu=6.0, seed=2)
    model = fit_garch(to_log_prices(returns), asymmetric=True, max_iterations=4000)
    assert model.converged
    assert model.persistence == pytest.approx(0.97, abs=0.02)
    assert model.beta == pytest.approx(0.89, abs=0.05)
    assert model.gamma == pytest.approx(0.08, abs=0.05)
    assert model.degrees_of_freedom == pytest.approx(6.0, abs=2.0)
    assert gbm_assumption_tests(to_log_prices(returns))["volatility_clusters"]


def test_garch_state_update_matches_the_fit_and_first_simulated_day_has_that_variance():
    log_prices = to_log_prices(simulate_gjr_garch_t(3000, 0.03, 0.05, 0.05, 0.88, 8.0, seed=3))
    model = fit_garch(log_prices, asymmetric=True, max_iterations=3000)
    assert model.with_data(log_prices).next_variance == pytest.approx(model.next_variance, rel=1e-6)
    shocks = model.simulate_daily_shocks(5, 200_000, np.random.default_rng(0))
    assert shocks[:, 0].std() == pytest.approx(math.sqrt(model.next_variance) / 100, rel=0.02)
    assert abs(shocks.mean()) < 1e-4


def test_merton_recovers_jump_structure():
    rng = np.random.default_rng(4)
    n, intensity, jump_mean, jump_sd, sigma = 8000, 0.03, -0.02, 0.05, 0.01
    counts = rng.poisson(intensity, n)
    returns = 0.0004 + sigma * rng.normal(size=n) + counts * jump_mean + np.sqrt(counts) * jump_sd * rng.normal(size=n)
    model = fit_merton(to_log_prices(returns), max_jumps_per_day=8, initial_jump_threshold_sd=3.0, max_iterations=4000)
    assert model.diffusion_volatility == pytest.approx(sigma, rel=0.1)
    assert 0.015 < model.jump_intensity < 0.06
    assert model.jump_volatility == pytest.approx(jump_sd, rel=0.3)
    shocks = model.simulate_daily_shocks(1, 400_000, np.random.default_rng(1))
    expected_variance = model.diffusion_volatility**2 + model.jump_intensity * (model.jump_mean**2 + model.jump_volatility**2)
    assert shocks.var() == pytest.approx(expected_variance, rel=0.05)
    assert abs(shocks.mean()) < 2e-4


def test_hmm_recovers_regimes_and_labels_turbulent_state_last():
    rng = np.random.default_rng(5)
    n, stay = 4000, np.array([0.98, 0.95])
    vols = np.array([0.008, 0.025])
    states = np.empty(n, int)
    states[0] = 0
    for t in range(1, n):
        states[t] = states[t - 1] if rng.uniform() < stay[states[t - 1]] else 1 - states[t - 1]
    returns = vols[states] * rng.normal(size=n)
    log_prices = to_log_prices(returns)

    model = fit_hmm(log_prices, n_states=2, max_iterations=300, tolerance=1e-8, random_restarts=3, horizon=5, rng=np.random.default_rng(0))

    assert np.sqrt(model.state_variances) == pytest.approx(vols, rel=0.15)
    assert np.diag(model.transition_matrix) == pytest.approx(stay, abs=0.03)
    turbulent_probability = filtered_turbulent_probability_series(model, log_prices)[1:]
    assert np.mean((turbulent_probability > 0.5) == (states == 1)) > 0.85


def test_kalman_slope_tracks_a_known_trend_and_is_causal():
    rng = np.random.default_rng(6)
    n = 3000
    true_slope = np.cumsum(rng.normal(0, 5e-5, n))
    level = np.cumsum(true_slope + rng.normal(0, 0.01, n))
    observed = level + rng.normal(0, 0.003, n)

    model = fit_kalman_trend(observed, max_iterations=800)
    slopes, variances = model.filter(observed)
    assert np.corrcoef(slopes[200:], true_slope[200:])[0, 1] > 0.8
    assert np.all(variances[10:] > 0)

    truncated_slopes, _ = model.filter(observed[:1500])
    np.testing.assert_allclose(truncated_slopes, slopes[:1500])  # no future data used


def test_ou_is_detected_and_recovered_only_when_mean_reversion_exists():
    rng = np.random.default_rng(7)
    b, theta, n = 0.98, math.log(100), 3000
    x = np.empty(n)
    x[0] = theta
    for t in range(1, n):
        x[t] = theta * (1 - b) + b * x[t - 1] + 0.01 * rng.normal()
    assert detect_mean_reversion(x, 0.05).is_stationary
    model = fit_ornstein_uhlenbeck(x, horizon=5)
    assert model.persistence == pytest.approx(b, abs=0.01)
    assert math.exp(model.long_run_log_price) == pytest.approx(100, rel=0.05)

    random_walk = math.log(100) + np.cumsum(rng.normal(0, 0.015, n))
    assert not detect_mean_reversion(random_walk, 0.05).is_stationary


def test_earnings_reaction_day_mapping_respects_the_close_weekends_and_timezones():
    trading_days = pd.bdate_range("2025-03-03", "2025-03-14")
    announcements = pd.DatetimeIndex(
        [
            pd.Timestamp("2025-03-03 10:00", tz="Asia/Kolkata"),  # during market hours -> same day
            pd.Timestamp("2025-03-05 16:00", tz="Asia/Kolkata"),  # after close -> next day (6th)
            pd.Timestamp("2025-03-08 11:00", tz="Asia/Kolkata"),  # Saturday -> Monday (10th)
            pd.Timestamp("2025-03-11 07:00", tz="America/New_York").tz_convert("Asia/Kolkata"),  # = 17:30 IST -> 12th
        ]
    )
    reaction = earnings_reaction_days(announcements, trading_days, exchange_timezone="Asia/Kolkata", market_close_time="15:30")
    assert list(reaction.strftime("%m-%d")) == ["03-03", "03-06", "03-10", "03-12"]

    # Yahoo delivers Indian announcement times in New York time; they must be converted, not read as IST.
    new_york_times = pd.DatetimeIndex([pd.Timestamp("2025-03-11 07:00")]).tz_localize("America/New_York")
    reaction = earnings_reaction_days(new_york_times, trading_days, exchange_timezone="Asia/Kolkata", market_close_time="15:30")
    assert list(reaction.strftime("%m-%d")) == ["03-12"]


def test_earnings_variance_multiplier_and_horizon_scaling():
    rng = np.random.default_rng(8)
    days = pd.bdate_range("2015-01-01", periods=2500)
    returns = pd.Series(rng.normal(0, 0.01, len(days)), index=days)
    reaction_days = days[::63]
    returns[reaction_days] = rng.normal(0, 0.03, len(reaction_days))  # 9x variance on results days

    multiplier = estimate_earnings_variance_multiplier(returns, reaction_days, min_events=8)
    assert multiplier == pytest.approx(9.0, rel=0.35)
    assert estimate_earnings_variance_multiplier(returns, reaction_days[:3], min_events=8) is None

    window = days[60:65]
    scale = horizon_shock_multipliers(window, reaction_days, 9.0)
    assert list(scale) == [1.0, 1.0, 1.0, 3.0, 1.0]


def test_price_band_caps_every_simulated_daily_move():
    model = fit_gbm(to_log_prices(np.random.default_rng(9).normal(0, 0.05, 500)))
    totals = simulate_cumulative_log_returns(model, 5, 10_000, np.random.default_rng(0), price_band=0.05)
    assert totals.max() <= 5 * math.log(1.05) + 1e-12
    assert totals.min() >= 5 * math.log(0.95) - 1e-12
