import numpy as np
import pandas as pd
import pytest

from finance_engine.synthesis.bayesian_drift import drift_prior, estimate_drift_posterior

H = 5


def make_data(n, signal_effect, seed):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2015-01-01", periods=n)
    signal = pd.Series(rng.normal(size=n), index=index)
    forward = pd.Series(0.002 + signal_effect * signal + rng.normal(0, 0.03, n), index=index)
    return pd.DataFrame({"trend": signal}), forward


def posterior(evidence, forward, origin, prior_mean=0.002, prior_std=0.004, tau=0.0025):
    return estimate_drift_posterior(evidence, forward, origin, horizon=H, prior_mean=prior_mean, prior_std=prior_std, signal_effect_prior_sd=tau)


def test_without_enough_history_the_posterior_is_the_prior():
    evidence, forward = make_data(60, 0.0, seed=1)
    result = posterior(evidence, forward, origin=50)
    assert (result.mean, result.std) == (0.002, 0.004)


def test_a_useless_signal_gets_shrunk_to_near_zero_effect():
    evidence, forward = make_data(8000, 0.0, seed=2)
    result = posterior(evidence, forward, origin=7990)
    coefficient, coefficient_std = result.coefficients["trend"]
    assert abs(coefficient) < 2.5 * coefficient_std
    assert abs(coefficient) < 0.002


def test_a_real_signal_is_learned_and_shifts_the_forecast():
    evidence, forward = make_data(8000, 0.01, seed=3)  # 1-sd signal moves the h-day return by 1%
    result = posterior(evidence, forward, origin=7990)
    assert result.coefficients["trend"][0] == pytest.approx(0.01, abs=0.002)
    expected = result.contributions["baseline"] + result.contributions["trend"]
    assert result.mean == pytest.approx(expected)


def test_only_already_realized_outcomes_are_used():
    evidence, forward = make_data(3000, 0.01, seed=4)
    origin = 2500
    poisoned = forward.copy()
    poisoned.iloc[origin - H + 1 :] = 1e6  # outcomes that ended after the origin must be invisible
    assert posterior(evidence, poisoned, origin).mean == pytest.approx(posterior(evidence, forward, origin).mean)


def test_drift_prior_scales_annual_settings_to_the_horizon():
    mean, std = drift_prior({"prior_annual_drift": 0.10, "prior_annual_drift_sd": 0.20}, horizon=5, trading_days_per_year=250, fundamental_annual_tilt=0.02)
    assert mean == pytest.approx(0.12 * 5 / 250)
    assert std == pytest.approx(0.20 * 5 / 250)
