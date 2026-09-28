import math

import numpy as np
import pytest

from finance_engine.stats.distributions import chi_square_survival, normal_cdf, sample_standardized_t, standardized_t_logpdf
from finance_engine.stats.hypothesis_tests import (
    arch_lm_test,
    augmented_dickey_fuller,
    diebold_mariano,
    jarque_bera,
    ks_test_uniform,
)
from finance_engine.stats.optimize import minimize_nelder_mead

RNG = np.random.default_rng(0)


def test_nelder_mead_finds_rosenbrock_minimum():
    rosenbrock = lambda p: (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2
    result = minimize_nelder_mead(rosenbrock, np.array([-1.2, 1.0]), max_iterations=5000, tolerance=1e-12)
    assert result.parameters == pytest.approx([1.0, 1.0], abs=1e-3)


def test_nelder_mead_handles_five_dimensions_and_infinite_regions():
    target = np.array([0.5, -1.0, 2.0, 0.0, 1.5])
    objective = lambda p: np.inf if p[0] < -5 else float(np.sum((p - target) ** 2))
    result = minimize_nelder_mead(objective, np.zeros(5), tolerance=1e-12)
    assert result.parameters == pytest.approx(target, abs=1e-4)


def test_normal_cdf_known_values():
    assert float(normal_cdf(0.0)) == pytest.approx(0.5)
    assert float(normal_cdf(1.959964)) == pytest.approx(0.975, abs=1e-6)


def test_standardized_t_is_a_unit_variance_density():
    grid = np.linspace(-60, 60, 400_001)
    density = np.exp(standardized_t_logpdf(grid, 5.0))
    assert np.trapezoid(density, grid) == pytest.approx(1.0, abs=1e-4)
    assert np.trapezoid(grid**2 * density, grid) == pytest.approx(1.0, abs=2e-3)
    assert np.var(sample_standardized_t(RNG, 5.0, 400_000)) == pytest.approx(1.0, abs=0.03)


def test_chi_square_survival_matches_tables():
    assert chi_square_survival(3.841459, 1) == pytest.approx(0.05, abs=1e-6)
    assert chi_square_survival(11.070498, 5) == pytest.approx(0.05, abs=1e-6)
    assert chi_square_survival(4.0, 2) == pytest.approx(math.exp(-2.0))
    assert chi_square_survival(40.0, 3) < 1e-7


def test_jarque_bera_accepts_normal_and_rejects_fat_tails():
    assert jarque_bera(RNG.normal(size=3000)).p_value > 0.01
    assert jarque_bera(RNG.standard_t(3, size=3000)).p_value < 1e-6


def simulate_garch(n, omega=0.05, alpha=0.1, beta=0.85, seed=1):
    rng = np.random.default_rng(seed)
    variance, out = omega / (1 - alpha - beta), np.empty(n)
    for t in range(n):
        out[t] = math.sqrt(variance) * rng.normal()
        variance = omega + alpha * out[t] ** 2 + beta * variance
    return out


def test_arch_lm_detects_volatility_clustering_only_when_present():
    rng = np.random.default_rng(11)
    false_alarm_rate = np.mean([arch_lm_test(rng.normal(size=1000), lags=5).p_value < 0.05 for _ in range(400)])
    assert false_alarm_rate == pytest.approx(0.05, abs=0.03)  # a correct test rejects true H0 ~5% of the time
    assert arch_lm_test(simulate_garch(3000), lags=5).p_value < 1e-4


def test_adf_distinguishes_random_walk_from_mean_reversion():
    random_walk = np.cumsum(RNG.normal(size=1000))
    assert not augmented_dickey_fuller(random_walk).is_stationary

    ar1 = np.zeros(1000)
    for t in range(1, 1000):
        ar1[t] = 0.9 * ar1[t - 1] + RNG.normal()
    result = augmented_dickey_fuller(ar1)
    assert result.is_stationary
    assert result.critical_values[0.05] == pytest.approx(-2.864, abs=0.005)


def test_ks_uniform():
    assert ks_test_uniform(RNG.uniform(size=2000)).p_value > 0.01
    assert ks_test_uniform(RNG.uniform(size=2000) ** 2).p_value < 1e-6


def test_diebold_mariano():
    benchmark = RNG.uniform(size=400)
    assert diebold_mariano(benchmark, benchmark, max_lag=1).p_value == 1.0
    better = benchmark - 0.1 + RNG.normal(0, 0.05, 400)
    result = diebold_mariano(better, benchmark, max_lag=1)
    assert result.mean_loss_difference < 0
    assert result.p_value < 1e-6
