"""Cross-check the hand-written statistics and models against reference libraries.

The engine itself needs only numpy/pandas; these tests use scipy, statsmodels and
arch purely as independent referees, and are skipped if those aren't installed.
"""

import math

import numpy as np
import pandas as pd
import pytest

stats = pytest.importorskip("scipy.stats")
optimize = pytest.importorskip("scipy.optimize")
sm = pytest.importorskip("statsmodels.api")
arch = pytest.importorskip("arch")

# statsmodels announces a future change to its return types; irrelevant to these comparisons.
pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

from statsmodels.stats.diagnostic import het_arch  # noqa: E402
from statsmodels.tsa.stattools import adfuller  # noqa: E402

from finance_engine.models.garch import fit_garch  # noqa: E402
from finance_engine.models.hmm import fit_hmm  # noqa: E402
from finance_engine.models.kalman import fit_kalman_trend  # noqa: E402
from finance_engine.stats.distributions import chi_square_survival, normal_cdf, standardized_t_logpdf  # noqa: E402
from finance_engine.stats.hypothesis_tests import arch_lm_test, augmented_dickey_fuller, jarque_bera, ks_test_uniform  # noqa: E402
from finance_engine.stats.optimize import minimize_nelder_mead  # noqa: E402
from tests.test_models import simulate_gjr_garch_t, to_log_prices  # noqa: E402

RNG = np.random.default_rng(2024)


def test_distributions_match_scipy_to_machine_precision():
    z = np.linspace(-6, 6, 25)
    nu = 5.3
    scipy_standardized_t = stats.t.logpdf(z * math.sqrt(nu / (nu - 2)), nu) + 0.5 * math.log(nu / (nu - 2))
    np.testing.assert_allclose(standardized_t_logpdf(z, nu), scipy_standardized_t, atol=1e-12)
    np.testing.assert_allclose(normal_cdf(z), stats.norm.cdf(z), atol=1e-14)
    for x, df in [(3.84, 1), (11.07, 5), (25.3, 7), (0.5, 3), (80.0, 12)]:
        assert chi_square_survival(x, df) == pytest.approx(stats.chi2.sf(x, df), rel=1e-9, abs=1e-15)


def test_hypothesis_tests_match_scipy_and_statsmodels():
    returns = simulate_gjr_garch_t(2500, 0.03, 0.05, 0.06, 0.88, 5.0, seed=11)
    log_prices = to_log_prices(returns)

    mine, reference = augmented_dickey_fuller(log_prices), adfuller(log_prices, regression="c", autolag="AIC")
    assert mine.statistic == pytest.approx(reference[0], rel=1e-9)
    assert mine.lags == reference[2]
    assert mine.critical_values[0.05] == pytest.approx(reference[4]["5%"], abs=1e-4)

    assert jarque_bera(returns).statistic == pytest.approx(stats.jarque_bera(returns).statistic, rel=1e-9)
    demeaned = returns - returns.mean()
    assert arch_lm_test(demeaned, 5).statistic == pytest.approx(het_arch(demeaned, nlags=5)[0], rel=1e-9)

    u = RNG.uniform(size=1500) ** 1.05
    assert ks_test_uniform(u).statistic == pytest.approx(stats.kstest(u, "uniform").statistic, rel=1e-12)
    assert ks_test_uniform(u).p_value == pytest.approx(stats.kstest(u, "uniform").pvalue, abs=0.02)  # asymptotic vs exact


def test_nelder_mead_matches_scipy():
    rosenbrock = lambda p: float(np.sum(100 * (p[1:] - p[:-1] ** 2) ** 2 + (1 - p[:-1]) ** 2))
    mine = minimize_nelder_mead(rosenbrock, np.zeros(4), max_iterations=20000, tolerance=1e-14)
    reference = optimize.minimize(rosenbrock, np.zeros(4), method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-14, "adaptive": True})
    np.testing.assert_allclose(mine.parameters, reference.x, atol=1e-4)


def test_gjr_garch_t_matches_the_arch_package():
    returns = simulate_gjr_garch_t(3000, 0.05, 0.04, 0.07, 0.88, 6.0, seed=12)
    mine = fit_garch(to_log_prices(returns), asymmetric=True, max_iterations=4000)
    reference = arch.arch_model(100 * (returns - returns.mean()), mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="StudentsT").fit(disp="off")
    params = reference.params
    assert mine.alpha == pytest.approx(params["alpha[1]"], abs=0.01)
    assert mine.gamma == pytest.approx(params["gamma[1]"], abs=0.015)
    assert mine.beta == pytest.approx(params["beta[1]"], abs=0.015)
    assert mine.degrees_of_freedom == pytest.approx(params["nu"], abs=0.5)
    # Only the variance recursion's starting value differs, so the optima should be within a fraction of a log-likelihood unit.
    assert mine.log_likelihood == pytest.approx(reference.loglikelihood, abs=1.0)
    reference_next_variance = reference.forecast(horizon=1).variance.to_numpy()[-1, 0]
    assert mine.next_variance == pytest.approx(reference_next_variance, rel=0.02)


def test_hmm_matches_statsmodels_markov_switching():
    rng = np.random.default_rng(13)
    states, vols = np.zeros(3000, int), np.array([0.009, 0.03])
    for t in range(1, 3000):
        states[t] = states[t - 1] if rng.uniform() < (0.985, 0.93)[states[t - 1]] else 1 - states[t - 1]
    returns = vols[states] * rng.normal(size=3000)

    mine = fit_hmm(to_log_prices(returns), n_states=2, max_iterations=500, tolerance=1e-9, random_restarts=4, horizon=5, rng=np.random.default_rng(0))
    reference = sm.tsa.MarkovRegression(returns, k_regimes=2, trend="c", switching_variance=True).fit(search_reps=20, disp=False)
    params = pd.Series(reference.params, index=reference.model.param_names)
    reference_vols = np.sort(np.sqrt([params["sigma2[0]"], params["sigma2[1]"]]))
    np.testing.assert_allclose(np.sqrt(mine.state_variances), reference_vols, rtol=0.01)
    assert mine.log_likelihood == pytest.approx(reference.llf, abs=0.5)


def test_kalman_trend_matches_statsmodels_unobserved_components():
    rng = np.random.default_rng(14)
    slope = np.cumsum(rng.normal(0, 2e-5, 2500))
    log_prices = np.cumsum(slope + rng.normal(0, 0.012, 2500)) + rng.normal(0, 0.002, 2500)

    mine = fit_kalman_trend(log_prices, max_iterations=800)
    reference = sm.tsa.UnobservedComponents(log_prices, level="local linear trend").fit(disp=False)
    params = pd.Series(reference.params, index=reference.model.param_names)
    assert mine.level_variance == pytest.approx(params["sigma2.level"], rel=0.05)
    my_slope, _ = mine.filter(log_prices)
    assert np.corrcoef(my_slope[200:], reference.filtered_state[1][200:])[0, 1] > 0.99
