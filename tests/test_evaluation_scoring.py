import math

import numpy as np
import pytest

from finance_engine.evaluation.scoring import continuous_ranked_probability_score, score_forecast
from finance_engine.stats.distributions import normal_cdf
from finance_engine.stats.hypothesis_tests import ks_test_uniform

RNG = np.random.default_rng(3)


def analytic_normal_crps(mean, std, y):
    z = (y - mean) / std
    pdf = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    return std * (z * (2 * float(normal_cdf(z)) - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


@pytest.mark.parametrize("realized", [-2.0, 0.0, 0.7, 3.5])
def test_sample_crps_matches_closed_form_for_normal(realized):
    samples = np.sort(RNG.normal(0.1, 1.3, 200_000))
    assert continuous_ranked_probability_score(samples, realized) == pytest.approx(analytic_normal_crps(0.1, 1.3, realized), abs=0.01)


def test_correct_model_is_calibrated_and_beats_an_overdispersed_one():
    levels = (0.5, 0.9)
    correct, too_wide = [], []
    for _ in range(800):
        realized = RNG.normal()
        correct.append(score_forecast(RNG.normal(size=4000), realized, levels))
        too_wide.append(score_forecast(RNG.normal(scale=2.0, size=4000), realized, levels))

    assert ks_test_uniform([s.pit for s in correct]).p_value > 0.01
    assert np.mean([s.interval_hits[0.9] for s in correct]) == pytest.approx(0.9, abs=0.035)
    assert np.mean([s.interval_hits[0.9] for s in too_wide]) > 0.97
    assert np.mean([s.crps for s in correct]) < np.mean([s.crps for s in too_wide])


def test_probability_up_and_brier():
    score = score_forecast(np.array([-1.0, 1.0, 2.0, 3.0]), realized=0.5, interval_levels=(0.5,))
    assert score.probability_up == 0.75
    assert score.brier == pytest.approx(0.0625)
