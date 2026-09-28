"""Score one probabilistic forecast (a sample of simulated outcomes) against the realized outcome.

Scores used and why:
    CRPS  — proper scoring rule for the whole distribution; rewards both being
            close to the outcome and being appropriately (not over-) confident.
            Same units as the outcome (log return), lower is better.
    PIT   — where the outcome fell inside the forecast distribution, F(y).
            Uniform over many forecasts ⇔ the forecasts are calibrated.
    Interval hits — did the outcome land inside each central interval?
            Over many forecasts, a 90% interval should contain ~90% of outcomes.
    Brier — squared error of the probability of an up move; scores the
            directional claim separately from the distribution's shape.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForecastScore:
    crps: float
    pit: float
    probability_up: float
    brier: float
    interval_hits: dict  # level -> bool


def continuous_ranked_probability_score(sorted_samples: np.ndarray, realized: float) -> float:
    """CRPS = E|X − y| − ½·E|X − X'| estimated from a sorted sample of X."""
    n = len(sorted_samples)
    distance_to_outcome = np.mean(np.abs(sorted_samples - realized))
    # E|X − X'| for sorted samples = (2/n²) Σ (2i − n − 1)·x_(i), an O(n) form of the pairwise mean.
    weights = 2 * np.arange(1, n + 1) - n - 1
    mean_pairwise_distance = 2 * np.dot(weights, sorted_samples) / n**2
    return float(distance_to_outcome - 0.5 * mean_pairwise_distance)


def probability_integral_transform(sorted_samples: np.ndarray, realized: float) -> float:
    """Fraction of simulated outcomes below the realized one (ties counted as half)."""
    below = np.searchsorted(sorted_samples, realized, side="left")
    at_or_below = np.searchsorted(sorted_samples, realized, side="right")
    return float((below + at_or_below) / (2 * len(sorted_samples)))


def central_interval(sorted_samples: np.ndarray, level: float) -> tuple[float, float]:
    """The interval holding the middle `level` share of simulated outcomes."""
    tail = (1 - level) / 2
    return float(np.quantile(sorted_samples, tail)), float(np.quantile(sorted_samples, 1 - tail))


def score_forecast(samples: np.ndarray, realized: float, interval_levels) -> ForecastScore:
    sorted_samples = np.sort(np.asarray(samples, float))
    probability_up = float(np.mean(sorted_samples > 0))
    went_up = float(realized > 0)
    hits = {}
    for level in interval_levels:
        low, high = central_interval(sorted_samples, level)
        hits[level] = bool(low <= realized <= high)
    return ForecastScore(
        crps=continuous_ranked_probability_score(sorted_samples, realized),
        pit=probability_integral_transform(sorted_samples, realized),
        probability_up=probability_up,
        brier=(probability_up - went_up) ** 2,
        interval_hits=hits,
    )
