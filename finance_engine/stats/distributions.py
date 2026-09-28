"""Probability distributions used by the models: normal and standardized Student-t, plus chi-square tails."""

import math

import numpy as np

_erfc = np.vectorize(math.erfc, otypes=[float])
LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)


def normal_cdf(x):
    """Φ(x), elementwise."""
    return 0.5 * _erfc(-np.asarray(x, float) / math.sqrt(2))


def normal_logpdf(x, mean=0.0, std=1.0):
    z = (np.asarray(x, float) - mean) / std
    return -0.5 * z**2 - np.log(std) - LOG_SQRT_2PI


def standardized_t_logpdf(z, degrees_of_freedom: float):
    """Log-density of a Student-t rescaled to unit variance (requires ν > 2).

    GARCH needs innovations with variance exactly 1 so that σ²_t is the true
    conditional variance; a plain t(ν) has variance ν/(ν−2), hence the rescale.
    """
    nu = degrees_of_freedom
    log_constant = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
    return log_constant - (nu + 1) / 2 * np.log1p(np.asarray(z, float) ** 2 / (nu - 2))


def sample_standardized_t(rng: np.random.Generator, degrees_of_freedom: float, size) -> np.ndarray:
    nu = degrees_of_freedom
    return rng.standard_t(nu, size=size) * math.sqrt((nu - 2) / nu)


def chi_square_survival(x: float, degrees_of_freedom: int) -> float:
    """P(χ²_k > x), via the regularized upper incomplete gamma function Q(k/2, x/2)."""
    if x <= 0:
        return 1.0
    return _regularized_upper_gamma(degrees_of_freedom / 2, x / 2)


def _regularized_upper_gamma(a: float, x: float) -> float:
    # Series for x < a+1, continued fraction otherwise (Numerical Recipes §6.2).
    log_prefactor = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1:
        term = total = 1.0 / a
        denominator = a
        for _ in range(1000):
            denominator += 1
            term *= x / denominator
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(log_prefactor))
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return min(1.0, math.exp(log_prefactor) * h)
