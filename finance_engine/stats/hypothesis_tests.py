"""Hypothesis tests used to check model assumptions and forecast quality.

    jarque_bera            — are returns normal? (GBM assumes yes)
    arch_lm_test           — does volatility cluster? (the reason GARCH exists)
    augmented_dickey_fuller — is the series mean-reverting? (OU requires yes)
    ks_test_uniform        — are PIT values uniform? (calibration of a forecast)
    diebold_mariano        — is one forecast's loss reliably lower than another's?
"""

import math
from dataclasses import dataclass

import numpy as np

from finance_engine.stats.distributions import chi_square_survival, normal_cdf

# MacKinnon (2010) response-surface coefficients for ADF with a constant, no trend:
# critical value(T) = b0 + b1/T + b2/T² + b3/T³.
ADF_CRITICAL_VALUE_COEFFICIENTS = {
    0.01: (-3.43035, -6.5393, -16.786, -79.433),
    0.05: (-2.86154, -2.8903, -4.234, -40.040),
    0.10: (-2.56677, -1.5384, -2.809, 0.0),
}


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float


@dataclass(frozen=True)
class AdfResult:
    statistic: float
    lags: int
    observations: int
    critical_values: dict
    significance: float
    is_stationary: bool


@dataclass(frozen=True)
class DieboldMarianoResult:
    mean_loss_difference: float  # negative = the model's loss is lower than the benchmark's
    statistic: float
    p_value: float  # two-sided
    observations: int


def jarque_bera(values) -> TestResult:
    """JB = n/6 · (S² + K²/4), where S is skewness and K excess kurtosis; χ²(2) under normality."""
    x = np.asarray(values, float)
    n = len(x)
    z = (x - x.mean()) / x.std()
    skewness = float(np.mean(z**3))
    excess_kurtosis = float(np.mean(z**4) - 3)
    statistic = n / 6 * (skewness**2 + excess_kurtosis**2 / 4)
    return TestResult(statistic, math.exp(-statistic / 2))  # χ²(2) survival is exactly e^(−x/2)


def arch_lm_test(residuals, lags: int) -> TestResult:
    """Engle's test: regress ε²_t on ε²_{t−1..t−q}; n·R² ~ χ²(q) if there is no volatility clustering."""
    squared = np.asarray(residuals, float) ** 2
    target = squared[lags:]
    design = np.column_stack([np.ones(len(target))] + [squared[lags - k : len(squared) - k] for k in range(1, lags + 1)])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ coefficients
    r_squared = 1 - np.sum((target - fitted) ** 2) / np.sum((target - target.mean()) ** 2)
    statistic = len(target) * r_squared
    return TestResult(float(statistic), chi_square_survival(statistic, lags))


def augmented_dickey_fuller(series, *, significance: float = 0.05, max_lags: int | None = None) -> AdfResult:
    """ADF with a constant: Δy_t = a + γ·y_{t−1} + Σ φ_i·Δy_{t−i} + e_t, testing H0: γ = 0 (unit root).

    Rejecting H0 (statistic below the critical value) is evidence of mean
    reversion. Lag count is chosen by AIC over a common sample, as in
    statsmodels' default. Only the 1%/5%/10% levels are supported because the
    test uses tabulated critical values rather than a p-value.
    """
    if significance not in ADF_CRITICAL_VALUE_COEFFICIENTS:
        raise ValueError(f"significance must be one of {sorted(ADF_CRITICAL_VALUE_COEFFICIENTS)}")
    y = np.asarray(series, float)
    dy = np.diff(y)
    max_lags = int(12 * (len(y) / 100) ** 0.25) if max_lags is None else max_lags

    def design_for(lags: int, start: int):
        rows = np.arange(start, len(dy))
        columns = [np.ones(len(rows)), y[rows]] + [dy[rows - k] for k in range(1, lags + 1)]
        return np.column_stack(columns), dy[rows]

    best_lags, best_aic = 0, np.inf
    for lags in range(max_lags + 1):
        design, target = design_for(lags, max_lags)
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual_variance = np.mean((target - design @ coefficients) ** 2)
        aic = len(target) * math.log(residual_variance) + 2 * design.shape[1]
        if aic < best_aic:
            best_lags, best_aic = lags, aic

    design, target = design_for(best_lags, best_lags)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residuals = target - design @ coefficients
    sigma_squared = residuals @ residuals / (len(target) - design.shape[1])
    covariance = sigma_squared * np.linalg.inv(design.T @ design)
    statistic = float(coefficients[1] / math.sqrt(covariance[1, 1]))

    n = len(target)
    critical_values = {level: sum(b / n**i for i, b in enumerate(coefs)) for level, coefs in ADF_CRITICAL_VALUE_COEFFICIENTS.items()}
    return AdfResult(statistic, best_lags, n, critical_values, significance, statistic < critical_values[significance])


def ks_test_uniform(values) -> TestResult:
    """Kolmogorov–Smirnov test against Uniform(0,1), asymptotic p-value with Stephens' small-sample correction."""
    u = np.sort(np.asarray(values, float))
    n = len(u)
    ranks = np.arange(1, n + 1)
    distance = float(max(np.max(ranks / n - u), np.max(u - (ranks - 1) / n)))
    scaled = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * distance
    p_value = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * scaled * scaled) for k in range(1, 101))
    return TestResult(distance, float(min(1.0, max(0.0, p_value))))


def diebold_mariano(model_losses, benchmark_losses, *, max_lag: int) -> DieboldMarianoResult:
    """Test H0: equal expected loss, using a Newey–West (Bartlett) long-run variance.

    Applies the Harvey–Leybourne–Newbold small-sample correction and a normal
    reference distribution (with hundreds of forecasts, t and normal coincide).
    """
    d = np.asarray(model_losses, float) - np.asarray(benchmark_losses, float)
    n = len(d)
    mean_difference = float(d.mean())
    centered = d - mean_difference
    long_run_variance = centered @ centered / n
    for lag in range(1, max_lag + 1):
        weight = 1 - lag / (max_lag + 1)
        long_run_variance += 2 * weight * (centered[lag:] @ centered[:-lag]) / n
    if long_run_variance <= 0:
        return DieboldMarianoResult(mean_difference, 0.0, 1.0, n)

    statistic = mean_difference / math.sqrt(long_run_variance / n)
    horizon = max_lag + 1
    statistic *= math.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    p_value = float(2 * (1 - normal_cdf(abs(statistic))))
    return DieboldMarianoResult(mean_difference, float(statistic), p_value, n)
