"""Out-of-sample comparison of every model against the GBM null model.

For each model, on the forecast dates where it existed:
    CRPS skill      = 1 − CRPS_model / CRPS_gbm    (> 0: better distribution than GBM)
    DM p-value      = Diebold–Mariano test of that CRPS difference (is it more than noise?)
    PIT KS p-value  = are outcomes uniformly spread through the forecast? (calibration)
    coverage        = realized share inside each central interval vs its nominal level
    Brier skill     = 1 − Brier_model / Brier_gbm   for P(up)  (directional claims only)
    Brier DM p      = is the directional improvement more than noise?

Forecasts are spaced `horizon` days apart, so outcome windows don't overlap; the
DM test still allows one lag of autocorrelation for safety.
"""

import numpy as np
import pandas as pd

from finance_engine.stats.hypothesis_tests import diebold_mariano, ks_test_uniform

BENCHMARK = "gbm"


def compare_models(records: pd.DataFrame, model_names: list[str], interval_levels: list[float]) -> pd.DataFrame:
    rows = []
    for name in model_names:
        present = records[f"{name}_crps"].notna()
        subset = records[present]
        crps, benchmark_crps = subset[f"{name}_crps"], subset[f"{BENCHMARK}_crps"]
        brier, benchmark_brier = subset[f"{name}_brier"], subset[f"{BENCHMARK}_brier"]
        row = {
            "model": name,
            "forecasts": int(present.sum()),
            "mean_crps": float(crps.mean()),
            "crps_skill_vs_gbm": float(1 - crps.mean() / benchmark_crps.mean()),
            "crps_dm_p_value": np.nan,
            "pit_ks_p_value": ks_test_uniform(subset[f"{name}_pit"]).p_value,
            "brier": float(brier.mean()),
            "brier_skill_vs_gbm": float(1 - brier.mean() / benchmark_brier.mean()),
            "brier_dm_p_value": np.nan,
        }
        if name != BENCHMARK:
            row["crps_dm_p_value"] = diebold_mariano(crps, benchmark_crps, max_lag=1).p_value
            row["brier_dm_p_value"] = diebold_mariano(brier, benchmark_brier, max_lag=1).p_value
        for level in interval_levels:
            row[f"coverage_{level}"] = float(subset[f"{name}_hit_{level}"].mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def coverage_tolerance(level: float, n: int) -> float:
    """Two binomial standard errors: how far observed coverage may drift from nominal by chance alone."""
    return 2 * np.sqrt(level * (1 - level) / n)


def base_rate_brier(records: pd.DataFrame) -> float:
    """Brier score of always forecasting the historical up-frequency known at each date (an honest naive baseline)."""
    went_up = (records["realized"] > 0).astype(float)
    known_rate = went_up.shift(1).expanding().mean().fillna(0.5)
    return float(np.mean((known_rate - went_up) ** 2))
