"""Where does the engine do worst? Skill and calibration broken down by market condition.

Segments (each known at the forecast date, so the breakdown itself has no lookahead):
    volatility regime  — HMM filtered P(turbulent) > 0.5 vs ≤ 0.5
    trend regime       — ADX trending vs ranging
    earnings           — a results reaction day inside the forecast window vs not
    year               — calendar year of the forecast
"""

import numpy as np
import pandas as pd

TAIL_LEVEL = 0.95


def segment_performance(records: pd.DataFrame, model: str = "engine", benchmark: str = "gbm") -> pd.DataFrame:
    segments = {
        "turbulent regime (HMM)": records["turbulent_probability"] > 0.5,
        "calm regime (HMM)": records["turbulent_probability"] <= 0.5,
        "trending (ADX)": records["trending"] == 1,
        "ranging (ADX)": records["trending"] == 0,
        "earnings in window": records["earnings_in_window"],
        "no earnings in window": ~records["earnings_in_window"],
    }
    for year in sorted(set(records.index.year)):
        segments[str(year)] = records.index.year == year

    rows = []
    for label, mask in segments.items():
        subset = records[np.asarray(mask, bool)]
        if len(subset) < 5:
            continue
        rows.append(
            {
                "segment": label,
                "forecasts": len(subset),
                "crps_skill_vs_gbm": float(1 - subset[f"{model}_crps"].mean() / subset[f"{benchmark}_crps"].mean()),
                "coverage_0.9": float(subset[f"{model}_hit_0.9"].mean()),
                "brier_skill_vs_gbm": float(1 - subset[f"{model}_brier"].mean() / subset[f"{benchmark}_brier"].mean()),
            }
        )
    return pd.DataFrame(rows).set_index("segment")


def worst_misses(records: pd.DataFrame, model: str = "engine", count: int = 5) -> pd.DataFrame:
    """Forecasts whose outcome fell furthest outside the forecast distribution (PIT nearest 0 or 1)."""
    extremeness = (records[f"{model}_pit"] - 0.5).abs()
    worst = records.loc[extremeness.sort_values(ascending=False).index[:count]]
    return pd.DataFrame(
        {
            "realized_return": np.expm1(worst["realized"]),
            "pit": worst[f"{model}_pit"],
            "turbulent_probability": worst["turbulent_probability"],
            "earnings_in_window": worst["earnings_in_window"],
        }
    )


def tail_miss_rate(records: pd.DataFrame, model: str = "engine") -> float:
    """Share of outcomes outside the 95% interval (should be ~5%)."""
    return float(1 - records[f"{model}_hit_{TAIL_LEVEL}"].mean())
