"""Find and drop redundant indicators so correlated ones aren't double-counted.

Pairwise Pearson correlation of the normalized indicators is measured on the
TRAINING window only. Walking the configured priority order, an indicator is
kept unless it correlates above the threshold with one already kept — so
between RSI and Stochastic (typically ρ > 0.8), the higher-priority one stays.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RedundancyDecision:
    kept: list[str]
    dropped: dict[str, tuple[str, float]]  # dropped indicator -> (indicator it duplicates, correlation)
    correlation: pd.DataFrame


def prune_redundant_indicators(normalized_training: pd.DataFrame, priority: list[str], max_abs_correlation: float) -> RedundancyDecision:
    ordered = [name for name in priority if name in normalized_training.columns]
    correlation = normalized_training[ordered].corr()
    kept, dropped = [], {}
    for name in ordered:
        duplicates = [(other, correlation.loc[name, other]) for other in kept if abs(correlation.loc[name, other]) > max_abs_correlation]
        if duplicates:
            dropped[name] = max(duplicates, key=lambda pair: abs(pair[1]))
        else:
            kept.append(name)
    return RedundancyDecision(kept, {k: (v[0], float(v[1])) for k, v in dropped.items()}, correlation)
