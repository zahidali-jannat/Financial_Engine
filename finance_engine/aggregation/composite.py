"""Category composites: average the kept indicators WITHIN each category first.

Averaging within a category before anything else means a category with four
look-alike oscillators gets one vote, not four. The composites (trend,
momentum, volatility, volume) are the technical evidence handed to Layer 4.
"""

import pandas as pd

CATEGORIES = ("trend", "momentum", "volatility", "volume")


def category_composites(normalized: pd.DataFrame, kept: list[str], categories: dict[str, str]) -> pd.DataFrame:
    """One column per category: mean of that category's kept indicators (NaN if a category has none yet)."""
    composites = {}
    for category in CATEGORIES:
        members = [name for name in kept if categories.get(name) == category]
        composites[category] = normalized[members].mean(axis=1, skipna=False) if members else pd.Series(float("nan"), index=normalized.index)
    return pd.DataFrame(composites, index=normalized.index)
