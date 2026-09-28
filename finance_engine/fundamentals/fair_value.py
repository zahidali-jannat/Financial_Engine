"""Combine fundamental components into one fair-value bias, and turn it into a prior drift tilt.

    bias = Σ w_c · score_c / Σ w_c      over AVAILABLE components (weights from settings)
    label: undervalued if bias > band, overvalued if bias < −band, else fairly valued
    confidence = coverage × agreement
        coverage  = share of total weight that was available
        agreement = 1 − (spread of component scores / 2)   (components pointing different ways → low)
    prior tilt (annual) = bias × confidence × max_annual_tilt   (a weak, conflicted rating tilts less)

How much this matters, honestly: at max_annual_tilt = 3%/yr, even a maximally
cheap rating moves the 5-day expected return by ~0.06% against a 5-day standard
deviation of ~3-4%. Valuation is a slow force; over days it barely registers.
It also can't be walk-forward validated with free data (no point-in-time
history for most inputs), so it enters only as a small, clearly labelled prior.
"""

from dataclasses import dataclass

import numpy as np

from finance_engine.fundamentals.valuation import ComponentScore


@dataclass(frozen=True)
class FairValueAssessment:
    bias: float  # + = undervalued
    label: str
    confidence: float  # 0-1
    confidence_label: str
    components: dict  # name -> ComponentScore
    annual_drift_tilt: float


def assess_fair_value(components: dict[str, ComponentScore], fundamentals_settings: dict) -> FairValueAssessment:
    weights = fundamentals_settings["component_weights"]
    available = {name: component for name, component in components.items() if component.score is not None}
    if not available:
        return FairValueAssessment(0.0, "not assessable", 0.0, "none", components, 0.0)

    total_weight = sum(weights[name] for name in available)
    bias = sum(weights[name] * component.score for name, component in available.items()) / total_weight
    coverage = total_weight / sum(weights.values())
    scores = [component.score for component in available.values()]
    agreement = 1 - (max(scores) - min(scores)) / 2 if len(scores) > 1 else 0.5
    confidence = coverage * agreement

    band = fundamentals_settings["fair_value_band"]
    label = "undervalued" if bias > band else "overvalued" if bias < -band else "fairly valued"
    confidence_label = "high" if confidence >= 0.6 else "medium" if confidence >= 0.35 else "low"
    return FairValueAssessment(
        bias=float(np.clip(bias, -1, 1)),
        label=label,
        confidence=float(confidence),
        confidence_label=confidence_label,
        components=components,
        annual_drift_tilt=float(np.clip(bias, -1, 1) * fundamentals_settings["max_annual_tilt"] * confidence),
    )
