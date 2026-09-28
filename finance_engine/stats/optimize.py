"""Derivative-free minimisation (Nelder–Mead) for the maximum-likelihood fits.

The model likelihoods (GARCH, Merton, Kalman, OU) have 2–5 parameters and are
optimised in an unconstrained space — each model maps real numbers to valid
parameters (exp for positives, logistic for fractions) — so a simplex method is
enough. The adaptive coefficients of Gao & Han (2012) keep it well behaved as
the dimension grows, and a restart from the best point guards against the
simplex collapsing early, a known Nelder–Mead failure mode.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class OptimizationResult:
    parameters: np.ndarray
    objective_value: float
    iterations: int
    converged: bool


def minimize_nelder_mead(
    objective: Callable[[np.ndarray], float],
    initial_parameters: np.ndarray,
    *,
    initial_step: float = 0.25,
    max_iterations: int = 2000,
    tolerance: float = 1e-8,
    restarts: int = 1,
) -> OptimizationResult:
    """Minimise `objective`; non-finite objective values are treated as +infinity."""
    best = _run_simplex(objective, np.asarray(initial_parameters, float), initial_step, max_iterations, tolerance)
    for _ in range(restarts):
        restarted = _run_simplex(objective, best.parameters, initial_step / 2, max_iterations, tolerance)
        improved = restarted.objective_value < best.objective_value - tolerance
        best = restarted if restarted.objective_value <= best.objective_value else best
        if not improved:
            break
    return best


def _safe(objective: Callable[[np.ndarray], float], x: np.ndarray) -> float:
    try:
        value = float(objective(x))
    except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError):
        return np.inf
    return value if np.isfinite(value) else np.inf


def _run_simplex(objective, x0, initial_step, max_iterations, tolerance) -> OptimizationResult:
    n = len(x0)
    reflect, expand, contract, shrink = 1.0, 1.0 + 2.0 / n, 0.75 - 1.0 / (2 * n), 1.0 - 1.0 / n

    simplex = np.vstack([x0] + [x0 + initial_step * np.eye(n)[i] for i in range(n)])
    values = np.array([_safe(objective, vertex) for vertex in simplex])

    for iteration in range(1, max_iterations + 1):
        order = np.argsort(values)
        simplex, values = simplex[order], values[order]

        value_spread = np.abs(values[-1] - values[0]) if np.isfinite(values[-1]) else np.inf
        simplex_size = np.max(np.abs(simplex[1:] - simplex[0]))
        if value_spread <= tolerance * (1 + abs(values[0])) and simplex_size <= np.sqrt(tolerance):
            return OptimizationResult(simplex[0], float(values[0]), iteration, True)

        centroid = simplex[:-1].mean(axis=0)
        reflected = centroid + reflect * (centroid - simplex[-1])
        reflected_value = _safe(objective, reflected)

        if values[0] <= reflected_value < values[-2]:
            simplex[-1], values[-1] = reflected, reflected_value
        elif reflected_value < values[0]:
            expanded = centroid + expand * (reflected - centroid)
            expanded_value = _safe(objective, expanded)
            if expanded_value < reflected_value:
                simplex[-1], values[-1] = expanded, expanded_value
            else:
                simplex[-1], values[-1] = reflected, reflected_value
        else:
            if reflected_value < values[-1]:
                contracted = centroid + contract * (reflected - centroid)
            else:
                contracted = centroid + contract * (simplex[-1] - centroid)
            contracted_value = _safe(objective, contracted)
            if contracted_value < min(reflected_value, values[-1]):
                simplex[-1], values[-1] = contracted, contracted_value
            else:
                simplex[1:] = simplex[0] + shrink * (simplex[1:] - simplex[0])
                values[1:] = [_safe(objective, vertex) for vertex in simplex[1:]]

    order = np.argsort(values)
    return OptimizationResult(simplex[order[0]], float(values[order[0]]), max_iterations, False)
