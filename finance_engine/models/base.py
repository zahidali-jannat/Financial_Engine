"""Common interface for every return model, and the simulator that turns a model into forecast samples.

Contract (what Layer 4 and the backtest rely on):
    A fitted model exposes
      name                         — stable identifier, e.g. "garch_t"
      daily_drift                  — the model's own expected daily log return
      simulate_daily_shocks(h, n, rng) -> ndarray (n, h) of ZERO-MEAN daily log-return shocks
      with_data(log_prices)        — same parameters, state updated to the end of `log_prices`
                                     (filtering, not re-estimation; never sees beyond its input)
      describe() -> dict           — fitted parameters in human units, for the report

Splitting drift from shocks is deliberate: the stochastic model supplies the
shape (fat tails, clustering, regimes) while Layer 4 can swap in a Bayesian
drift without disturbing that shape.
"""

import math
from abc import ABC, abstractmethod

import numpy as np


class FittedReturnModel(ABC):
    name: str
    daily_drift: float

    @abstractmethod
    def simulate_daily_shocks(self, horizon: int, n_paths: int, rng: np.random.Generator) -> np.ndarray: ...

    @abstractmethod
    def with_data(self, log_prices: np.ndarray) -> "FittedReturnModel": ...

    @abstractmethod
    def describe(self) -> dict: ...


def simulate_cumulative_log_returns(
    model: FittedReturnModel,
    horizon: int,
    n_paths: int,
    rng: np.random.Generator,
    *,
    daily_drift=None,
    shock_multipliers: np.ndarray | None = None,
    price_band: float | None = None,
) -> np.ndarray:
    """Simulate `n_paths` h-day cumulative log returns.

    daily_drift        scalar, or one value per path (to integrate over drift uncertainty);
                       defaults to the model's own drift.
    shock_multipliers  per-day scale on the shocks, length h (earnings days > 1).
    price_band         daily circuit limit as a fraction; simulated daily moves are capped there.
    """
    drift = model.daily_drift if daily_drift is None else daily_drift
    drift = np.broadcast_to(np.asarray(drift, float), (n_paths,))[:, None]
    shocks = model.simulate_daily_shocks(horizon, n_paths, rng)
    if shock_multipliers is not None:
        shocks = shocks * np.asarray(shock_multipliers, float)[None, :]
    daily_log_returns = drift + shocks
    if price_band:
        daily_log_returns = np.clip(daily_log_returns, math.log(1 - price_band), math.log(1 + price_band))
    return daily_log_returns.sum(axis=1)
