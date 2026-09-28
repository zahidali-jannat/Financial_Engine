"""Reality check after Indian trading costs: would acting on P(up) have beaten simply holding the stock?

Rule (deliberately simple — a test of the forecasts, not a trading system):
    at each forecast date, hold the stock for the next h days if P(up) ≥ threshold,
    otherwise hold cash earning the risk-free rate. Long-only, because delivery
    short-selling isn't available in the Indian cash market.
Costs: half the round-trip cost on each entry and each exit.
Compared with buy-and-hold over exactly the same out-of-sample dates.
"""

import numpy as np
import pandas as pd


def _performance(period_returns: pd.Series, periods_per_year: float, risk_free_per_period: float) -> dict:
    wealth = (1 + period_returns).cumprod()
    years = len(period_returns) / periods_per_year
    excess = period_returns - risk_free_per_period
    volatility = period_returns.std() * np.sqrt(periods_per_year)
    return {
        "annual_return": float(wealth.iloc[-1] ** (1 / years) - 1),
        "annual_volatility": float(volatility),
        "sharpe_ratio": float(excess.mean() / period_returns.std() * np.sqrt(periods_per_year)) if period_returns.std() > 0 else 0.0,
        "max_drawdown": float((wealth / wealth.cummax() - 1).min()),
        "total_return": float(wealth.iloc[-1] - 1),
    }


def evaluate_long_only_rule(
    records: pd.DataFrame,
    *,
    probability_column: str,
    enter_probability: float,
    horizon: int,
    trading_days_per_year: int,
    risk_free_rate_annual: float,
    round_trip_cost: float,
) -> dict:
    periods_per_year = trading_days_per_year / horizon
    risk_free = (1 + risk_free_rate_annual) ** (1 / periods_per_year) - 1
    stock_return = np.expm1(records["realized"])
    invested = records[probability_column] >= enter_probability

    switches = invested.astype(int).diff().abs().fillna(invested.iloc[0].astype(int))
    rule_return = np.where(invested, stock_return, risk_free) - switches * round_trip_cost / 2
    rule = _performance(pd.Series(rule_return, index=records.index), periods_per_year, risk_free)
    rule.update(
        {
            "share_of_time_invested": float(invested.mean()),
            "trades": int(switches.sum()),
            "hit_rate_when_invested": float((stock_return[invested] > 0).mean()) if invested.any() else float("nan"),
        }
    )
    hold = _performance(stock_return, periods_per_year, risk_free)
    hold["total_return"] -= round_trip_cost  # one entry and one exit over the whole period
    return {"rule": rule, "buy_and_hold": hold, "enter_probability": enter_probability}
