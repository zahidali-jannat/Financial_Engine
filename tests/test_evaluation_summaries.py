import numpy as np
import pandas as pd
import pytest

from finance_engine.evaluation.calibration import interval_calibration, pit_histogram, reliability_table
from finance_engine.evaluation.model_comparison import compare_models
from finance_engine.evaluation.strategy_check import evaluate_long_only_rule


def fake_records(n=300, seed=0):
    rng = np.random.default_rng(seed)
    realized = rng.normal(0.002, 0.03, n)
    records = pd.DataFrame({"realized": realized}, index=pd.bdate_range("2020-01-01", periods=n))
    for name, noise in (("gbm", 0.0), ("engine", -0.001)):
        records[f"{name}_crps"] = 0.02 + noise + rng.normal(0, 0.001, n)
        records[f"{name}_pit"] = rng.uniform(size=n)
        records[f"{name}_p_up"] = 0.5 + rng.normal(0, 0.02, n)
        records[f"{name}_brier"] = (records[f"{name}_p_up"] - (realized > 0)) ** 2
        for level in (0.5, 0.9):
            records[f"{name}_hit_{level}"] = rng.uniform(size=n) < level
    return records


def test_compare_models_reports_skill_and_significance_against_gbm():
    table = compare_models(fake_records(), ["gbm", "engine"], [0.5, 0.9])
    assert table.loc["gbm", "crps_skill_vs_gbm"] == 0.0
    assert table.loc["engine", "crps_skill_vs_gbm"] == pytest.approx(0.05, abs=0.01)
    assert table.loc["engine", "crps_dm_p_value"] < 1e-6
    assert table.loc["engine", "coverage_0.9"] == pytest.approx(0.9, abs=0.05)


def test_calibration_tables_for_a_calibrated_model():
    pit = pd.Series(np.random.default_rng(1).uniform(size=5000))
    table = interval_calibration(pit)
    assert np.all(np.abs(table["observed"] - table.index) <= table["tolerance"])
    assert pit_histogram(pit, 10).sum() == pytest.approx(1.0)

    probability = pd.Series(np.random.default_rng(2).uniform(0.3, 0.7, 5000))
    realized = pd.Series(np.where(np.random.default_rng(3).uniform(size=5000) < probability, 0.01, -0.01))
    reliability = reliability_table(probability, realized, bins=5)
    assert np.all(np.abs(reliability["observed_frequency"] - reliability["mean_forecast"]) <= reliability["tolerance"])


def test_strategy_check_charges_costs_on_every_switch():
    records = pd.DataFrame(
        {"realized": np.log([1.01, 1.01, 1.01, 1.01]), "p": [0.6, 0.4, 0.6, 0.4]}, index=pd.bdate_range("2024-01-01", periods=4)
    )
    result = evaluate_long_only_rule(
        records, probability_column="p", enter_probability=0.55, horizon=5, trading_days_per_year=250, risk_free_rate_annual=0.0, round_trip_cost=0.01
    )
    # in, out, in, out = 4 switches; each costs half the round trip.
    assert result["rule"]["trades"] == 4
    expected_wealth = (1.01 - 0.005) * (1 - 0.005) * (1.01 - 0.005) * (1 - 0.005)
    assert result["rule"]["total_return"] == pytest.approx(expected_wealth - 1)
    assert result["rule"]["share_of_time_invested"] == 0.5
