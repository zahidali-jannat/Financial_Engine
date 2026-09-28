import pandas as pd

from finance_engine.data.fundamentals import FundamentalData, _from_json, _standardize_statement, _to_json


def test_fundamentals_survive_the_cache_round_trip_unchanged():
    earnings = pd.DataFrame(
        {"eps_estimate": [10.0, 11.0], "reported_eps": [10.5, float("nan")], "surprise_pct": [5.0, float("nan")]},
        index=pd.DatetimeIndex(["2025-07-18 10:00", "2025-10-17 10:00"], tz="UTC", name="announced_at").as_unit("us"),
    )
    quarterly = _standardize_statement(
        pd.DataFrame({pd.Timestamp("2025-06-30"): [100.0, 10.0], pd.Timestamp("2025-03-31"): [90.0, 9.0]}, index=["Total Revenue", "Net Income"])
    )
    original = FundamentalData({"trailingPE": 20.0, "sector": "Energy"}, earnings, quarterly, pd.DataFrame())

    restored = _from_json(_to_json(original, pd.Timestamp("2026-01-01", tz="UTC")))

    pd.testing.assert_frame_equal(restored.earnings, original.earnings)
    pd.testing.assert_frame_equal(restored.quarterly_income, original.quarterly_income)
    assert restored.info == original.info
    assert restored.annual_income.empty
