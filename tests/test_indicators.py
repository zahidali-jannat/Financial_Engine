import numpy as np
import pandas as pd
import pytest

from finance_engine.aggregation.composite import category_composites
from finance_engine.aggregation.redundancy import prune_redundant_indicators
from finance_engine.config.settings import load_settings
from finance_engine.data.loader import standardize_ohlcv
from finance_engine.indicators.registry import build_indicators, compute_normalized_indicators, indicator_categories
from tests.conftest import make_ohlcv_bars

SETTINGS = load_settings()
INDICATORS = build_indicators(SETTINGS["indicators"], SETTINGS["normalization"])


@pytest.fixture(scope="module")
def bars():
    return standardize_ohlcv(make_ohlcv_bars(periods=900, seed=21), 0.005)


def trending_bars(direction: float, periods: int = 300) -> pd.DataFrame:
    close = 100 * np.exp(direction * 0.01 * np.arange(periods))
    index = pd.bdate_range("2020-01-01", periods=periods)
    return pd.DataFrame(
        {"open": close * (1 - direction * 0.002), "high": close * 1.005, "low": close * 0.995, "close": close, "volume": 1e6},
        index=index,
    )


@pytest.mark.parametrize("indicator", INDICATORS, ids=lambda i: i.name)
def test_every_indicator_is_causal(indicator, bars):
    """Lookahead guard: values up to day t must not change when later bars are removed."""
    full = indicator.compute_normalized(bars)
    for cutoff in (300, 555, 899):
        truncated = indicator.compute_normalized(bars.iloc[:cutoff])
        pd.testing.assert_series_equal(truncated, full.iloc[:cutoff], check_exact=False, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("indicator", INDICATORS, ids=lambda i: i.name)
def test_every_normalized_indicator_stays_in_unit_range(indicator, bars):
    values = indicator.compute_normalized(bars).dropna()
    assert len(values) > 400
    assert values.between(-1, 1).all()


def test_directional_indicators_agree_on_a_clean_uptrend_and_downtrend():
    directional = [i for i in INDICATORS if i.is_directional and i.category in ("trend", "momentum")]
    for direction in (+1, -1):
        normalized = compute_normalized_indicators(trending_bars(direction), directional).iloc[-1]
        for name, value in normalized.items():
            if name in ("macd",):  # MACD measures acceleration: a constant-rate trend has ~zero histogram
                continue
            assert np.sign(value) == direction, name


def test_rsi_saturates_at_the_extremes():
    rsi = next(i for i in INDICATORS if i.name == "rsi")
    assert rsi.compute(trending_bars(+1)).iloc[-1] == pytest.approx(100.0)
    assert rsi.normalize(rsi.compute(trending_bars(-1))).iloc[-1] == pytest.approx(-1.0)


def test_adx_reads_strong_trend_regardless_of_direction():
    adx = next(i for i in INDICATORS if i.name == "adx")
    assert adx.compute(trending_bars(+1)).iloc[-1] > 50
    assert adx.compute(trending_bars(-1)).iloc[-1] > 50


def test_redundancy_pruning_drops_the_lower_priority_duplicate():
    rng = np.random.default_rng(0)
    base = rng.normal(size=500)
    frame = pd.DataFrame({"rsi": base, "stochastic": base + rng.normal(0, 0.1, 500), "obv_slope": rng.normal(size=500)})
    decision = prune_redundant_indicators(frame, ["rsi", "stochastic", "obv_slope"], max_abs_correlation=0.8)
    assert decision.kept == ["rsi", "obv_slope"]
    assert decision.dropped["stochastic"][0] == "rsi"


def test_category_composites_average_within_each_category(bars):
    normalized = compute_normalized_indicators(bars, INDICATORS)
    composites = category_composites(normalized, ["rsi", "cci", "macd"], indicator_categories(INDICATORS))
    expected_momentum = normalized[["rsi", "cci"]].mean(axis=1, skipna=False)
    pd.testing.assert_series_equal(composites["momentum"], expected_momentum, check_names=False)
    assert composites["volume"].isna().all()  # no volume indicator kept
