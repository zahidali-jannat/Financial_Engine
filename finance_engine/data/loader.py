"""Public entry point of the data stage: load_ohlcv(ticker, data_settings).

Pulls bars from the cache or the configured source, standardizes them, and
validates them against the contract below before any other stage sees them.

OUTPUT CONTRACT — what every downstream stage may assume:
    pd.DataFrame
      index   : DatetimeIndex named "date", tz-naive, datetime64[ns], freq=None,
                strictly increasing, unique, one row per trading day
      columns : exactly ["open", "high", "low", "close", "volume"], all float64
      values  : no NaN; open/high/low/close > 0; volume >= 0;
                low <= min(open, close) and high >= max(open, close)
      prices  : split/dividend adjusted when data.yfinance.auto_adjust is true
      coverage: starts no earlier than data.history_years ago and spans at
                least data.min_history_years
"""

import logging

import pandas as pd

from finance_engine.config.settings import resolve_project_path
from finance_engine.data.cache import read_cached_ohlcv, write_cached_ohlcv
from finance_engine.data.fetch import fetch_ohlcv_from_csv, fetch_ohlcv_from_yfinance

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ["open", "high", "low", "close"]
OHLCV_COLUMNS = PRICE_COLUMNS + ["volume"]


class InvalidOHLCVError(ValueError):
    """Bars violate the data-stage output contract."""


class InsufficientHistoryError(ValueError):
    """Fewer years of history are available than data.min_history_years requires."""


def load_ohlcv(
    ticker: str,
    data_settings: dict,
    *,
    now: pd.Timestamp | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return contract-conforming daily bars for `ticker`, using the cache when it's usable."""
    ticker = ticker.upper()
    now = _as_utc(now if now is not None else pd.Timestamp.now(tz="UTC"))
    requested_start = (now.tz_localize(None) - pd.DateOffset(years=data_settings["history_years"])).normalize()
    repair_tolerance = data_settings["ohlc_repair_tolerance"]

    source = data_settings["source"]
    if source == "yfinance":
        bars = _load_from_yfinance_with_cache(ticker, data_settings, requested_start, now, force_refresh)
    elif source == "csv":
        bars = fetch_ohlcv_from_csv(ticker, resolve_project_path(data_settings["csv"]["directory"]))
    else:
        raise ValueError(f"Unknown data.source {source!r}; expected 'yfinance' or 'csv'")

    # Also re-checks cached bars, so a hand-edited or corrupted cache file can't slip through.
    bars = standardize_ohlcv(bars, repair_tolerance)
    if data_settings["drop_holiday_placeholder_bars"]:
        bars = drop_holiday_placeholder_bars(bars)
    bars = bars[bars.index >= requested_start]
    validate_ohlcv(bars)
    ensure_minimum_history(bars, data_settings["min_history_years"])
    return bars


def _load_from_yfinance_with_cache(
    ticker: str,
    data_settings: dict,
    requested_start: pd.Timestamp,
    now: pd.Timestamp,
    force_refresh: bool,
) -> pd.DataFrame:
    yfinance_settings = data_settings["yfinance"]
    fetch_options = {
        "source": "yfinance",
        "interval": yfinance_settings["interval"],
        "auto_adjust": yfinance_settings["auto_adjust"],
        "drop_todays_bar": yfinance_settings["drop_todays_bar"],
    }
    cache_directory = resolve_project_path(data_settings["cache"]["directory"])

    if not force_refresh:
        cached_bars = read_cached_ohlcv(
            cache_directory,
            ticker,
            requested_start=requested_start,
            fetch_options=fetch_options,
            max_age_hours=data_settings["cache"]["max_age_hours"],
            now=now,
        )
        if cached_bars is not None:
            return cached_bars

    logger.info("Fetching %s from yfinance since %s", ticker, requested_start.date())
    raw_bars = fetch_ohlcv_from_yfinance(
        ticker,
        requested_start,
        interval=fetch_options["interval"],
        auto_adjust=fetch_options["auto_adjust"],
        drop_todays_bar=fetch_options["drop_todays_bar"],
        now=now,
    )
    bars = standardize_ohlcv(raw_bars, data_settings["ohlc_repair_tolerance"])
    write_cached_ohlcv(
        bars,
        cache_directory,
        ticker,
        requested_start=requested_start,
        fetch_options=fetch_options,
        now=now,
    )
    return bars


def standardize_ohlcv(raw_bars: pd.DataFrame, ohlc_repair_tolerance: float) -> pd.DataFrame:
    """Coerce any source's bars into the contract's shape: names, dtypes, index, ordering, no gaps.

    Idempotent — running it on already-standardized bars returns them unchanged.
    """
    bars = raw_bars.copy()
    bars.columns = [str(col).strip().lower() for col in bars.columns]
    missing_columns = [col for col in OHLCV_COLUMNS if col not in bars.columns]
    if missing_columns:
        raise InvalidOHLCVError(f"Missing required columns: {missing_columns}")
    bars = bars[OHLCV_COLUMNS].astype("float64")

    index = pd.DatetimeIndex(bars.index)
    if index.tz is not None:
        # Keep the exchange-local calendar date; converting to UTC first could shift it by a day.
        index = index.tz_localize(None)
    # Trading calendars are irregular (holidays), so no fixed frequency is claimed.
    bars.index = pd.DatetimeIndex(index.as_unit("ns"), freq=None, name="date")

    bars = bars.sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]

    rows_with_gaps = bars.isna().any(axis=1)
    if rows_with_gaps.any():
        logger.warning("Dropping %d bars with missing values", int(rows_with_gaps.sum()))
        bars = bars[~rows_with_gaps]

    if (bars[PRICE_COLUMNS] <= 0).any().any():
        raise InvalidOHLCVError("Found non-positive prices; the source data is corrupt")

    return repair_small_ohlc_inconsistencies(bars, ohlc_repair_tolerance)


def drop_holiday_placeholder_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Remove bars where nothing traded and nothing moved: open = high = low = close and zero volume.

    Yahoo inserts these on some NSE holidays. Kept, they would add fake 0% return
    days that understate volatility and fat tails. Indices report zero volume on
    real trading days too, so volume alone is not enough to flag a placeholder.
    """
    is_flat = bars[PRICE_COLUMNS].nunique(axis=1).eq(1)
    is_placeholder = is_flat & bars["volume"].eq(0)
    if is_placeholder.any():
        logger.info("Dropping %d holiday placeholder bars (flat price, zero volume)", int(is_placeholder.sum()))
    return bars[~is_placeholder]


def repair_small_ohlc_inconsistencies(bars: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Clip high/low to enclose open and close when the violation is within `tolerance` (a fraction of price).

    Adjusted prices are rounded independently per column, so a high can land a
    hair below the close. Larger violations mean genuinely bad data and raise.
    """
    candle_body_top = bars[["open", "close"]].max(axis=1)
    candle_body_bottom = bars[["open", "close"]].min(axis=1)
    high_shortfall = (candle_body_top - bars["high"]) / candle_body_top
    low_overshoot = (bars["low"] - candle_body_bottom) / candle_body_bottom

    beyond_tolerance = (high_shortfall > tolerance) | (low_overshoot > tolerance)
    if beyond_tolerance.any():
        first_bad_date = bars.index[beyond_tolerance.to_numpy()][0].date()
        raise InvalidOHLCVError(
            f"{int(beyond_tolerance.sum())} bars have high/low outside open/close by more than "
            f"{tolerance:.2%} (first on {first_bad_date})"
        )

    needs_repair = (high_shortfall > 0) | (low_overshoot > 0)
    if needs_repair.any():
        logger.warning("Clipped high/low on %d bars with rounding-level inconsistencies", int(needs_repair.sum()))
        bars = bars.assign(
            high=bars["high"].clip(lower=candle_body_top),
            low=bars["low"].clip(upper=candle_body_bottom),
        )
    return bars


def validate_ohlcv(bars: pd.DataFrame) -> None:
    """Raise InvalidOHLCVError listing every way `bars` violates the output contract."""
    problems = []
    if bars.empty:
        raise InvalidOHLCVError("No bars")
    if list(bars.columns) != OHLCV_COLUMNS:
        problems.append(f"columns are {list(bars.columns)}, expected {OHLCV_COLUMNS}")
    if not all(dtype == "float64" for dtype in bars.dtypes):
        problems.append("not all columns are float64")
    if not isinstance(bars.index, pd.DatetimeIndex) or bars.index.tz is not None:
        problems.append("index is not a tz-naive DatetimeIndex")
    if not bars.index.is_monotonic_increasing or not bars.index.is_unique:
        problems.append("index is not strictly increasing")
    if bars.isna().any().any():
        problems.append("contains NaN")
    if problems:
        raise InvalidOHLCVError("; ".join(problems))

    if (bars[PRICE_COLUMNS] <= 0).any().any():
        problems.append("contains non-positive prices")
    if (bars["volume"] < 0).any():
        problems.append("contains negative volume")
    if (bars["high"] < bars[["open", "close"]].max(axis=1)).any():
        problems.append("high below open or close")
    if (bars["low"] > bars[["open", "close"]].min(axis=1)).any():
        problems.append("low above open or close")
    if problems:
        raise InvalidOHLCVError("; ".join(problems))


def ensure_minimum_history(bars: pd.DataFrame, min_history_years: float) -> None:
    """Raise InsufficientHistoryError if the bars span fewer than `min_history_years`."""
    span_years = (bars.index[-1] - bars.index[0]).days / 365.25
    if span_years < min_history_years:
        raise InsufficientHistoryError(
            f"Only {span_years:.1f} years of history ({bars.index[0].date()} to {bars.index[-1].date()}); "
            f"need at least {min_history_years}"
        )


def _as_utc(timestamp: pd.Timestamp) -> pd.Timestamp:
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
