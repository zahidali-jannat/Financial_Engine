"""Local on-disk cache for OHLCV frames, so repeated runs don't re-fetch.

Each cached ticker is two files:
    {TICKER}_{interval}.csv       the bars (human-inspectable)
    {TICKER}_{interval}.meta.json what request produced them and when

A cache entry is reused only if it is fresh, covers the requested start date,
and was fetched with identical options (interval, adjustment, etc.). Otherwise
the caller refetches the full history — never appends — because adjusted prices
are rewritten backwards on every dividend or split.
"""

import json
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def cache_file_paths(cache_directory: Path, ticker: str, interval: str) -> tuple[Path, Path]:
    """Return (bars_csv_path, metadata_json_path) for a ticker, with filesystem-unsafe characters replaced."""
    safe_ticker = re.sub(r"[^A-Za-z0-9._^-]", "_", ticker)
    stem = f"{safe_ticker}_{interval}"
    cache_directory = Path(cache_directory)
    return cache_directory / f"{stem}.csv", cache_directory / f"{stem}.meta.json"


def is_cache_usable(
    metadata: dict,
    *,
    requested_start: pd.Timestamp,
    fetch_options: dict,
    max_age_hours: float,
    now: pd.Timestamp,
) -> bool:
    """Decide whether a cache entry can answer this request without refetching."""
    fetched_at = pd.Timestamp(metadata["fetched_at_utc"])
    age_hours = (now - fetched_at).total_seconds() / 3600
    if age_hours > max_age_hours:
        return False
    if metadata["fetch_options"] != fetch_options:
        return False
    return pd.Timestamp(metadata["requested_start"]) <= requested_start


def read_cached_ohlcv(
    cache_directory: Path,
    ticker: str,
    *,
    requested_start: pd.Timestamp,
    fetch_options: dict,
    max_age_hours: float,
    now: pd.Timestamp,
) -> pd.DataFrame | None:
    """Return cached bars if a usable entry exists, else None."""
    bars_path, metadata_path = cache_file_paths(cache_directory, ticker, fetch_options["interval"])
    if not (bars_path.exists() and metadata_path.exists()):
        return None

    metadata = json.loads(metadata_path.read_text())
    if not is_cache_usable(
        metadata,
        requested_start=requested_start,
        fetch_options=fetch_options,
        max_age_hours=max_age_hours,
        now=now,
    ):
        logger.info("Cache for %s is stale or doesn't match this request; refetching", ticker)
        return None

    logger.info("Using cached bars for %s from %s (fetched %s)", ticker, bars_path, metadata["fetched_at_utc"])
    return pd.read_csv(bars_path, index_col="date", parse_dates=["date"])


def write_cached_ohlcv(
    bars: pd.DataFrame,
    cache_directory: Path,
    ticker: str,
    *,
    requested_start: pd.Timestamp,
    fetch_options: dict,
    now: pd.Timestamp,
) -> Path:
    """Save bars plus the metadata needed to judge their reusability later. Returns the CSV path."""
    bars_path, metadata_path = cache_file_paths(cache_directory, ticker, fetch_options["interval"])
    bars_path.parent.mkdir(parents=True, exist_ok=True)

    bars.to_csv(bars_path, index_label="date")
    metadata = {
        "ticker": ticker,
        "fetched_at_utc": now.isoformat(),
        "requested_start": requested_start.isoformat(),
        "fetch_options": fetch_options,
        "row_count": len(bars),
        "first_bar": bars.index[0].isoformat(),
        "last_bar": bars.index[-1].isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Cached %d bars for %s at %s", len(bars), ticker, bars_path)
    return bars_path
