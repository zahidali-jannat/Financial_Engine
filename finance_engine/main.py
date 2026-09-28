"""Command-line entry point: full probabilistic analysis of one stock.

Usage:   python -m finance_engine.main RELIANCE.NS [--refresh] [--settings path/to/settings.yaml]
Running this file directly (e.g. VS Code's "Run Python File") also works and prompts for the ticker.

Prints the plain-language report and saves it, with a JSON summary, the full
backtest records and two charts, under reports/<TICKER>_<date>/.
"""

import argparse
import logging
import sys
from pathlib import Path

if not __package__:
    # Run as a plain script, so Python can't see the finance_engine package yet; add the project root.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finance_engine.config.settings import load_settings
from finance_engine.output.report import render_report, write_report
from finance_engine.pipeline import run_analysis


def print_progress(done: int, total: int) -> None:
    if done == 1 or done % 50 == 0 or done == total:
        print(f"  backtesting: {done}/{total} out-of-sample forecasts", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Probabilistic stock forecast engine (NSE tickers use the .NS suffix)")
    parser.add_argument("ticker", nargs="?", help="e.g. RELIANCE.NS, TCS.NS (prompted for if omitted)")
    parser.add_argument("--settings", help="path to an alternative settings.yaml")
    parser.add_argument("--refresh", action="store_true", help="ignore cached prices and fundamentals and refetch")
    args = parser.parse_args(argv)
    ticker = args.ticker or input("Ticker symbol (e.g. RELIANCE.NS, TCS.NS): ").strip()
    if not ticker:
        parser.error("a ticker symbol is required")
    if "." not in ticker and not ticker.startswith("^"):
        print(f"Note: '{ticker}' has no exchange suffix, so Yahoo treats it as a US listing. For NSE use {ticker.upper()}.NS")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    settings = load_settings(args.settings)
    print(f"Analysing {ticker.upper()} — loading data, then walk-forward backtesting every model...")
    analysis = run_analysis(ticker, settings, refresh=args.refresh, progress=print_progress)
    paths = write_report(analysis, settings)
    print()
    print(render_report(analysis, settings))
    print(f"Saved to {paths['report'].parent}:")
    for label, path in paths.items():
        print(f"  {label:18s} {path.name}")


if __name__ == "__main__":
    main()
