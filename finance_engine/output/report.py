"""Plain-language report: what the engine expects, how far to trust it, and what drove it.

Never a bare number: every probability is followed by the out-of-sample evidence
for (or against) believing it, and by the components that produced it.

    build_verdicts(analysis, settings) -> dict of pass/fail judgements with their evidence
    render_report(analysis, settings)  -> the text report
    write_report(analysis, settings)   -> report.txt, summary.json, backtest_records.csv, two PNG charts
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from finance_engine.config.settings import resolve_project_path
from finance_engine.evaluation.failure_modes import tail_miss_rate
from finance_engine.evaluation.model_comparison import coverage_tolerance
from finance_engine.output.charts import plot_calibration, plot_forecast_distribution

MODEL_LABELS = {
    "gbm": "GBM (baseline)",
    "garch_t": "GJR-GARCH-t",
    "garch_t_earnings": "GJR-GARCH-t + earnings days",
    "merton_jump": "Merton jump-diffusion",
    "regime_switching": "HMM regime-switching",
    "ornstein_uhlenbeck": "Ornstein-Uhlenbeck",
    "engine": "ENGINE",
}
EVIDENCE_LABELS = {
    "baseline": "long-run average (prior + history)",
    "trend": "trend indicators",
    "momentum": "momentum indicators",
    "volatility": "volatility state",
    "volume": "volume / money flow",
    "kalman_trend": "Kalman trend",
    "trend_when_trending": "trend signals (trending-regime weight)",
    "momentum_when_trending": "momentum signals (trending-regime weight)",
}
COMPONENT_LABELS = {
    "valuation_vs_history": "Valuation vs own history",
    "valuation_vs_peers": "Valuation vs peers",
    "growth": "Growth",
    "earnings_surprise": "Earnings surprises",
}
RULE = "=" * 78


def _money(value: float, currency: str | None) -> str:
    symbol = "₹" if currency == "INR" else ""
    return f"{symbol}{value:,.2f}"


def _p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _p_is(value: float) -> str:
    return "p < 0.001" if value < 0.001 else f"p = {value:.3f}"


def _tick(passed: bool) -> str:
    return "[PASS]" if passed else "[FAIL]"


def build_verdicts(analysis, settings: dict) -> dict:
    alpha = settings["evaluation"]["significance_level"]
    engine = analysis.model_comparison.loc["engine"]
    n = int(engine["forecasts"])
    levels = analysis.backtest.interval_levels
    coverage_gaps = {level: engine[f"coverage_{level}"] - level for level in levels}
    coverage_ok = all(abs(gap) <= coverage_tolerance(level, n) for level, gap in coverage_gaps.items())
    rule, hold = analysis.strategy["rule"], analysis.strategy["buy_and_hold"]
    return {
        "forecasts_tested": n,
        "calibrated": bool(engine["pit_ks_p_value"] >= alpha and coverage_ok),
        "pit_ks_p_value": float(engine["pit_ks_p_value"]),
        "coverage": {level: float(engine[f"coverage_{level}"]) for level in levels},
        "distribution_edge_over_gbm": bool(engine["crps_skill_vs_gbm"] > 0 and engine["crps_dm_p_value"] < alpha),
        "crps_skill_vs_gbm": float(engine["crps_skill_vs_gbm"]),
        "crps_p_value": float(engine["crps_dm_p_value"]),
        "directional_edge_over_gbm": bool(engine["brier_skill_vs_gbm"] > 0 and engine["brier_dm_p_value"] < alpha),
        "brier_skill_vs_gbm": float(engine["brier_skill_vs_gbm"]),
        "brier_p_value": float(engine["brier_dm_p_value"]),
        "rule_beats_buy_and_hold": bool(rule["sharpe_ratio"] > hold["sharpe_ratio"]),
    }


def render_report(analysis, settings: dict) -> str:
    forecast, info = analysis.forecast, analysis.fundamentals_info
    currency = info.get("currency")
    verdicts = build_verdicts(analysis, settings)
    lines = [
        RULE,
        f"{analysis.ticker} — {info.get('longName') or ''}".rstrip(" —"),
        f"{forecast.horizon}-trading-day probabilistic forecast · data to {forecast.as_of.date()} · last close {_money(forecast.last_close, currency)}",
        RULE,
        "",
    ]
    lines += _section_forecast(forecast, settings, currency, verdicts)
    lines += _section_trust(analysis, verdicts)
    lines += _section_drivers(analysis, settings)
    lines += _section_fundamentals(analysis, settings)
    lines += _section_market(analysis)
    lines += _section_model_checks(analysis)
    lines += _section_failures(analysis)
    lines += [
        "NOTES",
        "  • A research tool, not investment advice. Probabilities describe the model's view given",
        "    past behaviour; markets can do things the past never showed (policy shocks, fraud, halts).",
        "  • Every trust statement above comes from walk-forward tests: each past forecast used only",
        "    data available on its date, and was scored on what happened next.",
        "",
    ]
    return "\n".join(lines)


def _section_forecast(forecast, settings, currency, verdicts) -> list[str]:
    last_day = forecast.forecast_days[-1].date()
    median_price = forecast.median_price
    lines = [f"WHAT THE ENGINE EXPECTS  (next {forecast.horizon} trading days, to about {last_day})"]
    for level in (0.68, 0.9, 0.95):
        low, high = forecast.price_interval(level)
        lines.append(f"  {level:.0%} probability the price ends between {_money(low, currency)} and {_money(high, currency)}")
    lines.append(f"  Middle outcome (median): {_money(median_price, currency)} ({median_price / forecast.last_close - 1:+.1%})")
    p_up = forecast.probability_of_return_above(0.0)
    direction_note = "" if verdicts["directional_edge_over_gbm"] else "  ← no proven directional edge; treat as close to a coin flip"
    lines.append(f"  Chance it ends higher than today: {p_up:.0%}{direction_note}")
    moves = []
    for threshold in settings["forecast"]["move_thresholds"]:
        moves.append(f"rise > {threshold:.0%}: {forecast.probability_of_return_above(threshold):.0%}")
        moves.append(f"fall > {threshold:.0%}: {1 - forecast.probability_of_return_above(-threshold):.0%}")
    lines.append("  Chance of a  " + " · ".join(moves))
    lines.append("")
    return lines


def _section_trust(analysis, verdicts) -> list[str]:
    n = verdicts["forecasts_tested"]
    coverage = verdicts["coverage"]
    rule, hold = analysis.strategy["rule"], analysis.strategy["buy_and_hold"]
    first, last = analysis.backtest.records.index[0].date(), analysis.backtest.records.index[-1].date()
    lines = [f"HOW MUCH TO TRUST THIS  ({n} past forecasts it had never seen, {first} to {last})"]
    lines.append(
        f"  {_tick(verdicts['calibrated'])} Calibrated probabilities: when it said 90%, the outcome landed inside "
        f"{coverage.get(0.9, float('nan')):.1%} of the time; 95% → {coverage.get(0.95, float('nan')):.1%}; 50% → {coverage.get(0.5, float('nan')):.1%}"
        f" (uniformity test {_p_is(verdicts['pit_ks_p_value'])})."
    )
    significant = "statistically significant" if verdicts["distribution_edge_over_gbm"] else "not distinguishable from noise"
    lines.append(
        f"  {_tick(verdicts['distribution_edge_over_gbm'])} Range of outcomes vs the GBM baseline: CRPS {verdicts['crps_skill_vs_gbm']:+.1%} "
        f"({_p_is(verdicts['crps_p_value'])}, {significant})."
    )
    if verdicts["directional_edge_over_gbm"]:
        lines.append(f"  [PASS] Direction: up/down probabilities beat GBM's (Brier {verdicts['brier_skill_vs_gbm']:+.1%}, {_p_is(verdicts['brier_p_value'])}).")
    else:
        lines.append(
            f"  [FAIL] Direction: up/down probabilities were no better than GBM's (Brier {verdicts['brier_skill_vs_gbm']:+.1%}, "
            f"{_p_is(verdicts['brier_p_value'])}). The engine has NO reliable directional edge on this stock."
        )
    # A trading rule beating buy-and-hold WITHOUT a significant directional edge is one lucky path, not evidence.
    unsupported_win = verdicts["rule_beats_buy_and_hold"] and not verdicts["directional_edge_over_gbm"]
    tick = "[LUCK?]" if unsupported_win else _tick(verdicts["rule_beats_buy_and_hold"])
    lines.append(
        f"  {tick} After Indian trading costs, holding only when P(up) ≥ {analysis.strategy['enter_probability']:.0%}: "
        f"{rule['annual_return']:+.1%}/yr (Sharpe {rule['sharpe_ratio']:.2f}, worst drawdown {rule['max_drawdown']:.0%}) vs buy-and-hold "
        f"{hold['annual_return']:+.1%}/yr (Sharpe {hold['sharpe_ratio']:.2f}, worst drawdown {hold['max_drawdown']:.0%})."
    )
    if unsupported_win:
        lines.append(
            "          The rule won on this one historical path, but the direction test above found no real skill —"
        )
        lines.append("          so this outperformance is not evidence of an edge and may not repeat. Do not trade on it alone.")
    if verdicts["distribution_edge_over_gbm"] and not verdicts["directional_edge_over_gbm"]:
        lines.append("  → Use it for RISK (how far the price may move, position sizing, stop distances), not for picking direction.")
    elif not verdicts["distribution_edge_over_gbm"] and not verdicts["directional_edge_over_gbm"]:
        lines.append("  → On this stock the engine adds nothing proven over a simple random-walk model (GBM).")
        if verdicts["calibrated"]:
            lines.append("    Its ranges are still calibrated, so they remain usable as risk bands — just not better ones.")
    lines.append("")
    return lines


def _section_drivers(analysis, settings) -> list[str]:
    forecast, suite = analysis.forecast, analysis.forecast.suite
    garch = suite.garch.describe()
    hmm = suite.hmm.describe()
    horizon = forecast.horizon
    comparison = analysis.model_comparison
    candidates = [name for name in comparison.index if name not in ("engine",)]
    lines = ["WHY — WHAT SHAPED THIS FORECAST"]
    lines.append(
        f"  Width: {MODEL_LABELS.get(forecast.volatility_model, forecast.volatility_model)} — best out-of-sample of {len(candidates)} models "
        f"(CRPS {comparison.loc[forecast.volatility_model, 'crps_skill_vs_gbm']:+.1%} vs GBM)."
    )
    level = "above" if garch["next_day_volatility"] > garch["long_run_daily_volatility"] else "below"
    lines.append(
        f"    Volatility now {garch['next_day_volatility']:.2%}/day vs long-run {garch['long_run_daily_volatility']:.2%} ({level} normal); "
        f"shocks fade with a half-life of ~{garch['volatility_half_life_days']:.0f} days."
    )
    lines.append(
        f"    Falls raise volatility more than rises (leverage γ = {garch['gamma_leverage']:.3f}); tails are fat (Student-t ν = {garch['tail_degrees_of_freedom']:.1f})."
    )
    regime = "TURBULENT" if hmm["current_turbulent_probability"] > 0.5 else "CALM"
    trend = "TRENDING" if forecast.trending == 1 else "RANGING" if forecast.trending == 0 else "unknown"
    lines.append(f"  Regime: {regime} (HMM P(turbulent) = {hmm['current_turbulent_probability']:.0%}) and {trend} (ADX vs {settings['regimes']['trend_adx_threshold']}).")
    if forecast.upcoming_reaction_days:
        day = forecast.upcoming_reaction_days[0].date()
        multiplier = suite.earnings_variance_multiplier
        lines.append(f"  Earnings: results reaction expected {day} — INSIDE the window; that day's variance is scaled ×{multiplier or 1:.1f} (this stock's history).")
    else:
        lines.append("  Earnings: no results reaction day expected inside the window.")

    posterior = forecast.posterior
    lines.append(
        f"  Centre (expected {horizon}-day return): prior {posterior.prior_mean:+.2%} → after evidence {posterior.mean:+.2%} "
        f"(± {posterior.std:.2%} uncertainty), from {posterior.training_rows} past non-overlapping windows."
    )
    ranked = sorted(posterior.contributions.items(), key=lambda item: -abs(item[1]))
    for name, contribution in ranked[:6]:
        lines.append(f"    {contribution:+.3%}  {EVIDENCE_LABELS.get(name, name)}")
    typical_move = np.std(forecast.samples)
    lines.append(f"    (For scale: a typical {horizon}-day move is ±{typical_move:.1%}; these shifts are small by design — see the Bayesian prior in settings.)")
    lines.append("")
    return lines


def _section_fundamentals(analysis, settings) -> list[str]:
    fair_value = analysis.fair_value
    horizon = settings["forecast"]["horizon_days"]
    per_horizon = fair_value.annual_drift_tilt * horizon / settings["market"]["trading_days_per_year"]
    lines = ["FUNDAMENTALS  (a small prior tilt only — free data can't be tested walk-forward)"]
    lines.append(
        f"  Overall: {fair_value.label.upper()} (bias {fair_value.bias:+.2f} on −1…+1, confidence {fair_value.confidence_label}) "
        f"→ drift tilt {fair_value.annual_drift_tilt:+.2%}/yr ≈ {per_horizon:+.3%} over {horizon} days."
    )
    for name, component in fair_value.components.items():
        score = "n/a" if component.score is None else f"{component.score:+.2f}"
        lines.append(f"  • {COMPONENT_LABELS[name]} [{score}]: {component.detail}")
    yahoo_pe = analysis.fundamentals_info.get("trailingPE")
    if analysis.rebuilt_pe and yahoo_pe:
        gap = abs(analysis.rebuilt_pe / yahoo_pe - 1)
        note = "" if gap < 0.05 else "  ← mismatch: Yahoo's EPS feed may mix standalone/consolidated figures"
        lines.append(f"  P/E cross-check: rebuilt from reported EPS {analysis.rebuilt_pe:.1f} vs Yahoo {yahoo_pe:.1f}{note}")
    lines.append("")
    return lines


def _section_market(analysis) -> list[str]:
    context = analysis.market_context
    if context is None:
        return ["MARKET CONTEXT", "  Index data unavailable.", ""]

    def pct(value):
        return "n/a" if value is None else f"{value:+.1%}"

    lines = ["MARKET CONTEXT  (last 3 months)"]
    lines.append(f"  Stock vs {context.benchmark}: {pct(context.relative_strength_vs_benchmark)}")
    if context.sector_index:
        lines.append(f"  Stock vs sector index {context.sector_index}: {pct(context.relative_strength_vs_sector)};  sector vs {context.benchmark}: {pct(context.sector_vs_benchmark)}")
    else:
        lines.append("  Sector index: none with history on Yahoo for this sector (only IT, Bank and Pharma have one).")
    if context.beta is not None:
        lines.append(f"  Beta to {context.benchmark} (1 year): {context.beta:.2f} (correlation {context.correlation:.2f})")
    lines.append("  Interest-rate sensitivity: not assessed — no reliable free Indian bond-yield series.")
    lines.append("")
    return lines


def _section_model_checks(analysis) -> list[str]:
    assumptions, refits = analysis.gbm_assumptions, analysis.backtest.refits
    adf_rejections = sum(refit["adf_rejects_unit_root"] for refit in refits)
    lines = ["MODEL CHECKS  (assumptions tested on this stock, not assumed)"]
    lines.append(
        f"  GBM's normal returns: {'hold' if assumptions['returns_are_normal'] else 'REJECTED'} — excess kurtosis {assumptions['excess_kurtosis']:.1f}, "
        f"{assumptions['moves_beyond_4_sd']} moves beyond 4σ vs {assumptions['moves_beyond_4_sd_expected_if_normal']:.2f} expected."
    )
    lines.append(f"  Volatility clustering (ARCH-LM): {'present' if assumptions['volatility_clusters'] else 'not detected'} ({_p_is(assumptions['arch_lm_p_value'])}) — the reason GARCH is used.")
    lines.append(
        f"  Mean reversion (ADF on log price): rejected the unit root in {adf_rejections} of {len(refits)} refits → "
        f"{'OU used where it applied' if adf_rejections else 'no mean reversion; OU not used'}."
    )
    dropped = analysis.forecast.redundancy.dropped
    if dropped:
        parts = [f"{name} (ρ={corr:.2f} with {twin})" for name, (twin, corr) in dropped.items()]
        lines.append(f"  Redundant indicators dropped (double-counting guard): {', '.join(parts)}.")
    lines.append("")
    header = f"  {'model':34s}{'CRPS vs GBM':>12s}{'p-value':>9s}{'calibr. p':>10s}{'90% cover':>11s}"
    lines.append(header)
    for name, row in analysis.model_comparison.iterrows():
        skill = "—" if name == "gbm" else f"{row['crps_skill_vs_gbm']:+.1%}"
        p_value = "—" if name == "gbm" else _p(row["crps_dm_p_value"])
        lines.append(f"  {MODEL_LABELS.get(name, name):34s}{skill:>12s}{p_value:>9s}{_p(row['pit_ks_p_value']):>10s}{row['coverage_0.9']:>11.1%}")
    lines.append("  (calibr. p < 0.05 = forecast probabilities are provably off; 90% cover should be ~90%)")
    lines.append("")
    return lines


def _section_failures(analysis) -> list[str]:
    segments = analysis.segments
    records = analysis.backtest.records
    lines = ["WHERE IT DOES WORST  (out of sample)"]
    non_year = segments[~segments.index.str.fullmatch(r"\d{4}")]
    for label, row in non_year.sort_values("crps_skill_vs_gbm").iterrows():
        caution = "  (small sample — read loosely)" if row["forecasts"] < 30 else ""
        lines.append(f"  {label:28s} n={int(row['forecasts']):4d}  skill vs GBM {row['crps_skill_vs_gbm']:+.1%}  90% coverage {row['coverage_0.9']:.0%}{caution}")
    years = segments[segments.index.str.fullmatch(r"\d{4}")]
    if len(years):
        worst_year = years["crps_skill_vs_gbm"].idxmin()
        lines.append(f"  Worst calendar year: {worst_year} (skill {years.loc[worst_year, 'crps_skill_vs_gbm']:+.1%}).")
    lines.append(f"  Outcomes outside the 95% interval: {tail_miss_rate(records):.1%} (expected 5%). Biggest misses:")
    for date, row in analysis.worst_misses.iterrows():
        tags = []
        if row["earnings_in_window"]:
            tags.append("results in window")
        if row["turbulent_probability"] > 0.5:
            tags.append("turbulent regime")
        lines.append(f"    {date.date()}  actual {row['realized_return']:+.1%}  (forecast percentile {row['pit']:.1%}){'  — ' + ', '.join(tags) if tags else ''}")
    lines.append("")
    return lines


def summary_json(analysis, settings: dict) -> dict:
    forecast = analysis.forecast
    verdicts = build_verdicts(analysis, settings)
    return {
        "ticker": analysis.ticker,
        "as_of": str(forecast.as_of.date()),
        "last_close": forecast.last_close,
        "horizon_trading_days": forecast.horizon,
        "price_intervals": {str(level): forecast.price_interval(level) for level in settings["forecast"]["interval_levels"]},
        "median_price": forecast.median_price,
        "probability_up": forecast.probability_of_return_above(0.0),
        "volatility_model": forecast.volatility_model,
        "drift_posterior": {"mean": forecast.posterior.mean, "std": forecast.posterior.std, "contributions": forecast.posterior.contributions},
        "fair_value": {"label": analysis.fair_value.label, "bias": analysis.fair_value.bias, "confidence": analysis.fair_value.confidence},
        "verdicts": {key: value for key, value in verdicts.items() if key != "coverage"},
        "coverage": {str(level): value for level, value in verdicts["coverage"].items()},
        "model_comparison": json.loads(analysis.model_comparison.to_json(orient="index")),
    }


def write_report(analysis, settings: dict) -> dict[str, Path]:
    directory = resolve_project_path(settings["output"]["directory"]) / f"{analysis.ticker}_{analysis.forecast.as_of.date()}"
    directory.mkdir(parents=True, exist_ok=True)
    forecast = analysis.forecast

    paths = {
        "report": directory / "report.txt",
        "summary": directory / "summary.json",
        "records": directory / "backtest_records.csv",
        "calibration_chart": directory / "calibration.png",
        "forecast_chart": directory / "forecast.png",
    }
    paths["report"].write_text(render_report(analysis, settings))
    paths["summary"].write_text(json.dumps(summary_json(analysis, settings), indent=2, default=float))
    analysis.backtest.records.to_csv(paths["records"])
    plot_calibration(
        analysis.engine_interval_calibration,
        analysis.baseline_interval_calibration,
        analysis.engine_reliability,
        analysis.engine_pit_histogram,
        len(analysis.backtest.records),
        paths["calibration_chart"],
    )
    intervals = {level: forecast.price_interval(level) for level in (0.68, 0.9)}
    plot_forecast_distribution(
        forecast.last_close * np.exp(forecast.samples),
        forecast.last_close,
        intervals,
        f"{analysis.ticker}: simulated price in {forecast.horizon} trading days ({len(forecast.samples):,} paths)",
        paths["forecast_chart"],
    )
    return paths
