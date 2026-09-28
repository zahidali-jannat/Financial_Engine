# Financial Engine

**An honest, probability-based forecasting engine for Indian (NSE) stocks.**

Give it a stock symbol like `RELIANCE.NS`. It tells you **how far the price is likely to move over the next 5 trading days, and how sure it is**, as price ranges with probabilities (for example: "90% chance the price ends between ₹1,169 and ₹1,301").

It then **proves how trustworthy those ranges are** by replaying 8 years of history, making a forecast every week using only the data that existed at that time, and checking what really happened.

> **It does not give a single "target price", and it does not claim to know whether a stock will go up or down.** Its own tests show it has no reliable edge on direction, and the report says so plainly. Its real strength is measuring **risk**.

---

## Contents

1. [What this project does](#1-what-this-project-does)
2. [Results so far](#2-results-so-far)
3. [Quick start](#3-quick-start)
4. [What you get after a run](#4-what-you-get-after-a-run)
5. [How it works](#5-how-it-works)
6. [How the engine is tested against history (the backtest)](#6-how-the-engine-is-tested-against-history-the-backtest)
7. [How forecasts are graded](#7-how-forecasts-are-graded)
8. [Built for the Indian market](#8-built-for-the-indian-market)
9. [Settings you can change](#9-settings-you-can-change)
10. [Project structure](#10-project-structure)
11. [Running the tests](#11-running-the-tests)
12. [Speed](#12-speed)
13. [Known limitations](#13-known-limitations)
14. [Technical specifications](#14-technical-specifications)
15. [Glossary](#15-glossary)
16. [Disclaimer](#16-disclaimer)

---

## 1. What this project does

Most "stock predictor" projects output one number ("RELIANCE will be ₹1,300 next week") and report an accuracy figure that quietly used future data. This engine is built the other way around:

| Typical stock predictor | This engine |
|---|---|
| One target price | A full range of possible prices, each with a probability |
| "95% accuracy" (often tested on data it was trained on) | Every forecast is tested on data it had **never seen** |
| Hides its weak spots | Reports clearly what works (**risk**) and what does not (**direction**) |
| One model | 6 mathematical models compete; the engine uses whichever has done best **so far** |
| Built for US stocks | Built for NSE: Nifty benchmark, Indian trading costs, no short selling |

**In one sentence:** it combines chart signals, company fundamentals and six statistical models into one forecast, simulates 10,000 possible futures, and grades itself honestly against history.

### Good uses
- How far could this stock move next week? (risk)
- Where should I put a stop-loss so normal noise doesn't trigger it?
- How much money should I put in this position, given how volatile it is right now?
- Is the market calm or turbulent right now?

### Not good for
- Deciding which stock will go up.
- A single price target.

---

## 2. Results so far

Tested on three large NSE stocks, each with **392 weekly forecasts** from **October 2018 to September 2026**. Every forecast used only data available on its own date.

### Are the price ranges honest?

When the engine says "90% chance the price lands in this range", it should land there about 90% of the time.

| Stock | Engine: 90% range held | Engine: 50% range held | Simple random-walk model: 50% range held |
|---|---|---|---|
| RELIANCE | 89.3% | 51.3% | 60.5% (too wide) |
| HDFC Bank | 89.0% | 51.0% | 56.6% |
| TCS | 91.3% | 51.8% | 57.9% (too wide) |

**Result: yes, the engine's ranges are honest on all three stocks.** The simple random-walk model gets them wrong because it assumes volatility never changes.

### Is it better than a simple random walk?

| Stock | Improvement over random walk (CRPS) | Chance it's just luck (p-value) | Verdict |
|---|---|---|---|
| RELIANCE | +2.3% | 0.011 | Real improvement |
| HDFC Bank | +1.8% | 0.031 | Real improvement |
| TCS | +0.7% | 0.21 | Could be luck |

### Can it predict direction (up or down)?

| Stock | Direction called correctly | Weeks the price rose anyway |
|---|---|---|
| RELIANCE | 53.3% | 54.6% |
| HDFC Bank | 53.1% | 53.3% |
| TCS | 50.8% | 48.7% |

**Result: no.** It is no better than a simple random-walk model at up/down calls. This matches what finance research generally finds for large, heavily traded stocks: **volatility can be predicted from past prices, direction cannot.**

### Paper-trading checks (pretend money, same fixed rules, after Indian trading costs)

| Test | Engine | Buy and hold | Nifty 50 | Engine's worst loss vs buy and hold |
|---|---|---|---|---|
| ₹1 crore, 3 stocks, Oct 2018 – Dec 2019 | +28.6% | +29.5% | +19.7% | −9.6% vs −9.9% |
| ₹1 crore, 3 stocks, 2024 – 2025 | +21.3% | +10.1% | +20.1% | −9.5% vs −15.8% |
| ₹2 lakh, RELIANCE only, 2024 | +8.0% | −5.3% | +9.2% | −13.3% vs −24.5% |

The engine made money in all three tests and lost much less in falling markets, mainly by holding more cash when volatility rose. Its extra gain over a fixed mix was positive each time, but **not large enough to rule out luck** yet, so treat these results as encouraging, not proven.

*Results were produced with the default settings and prices as cached in September 2026; rerunning later with fresh data will change them slightly.*

---

## 3. Quick start

### What you need
- A Mac, Linux or Windows computer
- **Python 3.12 or newer** (developed on Python 3.14)
- An internet connection for the first run (to download prices from Yahoo Finance)

### Step 1: open a terminal in the project folder

```bash
cd path/to/Financial_Engine
```

### Step 2: create a private Python environment (one time only)

On a Mac, the system command is `python3`, not `python`.

```bash
python3 -m venv .venv
```

### Step 3: switch it on (every time you open a new terminal)

```bash
source .venv/bin/activate        # Mac / Linux
.venv\Scripts\activate           # Windows
```

After this, your terminal prompt starts with `(.venv)`, and plain `python` works.

### Step 4: install the libraries (one time only)

```bash
pip install -r requirements.txt matplotlib
```

### Step 5: analyse a stock

```bash
python -m finance_engine.main RELIANCE.NS
```

The first run downloads about 10 years of prices and takes roughly 20–40 seconds. Later runs reuse the saved data for 12 hours.

**Options:**

| Command | What it does |
|---|---|
| `python -m finance_engine.main TCS.NS` | Analyse TCS |
| `python -m finance_engine.main RELIANCE.NS --refresh` | Ignore saved data and download fresh prices |
| `python -m finance_engine.main RELIANCE.NS --settings my_settings.yaml` | Use a different settings file |
| `python -m finance_engine.main` | Asks you to type the symbol |

**Symbols:** NSE stocks need the `.NS` ending (`RELIANCE.NS`, `HDFCBANK.NS`, `INFY.NS`). Indices start with `^` (`^NSEI` is the Nifty 50). Without `.NS`, Yahoo treats the symbol as a US stock.

**From VS Code:** open `finance_engine/main.py` and press the ▶ Run button. It will ask for the symbol.

---

## 4. What you get after a run

A plain-English report is printed on screen and saved, together with data and charts, in a new folder:

```
reports/RELIANCE.NS_2026-09-25/
├── report.txt              the full plain-English report
├── summary.json            the key numbers, for other programs to read
├── backtest_records.csv    every past weekly forecast and what really happened
├── calibration.png         chart: did the ranges hold as often as promised?
└── forecast.png            chart: the spread of 10,000 simulated prices for next week
```

### A shortened example report

```
RELIANCE.NS — Reliance Industries Limited
5-trading-day probabilistic forecast · data to 2026-09-25 · last close ₹1,226.00

WHAT THE ENGINE EXPECTS  (next 5 trading days, to about 2026-10-02)
  68% probability the price ends between ₹1,197.27 and ₹1,273.16
  90% probability the price ends between ₹1,168.73 and ₹1,300.85
  Middle outcome (median): ₹1,234.91 (+0.7%)
  Chance it ends higher than today: 59%  ← no proven directional edge; treat as close to a coin flip

HOW MUCH TO TRUST THIS  (392 past forecasts it had never seen)
  [PASS] Calibrated probabilities: when it said 90%, the outcome landed inside 90.6% of the time
  [PASS] Range of outcomes vs the GBM baseline: CRPS +2.5% (p = 0.002, statistically significant)
  [FAIL] Direction: up/down probabilities were no better than GBM's
  → Use it for RISK (how far the price may move, position sizing, stop distances), not for picking direction.

WHY — WHAT SHAPED THIS FORECAST
  Width: GJR-GARCH-t + earnings days — best out-of-sample of 5 models
  Regime: CALM (HMM P(turbulent) = 2%) and TRENDING (ADX vs 25)
  ...
```

The report has four parts:
1. **What the engine expects:** price ranges and probabilities.
2. **How much to trust this:** PASS/FAIL checks from the historical test.
3. **Why:** which model, which market regime, which signals pushed the forecast.
4. **Fundamentals:** whether the stock looks cheap or expensive versus its history and its peers.

---

## 5. How it works

The engine works in **five layers**, like stations in a factory. Each layer is tested on its own.

```
             ┌────────────────────────────────────────────────────┐
  Prices ───►│ 1. INDICATORS   13 chart signals, rescaled to −1..+1 │
             └──────────────────────────┬─────────────────────────┘
             ┌──────────────────────────▼─────────────────────────┐
  Company ──►│ 2. FUNDAMENTALS  cheap or expensive? (small nudge)  │
  data       └──────────────────────────┬─────────────────────────┘
             ┌──────────────────────────▼─────────────────────────┐
             │ 3. MODELS   6 maths models of how prices wiggle      │
             └──────────────────────────┬─────────────────────────┘
             ┌──────────────────────────▼─────────────────────────┐
             │ 4. SYNTHESIS  combine everything → simulate 10,000  │
             │               possible futures → price ranges       │
             └──────────────────────────┬─────────────────────────┘
             ┌──────────────────────────▼─────────────────────────┐
             │ 5. VALIDATION  replay 8 years week by week and      │
             │                grade every forecast                 │
             └────────────────────────────────────────────────────┘
```

### Layer 1: Technical indicators

The classic signals traders read from charts. Each one is rescaled to a common **−1 (very bearish) … +1 (very bullish)** scale, using only past data.

| Group | Indicators | Settings (from `settings.yaml`) |
|---|---|---|
| **Trend** | Moving-average spread, MACD, Directional Movement (+DI/−DI) | EMA 20 vs SMA 50; MACD 12/26/9; DMI 14 |
| **Momentum** | RSI, Stochastic, CCI, Bollinger position | RSI 14; Stochastic 14/3; CCI 20; Bollinger 20, 2σ |
| **Volatility** | ATR %, Bollinger bandwidth, Realized volatility | ATR 14; Bollinger 20; 21 days |
| **Volume** | OBV slope, VWAP distance, Chaikin Money Flow | 20; 20; 20 |
| **Regime** | ADX | 14 (above 25 = trending market) |

- **Duplicate removal:** indicators that say the same thing (correlation above 0.8) are dropped, so one idea is not counted twice.
- **Grouping:** the rest are averaged within each group (trend, momentum, volatility, volume).

### Layer 2: Fundamentals

A small nudge based on whether the company looks cheap or expensive:

| Check | Weight |
|---|---|
| P/E now vs its own 10-year history | 30% |
| Valuation vs peers (e.g. RELIANCE vs ONGC, IOC, BPCL) | 30% |
| Revenue and profit growth | 20% |
| Recent earnings surprises | 20% |

The nudge is capped at **±3% per year**, which is about ±0.06% over 5 days. It is deliberately tiny, because free fundamental data cannot be tested properly against history.

### Layer 3: Statistical models

Six models compete to describe how the price moves. Each one captures a different real-world behaviour:

| Model | What it captures, in plain English |
|---|---|
| **GBM (random walk)** | The baseline: prices move randomly with constant volatility. Every other model must beat this. |
| **GJR-GARCH with fat tails** | Calm days follow calm days, wild days follow wild days. Falls raise volatility more than rises. Big surprises happen more often than a bell curve suggests. |
| **GARCH + earnings days** | Same as above, plus extra volatility on the day the market reacts to quarterly results. |
| **Merton jump-diffusion** | Normal movement plus occasional sudden jumps. |
| **Hidden Markov regimes** | The market switches between a hidden "calm" state and a "turbulent" state. |
| **Ornstein–Uhlenbeck** | Prices pulled back towards an average. **Only used if a statistical test (ADF) shows the stock actually behaves that way**, which is rare. |
| **Kalman trend filter** | A smoothed trend estimate with its own uncertainty, used as a signal rather than as a price model. |

All models are fitted with maximum likelihood, the standard statistical method for choosing the parameters that best explain past data.

### Layer 4: Synthesis (the "engine")

The final forecast has two parts:

1. **Where the centre is (expected return):** a **Bayesian** model. It starts from a sensible assumption (Indian large caps return about 10% a year) and only moves away from it when the signals have **proven** themselves on past data. Weak signals get almost no influence. This is the main protection against fooling itself.
2. **How wide the range is (risk):** the volatility model with the **best track record so far** (lowest past error on forecasts already scored). It only switches models after at least 20 scored forecasts.

It then simulates **10,000 possible 5-day futures** (Monte Carlo) and reads the price ranges and probabilities from them.

### Layer 5: Validation

See the next two sections.

---

## 6. How the engine is tested against history (the backtest)

The engine is tested with a **walk-forward backtest**, the most honest way to test a forecaster:

```
2016 ─────────── 2018 ─┬─ week 1: learn from all data up to here → forecast next 5 days → check
                       ├─ week 2: learn from all data up to here → forecast next 5 days → check
                       ├─ ...
                       └─ week 392 (Sep 2026)
```

| Rule | Value |
|---|---|
| History needed before the first forecast | 504 trading days (about 2 years) |
| Forecast horizon | 5 trading days |
| How often a forecast is made | Every 5 trading days, so outcome periods never overlap |
| How often models are re-fitted | Every 21+ trading days (in between, they update with each new day's data) |
| Training window | All history up to the forecast date (expanding) |

### Guarding against cheating with future data (lookahead)

Cheating with future data is the most common reason stock backtests look good but fail in real life. The engine guards against it in several ways:

- **Models:** trained only on prices up to the forecast date.
- **Indicators:** use only past data. A test checks this by cutting the data short and confirming earlier values do not change.
- **Recent outcomes:** outcomes that were not yet known on the forecast date are hidden from the learning step.
- **Model choice:** the volatility model is chosen using only forecasts whose outcomes were already known.
- **End-to-end test:** the future prices are scrambled and every earlier forecast is confirmed to stay exactly the same.

A separate audit confirmed all of this, and also confirmed that **running the same analysis twice gives byte-for-byte identical results**, because every random simulation uses a fixed seed (42).

---

## 7. How forecasts are graded

| Check | Question it answers | Good result |
|---|---|---|
| **Range coverage** | When it said "90% range", did the price land inside 90% of the time? | Close to the stated %, within the chance margin shown |
| **PIT uniformity** | Across all forecasts, were outcomes spread evenly across the predicted range? | p-value above 0.05 |
| **CRPS** | How good was the whole predicted range? (One score that rewards being accurate *and* appropriately confident) | Lower than the random walk |
| **Diebold–Mariano test** | Is the improvement over the random walk real, or could it be luck? | p-value below 0.05 |
| **Brier score** | How good were the "chance of going up" numbers? | Lower than the random walk |
| **Trading reality check** | After Indian costs, does a simple rule (hold only when P(up) ≥ 55%, else cash) beat buy-and-hold? | Shown for context only, not proof of skill |

Every model is compared with the **random walk (GBM)**. If the engine cannot beat that, the report says so.

---

## 8. Built for the Indian market

| Feature | How it's handled |
|---|---|
| Stocks | NSE symbols with `.NS` (e.g. `RELIANCE.NS`) |
| Benchmark | Nifty 50 (`^NSEI`); sector indices for IT (`^CNXIT`), banks (`^NSEBANK`) and pharma (`^CNXPHARMA`) |
| Trading costs | 0.25% per round trip (STT, exchange, SEBI, GST, stamp duty, brokerage) |
| Risk-free rate | 6.5% per year (about the 91-day T-bill; update it periodically) |
| Short selling | Not used: delivery short selling isn't available in the Indian cash market |
| Trading days | 248 per year |
| Earnings timing | Results announced after the 15:30 IST close move the price on the next session |
| Holiday data errors | Fake flat "holiday" bars that Yahoo sometimes inserts are removed |
| Circuit limits | Optional daily price band per stock (e.g. 10%); simulated moves are capped at it |
| Currency | Reports show ₹ |

---

## 9. Settings you can change

**Every** adjustable number lives in one file: [`finance_engine/config/settings.yaml`](finance_engine/config/settings.yaml). Nothing is hidden inside the Python code, so you can change how the engine behaves without editing any Python.

The most useful settings:

| Setting | Default | What it controls |
|---|---|---|
| `data.history_years` | 10 | How many years of prices to download |
| `forecast.horizon_days` | 5 | How many trading days ahead to forecast |
| `forecast.monte_carlo_paths` | 10000 | How many possible futures to simulate |
| `forecast.random_seed` | 42 | Fixed seed so results are repeatable |
| `forecast.interval_levels` | 50%, 68%, 80%, 90%, 95% | Which probability ranges to report |
| `walk_forward.min_training_days` | 504 | History needed before the first test forecast |
| `walk_forward.refit_every_days` | 21 | How often models are re-fitted in the backtest |
| `market.round_trip_cost` | 0.0025 | Trading cost per buy + sell |
| `market.risk_free_rate_annual` | 0.065 | Return on idle cash |
| `market.price_bands` | none | Daily circuit limits, e.g. `{SOMESTOCK.NS: 0.10}` |
| `redundancy.max_abs_correlation` | 0.8 | When two indicators count as duplicates |
| `bayes.prior_annual_drift` | 0.10 | Starting assumption for yearly return |
| `bayes.signal_effect_prior_sd` | 0.0025 | How much any one signal is allowed to move the forecast |
| `fundamentals.max_annual_tilt` | 0.03 | Maximum yearly nudge from fundamentals |
| `fundamentals.peers` | list per stock | Which companies each stock is compared with |
| `strategy_check.enter_probability` | 0.55 | Threshold for the simple trading check |

To keep your own version, copy the file and run with `--settings my_settings.yaml`.

---

## 10. Project structure

```
Financial_Engine/
├── finance_engine/
│   ├── main.py                 ◄ START HERE: the command you run
│   ├── pipeline.py             ◄ the "manager": calls every step in order
│   ├── config/
│   │   ├── settings.yaml       ◄ every adjustable number
│   │   └── settings.py         reads settings.yaml
│   ├── data/                   download prices & fundamentals, save them in cache/
│   ├── indicators/             RSI, MACD, ADX, … (trend / momentum / volatility / volume)
│   ├── normalization/          rescale every indicator to −1..+1 using past data only
│   ├── aggregation/            remove duplicate indicators, group them, detect trending markets
│   ├── fundamentals/           valuation vs history and peers, growth, earnings surprise
│   ├── models/                 GBM, GARCH, jumps, regimes, Kalman, mean reversion, earnings days
│   ├── stats/                  statistical tests, distributions, optimizer (written from scratch)
│   ├── synthesis/              Bayesian centre + model mix + Monte Carlo = the final forecast
│   ├── backtest/               the walk-forward "time machine"
│   ├── evaluation/             grading: coverage, CRPS, Brier, failure analysis, trading check
│   └── output/                 the text report and the PNG charts
├── tests/                      112 automatic checks (no internet needed)
├── cache/                      downloaded data (created automatically, not in git)
├── reports/                    your saved reports
├── requirements.txt
└── pyproject.toml
```

**New to the code?** Read these five files in order: `main.py` → `pipeline.py` (the `run_analysis` function) → `settings.yaml` → `models/gbm.py` (the simplest model) → `backtest/walk_forward.py`. Every file starts with a plain-English explanation at the top.

---

## 11. Running the tests

```bash
python -m pytest
```

- **112 tests**, about 13 seconds, and **no internet needed** (they use made-up price data with known answers).
- They cover data loading, every indicator, every model, the scoring maths, the backtest's lookahead protection, and the report.
- `tests/test_cross_check_libraries.py` compares the engine's hand-written maths with well-known libraries (`scipy`, `statsmodels`, `arch`) to confirm the answers match.

---

## 12. Speed

Measured on an Apple M5 laptop:

| Task | Time |
|---|---|
| Full analysis of one stock (8 years, 392 backtest forecasts) | about 20 seconds |
| Of which: re-fitting the models (79 times) | about 77% of the time |
| 100 stocks, one after another | about 38 minutes |
| 100 stocks, 8 at a time in parallel | about 10 minutes |

Download time from Yahoo is extra.

---

## 13. Known limitations

These are real, and they are listed so nobody is misled:

1. **No proven directional edge.** Use the engine for risk, not for picking winners.
2. **Tested on three large stocks so far.** They tend to move together, so the three results support each other less than three independent tests would. Wider testing on 50–100 NSE stocks is the next step.
3. **Momentum signals are currently unused.** RSI, CCI, Stochastic and Bollinger position are removed as duplicates of MACD and ADX on most stocks, so the momentum group is usually empty.
4. **Rare signals can have too much pull.** On stocks that are rarely in a trending market, the "trend while trending" signal can move the forecast too much.
5. **Forecasts shorter than 5 days are less reliable.** The signal limit (`signal_effect_prior_sd`) is sized for 5-day returns and isn't scaled for other horizons.
6. **Earnings dates are assumed to be known in advance.** In rare cases (0.5–2% of forecasts) a results date may not have been announced yet. Some older Yahoo earnings timestamps are placeholders.
7. **Data comes from Yahoo Finance.** It's free and occasionally wrong. Prices are adjusted for dividends and bonus issues, so old prices look different from what was traded at the time.
8. **The trading check is simplified.** It ignores tax, slippage, flat brokerage fees on small accounts, and changes in interest rates.
9. **Some calibration checks are noisy with few forecasts.** The "chance of going up" table can flag a problem purely by chance about 1 time in 5.

---

## 14. Technical specifications

| Item | Specification |
|---|---|
| Language | Python 3.12+ (developed and tested on 3.14.7) |
| Required libraries | pandas ≥ 3.0, numpy ≥ 2.0, yfinance ≥ 1.0, PyYAML ≥ 6.0, matplotlib, pytest ≥ 8.0 |
| Optional libraries (tests only) | scipy ≥ 1.12, statsmodels ≥ 0.14, arch ≥ 7.0 |
| Maths | Written from scratch with numpy (optimizer, distributions, statistical tests, all models) |
| Data source | Yahoo Finance via `yfinance`: daily prices adjusted for splits and dividends, earnings dates, company financials. Local CSV files are also supported (`data.source: csv`) |
| Data cache | Prices reused for 12 hours, fundamentals for 24 hours (`cache/`) |
| Data checks | No gaps, positive prices, consistent high/low values, holiday placeholder bars removed, at least 3 years of history required |
| Forecast output | 10,000 simulated 5-day returns → price ranges at 50/68/80/90/95%, median, P(up), P(move > ±2%, ±5%) |
| Reproducibility | Fixed random seed; identical results on every rerun with the same data |
| Operating systems | macOS, Linux, Windows |
| Interface | Command line (`python -m finance_engine.main SYMBOL`) |
| Outputs | Text report, JSON summary, CSV of every backtest forecast, 2 PNG charts |

---

## 15. Glossary

| Term | Meaning |
|---|---|
| **Backtest** | Testing a method on past data as if you were living through it |
| **Walk-forward** | A backtest where each forecast only uses data from before its own date |
| **Lookahead bias** | Accidentally using future information in a test, which makes results look better than they are |
| **Calibrated** | When the engine says 90%, it happens about 90% of the time |
| **Volatility** | How much a price typically moves; high volatility means bigger swings |
| **Monte Carlo** | Simulating thousands of random possible futures to see the spread of outcomes |
| **GBM / random walk** | The simplest model: random daily moves of constant size. Used as the baseline to beat |
| **GARCH** | A model where today's volatility depends on recent volatility |
| **Fat tails** | Extreme moves happen more often than a normal bell curve predicts |
| **Regime** | A market "mood", e.g. calm vs turbulent |
| **Bayesian** | Start from a sensible assumption and update it only as far as the evidence justifies |
| **CRPS** | A single score for how good a whole predicted range is (lower = better) |
| **Brier score** | A score for how good "chance of going up" numbers are (lower = better) |
| **p-value** | The chance a result this good would appear by pure luck; below 0.05 is usually treated as real |
| **Kelly sizing** | A formula for how much to invest based on expected gain and risk; "half-Kelly" invests half that, for safety |
| **Drawdown** | The biggest fall from a peak to a low |
| **Sharpe ratio** | Return earned per unit of risk taken; higher is better |

---

## 16. Disclaimer

This project is for **education and research only**. It is **not investment advice** and not a recommendation to buy or sell any security. Past performance, real or simulated, does not guarantee future results. In India, giving buy/sell recommendations to the public generally requires registration with SEBI as a Research Analyst or Investment Adviser. Market data comes from Yahoo Finance and is subject to its terms of use.
