# APMA 365 Final Project: Black-Scholes Validation (AAPL)

Empirical comparison of **Black-Scholes** option prices and **Newton-Raphson implied volatility** to market data from the **OptionMetrics IvyDB US Trial**: Apple (**AAPL**) only, **2014-03-01** through **2014-03-15** (calendar dates fixed by the trial subscription).

The project:

- Pulls and cleans option chains and historical volatility (optional WRDS step).
- Prices options with Black-Scholes using **historical volatility** matched to the nearest OptionMetrics tenor by calendar day.
- Solves for **implied volatility** with a **Newton-Raphson** root finder (European call/put formulas).
- Reports **MAE**, **RMSE**, and **MAPE** for price errors (BS vs midpoint) and for IV errors (NR IV vs historical IV, where NR converges).
- Produces **core** exploratory figures and **model-comparison** diagnostics (including `R²` by volatility bucket).

---

## Requirements

- Python 3.10+ recommended  
- Dependencies:

```bash
pip install -r requirements.txt
```

Packages: `numpy`, `scipy`, `pandas`, `matplotlib`, `wrds`, `pytest`, `python-dotenv`, plus `yfinance` / `jupyter` as listed.

---

## Quick start (full pipeline)

With processed CSVs already under `data/processed/`:

```bash
python main.py --min-midpoint 0.25 --max-rows 100000000
```

To **refresh data from WRDS** first (requires `~/.pgpass` or interactive WRDS login and `WRDS_USERNAME` if you use it):

```bash
python main.py --extract --min-midpoint 0.25 --max-rows 100000000
```

This will:

1. Optionally run extraction (`--extract`).
2. Run the backtest and write CSVs.
3. Regenerate **all** figures (core + `figures/model_comparison/`).
4. Print metrics to stdout and write `data/processed/analysis_metrics.txt`.

---

## `main.py` reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--extract` | off | If set, runs WRDS extraction into `--options` path (and related trial outputs). |
| `--options` | `data/processed/aapl_ivydb_trial_2014-03-01_2014-03-15.csv` | Processed option-chain CSV. |
| `--historical-vol` | `data/processed/aapl_historical_volatility_2014-03-01_2014-03-15.csv` | Historical vol CSV. |
| `--backtest-out` | `data/processed/aapl_bs_backtest_2014-03-01_2014-03-15.csv` | Per-contract backtest rows. |
| `--summary-out` | `data/processed/aapl_bs_backtest_summary_2014-03-01_2014-03-15.csv` | One-row-per-metric summary. |
| `--figures-dir` | `figures` | Root for PNGs; model plots go to `figures/model_comparison/`. |
| `--metrics-out` | `data/processed/analysis_metrics.txt` | Human-readable metrics table. |
| `--risk-free-rate` | `0.01` | Continuously compounded rate used in BS and NR. |
| `--min-volume` | `1` | Drop option rows with volume below this before backtest. |
| `--max-rows` | `100000000` | Cap rows after filters (after sorting by volume if capped). |
| `--min-midpoint` | `0.25` | Exclude quotes with midpoint below this (filters illiquid / penny quotes). |
| `--log-level` | `INFO` | Logging level when `--extract` is used. |

---

## Repository layout

| Path | Role |
|------|------|
| `main.py` | Single entrypoint: extract (optional), backtest, figures, metrics file. |
| `black_scholes.py` | European call/put price, vega, and NR-style implied-vol solver. |
| `optionmetrics.py` | WRDS connection helpers, validation, option chain / prices / zero-curve loaders, quote cleaning. |
| `scripts/extract_ivydb_trial.py` | Trial-specific WRDS pull and CSV writes. |
| `scripts/backtest_black_scholes.py` | Merge HV tenors, compute BS prices, NR IV, errors, summary metrics. |
| `scripts/plots.py` | All matplotlib helpers (core + model comparison). |
| `scripts/generate_core_plots.py` | CLI wrapper for core figures only. |
| `scripts/generate_model_comparison_plots.py` | CLI wrapper for model figures; exposes `generate_model_comparison_figures()`. |
| `tests/` | `pytest` tests for BS, OptionMetrics helpers, and extraction helpers. |
| `data/processed/` | Processed CSVs and `analysis_metrics.txt` (large CSVs are gitignored by default). |
| `figures/` | PNG outputs (often gitignored). |

---

## Data outputs

### Extraction (`scripts/extract_ivydb_trial.py` or `main.py --extract`)

- **`aapl_ivydb_trial_2014-03-01_2014-03-15.csv`** — Option chain joined to underlying close as `spot`, plus midpoint, spreads, DTE, moneyness, and optional Greeks / OptionMetrics implied vol if the schema exposes them.
- **`aapl_historical_volatility_2014-03-01_2014-03-15.csv`** — Columns include `date`, `days`, `volatility` (and identifiers as written by the script).
- **`aapl_ivydb_trial_zero_curve_2014-03-01_2014-03-15.csv`** — Written only if the trial schema exposes the zero curve.

### Backtest (`scripts/backtest_black_scholes.py` or `main.py`)

**Option-level file** (`aapl_bs_backtest_*.csv`) — lean columns used for plots and metrics:

`cp_flag`, `midpoint`, `strike_price`, `spot`, `time_to_maturity_years`, `hv_tenor_days`, `historical_volatility`, `tenor_gap_days`, `bs_price_hv`, `bs_error`, `nr_implied_vol`, `iv_error_vs_hv`

**Summary file** (`aapl_bs_backtest_summary_*.csv`) — long format `metric`, `value`:

| Metric | Meaning |
|--------|---------|
| `n_quotes` | Rows after filters. |
| `n_nr_iv` | Rows with finite NR implied volatility (used for NR IV metrics). |
| `bs_price_mae` | Mean absolute error, \(\|\hat{C}-P\|\) (BS price \(\hat{C}\) vs midpoint \(P\)). |
| `bs_price_rmse` | RMSE of \((\hat{C}-P)\). |
| `bs_price_mape` | Mean \(\|\hat{C}-P\|/P\) (relative to midpoint). |
| `nr_iv_mae` | Mean \(\|\sigma_{NR}-\sigma_{HV}\|\) on rows with finite \(\sigma_{NR}\) and \(\sigma_{HV}>0\). |
| `nr_iv_rmse` | RMSE of \((\sigma_{NR}-\sigma_{HV})\). |
| `nr_iv_mape` | Mean \(\|\sigma_{NR}-\sigma_{HV}\|/\sigma_{HV}\). |
| `median_tenor_gap_days` | Median \(\lvert \text{HV tenor days} - \text{DTE}\rvert\) for matched rows. |

**Text metrics** — `data/processed/analysis_metrics.txt` duplicates the summary in a fixed-width block for reports.

MAE / RMSE / MAPE are **not** drawn as separate bar charts; use the summary CSV or `analysis_metrics.txt`.

---

## Figures

### Core (`figures/`)

Generated by `main.py` or `python scripts/generate_core_plots.py`:

| File | Content |
|------|---------|
| `spot_over_time.png` | Mean `spot` by trade date. |
| `option_activity_over_time.png` | Daily summed volume / open interest and quote count. |
| `historical_volatility_30d.png` | 30-day historical vol time series. |
| `price_vs_strike_calls.png` | Call midpoint vs strike on last date in sample; scatter colored by **days to expiry**; dashed **intrinsic** \((S-K)^+\). |
| `price_vs_strike_puts.png` | Same for puts; intrinsic \((K-S)^+\). |

### Model comparison (`figures/model_comparison/`)

Generated by `main.py` or `python scripts/generate_model_comparison_plots.py`:

| File | Content |
|------|---------|
| `bs_predicted_vs_actual.png` | BS price vs midpoint; calls vs puts; linear fit with \(R^2\). |
| `nr_iv_vs_historical_by_hv_tenor.png` | Median NR IV vs median historical IV by matched historical-volatility tenor (`hv_tenor_days`). |
| `bs_error_hist.png` / `bs_error_vs_price.png` | BS price error distribution and vs midpoint. |
| `iv_error_hist.png` / `iv_error_by_hv_tenor.png` | **Absolute** NR IV minus historical IV (histogram and median by historical-volatility tenor). |
| `r2_by_volatility_range_calls_puts.png` | \(R^2\) of BS price vs midpoint within historical-vol quantile bins, calls and puts. |

---

## WRDS and configuration

- Install the **`wrds`** client and configure credentials (typically **`~/.pgpass`**). Do not commit secrets; optional **`WRDS_USERNAME`** is read by `optionmetrics.connect_wrds()`.
- Default IvyDB trial schema is **`omtrial`** (see `optionmetrics.DEFAULT_SCHEMA`). Table names may differ by subscription; override via keyword arguments on the loaders in `optionmetrics.py` if needed.
- Extraction uses **`python-dotenv`** so a local **`.env`** can hold non-secret paths or flags only if you choose (never commit real passwords).

---

## Tests

```bash
pytest
```

---

## License / course use

Use and adapt for APMA 365 reporting; cite OptionMetrics / WRDS and trial data restrictions where required by your institution.
