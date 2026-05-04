# APMA 365 Final Project: Black-Scholes Validation

This repository evaluates Black-Scholes option pricing on AAPL OptionMetrics trial data (`2014-03-01` to `2014-03-15`), including:

- Black-Scholes predicted price vs observed option midpoint
- Newton-Raphson implied volatility vs historical volatility by tenor
- error diagnostics and volatility-range fit metrics (`R²`, adjusted `R²`)

## Setup

```bash
pip install -r requirements.txt
```

## Key Files

- `black_scholes.py` - call/put pricing, vega, implied-vol solver
- `optionmetrics.py` - WRDS/OptionMetrics query helpers and cleaning utilities
- `scripts/extract_ivydb_trial.py` - fetch and process AAPL trial dataset
- `scripts/backtest_black_scholes.py` - compute backtest outputs and summary metrics
- `scripts/plots.py` - reusable plotting utilities
- `scripts/generate_core_plots.py` - generate general market/chain diagnostic plots
- `scripts/generate_model_comparison_plots.py` - generate BS/NR comparison plots
- `main.py` - full pipeline (optional WRDS extract, backtest, all figures, metrics text)

## Data Pipeline

### 1) Extract trial data

```bash
python scripts/extract_ivydb_trial.py
```

Primary outputs:

- `data/processed/aapl_ivydb_trial_2014-03-01_2014-03-15.csv`
- `data/processed/aapl_historical_volatility_2014-03-01_2014-03-15.csv`

### 2) Run everything (recommended)

```bash
python main.py --min-midpoint 0.25 --max-rows 100000000
```

Optional WRDS pull first:

```bash
python main.py --extract --min-midpoint 0.25 --max-rows 100000000
```

Writes backtest CSVs, all figures under `figures/`, and a readable metrics file:

- `data/processed/analysis_metrics.txt`

### 3) Run backtest only (optional)

```bash
python scripts/backtest_black_scholes.py --min-volume 1 --min-midpoint 0.25 --max-rows 100000000
```

Outputs:

- `data/processed/aapl_bs_backtest_2014-03-01_2014-03-15.csv`
- `data/processed/aapl_bs_backtest_summary_2014-03-01_2014-03-15.csv` (includes `bs_price_mae` / `bs_price_rmse` / `bs_price_mape` and `nr_iv_mae` / `nr_iv_rmse` / `nr_iv_mape`)

## Plot Generation

### General core plots (spot, volume/open interest, volatility structure, liquidity)

```bash
python scripts/generate_core_plots.py
```

Generates figures in `figures/`, including:

- spot price over time
- option activity over time (daily volume, open interest, quote count)
- historical volatility time series and term structure
- implied-volatility smile/term-structure
- liquidity and price-by-strike views

### Model-comparison plots (BS/NR + diagnostics)

```bash
python scripts/generate_model_comparison_plots.py \
  --min-midpoint 0.25 \
  --max-rows 5000
```

Generates figures in `figures/model_comparison/`, including:

- Black Scholes predicted vs actual midpoint (with linear fit, `R²`, adjusted `R²`)
- Newton-Raphson derived implied volatility vs historical volatility by horizon
- BS and NR error diagnostics
- `R²` and adjusted `R²` by volatility range, split by calls vs puts

MAE / RMSE / MAPE are **not** plotted; they appear in `analysis_metrics.txt` and the backtest summary CSV.

## WRDS Notes

- Configure WRDS credentials outside the repo (e.g., `~/.pgpass`)
- schema/table names vary by subscription; override via `schema=` / `table=` args in `optionmetrics.py` helpers

## Tests

```bash
pytest
```
