# APMA 365 Final Project: Black-Scholes PDE

Empirical validation of the Black-Scholes option pricing model. We derive the Black-Scholes PDE,
solve it via the Fourier transform, and compare model prices to historical market option data.

## Structure

- `black_scholes.py` — closed-form pricing functions (call, put, vega) and implied volatility solver
- `test_bs.py` — sanity checks against textbook values
- `data_loader.py` — pulls option chains and underlying prices via yfinance
- `volatility.py` — historical volatility estimation
- `analysis.py` — model vs. market comparison, error metrics, smile regressions
- `plots.py` — figures for the report
- `main.py` — end-to-end pipeline
- `notebook.ipynb` — exploratory analysis

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

## Report

See `report/` (LaTeX, generated separately on Overleaf).
