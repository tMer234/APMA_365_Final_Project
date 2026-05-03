# APMA 365 Final Project: Black-Scholes PDE

Empirical validation of the Black-Scholes option pricing model. We derive the Black-Scholes PDE,
solve it via the Fourier transform, and compare model prices to historical market option data.

## Structure

- `black_scholes.py` — closed-form pricing functions (call, put, vega) and implied volatility solver
- `test_bs.py` — sanity checks against textbook values
- `optionmetrics.py` — utility access to OptionMetrics IvyDB US via WRDS (connection, validation, query helpers, light cleaning)
- `test_optionmetrics.py` — offline tests for the WRDS utilities
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

## OptionMetrics / WRDS data access

`optionmetrics.py` wraps the WRDS Python client for the empirical
validation. It contains no pricing logic — only connection management,
input validation, query helpers for IvyDB US tables, and light
post-query cleaning. Credentials are NEVER stored in this repository;
configure WRDS via `~/.pgpass` or interactive login as documented at
<https://wrds-www.wharton.upenn.edu/>. A username may also be provided
via the `WRDS_USERNAME` environment variable.

```python
import optionmetrics as om

conn = om.connect_wrds()                       # uses ~/.pgpass
chain = om.load_option_chain(
    conn, ["AAPL"], "2022-01-03", "2022-01-31",
    option_type="C", min_volume=10, max_spread_pct=0.5,
)
chain = om.add_time_to_maturity(chain)
spot  = om.load_underlying_prices(conn, ["AAPL"], "2022-01-03", "2022-01-31")
om.close_wrds(conn)
```

OptionMetrics WRDS schema/table names (`optionm.opprcd<YYYY>`,
`optionm.secnmd`, `optionm.secprd`, `optionm.zerocd`) can vary by
subscription; pass the `schema=` / `table=` keyword arguments on the
`load_*` functions to override the defaults.

## Report

See `report/` (LaTeX, generated separately on Overleaf).
