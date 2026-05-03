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

## IvyDB US Trial extraction (AAPL, 2014-03-01 .. 2014-03-15)

`scripts/extract_ivydb_trial.py` is a reproducible end-to-end pull of the
IvyDB US **Trial** subscription: AAPL only, calendar window
`2014-03-01` through `2014-03-15` inclusive (the trial does not contain
any other ticker or date). Configure WRDS as above, then run:

```bash
python scripts/extract_ivydb_trial.py
```

Output is written to
`data/processed/aapl_ivydb_trial_2014-03-01_2014-03-15.csv` and contains
the cleaned option chain merged with daily underlying close prices and
the columns required for Black-Scholes empirical analysis:

* identifiers: `secid`, `ticker`, `optionid`, `cp_flag`
* dates: `date`, `exdate`, `days_to_expiry`, `time_to_maturity_years`
* quotes: `best_bid`, `best_offer`, `midpoint`, `bid_ask_spread`,
  `bid_ask_spread_pct`, `volume`, `open_interest`
* contract / underlying: `strike_price` (in dollars; OptionMetrics'
  ×1000 integer scaling is undone by `optionmetrics.load_option_chain`),
  `spot`, `moneyness` (S/K), `log_moneyness`
* OptionMetrics analytics if exposed by the trial schema:
  `implied_volatility`, `delta`, `gamma`, `vega`, `theta`

If the trial schema does not expose Greeks, the script automatically
retries with a Greek-free projection.

Useful flags:

```bash
python scripts/extract_ivydb_trial.py \
    --max-spread-pct 1.0 \
    --min-volume 0 \
    --output data/processed/aapl_trial.csv
```

Caveats specific to the trial range:

* The 11-calendar-day window is not enough to compute a meaningful
  historical-volatility lookback. Pass `--underlying-start` if you have
  access to a longer `omtrial.secprd` range; otherwise use a fallback
  source (e.g. yfinance) from the analysis layer.
* The risk-free zero-coupon curve (`omtrial.zerocd`) may not be exposed
  in every trial account. The script tries it and writes
  `data/processed/aapl_ivydb_trial_zero_curve_*.csv` on success;
  otherwise it logs a clear `TODO` and continues without fabricating
  rates.

## Report

See `report/` (LaTeX, generated separately on Overleaf).
