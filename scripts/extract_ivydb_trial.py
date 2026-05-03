"""Extract OptionMetrics IvyDB US Trial data for AAPL (2014-03-01 .. 2014-03-15).

The IvyDB US Trial subscription on WRDS is restricted to Apple Inc. (ticker
AAPL) and to the calendar window from 2014-03-01 through 2014-03-15
inclusive. This script connects to WRDS via the existing
:mod:`optionmetrics` helpers, pulls the option chain, the underlying daily
prices, and (when available) the zero-coupon yield curve, derives the
columns needed for downstream Black-Scholes analysis, applies conservative
quote cleaning, and writes a single tidy CSV to
``data/processed/aapl_ivydb_trial_2014-03-01_2014-03-15.csv``.

Authentication
--------------
This script never reads credentials from environment variables or the
command line. Configure WRDS via ``~/.pgpass`` or supply a username via
``WRDS_USERNAME`` and let the ``wrds`` package prompt interactively. See
``optionmetrics.connect_wrds`` for details.

Caveats specific to the trial range
-----------------------------------
* The trial window is only ~11 calendar days. There is **not** enough data
  in the trial alone to compute a meaningful historical-volatility lookback
  (e.g. 30/60/90 calendar days). The script does not invent prior data; if
  the user has access to a longer ``omtrial.secprd`` window, the
  ``--underlying-start`` flag can extend the underlying-price pull only.
* The risk-free / zero-curve table (``omtrial.zerocd``) may or may not be
  exposed to a given trial account. The script tries to load it and writes
  a separate ``aapl_ivydb_trial_zero_curve_*.csv`` if it succeeds; if the
  table is unavailable the script logs a clear message and continues
  without silently fabricating rates.
* OptionMetrics stores ``strike_price`` as ``strike * 1000``. The
  ``load_option_chain`` helper already rescales to dollars; the merged
  output therefore reports strikes in dollars.

Usage
-----
::

    python scripts/extract_ivydb_trial.py
    python scripts/extract_ivydb_trial.py --output data/processed/custom.csv
    python scripts/extract_ivydb_trial.py --max-spread-pct 1.0 --min-volume 0
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the repo root importable so ``import optionmetrics`` works whether
# the script is run as ``python scripts/extract_ivydb_trial.py`` or via
# ``python -m scripts.extract_ivydb_trial``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import optionmetrics as om  # noqa: E402


# ---------------------------------------------------------------------------
# Trial-data constants
# ---------------------------------------------------------------------------
TRIAL_TICKER: str = "AAPL"
TRIAL_START_DATE: str = "2014-03-01"
TRIAL_END_DATE: str = "2014-03-15"

DEFAULT_OUTPUT_PATH: Path = (
    _REPO_ROOT
    / "data"
    / "processed"
    / "aapl_ivydb_trial_2014-03-01_2014-03-15.csv"
)
DEFAULT_ZERO_CURVE_OUTPUT_PATH: Path = (
    _REPO_ROOT
    / "data"
    / "processed"
    / "aapl_ivydb_trial_zero_curve_2014-03-01_2014-03-15.csv"
)

# Columns we ask the option-price table for. The trial table does not
# guarantee Greeks, but we request them so they survive into the output if
# present. Anything the helper rejects will surface as a clear SQL error,
# at which point the caller can prune via --option-columns.
TRIAL_OPTION_COLUMNS: tuple[str, ...] = (
    "secid",
    "date",
    "exdate",
    "cp_flag",
    "strike_price",
    "best_bid",
    "best_offer",
    "volume",
    "open_interest",
    "impl_volatility",
    "delta",
    "gamma",
    "vega",
    "theta",
    "optionid",
)

# Fallback projection if Greeks are not exposed in the trial schema.
TRIAL_OPTION_COLUMNS_NO_GREEKS: tuple[str, ...] = (
    "secid",
    "date",
    "exdate",
    "cp_flag",
    "strike_price",
    "best_bid",
    "best_offer",
    "volume",
    "open_interest",
    "impl_volatility",
    "optionid",
)

logger = logging.getLogger("extract_ivydb_trial")


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------
def default_output_path(
    ticker: str = TRIAL_TICKER,
    start: str = TRIAL_START_DATE,
    end: str = TRIAL_END_DATE,
    *,
    base_dir: Path | None = None,
) -> Path:
    """Build the canonical output CSV path for a trial extraction.

    Pure helper so tests can verify the naming scheme without touching
    WRDS or the filesystem.
    """
    base = base_dir if base_dir is not None else _REPO_ROOT / "data" / "processed"
    return base / f"{ticker.lower()}_ivydb_trial_{start}_{end}.csv"


# ---------------------------------------------------------------------------
# Cleaning / feature engineering (DB-free, unit-testable)
# ---------------------------------------------------------------------------
def add_derived_columns(
    options: pd.DataFrame,
    underlying: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge spot prices and add derived analytic columns.

    Adds ``midpoint`` (if not already present from cleaning), ``spot``
    (close from the underlying daily price table joined on
    ``secid+date``), ``bid_ask_spread``, ``bid_ask_spread_pct``,
    ``days_to_expiry``, ``time_to_maturity_years``, ``moneyness`` (S/K)
    and ``log_moneyness``. Idempotent — if a derived column is already
    present and finite, it is overwritten with the freshly-computed
    value.

    Returns a new DataFrame; the input is not mutated.
    """
    if options is None or len(options) == 0:
        return pd.DataFrame() if options is None else options.copy()

    out = options.copy()
    out["date"] = pd.to_datetime(out["date"])
    if "exdate" in out.columns:
        out["exdate"] = pd.to_datetime(out["exdate"])

    if "midpoint" not in out.columns and {"best_bid", "best_offer"}.issubset(out.columns):
        out["midpoint"] = (out["best_bid"] + out["best_offer"]) / 2.0

    if {"best_bid", "best_offer"}.issubset(out.columns):
        out["bid_ask_spread"] = out["best_offer"] - out["best_bid"]
        with np.errstate(divide="ignore", invalid="ignore"):
            out["bid_ask_spread_pct"] = np.where(
                out["midpoint"] > 0,
                out["bid_ask_spread"] / out["midpoint"],
                np.nan,
            )

    if "exdate" in out.columns:
        out["days_to_expiry"] = (out["exdate"] - out["date"]).dt.days
        out["time_to_maturity_years"] = out["days_to_expiry"] / 365.0

    if underlying is not None and len(underlying) > 0:
        u = underlying[["secid", "date", "close"]].copy()
        u["date"] = pd.to_datetime(u["date"])
        u = u.rename(columns={"close": "spot"})
        # If multiple rows per (secid, date), take the last — defensive
        # for noisy trial data.
        u = u.drop_duplicates(subset=["secid", "date"], keep="last")
        out = out.merge(u, on=["secid", "date"], how="left")

    if "spot" in out.columns and "strike_price" in out.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["moneyness"] = out["spot"] / out["strike_price"]
            out["log_moneyness"] = np.where(
                out["moneyness"] > 0, np.log(out["moneyness"]), np.nan
            )

    if "cp_flag" in out.columns:
        out["cp_flag"] = out["cp_flag"].astype(str).str.upper().str.strip()

    return out


# ---------------------------------------------------------------------------
# Live extraction
# ---------------------------------------------------------------------------
def _try_load_option_chain(
    conn,
    *,
    ticker: str,
    start: str,
    end: str,
    min_volume: int | None,
    max_spread_pct: float | None,
    schema: str,
) -> pd.DataFrame:
    """Load the option chain, falling back if Greeks aren't exposed."""
    try:
        return om.load_option_chain(
            conn,
            [ticker],
            start,
            end,
            min_volume=min_volume,
            max_spread_pct=max_spread_pct,
            columns=list(TRIAL_OPTION_COLUMNS),
            schema=schema,
        )
    except Exception as exc:  # pragma: no cover - depends on remote schema
        logger.warning(
            "Greek columns not available in trial schema (%s); "
            "retrying without delta/gamma/vega/theta.",
            exc,
        )
        return om.load_option_chain(
            conn,
            [ticker],
            start,
            end,
            min_volume=min_volume,
            max_spread_pct=max_spread_pct,
            columns=list(TRIAL_OPTION_COLUMNS_NO_GREEKS),
            schema=schema,
        )


def run_extraction(
    output_path: Path,
    *,
    ticker: str = TRIAL_TICKER,
    start: str = TRIAL_START_DATE,
    end: str = TRIAL_END_DATE,
    underlying_start: str | None = None,
    min_volume: int | None = 0,
    max_spread_pct: float | None = None,
    schema: str = om.DEFAULT_SCHEMA,
    zero_curve_output: Path | None = DEFAULT_ZERO_CURVE_OUTPUT_PATH,
) -> pd.DataFrame:
    """Connect to WRDS, pull the trial data, write the CSV, and return it.

    ``underlying_start`` lets callers extend the spot-price lookback
    independently of the option-chain window — useful if the trial has
    extra prior daily prices, or to clearly confirm that it does not.

    Returns the merged option-level DataFrame that was written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to WRDS (schema=%s) …", schema)
    conn = om.connect_wrds()
    try:
        logger.info(
            "Loading %s option chain %s .. %s", ticker, start, end
        )
        options = _try_load_option_chain(
            conn,
            ticker=ticker,
            start=start,
            end=end,
            min_volume=min_volume,
            max_spread_pct=max_spread_pct,
            schema=schema,
        )
        logger.info("Option-chain rows after cleaning: %d", len(options))

        u_start = underlying_start or start
        logger.info(
            "Loading %s underlying daily prices %s .. %s",
            ticker, u_start, end,
        )
        underlying = om.load_underlying_prices(
            conn, [ticker], u_start, end, schema=schema,
        )
        logger.info("Underlying rows: %d", len(underlying))

        # Try the zero curve. The trial may not expose this; we degrade
        # to a clear log line rather than fabricating rates.
        if zero_curve_output is not None:
            try:
                zc = om.load_zero_curve_or_rates(conn, start, end, schema=schema)
                if len(zc) > 0:
                    zero_curve_output.parent.mkdir(parents=True, exist_ok=True)
                    zc.to_csv(zero_curve_output, index=False)
                    logger.info(
                        "Zero curve rows: %d → %s", len(zc), zero_curve_output,
                    )
                else:
                    logger.warning(
                        "Zero curve query returned 0 rows; not writing %s. "
                        "TODO: configure a fallback risk-free source for "
                        "Black-Scholes analysis.",
                        zero_curve_output,
                    )
            except Exception as exc:  # pragma: no cover - depends on remote
                logger.warning(
                    "Zero curve unavailable in trial schema (%s). "
                    "TODO: supply a risk-free source (FRED DGS3MO, etc.) "
                    "from the analysis layer.",
                    exc,
                )
    finally:
        om.close_wrds(conn)

    merged = add_derived_columns(options, underlying)
    merged.to_csv(output_path, index=False)
    logger.info("Wrote %d rows to %s", len(merged), output_path)
    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH,
        help="Path to write the cleaned, merged option-level CSV.",
    )
    p.add_argument(
        "--ticker", default=TRIAL_TICKER,
        help="Ticker (trial only contains AAPL).",
    )
    p.add_argument("--start", default=TRIAL_START_DATE)
    p.add_argument("--end", default=TRIAL_END_DATE)
    p.add_argument(
        "--underlying-start", default=None,
        help="Optional earlier start date for the underlying-price pull.",
    )
    p.add_argument(
        "--min-volume", type=int, default=0,
        help="Minimum option volume to retain (default 0 = keep all).",
    )
    p.add_argument(
        "--max-spread-pct", type=float, default=None,
        help="Maximum (ask-bid)/midpoint to retain. Default: no filter.",
    )
    p.add_argument(
        "--schema", default=om.DEFAULT_SCHEMA,
        help=f"WRDS schema (default {om.DEFAULT_SCHEMA!r}).",
    )
    p.add_argument(
        "--no-zero-curve", action="store_true",
        help="Skip the zero-curve query entirely.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        help="Python logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.ticker.upper() != TRIAL_TICKER:
        logger.warning(
            "IvyDB US Trial only contains %s data; %s is unlikely to return rows.",
            TRIAL_TICKER, args.ticker,
        )
    run_extraction(
        output_path=args.output,
        ticker=args.ticker.upper(),
        start=args.start,
        end=args.end,
        underlying_start=args.underlying_start,
        min_volume=args.min_volume,
        max_spread_pct=args.max_spread_pct,
        schema=args.schema,
        zero_curve_output=None if args.no_zero_curve else DEFAULT_ZERO_CURVE_OUTPUT_PATH,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
