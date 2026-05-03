"""Utility module for accessing OptionMetrics IvyDB US data via WRDS.

This module provides thin wrappers around the WRDS Python client (``wrds``)
for the empirical Black-Scholes validation project. It is intentionally
narrow in scope: connection management, input validation, query helpers
for the standard OptionMetrics IvyDB tables (option chain, security
metadata, underlying daily prices, zero-coupon yield curve), and light
post-query cleaning utilities. It does NOT implement option pricing,
implied volatility solvers, statistical tests, or plotting; those belong
in ``black_scholes.py``, ``volatility.py``, ``analysis.py``, ``plots.py``.

Authentication
--------------
The WRDS Python client reads credentials from a ``~/.pgpass`` file on
first use, or interactively. This module never stores or logs
credentials. Username may be passed explicitly or supplied via the
``WRDS_USERNAME`` environment variable; passwords are NOT read from
environment variables to avoid accidental leakage to shell history /
process listings.

Schema and table names
----------------------
OptionMetrics IvyDB US table names on WRDS commonly look like
``optionm.opprcd<YYYY>`` (yearly option price tables),
``optionm.secnmd`` (security name dictionary), ``optionm.secprd``
(daily security prices), and ``optionm.zerocd`` (zero-coupon yield
curve). The exact names can drift between WRDS schema revisions
(``optionm`` vs. ``optionm_all`` vs. ``optionmetrics``). The module
exposes these as module-level constants and as optional keyword
arguments so future analysis code or tests can override them without
editing this file.

Example
-------
>>> import optionmetrics as om
>>> conn = om.connect_wrds()                       # uses ~/.pgpass
>>> ids = om.get_security_ids(conn, ["AAPL", "SPY"])
>>> chain = om.load_option_chain(
...     conn, ["AAPL"], "2022-01-03", "2022-01-31",
...     option_type="C", min_volume=10, max_spread_pct=0.5,
... )
>>> chain = om.add_time_to_maturity(chain)
>>> spot = om.load_underlying_prices(conn, ["AAPL"], "2022-01-03", "2022-01-31")
>>> om.close_wrds(conn)
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# ``wrds`` is an optional runtime dependency: validation, cleaning, and
# math helpers are usable without it (and without a database connection),
# which keeps unit tests cheap. Connection helpers will raise a clear
# error if the package is missing.
try:  # pragma: no cover - exercised only when wrds is installed
    import wrds  # type: ignore
except Exception:  # pragma: no cover
    wrds = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Default OptionMetrics IvyDB US table names on WRDS.
# Override via keyword arguments to load_* functions if your WRDS schema
# differs. These are documented defaults, not guarantees.
# ---------------------------------------------------------------------------
DEFAULT_SCHEMA: str = "optionm"
DEFAULT_OPTION_PRICE_TABLE_FMT: str = "opprcd{year}"  # yearly partition
DEFAULT_SECURITY_NAME_TABLE: str = "secnmd"
DEFAULT_SECURITY_PRICE_TABLE: str = "secprd"
DEFAULT_ZERO_CURVE_TABLE: str = "zerocd"

DEFAULT_OPTION_COLUMNS: tuple[str, ...] = (
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
DEFAULT_UNDERLYING_COLUMNS: tuple[str, ...] = (
    "secid",
    "date",
    "close",
    "open",
    "high",
    "low",
    "volume",
    "return",
)
DEFAULT_ZERO_CURVE_COLUMNS: tuple[str, ...] = ("date", "days", "rate")

# OptionMetrics stores strike_price * 1000 as an integer; expose the
# scaling so callers don't have to remember it.
STRIKE_PRICE_SCALE: int = 1000

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------
def connect_wrds(username: str | None = None, **kwargs):
    """Open a WRDS connection.

    Parameters
    ----------
    username:
        WRDS username. If ``None``, falls back to the ``WRDS_USERNAME``
        environment variable, then to the ``wrds`` library's own
        resolution (``~/.pgpass`` / interactive prompt).
    **kwargs:
        Forwarded to :class:`wrds.Connection` (e.g. ``autoconnect``).

    Returns
    -------
    wrds.Connection
        A live connection object. Caller is responsible for closing it
        via :func:`close_wrds`.

    Raises
    ------
    ImportError
        If the ``wrds`` package is not installed.
    """
    if wrds is None:
        raise ImportError(
            "The 'wrds' package is required for connect_wrds(). "
            "Install with `pip install wrds` and configure ~/.pgpass."
        )
    if username is None:
        username = os.environ.get("WRDS_USERNAME")
    if username is not None:
        kwargs.setdefault("wrds_username", username)
    return wrds.Connection(**kwargs)


def close_wrds(conn) -> None:
    """Close a WRDS connection if it exposes a ``close`` method.

    Safe to call with ``None`` or with already-closed connections.
    """
    if conn is None:
        return
    close = getattr(conn, "close", None)
    if callable(close):
        close()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def validate_date_range(
    start_date: str | _dt.date | _dt.datetime | pd.Timestamp,
    end_date: str | _dt.date | _dt.datetime | pd.Timestamp,
) -> tuple[_dt.date, _dt.date]:
    """Validate and normalize a (start, end) date pair.

    Returns a tuple of ``datetime.date`` with ``start <= end``. Accepts
    strings parseable by :func:`pandas.Timestamp`.
    """
    try:
        start = pd.Timestamp(start_date).date()
        end = pd.Timestamp(end_date).date()
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Could not parse date range: {exc}") from exc
    if start > end:
        raise ValueError(f"start_date ({start}) must be <= end_date ({end})")
    return start, end


def validate_tickers(tickers: str | Iterable[str]) -> list[str]:
    """Validate and normalize ticker symbols.

    Accepts a single string or an iterable of strings. Returns a sorted,
    de-duplicated list of upper-cased tickers. Rejects values that
    contain characters outside ``[A-Z0-9.-]`` to keep query inputs
    safe for parameterized SQL substitution.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    out: list[str] = []
    seen: set[str] = set()
    for raw in tickers:
        if not isinstance(raw, str):
            raise TypeError(f"Ticker must be str, got {type(raw).__name__}")
        t = raw.strip().upper()
        if not t:
            raise ValueError("Empty ticker is not allowed")
        if not _TICKER_RE.match(t):
            raise ValueError(
                f"Ticker {raw!r} contains unexpected characters; "
                "expected [A-Z0-9.-], length 1-10."
            )
        if t not in seen:
            seen.add(t)
            out.append(t)
    if not out:
        raise ValueError("At least one ticker is required")
    return sorted(out)


def _qualified(schema: str, table: str) -> str:
    """Return ``schema.table`` with safe identifier characters only."""
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", schema):
        raise ValueError(f"Invalid schema name: {schema!r}")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table):
        raise ValueError(f"Invalid table name: {table!r}")
    return f"{schema}.{table}"


def _select_clause(columns: Sequence[str] | None, default: Sequence[str]) -> str:
    cols = list(columns) if columns else list(default)
    for c in cols:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", c):
            raise ValueError(f"Invalid column name: {c!r}")
    return ", ".join(cols)


# ---------------------------------------------------------------------------
# Metadata: secid lookups
# ---------------------------------------------------------------------------
def get_security_ids(
    conn,
    tickers: str | Iterable[str],
    start_date: str | _dt.date | None = None,
    end_date: str | _dt.date | None = None,
    *,
    schema: str = DEFAULT_SCHEMA,
    table: str = DEFAULT_SECURITY_NAME_TABLE,
) -> pd.DataFrame:
    """Look up OptionMetrics ``secid`` values for a list of tickers.

    Returns a :class:`pandas.DataFrame` with columns
    ``[secid, ticker, issuer, effect_date, exp_date]`` (when present in
    the underlying table). If ``start_date`` / ``end_date`` are given,
    rows are filtered to security-name records overlapping that window.

    Notes
    -----
    A given ticker can map to multiple secids over time as issuers /
    listings change. Callers should join on ``secid`` for the trade
    date of interest rather than assume a 1:1 ticker→secid mapping.
    """
    tickers = validate_tickers(tickers)
    qtable = _qualified(schema, table)
    sql = f"SELECT secid, ticker, issuer, effect_date, exp_date FROM {qtable} WHERE ticker IN %(tickers)s"
    params: dict[str, object] = {"tickers": tuple(tickers)}
    if start_date is not None and end_date is not None:
        start, end = validate_date_range(start_date, end_date)
        sql += " AND (exp_date IS NULL OR exp_date >= %(start)s) AND effect_date <= %(end)s"
        params["start"] = start
        params["end"] = end
    sql += " ORDER BY ticker, effect_date"
    return conn.raw_sql(sql, params=params)


# ---------------------------------------------------------------------------
# Option chain
# ---------------------------------------------------------------------------
def load_option_chain(
    conn,
    tickers: str | Iterable[str],
    start_date: str | _dt.date,
    end_date: str | _dt.date,
    option_type: str | None = None,
    min_volume: int | None = None,
    max_spread_pct: float | None = None,
    columns: Sequence[str] | None = None,
    *,
    schema: str = DEFAULT_SCHEMA,
    option_table_fmt: str = DEFAULT_OPTION_PRICE_TABLE_FMT,
    name_table: str = DEFAULT_SECURITY_NAME_TABLE,
) -> pd.DataFrame:
    """Load the option chain for ``tickers`` between ``start_date`` and ``end_date``.

    OptionMetrics partitions the option price file by year
    (``opprcd2020``, ``opprcd2021``, ...). This helper unions the
    relevant yearly tables, joins to the security name dictionary so
    the result carries ``ticker`` alongside ``secid``, and applies
    optional bid/ask/volume filters via :func:`clean_option_quotes`.

    Strike prices are rescaled from OptionMetrics' integer
    representation to dollars (divided by ``STRIKE_PRICE_SCALE``).

    Parameters
    ----------
    option_type:
        ``"C"`` for calls, ``"P"`` for puts, or ``None`` for both.
    min_volume, max_spread_pct:
        Forwarded to :func:`clean_option_quotes` after the query.
    columns:
        Override the default projection. Always includes ``secid`` and
        ``date`` after de-duplication so joins remain valid.
    """
    tickers = validate_tickers(tickers)
    start, end = validate_date_range(start_date, end_date)
    if option_type is not None:
        option_type = option_type.upper()
        if option_type not in {"C", "P"}:
            raise ValueError("option_type must be 'C', 'P', or None")

    cols = list(columns) if columns else list(DEFAULT_OPTION_COLUMNS)
    for required in ("secid", "date"):
        if required not in cols:
            cols.insert(0, required)
    select_cols = _select_clause(cols, cols)

    name_qtable = _qualified(schema, name_table)
    years = list(range(start.year, end.year + 1))
    union_parts: list[str] = []
    params: dict[str, object] = {
        "tickers": tuple(tickers),
        "start": start,
        "end": end,
    }
    for year in years:
        opt_table = _qualified(schema, option_table_fmt.format(year=year))
        prefixed = ", ".join(f"o.{c}" for c in cols)
        part = (
            f"SELECT {prefixed}, n.ticker "
            f"FROM {opt_table} o "
            f"JOIN {name_qtable} n ON n.secid = o.secid "
            f"WHERE n.ticker IN %(tickers)s "
            f"AND o.date BETWEEN %(start)s AND %(end)s"
        )
        if option_type is not None:
            part += " AND o.cp_flag = %(cp_flag)s"
            params["cp_flag"] = option_type
        union_parts.append(part)
    sql = " UNION ALL ".join(union_parts) + " ORDER BY ticker, date, exdate, strike_price"

    df = conn.raw_sql(sql, params=params)
    if df.empty:
        return df

    if "strike_price" in df.columns:
        df["strike_price"] = pd.to_numeric(df["strike_price"]) / STRIKE_PRICE_SCALE

    df = clean_option_quotes(
        df,
        min_volume=min_volume,
        max_spread_pct=max_spread_pct,
        require_positive_bid=True,
    )
    if "impl_volatility" in df.columns:
        df = df.rename(columns={"impl_volatility": "implied_volatility"})
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Underlying daily prices
# ---------------------------------------------------------------------------
def load_underlying_prices(
    conn,
    tickers: str | Iterable[str],
    start_date: str | _dt.date,
    end_date: str | _dt.date,
    columns: Sequence[str] | None = None,
    *,
    schema: str = DEFAULT_SCHEMA,
    price_table: str = DEFAULT_SECURITY_PRICE_TABLE,
    name_table: str = DEFAULT_SECURITY_NAME_TABLE,
) -> pd.DataFrame:
    """Load daily underlying prices suitable for historical volatility.

    Joins ``optionm.secprd`` to ``optionm.secnmd`` so the result carries
    ``ticker`` for downstream grouping. The default column set includes
    OHLC, volume, and the OptionMetrics-supplied total return.
    """
    tickers = validate_tickers(tickers)
    start, end = validate_date_range(start_date, end_date)
    cols = list(columns) if columns else list(DEFAULT_UNDERLYING_COLUMNS)
    for required in ("secid", "date"):
        if required not in cols:
            cols.insert(0, required)
    prefixed = ", ".join(f"p.{c}" for c in cols)
    price_qtable = _qualified(schema, price_table)
    name_qtable = _qualified(schema, name_table)
    sql = (
        f"SELECT {prefixed}, n.ticker "
        f"FROM {price_qtable} p "
        f"JOIN {name_qtable} n ON n.secid = p.secid "
        f"WHERE n.ticker IN %(tickers)s "
        f"AND p.date BETWEEN %(start)s AND %(end)s "
        f"ORDER BY ticker, date"
    )
    params = {"tickers": tuple(tickers), "start": start, "end": end}
    return conn.raw_sql(sql, params=params)


# ---------------------------------------------------------------------------
# Zero-coupon yield curve
# ---------------------------------------------------------------------------
def load_zero_curve_or_rates(
    conn,
    start_date: str | _dt.date,
    end_date: str | _dt.date,
    columns: Sequence[str] | None = None,
    *,
    schema: str = DEFAULT_SCHEMA,
    table: str | None = DEFAULT_ZERO_CURVE_TABLE,
) -> pd.DataFrame:
    """Load the OptionMetrics zero-coupon yield curve over a date range.

    The IvyDB zero curve table is conventionally ``optionm.zerocd`` with
    columns ``(date, days, rate)``: continuously-compounded rate at a
    given maturity in days. If your WRDS subscription does not include
    this table, pass ``table=None`` to raise a clear
    :class:`NotImplementedError` instructing which table to configure;
    callers can then fall back to FRED or another source from the
    analysis layer.
    """
    if table is None:
        raise NotImplementedError(
            "Zero-coupon curve table is not configured. Pass table='zerocd' "
            "(or your WRDS-provided OptionMetrics curve table name, e.g. "
            "'optionm_all.zerocd') to load_zero_curve_or_rates(). If "
            "OptionMetrics IvyDB rates are unavailable in your WRDS "
            "subscription, use a FRED-based fallback (e.g. DGS3MO) "
            "from the analysis layer instead."
        )
    start, end = validate_date_range(start_date, end_date)
    cols = list(columns) if columns else list(DEFAULT_ZERO_CURVE_COLUMNS)
    select_cols = _select_clause(cols, cols)
    qtable = _qualified(schema, table)
    sql = (
        f"SELECT {select_cols} FROM {qtable} "
        f"WHERE date BETWEEN %(start)s AND %(end)s ORDER BY date, days"
    )
    return conn.raw_sql(sql, params={"start": start, "end": end})


# ---------------------------------------------------------------------------
# Cleaning / feature helpers (no DB connection required)
# ---------------------------------------------------------------------------
def clean_option_quotes(
    df: pd.DataFrame,
    min_volume: int | None = None,
    max_spread_pct: float | None = None,
    require_positive_bid: bool = True,
) -> pd.DataFrame:
    """Drop obviously bad option quotes and add a midpoint column.

    Filters applied (in order, only if the relevant columns exist):
    - ``best_offer > 0`` (a non-positive offer is meaningless).
    - ``best_bid >= 0`` and, if ``require_positive_bid``, ``best_bid > 0``.
    - Bid/ask not crossed: ``best_bid <= best_offer``.
    - ``volume >= min_volume`` if ``min_volume`` is given.
    - Relative spread ``(ask - bid) / midpoint <= max_spread_pct`` if
      ``max_spread_pct`` is given (computed only after midpoint exists).

    Adds a ``midpoint`` column equal to ``(best_bid + best_offer) / 2``.
    Returns a new DataFrame with a fresh index.
    """
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    out = df.copy()
    has_bid = "best_bid" in out.columns
    has_ask = "best_offer" in out.columns
    if has_ask:
        out = out[out["best_offer"] > 0]
    if has_bid:
        out = out[out["best_bid"] >= 0]
        if require_positive_bid:
            out = out[out["best_bid"] > 0]
    if has_bid and has_ask:
        out = out[out["best_bid"] <= out["best_offer"]]
        out["midpoint"] = (out["best_bid"] + out["best_offer"]) / 2.0
        if max_spread_pct is not None:
            if max_spread_pct <= 0:
                raise ValueError("max_spread_pct must be positive")
            spread = (out["best_offer"] - out["best_bid"]) / out["midpoint"]
            out = out[spread <= max_spread_pct]
    if min_volume is not None and "volume" in out.columns:
        if min_volume < 0:
            raise ValueError("min_volume must be non-negative")
        out = out[out["volume"].fillna(0) >= min_volume]
    return out.reset_index(drop=True)


def add_time_to_maturity(
    df: pd.DataFrame,
    date_col: str = "date",
    expiry_col: str = "exdate",
    trading_days: bool = False,
) -> pd.DataFrame:
    """Add a ``T`` column (years to expiry) to ``df``.

    With ``trading_days=False`` (default), uses calendar days / 365.0
    matching the conventional Black-Scholes inputs in textbooks. With
    ``trading_days=True``, uses business days / 252.0 — useful when the
    volatility input is annualized from trading-day returns.
    """
    if date_col not in df.columns or expiry_col not in df.columns:
        raise KeyError(
            f"Both {date_col!r} and {expiry_col!r} must be present to compute T"
        )
    out = df.copy()
    d = pd.to_datetime(out[date_col])
    e = pd.to_datetime(out[expiry_col])
    if trading_days:
        # np.busday_count needs date64; build it once.
        d_arr = d.values.astype("datetime64[D]")
        e_arr = e.values.astype("datetime64[D]")
        days = np.busday_count(d_arr, e_arr)
        out["T"] = days / 252.0
    else:
        out["T"] = (e - d).dt.days / 365.0
    return out


def add_moneyness(
    df: pd.DataFrame,
    spot_col: str = "spot",
    strike_col: str = "strike_price",
) -> pd.DataFrame:
    """Add ``moneyness`` (S/K) and ``log_moneyness`` columns.

    No-op if either column is missing — the function returns the
    DataFrame unchanged so callers can pipeline this even when spot
    has not been merged in yet.
    """
    if spot_col not in df.columns or strike_col not in df.columns:
        return df
    out = df.copy()
    s = pd.to_numeric(out[spot_col])
    k = pd.to_numeric(out[strike_col])
    out["moneyness"] = s / k
    out["log_moneyness"] = np.log(out["moneyness"])
    return out


__all__ = [
    "DEFAULT_SCHEMA",
    "DEFAULT_OPTION_PRICE_TABLE_FMT",
    "DEFAULT_SECURITY_NAME_TABLE",
    "DEFAULT_SECURITY_PRICE_TABLE",
    "DEFAULT_ZERO_CURVE_TABLE",
    "DEFAULT_OPTION_COLUMNS",
    "DEFAULT_UNDERLYING_COLUMNS",
    "DEFAULT_ZERO_CURVE_COLUMNS",
    "STRIKE_PRICE_SCALE",
    "connect_wrds",
    "close_wrds",
    "validate_date_range",
    "validate_tickers",
    "get_security_ids",
    "load_option_chain",
    "load_underlying_prices",
    "load_zero_curve_or_rates",
    "clean_option_quotes",
    "add_time_to_maturity",
    "add_moneyness",
]
