"""Unit tests for the WRDS-free helpers in ``optionmetrics``.

These tests cover input validation and the pure-pandas cleaning /
feature-engineering helpers. Tests requiring a live WRDS connection
are skipped unless ``WRDS_LIVE_TEST=1`` is set in the environment, and
even then only run if ``wrds`` is importable. By default the suite is
runnable in CI without credentials.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import optionmetrics as om


# ---------------------------------------------------------------------------
# validate_date_range
# ---------------------------------------------------------------------------
def test_validate_date_range_strings():
    start, end = om.validate_date_range("2022-01-03", "2022-12-30")
    assert (start.year, start.month, start.day) == (2022, 1, 3)
    assert (end.year, end.month, end.day) == (2022, 12, 30)


def test_validate_date_range_swapped_raises():
    with pytest.raises(ValueError):
        om.validate_date_range("2022-12-30", "2022-01-03")


def test_validate_date_range_bad_input():
    with pytest.raises(ValueError):
        om.validate_date_range("not-a-date", "2022-01-01")


# ---------------------------------------------------------------------------
# validate_tickers
# ---------------------------------------------------------------------------
def test_validate_tickers_single_string():
    assert om.validate_tickers("aapl") == ["AAPL"]


def test_validate_tickers_dedup_and_sort():
    assert om.validate_tickers(["spy", "AAPL", "aapl"]) == ["AAPL", "SPY"]


def test_validate_tickers_allows_dot_and_dash():
    assert om.validate_tickers(["BRK.B", "RDS-A"]) == ["BRK.B", "RDS-A"]


@pytest.mark.parametrize("bad", ["", "AAPL'; DROP TABLE x;--", "TOOLONGTICKER", "AA PL"])
def test_validate_tickers_rejects_bad(bad):
    with pytest.raises((ValueError, TypeError)):
        om.validate_tickers([bad])


def test_validate_tickers_requires_nonempty_iterable():
    with pytest.raises(ValueError):
        om.validate_tickers([])


def test_validate_tickers_rejects_non_string():
    with pytest.raises(TypeError):
        om.validate_tickers([123])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# clean_option_quotes
# ---------------------------------------------------------------------------
def _sample_quotes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "best_bid":   [1.0, 0.0, 2.0,  3.0, 1.0,  1.5],
            "best_offer": [1.2, 0.5, 1.5,  3.2, -0.1, 5.0],
            "volume":     [100, 50,  200,  0,   10,   1],
        }
    )


def test_clean_option_quotes_drops_nonpositive_offer_and_negative_bid():
    df = _sample_quotes()
    out = om.clean_option_quotes(df, require_positive_bid=False)
    assert (out["best_offer"] > 0).all()
    assert (out["best_bid"] >= 0).all()
    # crossed quote (bid 2.0 > offer 1.5) must be removed
    assert not ((out["best_bid"] == 2.0) & (out["best_offer"] == 1.5)).any()


def test_clean_option_quotes_require_positive_bid():
    df = _sample_quotes()
    out = om.clean_option_quotes(df, require_positive_bid=True)
    assert (out["best_bid"] > 0).all()


def test_clean_option_quotes_adds_midpoint():
    df = _sample_quotes()
    out = om.clean_option_quotes(df)
    assert "midpoint" in out.columns
    np.testing.assert_allclose(
        out["midpoint"].values,
        (out["best_bid"].values + out["best_offer"].values) / 2.0,
    )


def test_clean_option_quotes_min_volume_filter():
    df = _sample_quotes()
    out = om.clean_option_quotes(df, min_volume=50)
    assert (out["volume"] >= 50).all()


def test_clean_option_quotes_max_spread_pct():
    df = pd.DataFrame(
        {"best_bid": [1.0, 1.0], "best_offer": [1.05, 5.0], "volume": [10, 10]}
    )
    out = om.clean_option_quotes(df, max_spread_pct=0.5)
    # First row spread ~5%, retained; second row spread ~133%, dropped
    assert len(out) == 1
    assert float(out.iloc[0]["best_offer"]) == pytest.approx(1.05)


def test_clean_option_quotes_rejects_bad_thresholds():
    df = _sample_quotes()
    with pytest.raises(ValueError):
        om.clean_option_quotes(df, max_spread_pct=0)
    with pytest.raises(ValueError):
        om.clean_option_quotes(df, min_volume=-1)


def test_clean_option_quotes_empty_input():
    out = om.clean_option_quotes(pd.DataFrame())
    assert isinstance(out, pd.DataFrame) and out.empty


# ---------------------------------------------------------------------------
# add_time_to_maturity
# ---------------------------------------------------------------------------
def test_add_time_to_maturity_calendar():
    df = pd.DataFrame(
        {"date": ["2022-01-03", "2022-01-03"], "exdate": ["2022-01-03", "2023-01-03"]}
    )
    out = om.add_time_to_maturity(df)
    assert out["T"].iloc[0] == pytest.approx(0.0)
    assert out["T"].iloc[1] == pytest.approx(365 / 365.0)


def test_add_time_to_maturity_trading_days():
    df = pd.DataFrame({"date": ["2022-01-03"], "exdate": ["2022-01-10"]})
    out = om.add_time_to_maturity(df, trading_days=True)
    # Mon->Mon excluding the end date is 5 business days
    assert out["T"].iloc[0] == pytest.approx(5 / 252.0)


def test_add_time_to_maturity_missing_columns():
    with pytest.raises(KeyError):
        om.add_time_to_maturity(pd.DataFrame({"date": ["2022-01-01"]}))


# ---------------------------------------------------------------------------
# add_moneyness
# ---------------------------------------------------------------------------
def test_add_moneyness_adds_columns():
    df = pd.DataFrame({"spot": [100.0, 110.0], "strike_price": [100.0, 100.0]})
    out = om.add_moneyness(df)
    np.testing.assert_allclose(out["moneyness"], [1.0, 1.1])
    np.testing.assert_allclose(out["log_moneyness"], np.log([1.0, 1.1]))


def test_add_moneyness_noop_when_columns_missing():
    df = pd.DataFrame({"strike_price": [100.0]})
    out = om.add_moneyness(df)
    assert "moneyness" not in out.columns


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------
def test_close_wrds_safe_with_none():
    # Must not raise
    om.close_wrds(None)


def test_close_wrds_calls_close_method():
    class FakeConn:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    fake = FakeConn()
    om.close_wrds(fake)
    assert fake.closed is True


def test_connect_wrds_raises_when_package_missing(monkeypatch):
    monkeypatch.setattr(om, "wrds", None)
    with pytest.raises(ImportError):
        om.connect_wrds()


# ---------------------------------------------------------------------------
# zero curve placeholder
# ---------------------------------------------------------------------------
def test_load_zero_curve_raises_when_table_none():
    with pytest.raises(NotImplementedError):
        om.load_zero_curve_or_rates(conn=None, start_date="2022-01-01", end_date="2022-01-31", table=None)


# ---------------------------------------------------------------------------
# Live WRDS smoke test (skipped by default)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    os.environ.get("WRDS_LIVE_TEST") != "1",
    reason="Live WRDS test disabled; set WRDS_LIVE_TEST=1 to enable.",
)
def test_live_wrds_smoke():  # pragma: no cover - integration only
    pytest.importorskip("wrds")
    conn = om.connect_wrds()
    try:
        ids = om.get_security_ids(conn, ["AAPL"])
        assert "secid" in ids.columns
    finally:
        om.close_wrds(conn)
