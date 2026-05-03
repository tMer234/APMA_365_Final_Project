"""Unit tests for the optionmetrics module.

These tests cover input validation, data cleaning, and core functionality.
Tests requiring a live WRDS connection are skipped unless ``WRDS_LIVE_TEST=1`` 
is set in the environment.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
import optionmetrics as om
from dotenv import load_dotenv

load_dotenv()

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
def test_validate_tickers_basic():
    assert om.validate_tickers("aapl") == ["AAPL"]
    assert om.validate_tickers(["spy", "AAPL", "aapl"]) == ["AAPL", "SPY"]


def test_validate_tickers_rejects_bad():
    with pytest.raises(ValueError):
        om.validate_tickers([])
    with pytest.raises((ValueError, TypeError)):
        om.validate_tickers([""])
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


def test_clean_option_quotes_basic_functionality():
    df = _sample_quotes()
    out = om.clean_option_quotes(df)
    
    # Should add midpoint column
    assert "midpoint" in out.columns
    
    # Should filter out invalid quotes
    assert (out["best_offer"] > 0).all()
    assert len(out) < len(df)  # Some quotes should be filtered
    
    # Midpoint calculation should be correct
    expected_midpoint = (out["best_bid"] + out["best_offer"]) / 2.0
    np.testing.assert_allclose(out["midpoint"], expected_midpoint)


def test_clean_option_quotes_filters():
    df = _sample_quotes()
    
    # Test volume filter
    out_vol = om.clean_option_quotes(df, min_volume=50)
    assert (out_vol["volume"] >= 50).all()
    
    # Test spread filter
    df_spread = pd.DataFrame({
        "best_bid": [1.0, 1.0], 
        "best_offer": [1.05, 5.0], 
        "volume": [10, 10]
    })
    out_spread = om.clean_option_quotes(df_spread, max_spread_pct=0.5)
    assert len(out_spread) == 1  # Wide spread should be filtered


def test_clean_option_quotes_validation():
    df = _sample_quotes()
    with pytest.raises(ValueError):
        om.clean_option_quotes(df, max_spread_pct=0)
    with pytest.raises(ValueError):
        om.clean_option_quotes(df, min_volume=-1)


# ---------------------------------------------------------------------------
# add_time_to_maturity
# ---------------------------------------------------------------------------
def test_add_time_to_maturity():
    df = pd.DataFrame({
        "date": ["2022-01-03", "2022-01-03"], 
        "exdate": ["2022-01-03", "2023-01-03"]
    })
    out = om.add_time_to_maturity(df)
    assert out["T"].iloc[0] == pytest.approx(0.0)
    assert out["T"].iloc[1] == pytest.approx(1.0)  # 1 year
    
    # Test with missing columns
    with pytest.raises(KeyError):
        om.add_time_to_maturity(pd.DataFrame({"date": ["2022-01-01"]}))


# ---------------------------------------------------------------------------
# add_moneyness
# ---------------------------------------------------------------------------
def test_add_moneyness():
    df = pd.DataFrame({"spot": [100.0, 110.0], "strike_price": [100.0, 100.0]})
    out = om.add_moneyness(df)
    np.testing.assert_allclose(out["moneyness"], [1.0, 1.1])
    np.testing.assert_allclose(out["log_moneyness"], np.log([1.0, 1.1]))
    
    # Test when columns missing
    df_missing = pd.DataFrame({"strike_price": [100.0]})
    out_missing = om.add_moneyness(df_missing)
    assert "moneyness" not in out_missing.columns


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------
def test_close_wrds_safe_with_none():
    # Must not raise
    om.close_wrds(None)


def test_connect_wrds_raises_when_package_missing(monkeypatch):
    monkeypatch.setattr(om, "wrds", None)
    with pytest.raises(ImportError):
        om.connect_wrds()


# ---------------------------------------------------------------------------
# Live WRDS tests (skipped by default)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    os.environ.get("WRDS_LIVE_TEST") != "1",
    reason="Live WRDS test disabled; set WRDS_LIVE_TEST=1 to enable.",
)
def test_live_wrds_smoke():  # pragma: no cover - integration only
    pytest.importorskip("wrds")
    conn = om.connect_wrds()
    try:
        # Basic connection test - this should work for any WRDS account
        result = conn.raw_sql("SELECT current_user as username")
        assert len(result) == 1
        assert "username" in result.columns
        print(f"✓ WRDS connection successful for user: {result.iloc[0, 0]}")
        
        # Try OptionMetrics access if available (will be skipped if no subscription)
        try:
            ids = om.get_security_ids(conn, ["AAPL"])
            assert "secid" in ids.columns
            print("✓ OptionMetrics access confirmed")
        except Exception as e:
            if "permission denied" in str(e).lower() or "does not exist" in str(e).lower():
                pytest.skip("OptionMetrics subscription not available - basic WRDS connection works")
            else:
                raise
    finally:
        om.close_wrds(conn)


@pytest.mark.skipif(
    os.environ.get("WRDS_LIVE_TEST") != "1",
    reason="Live WRDS test disabled; set WRDS_LIVE_TEST=1 to enable.",
)
def test_load_option_chain():  
    pytest.importorskip("wrds")
    conn = om.connect_wrds()
    try:
        print("\nTesting option chain loading for AAPL...")
        
        # OptionMetrics IvyDB US Trial is limited to AAPL data from March 1-15, 2014
        try:
            print("Loading AAPL option chain for March 1-15, 2014 (trial data range)")
            options = om.load_option_chain(
                conn=conn,
                tickers=["AAPL"],
                start_date="2014-03-01",
                end_date="2014-03-15"
            )
            
            print(f"✓ Successfully loaded {len(options)} option records")
            print(f"Columns: {list(options.columns)}")
            
            # Verify expected columns are present
            expected_columns = ["ticker", "secid", "date"]
            for col in expected_columns:
                assert col in options.columns, f"Missing column: {col}"
            
            # Verify we got AAPL data in the correct date range
            if len(options) > 0:
                assert "AAPL" in options["ticker"].values
                print(f"Date range: {options['date'].min()} to {options['date'].max()}")
                
                
                print(f"✓ Option chain loading test passed with {len(options)} records!")
            else:
                pytest.fail("No option data found for AAPL in March 1-15, 2014 trial range")
                
        except Exception as e:
            error_msg = str(e).lower()
            if "does not exist" in error_msg or "permission denied" in error_msg:
                pytest.skip("Option price data not accessible in trial account")
            else:
                print(f"✗ Option chain loading failed: {e}")
                raise
                
    finally:
        om.close_wrds(conn)
