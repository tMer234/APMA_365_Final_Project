"""Offline tests for ``scripts/extract_ivydb_trial.py``.

These tests cover the trial-range constants, derived-column behaviour,
and the output-path helper. They never touch WRDS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import extract_ivydb_trial as ext  # noqa: E402


# ---------------------------------------------------------------------------
# Trial-range constants
# ---------------------------------------------------------------------------
def test_trial_constants_are_expected_window():
    assert ext.TRIAL_TICKER == "AAPL"
    assert ext.TRIAL_START_DATE == "2014-03-01"
    assert ext.TRIAL_END_DATE == "2014-03-15"
    start = pd.Timestamp(ext.TRIAL_START_DATE)
    end = pd.Timestamp(ext.TRIAL_END_DATE)
    assert start <= end
    assert (end - start).days == 14


# ---------------------------------------------------------------------------
# default_output_path
# ---------------------------------------------------------------------------
def test_default_output_path_uses_trial_window(tmp_path):
    p = ext.default_output_path(base_dir=tmp_path)
    assert p.parent == tmp_path
    assert p.name == "aapl_ivydb_trial_2014-03-01_2014-03-15.csv"


def test_default_output_path_lowercases_ticker(tmp_path):
    p = ext.default_output_path("MSFT", "2024-01-02", "2024-01-05", base_dir=tmp_path)
    assert p.name == "msft_ivydb_trial_2024-01-02_2024-01-05.csv"


def test_module_default_path_under_repo():
    # The module-level default lives under data/processed/ inside the repo.
    p = ext.DEFAULT_OUTPUT_PATH
    assert p.parts[-3:] == ("data", "processed", "aapl_ivydb_trial_2014-03-01_2014-03-15.csv")


# ---------------------------------------------------------------------------
# add_derived_columns
# ---------------------------------------------------------------------------
def _sample_options() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "secid": [101, 101, 101],
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "date": ["2014-03-03", "2014-03-03", "2014-03-04"],
            "exdate": ["2014-03-21", "2014-04-19", "2014-03-21"],
            "cp_flag": ["c", "P", "C"],
            "strike_price": [500.0, 525.0, 510.0],
            "best_bid": [25.0, 1.0, 18.0],
            "best_offer": [26.0, 1.2, 19.0],
            "midpoint": [25.5, 1.1, 18.5],
            "volume": [100, 50, 30],
            "open_interest": [200, 80, 40],
        }
    )


def _sample_underlying() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "secid": [101, 101],
            "date": ["2014-03-03", "2014-03-04"],
            "close": [527.5, 530.0],
        }
    )


def test_add_derived_columns_basic():
    out = ext.add_derived_columns(_sample_options(), _sample_underlying())

    for col in (
        "midpoint",
        "bid_ask_spread",
        "bid_ask_spread_pct",
        "days_to_expiry",
        "time_to_maturity_years",
        "spot",
        "moneyness",
        "log_moneyness",
    ):
        assert col in out.columns, col

    np.testing.assert_allclose(out["bid_ask_spread"], [1.0, 0.2, 1.0])
    np.testing.assert_allclose(
        out["bid_ask_spread_pct"], [1.0 / 25.5, 0.2 / 1.1, 1.0 / 18.5]
    )
    assert list(out["days_to_expiry"]) == [18, 47, 17]
    np.testing.assert_allclose(
        out["time_to_maturity_years"],
        np.array([18, 47, 17]) / 365.0,
    )
    np.testing.assert_allclose(out["spot"], [527.5, 527.5, 530.0])
    np.testing.assert_allclose(
        out["moneyness"], [527.5 / 500.0, 527.5 / 525.0, 530.0 / 510.0]
    )
    np.testing.assert_allclose(out["log_moneyness"], np.log(out["moneyness"]))
    assert list(out["cp_flag"]) == ["C", "P", "C"]


def test_add_derived_columns_empty():
    out = ext.add_derived_columns(pd.DataFrame(), None)
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_add_derived_columns_no_underlying():
    out = ext.add_derived_columns(_sample_options(), None)
    assert "spot" not in out.columns
    assert "moneyness" not in out.columns
    assert "bid_ask_spread" in out.columns
    assert "time_to_maturity_years" in out.columns


def test_add_derived_columns_does_not_mutate_input():
    df = _sample_options()
    df_before = df.copy()
    ext.add_derived_columns(df, _sample_underlying())
    pd.testing.assert_frame_equal(df, df_before)


def test_add_derived_columns_handles_zero_midpoint():
    df = pd.DataFrame(
        {
            "secid": [1],
            "date": ["2014-03-03"],
            "exdate": ["2014-03-21"],
            "cp_flag": ["C"],
            "strike_price": [500.0],
            "best_bid": [0.0],
            "best_offer": [0.0],
            "midpoint": [0.0],
        }
    )
    out = ext.add_derived_columns(df, None)
    assert np.isnan(out["bid_ask_spread_pct"].iloc[0])


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
def test_cli_parser_defaults():
    args = ext._parse_args([])
    assert args.ticker == "AAPL"
    assert args.start == ext.TRIAL_START_DATE
    assert args.end == ext.TRIAL_END_DATE
    assert args.min_volume == 0
    assert args.max_spread_pct is None
    assert args.no_zero_curve is False


def test_cli_parser_overrides(tmp_path):
    out = tmp_path / "x.csv"
    args = ext._parse_args(
        [
            "--output", str(out),
            "--max-spread-pct", "0.75",
            "--min-volume", "5",
            "--no-zero-curve",
        ]
    )
    assert args.output == out
    assert args.max_spread_pct == pytest.approx(0.75)
    assert args.min_volume == 5
    assert args.no_zero_curve is True
