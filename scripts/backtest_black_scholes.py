"""Backtest Black-Scholes prices and Newton-Raphson implied volatility."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from black_scholes import (  # noqa: E402
    black_scholes_call,
    black_scholes_implied_volatility,
    black_scholes_put,
)


DEFAULT_OPTIONS_PATH = (
    _REPO_ROOT / "data" / "processed" / "aapl_ivydb_trial_2014-03-01_2014-03-15.csv"
)
DEFAULT_HV_PATH = (
    _REPO_ROOT
    / "data"
    / "processed"
    / "aapl_historical_volatility_2014-03-01_2014-03-15.csv"
)
DEFAULT_BACKTEST_PATH = (
    _REPO_ROOT / "data" / "processed" / "aapl_bs_backtest_2014-03-01_2014-03-15.csv"
)
DEFAULT_SUMMARY_PATH = (
    _REPO_ROOT
    / "data"
    / "processed"
    / "aapl_bs_backtest_summary_2014-03-01_2014-03-15.csv"
)
DEFAULT_FIGURE_PATH = _REPO_ROOT / "figures" / "backtest_bs_error_histogram.png"


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    if "optionid" in df.columns:
        subset = ["date", "optionid", "best_bid", "best_offer", "volume", "open_interest"]
        subset = [c for c in subset if c in df.columns]
        return df.drop_duplicates(subset=subset).copy()
    return df.drop_duplicates().copy()


def _prepare_inputs(option_path: Path, hv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    options = pd.read_csv(
        option_path,
        usecols=[
            "date",
            "exdate",
            "cp_flag",
            "strike_price",
            "spot",
            "midpoint",
            "volume",
            "optionid",
            "best_bid",
            "best_offer",
            "days_to_expiry",
            "time_to_maturity_years",
        ],
    )
    hv = pd.read_csv(hv_path, usecols=["date", "days", "volatility"])

    options = _dedupe(options)
    options["date"] = pd.to_datetime(options["date"], errors="coerce")
    options["exdate"] = pd.to_datetime(options["exdate"], errors="coerce")
    options["cp_flag"] = options["cp_flag"].astype(str).str.upper()
    if "midpoint" not in options.columns:
        options["midpoint"] = (options["best_bid"] + options["best_offer"]) / 2.0
    if "days_to_expiry" not in options.columns:
        options["days_to_expiry"] = (options["exdate"] - options["date"]).dt.days
    if "time_to_maturity_years" not in options.columns:
        options["time_to_maturity_years"] = options["days_to_expiry"] / 365.0

    hv["date"] = pd.to_datetime(hv["date"], errors="coerce")
    hv["days"] = pd.to_numeric(hv["days"], errors="coerce")
    hv["volatility"] = pd.to_numeric(hv["volatility"], errors="coerce")

    hv_keys = hv[["date", "days", "volatility"]].dropna().sort_values(["date", "days"])
    merged = options.merge(hv_keys, on="date", how="left", suffixes=("", "_hv"))
    merged["days_to_expiry"] = pd.to_numeric(merged["days_to_expiry"], errors="coerce")
    merged["tenor_gap_days"] = (merged["days"] - merged["days_to_expiry"]).abs()

    merged = (
        merged.sort_values(["date", "optionid", "tenor_gap_days"])
        .drop_duplicates(subset=["date", "optionid", "best_bid", "best_offer"], keep="first")
        .reset_index(drop=True)
    )
    merged = merged.rename(columns={"volatility": "historical_volatility", "days": "hv_tenor_days"})
    return merged, hv


def _bs_price(row: pd.Series, r: float) -> float:
    s = float(row["spot"])
    k = float(row["strike_price"])
    t = float(row["time_to_maturity_years"])
    sigma = float(row["historical_volatility"])
    cp = row["cp_flag"]
    if cp == "C":
        return float(black_scholes_call(s, k, t, r, sigma))
    if cp == "P":
        return float(black_scholes_put(s, k, t, r, sigma))
    return np.nan


def _mae_rmse_mape_signed_error(error: np.ndarray, denom: np.ndarray) -> tuple[float, float, float]:
    """MAE, RMSE, MAPE for signed error; MAPE uses |error|/denom where denom > 0."""
    err = np.asarray(error, dtype=float)
    d = np.asarray(denom, dtype=float)
    mask = np.isfinite(err) & np.isfinite(d) & (d > 0)
    if not np.any(mask):
        return float("nan"), float("nan"), float("nan")
    e = err[mask]
    de = d[mask]
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e**2)))
    mape = float(np.mean(np.abs(e) / de))
    return mae, rmse, mape


def _nr_iv(row: pd.Series, r: float) -> float:
    cp = "call" if row["cp_flag"] == "C" else "put"
    try:
        return float(
            black_scholes_implied_volatility(
                Market_Price=float(row["midpoint"]),
                S=float(row["spot"]),
                K=float(row["strike_price"]),
                T=float(row["time_to_maturity_years"]),
                r=r,
                option_type=cp,
            )
        )
    except Exception:
        return np.nan


def run_backtest(
    options_path: Path,
    hv_path: Path,
    output_backtest_path: Path,
    output_summary_path: Path,
    risk_free_rate: float = 0.01,
    min_volume: int = 1,
    max_rows: int | None = 25000,
    min_midpoint: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, _ = _prepare_inputs(options_path, hv_path)
    numeric_cols = [
        "spot",
        "strike_price",
        "time_to_maturity_years",
        "midpoint",
        "historical_volatility",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = df[
        (df["spot"] > 0)
        & (df["strike_price"] > 0)
        & (df["time_to_maturity_years"] > 0)
        & (df["midpoint"] > 0)
        & (df["historical_volatility"] > 0)
        & (df["cp_flag"].isin(["C", "P"]))
    ].copy()
    if "volume" in valid.columns:
        valid["volume"] = pd.to_numeric(valid["volume"], errors="coerce").fillna(0.0)
        valid = valid[valid["volume"] >= float(min_volume)]
    if min_midpoint > 0:
        valid = valid[valid["midpoint"] >= float(min_midpoint)]
    if max_rows is not None and len(valid) > max_rows:
        if "volume" in valid.columns:
            valid = valid.sort_values("volume", ascending=False).head(max_rows).copy()
        else:
            valid = valid.head(max_rows).copy()

    valid["bs_price_hv"] = valid.apply(lambda r: _bs_price(r, risk_free_rate), axis=1)
    valid["bs_error"] = valid["bs_price_hv"] - valid["midpoint"]
    valid["abs_bs_error"] = valid["bs_error"].abs()
    valid["ape_bs"] = valid["abs_bs_error"] / valid["midpoint"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        valid["nr_implied_vol"] = valid.apply(lambda r: _nr_iv(r, risk_free_rate), axis=1)
    valid["iv_error_vs_hv"] = valid["nr_implied_vol"] - valid["historical_volatility"]
    valid["abs_iv_error_vs_hv"] = valid["iv_error_vs_hv"].abs()

    bs_mae, bs_rmse, bs_mape = _mae_rmse_mape_signed_error(
        (valid["bs_price_hv"] - valid["midpoint"]).to_numpy(),
        valid["midpoint"].to_numpy(),
    )
    nr_mask = valid["nr_implied_vol"].notna() & (valid["historical_volatility"] > 0)
    nr_iv_err = (valid.loc[nr_mask, "nr_implied_vol"] - valid.loc[nr_mask, "historical_volatility"]).to_numpy()
    nr_hv = valid.loc[nr_mask, "historical_volatility"].to_numpy()
    nr_mae, nr_rmse, nr_mape = _mae_rmse_mape_signed_error(nr_iv_err, nr_hv)
    n_nr_iv = int(nr_mask.sum())

    summary = pd.DataFrame(
        {
            "metric": [
                "n_quotes",
                "n_nr_iv",
                "bs_price_mae",
                "bs_price_rmse",
                "bs_price_mape",
                "nr_iv_mae",
                "nr_iv_rmse",
                "nr_iv_mape",
                "median_tenor_gap_days",
            ],
            "value": [
                float(len(valid)),
                float(n_nr_iv),
                bs_mae,
                bs_rmse,
                bs_mape,
                nr_mae,
                nr_rmse,
                nr_mape,
                float(valid["tenor_gap_days"].median()),
            ],
        }
    )

    output_backtest_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_cols = [
        "cp_flag",
        "midpoint",
        "strike_price",
        "spot",
        "time_to_maturity_years",
        "hv_tenor_days",
        "historical_volatility",
        "tenor_gap_days",
        "bs_price_hv",
        "bs_error",
        "nr_implied_vol",
        "iv_error_vs_hv",
    ]
    valid[output_cols].to_csv(output_backtest_path, index=False)
    summary.to_csv(output_summary_path, index=False)
    return valid[output_cols], summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--options", type=Path, default=DEFAULT_OPTIONS_PATH)
    parser.add_argument("--historical-vol", type=Path, default=DEFAULT_HV_PATH)
    parser.add_argument("--output-backtest", type=Path, default=DEFAULT_BACKTEST_PATH)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--risk-free-rate", type=float, default=0.01)
    parser.add_argument(
        "--min-volume",
        type=int,
        default=1,
        help="Minimum volume filter before backtest metrics.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=25000,
        help="Cap rows used for backtest (highest-volume rows retained).",
    )
    parser.add_argument(
        "--min-midpoint",
        type=float,
        default=0.0,
        help="Exclude contracts with midpoint below this threshold.",
    )
    parser.add_argument("--plot", action="store_true", help="Save BS error histogram.")
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_FIGURE_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    backtest, summary = run_backtest(
        options_path=args.options,
        hv_path=args.historical_vol,
        output_backtest_path=args.output_backtest,
        output_summary_path=args.output_summary,
        risk_free_rate=args.risk_free_rate,
        min_volume=args.min_volume,
        max_rows=args.max_rows,
        min_midpoint=args.min_midpoint,
    )
    if args.plot:
        from scripts.plots import plot_backtest_error_histogram  # noqa: WPS433,E402

        plot_backtest_error_histogram(backtest, error_col="bs_error", save_path=args.plot_path, show=False)

    print("Backtest complete.")
    print(f"Wrote option-level results: {args.output_backtest}")
    print(f"Wrote summary metrics:     {args.output_summary}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
