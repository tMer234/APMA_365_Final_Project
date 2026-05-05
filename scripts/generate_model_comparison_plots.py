"""Generate BS/NR comparison plots and diagnostics."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_MPL_CACHE_DIR = _REPO_ROOT / ".cache" / "matplotlib"
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_XDG_CACHE_DIR = _REPO_ROOT / ".cache"
_XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

from scripts.backtest_black_scholes import run_backtest  # noqa: E402
from scripts.plots import (  # noqa: E402
    plot_bs_error_diagnostics,
    plot_bs_predicted_vs_actual,
    plot_iv_error_diagnostics,
    plot_nr_iv_vs_historical_by_hv_tenor,
    plot_r2_by_volatility_range,
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
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "figures" / "model_comparison"


def generate_model_comparison_figures(
    backtest_df: pd.DataFrame,
    output_dir: Path,
    *,
    show: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_bs_predicted_vs_actual(
        backtest_df, save_path=output_dir / "bs_predicted_vs_actual.png", show=show
    )
    plot_nr_iv_vs_historical_by_hv_tenor(
        backtest_df, save_path=output_dir / "nr_iv_vs_historical_by_hv_tenor.png", show=show
    )
    plot_bs_error_diagnostics(backtest_df, prefix=output_dir / "bs_error", show=show)
    plot_iv_error_diagnostics(backtest_df, prefix=output_dir / "iv_error", show=show)
    plot_r2_by_volatility_range(
        backtest_df,
        save_path=output_dir / "r2_by_volatility_range_calls_puts.png",
        show=show,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--options", type=Path, default=DEFAULT_OPTIONS_PATH)
    parser.add_argument("--historical-vol", type=Path, default=DEFAULT_HV_PATH)
    parser.add_argument("--backtest", type=Path, default=DEFAULT_BACKTEST_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--risk-free-rate", type=float, default=0.01)
    parser.add_argument("--min-volume", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=25000)
    parser.add_argument(
        "--min-midpoint",
        type=float,
        default=0.25,
        help="Exclude contracts with midpoint below this threshold.",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--reuse-existing-backtest",
        action="store_true",
        help="Use existing backtest CSV instead of recomputing it.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_existing_backtest and args.backtest.exists():
        backtest = pd.read_csv(args.backtest)
    else:
        backtest, _ = run_backtest(
            options_path=args.options,
            hv_path=args.historical_vol,
            output_backtest_path=args.backtest,
            output_summary_path=args.summary,
            risk_free_rate=args.risk_free_rate,
            min_volume=args.min_volume,
            max_rows=args.max_rows,
            min_midpoint=args.min_midpoint,
        )

    generate_model_comparison_figures(backtest, args.output_dir, show=args.show)

    print(f"Generated focused model-comparison plots in: {args.output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
