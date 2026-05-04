"""End-to-end pipeline: optional WRDS extract, backtest, figures, metrics text."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_CACHE = REPO_ROOT / ".cache"
_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
(_CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from scripts.backtest_black_scholes import (
    DEFAULT_BACKTEST_PATH,
    DEFAULT_HV_PATH,
    DEFAULT_OPTIONS_PATH,
    DEFAULT_SUMMARY_PATH,
    run_backtest,
)
from scripts.extract_ivydb_trial import (
    DEFAULT_OUTPUT_PATH as DEFAULT_OPTION_CSV,
    DEFAULT_ZERO_CURVE_OUTPUT_PATH,
    TRIAL_END_DATE,
    TRIAL_START_DATE,
    TRIAL_TICKER,
    run_extraction,
)
from scripts.generate_model_comparison_plots import generate_model_comparison_figures
from scripts.plots import (
    deduplicate_option_quotes,
    generate_all_core_plots,
    load_historical_volatility_data,
    load_processed_option_data,
)

OPTION_USECOLS = [
    "date",
    "exdate",
    "cp_flag",
    "strike_price",
    "midpoint",
    "spot",
    "days_to_expiry",
    "volume",
    "open_interest",
    "optionid",
    "best_bid",
    "best_offer",
]


def _format_metrics_table(summary: pd.DataFrame) -> str:
    lines = ["Backtest summary metrics", "=" * 40]
    for _, row in summary.iterrows():
        m = str(row["metric"])
        v = row["value"]
        if isinstance(v, float):
            lines.append(f"{m:28s} {v:.6g}")
        else:
            lines.append(f"{m:28s} {v}")
    lines.append("")
    lines.append("Notes:")
    lines.append("  bs_price_* : Black-Scholes price (historical vol) vs midpoint")
    lines.append("  nr_iv_*    : Newton-Raphson IV vs historical IV (finite NR only)")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--extract",
        action="store_true",
        help="Pull AAPL trial data from WRDS (requires credentials).",
    )
    p.add_argument("--options", type=Path, default=DEFAULT_OPTION_CSV)
    p.add_argument("--historical-vol", type=Path, default=DEFAULT_HV_PATH)
    p.add_argument("--backtest-out", type=Path, default=DEFAULT_BACKTEST_PATH)
    p.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_PATH)
    p.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "figures")
    p.add_argument(
        "--metrics-out",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "analysis_metrics.txt",
        help="Write formatted metrics to this path.",
    )
    p.add_argument("--risk-free-rate", type=float, default=0.01)
    p.add_argument("--min-volume", type=int, default=1)
    p.add_argument("--max-rows", type=int, default=10**9)
    p.add_argument("--min-midpoint", type=float, default=0.25)
    p.add_argument("--log-level", default="INFO", help="Used when --extract is set.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.extract:
        logging.basicConfig(
            level=getattr(logging, args.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        run_extraction(
            args.options,
            ticker=TRIAL_TICKER,
            start=TRIAL_START_DATE,
            end=TRIAL_END_DATE,
            zero_curve_output=DEFAULT_ZERO_CURVE_OUTPUT_PATH,
        )

    if not args.options.exists():
        print(f"Missing option CSV: {args.options}", file=sys.stderr)
        print("Run with --extract after configuring WRDS, or place processed CSVs.", file=sys.stderr)
        return 1
    if not args.historical_vol.exists():
        print(f"Missing historical vol CSV: {args.historical_vol}", file=sys.stderr)
        return 1

    _, summary = run_backtest(
        options_path=args.options,
        hv_path=args.historical_vol,
        output_backtest_path=args.backtest_out,
        output_summary_path=args.summary_out,
        risk_free_rate=args.risk_free_rate,
        min_volume=args.min_volume,
        max_rows=args.max_rows,
        min_midpoint=args.min_midpoint,
    )
    backtest = pd.read_csv(args.backtest_out)

    options = load_processed_option_data(args.options, usecols=OPTION_USECOLS)
    historical_vol = load_historical_volatility_data(args.historical_vol)
    generate_all_core_plots(
        option_data=deduplicate_option_quotes(options),
        historical_vol_data=historical_vol,
        output_dir=args.figures_dir,
        show=False,
    )

    model_dir = args.figures_dir / "model_comparison"
    generate_model_comparison_figures(backtest, model_dir, show=False)

    metrics_text = _format_metrics_table(summary)
    print(metrics_text)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(metrics_text, encoding="utf-8")
    print(f"Wrote metrics: {args.metrics_out}")
    print(f"Core figures: {args.figures_dir}")
    print(f"Model figures: {model_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
