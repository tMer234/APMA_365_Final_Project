"""Generate core AAPL diagnostic figures."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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

from scripts.plots import (  # noqa: E402
    generate_all_core_plots,
    load_historical_volatility_data,
    load_processed_option_data,
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
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "figures"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--options", type=Path, default=DEFAULT_OPTIONS_PATH)
    parser.add_argument("--historical-vol", type=Path, default=DEFAULT_HV_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving PNGs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    options = load_processed_option_data(
        args.options,
        usecols=[
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
        ],
    )
    historical_vol = load_historical_volatility_data(args.historical_vol)
    generate_all_core_plots(
        option_data=options,
        historical_vol_data=historical_vol,
        output_dir=args.output_dir,
        show=args.show,
    )
    print(f"Generated core plots in: {args.output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
