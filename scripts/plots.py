"""Plot helpers for AAPL trial analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_processed_option_data(
    path: str | Path,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """Load option-level CSV and normalize key columns."""
    df = pd.read_csv(path, usecols=usecols)
    for col in ("date", "exdate"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_historical_volatility_data(path: str | Path) -> pd.DataFrame:
    """Load historical volatility CSV and normalize key columns."""
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def deduplicate_option_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate quotes to avoid over-weighting repeated rows."""
    if "optionid" in df.columns:
        subset = ["date", "optionid", "best_bid", "best_offer", "volume", "open_interest"]
        existing = [c for c in subset if c in df.columns]
        if existing:
            return df.drop_duplicates(subset=existing).copy()
    return df.drop_duplicates().copy()


def _finalize_plot(
    fig: plt.Figure,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_underlying_spot_over_time(
    option_chain: pd.DataFrame,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    daily = (
        option_chain.dropna(subset=["date", "spot"])
        .groupby("date", as_index=False)["spot"]
        .mean()
        .sort_values("date")
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily["date"], daily["spot"], marker="o", linewidth=1.8)
    ax.set_title("AAPL Spot Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Spot Price")
    ax.grid(alpha=0.25)
    _finalize_plot(fig, save_path, show)


def plot_option_activity_over_time(
    option_chain: pd.DataFrame,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Daily option activity: volume, open interest, and quote count."""
    daily = (
        option_chain.dropna(subset=["date"])
        .groupby("date", as_index=False)
        .agg(
            volume=("volume", "sum"),
            open_interest=("open_interest", "sum"),
            quote_count=("date", "size"),
        )
        .sort_values("date")
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(daily["date"], daily["volume"], marker="o", label="Volume")
    axes[0].plot(daily["date"], daily["open_interest"], marker="s", label="Open Interest")
    axes[0].set_title("AAPL Option Activity Over Time")
    axes[0].set_ylabel("Contracts")
    axes[0].legend(loc="best", framealpha=0.95)
    axes[0].grid(alpha=0.25)

    axes[1].bar(daily["date"], daily["quote_count"], width=0.8)
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Number of Quotes")
    axes[1].grid(alpha=0.25)
    _finalize_plot(fig, save_path, show)


def plot_historical_volatility_timeseries(
    historical_volatility: pd.DataFrame,
    tenor_days: int = 30,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    hv = historical_volatility[historical_volatility["days"] == tenor_days].copy()
    hv = hv.dropna(subset=["date", "volatility"]).sort_values("date")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hv["date"], hv["volatility"], marker="o", linewidth=1.8)
    ax.set_title(f"Historical Volatility Time Series ({tenor_days}-Day)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.grid(alpha=0.25)
    _finalize_plot(fig, save_path, show)


def plot_price_vs_strike(
    option_chain: pd.DataFrame,
    cp_flag: str = "C",
    snapshot_date: str | pd.Timestamp | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Strike vs midpoint for one trade date; color = days to expiry; dashed = intrinsic."""
    df = option_chain.copy()
    df = df[df["cp_flag"].astype(str).str.upper() == cp_flag.upper()]
    if snapshot_date is None:
        snapshot_date = df["date"].max()
    snapshot_date = pd.Timestamp(snapshot_date)
    snap = df[df["date"] == snapshot_date].copy()
    if "exdate" in snap.columns:
        snap["exdate"] = pd.to_datetime(snap["exdate"], errors="coerce")
    if "days_to_expiry" not in snap.columns and "exdate" in snap.columns:
        snap["days_to_expiry"] = (snap["exdate"] - snap["date"]).dt.days
    snap = snap.dropna(subset=["strike_price", "midpoint", "days_to_expiry"])
    snap = snap[snap["days_to_expiry"] >= 0]
    if len(snap) == 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(f"No data: {cp_flag.upper()}, {snapshot_date.date()}")
        _finalize_plot(fig, save_path, show)
        return

    spot_series = snap["spot"] if "spot" in snap.columns else pd.Series(dtype=float)
    spot_val = float(spot_series.dropna().median()) if spot_series.notna().any() else float("nan")
    k_min = float(snap["strike_price"].min())
    k_max = float(snap["strike_price"].max())
    k_fine = np.linspace(k_min, k_max, 400)

    fig, ax = plt.subplots(figsize=(10, 5))
    if np.isfinite(spot_val) and spot_val > 0:
        if cp_flag.upper() == "C":
            intrinsic = np.maximum(spot_val - k_fine, 0.0)
        else:
            intrinsic = np.maximum(k_fine - spot_val, 0.0)
        ax.plot(
            k_fine,
            intrinsic,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="Intrinsic (European, T→0)",
        )

    dte = snap["days_to_expiry"].astype(float)
    sc = ax.scatter(
        snap["strike_price"],
        snap["midpoint"],
        c=dte,
        cmap="viridis",
        s=22,
        alpha=0.78,
        edgecolors="white",
        linewidths=0.35,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Days to expiry")

    kind = "Call" if cp_flag.upper() == "C" else "Put"
    spot_note = f", spot ≈ {spot_val:.2f}" if np.isfinite(spot_val) else ""
    ax.set_title(f"{kind} midpoint vs strike ({snapshot_date.date()}){spot_note}")
    ax.set_xlabel("Strike price K")
    ax.set_ylabel("Option midpoint")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="best", framealpha=0.95)
    _finalize_plot(fig, save_path, show)


def plot_backtest_error_histogram(
    backtest_df: pd.DataFrame,
    error_col: str = "bs_error",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(backtest_df[error_col].dropna(), bins=50, alpha=0.8)
    ax.set_title(f"Distribution of {error_col}")
    ax.set_xlabel("Error")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    _finalize_plot(fig, save_path, show)


def generate_all_core_plots(
    option_data: pd.DataFrame,
    historical_vol_data: pd.DataFrame,
    output_dir: str | Path = "figures",
    show: bool = False,
) -> None:
    """Create the reduced core set of AAPL diagnostics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    options = deduplicate_option_quotes(option_data)

    plot_underlying_spot_over_time(options, output_dir / "spot_over_time.png", show)
    plot_option_activity_over_time(options, output_dir / "option_activity_over_time.png", show)
    plot_historical_volatility_timeseries(
        historical_vol_data,
        tenor_days=30,
        save_path=output_dir / "historical_volatility_30d.png",
        show=show,
    )
    plot_price_vs_strike(options, "C", save_path=output_dir / "price_vs_strike_calls.png", show=show)
    plot_price_vs_strike(options, "P", save_path=output_dir / "price_vs_strike_puts.png", show=show)


def plot_bs_predicted_vs_actual(
    backtest_df: pd.DataFrame,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Scatter plot of predicted Black-Scholes prices vs actual mid prices."""
    df = backtest_df.dropna(subset=["bs_price_hv", "midpoint"]).copy()
    fig, ax = plt.subplots(figsize=(7, 7))
    calls = df[df["cp_flag"] == "C"]
    puts = df[df["cp_flag"] == "P"]
    ax.scatter(
        calls["midpoint"],
        calls["bs_price_hv"],
        c="#1f77b4",
        s=12,
        alpha=0.35,
        label="Calls",
    )
    ax.scatter(
        puts["midpoint"],
        puts["bs_price_hv"],
        c="#ff7f0e",
        s=12,
        alpha=0.35,
        label="Puts",
    )
    mn = min(df["midpoint"].min(), df["bs_price_hv"].min())
    mx = max(df["midpoint"].max(), df["bs_price_hv"].max())
    x = df["midpoint"].to_numpy()
    y = df["bs_price_hv"].to_numpy()
    slope, intercept = np.polyfit(x, y, deg=1)
    y_hat = intercept + slope * x
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    n = len(x)
    p = 1  # one regressor: actual midpoint
    if n > p + 1 and not np.isnan(r2):
        r2_adj = 1.0 - (1.0 - r2) * (n - 1.0) / (n - p - 1.0)
    else:
        r2_adj = np.nan

    line_x = np.array([mn, mx])
    line_y = intercept + slope * line_x
    ax.plot(line_x, line_y, color="#2ca02c", linewidth=1.5, label="Linear fit")

    ax.text(
        0.03,
        0.97,
        f"R² = {r2:.4f}\nAdj. R² = {r2_adj:.4f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#666666"},
    )
    ax.set_title("Black-Scholes Predicted vs Actual Option Prices", fontsize=12)
    ax.set_xlabel("Actual Option Mid Price", fontsize=11)
    ax.set_ylabel("Predicted Black-Scholes Price", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", framealpha=0.95)
    _finalize_plot(fig, save_path, show)


def plot_nr_iv_vs_historical_by_horizon(
    backtest_df: pd.DataFrame,
    horizon_col: str = "hv_tenor_days",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Compare NR implied vol and historical vol over time horizons."""
    df = backtest_df.dropna(subset=["nr_implied_vol", "historical_volatility", horizon_col]).copy()
    grouped = (
        df.groupby(horizon_col, as_index=False)[["nr_implied_vol", "historical_volatility"]]
        .median()
        .sort_values(horizon_col)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        grouped[horizon_col],
        grouped["nr_implied_vol"],
        marker="o",
        linewidth=1.8,
        label="Newton-Raphson implied volatility (median)",
    )
    ax.plot(
        grouped[horizon_col],
        grouped["historical_volatility"],
        marker="s",
        linewidth=1.8,
        label="Historical volatility (median)",
    )
    ax.set_title("Implied vs Historical Volatility Across Time Horizons", fontsize=12)
    ax.set_xlabel("Time Horizon (days)", fontsize=11)
    ax.set_ylabel("Volatility", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", framealpha=0.95)
    _finalize_plot(fig, save_path, show)


def plot_bs_error_diagnostics(
    backtest_df: pd.DataFrame,
    prefix: str | Path = "figures/bs_error",
    show: bool = True,
) -> None:
    """Diagnostic plots for Black-Scholes pricing error."""
    prefix = Path(prefix)
    df = backtest_df.dropna(subset=["bs_error", "midpoint"]).copy()

    calls = df[df["cp_flag"] == "C"]
    puts = df[df["cp_flag"] == "P"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(calls["bs_error"], bins=60, alpha=0.55, label="Calls", color="#1f77b4")
    ax.hist(puts["bs_error"], bins=60, alpha=0.55, label="Puts", color="#ff7f0e")
    ax.set_title("Black-Scholes Pricing Error Distribution", fontsize=12)
    ax.set_xlabel("Pricing Error (Predicted - Actual)", fontsize=11)
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", framealpha=0.95)
    _finalize_plot(fig, prefix.with_name(f"{prefix.name}_hist.png"), show)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(calls["midpoint"], calls["bs_error"], s=12, alpha=0.35, label="Calls", c="#1f77b4")
    ax.scatter(puts["midpoint"], puts["bs_error"], s=12, alpha=0.35, label="Puts", c="#ff7f0e")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Black-Scholes Pricing Error vs Actual Price", fontsize=12)
    ax.set_xlabel("Actual Option Mid Price", fontsize=11)
    ax.set_ylabel("Pricing Error (Predicted - Actual)", fontsize=11)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", framealpha=0.95)
    _finalize_plot(fig, prefix.with_name(f"{prefix.name}_vs_price.png"), show)


def plot_iv_error_diagnostics(
    backtest_df: pd.DataFrame,
    prefix: str | Path = "figures/iv_error",
    show: bool = True,
) -> None:
    """Diagnostic plots for absolute Newton-Raphson implied-vol error vs HV."""
    prefix = Path(prefix)
    df = backtest_df.dropna(subset=["iv_error_vs_hv", "hv_tenor_days"]).copy()
    df["abs_iv_error"] = df["iv_error_vs_hv"].abs()

    calls = df[df["cp_flag"] == "C"]
    puts = df[df["cp_flag"] == "P"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(calls["abs_iv_error"], bins=60, alpha=0.55, label="Calls", color="#1f77b4")
    ax.hist(puts["abs_iv_error"], bins=60, alpha=0.55, label="Puts", color="#ff7f0e")
    ax.set_title("Absolute Newton-Raphson Implied-Volatility Error Distribution", fontsize=12)
    ax.set_xlabel("Absolute IV Error |Newton-Raphson IV - Historical IV|", fontsize=11)
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", framealpha=0.95)
    _finalize_plot(fig, prefix.with_name(f"{prefix.name}_hist.png"), show)

    grouped = (
        df.groupby(["hv_tenor_days", "cp_flag"], as_index=False)["abs_iv_error"]
        .median()
        .sort_values(["cp_flag", "hv_tenor_days"])
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    for cp, color, label in [("C", "#1f77b4", "Calls"), ("P", "#ff7f0e", "Puts")]:
        sub = grouped[grouped["cp_flag"] == cp]
        ax.plot(sub["hv_tenor_days"], sub["abs_iv_error"], marker="o", color=color, label=label)
    ax.set_title("Median Absolute Newton-Raphson IV Error by Time Horizon", fontsize=12)
    ax.set_xlabel("Time Horizon (days)", fontsize=11)
    ax.set_ylabel("Median Absolute IV Error", fontsize=11)
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", framealpha=0.95)
    _finalize_plot(fig, prefix.with_name(f"{prefix.name}_by_horizon.png"), show)


def _compute_r2_metrics(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (R2, adjusted R2) for simple linear regression y ~ x."""
    if len(x) < 3:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x, y, deg=1)
    y_hat = intercept + slope * x
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot <= 0:
        return np.nan, np.nan
    r2 = 1.0 - (ss_res / ss_tot)
    n = len(x)
    p = 1
    if n <= p + 1:
        return r2, np.nan
    r2_adj = 1.0 - (1.0 - r2) * (n - 1.0) / (n - p - 1.0)
    return r2, r2_adj


def plot_r2_by_volatility_range(
    backtest_df: pd.DataFrame,
    n_bins: int = 6,
    vol_col: str = "historical_volatility",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot R2 and adjusted R2 by volatility range for calls and puts."""
    df = backtest_df.dropna(subset=["midpoint", "bs_price_hv", "cp_flag", vol_col]).copy()
    df = df[df[vol_col] > 0]
    if len(df) == 0:
        return

    df["vol_bin"] = pd.qcut(df[vol_col], q=n_bins, duplicates="drop")
    bin_order = list(df["vol_bin"].cat.categories)
    bin_labels = [f"{interval.left:.3f}-{interval.right:.3f}" for interval in bin_order]

    records: list[dict[str, object]] = []
    for cp_flag, option_type in [("C", "Calls"), ("P", "Puts")]:
        sub = df[df["cp_flag"] == cp_flag]
        for interval in bin_order:
            chunk = sub[sub["vol_bin"] == interval]
            x = chunk["midpoint"].to_numpy()
            y = chunk["bs_price_hv"].to_numpy()
            r2, r2_adj = _compute_r2_metrics(x, y)
            records.append(
                {
                    "option_type": option_type,
                    "bin": interval,
                    "r2": r2,
                    "r2_adj": r2_adj,
                }
            )
    metrics = pd.DataFrame(records)
    x_idx = np.arange(len(bin_order))

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for ax, metric_col, title in [
        (axes[0], "r2", "R² by Volatility Range"),
        (axes[1], "r2_adj", "Adjusted R² by Volatility Range"),
    ]:
        for option_type, color in [("Calls", "#1f77b4"), ("Puts", "#ff7f0e")]:
            s = metrics[metrics["option_type"] == option_type].sort_values("bin")
            ax.plot(x_idx, s[metric_col], marker="o", linewidth=1.8, color=color, label=option_type)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(metric_col.replace("_", " ").upper(), fontsize=11)
        ax.set_ylim(-0.1, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", framealpha=0.95)

    axes[1].set_xticks(x_idx)
    axes[1].set_xticklabels(bin_labels, rotation=30, ha="right")
    axes[1].set_xlabel("Historical Volatility Range", fontsize=11)
    _finalize_plot(fig, save_path, show)