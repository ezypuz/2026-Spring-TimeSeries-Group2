"""
Reusable matplotlib plots for the crypto pairs trading project.

All functions return (fig, ax) so callers can further customise or save.
Pass save_path to write to disk automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ZSCORE_ENTRY, ZSCORE_EXIT, RESULTS_DIR

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        11,
})

_DEFAULT_FIGSIZE = (14, 5)


# ---------------------------------------------------------------------------
# Price charts
# ---------------------------------------------------------------------------

def plot_prices(
    prices: pd.DataFrame,
    title: str = "Close Prices",
    log_scale: bool = False,
    save_path: Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot raw (or log) close prices for multiple symbols on one axes."""
    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)

    for col in prices.columns:
        ax.plot(prices.index, prices[col], label=col, linewidth=1.2)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("ln(Price)" if log_scale else "Price (USDT)")
    ax.legend()
    _fmt_xaxis(ax)
    _save(fig, save_path)
    return fig, ax


def plot_log_prices(
    prices: pd.DataFrame,
    title: str = "Log Close Prices",
    save_path: Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Convenience wrapper that log-transforms prices before plotting."""
    return plot_prices(np.log(prices), title=title, log_scale=True,
                       save_path=save_path)


# ---------------------------------------------------------------------------
# Spread
# ---------------------------------------------------------------------------

def plot_spread(
    spread: pd.Series,
    title: str = "Spread (ECT)",
    show_bands: bool = True,
    n_sigma: float = 2.0,
    save_path: Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the ECT / spread with optional ±n_sigma bands."""
    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)

    ax.plot(spread.index, spread.values, color="#2c7bb6", linewidth=1.0,
            label="Spread")

    mu  = spread.mean()
    sig = spread.std(ddof=1)
    ax.axhline(mu, color="black", linestyle="--", linewidth=1.0, label="Mean")

    if show_bands:
        for sign, label in [(+1, f"+{n_sigma}σ"), (-1, f"−{n_sigma}σ")]:
            ax.axhline(mu + sign * n_sigma * sig, color="#d7191c",
                       linestyle=":", linewidth=1.0, label=label)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Spread")
    ax.legend()
    _fmt_xaxis(ax)
    _save(fig, save_path)
    return fig, ax


# ---------------------------------------------------------------------------
# Z-score with trading signals
# ---------------------------------------------------------------------------

def plot_zscore_signals(
    zscore:   pd.Series,
    signals:  pd.Series | None = None,
    entry:    float = ZSCORE_ENTRY,
    exit_:    float = ZSCORE_EXIT,
    title:    str   = "Z-Score & Trading Signals",
    save_path: Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot Z-score with entry/exit thresholds and optional signal markers."""
    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)

    ax.plot(zscore.index, zscore.values, color="#555555", linewidth=0.9,
            label="Z-score", zorder=2)

    # Threshold lines
    for val, color, label in [
        ( entry, "#d7191c", f"+{entry}"),
        (-entry, "#1a9641", f"−{entry}"),
        ( exit_, "#aaaaaa", f"Exit ({exit_})") if exit_ != 0 else (None, None, None),
    ]:
        if val is not None:
            ax.axhline(val, color=color, linestyle="--", linewidth=1.0,
                       label=label)
    ax.axhline(0, color="black", linestyle="-", linewidth=0.6)

    # Signal markers
    if signals is not None:
        long_entries  = signals.index[(signals == 1)  & (signals.shift(1) != 1)]
        short_entries = signals.index[(signals == -1) & (signals.shift(1) != -1)]
        exits         = signals.index[(signals == 0)  & (signals.shift(1) != 0)]

        ax.scatter(long_entries,  zscore[long_entries],
                   marker="^", color="#1a9641", zorder=5, s=60, label="Long entry")
        ax.scatter(short_entries, zscore[short_entries],
                   marker="v", color="#d7191c", zorder=5, s=60, label="Short entry")
        ax.scatter(exits,         zscore[exits],
                   marker="x", color="#999999", zorder=5, s=40, label="Exit")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Z-score")
    ax.legend(loc="upper right", fontsize=9)
    _fmt_xaxis(ax)
    _save(fig, save_path)
    return fig, ax


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

def plot_equity_curve(
    equity:    pd.Series,
    metrics:   dict | None = None,
    benchmark: pd.Series | None = None,
    title:     str = "Portfolio Equity Curve",
    save_path: Path | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot equity curve (top) and drawdown (bottom) as a 2-panel chart."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    # ---- Equity ----
    ax1.plot(equity.index, equity.values, color="#2c7bb6", linewidth=1.3,
             label="Strategy")

    if benchmark is not None:
        bench_scaled = benchmark / benchmark.iloc[0] * equity.iloc[0]
        ax1.plot(bench_scaled.index, bench_scaled.values, color="#aaaaaa",
                 linewidth=1.0, linestyle="--", label="Benchmark")

    ax1.set_title(title, fontsize=13)
    ax1.set_ylabel("Portfolio Value (USDT)")
    ax1.legend()

    if metrics:
        info = (
            f"Total Ret: {metrics['total_return']:+.2%}  |  "
            f"CAGR: {metrics['cagr']:+.2%}  |  "
            f"Sharpe: {metrics['sharpe']:.2f}  |  "
            f"MDD: {metrics['max_drawdown']:.2%}"
        )
        ax1.set_xlabel(info)

    # ---- Drawdown ----
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max * 100

    ax2.fill_between(drawdown.index, drawdown.values, 0,
                     color="#d7191c", alpha=0.4, label="Drawdown")
    ax2.plot(drawdown.index, drawdown.values, color="#d7191c", linewidth=0.7)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    _fmt_xaxis(ax2)
    _save(fig, save_path)
    return fig, (ax1, ax2)


# ---------------------------------------------------------------------------
# Stationarity summary heatmap
# ---------------------------------------------------------------------------

def plot_stationarity_summary(
    report: pd.DataFrame,
    title:  str = "Stationarity Summary (ADF p-values)",
    save_path: Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Bar chart of ADF p-values with significance threshold line."""
    fig, ax = plt.subplots(figsize=(max(10, len(report) * 0.4), 5))

    colors = ["#1a9641" if p < 0.05 else "#d7191c"
              for p in report["adf_pvalue"]]
    ax.bar(report.index, report["adf_pvalue"], color=colors, edgecolor="white",
           width=0.7)
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1.2,
               label="5% threshold")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Symbol")
    ax.set_ylabel("ADF p-value")
    ax.tick_params(axis="x", rotation=90)
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_xaxis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


def _save(fig: plt.Figure, path: Path | None) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
