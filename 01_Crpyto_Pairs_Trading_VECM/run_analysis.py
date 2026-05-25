"""
CLI entry point: full analysis pipeline.

Runs the complete workflow for a given symbol pair:
    1. Load & preprocess close prices
    2. ADF stationarity test (level + 1st difference)
    3. Johansen cointegration test
    4. VECM fitting + ECT extraction
    5. Z-score signal generation
    6. Backtest + performance metrics
    7. Save all charts to results/

Usage examples
--------------
# Default pair (BTC/ETH), all defaults
python run_analysis.py

# Custom pair and parameters
python run_analysis.py --sym-a BTCUSDT --sym-b ETHUSDT --interval 1d
python run_analysis.py --sym-a SOLUSDT --sym-b AVAXUSDT --zscore-entry 1.5

# Skip backtest, just run statistics
python run_analysis.py --no-backtest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from src.config import (
    DATA_DIR, RESULTS_DIR, DEFAULT_INTERVAL,
    ZSCORE_ENTRY, ZSCORE_EXIT, ZSCORE_WINDOW,
    VECM_K_AR_DIFF, INITIAL_CAPITAL, TRANSACTION_COST,
)
from src.data        import DataLoader
from src.preprocessing import log_transform, align_and_clean
from src.stats       import adf_test, stationarity_report, johansen_test
from src.models      import fit_vecm, get_ect, vecm_summary
from src.trading     import compute_zscore, generate_signals, run_backtest
from src.trading.backtest import metrics_table
from src.viz         import (
    plot_log_prices, plot_spread, plot_zscore_signals, plot_equity_curve,
)


def run_pipeline(
    sym_a:     str   = "BTCUSDT",
    sym_b:     str   = "ETHUSDT",
    interval:  str   = DEFAULT_INTERVAL,
    data_dir:  Path  = DATA_DIR,
    results_dir: Path = RESULTS_DIR,
    zscore_entry: float = ZSCORE_ENTRY,
    zscore_exit:  float = ZSCORE_EXIT,
    zscore_window: int | None = ZSCORE_WINDOW,
    k_ar_diff: int = VECM_K_AR_DIFF,
    run_backtest_flag: bool = True,
    show_plots: bool = True,
) -> dict:
    """Execute the full analysis pipeline and return a results dict."""

    results_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{sym_a}_{sym_b}_{interval}"

    # -------------------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------------------
    print("\n[1/6] Loading data …")
    loader = DataLoader(data_dir=data_dir, interval=interval)
    close_a = loader.load(sym_a)
    close_b = loader.load(sym_b)
    close   = align_and_clean({sym_a: close_a, sym_b: close_b})
    log_px  = log_transform(close)

    print(f"  {sym_a}: {len(close_a)} rows")
    print(f"  {sym_b}: {len(close_b)} rows")
    print(f"  Aligned: {len(close)} rows  ({close.index[0].date()} → {close.index[-1].date()})")

    # -------------------------------------------------------------------------
    # 2. Stationarity tests
    # -------------------------------------------------------------------------
    print("\n[2/6] Stationarity tests (ADF + KPSS) …")
    stat_report = stationarity_report(log_px)
    print(stat_report.to_string())

    # Also test first differences
    print("\n  First differences:")
    diff_report = stationarity_report(log_px.diff().dropna())
    print(diff_report[["adf_pvalue", "adf_stationary", "conclusion"]].to_string())

    # -------------------------------------------------------------------------
    # 3. Johansen cointegration test
    # -------------------------------------------------------------------------
    print("\n[3/6] Johansen cointegration test …")
    j_result = johansen_test(log_px)
    print(j_result)
    print(f"  Trace stats    : {j_result.trace_stats.round(4)}")
    print(f"  95% crit vals  : {j_result.trace_crit_95.round(4)}")

    if not j_result.is_cointegrated:
        print("  WARNING: No cointegration found at 5% level.")

    # -------------------------------------------------------------------------
    # 4. VECM fit
    # -------------------------------------------------------------------------
    print("\n[4/6] Fitting VECM …")
    vecm = fit_vecm(log_px, k_ar_diff=k_ar_diff)
    print(f"  {vecm}")
    for k, v in vecm_summary(vecm).items():
        print(f"  {k:<22}: {v}")

    ect = get_ect(vecm, log_px)

    # -------------------------------------------------------------------------
    # 5. Signal generation
    # -------------------------------------------------------------------------
    print("\n[5/6] Generating trading signals …")
    zscore  = compute_zscore(ect, window=zscore_window)
    signals = generate_signals(zscore, entry=zscore_entry, exit_=zscore_exit)

    n_long  = int((signals ==  1).sum())
    n_short = int((signals == -1).sum())
    n_flat  = int((signals ==  0).sum())
    print(f"  Long  bars : {n_long:>6}")
    print(f"  Short bars : {n_short:>6}")
    print(f"  Flat  bars : {n_flat:>6}")

    # -------------------------------------------------------------------------
    # 6. Backtest
    # -------------------------------------------------------------------------
    bt_result = None
    if run_backtest_flag:
        print("\n[6/6] Running backtest …")
        bt_result = run_backtest(
            signals       = signals,
            log_price_a   = log_px[sym_a],
            log_price_b   = log_px[sym_b],
            beta          = vecm.ect_coef,
            initial_capital  = INITIAL_CAPITAL,
            transaction_cost = TRANSACTION_COST,
        )
        print(metrics_table(bt_result).to_string())

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    print("\nGenerating charts …")

    fig1, _ = plot_log_prices(
        log_px, title=f"Log Prices — {sym_a} / {sym_b}",
        save_path=results_dir / f"{tag}_log_prices.png",
    )

    fig2, _ = plot_spread(
        ect, title=f"ECT Spread — {sym_a} / {sym_b}",
        save_path=results_dir / f"{tag}_spread.png",
    )

    fig3, _ = plot_zscore_signals(
        zscore, signals=signals,
        entry=zscore_entry, exit_=zscore_exit,
        title=f"Z-Score Signals — {sym_a} / {sym_b}",
        save_path=results_dir / f"{tag}_zscore.png",
    )

    if bt_result is not None:
        fig4, _ = plot_equity_curve(
            bt_result.equity_curve,
            metrics=bt_result.metrics,
            title=f"Equity Curve — {sym_a} / {sym_b}",
            save_path=results_dir / f"{tag}_equity.png",
        )

    if show_plots:
        import matplotlib.pyplot as plt
        plt.show()

    print(f"\nAll charts saved to {results_dir}")

    return {
        "log_prices":    log_px,
        "stat_report":   stat_report,
        "johansen":      j_result,
        "vecm":          vecm,
        "ect":           ect,
        "zscore":        zscore,
        "signals":       signals,
        "backtest":      bt_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crypto pairs trading analysis pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sym-a",      default="BTCUSDT",  dest="sym_a")
    parser.add_argument("--sym-b",      default="ETHUSDT",  dest="sym_b")
    parser.add_argument("--interval",   default=DEFAULT_INTERVAL)
    parser.add_argument("--data-dir",   type=Path, default=DATA_DIR,   dest="data_dir")
    parser.add_argument("--results-dir",type=Path, default=RESULTS_DIR,dest="results_dir")
    parser.add_argument("--zscore-entry",  type=float, default=ZSCORE_ENTRY,  dest="zscore_entry")
    parser.add_argument("--zscore-exit",   type=float, default=ZSCORE_EXIT,   dest="zscore_exit")
    parser.add_argument("--zscore-window", type=int,   default=ZSCORE_WINDOW, dest="zscore_window")
    parser.add_argument("--k-ar-diff",  type=int,   default=VECM_K_AR_DIFF, dest="k_ar_diff")
    parser.add_argument("--no-backtest", action="store_true", dest="no_backtest")
    parser.add_argument("--no-show",     action="store_true", dest="no_show",
                        help="Do not call plt.show() (useful in headless environments)")
    args = parser.parse_args()

    run_pipeline(
        sym_a       = args.sym_a,
        sym_b       = args.sym_b,
        interval    = args.interval,
        data_dir    = args.data_dir,
        results_dir = args.results_dir,
        zscore_entry  = args.zscore_entry,
        zscore_exit   = args.zscore_exit,
        zscore_window = args.zscore_window,
        k_ar_diff     = args.k_ar_diff,
        run_backtest_flag = not args.no_backtest,
        show_plots    = not args.no_show,
    )


if __name__ == "__main__":
    main()
