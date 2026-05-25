"""
Cointegration analysis: Johansen test and pair screening.

The Johansen procedure tests for the number of cointegrating vectors
among a set of I(1) series. We also provide a pairwise screening helper
that iterates over all (N choose 2) pairs in a price panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from src.config import (
    JOHANSEN_DET_ORDER,
    JOHANSEN_K_AR_DIFF,
    COINT_ALPHA,
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class JohansenResult:
    symbols:          tuple[str, ...]
    trace_stats:      np.ndarray          # shape (n,)
    trace_crit_95:    np.ndarray          # shape (n,)
    max_eig_stats:    np.ndarray          # shape (n,)
    max_eig_crit_95:  np.ndarray          # shape (n,)
    coint_vectors:    np.ndarray          # eigenvectors, shape (n, n)
    coint_rank_trace: int                 # rank by trace test at 5 %
    coint_rank_maxeig: int                # rank by max-eigenvalue test at 5 %

    @property
    def is_cointegrated(self) -> bool:
        """True if at least one cointegrating relationship found (trace, 5 %)."""
        return self.coint_rank_trace >= 1

    @property
    def beta(self) -> np.ndarray:
        """First (dominant) cointegrating vector, shape (n,)."""
        return self.coint_vectors[:, 0]

    def __str__(self) -> str:
        ci = "YES" if self.is_cointegrated else "NO"
        return (
            f"Johansen({', '.join(self.symbols)})  "
            f"rank(trace)={self.coint_rank_trace}  "
            f"cointegrated={ci}"
        )


@dataclass
class CointPair:
    sym_a:         str
    sym_b:         str
    result:        JohansenResult
    beta:          float    # coefficient: ln(A) - beta * ln(B) ≈ stationary
    is_cointegrated: bool

    def __str__(self) -> str:
        return (
            f"{self.sym_a}/{self.sym_b}  "
            f"beta={self.beta:.4f}  "
            f"cointegrated={self.is_cointegrated}"
        )


# ---------------------------------------------------------------------------
# Core test
# ---------------------------------------------------------------------------

def johansen_test(
    log_prices: pd.DataFrame,
    det_order:  int = JOHANSEN_DET_ORDER,
    k_ar_diff:  int = JOHANSEN_K_AR_DIFF,
) -> JohansenResult:
    """Run the Johansen cointegration test on a log-price DataFrame.

    Parameters
    ----------
    log_prices:
        DataFrame of log prices, one column per asset, datetime index.
        Should contain only I(1) series.
    det_order:
        Deterministic specification passed to coint_johansen:
        -1 = no constant, 0 = constant outside ECT, 1 = constant inside ECT.
    k_ar_diff:
        Number of lagged differences to include.

    Returns
    -------
    JohansenResult
    """
    data    = log_prices.dropna().values
    symbols = tuple(log_prices.columns)
    n       = data.shape[1]

    res = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)

    # res.cvt[:, 1] → 95 % critical values for trace test
    # res.cvm[:, 1] → 95 % critical values for max-eigenvalue test
    trace_stats      = res.lr1           # shape (n,)
    trace_crit_95    = res.cvt[:, 1]
    max_eig_stats    = res.lr2
    max_eig_crit_95  = res.cvm[:, 1]

    rank_trace  = int(np.sum(trace_stats   > trace_crit_95))
    rank_maxeig = int(np.sum(max_eig_stats > max_eig_crit_95))

    return JohansenResult(
        symbols           = symbols,
        trace_stats       = trace_stats,
        trace_crit_95     = trace_crit_95,
        max_eig_stats     = max_eig_stats,
        max_eig_crit_95   = max_eig_crit_95,
        coint_vectors     = res.evec,
        coint_rank_trace  = rank_trace,
        coint_rank_maxeig = rank_maxeig,
    )


# ---------------------------------------------------------------------------
# Pairwise screening
# ---------------------------------------------------------------------------

def find_cointegrated_pairs(
    log_price_panel: pd.DataFrame,
    det_order:  int   = JOHANSEN_DET_ORDER,
    k_ar_diff:  int   = JOHANSEN_K_AR_DIFF,
    min_alpha:  float = COINT_ALPHA,
    verbose:    bool  = True,
) -> list[CointPair]:
    """Screen every pair in a log-price panel for cointegration.

    Parameters
    ----------
    log_price_panel:
        DataFrame of log prices, columns = symbols.
    det_order, k_ar_diff:
        Passed to johansen_test.
    min_alpha:
        Not directly used in the Johansen trace criterion (which is fixed at
        5 % by the critical value table), but kept as a hook for callers.
    verbose:
        Print summary for each tested pair.

    Returns
    -------
    List of CointPair, sorted by beta coefficient (ascending).
    Only cointegrated pairs are included.
    """
    symbols = log_price_panel.columns.tolist()
    pairs: list[CointPair] = []

    total = len(list(combinations(symbols, 2)))
    print(f"Testing {total} pairs for cointegration …\n")

    for idx, (sym_a, sym_b) in enumerate(combinations(symbols, 2), 1):
        pair_df = log_price_panel[[sym_a, sym_b]].dropna()
        if len(pair_df) < 100:
            continue

        try:
            result = johansen_test(pair_df, det_order=det_order, k_ar_diff=k_ar_diff)
        except Exception as exc:
            if verbose:
                print(f"  [{idx:>5}/{total}] {sym_a}/{sym_b} — error: {exc}")
            continue

        # beta: ln(A) = beta * ln(B) + c  →  vector is [1, -beta]
        vec   = result.coint_vectors[:, 0]
        beta  = -vec[1] / vec[0] if abs(vec[0]) > 1e-10 else np.nan

        pair = CointPair(
            sym_a           = sym_a,
            sym_b           = sym_b,
            result          = result,
            beta            = float(beta),
            is_cointegrated = result.is_cointegrated,
        )

        if verbose:
            mark = "✓" if result.is_cointegrated else " "
            print(
                f"  [{idx:>5}/{total}] [{mark}] "
                f"{sym_a:<14}/{sym_b:<14}  "
                f"rank={result.coint_rank_trace}  beta={beta:.4f}"
            )

        if result.is_cointegrated:
            pairs.append(pair)

    print(f"\n→ {len(pairs)} cointegrated pairs found out of {total}")
    return sorted(pairs, key=lambda p: p.beta)


def cointegration_summary(pairs: list[CointPair]) -> pd.DataFrame:
    """Convert a list of CointPair to a summary DataFrame."""
    return pd.DataFrame([
        {
            "pair":           f"{p.sym_a}/{p.sym_b}",
            "sym_a":          p.sym_a,
            "sym_b":          p.sym_b,
            "beta":           p.beta,
            "rank_trace":     p.result.coint_rank_trace,
            "rank_maxeig":    p.result.coint_rank_maxeig,
        }
        for p in pairs
    ])
