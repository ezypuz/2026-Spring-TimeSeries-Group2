"""
VECM (Vector Error Correction Model) fitting and interpretation.

Wraps statsmodels VECM with a cleaner result surface.
The model equation for a 2-asset pair is:

    Δln(A_t) = α₁ [ln(A_{t-1}) - β·ln(B_{t-1}) - c]
               + Σᵢ γᵢ Δln(A_{t-i}) + Σᵢ δᵢ Δln(B_{t-i}) + ε₁ₜ

where [·] is the Error Correction Term (ECT / spread).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM, VECMResults

from src.config import (
    VECM_K_AR_DIFF,
    VECM_COINT_RANK,
    VECM_DET_ORDER,
)


# ---------------------------------------------------------------------------
# Result wrapper
# ---------------------------------------------------------------------------

@dataclass
class VECMSummary:
    """Parsed VECM estimation output."""
    symbols:           tuple[str, ...]
    coint_rank:        int
    k_ar_diff:         int
    alpha:             np.ndarray        # adjustment speeds  (n × r)
    beta:              np.ndarray        # cointegrating vectors (n × r)
    beta_normalized:   np.ndarray        # beta normalized so beta[0, 0] = 1
    ect_coef:          float             # scalar β for 2-asset case
    alpha_a:           float             # adjustment speed for first asset
    alpha_b:           float             # adjustment speed for second asset
    llf:               float             # log-likelihood
    aic:               float
    bic:               float
    _fitted:           VECMResults       # raw statsmodels object

    def __str__(self) -> str:
        return (
            f"VECM({', '.join(self.symbols)})  "
            f"rank={self.coint_rank}  "
            f"β={self.ect_coef:.4f}  "
            f"α=({self.alpha_a:.4f}, {self.alpha_b:.4f})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_vecm(
    log_prices: pd.DataFrame,
    k_ar_diff:  int = VECM_K_AR_DIFF,
    coint_rank: int = VECM_COINT_RANK,
    det_order:  str = VECM_DET_ORDER,
) -> VECMSummary:
    """Fit a VECM to a log-price DataFrame.

    Parameters
    ----------
    log_prices:
        DataFrame of log prices, columns = symbols (must be I(1) series).
        Minimum 2 columns required.
    k_ar_diff:
        Number of lagged differences in the VECM (p-1 in VAR notation).
    coint_rank:
        Number of cointegrating vectors (r). For a 2-asset system r=1.
    det_order:
        Deterministic specification string for statsmodels VECM:
        'n', 'co', 'ci', 'lo', 'li'  (see statsmodels docs).

    Returns
    -------
    VECMSummary
    """
    data    = log_prices.dropna().values
    symbols = tuple(log_prices.columns)

    model  = VECM(endog=data, k_ar_diff=k_ar_diff, coint_rank=coint_rank,
                  deterministic=det_order)
    fitted = model.fit()

    # statsmodels returns beta as (n_endog, coint_rank)
    beta_raw = fitted.beta                       # shape (n, r)
    alpha_raw = fitted.alpha                     # shape (n, r)

    # Normalize so the first element of the first vector = 1
    scale = beta_raw[0, 0]
    beta_norm = beta_raw / scale if abs(scale) > 1e-10 else beta_raw

    # For a 2-asset system: spread = ln(A) + beta_norm[1,0] * ln(B)
    # The cointegrating equation is ln(A) = -beta_norm[1,0] * ln(B) + c
    ect_coef = float(-beta_norm[1, 0]) if beta_norm.shape[0] >= 2 else float("nan")

    return VECMSummary(
        symbols          = symbols,
        coint_rank       = coint_rank,
        k_ar_diff        = k_ar_diff,
        alpha            = alpha_raw,
        beta             = beta_raw,
        beta_normalized  = beta_norm,
        ect_coef         = ect_coef,
        alpha_a          = float(alpha_raw[0, 0]),
        alpha_b          = float(alpha_raw[1, 0]) if alpha_raw.shape[0] >= 2 else float("nan"),
        llf              = float(fitted.llf),
        aic              = float(getattr(fitted, "aic", float("nan"))),
        bic              = float(getattr(fitted, "bic", float("nan"))),
        _fitted          = fitted,
    )


def get_ect(
    summary: VECMSummary,
    log_prices: pd.DataFrame,
) -> pd.Series:
    """Reconstruct the Error Correction Term (spread) time series.

    ECT_t = ln(A_t) + beta_norm[1,0] * ln(B_t) + (constant term if any)

    This is the "spread" that the trading strategy monitors.

    Parameters
    ----------
    summary:
        Fitted VECMSummary from fit_vecm.
    log_prices:
        The same log-price DataFrame used for fitting (or a new window).

    Returns
    -------
    pd.Series  named 'ect', same datetime index as log_prices.
    """
    data = log_prices.dropna()
    beta = summary.beta_normalized[:, 0]       # shape (n,) or (n+1,) with const

    n_endog = log_prices.shape[1]
    if len(beta) == n_endog:
        ect = data.values @ beta
    elif len(beta) == n_endog + 1:
        # Last element is the constant term
        ect = data.values @ beta[:n_endog] + beta[-1]
    else:
        raise ValueError(
            f"Unexpected beta length {len(beta)} for {n_endog} endog variables."
        )

    return pd.Series(ect, index=data.index, name="ect")


def select_optimal_lag(
    log_prices: pd.DataFrame,
    max_lags:   int = 10,
    criterion:  str = "aic",
) -> int:
    """Determine optimal lag order by fitting VECM with varying k_ar_diff.

    Parameters
    ----------
    log_prices:
        Log-price DataFrame.
    max_lags:
        Maximum k_ar_diff to try.
    criterion:
        'aic' or 'bic'.

    Returns
    -------
    Optimal k_ar_diff (1-indexed).
    """
    scores: dict[int, float] = {}

    for lag in range(1, max_lags + 1):
        try:
            summary = fit_vecm(log_prices, k_ar_diff=lag)
            scores[lag] = summary.aic if criterion == "aic" else summary.bic
        except Exception:
            continue

    if not scores:
        return 1

    return min(scores, key=scores.__getitem__)


def vecm_summary(summary: VECMSummary) -> dict:
    """Return a plain dict of key model statistics for reporting."""
    return {
        "symbols":        list(summary.symbols),
        "coint_rank":     summary.coint_rank,
        "k_ar_diff":      summary.k_ar_diff,
        "beta (ECT coef)": summary.ect_coef,
        "alpha_a":        summary.alpha_a,
        "alpha_b":        summary.alpha_b,
        "AIC":            summary.aic,
        "BIC":            summary.bic,
        "log-likelihood": summary.llf,
    }
