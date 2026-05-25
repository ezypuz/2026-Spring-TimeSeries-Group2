"""
Stationarity tests: ADF and KPSS.

Each test returns a typed result dataclass so callers never have to
remember the position of individual items in a tuple.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from src.config import ADF_MAX_LAGS, ADF_ALPHA, KPSS_NLAGS


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ADFResult:
    symbol:          str
    stat:            float
    pvalue:          float
    used_lags:       int
    nobs:            int
    critical_values: dict[str, float]   # {"1%": ..., "5%": ..., "10%": ...}
    is_stationary:   bool               # True if we reject H0 (unit root)

    def __str__(self) -> str:
        decision = "Stationary" if self.is_stationary else "Non-stationary"
        return (
            f"ADF({self.symbol})  stat={self.stat:.4f}  "
            f"p={self.pvalue:.4f}  [{decision}]"
        )


@dataclass
class KPSSResult:
    symbol:          str
    stat:            float
    pvalue:          float
    used_lags:       int
    critical_values: dict[str, float]
    is_stationary:   bool               # True if we fail to reject H0 (stationary)

    def __str__(self) -> str:
        decision = "Stationary" if self.is_stationary else "Non-stationary"
        return (
            f"KPSS({self.symbol})  stat={self.stat:.4f}  "
            f"p={self.pvalue:.4f}  [{decision}]"
        )


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------

def adf_test(
    series: pd.Series,
    max_lags: int | None = ADF_MAX_LAGS,
    alpha:    float      = ADF_ALPHA,
    regression: str      = "c",
) -> ADFResult:
    """Augmented Dickey-Fuller test for a unit root.

    H0: Series has a unit root (non-stationary).
    Reject H0 → series is stationary.

    Parameters
    ----------
    series:
        1-D time series (NaN-free recommended).
    max_lags:
        Maximum lags to consider; None → automatic AIC selection.
    alpha:
        Significance level for the is_stationary decision (default 0.05).
    regression:
        Deterministic component — 'c' (constant), 'ct' (constant + trend),
        'ctt', 'n'.

    Returns
    -------
    ADFResult
    """
    clean = series.dropna()
    stat, pvalue, lags, nobs, crit, _ = adfuller(
        clean, maxlag=max_lags, regression=regression, autolag="AIC"
    )
    return ADFResult(
        symbol          = series.name or "unknown",
        stat            = float(stat),
        pvalue          = float(pvalue),
        used_lags       = int(lags),
        nobs            = int(nobs),
        critical_values = {k: float(v) for k, v in crit.items()},
        is_stationary   = pvalue < alpha,
    )


def kpss_test(
    series: pd.Series,
    nlags:  int | str = KPSS_NLAGS,
    alpha:  float     = ADF_ALPHA,
    regression: str   = "c",
) -> KPSSResult:
    """KPSS stationarity test.

    H0: Series is stationary (trend-stationary).
    Reject H0 → series is non-stationary (has a unit root).

    Parameters
    ----------
    series:
        1-D time series.
    nlags:
        Lag truncation parameter; 'auto' for automatic selection.
    alpha:
        Significance level.
    regression:
        'c' (level stationary) or 'ct' (trend stationary).
    """
    clean = series.dropna()
    stat, pvalue, lags, crit = kpss(clean, regression=regression, nlags=nlags)
    return KPSSResult(
        symbol          = series.name or "unknown",
        stat            = float(stat),
        pvalue          = float(pvalue),
        used_lags       = int(lags),
        critical_values = {k: float(v) for k, v in crit.items()},
        is_stationary   = pvalue >= alpha,   # fail-to-reject → stationary
    )


# ---------------------------------------------------------------------------
# Batch summary
# ---------------------------------------------------------------------------

def stationarity_report(
    price_panel: pd.DataFrame,
    max_lags: int | None = ADF_MAX_LAGS,
    alpha:    float      = ADF_ALPHA,
    run_kpss: bool       = True,
) -> pd.DataFrame:
    """Run ADF (and optionally KPSS) on every column of a price panel.

    Typical usage:
        log_prices = log_transform(close_panel)
        report     = stationarity_report(log_prices)

    Parameters
    ----------
    price_panel:
        DataFrame with one column per symbol, datetime index.
    max_lags:
        Passed to adf_test.
    alpha:
        Significance level.
    run_kpss:
        Also compute KPSS and add its columns to the report.

    Returns
    -------
    pd.DataFrame with index = symbol and columns:
        adf_stat, adf_pvalue, adf_lags, adf_stationary
        [kpss_stat, kpss_pvalue, kpss_lags, kpss_stationary]  (if run_kpss)
        conclusion  ('I(0)' or 'I(1)' or 'Ambiguous')
    """
    rows: list[dict] = []

    for col in price_panel.columns:
        adf = adf_test(price_panel[col], max_lags=max_lags, alpha=alpha)
        row: dict = {
            "adf_stat":        adf.stat,
            "adf_pvalue":      adf.pvalue,
            "adf_lags":        adf.used_lags,
            "adf_stationary":  adf.is_stationary,
        }

        if run_kpss:
            kp = kpss_test(price_panel[col], alpha=alpha)
            row.update({
                "kpss_stat":       kp.stat,
                "kpss_pvalue":     kp.pvalue,
                "kpss_lags":       kp.used_lags,
                "kpss_stationary": kp.is_stationary,
            })
            # Both tests agree on stationary → I(0)
            # Both tests agree on non-stationary → I(1)
            if adf.is_stationary and kp.is_stationary:
                conclusion = "I(0)"
            elif not adf.is_stationary and not kp.is_stationary:
                conclusion = "I(1)"
            else:
                conclusion = "Ambiguous"
        else:
            conclusion = "I(0)" if adf.is_stationary else "I(1)"

        row["conclusion"] = conclusion
        rows.append(row)

    return pd.DataFrame(rows, index=price_panel.columns)
