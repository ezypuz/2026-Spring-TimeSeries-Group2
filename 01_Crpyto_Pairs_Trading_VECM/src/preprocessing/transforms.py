"""
Price series transformations.

Design principle: every function is a pure transformation — it returns a
new object and never modifies its input in-place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_transform(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Apply natural-log transformation to price(s).

    Parameters
    ----------
    prices:
        Raw price DataFrame (columns = symbols) or a single Series.

    Returns
    -------
    Same type as input, natural-log transformed.
    """
    return np.log(prices)


def compute_returns(
    prices: pd.DataFrame | pd.Series,
    log: bool = True,
) -> pd.DataFrame | pd.Series:
    """Compute period-over-period returns.

    Parameters
    ----------
    prices:
        Price DataFrame or Series.
    log:
        True  → log returns: ln(P_t / P_{t-1})
        False → simple returns: (P_t - P_{t-1}) / P_{t-1}

    Returns
    -------
    Same type as input, first row will be NaN (dropped automatically
    when used downstream).
    """
    if log:
        return np.log(prices).diff()
    return prices.pct_change()


def align_and_clean(
    prices: dict[str, pd.Series] | pd.DataFrame,
    how: str = "inner",
    dropna: bool = True,
) -> pd.DataFrame:
    """Align multiple price series to a common index and drop NaN rows.

    Parameters
    ----------
    prices:
        Either a dict {symbol: pd.Series} or an already-combined DataFrame.
    how:
        'inner' (default) keeps only timestamps present in ALL series.
        'outer' keeps all timestamps and fills gaps with NaN.
    dropna:
        Drop rows that still contain NaN after alignment (default True).

    Returns
    -------
    pd.DataFrame  shape (T, N), columns = symbol names.
    """
    if isinstance(prices, dict):
        panel = pd.concat(prices.values(), axis=1, keys=prices.keys(), join=how)
    else:
        panel = prices.copy()

    if dropna:
        panel = panel.dropna()

    return panel.sort_index()


def normalize(series: pd.Series, window: int | None = None) -> pd.Series:
    """Z-score normalize a Series.

    Parameters
    ----------
    series:
        The time series to normalize.
    window:
        Rolling window size.  None (default) → full-sample statistics.

    Returns
    -------
    pd.Series with mean ≈ 0 and std ≈ 1.
    """
    if window is None:
        mu  = series.mean()
        sig = series.std(ddof=1)
    else:
        mu  = series.rolling(window).mean()
        sig = series.rolling(window).std(ddof=1)

    return (series - mu) / sig


def winsorize(
    series: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """Clip extreme values at given quantile bounds.

    Useful for returns series before backtesting to remove data errors.
    """
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)
