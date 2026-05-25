"""
Signal generation for the Z-Score pairs trading strategy.

Pipeline:
    log_prices → spread → z-score → position signals

Signal convention
-----------------
+1  Long spread  (spread below lower band: BTC undervalued → Long BTC / Short ETH)
-1  Short spread (spread above upper band: BTC overvalued → Short BTC / Long ETH)
 0  No position / flat
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ZSCORE_ENTRY, ZSCORE_EXIT, ZSCORE_WINDOW


# ---------------------------------------------------------------------------
# Spread
# ---------------------------------------------------------------------------

def compute_spread(
    log_price_a: pd.Series,
    log_price_b: pd.Series,
    beta:        float,
) -> pd.Series:
    """Compute the log-price spread (ECT proxy).

    spread_t = ln(A_t) - β · ln(B_t)

    This is the simplest spread definition.  For a more precise ECT use
    models.vecm.get_ect() which includes the constant term.

    Parameters
    ----------
    log_price_a, log_price_b:
        Log-price Series for asset A and B, same datetime index.
    beta:
        Cointegrating coefficient (from JohansenResult.beta or VECMSummary).

    Returns
    -------
    pd.Series  named 'spread'
    """
    aligned = pd.concat([log_price_a, log_price_b], axis=1).dropna()
    a = aligned.iloc[:, 0]
    b = aligned.iloc[:, 1]
    return (a - beta * b).rename("spread")


# ---------------------------------------------------------------------------
# Z-score
# ---------------------------------------------------------------------------

def compute_zscore(
    spread: pd.Series,
    window: int | None = ZSCORE_WINDOW,
) -> pd.Series:
    """Standardize the spread into a Z-score.

    Parameters
    ----------
    spread:
        Spread Series (output of compute_spread or get_ect).
    window:
        None → use full-sample mean and std (fixed parameters).
        int  → rolling window (re-estimated at each step, walk-forward safe).

    Returns
    -------
    pd.Series  named 'zscore'
    """
    if window is None:
        mu  = spread.mean()
        sig = spread.std(ddof=1)
        zscore = (spread - mu) / sig
    else:
        mu  = spread.rolling(window, min_periods=window).mean()
        sig = spread.rolling(window, min_periods=window).std(ddof=1)
        zscore = (spread - mu) / sig

    return zscore.rename("zscore")


# ---------------------------------------------------------------------------
# Position signals
# ---------------------------------------------------------------------------

def generate_signals(
    zscore:    pd.Series,
    entry:     float = ZSCORE_ENTRY,
    exit_:     float = ZSCORE_EXIT,
) -> pd.Series:
    """Convert Z-score series into discrete position signals.

    Entry / exit logic
    ------------------
    Open long  (+1) when z <  -entry   (spread abnormally low)
    Open short (-1) when z >  +entry   (spread abnormally high)
    Close (+1→0)    when z >=  exit_   (spread mean-reverted from below)
    Close (-1→0)    when z <= -exit_   (spread mean-reverted from above)

    This implements a sticky regime: once a position is open it is held
    until the exit condition is met, even if the entry threshold is not
    re-crossed.

    Parameters
    ----------
    zscore:
        Z-score Series.
    entry:
        Absolute z-score threshold to enter a trade (default 2.0).
    exit_:
        Absolute z-score threshold to exit / close (default 0.0).

    Returns
    -------
    pd.Series  of int8 {-1, 0, 1}, named 'signal'
    """
    signals = pd.Series(np.zeros(len(zscore), dtype=np.int8),
                        index=zscore.index, name="signal")
    position = 0

    for i, z in enumerate(zscore):
        if np.isnan(z):
            signals.iloc[i] = 0
            continue

        if position == 0:
            if z > entry:
                position = -1      # short spread
            elif z < -entry:
                position = 1       # long spread
        elif position == 1:
            if z >= exit_:
                position = 0       # close long
        elif position == -1:
            if z <= -exit_:
                position = 0       # close short

        signals.iloc[i] = position

    return signals


def signals_summary(signals: pd.Series, zscore: pd.Series) -> pd.DataFrame:
    """Return a DataFrame with z-score, signal, and trade annotation."""
    df = pd.concat([zscore, signals], axis=1)
    df.columns = ["zscore", "signal"]

    prev = df["signal"].shift(1).fillna(0)
    df["trade"] = np.where(
        (df["signal"] != 0) & (prev == 0), "entry",
        np.where((df["signal"] == 0) & (prev != 0), "exit", "")
    )
    return df
