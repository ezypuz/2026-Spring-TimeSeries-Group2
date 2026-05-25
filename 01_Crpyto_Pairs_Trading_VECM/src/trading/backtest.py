"""
Vectorized backtest engine for the Z-Score pairs trading strategy.

Return model
------------
We hold "one unit of spread":
    position = +1  →  log_return ≈ Δln(A) - β · Δln(B)
    position = -1  →  log_return ≈ -(Δln(A) - β · Δln(B))

Signals are shifted by 1 period to avoid look-ahead bias (we act on the
*close* of the signal bar, not the open of the next bar – conservative).

Transaction costs are charged on every position change proportional to
the absolute change in position.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import INITIAL_CAPITAL, TRANSACTION_COST


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    equity_curve:  pd.Series            # portfolio value over time
    positions:     pd.Series            # lagged signals (actual holdings)
    spread_returns: pd.Series           # raw spread log-returns
    pnl:           pd.Series            # period-level P&L
    metrics:       dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.metrics:
            self.metrics = compute_metrics(self.equity_curve, self.pnl)


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------

def run_backtest(
    signals:      pd.Series,
    log_price_a:  pd.Series,
    log_price_b:  pd.Series,
    beta:         float,
    initial_capital: float = INITIAL_CAPITAL,
    transaction_cost: float = TRANSACTION_COST,
) -> BacktestResult:
    """Run a vectorized pairs-trading backtest.

    Parameters
    ----------
    signals:
        Position signal Series {-1, 0, 1} from generate_signals().
    log_price_a, log_price_b:
        Log-price Series for asset A (BTC) and B (ETH).
    beta:
        Cointegrating coefficient used to build the spread.
    initial_capital:
        Starting portfolio value in USDT.
    transaction_cost:
        One-way cost as a fraction of notional (e.g. 0.001 = 0.1 %).
        Applied on every position change.

    Returns
    -------
    BacktestResult
    """
    # Align all series to a common index
    common = signals.index \
        .intersection(log_price_a.index) \
        .intersection(log_price_b.index)

    sig  = signals.reindex(common)
    la   = log_price_a.reindex(common)
    lb   = log_price_b.reindex(common)

    # Spread log-return per period
    spread_ret = la.diff() - beta * lb.diff()

    # Lag signals: we act at the END of the bar that generated the signal
    position = sig.shift(1).fillna(0).astype(float)

    # Gross P&L
    pnl = position * spread_ret

    # Transaction costs: charged on absolute position change
    pos_change = position.diff().abs().fillna(0)
    cost = pos_change * transaction_cost
    pnl  = pnl - cost

    # Equity curve (multiplicative compounding)
    equity = initial_capital * (1 + pnl).cumprod()

    return BacktestResult(
        equity_curve   = equity,
        positions      = position,
        spread_returns = spread_ret,
        pnl            = pnl,
    )


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    equity: pd.Series,
    pnl:    pd.Series | None = None,
    periods_per_year: int | None = None,
) -> dict:
    """Compute standard quantitative trading metrics.

    Parameters
    ----------
    equity:
        Equity curve (portfolio value, not returns).
    pnl:
        Period P&L Series.  If None, derived from equity curve.
    periods_per_year:
        Annualisation factor.  Inferred from the index frequency if None.
        Hourly  → 24 × 365 = 8760
        Daily   → 365
        15-min  → 96 × 365

    Returns
    -------
    dict with keys:
        total_return, cagr, sharpe, sortino, max_drawdown,
        calmar, num_trades, win_rate, avg_win, avg_loss
    """
    if pnl is None:
        pnl = equity.pct_change().dropna()

    epy = _infer_periods_per_year(equity, periods_per_year)

    # --- Returns ---
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    n_years      = len(equity) / epy
    cagr         = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1) \
                   if n_years > 0 else float("nan")

    # --- Risk-adjusted ---
    mean_r = pnl.mean()
    std_r  = pnl.std(ddof=1)
    sharpe = float(mean_r / std_r * np.sqrt(epy)) if std_r > 0 else float("nan")

    downside = pnl[pnl < 0].std(ddof=1)
    sortino  = float(mean_r / downside * np.sqrt(epy)) if downside > 0 else float("nan")

    # --- Drawdown ---
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_dd   = float(drawdown.min())
    calmar   = float(cagr / abs(max_dd)) if max_dd != 0 else float("nan")

    # --- Trade stats (count position changes as trade boundaries) ---
    non_zero = pnl[pnl != 0]
    num_trades = int((pnl != 0).sum())
    wins  = non_zero[non_zero > 0]
    losses= non_zero[non_zero < 0]
    win_rate = float(len(wins) / len(non_zero)) if len(non_zero) > 0 else float("nan")
    avg_win  = float(wins.mean())   if len(wins)   > 0 else float("nan")
    avg_loss = float(losses.mean()) if len(losses) > 0 else float("nan")

    return {
        "total_return":  total_return,
        "cagr":          cagr,
        "sharpe":        sharpe,
        "sortino":       sortino,
        "max_drawdown":  max_dd,
        "calmar":        calmar,
        "num_trades":    num_trades,
        "win_rate":      win_rate,
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
    }


def metrics_table(result: BacktestResult) -> pd.DataFrame:
    """Format BacktestResult metrics as a single-column display table."""
    m = result.metrics
    rows = {
        "Total Return":  f"{m['total_return']:+.2%}",
        "CAGR":          f"{m['cagr']:+.2%}",
        "Sharpe Ratio":  f"{m['sharpe']:.3f}",
        "Sortino Ratio": f"{m['sortino']:.3f}",
        "Max Drawdown":  f"{m['max_drawdown']:.2%}",
        "Calmar Ratio":  f"{m['calmar']:.3f}",
        "# Trades":      f"{m['num_trades']:,}",
        "Win Rate":      f"{m['win_rate']:.2%}",
        "Avg Win":       f"{m['avg_win']:+.4f}",
        "Avg Loss":      f"{m['avg_loss']:+.4f}",
    }
    return pd.DataFrame.from_dict(rows, orient="index", columns=["Value"])


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _infer_periods_per_year(equity: pd.Series, override: int | None) -> int:
    if override is not None:
        return override

    if not isinstance(equity.index, pd.DatetimeIndex) or len(equity) < 2:
        return 365   # fallback

    delta = (equity.index[-1] - equity.index[0]).total_seconds()
    freq_sec = delta / (len(equity) - 1)

    HOUR  = 3_600
    DAY   = 86_400
    YEAR  = 365 * DAY

    return max(1, round(YEAR / freq_sec))
