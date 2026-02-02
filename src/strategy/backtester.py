"""Vectorised backtesting utilities for the JGB repricing framework.

Provides three backtesting modes:

1. **Signal backtest** -- apply a long/short signal to a return series
   with configurable transaction costs.
2. **Event-study backtest** -- measure average strategy performance in
   windows around pre-specified event dates (e.g. BoJ meetings, YCC
   adjustments).
3. **Regime-conditional backtest** -- only trade when the regime
   probability exceeds a threshold.

All functions are vectorised using pandas/numpy for speed and
reproducibility.  No path-dependent simulation state is maintained
beyond what is strictly necessary for P&L accounting.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Helper: max drawdown
# ------------------------------------------------------------------
def _max_drawdown(cum_returns: pd.Series) -> float:
    """Compute maximum drawdown from a cumulative-return series.

    Parameters
    ----------
    cum_returns : pd.Series
        Cumulative (compounded) return series, **not** equity curve.
        Expected to start near 0.0.

    Returns
    -------
    float
        Maximum drawdown as a positive fraction (e.g. 0.15 = 15%).
    """
    equity = 1.0 + cum_returns
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(-drawdown.min()) if len(drawdown) > 0 else 0.0


# ------------------------------------------------------------------
# 1. Simple signal backtest
# ------------------------------------------------------------------
def backtest_signal(
    returns: pd.Series,
    signal: pd.Series,
    cost_bps: float = 2.0,
) -> dict:
    """Backtest a long/short signal against an asset return series.

    The signal is applied with a one-period lag (signal at *t* determines
    the position held from *t* to *t+1*) to avoid look-ahead bias.
    Transaction costs are charged on each change in position.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the underlying instrument (e.g. JGB 10Y).
    signal : pd.Series
        Trading signal, typically in {-1, 0, +1} or continuous [-1, 1].
        Must share the same index (dates) as ``returns``, or at least
        overlap.
    cost_bps : float, optional
        One-way transaction cost in basis points.  Default is 2.0 bps.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``cumulative_returns`` : pd.Series -- cumulative strategy P&L.
        - ``sharpe_ratio`` : float -- annualised Sharpe ratio (252 days).
        - ``max_drawdown`` : float -- maximum drawdown (positive number).
        - ``hit_rate`` : float -- fraction of days with positive P&L
          (when position is non-zero).
        - ``avg_win`` : float -- average daily return on winning days.
        - ``avg_loss`` : float -- average daily return on losing days
          (negative number).
        - ``n_trades`` : int -- number of position changes (round-trips
          counted as two changes).
        - ``total_return`` : float -- final cumulative return.
        - ``annual_return`` : float -- annualised arithmetic return.
        - ``annual_vol`` : float -- annualised volatility of strategy
          returns.

    Notes
    -----
    Aligns ``signal`` and ``returns`` on their common index.  Periods
    where either series has NaN are dropped.
    """
    # Align on common dates
    common_idx = returns.index.intersection(signal.index)
    if len(common_idx) == 0:
        return _empty_backtest_result()

    ret = returns.loc[common_idx].copy()
    sig = signal.loc[common_idx].copy()

    # Lag signal by one period to avoid look-ahead
    position = sig.shift(1).fillna(0.0)

    # Transaction costs
    cost_frac = cost_bps / 10_000.0
    turnover = position.diff().abs().fillna(0.0)
    tc = turnover * cost_frac

    # Strategy returns
    strategy_returns = position * ret - tc

    # Drop NaN
    strategy_returns = strategy_returns.dropna()
    if len(strategy_returns) == 0:
        return _empty_backtest_result()

    # Cumulative returns
    cumulative = strategy_returns.cumsum()

    # Metrics
    n_days = len(strategy_returns)
    active_days = strategy_returns[position.loc[strategy_returns.index] != 0]

    wins = active_days[active_days > 0]
    losses = active_days[active_days < 0]

    hit_rate = len(wins) / len(active_days) if len(active_days) > 0 else 0.0
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

    n_trades = int((turnover > 0).sum())

    annual_vol = float(strategy_returns.std() * np.sqrt(252))
    annual_return = float(strategy_returns.mean() * 252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0
    mdd = _max_drawdown(cumulative)

    return {
        "cumulative_returns": cumulative,
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(mdd, 4),
        "hit_rate": round(hit_rate, 4),
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "n_trades": n_trades,
        "total_return": round(float(cumulative.iloc[-1]), 6),
        "annual_return": round(annual_return, 6),
        "annual_vol": round(annual_vol, 6),
    }


def _empty_backtest_result() -> dict:
    """Return a placeholder result when there is no data to backtest."""
    return {
        "cumulative_returns": pd.Series(dtype=float),
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "hit_rate": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "n_trades": 0,
        "total_return": 0.0,
        "annual_return": 0.0,
        "annual_vol": 0.0,
    }


# ------------------------------------------------------------------
# 2. Event-study backtest
# ------------------------------------------------------------------
def backtest_around_events(
    returns: pd.Series,
    signal: pd.Series,
    event_dates: list,
    window_before: int = 20,
    window_after: int = 60,
) -> pd.DataFrame:
    """Event study: average strategy performance around event dates.

    For each event date, the function extracts a window of strategy
    returns from ``window_before`` days before to ``window_after`` days
    after the event, computes the cumulative P&L path, and averages
    across all events.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the underlying instrument.
    signal : pd.Series
        Trading signal (same conventions as :func:`backtest_signal`).
    event_dates : list
        List of event dates (datetime-like).  Dates not present in the
        index are snapped to the nearest available trading day.
    window_before : int, optional
        Number of trading days before the event to include.  Default 20.
    window_after : int, optional
        Number of trading days after the event to include.  Default 60.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``relative_day`` (from ``-window_before``
        to ``+window_after``), with columns:

        - ``avg_cumulative_pnl`` : average cumulative P&L across events.
        - ``median_cumulative_pnl`` : median cumulative P&L.
        - ``std_cumulative_pnl`` : standard deviation of cumulative P&L.
        - ``n_events`` : number of events contributing at each day.
    """
    # Align
    common_idx = returns.index.intersection(signal.index)
    ret = returns.loc[common_idx]
    sig = signal.loc[common_idx]

    # Lagged position
    position = sig.shift(1).fillna(0.0)
    strategy_returns = position * ret

    # Numeric index for efficient slicing
    all_dates = strategy_returns.index
    total_window = window_before + window_after + 1

    event_paths: list[pd.Series] = []

    for event_date in event_dates:
        event_date = pd.Timestamp(event_date)
        # Find nearest trading day
        idx_loc = all_dates.searchsorted(event_date)
        if idx_loc >= len(all_dates):
            idx_loc = len(all_dates) - 1

        start_loc = idx_loc - window_before
        end_loc = idx_loc + window_after + 1

        if start_loc < 0 or end_loc > len(all_dates):
            continue  # skip if window falls outside data range

        window_returns = strategy_returns.iloc[start_loc:end_loc].values
        if len(window_returns) != total_window:
            continue

        cum_pnl = np.cumsum(window_returns)
        relative_days = list(range(-window_before, window_after + 1))
        path = pd.Series(cum_pnl, index=relative_days)
        event_paths.append(path)

    if not event_paths:
        relative_days = list(range(-window_before, window_after + 1))
        return pd.DataFrame(
            {
                "avg_cumulative_pnl": np.zeros(total_window),
                "median_cumulative_pnl": np.zeros(total_window),
                "std_cumulative_pnl": np.zeros(total_window),
                "n_events": np.zeros(total_window, dtype=int),
            },
            index=pd.Index(relative_days, name="relative_day"),
        )

    paths_df = pd.DataFrame(event_paths)

    result = pd.DataFrame(
        {
            "avg_cumulative_pnl": paths_df.mean(axis=0),
            "median_cumulative_pnl": paths_df.median(axis=0),
            "std_cumulative_pnl": paths_df.std(axis=0).fillna(0.0),
            "n_events": paths_df.count(axis=0).astype(int),
        },
    )
    result.index.name = "relative_day"
    return result


# ------------------------------------------------------------------
# 3. Regime-conditional backtest
# ------------------------------------------------------------------
def regime_backtest(
    returns: pd.Series,
    signal: pd.Series,
    regime_prob: pd.Series,
    threshold: float = 0.5,
    cost_bps: float = 2.0,
) -> dict:
    """Backtest a signal only during periods when regime probability exceeds a threshold.

    This is identical to :func:`backtest_signal` except that the signal
    is zeroed-out whenever ``regime_prob <= threshold``, effectively
    keeping the strategy flat outside the target regime.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the underlying instrument.
    signal : pd.Series
        Trading signal.
    regime_prob : pd.Series
        Time series of regime probability (0-1).  Must share index with
        ``returns``.
    threshold : float, optional
        Minimum regime probability required to hold a position.
        Default is 0.5.
    cost_bps : float, optional
        One-way transaction cost in basis points.  Default is 2.0 bps.

    Returns
    -------
    dict
        Same output format as :func:`backtest_signal`, plus:

        - ``pct_time_active`` : float -- fraction of days the strategy
          was active (regime_prob > threshold).
        - ``regime_threshold`` : float -- the threshold used.
    """
    # Align all three series
    common_idx = (
        returns.index
        .intersection(signal.index)
        .intersection(regime_prob.index)
    )

    if len(common_idx) == 0:
        result = _empty_backtest_result()
        result["pct_time_active"] = 0.0
        result["regime_threshold"] = threshold
        return result

    ret = returns.loc[common_idx]
    sig = signal.loc[common_idx].copy()
    rp = regime_prob.loc[common_idx]

    # Zero out signal when regime probability is below threshold
    regime_mask = rp > threshold
    filtered_signal = sig.where(regime_mask, 0.0)

    pct_active = float(regime_mask.mean())

    result = backtest_signal(ret, filtered_signal, cost_bps=cost_bps)
    result["pct_time_active"] = round(pct_active, 4)
    result["regime_threshold"] = threshold

    return result
