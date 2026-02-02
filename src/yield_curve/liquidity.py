"""
Bond-market liquidity metrics for JGB analysis.

Implements standard microstructure-based liquidity measures used to monitor
trading conditions in the JGB market:

* **Amihud (2002)** illiquidity ratio -- price impact per unit of volume.
* **Roll (1984)** effective spread -- inferred from negative auto-covariance
  of returns.
* **Composite liquidity index** -- z-score aggregation of multiple metrics
  into a single tradeable signal.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Amihud illiquidity
# ---------------------------------------------------------------------------

def amihud_illiquidity(
    returns: pd.Series,
    volume: pd.Series,
    window: int = 22,
) -> pd.Series:
    """Compute the Amihud (2002) illiquidity ratio.

    .. math::

        ILLIQ_t = \\frac{1}{W} \\sum_{d=t-W+1}^{t}
                  \\frac{|r_d|}{V_d}

    where *r_d* is the daily return and *V_d* is the daily trading volume
    (in monetary terms).

    Parameters
    ----------
    returns : pd.Series
        Daily bond returns (or yield changes as a proxy).
    volume : pd.Series
        Daily trading volume, aligned with *returns*.  Must be strictly
        positive where non-NaN.
    window : int, default 22
        Rolling window length (trading days).  22 ~ one month.

    Returns
    -------
    pd.Series
        Rolling Amihud illiquidity ratio.  Higher values indicate *less*
        liquid conditions.

    Raises
    ------
    ValueError
        If *returns* and *volume* have mismatched indices.
    """
    if not returns.index.equals(volume.index):
        # Align on intersection
        common = returns.index.intersection(volume.index)
        if common.empty:
            raise ValueError("returns and volume share no common dates.")
        returns = returns.loc[common]
        volume = volume.loc[common]

    # Guard against zero volume
    safe_volume = volume.replace(0, np.nan)
    daily_ratio = returns.abs() / safe_volume

    illiq = daily_ratio.rolling(window=window, min_periods=max(window // 2, 1)).mean()
    illiq.name = "amihud_illiquidity"
    return illiq


# ---------------------------------------------------------------------------
# Roll effective spread
# ---------------------------------------------------------------------------

def roll_measure(
    returns: pd.Series,
    window: int = 22,
) -> pd.Series:
    """Estimate the Roll (1984) effective bid-ask spread.

    .. math::

        \\text{Roll} = 2 \\sqrt{-\\text{Cov}(r_t, r_{t-1})}

    The measure is defined only when the first-order auto-covariance of
    returns is negative (consistent with bid-ask bounce).  Where the
    auto-covariance is non-negative the spread is set to zero (market is
    not exhibiting bounce behaviour).

    Parameters
    ----------
    returns : pd.Series
        Daily returns.
    window : int, default 22
        Rolling window for the auto-covariance estimate.

    Returns
    -------
    pd.Series
        Rolling Roll effective spread estimate.
    """
    ret = returns.copy().dropna()

    def _roll_window(arr: np.ndarray) -> float:
        if len(arr) < 2:
            return np.nan
        autocov = np.cov(arr[1:], arr[:-1])[0, 1]
        if autocov < 0:
            return 2.0 * np.sqrt(-autocov)
        return 0.0

    roll = ret.rolling(window=window, min_periods=max(window // 2, 1)).apply(
        _roll_window, raw=True,
    )
    roll.name = "roll_spread"
    return roll


# ---------------------------------------------------------------------------
# Composite liquidity index
# ---------------------------------------------------------------------------

def composite_liquidity_index(
    metrics_dict: Dict[str, pd.Series],
    method: Literal["z_score", "min_max", "rank"] = "z_score",
    expanding_window: Optional[int] = None,
) -> pd.Series:
    """Combine multiple liquidity metrics into a single composite index.

    Each metric is first normalised (sign-adjusted so that *higher* values
    always mean *less* liquidity), then aggregated by equal-weight average.

    Parameters
    ----------
    metrics_dict : dict of str -> pd.Series
        Mapping of metric names to their time series.  All series should be
        aligned to the same date index (missing dates are handled via outer
        join + forward fill).
    method : {'z_score', 'min_max', 'rank'}, default 'z_score'
        Normalisation method:

        * ``'z_score'``  -- subtract expanding mean, divide by expanding std.
        * ``'min_max'``  -- scale to [0, 1] using expanding min/max.
        * ``'rank'``     -- percentile rank within expanding window.
    expanding_window : int, optional
        If provided, use a rolling (rather than expanding) window for the
        normalisation statistics.  ``None`` means full expanding window.

    Returns
    -------
    pd.Series
        Composite liquidity index.  Higher values signal *worse* liquidity.
    """
    if not metrics_dict:
        raise ValueError("metrics_dict must contain at least one metric.")

    # Align all series
    combined = pd.DataFrame(metrics_dict)
    combined = combined.sort_index().ffill()

    normalised = pd.DataFrame(index=combined.index)

    for col in combined.columns:
        s = combined[col]

        if method == "z_score":
            if expanding_window is not None:
                roll_mean = s.rolling(window=expanding_window, min_periods=1).mean()
                roll_std = s.rolling(window=expanding_window, min_periods=1).std()
            else:
                roll_mean = s.expanding(min_periods=1).mean()
                roll_std = s.expanding(min_periods=1).std()
            normalised[col] = (s - roll_mean) / roll_std.replace(0, np.nan)

        elif method == "min_max":
            if expanding_window is not None:
                roll_min = s.rolling(window=expanding_window, min_periods=1).min()
                roll_max = s.rolling(window=expanding_window, min_periods=1).max()
            else:
                roll_min = s.expanding(min_periods=1).min()
                roll_max = s.expanding(min_periods=1).max()
            denom = (roll_max - roll_min).replace(0, np.nan)
            normalised[col] = (s - roll_min) / denom

        elif method == "rank":
            if expanding_window is not None:
                normalised[col] = s.rolling(
                    window=expanding_window, min_periods=1,
                ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
            else:
                normalised[col] = s.expanding(min_periods=1).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False,
                )
        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose from 'z_score', 'min_max', 'rank'."
            )

    composite = normalised.mean(axis=1)
    composite.name = "composite_liquidity_index"
    return composite
