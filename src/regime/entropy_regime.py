"""
Entropy-based regime detection for JGB yield dynamics.

Uses information-theoretic measures -- permutation entropy and sample
entropy -- computed on a rolling basis to detect changes in the
*complexity* or *predictability* of yield movements.

    * **Low entropy** periods correspond to highly predictable,
      BoJ-suppressed dynamics.
    * **High entropy** spikes indicate increased randomness /
      market-driven repricing.

Depends on ``antropy >= 0.1.6``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import antropy

logger = logging.getLogger(__name__)


def rolling_permutation_entropy(
    series: pd.Series,
    window: int = 120,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> pd.Series:
    """Compute rolling permutation entropy over a time series.

    Permutation entropy (Bandt & Pompe, 2002) quantifies the complexity
    of a time series by examining the relative ordering of consecutive
    values.

    Parameters
    ----------
    series : pd.Series
        Input time series (e.g. daily JGB yield changes).
    window : int, default 120
        Rolling window size in observations (~6 months of daily data).
    order : int, default 3
        Embedding dimension (order of permutation patterns).  Typical
        values are 3--7.
    delay : int, default 1
        Time delay between elements within each pattern.
    normalize : bool, default True
        If ``True``, entropy is normalised to [0, 1].

    Returns
    -------
    pd.Series
        Rolling permutation entropy, indexed by the *end* of each
        window.  The first ``window - 1`` values are ``NaN``.
    """
    if window < 2 * order:
        raise ValueError(
            f"Window ({window}) must be at least 2 * order ({2 * order})."
        )

    values = series.values
    n = len(values)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        segment = values[i - window + 1 : i + 1]
        if np.isnan(segment).any():
            continue
        result[i] = antropy.perm_entropy(
            segment, order=order, delay=delay, normalize=normalize
        )

    out = pd.Series(result, index=series.index, name="perm_entropy")
    logger.info(
        "Computed rolling permutation entropy (window=%d, order=%d).", window, order
    )
    return out


def rolling_sample_entropy(
    series: pd.Series,
    window: int = 120,
    order: int = 2,
    metric: str = "chebyshev",
) -> pd.Series:
    """Compute rolling sample entropy over a time series.

    Sample entropy (Richman & Moorman, 2000) measures the regularity of
    a time series -- lower values indicate more self-similar (regular)
    dynamics.

    Parameters
    ----------
    series : pd.Series
        Input time series.
    window : int, default 120
        Rolling window size.
    order : int, default 2
        Embedding dimension.
    metric : str, default "chebyshev"
        Distance metric passed to ``antropy.sample_entropy``.

    Returns
    -------
    pd.Series
        Rolling sample entropy with the same index as *series*.
    """
    values = series.values
    n = len(values)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        segment = values[i - window + 1 : i + 1]
        if np.isnan(segment).any():
            continue
        try:
            result[i] = antropy.sample_entropy(segment, order=order)
        except Exception:  # noqa: BLE001
            # sample_entropy can fail on constant segments
            result[i] = 0.0

    out = pd.Series(result, index=series.index, name="sample_entropy")
    logger.info(
        "Computed rolling sample entropy (window=%d, order=%d).", window, order
    )
    return out


def entropy_regime_signal(
    entropy_series: pd.Series,
    threshold_std: float = 1.5,
    rolling_window: int = 252,
) -> pd.Series:
    """Detect regime-change signals from an entropy series.

    A regime-change signal is raised when the entropy deviates from its
    own rolling mean by more than ``threshold_std`` rolling standard
    deviations.

    Parameters
    ----------
    entropy_series : pd.Series
        Rolling entropy series (e.g. from
        :func:`rolling_permutation_entropy`).
    threshold_std : float, default 1.5
        Number of rolling standard deviations beyond the rolling mean
        that triggers a regime signal.
    rolling_window : int, default 252
        Window for computing rolling mean and standard deviation of the
        entropy series (~1 year of daily data).

    Returns
    -------
    pd.Series
        Binary signal: ``1`` where entropy exceeds the upper threshold
        (market-driven regime), ``0`` otherwise (suppressed).  Values
        are ``NaN`` where the rolling statistics are not yet available.
    """
    rolling_mean = entropy_series.rolling(window=rolling_window, min_periods=60).mean()
    rolling_std = entropy_series.rolling(window=rolling_window, min_periods=60).std()

    upper_threshold = rolling_mean + threshold_std * rolling_std

    signal = (entropy_series > upper_threshold).astype(float)
    signal[rolling_mean.isna()] = np.nan

    signal.name = "entropy_regime_signal"
    logger.info(
        "Entropy regime signal: %d regime-change observations detected "
        "out of %d valid observations.",
        int(signal.sum()),
        int(signal.notna().sum()),
    )
    return signal
