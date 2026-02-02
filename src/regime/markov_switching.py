"""
Markov regime-switching models for JGB yield dynamics.

Implements Hamilton (1989) Markov-switching regression to estimate smoothed
regime probabilities from univariate JGB yield-change series.  Two regimes
are identified by default:

    * Regime 0 -- **BoJ-suppressed**: low-mean, low-variance yield changes
      consistent with yield-curve control (YCC).
    * Regime 1 -- **Market-driven**: higher mean/variance reflecting
      repricing episodes or policy normalisation.

Depends on ``statsmodels >= 0.14``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

logger = logging.getLogger(__name__)


def fit_markov_regime(
    series: pd.Series,
    k_regimes: int = 2,
    switching_variance: bool = True,
) -> Dict[str, Any]:
    """Fit a Hamilton Markov-switching regression on a yield-change series.

    Parameters
    ----------
    series : pd.Series
        Stationary yield-change series (e.g. daily first-differences of
        10-year JGB yield in basis points).  Must have a
        ``DatetimeIndex``.
    k_regimes : int, default 2
        Number of latent regimes.
    switching_variance : bool, default True
        If ``True`` the model allows the error variance to switch across
        regimes, capturing the well-documented difference in volatility
        between BoJ-suppressed and market-driven periods.

    Returns
    -------
    dict
        ``regime_probabilities`` : pd.DataFrame
            Smoothed (Kim) regime probabilities, shape (T, k_regimes).
        ``regime_means`` : np.ndarray
            Estimated mean for each regime.
        ``regime_variances`` : np.ndarray
            Estimated variance for each regime.
        ``model_result`` : MarkovRegressionResults
            Full statsmodels result object for diagnostics.

    Raises
    ------
    ValueError
        If the input series contains NaN values or is too short for
        reliable estimation (< 100 observations).
    """
    if series.isna().any():
        raise ValueError(
            "Input series contains NaN values. Please drop or fill them "
            "before fitting the Markov-switching model."
        )
    if len(series) < 100:
        raise ValueError(
            f"Series length {len(series)} is too short for reliable "
            f"Markov-switching estimation (need >= 100 observations)."
        )

    logger.info(
        "Fitting MarkovRegression with k_regimes=%d, switching_variance=%s "
        "on %d observations.",
        k_regimes,
        switching_variance,
        len(series),
    )

    model = MarkovRegression(
        series,
        k_regimes=k_regimes,
        switching_variance=switching_variance,
    )
    result = model.fit(disp=False)

    # Smoothed (Kim) probabilities -- statsmodels may return a DataFrame or ndarray
    raw_probs = result.smoothed_marginal_probabilities
    if isinstance(raw_probs, pd.DataFrame):
        smoothed_probs = raw_probs.copy()
        smoothed_probs.columns = [f"regime_{i}" for i in range(k_regimes)]
        smoothed_probs.index = series.index
    else:
        # ndarray: may be (k_regimes, T) or (T, k_regimes)
        if raw_probs.shape[0] == k_regimes and raw_probs.shape[1] == len(series):
            raw_probs = raw_probs.T
        smoothed_probs = pd.DataFrame(
            raw_probs,
            index=series.index,
            columns=[f"regime_{i}" for i in range(k_regimes)],
        )

    # Extract regime-specific means and variances from parameters
    regime_means = np.array(
        [result.params[f"const[{i}]"] for i in range(k_regimes)]
    )
    regime_variances = np.array(
        [
            result.params.get(f"sigma2[{i}]", result.params.get("sigma2", np.nan))
            for i in range(k_regimes)
        ]
    )

    logger.info(
        "Estimation complete. Regime means: %s, regime variances: %s",
        regime_means,
        regime_variances,
    )

    return {
        "regime_probabilities": smoothed_probs,
        "regime_means": regime_means,
        "regime_variances": regime_variances,
        "model_result": result,
    }


def classify_current_regime(regime_probs: pd.DataFrame) -> int:
    """Return the most likely current regime from smoothed probabilities.

    Parameters
    ----------
    regime_probs : pd.DataFrame
        Smoothed regime probabilities as returned by
        :func:`fit_markov_regime` (key ``regime_probabilities``).  Each
        column corresponds to one regime.

    Returns
    -------
    int
        Index of the most probable regime at the last observation.
        Convention: 0 = suppressed, 1 = market-driven.
    """
    last_row: pd.Series = regime_probs.iloc[-1]
    current_regime: int = int(last_row.values.argmax())
    logger.info(
        "Current regime classified as %d (probability %.4f).",
        current_regime,
        last_row.iloc[current_regime],
    )
    return current_regime
