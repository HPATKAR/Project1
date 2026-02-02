"""Granger causality analysis for cross-market lead-lag relationships.

Identifies statistically significant predictive relationships between
yield changes, FX moves, and equity returns to map information flow
across the JGB-related universe.
"""

import logging
from itertools import permutations
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

logger = logging.getLogger(__name__)


def pairwise_granger(
    data: pd.DataFrame,
    max_lag: int = 10,
    significance: float = 0.05,
) -> pd.DataFrame:
    """Run Granger causality tests for all directed pairs in the DataFrame.

    For each ordered pair (X, Y), tests whether past values of X help
    predict Y beyond Y's own past values.  The optimal lag is chosen as
    the one producing the lowest p-value across 1..max_lag.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame of stationary time series (e.g. daily returns or yield
        changes).  Each column is treated as a separate variable.
    max_lag : int, default 10
        Maximum number of lags to test.
    significance : float, default 0.05
        P-value threshold for declaring significance.

    Returns
    -------
    pd.DataFrame
        Columns: cause, effect, optimal_lag, f_stat, p_value, significant.
        One row per directed pair.
    """
    if data.shape[1] < 2:
        raise ValueError("DataFrame must contain at least two columns.")

    data = data.dropna()
    columns = data.columns.tolist()
    results = []

    for cause, effect in permutations(columns, 2):
        pair_data = data[[effect, cause]].copy()

        # Skip pairs with insufficient observations
        if len(pair_data) < max_lag + 2:
            logger.warning(
                "Insufficient observations for pair (%s -> %s), skipping.",
                cause,
                effect,
            )
            continue

        try:
            gc_result = grangercausalitytests(
                pair_data, maxlag=max_lag, verbose=False
            )

            # Find the lag with the smallest p-value (using ssr_ftest)
            best_lag: Optional[int] = None
            best_p: float = 1.0
            best_f: float = 0.0

            for lag in range(1, max_lag + 1):
                test_stats = gc_result[lag][0]
                f_stat, p_value, _, _ = test_stats["ssr_ftest"]
                if p_value < best_p:
                    best_p = p_value
                    best_f = f_stat
                    best_lag = lag

            results.append(
                {
                    "cause": cause,
                    "effect": effect,
                    "optimal_lag": best_lag,
                    "f_stat": round(best_f, 4),
                    "p_value": round(best_p, 6),
                    "significant": best_p < significance,
                }
            )

        except Exception as exc:
            logger.warning(
                "Granger test failed for pair (%s -> %s): %s",
                cause,
                effect,
                exc,
            )
            results.append(
                {
                    "cause": cause,
                    "effect": effect,
                    "optimal_lag": np.nan,
                    "f_stat": np.nan,
                    "p_value": np.nan,
                    "significant": False,
                }
            )

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("p_value", ascending=True).reset_index(
        drop=True
    )

    n_sig = result_df["significant"].sum()
    logger.info(
        "Granger causality: %d/%d pairs significant at %.1f%% level.",
        n_sig,
        len(result_df),
        significance * 100,
    )

    return result_df
