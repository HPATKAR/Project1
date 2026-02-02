"""Diebold-Yilmaz spillover index based on generalized VAR forecast error
variance decomposition.

References
----------
Diebold, F.X. and Yilmaz, K. (2012), "Better to Give than to Receive:
Predictive Directional Measurement of Volatility Spillovers",
International Journal of Forecasting, 28(1), 57-66.
"""

import logging
from typing import Dict, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

logger = logging.getLogger(__name__)


def _generalized_fevd(
    var_result,
    forecast_horizon: int,
) -> np.ndarray:
    """Compute generalized forecast error variance decomposition (GFEVD).

    Unlike orthogonalized FEVD, the generalized approach does not depend
    on variable ordering.  The method follows Pesaran & Shin (1998).

    Parameters
    ----------
    var_result : statsmodels VARResults
        Fitted VAR model.
    forecast_horizon : int
        Number of steps ahead for FEVD.

    Returns
    -------
    np.ndarray
        (K x K) matrix where element (i, j) is the fraction of the
        H-step forecast error variance of variable i attributable to
        shocks in variable j.  Rows are normalized to sum to 1.
    """
    ma_coefs = var_result.ma_rep(forecast_horizon - 1)  # (H, K, K)
    sigma_u = np.array(var_result.sigma_u)  # (K, K) residual covariance
    k = sigma_u.shape[0]

    # Diagonal standard deviations of the residuals
    sigma_diag = np.sqrt(np.diag(sigma_u))

    # Accumulate the generalized FEVD matrix
    theta = np.zeros((k, k))

    for i in range(k):
        for j in range(k):
            numerator = 0.0
            denominator = 0.0
            for h in range(forecast_horizon):
                ei = np.zeros(k)
                ei[i] = 1.0
                ej = np.zeros(k)
                ej[j] = 1.0

                # Generalized impulse: (sigma_jj)^{-0.5} * Psi_h * Sigma * e_j
                impulse = ma_coefs[h] @ sigma_u @ ej
                numerator += (ei @ impulse) ** 2

                # Total variance contribution
                denominator += ei @ ma_coefs[h] @ sigma_u @ ma_coefs[h].T @ ei

            # Scale by inverse of sigma_jj
            theta[i, j] = (numerator / (sigma_diag[j] ** 2)) / max(
                denominator, 1e-12
            )

    # Normalize rows to sum to 1
    row_sums = theta.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    theta_tilde = theta / row_sums

    return theta_tilde


def compute_spillover_index(
    data: pd.DataFrame,
    var_lags: int = 4,
    forecast_horizon: int = 10,
) -> Dict[str, Union[float, pd.Series, pd.DataFrame]]:
    """Compute the Diebold-Yilmaz spillover index from a VAR model.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame of stationary time series (e.g. daily yield changes
        or returns).  Columns are variable names.
    var_lags : int, default 4
        Number of lags in the VAR specification.
    forecast_horizon : int, default 10
        Forecast horizon for the variance decomposition.

    Returns
    -------
    dict
        - ``total_spillover`` : float
            System-wide spillover index (0-100 scale).
        - ``directional_to`` : pd.Series
            Spillover *to* each variable from all others.
        - ``directional_from`` : pd.Series
            Spillover *from* each variable to all others.
        - ``net_spillover`` : pd.Series
            Net spillover (to - from) for each variable.
        - ``spillover_matrix`` : pd.DataFrame
            Full (K x K) normalized variance decomposition matrix.
    """
    data = data.dropna()
    columns = data.columns.tolist()
    k = len(columns)

    if k < 2:
        raise ValueError("Need at least 2 variables for spillover analysis.")

    # Fit VAR
    model = VAR(data)
    var_result = model.fit(maxlags=var_lags, ic=None)
    logger.info(
        "VAR(%d) fitted on %d observations, %d variables.",
        var_lags,
        len(data),
        k,
    )

    # Generalized FEVD
    theta = _generalized_fevd(var_result, forecast_horizon)

    spillover_matrix = pd.DataFrame(theta, index=columns, columns=columns)

    # Total spillover index: sum of off-diagonal elements / k * 100
    off_diag_sum = theta.sum() - np.trace(theta)
    total_spillover = (off_diag_sum / k) * 100.0

    # Directional TO others: column sums minus own (how much j spills to others)
    directional_to = pd.Series(
        [theta[:, j].sum() - theta[j, j] for j in range(k)],
        index=columns,
        name="directional_to",
    ) * 100.0

    # Directional FROM others: row sums minus own (how much i receives)
    directional_from = pd.Series(
        [theta[i, :].sum() - theta[i, i] for i in range(k)],
        index=columns,
        name="directional_from",
    ) * 100.0

    # Net spillover
    net_spillover = directional_to - directional_from
    net_spillover.name = "net_spillover"

    logger.info("Total spillover index: %.2f%%", total_spillover)

    return {
        "total_spillover": round(total_spillover, 4),
        "directional_to": directional_to.round(4),
        "directional_from": directional_from.round(4),
        "net_spillover": net_spillover.round(4),
        "spillover_matrix": spillover_matrix.round(6),
    }


def rolling_spillover(
    data: pd.DataFrame,
    window: int = 200,
    var_lags: int = 4,
    forecast_horizon: int = 10,
) -> pd.Series:
    """Compute a rolling-window Diebold-Yilmaz total spillover index.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame of stationary time series.
    window : int, default 200
        Rolling window size (business days).
    var_lags : int, default 4
        VAR lag order.
    forecast_horizon : int, default 10
        FEVD forecast horizon.

    Returns
    -------
    pd.Series
        Time series of total spillover index values, indexed by the
        last date of each rolling window.
    """
    data = data.dropna()
    n = len(data)

    if n < window:
        raise ValueError(
            f"Insufficient data: {n} observations for window={window}."
        )

    dates = []
    spillovers = []

    for end in range(window, n + 1):
        start = end - window
        window_data = data.iloc[start:end]

        try:
            result = compute_spillover_index(
                window_data,
                var_lags=var_lags,
                forecast_horizon=forecast_horizon,
            )
            spillovers.append(result["total_spillover"])
        except Exception as exc:
            logger.debug(
                "Rolling spillover failed at index %d: %s", end - 1, exc
            )
            spillovers.append(np.nan)

        dates.append(data.index[end - 1])

    result_series = pd.Series(spillovers, index=dates, name="total_spillover")

    logger.info(
        "Rolling spillover computed: %d windows, mean=%.2f%%.",
        len(result_series),
        result_series.mean(),
    )

    return result_series
