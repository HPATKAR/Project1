"""Dynamic Conditional Correlation (DCC) estimation via GARCH + EWMA proxy.

Estimates time-varying correlations between asset returns using:
1. Univariate GARCH(p,q) for conditional volatilities (via ``arch`` package).
2. Exponentially weighted moving average (EWMA) correlation on standardized
   residuals as a practical DCC approximation (avoids fragile bivariate
   likelihood optimization while capturing the key dynamics).

References
----------
Engle, R. (2002), "Dynamic Conditional Correlation: A Simple Class of
Multivariate Generalized Autoregressive Conditional Heteroskedasticity
Models", Journal of Business & Economic Statistics, 20(3), 339-350.
"""

import logging
from itertools import combinations
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from arch import arch_model

logger = logging.getLogger(__name__)


def fit_univariate_garch(
    series: pd.Series,
    p: int = 1,
    q: int = 1,
) -> Tuple[pd.Series, pd.Series]:
    """Fit a univariate GARCH(p, q) model to a return series.

    Parameters
    ----------
    series : pd.Series
        Return series (should be demeaned or approximately zero-mean).
    p : int, default 1
        GARCH lag order (for conditional variance).
    q : int, default 1
        ARCH lag order (for squared residual terms).

    Returns
    -------
    tuple of (pd.Series, pd.Series)
        - ``std_resid``: Standardized residuals (residuals / conditional vol).
        - ``cond_vol``: Conditional volatility (annualized not applied;
          same frequency as input).
    """
    series = series.dropna()

    if len(series) < 50:
        raise ValueError(
            f"Insufficient data for GARCH estimation: {len(series)} obs."
        )

    # Scale returns to percentage for numerical stability in arch package
    scale_factor = 100.0
    scaled = series * scale_factor

    model = arch_model(
        scaled,
        vol="Garch",
        p=p,
        q=q,
        mean="Constant",
        dist="normal",
        rescale=False,
    )
    result = model.fit(disp="off", show_warning=False)

    # Conditional volatility (rescale back)
    cond_vol = result.conditional_volatility / scale_factor
    cond_vol.name = series.name

    # Standardized residuals
    resid = result.resid / scale_factor
    std_resid = resid / cond_vol.replace(0, np.nan)
    std_resid.name = series.name

    logger.debug(
        "GARCH(%d,%d) fitted for %s: omega=%.4e, alpha=%.4f, beta=%.4f",
        p,
        q,
        series.name,
        result.params.get("omega", np.nan),
        result.params.get("alpha[1]", np.nan),
        result.params.get("beta[1]", np.nan),
    )

    return std_resid, cond_vol


def _ewma_correlation(
    resid_i: np.ndarray,
    resid_j: np.ndarray,
    decay: float = 0.94,
) -> np.ndarray:
    """Compute EWMA dynamic correlation between two standardized residual
    series.

    Uses the RiskMetrics-style exponential smoother:
        Q_{t} = (1 - lambda) * e_{i,t} * e_{j,t} + lambda * Q_{t-1}

    Then normalizes to get correlations.

    Parameters
    ----------
    resid_i, resid_j : np.ndarray
        Standardized residual arrays of equal length.
    decay : float, default 0.94
        EWMA decay factor (lambda).  0.94 is the RiskMetrics daily default.

    Returns
    -------
    np.ndarray
        Time-varying correlation series.
    """
    n = len(resid_i)
    q_ij = np.zeros(n)
    q_ii = np.zeros(n)
    q_jj = np.zeros(n)
    rho = np.zeros(n)

    # Initialize with unconditional values (first 10 obs or available)
    init_window = min(20, n)
    q_ij[0] = np.mean(resid_i[:init_window] * resid_j[:init_window])
    q_ii[0] = np.mean(resid_i[:init_window] ** 2)
    q_jj[0] = np.mean(resid_j[:init_window] ** 2)

    denom = np.sqrt(max(q_ii[0], 1e-12) * max(q_jj[0], 1e-12))
    rho[0] = q_ij[0] / denom if denom > 0 else 0.0

    for t in range(1, n):
        q_ij[t] = (1 - decay) * resid_i[t] * resid_j[t] + decay * q_ij[t - 1]
        q_ii[t] = (1 - decay) * resid_i[t] ** 2 + decay * q_ii[t - 1]
        q_jj[t] = (1 - decay) * resid_j[t] ** 2 + decay * q_jj[t - 1]

        denom = np.sqrt(max(q_ii[t], 1e-12) * max(q_jj[t], 1e-12))
        rho[t] = np.clip(q_ij[t] / denom, -1.0, 1.0)

    return rho


def compute_dcc(
    data: pd.DataFrame,
    p: int = 1,
    q: int = 1,
    decay: float = 0.94,
) -> Dict[str, Union[Dict[str, pd.Series], pd.DataFrame]]:
    """Estimate Dynamic Conditional Correlations for all pairs.

    Procedure:
    1. Fit univariate GARCH(p, q) to each column to obtain standardized
       residuals and conditional volatilities.
    2. Estimate time-varying correlations on the standardized residuals
       using an EWMA smoother (practical DCC proxy).

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame of return series.  Each column is an asset.
    p : int, default 1
        GARCH p order.
    q : int, default 1
        GARCH q order.
    decay : float, default 0.94
        EWMA decay factor for the correlation dynamics.

    Returns
    -------
    dict
        - ``conditional_correlations`` : dict of str -> pd.Series
            Keys are ``"Asset1_Asset2"`` pair labels; values are the
            time-varying correlation series.
        - ``conditional_vols`` : pd.DataFrame
            Conditional volatility for each asset (columns aligned with
            input).
    """
    data = data.dropna()
    columns = data.columns.tolist()

    if len(columns) < 2:
        raise ValueError("Need at least 2 series for DCC estimation.")

    # ── Stage 1: Univariate GARCH ────────────────────────────────────────
    std_resids: Dict[str, pd.Series] = {}
    cond_vols: Dict[str, pd.Series] = {}

    for col in columns:
        try:
            sr, cv = fit_univariate_garch(data[col], p=p, q=q)
            std_resids[col] = sr
            cond_vols[col] = cv
        except Exception as exc:
            logger.warning("GARCH fit failed for %s: %s", col, exc)
            # Fallback: rolling vol estimator
            rolling_vol = data[col].rolling(window=21).std()
            cond_vols[col] = rolling_vol
            std_resids[col] = data[col] / rolling_vol.replace(0, np.nan)

    cond_vol_df = pd.DataFrame(cond_vols)

    # ── Stage 2: EWMA Correlation on Standardized Residuals ──────────────
    # Align all standardized residuals to a common index
    resid_df = pd.DataFrame(std_resids).dropna()
    common_index = resid_df.index

    conditional_correlations: Dict[str, pd.Series] = {}

    for col_i, col_j in combinations(columns, 2):
        if col_i not in resid_df.columns or col_j not in resid_df.columns:
            continue

        ri = resid_df[col_i].values
        rj = resid_df[col_j].values

        rho = _ewma_correlation(ri, rj, decay=decay)
        pair_label = f"{col_i}_{col_j}"
        conditional_correlations[pair_label] = pd.Series(
            rho, index=common_index, name=pair_label
        )

    n_pairs = len(conditional_correlations)
    logger.info(
        "DCC estimated for %d pairs across %d assets (%d obs).",
        n_pairs,
        len(columns),
        len(common_index),
    )

    return {
        "conditional_correlations": conditional_correlations,
        "conditional_vols": cond_vol_df,
    }
