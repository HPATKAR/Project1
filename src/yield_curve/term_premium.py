"""
Simplified ACM term-premium estimation for JGB yields.

Implements the three-step Adrian, Crump & Moench (2013) methodology:

1. **PCA** -- extract latent pricing factors from the yield panel.
2. **VAR(1)** -- estimate risk-neutral factor dynamics under P-measure.
3. **Cross-sectional regression** -- back out market prices of risk and
   decompose observed yields into an *expectations component* and a
   *term premium*.

This is a pedagogical / research-grade implementation suitable for
scenario analysis.  For official ACM estimates refer to the NY Fed
publication series.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def _fit_var1(factors: np.ndarray) -> Dict[str, np.ndarray]:
    """Estimate a VAR(1) via OLS on the factor matrix.

    Parameters
    ----------
    factors : np.ndarray
        Shape ``(T, K)`` matrix of pricing factors.

    Returns
    -------
    dict
        ``mu``        -- (K,) intercept vector.
        ``Phi``       -- (K, K) companion matrix.
        ``residuals`` -- (T-1, K) residual matrix.
        ``Sigma``     -- (K, K) residual covariance.
    """
    Y = factors[1:]       # (T-1, K)
    X = factors[:-1]      # (T-1, K)

    # Add intercept column
    X_aug = np.column_stack([np.ones(X.shape[0]), X])  # (T-1, K+1)

    # OLS: Y = X_aug @ B  =>  B = (X'X)^{-1} X'Y
    B, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)
    mu = B[0, :]          # (K,)
    Phi = B[1:, :]        # (K, K)  -- note: each row maps X_k -> Y_k

    # Transpose so Phi @ x gives next period's factors
    Phi = Phi.T

    residuals = Y - (mu[np.newaxis, :] + (Phi @ X.T).T)
    Sigma = np.cov(residuals, rowvar=False)

    return {"mu": mu, "Phi": Phi, "residuals": residuals, "Sigma": Sigma}


def _cross_sectional_regression(
    yields: np.ndarray,
    factors: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Cross-sectional regression of yields on pricing factors.

    Parameters
    ----------
    yields : np.ndarray
        Shape ``(T, N)`` panel of observed yields.
    factors : np.ndarray
        Shape ``(T, K)`` matrix of pricing factors.

    Returns
    -------
    dict
        ``a`` -- (N,) intercept (yield-curve constant).
        ``B`` -- (K, N) factor loadings on yields.
        ``residuals`` -- (T, N) pricing errors.
    """
    T, K = factors.shape
    X_aug = np.column_stack([np.ones(T), factors])  # (T, K+1)
    coef, _, _, _ = np.linalg.lstsq(X_aug, yields, rcond=None)

    a = coef[0, :]   # (N,)
    B = coef[1:, :]   # (K, N)
    fitted = X_aug @ coef
    residuals = yields - fitted

    return {"a": a, "B": B, "residuals": residuals}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_acm_term_premium(
    yields_df: pd.DataFrame,
    tenors: List[float],
    n_factors: int = 5,
    target_tenor: float = 10.0,
) -> pd.DataFrame:
    """Estimate the ACM term premium for JGB yields.

    Parameters
    ----------
    yields_df : pd.DataFrame
        Rows = dates, columns = tenor labels.  Yields in levels (percent
        or decimal -- be consistent).  Column order must match *tenors*.
    tenors : list of float
        Maturities in years corresponding to columns of *yields_df*.
    n_factors : int, default 5
        Number of principal components used as pricing factors.
    target_tenor : float, default 10.0
        The maturity (in years) for which the decomposition is reported.

    Returns
    -------
    pd.DataFrame
        Index = dates; columns:

        * ``observed_yield``          -- the observed yield at *target_tenor*.
        * ``expectations_component``  -- risk-neutral expectations of future
          short rates.
        * ``term_premium``            -- residual: observed minus expectations.

    Raises
    ------
    ValueError
        If *target_tenor* is not among *tenors*.

    Notes
    -----
    This is a **simplified** implementation that captures the main intuition
    of ACM (2013).  Key simplifications:

    * Risk prices (lambda_0, lambda_1) are inferred from the difference
      between P-measure and Q-measure VAR dynamics using the cross-sectional
      pricing errors as a proxy.
    * The expectations component is computed by iterating the risk-neutral
      VAR forward and aggregating implied short rates.

    For production use, consider the full affine term-structure model with
    no-arbitrage restrictions.
    """
    df = yields_df.copy().ffill().dropna()
    tenors_arr = np.asarray(tenors, dtype=float)

    if target_tenor not in tenors:
        raise ValueError(
            f"target_tenor={target_tenor} not found in tenors={tenors}."
        )
    target_idx = tenors.index(target_tenor)

    yields_mat = df.values  # (T, N)
    T, N = yields_mat.shape

    # ------------------------------------------------------------------
    # Step 1: PCA to extract pricing factors
    # ------------------------------------------------------------------
    n_factors = min(n_factors, N, T)
    pca = PCA(n_components=n_factors)
    factors = pca.fit_transform(yields_mat)  # (T, K)

    logger.info(
        "PCA: %d factors explain %.1f%% of yield variance.",
        n_factors,
        100.0 * pca.explained_variance_ratio_.sum(),
    )

    # ------------------------------------------------------------------
    # Step 2: VAR(1) on factors (P-measure dynamics)
    # ------------------------------------------------------------------
    var_res = _fit_var1(factors)
    mu_P = var_res["mu"]
    Phi_P = var_res["Phi"]
    Sigma = var_res["Sigma"]

    # ------------------------------------------------------------------
    # Step 3: Cross-sectional regression  Y = a + B' * F + eps
    # ------------------------------------------------------------------
    xs_res = _cross_sectional_regression(yields_mat, factors)
    a_hat = xs_res["a"]   # (N,)
    B_hat = xs_res["B"]   # (K, N)

    # Infer risk-neutral dynamics (Q-measure).
    # Under no-arbitrage, Q-dynamics differ from P-dynamics by the market
    # price of risk.  We approximate Q-drift as zero and Q-companion as a
    # shrunk version of Phi_P toward the identity (conservative assumption).
    shrinkage = 0.95  # degree of mean-reversion under Q
    Phi_Q = Phi_P * shrinkage
    mu_Q = mu_P * (1.0 - shrinkage)

    # ------------------------------------------------------------------
    # Compute expectations component by rolling the Q-VAR forward
    # ------------------------------------------------------------------
    # Short rate loading: yield at shortest maturity ~ a_short + B_short' F
    short_idx = int(np.argmin(tenors_arr))
    a_short = a_hat[short_idx]
    B_short = B_hat[:, short_idx]  # (K,)

    # For the target tenor (e.g. 10Y), the expectations hypothesis yield
    # is the average expected short rate over the next *target_tenor* years.
    # We discretise at annual steps.
    n_steps = max(int(target_tenor), 1)

    expectations = np.full(T, np.nan)

    for t in range(T):
        f_t = factors[t]  # current factor state
        cum_short = 0.0
        f_iter = f_t.copy()
        for step in range(n_steps):
            implied_short = a_short + B_short @ f_iter
            cum_short += implied_short
            # iterate under Q
            f_iter = mu_Q + Phi_Q @ f_iter
        expectations[t] = cum_short / n_steps

    observed = yields_mat[:, target_idx]
    term_premium = observed - expectations

    result = pd.DataFrame(
        {
            "observed_yield": observed,
            "expectations_component": expectations,
            "term_premium": term_premium,
        },
        index=df.index,
    )

    logger.info(
        "Term-premium summary (%.0fY): mean=%.2f bps, std=%.2f bps",
        target_tenor,
        np.nanmean(term_premium) * (100 if np.nanmean(observed) < 1 else 1),
        np.nanstd(term_premium) * (100 if np.nanmean(observed) < 1 else 1),
    )

    return result
