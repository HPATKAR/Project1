"""
Nelson-Siegel (-Svensson) parametric yield-curve fitting.

Fits the classic Nelson-Siegel three-factor model to JGB yield cross-sections:

    y(tau) = beta0
           + beta1 * (1 - exp(-tau/lambda)) / (tau/lambda)
           + beta2 * ((1 - exp(-tau/lambda)) / (tau/lambda) - exp(-tau/lambda))

where *beta0* captures **level**, *beta1* captures **slope**, *beta2*
captures **curvature**, and *lambda* (tau parameter) governs the decay rate.

Provides both single-date and full time-series fitting.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal NS model evaluation
# ---------------------------------------------------------------------------

def _ns_curve(
    tenors: np.ndarray,
    beta0: float,
    beta1: float,
    beta2: float,
    tau: float,
) -> np.ndarray:
    """Evaluate the Nelson-Siegel curve at the given tenors.

    Parameters
    ----------
    tenors : np.ndarray
        Maturities in years.
    beta0, beta1, beta2 : float
        Level, slope, and curvature parameters.
    tau : float
        Decay factor (must be > 0).

    Returns
    -------
    np.ndarray
        Fitted yields for each tenor.
    """
    tau = max(tau, 1e-6)  # guard against zero / negative
    x = tenors / tau
    factor1 = np.where(x < 1e-8, 1.0, (1.0 - np.exp(-x)) / x)
    factor2 = factor1 - np.exp(-x)
    return beta0 + beta1 * factor1 + beta2 * factor2


def _ns_loss(
    params: np.ndarray,
    tenors: np.ndarray,
    observed: np.ndarray,
) -> float:
    """Sum-of-squared-errors loss for NS optimisation."""
    beta0, beta1, beta2, tau = params
    fitted = _ns_curve(tenors, beta0, beta1, beta2, tau)
    return float(np.sum((observed - fitted) ** 2))


# ---------------------------------------------------------------------------
# Single cross-section fit
# ---------------------------------------------------------------------------

def fit_ns(
    yields: pd.Series,
    tenors: List[float],
    tau_bounds: Tuple[float, float] = (0.1, 30.0),
) -> Dict[str, float]:
    """Fit a Nelson-Siegel model to one cross-section of yields.

    Parameters
    ----------
    yields : pd.Series
        Observed yields indexed by tenor label (ordering must match *tenors*).
    tenors : list of float
        Maturities in years corresponding to each element of *yields*.
    tau_bounds : tuple of float, default (0.1, 30.0)
        Lower and upper bounds for the decay parameter *tau*.

    Returns
    -------
    dict
        Keys: ``beta0``, ``beta1``, ``beta2``, ``tau``, ``fitted``, ``residuals``.
        ``fitted`` and ``residuals`` are np.ndarrays aligned to *tenors*.

    Notes
    -----
    Uses ``scipy.optimize.minimize`` (L-BFGS-B) with multiple random restarts
    to avoid poor local minima.  Falls back to a simple initial guess when the
    ``nelson_siegel_svensson`` package is unavailable.
    """
    tenors_arr = np.asarray(tenors, dtype=float)
    obs = np.asarray(yields.values, dtype=float)

    # Try to use nelson_siegel_svensson for a warm-start guess
    nss_guess: Optional[np.ndarray] = None
    try:
        from nelson_siegel_svensson.calibrate import calibrate_ns_ols
        curve, _ = calibrate_ns_ols(tenors_arr, obs)
        nss_guess = np.array([curve.beta0, curve.beta1, curve.beta2, curve.tau])
    except Exception:
        # Package not installed or calibration failed -- proceed with heuristic
        pass

    # Heuristic initial guesses
    long_yield = obs[-1] if len(obs) > 0 else 0.01
    short_yield = obs[0] if len(obs) > 0 else 0.005
    init_guesses = [
        np.array([long_yield, short_yield - long_yield, 0.0, 1.5]),
        np.array([np.mean(obs), -0.01, 0.01, 2.0]),
        np.array([long_yield, -0.02, 0.02, 5.0]),
    ]
    if nss_guess is not None:
        init_guesses.insert(0, nss_guess)

    bounds = [(None, None), (None, None), (None, None), tau_bounds]
    best_loss = np.inf
    best_params = init_guesses[0]

    for x0 in init_guesses:
        try:
            res = minimize(
                _ns_loss,
                x0=x0,
                args=(tenors_arr, obs),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 2000},
            )
            if res.fun < best_loss:
                best_loss = res.fun
                best_params = res.x
        except Exception as exc:
            logger.debug("NS optimisation restart failed: %s", exc)

    beta0, beta1, beta2, tau = best_params
    fitted = _ns_curve(tenors_arr, beta0, beta1, beta2, tau)

    return {
        "beta0": float(beta0),
        "beta1": float(beta1),
        "beta2": float(beta2),
        "tau": float(tau),
        "fitted": fitted,
        "residuals": obs - fitted,
    }


# ---------------------------------------------------------------------------
# Time-series fit
# ---------------------------------------------------------------------------

def fit_ns_timeseries(
    yields_df: pd.DataFrame,
    tenors: List[float],
    tau_bounds: Tuple[float, float] = (0.1, 30.0),
) -> pd.DataFrame:
    """Fit the Nelson-Siegel model at every date in a panel of yields.

    Parameters
    ----------
    yields_df : pd.DataFrame
        Rows = dates, columns = tenor labels.  Column order must match
        *tenors*.
    tenors : list of float
        Maturities in years.
    tau_bounds : tuple of float, default (0.1, 30.0)
        Bounds for *tau* passed to :func:`fit_ns`.

    Returns
    -------
    pd.DataFrame
        Index = dates; columns = ``['beta0', 'beta1', 'beta2', 'tau']``.
        ``beta0`` is the level factor, ``beta1`` the slope factor, and
        ``beta2`` the curvature factor.
    """
    records: List[Dict[str, float]] = []
    dates = []

    for date, row in yields_df.iterrows():
        if row.isna().any():
            logger.debug("Skipping %s due to NaN yields.", date)
            continue
        try:
            params = fit_ns(row, tenors, tau_bounds=tau_bounds)
            records.append({
                "beta0": params["beta0"],
                "beta1": params["beta1"],
                "beta2": params["beta2"],
                "tau": params["tau"],
            })
            dates.append(date)
        except Exception as exc:
            logger.warning("NS fit failed on %s: %s", date, exc)

    if not records:
        raise RuntimeError("Nelson-Siegel fitting failed on all dates.")

    return pd.DataFrame(records, index=dates)
