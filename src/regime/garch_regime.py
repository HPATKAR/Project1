"""
GARCH volatility regime modelling for JGB yields.

Fits GARCH and EGARCH models to yield-change series to extract
conditional volatility, then applies structural break detection to the
volatility series to identify volatility regime shifts.

    * **GARCH(1,1)** with Student-t innovations captures fat-tailed
      yield dynamics.
    * **EGARCH** captures asymmetric volatility responses (leverage
      effects) that may arise when yields break through BoJ bands.

Depends on ``arch >= 5.0`` and ``ruptures >= 1.1``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from arch import arch_model
import ruptures

logger = logging.getLogger(__name__)


def fit_garch(
    series: pd.Series,
    p: int = 1,
    q: int = 1,
    vol: str = "Garch",
    dist: str = "StudentsT",
    mean: str = "Constant",
    rescale: bool = True,
) -> Dict[str, Any]:
    """Fit a GARCH-family model to a yield-change series.

    Parameters
    ----------
    series : pd.Series
        Stationary yield-change series (basis-point changes recommended).
    p : int, default 1
        Lag order of the symmetric innovation (ARCH term).
    q : int, default 1
        Lag order of the lagged variance (GARCH term).
    vol : str, default "Garch"
        Volatility process: ``"Garch"``, ``"EGARCH"``, ``"TARCH"``, etc.
    dist : str, default "StudentsT"
        Error distribution: ``"Normal"``, ``"StudentsT"``,
        ``"SkewStudent"``, ``"GED"``.
    mean : str, default "Constant"
        Mean model: ``"Constant"``, ``"Zero"``, ``"AR"``, etc.
    rescale : bool, default True
        Whether to let ``arch`` rescale the data for numerical stability.

    Returns
    -------
    dict
        ``conditional_volatility`` : pd.Series
            Estimated conditional standard deviation, same index as
            input.
        ``standardized_residuals`` : pd.Series
            Residuals divided by conditional volatility.
        ``parameters`` : pd.Series
            Estimated model parameters.
        ``model_result`` : ARCHModelResult
            Full result object for further diagnostics.

    Raises
    ------
    ValueError
        If the series contains NaN values.
    """
    if series.isna().any():
        raise ValueError("Input series contains NaN values.")

    logger.info(
        "Fitting %s(%d,%d) with dist='%s' on %d observations.",
        vol,
        p,
        q,
        dist,
        len(series),
    )

    am = arch_model(
        series,
        mean=mean,
        vol=vol,
        p=p,
        q=q,
        dist=dist,
        rescale=rescale,
    )
    result = am.fit(disp="off")

    cond_vol = result.conditional_volatility
    cond_vol.name = "conditional_volatility"

    std_resid = result.std_resid
    if isinstance(std_resid, np.ndarray):
        std_resid = pd.Series(
            std_resid, index=series.index, name="standardized_residuals"
        )
    else:
        std_resid.name = "standardized_residuals"

    logger.info("GARCH fit complete.  Parameters:\n%s", result.params)

    return {
        "conditional_volatility": cond_vol,
        "standardized_residuals": std_resid,
        "parameters": result.params,
        "model_result": result,
    }


def fit_egarch(
    series: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "StudentsT",
    mean: str = "Constant",
) -> Dict[str, Any]:
    """Fit an EGARCH model for asymmetric volatility response.

    EGARCH (Nelson, 1991) models the logarithm of conditional variance,
    allowing negative shocks to have a different impact than positive
    shocks -- relevant when JGB yields breach BoJ tolerance bands and
    trigger asymmetric market reactions.

    Parameters
    ----------
    series : pd.Series
        Stationary yield-change series.
    p : int, default 1
        ARCH lag order.
    q : int, default 1
        GARCH lag order.
    dist : str, default "StudentsT"
        Innovation distribution.
    mean : str, default "Constant"
        Mean model specification.

    Returns
    -------
    dict
        Same structure as :func:`fit_garch`.
    """
    return fit_garch(
        series=series,
        p=p,
        q=q,
        vol="EGARCH",
        dist=dist,
        mean=mean,
    )


def volatility_regime_breaks(
    conditional_vol: pd.Series,
    n_bkps: int = 3,
    model: str = "rbf",
    min_size: int = 60,
) -> List[pd.Timestamp]:
    """Detect structural breaks in the conditional volatility series.

    Applies binary segmentation (``ruptures``) to the GARCH conditional
    volatility to identify discrete volatility regime transitions.

    Parameters
    ----------
    conditional_vol : pd.Series
        Conditional volatility series (from :func:`fit_garch` or
        :func:`fit_egarch`).
    n_bkps : int, default 3
        Number of breakpoints to detect.
    model : str, default "rbf"
        Cost model for ``ruptures.Binseg``.
    min_size : int, default 60
        Minimum segment length.

    Returns
    -------
    list of pd.Timestamp
        Dates of detected volatility regime breaks.
    """
    signal: np.ndarray = conditional_vol.dropna().values

    algo = ruptures.Binseg(model=model, min_size=min_size).fit(signal)
    breakpoint_indices: List[int] = algo.predict(n_bkps=n_bkps)

    # Remove terminal index appended by ruptures
    breakpoint_indices = [bp for bp in breakpoint_indices if bp < len(signal)]

    dates: List[pd.Timestamp] = [
        conditional_vol.dropna().index[bp] for bp in breakpoint_indices
    ]

    logger.info(
        "Volatility regime breaks (%d): %s",
        len(dates),
        [str(d.date()) for d in dates],
    )
    return dates
