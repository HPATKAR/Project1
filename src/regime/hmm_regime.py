"""
Hidden Markov Model for multivariate regime detection.

Uses a Gaussian HMM (``hmmlearn``) to jointly model multiple market
variables -- e.g. JGB 10-year yield changes, USD/JPY log-returns, and
VIX changes -- in order to infer latent market regimes.

Two states are estimated by default:

    * State 0 -- **Calm / BoJ-suppressed**: low cross-asset volatility,
      muted co-movement.
    * State 1 -- **Stress / market-driven**: elevated volatility and
      stronger cross-asset linkages indicative of repricing.

Depends on ``hmmlearn >= 0.3``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)


def fit_multivariate_hmm(
    data: pd.DataFrame,
    n_states: int = 2,
    n_iter: int = 100,
    covariance_type: str = "full",
    random_state: int = 42,
) -> Dict[str, Any]:
    """Fit a Gaussian HMM on multiple market series for regime detection.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame where each column is a stationary market variable
        (e.g. JGB 10Y changes, USD/JPY changes, VIX changes).  Rows
        are time-ordered observations.  Must not contain NaN.
    n_states : int, default 2
        Number of hidden states (regimes).
    n_iter : int, default 100
        Maximum number of EM iterations.
    covariance_type : str, default "full"
        Type of covariance matrix.  One of ``"full"``, ``"diag"``,
        ``"spherical"``, ``"tied"``.
    random_state : int, default 42
        Seed for reproducibility.

    Returns
    -------
    dict
        ``states`` : pd.Series
            Most-likely state sequence (Viterbi), indexed like *data*.
        ``state_means`` : np.ndarray, shape (n_states, n_features)
            Estimated mean vector for each state.
        ``state_covariances`` : np.ndarray
            Estimated covariance matrices for each state.
        ``transition_matrix`` : np.ndarray, shape (n_states, n_states)
            Estimated transition probability matrix.
        ``model`` : GaussianHMM
            Fitted model object.

    Raises
    ------
    ValueError
        If *data* contains NaN or has fewer rows than *n_states*.
    """
    if data.isna().any().any():
        raise ValueError(
            "Input DataFrame contains NaN values.  Drop or impute them "
            "before fitting the HMM."
        )
    if len(data) < n_states:
        raise ValueError(
            f"Need at least {n_states} observations; got {len(data)}."
        )

    logger.info(
        "Fitting GaussianHMM with n_states=%d, n_iter=%d, "
        "covariance_type='%s' on data of shape %s.",
        n_states,
        n_iter,
        covariance_type,
        data.shape,
    )

    X: np.ndarray = data.values

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X)

    hidden_states: np.ndarray = model.predict(X)

    states_series = pd.Series(
        hidden_states, index=data.index, name="hmm_state"
    )

    logger.info(
        "HMM fit complete.  Transition matrix:\n%s", model.transmat_
    )

    return {
        "states": states_series,
        "state_means": model.means_,
        "state_covariances": model.covars_,
        "transition_matrix": model.transmat_,
        "model": model,
    }


def predict_regime(
    model: GaussianHMM,
    new_data: Union[pd.DataFrame, np.ndarray],
) -> np.ndarray:
    """Predict the most likely regime for new observations.

    Parameters
    ----------
    model : GaussianHMM
        A fitted ``GaussianHMM`` instance (e.g. from
        :func:`fit_multivariate_hmm`).
    new_data : pd.DataFrame or np.ndarray
        New observations with the same feature ordering used during
        fitting.

    Returns
    -------
    np.ndarray
        Predicted hidden-state labels for each row of *new_data*.
    """
    if isinstance(new_data, pd.DataFrame):
        X = new_data.values
    else:
        X = np.asarray(new_data)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    predictions: np.ndarray = model.predict(X)
    logger.info("Predicted %d regime labels.", len(predictions))
    return predictions
