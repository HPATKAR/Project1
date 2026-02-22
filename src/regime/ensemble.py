"""
Ensemble regime probability for the JGB repricing framework.

Combines signals from all five regime-detection methods into a single
composite probability:

    1. Markov-switching smoothed probabilities
    2. HMM hidden-state labels
    3. Entropy-based regime signal
    4. GARCH conditional-volatility regime breaks

Each signal is first normalised to the [0, 1] interval, then combined
via a weighted average.  The resulting series can be interpreted as:

    * Values near **0** -- BoJ-suppressed regime (low volatility, stable
      yields within YCC band).
    * Values near **1** -- Market-driven repricing regime (elevated
      volatility, yields moving beyond policy anchors).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _normalize_to_unit(s: pd.Series) -> pd.Series:
    """Min-max normalise a series to [0, 1], handling constant series."""
    s_min = s.min()
    s_max = s.max()
    if s_max == s_min:
        return pd.Series(0.5, index=s.index, name=s.name)
    return (s - s_min) / (s_max - s_min)


def _breakpoints_to_regime_series(
    breakpoints: List[pd.Timestamp],
    index: pd.DatetimeIndex,
) -> pd.Series:
    """Convert breakpoint dates to a binary regime series.

    Segments between consecutive breakpoints alternate between 0 and 1,
    starting at 0 (suppressed).
    """
    regime = pd.Series(0.0, index=index, name="garch_vol_regime")
    current_regime = 0.0
    for bp in sorted(breakpoints):
        current_regime = 1.0 - current_regime
        regime.loc[regime.index >= bp] = current_regime
    return regime


def ensemble_regime_probability(
    markov_prob: pd.Series,
    hmm_states: pd.Series,
    entropy_signal: pd.Series,
    garch_vol_regime: Union[pd.Series, List[pd.Timestamp]],
    weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """Combine multiple regime-detection signals into one probability.

    Parameters
    ----------
    markov_prob : pd.Series
        Smoothed probability of the market-driven regime from the
        Markov-switching model (values in [0, 1]).  Typically column
        ``regime_1`` from :func:`~src.regime.markov_switching.fit_markov_regime`.
    hmm_states : pd.Series
        HMM state labels (integer-valued).  The state with the higher
        mean is assumed to be the market-driven state and is normalised
        accordingly.
    entropy_signal : pd.Series
        Binary or continuous entropy regime signal from
        :func:`~src.regime.entropy_regime.entropy_regime_signal`.
    garch_vol_regime : pd.Series or list of pd.Timestamp
        Either a pre-computed binary regime series **or** a list of
        volatility breakpoint dates (from
        :func:`~src.regime.garch_regime.volatility_regime_breaks`).  If
        a list is passed it is converted to a binary series internally.
    weights : dict or None
        Mapping of signal name to weight.  Keys must be a subset of
        ``{"markov", "hmm", "entropy", "garch"}``.  If ``None``,
        equal weights (0.25 each) are used.

    Returns
    -------
    pd.Series
        Ensemble regime probability (0 = suppressed, 1 = market-driven),
        indexed by the intersection of all input indices.
    """
    # Default equal weights
    if weights is None:
        weights = {
            "markov": 0.25,
            "hmm": 0.25,
            "entropy": 0.25,
            "garch": 0.25,
        }

    # Validate weights sum to ~1
    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0):
        logger.warning(
            "Weights sum to %.4f, not 1.0.  Normalising.", total_weight
        )
        weights = {k: v / total_weight for k, v in weights.items()}

    # --- Normalise each signal to [0, 1] ---
    norm_markov = _normalize_to_unit(markov_prob).rename("markov")
    norm_hmm = _normalize_to_unit(hmm_states.astype(float)).rename("hmm")
    norm_entropy = _normalize_to_unit(entropy_signal).rename("entropy")

    if isinstance(garch_vol_regime, list):
        # Convert breakpoint list to binary regime series using the
        # broadest available index
        ref_index = markov_prob.index.union(hmm_states.index)
        garch_series = _breakpoints_to_regime_series(garch_vol_regime, ref_index)
    else:
        garch_series = garch_vol_regime.copy()
    norm_garch = _normalize_to_unit(garch_series).rename("garch")

    # --- Align all signals on a common index ---
    combined = pd.DataFrame(
        {
            "markov": norm_markov,
            "hmm": norm_hmm,
            "entropy": norm_entropy,
            "garch": norm_garch,
        }
    )

    # Weighted average (NaN-aware: only use available signals per row)
    weight_series = pd.Series(weights)
    weighted_sum = combined.mul(weight_series, axis=1).sum(axis=1, min_count=1)
    weight_total = combined.notna().astype(float).mul(weight_series, axis=1).sum(axis=1)
    ensemble = (weighted_sum / weight_total).clip(0.0, 1.0)

    ensemble.name = "ensemble_regime_probability"

    logger.info(
        "Ensemble regime probability computed.  Mean=%.4f, "
        "regime-1 fraction=%.2f%%.",
        ensemble.mean(),
        (ensemble > 0.5).mean() * 100,
    )
    return ensemble


# ---------------------------------------------------------------------------
# Validation against known BOJ events
# ---------------------------------------------------------------------------

def validate_ensemble_vs_boj(
    ensemble: pd.Series,
    boj_events: Dict[str, str],
    window_days: int = 10,
    spike_threshold: float = 0.6,
) -> Dict[str, object]:
    """Check whether the ensemble probability spiked around known BOJ events.

    For each BOJ event date that falls within the ensemble's date range,
    we check whether the ensemble probability exceeded *spike_threshold*
    within ±*window_days* trading days of the event.

    Parameters
    ----------
    ensemble : pd.Series
        Ensemble regime probability (0–1), datetime-indexed.
    boj_events : dict
        Mapping ``"YYYY-MM-DD"`` → description (e.g. from ``config.BOJ_EVENTS``).
    window_days : int
        Number of trading days on each side of the event to search.
    spike_threshold : float
        Minimum ensemble probability to count as a "detection".

    Returns
    -------
    dict
        ``detection_rate``  : float – fraction of in-sample events detected.
        ``avg_lead_lag``    : float – mean signed offset in trading days
                              (negative = ensemble spiked *before* the event).
        ``n_detected``      : int
        ``n_in_sample``     : int
        ``details``         : list of dicts with per-event results.
    """
    ens = ensemble.dropna()
    if len(ens) == 0:
        return {
            "detection_rate": 0.0,
            "avg_lead_lag": 0.0,
            "n_detected": 0,
            "n_in_sample": 0,
            "details": [],
        }

    details: List[Dict[str, object]] = []
    n_detected = 0
    lead_lags: List[int] = []

    for date_str, description in boj_events.items():
        event_date = pd.Timestamp(date_str)

        # Skip events outside ensemble range
        if event_date < ens.index.min() or event_date > ens.index.max():
            continue

        # Find nearest index position for the event date
        idx_loc = ens.index.searchsorted(event_date)
        lo = max(idx_loc - window_days, 0)
        hi = min(idx_loc + window_days + 1, len(ens))
        window = ens.iloc[lo:hi]

        peak_val = float(window.max())
        detected = peak_val >= spike_threshold

        if detected:
            n_detected += 1
            peak_date = window.idxmax()
            # Signed offset: negative means ensemble spiked before event
            offset = int(ens.index.searchsorted(peak_date) - idx_loc)
            lead_lags.append(offset)
        else:
            offset = None

        details.append({
            "date": date_str,
            "event": description,
            "detected": detected,
            "peak_prob": peak_val,
            "lead_lag_days": offset,
        })

    n_in_sample = len(details)
    detection_rate = n_detected / n_in_sample if n_in_sample > 0 else 0.0
    avg_lead_lag = float(np.mean(lead_lags)) if lead_lags else 0.0

    logger.info(
        "Ensemble BOJ validation: %d/%d events detected (%.0f%%), "
        "avg lead/lag=%.1f days.",
        n_detected,
        n_in_sample,
        detection_rate * 100,
        avg_lead_lag,
    )

    return {
        "detection_rate": detection_rate,
        "avg_lead_lag": avg_lead_lag,
        "n_detected": n_detected,
        "n_in_sample": n_in_sample,
        "details": details,
    }
