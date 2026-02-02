"""Transfer entropy computation for directional information flow analysis.

Transfer entropy measures the directed information flow from a source
process X to a target process Y, quantifying how much uncertainty about
the future of Y is reduced by knowing the past of X *beyond* what is
already explained by the past of Y itself.

    TE(X -> Y) = H(Y_t | Y_{t-1:t-k}) - H(Y_t | Y_{t-1:t-k}, X_{t-1:t-k})

Implemented using histogram-based probability estimation (no external TE
package required) for robustness and transparency.

References
----------
Schreiber, T. (2000), "Measuring Information Transfer",
Physical Review Letters, 85(2), 461-464.
"""

import logging
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

logger = logging.getLogger(__name__)


def discretize_series(
    series: pd.Series,
    n_bins: int = 3,
) -> pd.Series:
    """Discretize a continuous return series into categorical bins.

    Parameters
    ----------
    series : pd.Series
        Continuous series (e.g. daily returns).
    n_bins : int, default 3
        Number of bins.  With n_bins=3, the bins represent
        approximately ``down / flat / up`` terciles based on
        equal-frequency quantile binning.

    Returns
    -------
    pd.Series
        Integer-coded series with values in [0, n_bins - 1].
    """
    series = series.dropna()

    if len(series) < n_bins:
        raise ValueError(
            f"Series too short ({len(series)}) for {n_bins} bins."
        )

    # Use quantile-based binning for equal-frequency bins
    try:
        binned = pd.qcut(series, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        # Fallback to equal-width bins if quantile binning fails
        # (e.g. too many identical values)
        binned = pd.cut(series, bins=n_bins, labels=False, duplicates="drop")

    return binned.astype(int)


def _joint_histogram(
    *arrays: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Compute a normalized joint histogram (empirical joint PMF).

    Parameters
    ----------
    *arrays : np.ndarray
        One or more 1-D integer arrays (discretized), all of equal length.
    n_bins : int
        Number of discrete levels per variable.

    Returns
    -------
    np.ndarray
        Normalized histogram (sums to 1) with shape (n_bins,) * len(arrays).
    """
    stacked = np.column_stack(arrays)
    n = len(stacked)

    # Build multi-dimensional histogram
    edges = [np.arange(-0.5, n_bins + 0.5, 1.0)] * stacked.shape[1]
    hist, _ = np.histogramdd(stacked, bins=edges)

    # Normalize
    hist = hist / max(n, 1)

    return hist


def compute_transfer_entropy(
    source: pd.Series,
    target: pd.Series,
    lag: int = 1,
    n_bins: int = 3,
) -> float:
    """Compute transfer entropy from source to target.

    TE(source -> target) = H(target_t, target_{t-lag}) - H(target_{t-lag})
                         - H(target_t, target_{t-lag}, source_{t-lag})
                         + H(target_{t-lag}, source_{t-lag})

    This is equivalent to the conditional mutual information:
        I(target_t ; source_{t-lag} | target_{t-lag})

    Parameters
    ----------
    source : pd.Series
        Source (cause) time series.
    target : pd.Series
        Target (effect) time series.
    lag : int, default 1
        Number of time steps for the lagged conditioning.
    n_bins : int, default 3
        Number of discretization bins.

    Returns
    -------
    float
        Transfer entropy in nats (natural logarithm base).
        Non-negative by construction (within numerical precision).
    """
    # Align and discretize
    combined = pd.DataFrame(
        {"source": source, "target": target}
    ).dropna()

    if len(combined) < lag + 10:
        logger.warning(
            "Insufficient data for TE computation: %d obs.", len(combined)
        )
        return 0.0

    src_disc = discretize_series(combined["source"], n_bins=n_bins).values
    tgt_disc = discretize_series(combined["target"], n_bins=n_bins).values

    n = len(src_disc)

    # Build lagged arrays
    # target_t      = tgt_disc[lag:]
    # target_past   = tgt_disc[:n-lag]  (i.e. tgt_{t-lag})
    # source_past   = src_disc[:n-lag]  (i.e. src_{t-lag})
    target_t = tgt_disc[lag:]
    target_past = tgt_disc[:n - lag]
    source_past = src_disc[:n - lag]

    # TE = H(target_t, target_past) - H(target_past)
    #    - H(target_t, target_past, source_past) + H(target_past, source_past)
    #
    # Using the relationship: TE = H(A,B) - H(B) - H(A,B,C) + H(B,C)
    # where A = target_t, B = target_past, C = source_past

    # Compute joint distributions
    p_ab = _joint_histogram(target_t, target_past, n_bins=n_bins)
    p_b = _joint_histogram(target_past, n_bins=n_bins)
    p_abc = _joint_histogram(target_t, target_past, source_past, n_bins=n_bins)
    p_bc = _joint_histogram(target_past, source_past, n_bins=n_bins)

    # Compute entropies (flatten for scipy.stats.entropy)
    h_ab = scipy_entropy(p_ab.ravel() + 1e-12)
    h_b = scipy_entropy(p_b.ravel() + 1e-12)
    h_abc = scipy_entropy(p_abc.ravel() + 1e-12)
    h_bc = scipy_entropy(p_bc.ravel() + 1e-12)

    te = h_ab - h_b - h_abc + h_bc

    # TE should be non-negative; clip numerical noise
    te = max(te, 0.0)

    return te


def pairwise_transfer_entropy(
    data: pd.DataFrame,
    lag: int = 1,
    n_bins: int = 3,
) -> pd.DataFrame:
    """Compute transfer entropy for all directed pairs in the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame of time series (e.g. returns).  Each column is a variable.
    lag : int, default 1
        Lag for the TE computation.
    n_bins : int, default 3
        Number of discretization bins.

    Returns
    -------
    pd.DataFrame
        Columns: source, target, te_value.
        One row per directed pair, sorted by te_value descending.
    """
    data = data.dropna()
    columns = data.columns.tolist()

    if len(columns) < 2:
        raise ValueError("Need at least 2 columns for pairwise TE.")

    results = []

    for src, tgt in permutations(columns, 2):
        te_val = compute_transfer_entropy(
            source=data[src],
            target=data[tgt],
            lag=lag,
            n_bins=n_bins,
        )
        results.append(
            {
                "source": src,
                "target": tgt,
                "te_value": round(te_val, 6),
            }
        )

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("te_value", ascending=False).reset_index(
        drop=True
    )

    logger.info(
        "Pairwise TE computed for %d directed pairs (lag=%d, bins=%d).",
        len(result_df),
        lag,
        n_bins,
    )

    return result_df


def information_flow_network(
    te_matrix: pd.DataFrame,
    threshold: float = 0.01,
) -> List[Dict[str, object]]:
    """Filter significant information flows for network visualization.

    Parameters
    ----------
    te_matrix : pd.DataFrame
        Output of ``pairwise_transfer_entropy``, with columns
        ``source``, ``target``, ``te_value``.
    threshold : float, default 0.01
        Minimum TE value to include an edge.

    Returns
    -------
    list of dict
        Adjacency list where each element is a dict with keys:
        ``source``, ``target``, ``weight`` (the TE value).
        Suitable for direct use with ``networkx.DiGraph.add_edges_from``
        or similar visualization libraries.
    """
    significant = te_matrix[te_matrix["te_value"] >= threshold].copy()

    edges = []
    for _, row in significant.iterrows():
        edges.append(
            {
                "source": row["source"],
                "target": row["target"],
                "weight": row["te_value"],
            }
        )

    logger.info(
        "Information flow network: %d/%d edges above threshold=%.4f.",
        len(edges),
        len(te_matrix),
        threshold,
    )

    return edges
