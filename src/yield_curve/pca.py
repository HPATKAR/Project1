"""
PCA decomposition of the JGB yield curve.

Decomposes daily yield-change cross-sections into orthogonal factors that
correspond to *level*, *slope*, and *curvature* movements.  Supports both
full-sample and rolling-window estimation so that time-varying factor
structures can be monitored.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Full-sample PCA
# ---------------------------------------------------------------------------

def fit_yield_pca(
    yield_changes: pd.DataFrame,
    n_components: int = 3,
) -> Dict[str, Any]:
    """Fit PCA on a panel of daily yield changes.

    Parameters
    ----------
    yield_changes : pd.DataFrame
        Rows = dates, columns = tenor labels (e.g. ``['2Y', '5Y', '10Y', …]``).
        Values are daily changes in yields (bps or decimals -- just be
        consistent).  Missing values are forward-filled then dropped.
    n_components : int, default 3
        Number of principal components to retain.

    Returns
    -------
    dict
        ``scores``              – pd.DataFrame (dates x components) of factor scores.
        ``loadings``            – pd.DataFrame (components x tenors) of factor loadings.
        ``explained_variance``  – np.ndarray of variance explained per component.
        ``explained_variance_ratio`` – np.ndarray of fraction of total variance.
        ``pca_model``           – fitted ``sklearn.decomposition.PCA`` object.
    """
    df = yield_changes.copy().ffill().dropna()
    if df.empty:
        raise ValueError("yield_changes is empty after cleaning.")

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(df.values)

    scores_df = pd.DataFrame(
        scores,
        index=df.index,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    loadings_df = pd.DataFrame(
        pca.components_,
        index=[f"PC{i+1}" for i in range(n_components)],
        columns=df.columns,
    )

    return {
        "scores": scores_df,
        "loadings": loadings_df,
        "explained_variance": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "pca_model": pca,
    }


# ---------------------------------------------------------------------------
# Rolling-window PCA
# ---------------------------------------------------------------------------

def rolling_pca(
    yield_changes: pd.DataFrame,
    window: int = 252,
    n_components: int = 3,
) -> Dict[str, pd.DataFrame]:
    """Rolling-window PCA to track time-varying loadings.

    Parameters
    ----------
    yield_changes : pd.DataFrame
        Same format as :func:`fit_yield_pca`.
    window : int, default 252
        Number of trading days in the rolling window.
    n_components : int, default 3
        Number of principal components to retain.

    Returns
    -------
    dict
        ``rolling_var_explained`` – pd.DataFrame  (dates x components) of
            variance-explained ratios at each window end-date.
        ``rolling_loadings``     – dict mapping ``'PC1'``, ``'PC2'``, … to
            pd.DataFrame (dates x tenors) of loadings over time.
    """
    df = yield_changes.copy().ffill().dropna()
    n_obs, n_tenors = df.shape
    if n_obs < window:
        raise ValueError(
            f"Need at least {window} observations; got {n_obs}."
        )

    dates: List[pd.Timestamp] = []
    var_explained_rows: List[np.ndarray] = []
    loadings_by_pc: Dict[str, List[np.ndarray]] = {
        f"PC{i+1}": [] for i in range(n_components)
    }

    pca = PCA(n_components=n_components)

    for end in range(window, n_obs + 1):
        start = end - window
        chunk = df.iloc[start:end].values
        pca.fit(chunk)

        dates.append(df.index[end - 1])
        var_explained_rows.append(pca.explained_variance_ratio_.copy())
        for i in range(n_components):
            loadings_by_pc[f"PC{i+1}"].append(pca.components_[i].copy())

    var_explained_df = pd.DataFrame(
        var_explained_rows,
        index=dates,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    rolling_loadings: Dict[str, pd.DataFrame] = {}
    for pc_name, rows in loadings_by_pc.items():
        rolling_loadings[pc_name] = pd.DataFrame(
            rows, index=dates, columns=df.columns,
        )

    return {
        "rolling_var_explained": var_explained_df,
        "rolling_loadings": rolling_loadings,
    }


# ---------------------------------------------------------------------------
# Interpretation helper
# ---------------------------------------------------------------------------

def interpret_pca(
    loadings: pd.DataFrame,
    tenors: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Label principal components as *level*, *slope*, or *curvature*.

    Heuristic rules (standard in fixed-income literature):

    * **Level**     -- loadings are roughly uniform across tenors.
    * **Slope**     -- loadings change sign once (short end vs. long end).
    * **Curvature** -- loadings change sign twice (hump-shaped or U-shaped).

    Parameters
    ----------
    loadings : pd.DataFrame
        Shape ``(n_components, n_tenors)``.  As returned by
        :func:`fit_yield_pca` under key ``'loadings'``.
    tenors : list of str, optional
        Tenor labels.  If *None*, ``loadings.columns`` are used directly.

    Returns
    -------
    pd.DataFrame
        Copy of *loadings* with an added ``'interpretation'`` column.
    """
    if tenors is not None:
        loadings = loadings.copy()
        loadings.columns = tenors

    result = loadings.copy()
    labels: List[str] = []

    for idx in loadings.index:
        row = loadings.loc[idx].values.astype(float)
        sign_changes = int(np.sum(np.diff(np.sign(row)) != 0))

        if sign_changes == 0:
            # All same sign and relatively flat -> level
            coeff_of_var = np.std(np.abs(row)) / (np.mean(np.abs(row)) + 1e-12)
            if coeff_of_var < 0.5:
                labels.append("level")
            else:
                labels.append("level (uneven)")
        elif sign_changes == 1:
            labels.append("slope")
        elif sign_changes >= 2:
            labels.append("curvature")
        else:
            labels.append("unknown")

    result["interpretation"] = labels
    return result


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------

def validate_pca_factors(
    pca_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate PCA factors against known yield curve dynamics.

    Compares the fitted PCA decomposition to the classical fixed-income
    factor structure (Litterman-Scheinkman 1991):

    * PC1 should explain 60-90% of variance (level factor).
    * PC2 should explain 10-20% (slope factor).
    * PC3 should explain 3-10% (curvature factor).
    * Cumulative PC1-3 should exceed 95%.

    Parameters
    ----------
    pca_result : dict
        Output from :func:`fit_yield_pca`.

    Returns
    -------
    dict
        ``factor_checks``       – list of (name, passed, detail) tuples.
        ``explained_variance``  – dict mapping factor name to ratio.
        ``cumulative_variance`` – float, total explained by all retained PCs.
        ``interpretation``      – pd.DataFrame from :func:`interpret_pca`.
        ``summary``             – str, human-readable validation summary.
    """
    evr = pca_result["explained_variance_ratio"]
    loadings = pca_result["loadings"]
    interp = interpret_pca(loadings)

    n = len(evr)
    cum_var = float(np.sum(evr))

    checks = []

    # PC1: Level factor
    if n >= 1:
        pc1_var = float(evr[0])
        pc1_label = interp.iloc[0]["interpretation"] if "interpretation" in interp.columns else "unknown"
        pc1_pass = pc1_var >= 0.50 and "level" in pc1_label.lower()
        checks.append((
            "PC1 (Level)",
            pc1_pass,
            f"{pc1_var:.1%} variance, identified as '{pc1_label}'"
        ))

    # PC2: Slope factor
    if n >= 2:
        pc2_var = float(evr[1])
        pc2_label = interp.iloc[1]["interpretation"] if "interpretation" in interp.columns else "unknown"
        pc2_pass = "slope" in pc2_label.lower()
        checks.append((
            "PC2 (Slope)",
            pc2_pass,
            f"{pc2_var:.1%} variance, identified as '{pc2_label}'"
        ))

    # PC3: Curvature factor
    if n >= 3:
        pc3_var = float(evr[2])
        pc3_label = interp.iloc[2]["interpretation"] if "interpretation" in interp.columns else "unknown"
        pc3_pass = "curvature" in pc3_label.lower()
        checks.append((
            "PC3 (Curvature)",
            pc3_pass,
            f"{pc3_var:.1%} variance, identified as '{pc3_label}'"
        ))

    # Cumulative variance check
    cum_pass = cum_var >= 0.90
    checks.append((
        "Cumulative (PC1-3)",
        cum_pass,
        f"{cum_var:.1%} total variance explained (benchmark: >95%)"
    ))

    # Build summary
    passed = sum(1 for _, p, _ in checks if p)
    total = len(checks)
    lines = [f"PCA Factor Validation: {passed}/{total} checks passed"]
    for name, ok, detail in checks:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    return {
        "factor_checks": checks,
        "explained_variance": {
            f"PC{i+1}": float(evr[i]) for i in range(n)
        },
        "cumulative_variance": cum_var,
        "interpretation": interp,
        "summary": "\n".join(lines),
    }
