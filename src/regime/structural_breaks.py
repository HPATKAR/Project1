"""
Structural break detection for JGB yield and spread series.

Provides change-point detection via the PELT (Pruned Exact Linear Time)
algorithm and binary segmentation, using the ``ruptures`` library.
Detected breakpoints correspond to dates where the statistical properties
of a series change significantly -- e.g. BoJ policy shifts, YCC band
adjustments, or sudden repricing episodes.

Depends on ``ruptures >= 1.1`` and ``matplotlib`` for plotting.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures

logger = logging.getLogger(__name__)


def detect_breaks_pelt(
    series: pd.Series,
    penalty: Optional[float] = None,
    min_size: int = 60,
    model: str = "rbf",
) -> List[pd.Timestamp]:
    """Detect structural breakpoints using the PELT algorithm.

    PELT (Pruned Exact Linear Time) searches for an optimal number of
    change points by minimising a penalised cost function.

    Parameters
    ----------
    series : pd.Series
        Time series with ``DatetimeIndex``.
    penalty : float or None
        Penalty value for adding a breakpoint.  If ``None`` a
        data-driven BIC-style penalty of ``log(n) * variance`` is used.
    min_size : int, default 60
        Minimum number of observations between two breakpoints (e.g.
        ~3 months of daily data).
    model : str, default "rbf"
        Cost model passed to ``ruptures.Pelt``.  Common choices are
        ``"rbf"``, ``"l2"``, ``"normal"``.

    Returns
    -------
    list of pd.Timestamp
        Dates at which structural breaks were detected (excluding the
        terminal index which ``ruptures`` always appends).
    """
    signal: np.ndarray = series.dropna().values

    if penalty is None:
        penalty = float(np.log(len(signal)) * np.var(signal))
        logger.info("Using data-driven penalty: %.6f", penalty)

    algo = ruptures.Pelt(model=model, min_size=min_size).fit(signal)
    breakpoint_indices: List[int] = algo.predict(pen=penalty)

    # ruptures appends len(signal) as the last element; remove it
    breakpoint_indices = [bp for bp in breakpoint_indices if bp < len(signal)]

    dates: List[pd.Timestamp] = [
        series.dropna().index[bp] for bp in breakpoint_indices
    ]

    logger.info(
        "PELT detected %d breakpoints: %s",
        len(dates),
        [str(d.date()) for d in dates],
    )
    return dates


def detect_breaks_binseg(
    series: pd.Series,
    n_bkps: int = 5,
    model: str = "rbf",
    min_size: int = 30,
) -> List[pd.Timestamp]:
    """Detect a fixed number of breakpoints via binary segmentation.

    Binary segmentation is a greedy, top-down approach that recursively
    splits the series at the point of greatest cost improvement.

    Parameters
    ----------
    series : pd.Series
        Time series with ``DatetimeIndex``.
    n_bkps : int, default 5
        Number of breakpoints to detect.
    model : str, default "rbf"
        Cost model.
    min_size : int, default 30
        Minimum segment length.

    Returns
    -------
    list of pd.Timestamp
        Detected breakpoint dates.
    """
    signal: np.ndarray = series.dropna().values

    algo = ruptures.Binseg(model=model, min_size=min_size).fit(signal)
    breakpoint_indices: List[int] = algo.predict(n_bkps=n_bkps)

    breakpoint_indices = [bp for bp in breakpoint_indices if bp < len(signal)]

    dates: List[pd.Timestamp] = [
        series.dropna().index[bp] for bp in breakpoint_indices
    ]

    logger.info(
        "BinSeg detected %d breakpoints: %s",
        len(dates),
        [str(d.date()) for d in dates],
    )
    return dates


def plot_breaks(
    series: pd.Series,
    breakpoints: List[pd.Timestamp],
    title: str = "",
    figsize: tuple = (14, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot a time series with vertical lines at detected breakpoints.

    Parameters
    ----------
    series : pd.Series
        The time series to plot.
    breakpoints : list of pd.Timestamp
        Breakpoint dates to annotate.
    title : str, default ""
        Plot title.
    figsize : tuple, default (14, 5)
        Figure size if a new figure is created.
    ax : matplotlib.axes.Axes or None
        If provided, plot on the given axes; otherwise create a new
        figure.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(series.index, series.values, linewidth=0.8, color="steelblue")

    for bp in breakpoints:
        ax.axvline(
            x=bp,
            color="crimson",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label=str(bp.date()),
        )

    ax.set_title(title or "Structural Breaks", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel(series.name or "Value")

    # Avoid duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper left",
            fontsize=8,
            ncol=2,
        )

    fig.tight_layout()
    return fig
