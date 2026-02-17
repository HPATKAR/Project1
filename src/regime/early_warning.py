"""
Early Warning System for JGB repricing events.

Combines entropy divergence, carry stress, and spillover intensity
into a composite warning score with configurable thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Warning:
    """A single warning signal."""
    timestamp: pd.Timestamp
    component: str
    value: float
    threshold: float
    severity: str  # INFO, WARNING, CRITICAL
    message: str


def entropy_divergence(
    entropy: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Compute rolling z-score of entropy to detect divergence from normal."""
    mean = entropy.rolling(window, min_periods=30).mean()
    std = entropy.rolling(window, min_periods=30).std()
    return ((entropy - mean) / std.replace(0, np.nan)).fillna(0)


def carry_stress_indicator(
    jp_rate: pd.Series,
    us_rate: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Carry stress: z-score of US-JP rate differential."""
    spread = us_rate - jp_rate
    mean = spread.rolling(window, min_periods=60).mean()
    std = spread.rolling(window, min_periods=60).std()
    return ((spread - mean) / std.replace(0, np.nan)).fillna(0)


def spillover_intensity(
    correlation: pd.Series,
    threshold: float = 0.5,
) -> pd.Series:
    """Spillover intensity based on rolling correlation exceeding threshold."""
    return (correlation.abs() - threshold).clip(lower=0) * 2


def composite_warning_score(
    entropy_z: pd.Series,
    carry_z: pd.Series,
    spillover: pd.Series,
    weights: Optional[dict] = None,
) -> pd.Series:
    """Composite early warning score (0-100 scale).

    Default weights: entropy=0.4, carry=0.3, spillover=0.3.
    """
    w = weights or {"entropy": 0.4, "carry": 0.3, "spillover": 0.3}

    # Normalize each to 0-100
    def _to_score(s: pd.Series) -> pd.Series:
        abs_s = s.abs()
        s_min = abs_s.rolling(252, min_periods=30).min()
        s_max = abs_s.rolling(252, min_periods=30).max()
        rng = (s_max - s_min).replace(0, np.nan)
        return ((abs_s - s_min) / rng * 100).fillna(0).clip(0, 100)

    common = entropy_z.index.intersection(carry_z.index).intersection(spillover.index)
    if len(common) == 0:
        return pd.Series(dtype=float)

    score = (
        w["entropy"] * _to_score(entropy_z.loc[common])
        + w["carry"] * _to_score(carry_z.loc[common])
        + w["spillover"] * _to_score(spillover.loc[common])
    )
    return score.clip(0, 100)


def compute_simple_warning_score(
    df: pd.DataFrame,
    entropy_window: int = 60,
) -> pd.Series:
    """Simplified composite warning score using only the unified DataFrame.

    Works without FRED data by using available yield columns.
    """
    jp10 = df["JP_10Y"] if "JP_10Y" in df.columns else pd.Series(dtype=float)
    us10 = df["US_10Y"] if "US_10Y" in df.columns else pd.Series(dtype=float)

    if len(jp10.dropna()) < 60:
        return pd.Series(dtype=float)

    # Entropy divergence from rolling vol
    changes = jp10.diff()
    vol = changes.rolling(entropy_window).std()
    ent_z = entropy_divergence(vol, window=entropy_window)

    # Carry stress
    if len(us10.dropna()) > 60:
        carry_z = carry_stress_indicator(jp10, us10)
    else:
        carry_z = pd.Series(0, index=jp10.index)

    # Spillover (correlation proxy)
    if len(us10.dropna()) > 60:
        corr = jp10.rolling(60).corr(us10)
        spill = spillover_intensity(corr)
    else:
        spill = pd.Series(0, index=jp10.index)

    return composite_warning_score(ent_z, carry_z, spill)


def generate_warnings(
    score: pd.Series,
    cooldown_days: int = 5,
) -> List[Warning]:
    """Generate Warning objects from the composite score.

    Thresholds:
    - CRITICAL: score > 80
    - WARNING: score > 50
    - INFO: score > 30
    """
    warnings = []
    last_warning_date = None

    for dt, val in score.items():
        if last_warning_date is not None:
            days_since = (dt - last_warning_date).days
            if days_since < cooldown_days:
                continue

        if val > 80:
            severity = "CRITICAL"
            msg = f"Composite warning score at {val:.0f}/100. Multiple stress indicators firing simultaneously."
        elif val > 50:
            severity = "WARNING"
            msg = f"Warning score elevated at {val:.0f}/100. Monitor closely for regime shift signals."
        elif val > 30:
            severity = "INFO"
            msg = f"Warning score at {val:.0f}/100. Early stress signals detected but not yet actionable."
        else:
            continue

        warnings.append(Warning(
            timestamp=pd.Timestamp(dt),
            component="composite",
            value=float(val),
            threshold=30.0,
            severity=severity,
            message=msg,
        ))
        last_warning_date = dt

    return warnings
