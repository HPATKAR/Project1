"""Position sizing utilities for the JGB repricing framework.

Provides three complementary sizing approaches:

1. **Volatility targeting** -- scales notional so that the position
   contributes a fixed amount of portfolio volatility.
2. **Regime-adjusted sizing** -- modulates a base position by the current
   regime probability and conviction score.
3. **Kelly criterion** -- computes the optimal fraction of capital to risk
   given a win rate and payoff ratio.

All functions are pure (no side-effects) and operate on scalar inputs so
they can be called element-wise inside vectorised pipelines.
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------
# 1. Volatility-target sizing
# ------------------------------------------------------------------
def volatility_target_size(
    signal: float,
    realized_vol: float,
    target_vol: float = 0.10,
    capital: float = 1_000_000.0,
) -> float:
    """Compute position size using constant-volatility targeting.

    The idea is to allocate notional such that the expected annualised
    volatility contribution of the position equals ``target_vol``
    multiplied by capital.

    .. math::

        \\text{position} = \\text{signal} \\times
        \\frac{\\text{target\\_vol}}{\\text{realized\\_vol}}
        \\times \\text{capital}

    Parameters
    ----------
    signal : float
        Directional signal strength, typically in [-1, 1].  The sign
        determines direction; magnitude scales size.
    realized_vol : float
        Current annualised realised volatility of the underlying
        instrument.  Must be strictly positive.
    target_vol : float, optional
        Desired annualised volatility contribution as a fraction
        (e.g. 0.10 for 10%).  Default is 0.10.
    capital : float, optional
        Total capital base in currency units.  Default is 1,000,000.

    Returns
    -------
    float
        Position size in currency units.  Positive means long; negative
        means short.

    Raises
    ------
    ValueError
        If ``realized_vol`` is not positive, or ``target_vol`` or
        ``capital`` is negative.

    Examples
    --------
    >>> volatility_target_size(signal=1.0, realized_vol=0.20)
    500000.0
    >>> volatility_target_size(signal=-0.5, realized_vol=0.10, target_vol=0.05)
    -250000.0
    """
    if realized_vol <= 0:
        raise ValueError(f"realized_vol must be > 0, got {realized_vol}")
    if target_vol < 0:
        raise ValueError(f"target_vol must be >= 0, got {target_vol}")
    if capital < 0:
        raise ValueError(f"capital must be >= 0, got {capital}")

    position = signal * (target_vol / realized_vol) * capital
    return float(position)


# ------------------------------------------------------------------
# 2. Regime-adjusted sizing
# ------------------------------------------------------------------
def regime_adjusted_size(
    base_size: float,
    regime_prob: float,
    conviction: float,
) -> float:
    """Scale a base position by regime probability and conviction.

    This is a multiplicative overlay: the final position is the base
    size attenuated (or amplified) by how confident the model is in the
    current regime and how high the trade-level conviction is.

    .. math::

        \\text{adjusted} = \\text{base\\_size} \\times
        \\text{regime\\_prob} \\times \\text{conviction}

    Parameters
    ----------
    base_size : float
        Base position size in currency units (from vol-target or other
        method).
    regime_prob : float
        Regime probability, must be in [0, 1].
    conviction : float
        Trade-level conviction score, must be in [0, 1].

    Returns
    -------
    float
        Regime- and conviction-adjusted position size.

    Raises
    ------
    ValueError
        If ``regime_prob`` or ``conviction`` is outside [0, 1].

    Examples
    --------
    >>> regime_adjusted_size(base_size=500_000, regime_prob=0.8, conviction=0.7)
    280000.0
    """
    if not 0.0 <= regime_prob <= 1.0:
        raise ValueError(f"regime_prob must be in [0, 1], got {regime_prob}")
    if not 0.0 <= conviction <= 1.0:
        raise ValueError(f"conviction must be in [0, 1], got {conviction}")

    adjusted = base_size * regime_prob * conviction
    return float(adjusted)


# ------------------------------------------------------------------
# 3. Kelly criterion
# ------------------------------------------------------------------
def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """Compute the Kelly criterion optimal bet fraction.

    The Kelly fraction maximises the expected logarithmic growth rate of
    capital under repeated bets with a fixed win rate and payoff ratio.

    .. math::

        f^* = \\frac{p \\cdot W - (1 - p) \\cdot L}{W}

    where *p* is the win rate, *W* is the average win, and *L* is the
    average loss (expressed as a positive number).

    Parameters
    ----------
    win_rate : float
        Probability of a winning trade, in (0, 1).
    avg_win : float
        Average profit on a winning trade (positive number).
    avg_loss : float
        Average loss on a losing trade (positive number representing the
        magnitude of loss).

    Returns
    -------
    float
        Optimal Kelly fraction.  A negative value indicates the edge is
        negative and the trade should not be taken.  Values above 1.0
        imply leverage.

    Raises
    ------
    ValueError
        If ``win_rate`` is not in (0, 1), or ``avg_win`` / ``avg_loss``
        are not positive.

    Notes
    -----
    In practice, most systematic strategies use a *half-Kelly* or
    *quarter-Kelly* fraction to account for estimation error and reduce
    drawdown severity.

    Examples
    --------
    >>> kelly_fraction(win_rate=0.55, avg_win=1.0, avg_loss=1.0)
    0.1
    >>> kelly_fraction(win_rate=0.60, avg_win=2.0, avg_loss=1.0)
    0.4
    """
    if not 0.0 < win_rate < 1.0:
        raise ValueError(f"win_rate must be in (0, 1), got {win_rate}")
    if avg_win <= 0:
        raise ValueError(f"avg_win must be > 0, got {avg_win}")
    if avg_loss <= 0:
        raise ValueError(f"avg_loss must be > 0, got {avg_loss}")

    fraction = (win_rate * avg_win - (1.0 - win_rate) * avg_loss) / avg_win
    return float(fraction)
