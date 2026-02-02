"""CTA positioning proxy for systematic trend-following exposure estimation.

Commodity Trading Advisors (CTAs) and managed futures funds are major
participants in JGB and USDJPY markets.  Their positioning is driven by
price trends and volatility.  This module builds a proxy for CTA positioning
using multi-timeframe trend signals with volatility-targeted sizing.

The proxy can be used to:
- Estimate crowding in trend-following positions
- Anticipate forced liquidation during trend reversals
- Gauge the vulnerability of JPY carry/JGB shorts to CTA unwinds
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def trend_signal(
    prices: pd.Series,
    lookbacks: Optional[List[int]] = None,
) -> pd.Series:
    """Compute a multi-timeframe trend signal.

    For each lookback period, the signal is:
        sign(price - SMA(price, lookback))

    The final signal is the average across all lookback periods,
    producing a value in [-1, 1].  Values near +1 indicate strong
    uptrend consensus across timeframes; values near -1 indicate
    strong downtrend consensus.

    Parameters
    ----------
    prices : pd.Series
        Price series (levels, not returns).
    lookbacks : list of int, optional
        SMA lookback periods in business days.
        Default: [21, 63, 126, 252] (1M, 3M, 6M, 12M).

    Returns
    -------
    pd.Series
        Trend signal in [-1, 1], indexed to the price series.
    """
    if lookbacks is None:
        lookbacks = [21, 63, 126, 252]

    prices = prices.dropna()

    if len(prices) < max(lookbacks):
        logger.warning(
            "Price series (%d obs) shorter than longest lookback (%d).",
            len(prices),
            max(lookbacks),
        )

    signals = pd.DataFrame(index=prices.index)

    for lb in lookbacks:
        sma = prices.rolling(window=lb, min_periods=lb).mean()
        # sign(price - SMA): +1 if above, -1 if below, 0 if equal
        sig = np.sign(prices - sma)
        signals[f"signal_{lb}"] = sig

    # Average across lookbacks
    avg_signal = signals.mean(axis=1)
    avg_signal.name = "trend_signal"

    return avg_signal


def trend_position(
    signal: pd.Series,
    realized_vol: pd.Series,
    target_vol: float = 0.10,
) -> pd.Series:
    """Convert trend signal to volatility-targeted position size.

    Position = signal * (target_vol / realized_vol)

    This ensures that the portfolio's expected volatility contribution
    from each asset is approximately equal to ``target_vol``, regardless
    of the asset's realized volatility.

    Parameters
    ----------
    signal : pd.Series
        Trend signal in [-1, 1] (output of ``trend_signal``).
    realized_vol : pd.Series
        Annualized realized volatility of the asset.
    target_vol : float, default 0.10
        Target annualized volatility (e.g. 0.10 for 10%).

    Returns
    -------
    pd.Series
        Position size.  Magnitude indicates notional scaling (e.g. 2.0
        means 2x notional); sign indicates direction (long/short).
        Capped at +/- 5x to prevent extreme leverage in low-vol regimes.
    """
    # Prevent division by zero and extreme leverage
    safe_vol = realized_vol.replace(0, np.nan).clip(lower=1e-6)

    position = signal * (target_vol / safe_vol)

    # Cap extreme positions (low-vol environments can produce huge sizing)
    max_leverage = 5.0
    position = position.clip(lower=-max_leverage, upper=max_leverage)
    position.name = "position"

    return position.dropna()


def compute_cta_proxy(
    price_dict: Dict[str, pd.Series],
    target_vol: float = 0.10,
    lookbacks: Optional[List[int]] = None,
    vol_window: int = 63,
) -> pd.DataFrame:
    """Compute CTA proxy positioning across multiple assets.

    For each asset:
    1. Compute multi-timeframe trend signal.
    2. Estimate realized volatility from returns.
    3. Size position using vol-targeting.

    Parameters
    ----------
    price_dict : dict of str -> pd.Series
        Dictionary mapping asset names to price series.
    target_vol : float, default 0.10
        Target annualized volatility per asset.
    lookbacks : list of int, optional
        Lookback periods for trend signal.
        Default: [21, 63, 126, 252].
    vol_window : int, default 63
        Rolling window for realized volatility estimation.

    Returns
    -------
    pd.DataFrame
        DataFrame of position sizes, with columns for each asset and
        a DatetimeIndex.  Positive = long, negative = short.
    """
    if lookbacks is None:
        lookbacks = [21, 63, 126, 252]

    positions: Dict[str, pd.Series] = {}

    for asset_name, prices in price_dict.items():
        prices = prices.dropna()

        if len(prices) < max(lookbacks) + vol_window:
            logger.warning(
                "Insufficient data for %s (%d obs), skipping.",
                asset_name,
                len(prices),
            )
            continue

        # Trend signal
        sig = trend_signal(prices, lookbacks=lookbacks)

        # Realized vol from log returns
        log_returns = np.log(prices / prices.shift(1)).dropna()
        realized_vol = (
            log_returns.rolling(window=vol_window, min_periods=vol_window // 2)
            .std()
            * np.sqrt(252)
        )

        # Vol-targeted position
        pos = trend_position(sig, realized_vol, target_vol=target_vol)
        positions[asset_name] = pos

    if not positions:
        logger.warning("No assets had sufficient data for CTA proxy.")
        return pd.DataFrame()

    result = pd.DataFrame(positions)
    result = result.dropna(how="all")

    logger.info(
        "CTA proxy positions computed for %d assets, %d observations.",
        len(positions),
        len(result),
    )

    return result
