"""FX carry trade analytics for the JGB repricing framework.

Carry trades exploit interest rate differentials between currencies.
For JPY-funded carry, the trader borrows in JPY (low rate) and invests
in a higher-yielding currency.  The carry return is the rate differential,
but is exposed to FX risk (JPY appreciation erodes P&L).

This module provides tools for monitoring carry attractiveness, which
is a key driver of USDJPY and an indirect influence on JGB demand
from foreign investors.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_carry(
    domestic_rate: pd.Series,
    foreign_rate: pd.Series,
) -> pd.Series:
    """Compute annualized carry (interest rate differential).

    Carry = foreign_rate - domestic_rate.

    For a JPY-funded carry trade into USD:
    - ``domestic_rate`` = Japan overnight call rate
    - ``foreign_rate`` = US Fed Funds rate

    A positive carry means the foreign currency offers a higher yield,
    making the carry trade attractive.

    Parameters
    ----------
    domestic_rate : pd.Series
        Domestic (funding) interest rate in annualized percentage points
        (e.g. 0.10 for 10 bps).
    foreign_rate : pd.Series
        Foreign (investment) interest rate in annualized percentage points.

    Returns
    -------
    pd.Series
        Annualized carry (rate differential), same units as input.
    """
    carry = foreign_rate.sub(domestic_rate, fill_value=np.nan)
    carry.name = "carry"
    carry = carry.dropna()
    return carry


def carry_to_vol(
    carry: pd.Series,
    fx_vol: pd.Series,
) -> pd.Series:
    """Compute carry-to-volatility ratio (Sharpe-like attractiveness metric).

    Higher carry-to-vol means the carry trade offers better compensation
    for FX risk.  This is widely used by macro funds to gauge when carry
    trades are attractive vs. vulnerable.

    Parameters
    ----------
    carry : pd.Series
        Annualized carry (output of ``compute_carry``).
    fx_vol : pd.Series
        FX realized or implied volatility (annualized, same units as carry —
        both in percentage points or both in decimal).

    Returns
    -------
    pd.Series
        Carry-to-vol ratio.  Values above 0.5 are generally considered
        attractive; values below 0.2 suggest carry trades are vulnerable
        to unwind.
    """
    # Avoid division by zero
    safe_vol = fx_vol.replace(0, np.nan)
    ratio = carry / safe_vol
    ratio.name = "carry_to_vol"
    return ratio.dropna()


def rolling_carry_metrics(
    rates_df: pd.DataFrame,
    fx_returns: pd.Series,
    window: int = 63,
) -> pd.DataFrame:
    """Compute rolling carry trade metrics for JPY-funded carry into USD.

    Parameters
    ----------
    rates_df : pd.DataFrame
        Must contain columns ``'US_FF'`` (Fed Funds rate) and
        ``'JP_CALL_RATE'`` (BOJ overnight call rate), both in annualized
        percentage points.
    fx_returns : pd.Series
        Daily USDJPY log returns (or simple returns).  Used to compute
        realized FX volatility.
    window : int, default 63
        Rolling window in business days (63 ~ 3 months).

    Returns
    -------
    pd.DataFrame
        Columns:
        - ``carry`` : rolling carry (rate differential)
        - ``realized_vol`` : annualized realized FX volatility
        - ``carry_to_vol`` : carry / realized_vol
    """
    required_cols = {"US_FF", "JP_CALL_RATE"}
    missing = required_cols - set(rates_df.columns)
    if missing:
        raise ValueError(f"rates_df missing required columns: {missing}")

    # Align indices
    rates = rates_df[["US_FF", "JP_CALL_RATE"]].copy()
    fx_ret = fx_returns.copy()
    fx_ret.name = "fx_returns"

    combined = pd.concat([rates, fx_ret], axis=1).dropna()

    # Rolling carry (use rolling mean to smooth noisy daily rate data)
    rolling_carry = (
        combined["US_FF"].rolling(window=window, min_periods=window // 2).mean()
        - combined["JP_CALL_RATE"]
        .rolling(window=window, min_periods=window // 2)
        .mean()
    )
    rolling_carry.name = "carry"

    # Realized volatility: annualized std of daily FX returns
    realized_vol = (
        combined["fx_returns"]
        .rolling(window=window, min_periods=window // 2)
        .std()
        * np.sqrt(252)
    )
    realized_vol.name = "realized_vol"

    # Carry-to-vol ratio
    # Carry is in percentage points (e.g. 5.0 for 5%), vol is in decimal
    # (e.g. 0.10 for 10%).  Normalize carry to decimal form for ratio.
    carry_decimal = rolling_carry / 100.0  # convert from pct pts to decimal
    c2v = carry_to_vol(carry_decimal, realized_vol)

    result = pd.DataFrame(
        {
            "carry": rolling_carry,
            "realized_vol": realized_vol,
            "carry_to_vol": c2v,
        }
    ).dropna()

    logger.info(
        "Rolling carry metrics computed: %d obs, window=%d days.",
        len(result),
        window,
    )

    return result
