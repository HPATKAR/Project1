"""FRED API client for macroeconomic and rates data."""

import os
import pandas as pd
from typing import Optional

from src.data.config import FRED_SERIES, DEFAULT_START, DEFAULT_END


def fetch_fred_series(
    series_id: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    api_key: Optional[str] = None,
) -> pd.Series:
    """Fetch a single FRED series."""
    from fredapi import Fred

    key = api_key or os.environ.get("FRED_API_KEY")
    if not key or len(key) != 32 or not key.isalnum():
        raise ValueError(
            "FRED API key required (32-char alphanumeric). "
            "Set FRED_API_KEY env var or pass api_key."
        )

    fred = Fred(api_key=key)
    start = start or str(DEFAULT_START)
    end = end or str(DEFAULT_END)

    data = fred.get_series(series_id, observation_start=start, observation_end=end)
    data.name = series_id
    return data


def fetch_all_fred(
    api_key: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch all configured FRED series into a single DataFrame."""
    results = {}
    errors = {}

    for name, series_id in FRED_SERIES.items():
        try:
            results[name] = fetch_fred_series(series_id, start, end, api_key)
        except Exception as e:
            errors[name] = str(e)

    if errors:
        print(f"FRED fetch errors: {errors}")

    df = pd.DataFrame(results)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def generate_simulated_fred(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Generate simulated FRED data for development without API key."""
    import numpy as np

    start = start or str(DEFAULT_START)
    end = end or str(DEFAULT_END)
    dates = pd.bdate_range(start=start, end=end)
    np.random.seed(42)
    n = len(dates)

    # Simulate realistic yield levels with regime shifts
    t = np.linspace(0, 1, n)

    # Japan 10Y: near zero under YCC, rising after 2022
    jp_10y = np.where(
        t < 0.85,
        0.05 + 0.1 * np.random.randn(n) * 0.01,
        0.05 + 0.8 * (t - 0.85) / 0.15 + 0.15 * np.random.randn(n) * 0.01,
    )

    # US 10Y: rising cycle
    us_10y = 2.0 + 1.5 * np.sin(2 * np.pi * t) + 0.5 * t + np.random.randn(n) * 0.05

    data = {
        "JP_10Y": jp_10y,
        "JP_CALL_RATE": np.clip(jp_10y - 0.1, -0.1, 0.5),
        "US_2Y": us_10y - 0.5 + np.random.randn(n) * 0.03,
        "US_5Y": us_10y - 0.2 + np.random.randn(n) * 0.03,
        "US_10Y": us_10y,
        "US_30Y": us_10y + 0.3 + np.random.randn(n) * 0.03,
        "US_FF": np.clip(us_10y - 1.0, 0, 5.5) + np.random.randn(n) * 0.02,
        "DE_10Y": us_10y - 1.0 + np.random.randn(n) * 0.04,
        "VIX": 18 + 5 * np.abs(np.random.randn(n)),
    }

    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    return df
