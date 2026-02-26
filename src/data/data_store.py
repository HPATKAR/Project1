"""Unified data store: fetches, aligns, and caches all data sources.

Supports three modes:
- ``use_simulated=True``:  All synthetic data (no network calls).
- ``use_simulated=False``: Live yfinance market data; FRED rates if API
  key available, otherwise falls back to yfinance yield proxies and MOF
  Japan JGB yields.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from src.data.config import (
    DATA_DIR, DEFAULT_START, DEFAULT_END, FRED_SERIES, MOF_JGB_URL,
)
from src.data.fred_client import fetch_all_fred, generate_simulated_fred
from src.data.market_data import fetch_yf_prices, generate_simulated_market

logger = logging.getLogger(__name__)

# yfinance symbols for yield proxies (used when FRED key is unavailable)
_YF_YIELD_TICKERS = {
    "US_10Y": "^TNX",     # CBOE 10Y Treasury Yield Index (×10 = %)
    "US_5Y": "^FVX",      # CBOE 5Y Treasury Yield Index
    "US_30Y": "^TYX",     # CBOE 30Y Treasury Yield Index
    "VIX": "^VIX",        # CBOE VIX
}


def _fetch_yf_yields(start: str, end: str) -> pd.DataFrame:
    """Fetch US Treasury yields and VIX from yfinance as FRED alternative."""
    import yfinance as yf

    ticker_list = list(_YF_YIELD_TICKERS.values())
    name_map = {v: k for k, v in _YF_YIELD_TICKERS.items()}

    data = yf.download(ticker_list, start=start, end=end, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    prices = prices.rename(columns=name_map)
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"

    # ^TNX, ^FVX, ^TYX report yields ×10 — no longer the case in recent
    # yfinance versions but check and correct if needed
    for col in ["US_10Y", "US_5Y", "US_30Y"]:
        if col in prices.columns:
            # If max > 20 it's likely in basis-point-like format; rescale
            if prices[col].max() > 20:
                prices[col] = prices[col] / 10.0

    return prices


def _fetch_mof_jgb_yields(start: str, end: str) -> pd.DataFrame:
    """Fetch JGB yields from Japan Ministry of Finance CSV endpoint."""
    import re

    try:
        # Try multiple encodings
        for enc in ["shift_jis", "utf-8", "cp932"]:
            try:
                df = pd.read_csv(MOF_JGB_URL, encoding=enc, skiprows=1)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            logger.warning("Could not decode MOF CSV with any encoding.")
            return pd.DataFrame()

        # Parse date column (first column)
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df.set_index(date_col)
        df.index.name = "date"

        # Map MOF column names to our schema using exact tenor matching
        # MOF columns are like "1年", "2年", "5年", "10年", "20年", "30年"
        # or in English: "1yr", "2yr", etc.
        tenor_map = {
            "JP_2Y": [r"\b2\b", r"^2年$", r"^2yr"],
            "JP_5Y": [r"\b5\b", r"^5年$", r"^5yr"],
            "JP_10Y": [r"\b10\b", r"^10年$", r"^10yr"],
            "JP_20Y": [r"\b20\b", r"^20年$", r"^20yr"],
            "JP_30Y": [r"\b30\b", r"^30年$", r"^30yr"],
        }

        col_map = {}
        used_targets = set()
        for col in df.columns:
            col_str = str(col).strip()
            for target, patterns in tenor_map.items():
                if target in used_targets:
                    continue
                for pat in patterns:
                    if re.search(pat, col_str):
                        col_map[col] = target
                        used_targets.add(target)
                        break
                if col in col_map:
                    break

        if col_map:
            df = df[list(col_map.keys())]
            df = df.rename(columns=col_map)
            df = df.apply(pd.to_numeric, errors="coerce")
            # Remove any remaining duplicate columns
            df = df.loc[:, ~df.columns.duplicated()]
            df = df.loc[start:end]
            logger.info("Fetched %d rows of JGB yields from MOF Japan.", len(df))
            return df
    except Exception as exc:
        logger.warning("MOF JGB fetch failed: %s", exc)

    return pd.DataFrame()


def _synthesize_missing_rates(
    df: pd.DataFrame, start: str, end: str,
) -> pd.DataFrame:
    """Fill in missing rate columns with simple proxies."""
    dates = pd.bdate_range(start=start, end=end)
    if df.empty:
        df = pd.DataFrame(index=dates)
        df.index.name = "date"

    # JP_10Y: if missing, estimate from available data or use placeholder
    if "JP_10Y" not in df.columns:
        np.random.seed(42)
        n = len(df)
        t = np.linspace(0, 1, n)
        # Approximate JGB 10Y trajectory: near zero → rising
        jp_10y = np.where(
            t < 0.85,
            0.05 + np.random.randn(n) * 0.01,
            0.05 + 0.8 * (t - 0.85) / 0.15 + np.random.randn(n) * 0.015,
        )
        df["JP_10Y"] = jp_10y
        logger.info("JP_10Y: using synthetic proxy (no FRED/MOF data).")

    # JP_CALL_RATE: approximate from JP_10Y
    if "JP_CALL_RATE" not in df.columns:
        df["JP_CALL_RATE"] = np.clip(df["JP_10Y"] - 0.1, -0.1, 0.5)

    # US_2Y: approximate from US_10Y if available
    if "US_2Y" not in df.columns and "US_10Y" in df.columns:
        df["US_2Y"] = df["US_10Y"] - 0.5 + np.random.randn(len(df)) * 0.03

    # US_FF: approximate
    if "US_FF" not in df.columns and "US_10Y" in df.columns:
        df["US_FF"] = np.clip(df["US_10Y"] - 1.0, 0, 5.5)

    # DE_10Y: approximate from US_10Y
    if "DE_10Y" not in df.columns and "US_10Y" in df.columns:
        df["DE_10Y"] = df["US_10Y"] - 1.0 + np.random.randn(len(df)) * 0.04

    # UK_10Y: approximate from US_10Y (Gilts track Treasuries with ~20bp spread)
    if "UK_10Y" not in df.columns and "US_10Y" in df.columns:
        df["UK_10Y"] = df["US_10Y"] - 0.2 + np.random.randn(len(df)) * 0.05
        logger.info("UK_10Y: using synthetic proxy (US_10Y - 20bp + noise).")

    # AU_10Y: approximate from US_10Y (AU trades ~30bp above US historically)
    if "AU_10Y" not in df.columns and "US_10Y" in df.columns:
        df["AU_10Y"] = df["US_10Y"] + 0.3 + np.random.randn(len(df)) * 0.04
        logger.info("AU_10Y: using synthetic proxy (US_10Y + 30bp + noise).")

    # JGBi breakeven: JP_10Y minus real rate proxy (CPI-adjusted)
    if "JP_BREAKEVEN" not in df.columns and "JP_10Y" in df.columns:
        if "JP_CPI_CORE" in df.columns:
            # Breakeven ~ nominal yield - (CPI-implied real rate proxy)
            cpi_yoy = df["JP_CPI_CORE"].pct_change(12).fillna(0) * 100
            df["JP_BREAKEVEN"] = df["JP_10Y"] - (df["JP_10Y"] - cpi_yoy.clip(-1, 5))
            logger.info("JP_BREAKEVEN: computed from JP_10Y and JP_CPI_CORE.")
        else:
            # Simple proxy: breakeven ≈ 0.5-1.5% for Japan
            np.random.seed(43)
            df["JP_BREAKEVEN"] = 0.8 + np.random.randn(len(df)) * 0.15
            logger.info("JP_BREAKEVEN: using synthetic proxy (~0.8% ± noise).")

    return df


class DataStore:
    """Unified data access layer for the JGB repricing framework."""

    def __init__(self, data_dir: Optional[str] = None, use_simulated: bool = False):
        self.data_dir = Path(data_dir or DATA_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.use_simulated = use_simulated
        self._cache = {}

    def get_rates(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        fred_api_key: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get interest rate data.

        Priority: FRED API → yfinance yields + MOF Japan → simulated.
        """
        cache_key = "rates"
        if cache_key in self._cache:
            return self._cache[cache_key]

        parquet_path = self.data_dir / "rates.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            self._cache[cache_key] = df
            return df

        start_str = start or str(DEFAULT_START)
        end_str = end or str(DEFAULT_END)

        if self.use_simulated:
            df = generate_simulated_fred(start_str, end_str)
        else:
            # Try FRED first
            api_key = fred_api_key or os.environ.get("FRED_API_KEY")
            if api_key:
                try:
                    df = fetch_all_fred(api_key=api_key, start=start_str, end=end_str)
                    if not df.empty:
                        logger.info("Rates loaded from FRED (%d rows).", len(df))
                    else:
                        raise ValueError("FRED returned empty data.")
                except Exception as exc:
                    logger.warning("FRED fetch failed: %s. Falling back.", exc)
                    api_key = None  # trigger fallback below

            if not api_key:
                # Fallback: yfinance yields + MOF Japan JGB
                logger.info("No FRED API key — using yfinance yields + MOF Japan JGB")
                yf_yields = pd.DataFrame()
                mof_jgb = pd.DataFrame()

                try:
                    yf_yields = _fetch_yf_yields(start_str, end_str)
                    logger.info("yfinance yields: %d rows, cols=%s", len(yf_yields), list(yf_yields.columns))
                except Exception as exc:
                    logger.warning("yfinance yields fetch failed: %s", exc)

                try:
                    mof_jgb = _fetch_mof_jgb_yields(start_str, end_str)
                    if not mof_jgb.empty:
                        logger.info("MOF Japan JGB: %d rows, cols=%s", len(mof_jgb), list(mof_jgb.columns))
                except Exception as exc:
                    logger.warning("MOF JGB fetch failed: %s", exc)

                if not mof_jgb.empty and not yf_yields.empty:
                    # Strip timezone info for join compatibility
                    if yf_yields.index.tz is not None:
                        yf_yields.index = yf_yields.index.tz_localize(None)
                    if mof_jgb.index.tz is not None:
                        mof_jgb.index = mof_jgb.index.tz_localize(None)
                    df = yf_yields.join(mof_jgb, how="outer").sort_index()
                elif not yf_yields.empty:
                    if yf_yields.index.tz is not None:
                        yf_yields.index = yf_yields.index.tz_localize(None)
                    df = yf_yields
                elif not mof_jgb.empty:
                    if mof_jgb.index.tz is not None:
                        mof_jgb.index = mof_jgb.index.tz_localize(None)
                    df = mof_jgb
                else:
                    df = pd.DataFrame()

                # Remove duplicate columns
                df = df.loc[:, ~df.columns.duplicated()]
                df = df.ffill()
                df = _synthesize_missing_rates(df, start_str, end_str)

        df.to_parquet(parquet_path)
        self._cache[cache_key] = df
        return df

    def get_market(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get market data (FX, equities, ETFs — yfinance or simulated)."""
        cache_key = "market"
        if cache_key in self._cache:
            return self._cache[cache_key]

        parquet_path = self.data_dir / "market.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            self._cache[cache_key] = df
            return df

        if self.use_simulated:
            df = generate_simulated_market(start, end)
        else:
            df = fetch_yf_prices(start=start, end=end)

        df.to_parquet(parquet_path)
        self._cache[cache_key] = df
        return df

    def get_unified(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        fred_api_key: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get all data merged into a single aligned DataFrame."""
        cache_key = "unified"
        if cache_key in self._cache:
            return self._cache[cache_key]

        rates = self.get_rates(start, end, fred_api_key)
        market = self.get_market(start, end)

        # Align on business days via outer join, forward-fill
        df = rates.join(market, how="outer").sort_index()
        df = df.ffill().dropna(how="all")

        parquet_path = self.data_dir / "unified.parquet"
        df.to_parquet(parquet_path)
        self._cache[cache_key] = df
        return df

    def clear_cache(self):
        """Clear in-memory and disk cache."""
        self._cache.clear()
        for f in self.data_dir.glob("*.parquet"):
            f.unlink()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JGB Data Store")
    parser.add_argument("--simulated", action="store_true", help="Use simulated data")
    parser.add_argument("--start", default=str(DEFAULT_START))
    parser.add_argument("--end", default=str(DEFAULT_END))
    args = parser.parse_args()

    store = DataStore(use_simulated=args.simulated)
    df = store.get_unified(start=args.start, end=args.end)
    print(f"Unified dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample:\n{df.tail()}")
