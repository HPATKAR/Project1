"""Market data fetcher for FX, equities, and ETFs via yfinance."""

import pandas as pd
import numpy as np
from typing import Optional

from src.data.config import YF_TICKERS, DEFAULT_START, DEFAULT_END


def fetch_yf_prices(
    tickers: Optional[dict] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch closing prices for configured yfinance tickers."""
    import yfinance as yf

    tickers = tickers or YF_TICKERS
    start = start or str(DEFAULT_START)
    end = end or str(DEFAULT_END)

    ticker_list = list(tickers.values())
    name_map = {v: k for k, v in tickers.items()}

    data = yf.download(ticker_list, start=start, end=end, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    prices = prices.rename(columns=name_map)
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"
    return prices


def fetch_yf_returns(
    tickers: Optional[dict] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch daily returns for configured tickers."""
    prices = fetch_yf_prices(tickers, start, end)
    returns = prices.pct_change().dropna()
    return returns


def generate_simulated_market(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Generate simulated market data for development."""
    start = start or str(DEFAULT_START)
    end = end or str(DEFAULT_END)
    dates = pd.bdate_range(start=start, end=end)
    np.random.seed(43)
    n = len(dates)

    # USD/JPY: trending higher as rate differential widens
    usdjpy_returns = np.random.normal(0.0001, 0.005, n)
    usdjpy = 100 * np.exp(np.cumsum(usdjpy_returns))

    # Nikkei
    nk_returns = np.random.normal(0.0003, 0.012, n)
    nikkei = 15000 * np.exp(np.cumsum(nk_returns))

    # S&P 500
    spx_returns = np.random.normal(0.0003, 0.010, n)
    spx = 2000 * np.exp(np.cumsum(spx_returns))

    # TLT (inverse relationship with yields)
    tlt_returns = np.random.normal(-0.0001, 0.008, n)
    tlt = 120 * np.exp(np.cumsum(tlt_returns))

    # Sensex
    sensex_returns = np.random.normal(0.0003, 0.011, n)
    sensex = 30000 * np.exp(np.cumsum(sensex_returns))

    # Hang Seng
    hsi_returns = np.random.normal(0.0001, 0.013, n)
    hangseng = 22000 * np.exp(np.cumsum(hsi_returns))

    # Shanghai Composite
    sse_returns = np.random.normal(0.0001, 0.014, n)
    shanghai = 3000 * np.exp(np.cumsum(sse_returns))

    # KOSPI
    kospi_returns = np.random.normal(0.0002, 0.012, n)
    kospi = 2000 * np.exp(np.cumsum(kospi_returns))

    data = {
        "USDJPY": usdjpy,
        "EURJPY": usdjpy * 1.1 + np.random.randn(n) * 0.5,
        "EURUSD": 1.1 + np.random.randn(n) * 0.01,
        "NIKKEI": nikkei,
        "SPX": spx,
        "TLT": tlt,
        "IEF": tlt * 0.85 + np.random.randn(n) * 0.3,
        "SHY": 82 + np.random.randn(n) * 0.2,
        "SENSEX": sensex,
        "HANGSENG": hangseng,
        "SHANGHAI": shanghai,
        "KOSPI": kospi,
    }

    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    return df
