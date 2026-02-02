"""Configuration for data sources, tickers, tenors, and date ranges."""

from datetime import date

# ── JGB Tenors (MOF Japan publishes these) ────────────────────────────
JGB_TENORS = [2, 5, 7, 10, 20, 30, 40]

# ── FRED Series IDs ───────────────────────────────────────────────────
FRED_SERIES = {
    # Japan
    "JP_10Y": "IRLTLT01JPM156N",        # Japan 10Y government bond yield
    "JP_CALL_RATE": "IRSTCB01JPM156N",   # BOJ overnight call rate
    "JP_CPI": "JPNCPIALLMINMEI",         # Japan CPI
    # US
    "US_2Y": "DGS2",                     # US 2Y Treasury
    "US_5Y": "DGS5",                     # US 5Y Treasury
    "US_10Y": "DGS10",                   # US 10Y Treasury
    "US_30Y": "DGS30",                   # US 30Y Treasury
    "US_FF": "DFF",                      # Fed Funds effective rate
    # Germany
    "DE_10Y": "IRLTLT01DEM156N",         # Germany 10Y Bund yield
    # Volatility
    "VIX": "VIXCLS",                     # CBOE VIX
    "MOVE": "BAMLHYH0A0HYM2TRIV",       # ICE BofA MOVE index (proxy)
}

# ── yfinance Tickers ──────────────────────────────────────────────────
YF_TICKERS = {
    # FX
    "USDJPY": "JPY=X",
    "EURJPY": "EURJPY=X",
    "EURUSD": "EURUSD=X",
    # Equity indices
    "NIKKEI": "^N225",
    "SPX": "^GSPC",
    "EUROSTOXX": "^STOXX50E",
    # Bond ETFs (for flow proxy)
    "TLT": "TLT",       # iShares 20+ Year Treasury
    "IEF": "IEF",       # iShares 7-10 Year Treasury
    "SHY": "SHY",       # iShares 1-3 Year Treasury
    "BNDX": "BNDX",     # Vanguard Total International Bond
    # JGB futures proxy (10Y JGB futures on OSE)
    "JGB_FUT": "JBH5.OSE",
}

# ── Date Ranges ───────────────────────────────────────────────────────
DEFAULT_START = date(2010, 1, 1)
DEFAULT_END = date.today()

# Analysis windows for backtesting
ANALYSIS_WINDOWS = {
    "full": (date(2010, 1, 1), date.today()),
    "pre_abenomics": (date(2010, 1, 1), date(2013, 4, 3)),
    "qqe_era": (date(2013, 4, 4), date(2016, 1, 28)),
    "nirp_ycc": (date(2016, 1, 29), date(2022, 12, 19)),
    "ycc_exit": (date(2022, 12, 20), date(2024, 3, 18)),
    "post_ycc": (date(2024, 3, 19), date.today()),
}

# ── BOJ Policy Shift Dates ───────────────────────────────────────────
BOJ_EVENTS = {
    "2013-04-04": "Kuroda QQE launch",
    "2014-10-31": "QQE expansion (Halloween surprise)",
    "2016-01-29": "Negative Interest Rate Policy",
    "2016-09-21": "Yield Curve Control introduced",
    "2018-07-31": "YCC flexibility (forward guidance change)",
    "2022-12-20": "YCC band widened to +/-0.50%",
    "2023-07-28": "YCC flexibility (+1.0% effective cap)",
    "2023-10-31": "1.0% reference (soft cap removed)",
    "2024-03-19": "BOJ exits NIRP and YCC formally",
}

# ── MOF Japan JGB Yield URL ──────────────────────────────────────────
MOF_JGB_URL = "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/jgbcme.csv"

# ── Data Storage ──────────────────────────────────────────────────────
DATA_DIR = "output/data"
