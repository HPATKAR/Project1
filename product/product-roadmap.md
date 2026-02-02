# JGB Repricing Framework — Product Roadmap

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    SECTION 5: TRADE ENGINE                   │
│  Regime-conditional trade generation, sizing, backtesting   │
├──────────────────────┬──────────────────────────────────────┤
│   SECTION 3:         │   SECTION 4:                         │
│   REGIME DETECTION   │   CROSS-ASSET SPILLOVER              │
│   Markov, breaks,    │   Transfer entropy, Diebold-Yilmaz,  │
│   entropy, GARCH     │   DCC-GARCH, FX/carry                │
├──────────────────────┴──────────────────────────────────────┤
│              SECTION 2: YIELD CURVE ANALYTICS                │
│  PCA, Nelson-Siegel, term premium (ACM), basis analysis     │
├─────────────────────────────────────────────────────────────┤
│              SECTION 1: DATA INFRASTRUCTURE                  │
│  FRED, MOF, BOJ, yfinance, COT — unified data pipeline     │
└─────────────────────────────────────────────────────────────┘
```

Each section is independently buildable and demo-able. Later sections build on earlier ones.

---

## Section 1: Data Infrastructure

**Goal:** Unified data pipeline that fetches, cleans, and stores all required time series.

**Delivers:**
- JGB yields (2Y, 5Y, 7Y, 10Y, 20Y, 30Y, 40Y) from MOF Japan
- Global benchmark yields (UST, Bund) from FRED
- FX rates (USD/JPY, EUR/JPY, EUR/USD) from yfinance
- BOJ balance sheet / operations data from BOJ Statistics
- CFTC COT positioning (JPY futures) from cot_reports
- Equity indices (Nikkei, S&P, EuroStoxx) from yfinance
- VIX and MOVE index from yfinance/FRED

**Key files:**
```
src/data/
  fred_client.py        # FRED API wrapper (fredapi)
  mof_scraper.py        # MOF Japan JGB yield scraper
  boj_scraper.py        # BOJ balance sheet + operations
  market_data.py        # yfinance wrapper for FX, equities, ETFs
  cot_client.py         # CFTC COT data
  data_store.py         # Unified storage (parquet files)
  config.py             # Tickers, tenors, date ranges, API keys
```

**Demo:** Run `python -m src.data.data_store --fetch-all` and get a unified DataFrame with all series aligned by date.

**Dependencies:** fredapi, yfinance, beautifulsoup4, cot_reports, pandas, requests

---

## Section 2: Yield Curve Analytics

**Goal:** Decompose JGB yield curve into tradeable factors and estimate term premium.

**Delivers:**
- PCA decomposition (level/slope/curvature) with rolling factor analysis
- Nelson-Siegel-Svensson parameter fitting and evolution tracking
- ACM term premium estimation for JGBs (custom implementation)
- JGB futures basis analysis (gross/net basis, CTD identification)
- Liquidity metrics (Amihud, Roll measure, on/off-the-run spread)

**Key files:**
```
src/yield_curve/
  pca.py                # PCA on JGB yield changes (sklearn)
  nelson_siegel.py      # NS/NSS fitting and parameter tracking
  term_premium.py       # ACM three-step regression for JGBs
  basis_analysis.py     # JGB futures basis, CTD, carry
  liquidity.py          # Amihud, Roll, composite liquidity index
```

**Demo:** Generate a dashboard showing JGB curve PCA loadings, term premium time series, and liquidity composite — highlighting how each shifted at known BOJ policy dates.

**Dependencies:** Section 1 data + scikit-learn, nelson-siegel-svensson, statsmodels, numpy

---

## Section 3: Regime Detection Engine

**Goal:** Multi-method regime detection that outputs a unified regime probability.

**Delivers:**
- Markov-switching model on JGB 10Y yield changes (2-state: suppressed vs. market-driven)
- Multivariate HMM on [JGB 10Y, USD/JPY, JGB vol] jointly
- Structural break detection via PELT on yield levels, volatility, and liquidity
- Rolling permutation entropy on JGB yield changes (complexity regime)
- GARCH conditional volatility with regime-switching variance
- BOJ policy function model (reaction function with structural breaks)
- **Ensemble regime probability** combining all methods

**Key files:**
```
src/regime/
  markov_switching.py   # statsmodels MarkovRegression
  hmm_regime.py         # hmmlearn multivariate HMM
  structural_breaks.py  # ruptures PELT/BinSeg
  entropy_regime.py     # antropy permutation entropy
  garch_regime.py       # arch GARCH with switching
  boj_policy.py         # BOJ reaction function model
  ensemble.py           # Combine all methods into unified regime probability
```

**Demo:** Plot regime probability time series overlaid on JGB 10Y yields, with vertical lines at known BOJ policy dates. Show that the ensemble detects shifts early.

**Dependencies:** Sections 1-2 + statsmodels, hmmlearn, ruptures, antropy, arch

---

## Section 4: Cross-Asset Spillover & Information Flow

**Goal:** Map how JGB repricing transmits to global markets and FX.

**Delivers:**
- Transfer entropy network: JGB -> UST, JGB -> USD/JPY, UST -> JGB, VIX -> JPY vol
- Diebold-Yilmaz spillover index (rolling, directional, net)
- DCC-GARCH time-varying correlation between JGB/UST/Bund
- FX carry trade analytics: carry-to-vol ratio, JPY funding cost
- CTA positioning proxy: trend-following signal replication + CFTC COT
- Bond ETF flow proxy: AUM-based flow estimation for TLT, BNDX, Japan ETFs

**Key files:**
```
src/spillover/
  transfer_entropy.py   # IDTxl/pyinform TE computation
  granger.py            # Granger causality baseline
  diebold_yilmaz.py     # VAR-based spillover index
  dcc_garch.py          # arch DCC-GARCH
src/fx/
  carry_analytics.py    # Carry-to-vol, rate differentials
  positioning.py        # CTA proxy + CFTC COT
  etf_flows.py          # Bond ETF flow proxy
```

**Demo:** Generate a spillover network graph showing information flow intensity between JGBs, USTs, Bunds, USD/JPY, and VIX. Show how the network topology changes across regimes.

**Dependencies:** Sections 1-3 + IDTxl or pyinform, arch (DCC), ta (trend signals)

---

## Section 5: Trade Engine & Strategy Backtesting

**Goal:** Convert regime detection and spillover signals into executable trade ideas.

**Delivers:**
- Regime-conditional trade generator: maps regime states to trade expressions
- Four trade categories: Rates / FX / Volatility / Cross-asset hedges
- Position sizing: volatility-targeting with regime-adjusted leverage
- Backtesting engine: validate against historical BOJ shifts (2013-2024)
- Trade card format: regime conditions, edge source, entry/exit, failure scenario
- Portfolio-level risk: correlation-aware allocation across trade legs

**Key files:**
```
src/strategy/
  trade_generator.py    # Regime -> trade mapping
  rates_trades.py       # JGB curve trades (steepeners, duration)
  fx_trades.py          # USD/JPY, carry trade expressions
  vol_trades.py         # Swaption/FX vol strategies
  cross_asset.py        # Global hedges, relative value
  sizing.py             # Vol-targeting, regime-adjusted
  backtester.py         # Historical validation engine
  trade_card.py         # Structured trade output format
  portfolio.py          # Cross-trade correlation and risk
```

**Demo:** Generate a set of trade cards for the current regime state, with historical P&L attribution showing how each trade performed around past BOJ shifts.

**Dependencies:** Sections 1-4 + vectorbt or pysystemtrade, Riskfolio-Lib

---

## Build Order & Dependencies

```
Section 1 (Data) ──→ Section 2 (Yield Curve) ──→ Section 3 (Regime) ──┐
                                                                        ├──→ Section 5 (Trades)
                     Section 1 (Data) ──→ Section 4 (Spillover) ───────┘
```

Sections 3 and 4 can be built in parallel once Sections 1 and 2 are complete.
Section 5 requires both 3 and 4.

## Package Installation

```bash
# Core (Tier 1)
pip install numpy pandas scipy scikit-learn statsmodels arch ruptures matplotlib

# Yield Curve (Tier 1)
pip install nelson-siegel-svensson

# Regime Detection (Tier 1)
pip install hmmlearn antropy

# Data (Tier 1)
pip install fredapi yfinance cot-reports beautifulsoup4 requests

# Information Flow (Tier 2)
pip install idtxl pyinform

# Strategy (Tier 2)
pip install vectorbt riskfolio-lib

# Advanced (Tier 3 — as needed)
pip install QuantLib rateslib pysystemtrade finmarketpy
```
