# JGB Repricing Framework

**Course:** MGMT 69000 — Mastering AI for Finance | Purdue University
**Author:** Heramb Patkar
**Thesis:** Japan Bond Market Repricing — regime shift from BOJ-suppressed yields to market-driven pricing.

## Overview

A production-grade quantitative research platform that **detects regime shifts** in the Japanese Government Bond market and **generates tradeable strategies** across rates, FX, volatility, and cross-asset portfolios.

The framework integrates five analytical layers:

1. **Yield Curve Analytics** — PCA decomposition (Level/Slope/Curvature), Nelson-Siegel parametric fitting, liquidity metrics (Roll measure, Amihud), ACM term premium estimation
2. **Regime Detection** — Four independent models (Markov-Switching, HMM, Permutation Entropy, GARCH) combined via weighted ensemble into a single regime probability
3. **Spillover & Information Flow** — Diebold-Yilmaz VAR-FEVD spillover index, DCC-GARCH time-varying correlations, Granger causality, Transfer Entropy network
4. **FX & Carry Analytics** — JPY carry-to-vol ratio, rate differentials, positioning indicators
5. **Trade Generation** — Regime-conditional trade cards with entry/exit signals, DV01-neutral sizing, and explicit failure scenarios

## Architecture

```
jgb-repricing-framework/
|-- app.py                          # Streamlit 5-page dashboard (1858 lines)
|-- run_analysis.py                 # CLI batch analysis runner
|-- requirements.txt                # 15 core dependencies
|-- Dockerfile                      # Docker deployment (Python 3.11-slim)
|-- render.yaml                     # Render.com cloud deployment config
|-- .streamlit/config.toml          # Institutional desk theme (Inter font, muted palette)
|-- src/
|   |-- data/
|   |   |-- config.py               # FRED/yfinance tickers, BOJ events, date ranges
|   |   |-- data_store.py           # Unified data layer (FRED -> yfinance -> MOF -> synthetic)
|   |   |-- fred_client.py          # FRED API client
|   |   `-- market_data.py          # yfinance integration
|   |-- yield_curve/
|   |   |-- pca.py                  # PCA decomposition + rolling PCA + factor interpretation
|   |   |-- nelson_siegel.py        # Nelson-Siegel 3-factor parametric fitting
|   |   |-- liquidity.py            # Roll measure, Amihud illiquidity, composite index
|   |   `-- term_premium.py         # ACM affine term premium estimation
|   |-- regime/
|   |   |-- markov_switching.py     # Hamilton (1989) 2-state Markov-switching regression
|   |   |-- hmm_regime.py           # Gaussian HMM on multivariate features
|   |   |-- entropy_regime.py       # Permutation entropy + sample entropy regime signal
|   |   |-- garch_regime.py         # GARCH(1,1) conditional vol + PELT breakpoints
|   |   |-- structural_breaks.py    # PELT structural break detection
|   |   `-- ensemble.py             # Weighted ensemble of 4 regime detectors
|   |-- spillover/
|   |   |-- diebold_yilmaz.py       # VAR(4) generalized FEVD spillover index
|   |   |-- dcc_garch.py            # DCC time-varying correlations (GARCH + EWMA)
|   |   |-- granger.py              # Pairwise Granger causality (F-test, lags 1-5)
|   |   `-- transfer_entropy.py     # Histogram-based directional information flow
|   |-- fx/
|   |   |-- carry_analytics.py      # Carry-to-vol ratio, rate differentials
|   |   `-- positioning.py          # Positioning-based regime indicators
|   `-- strategy/
|       |-- trade_card.py           # TradeCard dataclass with failure scenarios
|       |-- trade_generator.py      # Regime-conditional rates/FX/vol/cross-asset rules
|       |-- sizing.py               # Vol-target and DV01-neutral position sizing
|       `-- backtester.py           # Strategy backtesting engine
|-- tests/
|   |-- test_yield_curve.py         # PCA shapes, explained variance, NS, liquidity
|   |-- test_regime.py              # Markov, HMM, entropy, ensemble integration
|   |-- test_spillover.py           # DY spillover, Granger, transfer entropy
|   `-- test_data.py                # Data pipeline and column validation
|-- product/
|   |-- README.md                   # DRIVER workflow & thesis summary
|   |-- product-overview.md         # Problem statement, tech stack, research lenses
|   `-- product-roadmap.md          # Section breakdown
|-- docs/
|   `-- ai_collaboration_log.md     # AI tool usage documentation
`-- output/data/                    # Cached parquet files
```

## Methods

### PCA Yield Curve Decomposition

Principal Component Analysis on daily yield changes across tenors (2Y, 5Y, 7Y, 10Y, 20Y, 30Y):

| Component | Interpretation | Typical Variance | Validation |
|-----------|---------------|-----------------|------------|
| PC1 | **Level** (parallel shift) | 60-80% | All loadings same sign, low coefficient of variation |
| PC2 | **Slope** (twist) | 10-20% | Loadings change sign once (short vs long end) |
| PC3 | **Curvature** (butterfly) | 5-10% | Loadings change sign twice (hump-shaped) |

- **Input:** Daily yield changes (not levels) for stationarity
- **Scaling:** Covariance-based (preserves bps-scale economic meaning)
- **Rolling PCA:** 252-day window tracks time-varying factor structure
- **Interpretation:** Automated heuristic validates factor identity against fixed-income literature (Litterman-Scheinkman 1991)
- **Implementation:** `src/yield_curve/pca.py` — `fit_yield_pca()`, `rolling_pca()`, `interpret_pca()`

### 4-Model Ensemble Regime Detection

| Model | Package | Signal | Strength |
|-------|---------|--------|----------|
| Markov-Switching | statsmodels | Smoothed state probabilities | Volatility clustering |
| Gaussian HMM | hmmlearn | Viterbi state sequence | Cross-asset co-movement |
| Permutation Entropy | antropy | Z-score threshold signal | Early warning (fires first) |
| GARCH(1,1) + PELT | arch + ruptures | Vol-regime breakpoints | Structural break timing |

- **Ensemble:** Min-max normalize each signal to [0,1], weighted average (default: 25% each)
- **Output:** Single probability series — 0 = BOJ-suppressed, 1 = market-driven repricing
- **Conviction thresholds:** >0.7 STRONG | 0.5-0.7 MODERATE | 0.3-0.5 TRANSITION | <0.3 SUPPRESSED
- **Implementation:** `src/regime/ensemble.py`

### Cross-Asset Spillover Analysis

| Method | What It Measures | Key Output |
|--------|-----------------|------------|
| Diebold-Yilmaz (2012) | Forecast error variance decomposition | Total spillover % + net directional flows |
| DCC-GARCH | Time-varying correlations | Crisis-driven correlation spikes |
| Granger Causality | Lagged predictability | Significant cause-effect pairs with optimal lag |
| Transfer Entropy | Directional information flow | Asymmetric leader/follower network |

- **Variables:** JP_10Y, US_10Y, DE_10Y, USDJPY, NIKKEI, VIX
- **Interpretation:** Spillover >30% = markets tightly coupled, diversification less effective

### Trade Generation

Every regime state maps to specific trades with:

- **Instruments** (JGB futures, FX, bond ETFs)
- **Regime condition** (ensemble probability threshold)
- **Edge source** (which model provides the signal)
- **Entry/exit signals** (concrete trigger + target/stop)
- **Failure scenario** (what kills the thesis)
- **Sizing method** (vol-target or DV01-neutral)
- **Conviction** (0-1, derived from model consensus)

Trade categories: **Rates** (JGB shorts, curve steepeners, butterflies, liquidity premium) | **FX** (carry, trend) | **Volatility** (straddles, skew) | **Cross-Asset** (relative value, diversification)

## BOJ Policy Timeline

| Date | Event | Regime Tag |
|------|-------|-----------|
| Apr 2013 | Kuroda QQE launch | QQE |
| Oct 2014 | QQE expansion (Halloween surprise) | QQE |
| Jan 2016 | Negative Interest Rate Policy | NIRP |
| Sep 2016 | Yield Curve Control introduced | YCC |
| Jul 2018 | YCC flexibility (forward guidance) | YCC |
| Dec 2022 | YCC band widened to +/-0.50% | YCC exit |
| Jul 2023 | YCC flexibility (+1.0% effective cap) | YCC exit |
| Oct 2023 | 1.0% reference (soft cap removed) | YCC exit |
| Mar 2024 | BOJ exits NIRP and YCC formally | Post-YCC |

All charts overlay these events as red vertical dashed lines for policy-model alignment validation.

## Dashboard (5 Pages)

The Streamlit dashboard (`app.py`) provides five interactive pages:

1. **Overview & Data** — KPI metrics, sovereign yields + VIX chart, FX + equity chart, actionable spread/trend/vol insights
2. **Yield Curve Analytics** — PCA explained variance + loadings heatmap + score time series, liquidity metrics, Nelson-Siegel factor evolution
3. **Regime Detection** — Ensemble probability gauge + time series, Markov smoothed probabilities, PELT structural breakpoints, permutation entropy signal, GARCH conditional volatility
4. **Spillover & Info Flow** — Granger causality table, transfer entropy heatmap, Diebold-Yilmaz spillover index + net directional bars, DCC correlations, carry-to-vol ratio
5. **Trade Ideas** — Regime-conditional trade cards with conviction scores, filterable by category/conviction, CSV export

## Data Pipeline

Three-tier fallback for robustness:

```
FRED API (premium, full history) → yfinance (free, FX/equity/ETFs) → MOF Japan (JGB yields) → Synthetic (reproducible demo)
```

- **Caching:** In-memory dict + disk parquet + Streamlit `@st.cache_data` (TTL: 15 min)
- **Pre-warming:** All caches populated on first load for instant page switching
- **Unification:** Rates + market data outer-joined on business dates, forward-filled

## Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Regime models | 4-model ensemble | No single model reliable enough; ensemble reduces false positives |
| PCA input | Daily yield changes | Stationarity requirement; levels have unit root |
| PCA scaling | Covariance (no z-score) | Preserves bps-scale meaning across tenors |
| DCC method | GARCH + EWMA (lambda=0.94) | Avoids fragile bivariate DCC optimization |
| TE discretization | Quantile bins (terciles) | Robust to outliers; interpretable (down/flat/up) |
| Spillover VAR | VAR(4), 10-step horizon | Standard in Diebold-Yilmaz (2012) literature |
| Entropy | Permutation (order=3, window=120) | Early warning signal; complexity measure |
| Trade sizing | Vol-target (5 bps/day) or DV01-neutral | Risk parity across trade legs |
| Dashboard theme | Inter font, institutional blue (#1e3a5f) | Professional desk presentation standard |
| Deployment | Docker + Render.com | Reproducible, cloud-deployable |

## Getting Started

```bash
# Clone and install
cd jgb-repricing-framework
pip install -r requirements.txt

# Launch dashboard
streamlit run app.py

# Run CLI batch analysis
python run_analysis.py

# Run tests
python -m pytest tests/ -v

# Docker deployment
docker build -t jgb-repricing .
docker run -p 7860:7860 jgb-repricing
```

### Optional: FRED API Key

For full historical data access, set your FRED API key in the dashboard sidebar or as an environment variable:

```bash
export FRED_API_KEY=your_key_here
```

Without a FRED key, the dashboard falls back to yfinance + synthetic data.

## Development Process

- **13 iterative commits** showing genuine development progression — from initial framework to institutional UX refinement
- **Docker containerized** with proper `src/` module structure
- **Cache pre-warming** optimization for production-level performance
- **Graceful degradation** across data sources (FRED -> yfinance -> MOF -> synthetic)

## Limitations & Next Steps

- MOF Japan CSV endpoint may have availability gaps; FRED requires API key for full JGB tenor coverage
- GARCH DCC uses EWMA proxy instead of full MLE bivariate optimization (stability tradeoff)
- Transfer entropy uses simple histogram binning; could upgrade to KDE or conditional TE (IDTxl)
- **Future:** Add Japan sovereign CDS spreads as market-implied trust/credibility metric
- **Future:** Japanese language NLP on BOJ minutes for hawkishness scoring
- **Future:** Real-time execution layer via broker API integration

## AI Collaboration

AI tools were used as pair-programming partners for scaffolding, debugging, and code review. All analytical decisions — model selection, parameter choices, thesis formulation, trade logic — were my own. See [`docs/ai_collaboration_log.md`](docs/ai_collaboration_log.md) for the structured collaboration log.
