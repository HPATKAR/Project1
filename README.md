# JGB Repricing Framework

Regime detection and trade generation for the Japanese Government Bond market.

![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![CI](https://github.com/HPATKAR/Project1/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

**Author:** Heramb Patkar — Purdue University, Daniels School of Business
**Course:** MGMT 69000 — Mastering AI for Finance (Prof. Yanjun Zhang)
**Methodology:** Built using the [DRIVER Framework](product/README.md) — a structured product-development workflow (Define → Represent → Implement → Validate → Evolve → Reflect) designed by Prof. Zhang for AI-augmented financial engineering.

## Overview

Quantitative research platform that detects regime shifts in JGB markets — from BOJ-suppressed yields to market-driven repricing — and generates tradeable strategies across rates, FX, volatility, and cross-asset portfolios. The framework combines PCA yield curve decomposition, a four-model ensemble regime detector, VAR-based spillover analysis, and transfer entropy networks into a unified analytical pipeline. Output is delivered through an interactive Streamlit dashboard and structured trade cards with explicit failure scenarios.

## Architecture

| Module | Description |
|--------|-------------|
| `src/yield_curve/` | PCA decomposition, Nelson-Siegel fitting, liquidity metrics, ACM term premium |
| `src/regime/` | Markov-Switching, HMM, Permutation Entropy, GARCH + PELT, weighted ensemble |
| `src/spillover/` | Diebold-Yilmaz FEVD, DCC-GARCH, Granger causality, Transfer Entropy |
| `src/fx/` | Carry-to-vol ratio, rate differentials, positioning indicators |
| `src/strategy/` | Regime-conditional trade cards, DV01-neutral sizing, backtester |
| `src/data/` | FRED/yfinance/MOF Japan data layer with three-tier fallback |
| `tests/` | Yield curve, regime, spillover, and data pipeline validation |

<details>
<summary>Full directory tree</summary>

```
jgb-repricing-framework/
|-- app.py                          # Streamlit dashboard
|-- run_analysis.py                 # CLI batch analysis runner
|-- requirements.txt                # Core dependencies
|-- Dockerfile                      # Docker deployment (Python 3.11-slim)
|-- render.yaml                     # Render.com cloud deployment config
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

</details>

## Methods

### PCA Yield Curve Decomposition

Principal Component Analysis on daily yield changes across tenors (2Y, 5Y, 7Y, 10Y, 20Y, 30Y):

| Component | Interpretation | Typical Variance | Reference |
|-----------|---------------|-----------------|-----------|
| PC1 | **Level** (parallel shift) | 60-80% | Litterman & Scheinkman (1991) |
| PC2 | **Slope** (twist) | 10-20% | Litterman & Scheinkman (1991) |
| PC3 | **Curvature** (butterfly) | 5-10% | Litterman & Scheinkman (1991) |

- **Input:** Daily yield changes (not levels) for stationarity
- **Scaling:** Covariance-based (preserves bps-scale economic meaning)
- **Rolling PCA:** 252-day window tracks time-varying factor structure
- **Validation:** Automated heuristic checks sign-change patterns and explained variance ratios against Litterman-Scheinkman (1991) factor benchmarks

### 4-Model Ensemble Regime Detection

| Model | Package | Signal | Reference |
|-------|---------|--------|-----------|
| Markov-Switching | statsmodels | Smoothed state probabilities | Hamilton (1989) |
| Gaussian HMM | hmmlearn | Viterbi state sequence | Rabiner (1989) |
| Permutation Entropy | antropy | Z-score threshold signal | Bandt & Pompe (2002) |
| GARCH(1,1) + PELT | arch + ruptures | Vol-regime breakpoints | Bollerslev (1986) |

- **Ensemble:** Min-max normalize each signal to [0,1], weighted average (default: 25% each)
- **Output:** Single probability series — 0 = BOJ-suppressed, 1 = market-driven repricing
- **Conviction thresholds:** >0.7 STRONG | 0.5-0.7 MODERATE | 0.3-0.5 TRANSITION | <0.3 SUPPRESSED

### Cross-Asset Spillover Analysis

| Method | What It Measures | Key Output |
|--------|-----------------|------------|
| Diebold-Yilmaz (2012) | Forecast error variance decomposition | Total spillover % + net directional flows |
| DCC-GARCH | Time-varying correlations | Crisis-driven correlation spikes |
| Granger Causality | Lagged predictability | Significant cause-effect pairs with optimal lag |
| Transfer Entropy | Directional information flow (Schreiber, 2000) | Asymmetric leader/follower network |

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

## Dashboard

The Streamlit dashboard (`app.py`) provides six analytical pages:

1. **Overview & Data** — KPI metrics, sovereign yields + VIX chart, FX + equity chart, actionable spread/trend/vol insights, Japan sovereign credit ratings and BOJ credibility events timeline
2. **Yield Curve Analytics** — PCA explained variance + loadings heatmap + score time series, PCA factor validation against Litterman-Scheinkman (1991), liquidity metrics, Nelson-Siegel factor evolution
3. **Regime Detection** — Ensemble probability gauge + time series, Markov smoothed probabilities, PELT structural breakpoints, permutation entropy signal, GARCH conditional volatility
4. **Spillover & Info Flow** — Granger causality table, transfer entropy heatmap, Diebold-Yilmaz spillover index + net directional bars, DCC correlations, carry-to-vol ratio
5. **Trade Ideas** — Regime-conditional trade cards with conviction scores, filterable by category/conviction, CSV export
6. **AI Q&A** — Conversational interface with full analysis context (regime state, PCA, spillover, carry, liquidity, trade ideas) injected as system prompt

## Data Pipeline

Three-tier fallback for robustness:

```
FRED API (full history) → yfinance (free, FX/equity/ETFs) → MOF Japan (JGB yields) → Synthetic (reproducible demo)
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

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
python -m pytest tests/ -v
```

<details>
<summary>Docker & FRED configuration</summary>

```bash
# Docker deployment
docker build -t jgb-repricing .
docker run -p 7860:7860 jgb-repricing

# Optional: FRED API key for full historical data
export FRED_API_KEY=your_key_here
```

Without a FRED key, the dashboard falls back to yfinance + synthetic data.

</details>

## Development Process

- **21+ iterative commits** spanning architecture design, data pipeline hardening, 4-model regime ensemble, spillover analysis, trade generation logic, dashboard UX, Purdue Daniels branding, and CI/CD pipeline
- **DRIVER Framework** applied end-to-end: thesis formulation (Define), module architecture (Represent), full implementation across 6 source packages (Implement), 37 automated tests (Validate), iterative refinement from professor feedback (Evolve), and structured reflection on AI collaboration patterns (Reflect)
- **Docker containerized** with proper `src/` module structure and Render.com cloud deployment
- **Graceful degradation** across four data source tiers (FRED → yfinance → MOF → synthetic)
- **Cache pre-warming** optimization for production-level dashboard performance

## Known Limitations

- MOF Japan CSV endpoint may have availability gaps; FRED requires API key for full JGB tenor coverage
- GARCH DCC uses EWMA proxy instead of full MLE bivariate optimization (stability tradeoff)
- Transfer entropy uses simple histogram binning; could upgrade to KDE or conditional TE (IDTxl)

## References

1. Litterman, R. & Scheinkman, J. (1991). Common factors affecting bond returns. *Journal of Fixed Income*, 1(1), 54-61.
2. Hamilton, J.D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357-384.
3. Diebold, F.X. & Yilmaz, K. (2012). Better to give than to receive: Predictive directional measurement of volatility spillovers. *International Journal of Forecasting*, 28(1), 57-66.
4. Bandt, C. & Pompe, B. (2002). Permutation entropy: A natural complexity measure for time series. *Physical Review Letters*, 88(17), 174102.
5. Schreiber, T. (2000). Measuring information transfer. *Physical Review Letters*, 85(2), 461-464.
6. Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.

## Disclaimer

This framework is provided for research and educational purposes only. Nothing herein constitutes investment advice, a solicitation, or a recommendation to buy or sell any security. The authors accept no liability for any losses arising from use of this software.

## License

MIT. See [LICENSE](LICENSE).

## Contact

Heramb Patkar — Purdue University, Daniels School of Business | [GitHub](https://github.com/HPATKAR)
MGMT 69000: Mastering AI for Finance — Prof. Yanjun Zhang

AI tools were used as pair-programming partners for scaffolding, debugging, and code review. All analytical decisions — model selection, parameter choices, thesis formulation, trade logic — were my own. See [`docs/ai_collaboration_log.md`](docs/ai_collaboration_log.md) for the structured collaboration log.
