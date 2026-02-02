# JGB Repricing Framework: Regime Detection & Systematic Macro Trading

## The Problem

Japan's bond market is undergoing a structural regime shift from BOJ-suppressed yields (ZIRP/NIRP/YCC era, 2013-2024) to market-driven pricing. This creates:

1. **Detection problem**: Identifying structural change early — before consensus — using quantitative signals across yield curve dynamics, volatility, liquidity, and information flow
2. **Translation problem**: Converting regime detection into executable trades across rates, FX, volatility, and cross-asset portfolios
3. **Spillover problem**: Mapping how JGB repricing transmits to USTs, Bunds, EM bonds, USD/JPY, and global duration risk

No integrated open-source framework exists for this. The building blocks exist across 15+ Python packages, but the integration layer — connecting regime detection to trade generation — must be built.

## Success Looks Like

- **Regime detection dashboard**: Real-time probability of being in "suppressed" vs. "market-driven" vs. "crisis" regime, updated with each data point
- **Signal generation**: Quantitative signals from 8 research lenses that feed trade ideas with explicit regime conditions, edge source, and failure scenarios
- **Backtested strategies**: Validated against historical BOJ policy shift dates (2013 QQE, 2016 YCC, 2022-2024 YCC exit)
- **Cross-asset information flow map**: Transfer entropy network showing lead-lag relationships JGB <-> UST <-> Bund <-> USD/JPY <-> VIX
- **Executable trade recommendations**: Rates / FX / Vol / Cross-asset hedges with sizing logic

## Building On (Existing Foundations)

### Regime Detection
- **statsmodels.MarkovRegression** — Hamilton (1989) Markov-switching for 2-state yield regime models
- **hmmlearn.GaussianHMM** — Multivariate HMM for joint JGB + FX + vol regime detection
- **ruptures** (PELT algorithm) — Structural break detection in yield/liquidity series
- **antropy** — Permutation entropy for complexity-based regime shifts
- **arch** — GARCH family (EGARCH, GJR-GARCH) for volatility regime detection

### Yield Curve & Term Premium
- **scikit-learn PCA** — Level/slope/curvature decomposition of JGB curve
- **nelson_siegel_svensson** — Parametric curve fitting, parameter evolution tracking
- **Custom ACM implementation** — Adrian-Crump-Moench term premium via statsmodels VAR + PCA

### Information Flow
- **IDTxl** — Multivariate transfer entropy with conditional testing (best for financial TE)
- **PyInform** — Faster discrete transfer entropy for screening
- **statsmodels Granger causality** — Linear baseline

### Cross-Asset Spillover
- **statsmodels VAR + custom FEVD** — Diebold-Yilmaz connectedness index
- **arch DCC-GARCH** — Time-varying correlation JGB/UST/Bund

### Systematic Trading
- **pysystemtrade** — Rob Carver's futures framework (supports JGB futures, FX, carry/trend)
- **vectorbt** — Fast signal backtesting
- **finmarketpy** — CueMacro carry trade framework (Saeed Amen)
- **Riskfolio-Lib** — Portfolio construction across trade legs

### Data
- **FRED / fredapi** — US rates, Japan 10Y, macro series
- **MOF Japan** — JGB reference yields (all tenors, daily)
- **BOJ Statistics** — BOJ balance sheet, operations, call rate
- **yfinance** — FX spot, JGB/UST ETFs, equity indices
- **cot_reports** — CFTC COT positioning (JPY futures)

## The Unique Part

What we're building that doesn't exist:

1. **Unified regime detection engine** — Combines Markov-switching, structural breaks, entropy, and GARCH into a single regime probability output. No package does this.
2. **BOJ policy function model** — Quantitative model of BOJ reaction function with structural break detection at policy shift dates. Custom build.
3. **JGB -> global spillover map** — Transfer entropy + Diebold-Yilmaz network showing real-time information flow from JGBs to global markets.
4. **Regime-conditional trade generator** — Maps regime states to specific tradeable expressions with sizing, entry/exit rules, and failure scenarios.
5. **Historical validation against known BOJ shifts** — Backtested against 2013 QQE launch, 2016 NIRP/YCC, 2022 YCC widening, 2024 YCC exit.

## Tech Stack

- **Language:** Python 3.13
- **Core:** numpy, pandas, scipy, matplotlib
- **Regime Detection:** statsmodels, hmmlearn, ruptures, antropy, arch
- **Yield Curve:** scikit-learn (PCA), nelson-siegel-svensson, QuantLib-Python
- **Information Flow:** IDTxl, pyinform, statsmodels (Granger)
- **Spillover:** statsmodels (VAR/FEVD), arch (DCC-GARCH)
- **Strategy:** pysystemtrade, vectorbt, Riskfolio-Lib
- **Data:** fredapi, yfinance, cot_reports, beautifulsoup4 (BOJ scraping)
- **Testing:** pytest
- **Visualization:** matplotlib, plotly

## Key Historical BOJ Policy Shift Dates (for backtesting)

| Date | Event | Market Impact |
|------|-------|---------------|
| 2013-04-04 | Kuroda QQE launch | Massive rally then VaR shock (May 2013) |
| 2016-01-29 | Negative Interest Rate Policy | Bull-flattening |
| 2016-09-21 | Yield Curve Control introduced | 10Y pinned at ~0% |
| 2022-12-20 | YCC band widened to +/-0.50% | 10Y spiked 20bp in a day |
| 2023-07-28 | YCC "flexibility" (+1.0% effective cap) | Gradual sell-off |
| 2023-10-31 | 1.0% "reference" (soft cap removed) | 10Y hit 0.95% |
| 2024-03-19 | BOJ exits NIRP and YCC formally | Normalization begins |

## Research Lenses → Trade Output

Every research lens produces tradeable output:

| Lens | Primary Signal | Trade Expression |
|------|---------------|-----------------|
| Yield Curve PCA | PC loadings shift, term premium release | Curve steepeners, duration shorts |
| Liquidity/Microstructure | Basis blowout, Amihud spike | Basis trades, liquidity premium capture |
| BOJ Policy Function | Reaction function break, absorption ratio | Pre-positioning around meetings |
| Structural Change | Regime probability > threshold | Directional rates, vol buying |
| Transfer Entropy | Information flow reversal JGB→UST | Cross-market relative value |
| FX & Carry | JPY funding cost rise, carry-to-vol collapse | Short JPY carry, long JPY vol |
| Global Spillovers | FEVD shock transmission spike | UST/Bund hedges, EM duration reduction |
| Positioning & Flows | CTA trend signal flip, ETF outflows | Contrarian or momentum overlay |

## Open Questions

1. Data access: Bloomberg vs. free alternatives — which tenors/instruments are available via free APIs?
2. Intraday data: Do we need tick data for realized vol and microstructure, or is daily sufficient for Phase 1?
3. Execution layer: Is this research-only or should we build toward API-connected execution?
4. Japanese language NLP: Do we want to parse BOJ minutes in Japanese for hawkishness scoring?
