# Validation: Data and Code Ownership

This document maps every validation mechanism built into the JGB Repricing Framework, where to find it, and what it proves.

## 1. Regime Detection Validation

**Where:** Dashboard > Regime Detection page + `src/regime/ensemble.py`

The function `validate_ensemble_vs_boj()` tests the four-model ensemble against every known BOJ policy event (Dec 2022 band widening, Jul 2023 band widening, Oct 2023 YCC flexibility, Mar 2024 YCC exit, etc.).

For each event it records:
- Whether the ensemble probability spiked above the 0.6 detection threshold within +/- 10 trading days
- The peak probability reached
- The lead/lag offset in trading days (negative = early detection, positive = delayed)
- Overall detection rate across all events

**Why it matters:** If the model cannot detect known historical events, it has no predictive value. Lead/lag analysis confirms whether detection is early enough to be tradeable.

## 2. PCA Factor Validation

**Where:** Dashboard > Yield Curve Analytics page + `src/yield_curve/pca.py`

The function `validate_pca_factors()` checks the PCA decomposition against the Litterman-Scheinkman (1991) factor structure:

- PC1 (Level) should explain 60-80% of variance with uniform loadings across tenors
- PC2 (Slope) should show a sign change between short and long tenors
- PC3 (Curvature) should show hump-shaped or U-shaped loadings
- Cumulative variance explained by PC1-PC3 should exceed 90%

Each check returns PASS or FAIL with the actual value. Results are displayed in the dashboard with color-coded metrics.

**Why it matters:** If PCA factors do not match the classical fixed-income factor structure, the decomposition is unreliable and any trade ideas based on factor scores would be misinformed.

## 3. Performance Review Page

**Where:** Dashboard > Diagnostics > Performance Review + `src/pages/performance_review.py`

A dedicated diagnostics page that evaluates model performance:

- **Accuracy metrics:** Accuracy, precision, recall, false positive rate tracked over time
- **Confusion matrix:** True positives, false positives, true negatives, false negatives visualised
- **Ensemble probability vs actual outcomes:** Charts predicted regime probability against realised market moves
- **Model agreement analysis:** Measures how often the four regime models (Markov, HMM, Entropy, GARCH) agree
- **ML regime predictor:** Walk-forward RandomForest model with feature importance, providing an independent check on the ensemble
- **Auto-generated improvement suggestions:** Rule-based recommendations when metrics fall below thresholds
- **Export:** Full results exportable as PDF (three profiles) or CSV

**Why it matters:** This page is the central validation hub. It answers: does the model work, where does it fail, and what should be improved?

## 4. Data Source Validation

**Where:** `src/data/data_store.py` + `src/data/config.py`

- FRED series IDs are defined in `config.py` and were manually verified against the St. Louis Fed website
- MOF Japan CSV parsing handles three encodings (Shift-JIS, UTF-8, CP932) with tenor column matching validated against the official schema
- yfinance yield indices (^TNX, ^FVX, ^TYX) include a scaling check: if max value exceeds 20, the series is divided by 10 to correct for historical x10 reporting
- The 4-tier fallback chain (FRED > yfinance > MOF Japan > synthetic) ensures the dashboard never crashes on missing data, with logging at each fallback step

## 5. ACM Term Premium Validation

**Where:** Dashboard > Yield Curve Analytics page + `src/yield_curve/term_premium.py`

Term premium estimates are computed using the Adrian-Crump-Moench (2013) affine term structure methodology and are compared against NY Fed published reference ranges. The dashboard displays the time series alongside the reference bands.

**Why it matters:** Term premium is a key input for trade ideas. If estimates fall outside known reference ranges, the model is miscalibrated.

## 6. Test Suite

**Where:** `tests/` directory

- **54 unit tests** across 5 test files:
  - `test_yield_curve.py` - PCA decomposition, Nelson-Siegel fitting, liquidity metrics
  - `test_regime.py` - Markov-switching, HMM, entropy, GARCH, ensemble
  - `test_spillover.py` - Diebold-Yilmaz, Granger causality, transfer entropy
  - `test_equity_spillover.py` - Nikkei integration tests
  - `test_data.py` - Data pipeline, fallback chain, simulated data generation
- **CI/CD:** GitHub Actions runs the full test suite on every push to main

Run locally:
```bash
pytest tests/ -v
```

## 7. Code Quality Audit

Performed before final submission:

- Zero `print()` statements in production code (all replaced with `logging`)
- Zero hardcoded API keys or credentials in any source file
- `.gitignore` covers: `.env`, `.streamlit/secrets.toml`, `__pycache__/`, `*.pyc`, `*.parquet`, `.venv/`
- No `TODO`, `FIXME`, or `HACK` comments left in source
- All `except` clauses are typed (`except Exception`), no bare `except:`
- MIT License with correct copyright year (2025-2026)

## 8. Structural Validation via BOJ Event Overlay

**Where:** Every time-series chart in the dashboard

All charts overlay known BOJ policy dates (vertical dashed lines with annotations). This allows visual confirmation that model signals (regime probability spikes, volatility surges, spillover jumps) align with real-world policy events.

---

## Summary

| Validation Type | Location | What It Checks |
|----------------|----------|----------------|
| Regime vs BOJ events | Regime page, `ensemble.py` | Detection rate + lead/lag on known dates |
| PCA factor structure | Yield Curve page, `pca.py` | Litterman-Scheinkman (1991) benchmarks |
| Performance review | Diagnostics page | Accuracy, precision, recall, confusion matrix |
| Data sources | `data_store.py` | Encoding, scaling, fallback chain |
| Term premium | Yield Curve page | NY Fed reference ranges |
| Unit tests | `tests/` (54 tests) | All analytical modules |
| Code audit | Pre-submission | No secrets, no print(), proper logging |
| BOJ overlays | All charts | Visual alignment of signals with events |

---

*Heramb S. Patkar | MS Finance, Purdue University | Daniels School of Business*
