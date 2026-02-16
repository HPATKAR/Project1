# AI Collaboration Log

**Project:** JGB Repricing Framework
**Course:** MGMT 69000: Mastering AI for Finance
**Author:** Heramb Patkar

## Collaboration Philosophy

AI was used as a **pair-programming partner and research accelerator** — providing scaffolding, code generation, debugging, and architecture feedback — while all analytical decisions, model selection, thesis formulation, trade logic, and interpretation remained my own. Every AI-generated suggestion was reviewed, tested against domain knowledge, and modified before inclusion.

## Session Log

### Session 1: Project Architecture & DRIVER Framework
- **AI Tool:** Claude Code (via DRIVER methodology)
- **What I asked:** `/driver:define` — establish macro thesis, research lenses, and system architecture for JGB repricing analysis.
- **What I decided:**
  - Chose 8 research lenses (yield curve, liquidity, BOJ policy, structural change, TE, carry, spillover, positioning)
  - Selected specific packages for each lens (statsmodels for Markov, hmmlearn for HMM, etc.)
  - Designed the `src/` module hierarchy: `yield_curve/`, `regime/`, `spillover/`, `fx/`, `strategy/`
- **Verification:** Validated architecture against published quant research workflows (Diebold-Yilmaz 2012, Hamilton 1989).

### Session 2: Data Pipeline Design
- **AI Tool:** Claude Code
- **What I asked:** Build a unified data layer with three-tier fallback (FRED -> yfinance -> MOF Japan -> synthetic).
- **What I decided:**
  - FRED as primary source for yield history (best coverage for JP_10Y)
  - yfinance for FX, equities, and bond ETFs (free, no API key required)
  - MOF Japan CSV endpoint for actual JGB yields across tenors
  - Synthetic data as guaranteed fallback (seed=42 for reproducibility)
- **What I rejected:** AI suggested using Bloomberg API — not available in academic setting.
- **Verification:** Tested all four fallback paths independently; confirmed data alignment on business dates.

### Session 3: PCA Implementation
- **AI Tool:** Claude Code
- **What I asked:** Implement PCA yield curve decomposition with full-sample and rolling-window modes, plus automated factor interpretation.
- **What I decided:**
  - PCA on daily yield changes (not levels) for stationarity
  - Covariance-based PCA (no standardization) to preserve bps-scale economic meaning
  - Rolling window of 252 trading days (1 year) for time-varying analysis
  - Automated heuristic: sign-change counting to identify level/slope/curvature
- **What I rejected:** AI initially suggested correlation-based PCA (z-scored). I overrode this — in fixed income, the relative magnitude of moves across tenors carries economic meaning that z-scoring destroys.
- **Verification:** Validated PC1/2/3 loadings against Litterman-Scheinkman (1991) factor structure. Added `interpret_pca()` and `validate_pca_factors()` functions.

### Session 4: Regime Detection Models
- **AI Tool:** Claude Code
- **What I asked:** Implement four independent regime detection models and an ensemble combiner.
- **What I decided:**
  - Markov-Switching (2 regimes, switching variance) for volatility clustering
  - Gaussian HMM on multivariate features (JP_10Y + USDJPY + VIX) for cross-asset co-movement
  - Permutation entropy (order=3, window=120) as early warning signal
  - GARCH(1,1) + PELT for volatility regime breakpoints
  - Equal-weighted ensemble (25% each) as default; NaN-aware averaging
- **What I rejected:**
  - AI suggested 3-state Markov model — I reduced to 2 states for parsimony (suppressed vs repricing)
  - AI suggested EGARCH — I chose standard GARCH(1,1) for stability and interpretability
- **Verification:** Backtested regime signals against known BOJ policy dates. Confirmed permutation entropy fires earliest (1-2 weeks before Markov).

### Session 5: Spillover & Information Flow
- **AI Tool:** Claude Code
- **What I asked:** Implement Diebold-Yilmaz VAR-FEVD spillover, DCC-GARCH correlations, Granger causality, and transfer entropy.
- **What I decided:**
  - VAR(4) with 10-step horizon (standard DY2012 specification)
  - DCC via GARCH + EWMA (lambda=0.94) instead of full MLE bivariate DCC (stability tradeoff)
  - Transfer entropy with quantile binning (terciles: down/flat/up)
  - Granger causality: lags 1-5, optimal lag by minimum p-value
- **What I rejected:** AI suggested bivariate DCC-GARCH via arch package's DCC class. In testing, it frequently failed to converge on this dataset. Switched to EWMA proxy which is numerically stable and produces similar dynamics.
- **Verification:** Checked spillover index against BIS working papers on JGB contagion channels.

### Session 6: Trade Generation System
- **AI Tool:** Claude Code
- **What I asked:** Design a regime-conditional trade generation system with explicit failure scenarios.
- **What I decided:**
  - TradeCard dataclass with mandatory `failure_scenario` field
  - Four categories: rates, FX, volatility, cross-asset
  - Conviction scoring: base + regime_prob * weight + model-specific signal * weight
  - DV01-neutral sizing for spread/butterfly trades
  - Vol-target sizing (5 bps/day) for directional trades
- **Key design choice:** Every trade card must answer "What would kill this thesis?" — this was my requirement, not AI's suggestion. It forces critical thinking beyond entry signals.
- **Verification:** Reviewed trade logic against professional rates desk playbooks.

### Session 7: Dashboard UX & Institutional Styling
- **AI Tool:** Claude Code
- **What I asked:** Build a 5-page Streamlit dashboard with institutional desk styling, inline actionable insights, and cache pre-warming.
- **What I decided:**
  - Inter font, muted institutional palette (#1e3a5f primary)
  - Every chart has a data-driven insight paragraph (not generic descriptions)
  - Page verdicts synthesize findings into actionable conclusions
  - Cache pre-warming on first load for instant page switches
  - TTL=900s + max_entries=3 for resource management
- **Iterations:** 6 commits dedicated to UX refinement (from generic captions to chart-specific insights to institutional styling)
- **Verification:** Tested on Render.com cloud deployment.

### Session 8: Post-Feedback Improvements
- **AI Tool:** Claude Code (multi-model: Claude Opus + GPT-5.2 via PAL)
- **What I asked:** Apply professor's feedback — add README, PCA validation documentation, credit ratings/trust metrics, and AI collaboration log.
- **What I decided:**
  - README as engineering documentation (not assignment brief): architecture, methods, decisions, limitations
  - PCA validation: automated factor checks against Litterman-Scheinkman, displayed in dashboard
  - Credit ratings: static reference table (Moody's A1, S&P A+, Fitch A, R&I AA+) with credibility events
  - This collaboration log: structured, honest, decision-focused
- **Verification:** All existing tests pass after changes. Dashboard renders correctly with new sections.

### Session 9: Purdue Daniels Theme Redesign, Navigation, and New Pages
- **AI Tool:** Claude Code (primary) + **GPT-5.2 via PAL MCP** (targeted CSS design guidance, color palette validation)
- **What I asked:** Redesign the Streamlit UI to match Purdue Daniels School of Business branding; expand dashboard from 5 to 8 pages; add full-bleed footer navigation; add AI Q&A page; create two About pages (student + instructor).
- **What I decided:**
  - Adopted Purdue Daniels theme tokens: Black `#000000`, Boilermaker Gold `#CFB991`, Aged `#8E6F3E`, DM Sans typography; removed the prior institutional blue (`#1e3a5f`) styling entirely
  - Implemented a full-bleed footer with 5-column grid (branding, navigate, about, connect, source code) and persistent navigation links to reduce friction when switching among 8 pages
  - Added AI Q&A page powered by OpenAI/Anthropic endpoints with full analysis context injection (regime state, PCA, spillover, carry, trade ideas)
  - Built About pages with hero banners, card-based layouts, and profile photo integration
  - Used GPT-5.2 via PAL specifically for CSS best practices (sidebar collapse button removal across Streamlit versions, institutional-grade table styling, Purdue palette gradient validation)
- **What AI generated vs. what I owned:**
  - AI proposed CSS layout patterns, footer structure, About page card designs, and draft copy; I selected the final visual design, verified brand color usage against official Purdue guidelines, and integrated components into the existing app architecture
  - AI suggested the AI Q&A system prompt structure; I defined the allowed behaviors, disclaimers, and failure modes (no confidential data, no investment advice framing, graceful API key handling)
  - AI recommended gradient values for the footer gold bar; I rejected the initial Rush `#DAAA00` endpoint as too yellow and selected an Aged-to-Gold sweep that matches Purdue brand standards
- **What I rejected:**
  - AI initially used Rush (#DAAA00) in footer gradient, which looked off-brand on wide-gamut displays; replaced with Aged-to-Boilermaker Gold gradient
  - AI suggested hiding only the sidebar collapse button via CSS; I opted to hide the entire Streamlit header for a cleaner institutional look
- **Verification:** Manually validated all 8 pages render under the new theme; checked footer navigation links across all pages; confirmed AI Q&A page handles missing API keys gracefully; verified profile photo renders correctly via base64 embedding.

### Session 10: Typography Unification & AI Q&A Deep Grounding
- **AI Tool:** Claude Code (primary) + **GPT-5.2 via PAL MCP** (AI Q&A grounding audit and improvement recommendations)
- **What I asked:** (1) Optimize fonts across all dashboard pages for visual uniformity. (2) Audit the AI Q&A page against the professor's requirement of "a chat interface grounded in the case study" and implement improvements.
- **What I decided:**
  - Created a 13-step CSS custom property type scale (`--fs-micro` through `--fs-hero`) with letter-spacing and font-stack tokens, consolidating 28 scattered font sizes into a maintainable design system
  - Enriched AI Q&A context injection from 7 summary metrics to 12 deep analytical sections: added PCA loadings (not just variance), Nelson-Siegel curve factors, spillover directional edges (top 5), DCC correlations with deviation flags, Granger causality significant pairs, GARCH conditional vol with percentile, entropy early-warning signals, structural break dates, curve slopes, trade failure scenarios
  - Rewrote system prompt to enforce thesis-grounded answers: BOJ suppression → repricing → trade implications, with hard rules requiring citation of 2+ specific metrics and prohibition on inventing data
  - Replaced generic topic cards with case-study-aligned prompts: "Regime Call," "PCA Factors," "Spillover Chain," "Trade Thesis"
- **What AI generated vs. what I owned:**
  - GPT-5.2 via PAL provided the audit framework (P0/P1/P2 priorities, system prompt tightening recommendations); I selected which improvements to implement and designed the final context structure
  - AI generated the enriched `_build_analysis_context()` and tightened system prompt; I verified all data paths connect to existing cached functions and validated the analytical accuracy of injected content
- **Verification:** All 37 tests pass. Context injection exercises all 12 analytical modules without error.

## AI Usage Summary

| Category | AI Role | My Role |
|----------|---------|---------|
| Thesis & Research Lenses | Not used | Formulated from BOJ policy analysis and market structure |
| Architecture | Scaffolding and module structure | Final design decisions on separation of concerns |
| Model Selection | Implementation assistance | Chose models based on fixed-income literature |
| Parameter Choices | Suggested defaults | Overrode where domain knowledge differed (PCA scaling, GARCH spec, regime count) |
| Trade Logic | Code generation | Designed conviction scoring, failure scenarios, sizing methods |
| Dashboard | Layout and chart generation | Designed page flow, insight narratives, UX refinements |
| Testing | Test structure | Wrote all test cases with domain-appropriate assertions |
| Documentation | Drafting assistance | Accuracy review, technical claims verification |

## Key Decisions Where I Overrode AI Suggestions

1. **Covariance PCA over Correlation PCA:** AI suggested z-scoring all tenors. I kept covariance-based PCA because in fixed income, the relative magnitude of yield moves across tenors is economically meaningful.
2. **2-state Markov over 3-state:** AI proposed calm/stress/crisis. I simplified to suppressed/repricing for clarity and parsimony — the data doesn't support a reliable third state.
3. **EWMA DCC over MLE DCC:** AI implemented full bivariate DCC. It failed to converge on 40% of test runs. I switched to EWMA (lambda=0.94, RiskMetrics standard) for numerical stability.
4. **Mandatory failure scenarios:** AI-generated trade cards initially lacked risk analysis. I added the `failure_scenario` field as a required dataclass attribute, ensuring every trade answers "what kills this thesis?"
5. **FRED primary over yfinance:** AI defaulted to yfinance for all data. I restructured the pipeline to prioritize FRED for yield data (better JGB coverage) with yfinance as fallback.
