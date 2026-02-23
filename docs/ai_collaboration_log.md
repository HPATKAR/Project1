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
- **AI Tool:** Claude Code (multi-model: Claude Opus + GPT-4o via PAL MCP)
- **What I asked:** Apply professor's feedback — add README, PCA validation documentation, credit ratings/trust metrics, and AI collaboration log.
- **What I decided:**
  - README as engineering documentation (not assignment brief): architecture, methods, decisions, limitations
  - PCA validation: automated factor checks against Litterman-Scheinkman, displayed in dashboard
  - Credit ratings: static reference table (Moody's A1, S&P A+, Fitch A, R&I AA+) with credibility events
  - This collaboration log: structured, honest, decision-focused
- **Verification:** All existing tests pass after changes. Dashboard renders correctly with new sections.

### Session 9: Purdue Daniels Theme Redesign, Navigation, and New Pages
- **AI Tool:** Claude Code (primary) + **GPT-4o via PAL MCP** (targeted CSS design guidance, color palette validation)
- **What I asked:** Redesign the Streamlit UI to match Purdue Daniels School of Business branding; expand dashboard from 5 to 8 pages; add full-bleed footer navigation; add AI Q&A page; create two About pages (student + instructor).
- **What I decided:**
  - Adopted Purdue Daniels theme tokens: Black `#000000`, Boilermaker Gold `#CFB991`, Aged `#8E6F3E`, DM Sans typography; removed the prior institutional blue (`#1e3a5f`) styling entirely
  - Implemented a full-bleed footer with 5-column grid (branding, navigate, about, connect, source code) and persistent navigation links to reduce friction when switching among 8 pages
  - Added AI Q&A page powered by OpenAI/Anthropic endpoints with full analysis context injection (regime state, PCA, spillover, carry, trade ideas)
  - Built About pages with hero banners, card-based layouts, and profile photo integration
  - Used GPT-4o via PAL MCP specifically for CSS best practices (sidebar collapse button removal across Streamlit versions, institutional-grade table styling, Purdue palette gradient validation)
- **What AI generated vs. what I owned:**
  - AI proposed CSS layout patterns, footer structure, About page card designs, and draft copy; I selected the final visual design, verified brand color usage against official Purdue guidelines, and integrated components into the existing app architecture
  - AI suggested the AI Q&A system prompt structure; I defined the allowed behaviors, disclaimers, and failure modes (no confidential data, no investment advice framing, graceful API key handling)
  - AI recommended gradient values for the footer gold bar; I rejected the initial Rush `#DAAA00` endpoint as too yellow and selected an Aged-to-Gold sweep that matches Purdue brand standards
- **What I rejected:**
  - AI initially used Rush (#DAAA00) in footer gradient, which looked off-brand on wide-gamut displays; replaced with Aged-to-Boilermaker Gold gradient
  - AI suggested hiding only the sidebar collapse button via CSS; I opted to hide the entire Streamlit header for a cleaner institutional look
- **Verification:** Manually validated all 8 pages render under the new theme; checked footer navigation links across all pages; confirmed AI Q&A page handles missing API keys gracefully; verified profile photo renders correctly via base64 embedding.

### Session 10: Typography Unification & AI Q&A Deep Grounding
- **AI Tool:** Claude Code (primary) + **GPT-4o via PAL MCP** (AI Q&A grounding audit and improvement recommendations)
- **What I asked:** (1) Optimize fonts across all dashboard pages for visual uniformity. (2) Audit the AI Q&A page against the professor's requirement of "a chat interface grounded in the case study" and implement improvements.
- **What I decided:**
  - Created a 13-step CSS custom property type scale (`--fs-micro` through `--fs-hero`) with letter-spacing and font-stack tokens, consolidating 28 scattered font sizes into a maintainable design system
  - Enriched AI Q&A context injection from 7 summary metrics to 12 deep analytical sections: added PCA loadings (not just variance), Nelson-Siegel curve factors, spillover directional edges (top 5), DCC correlations with deviation flags, Granger causality significant pairs, GARCH conditional vol with percentile, entropy early-warning signals, structural break dates, curve slopes, trade failure scenarios
  - Rewrote system prompt to enforce thesis-grounded answers: BOJ suppression → repricing → trade implications, with hard rules requiring citation of 2+ specific metrics and prohibition on inventing data
  - Replaced generic topic cards with case-study-aligned prompts: "Regime Call," "PCA Factors," "Spillover Chain," "Trade Thesis"
- **What AI generated vs. what I owned:**
  - GPT-4o via PAL MCP provided the audit framework (P0/P1/P2 priorities, system prompt tightening recommendations); I selected which improvements to implement and designed the final context structure
  - AI generated the enriched `_build_analysis_context()` and tightened system prompt; I verified all data paths connect to existing cached functions and validated the analytical accuracy of injected content
- **Verification:** All 37 tests pass. Context injection exercises all 12 analytical modules without error.

### Session 11: CI/CD Pipeline Setup
- **AI Tool:** Claude Code (Claude Opus 4.6)
- **What I asked:** Set up a proper CI/CD pipeline with GitHub Actions so code is tested and deployable on every push, per professor's requirement.
- **What I decided:**
  - Two-workflow design: `ci.yml` (test + lint on every push/PR) and `deploy.yml` (deploy on push to main, gated behind passing tests)
  - Python 3.11 to match Dockerfile and render.yaml
  - Added `ruff` as linter (fast, modern, minimal config)
  - Deploy workflow triggers Render.com deploy hook via secret (`RENDER_DEPLOY_HOOK_URL`), keeping credentials out of code
  - CI workflow is reusable (`workflow_call`) so deploy.yml can depend on it without duplicating test steps
  - Added CI status badge to README for visibility
- **What AI generated vs. what I owned:**
  - AI generated the workflow YAML files and badge markdown; I defined the pipeline architecture (separate CI vs deploy, test-gated deployment, secret-based deploy hook)
- **Verification:** Workflow syntax validated. All 37 existing tests expected to pass in CI. Badge will render after first workflow run.

### Session 12: README Institutional Polish & LICENSE
- **AI Tool:** Claude Code (Claude Opus 4.6)
- **What I asked:** Rewrite README to institutional-grade quality (modeled after gs-quant, Qlib, Bloomberg quant-research repos). Add MIT LICENSE file. Keep Purdue Daniels affiliation framed professionally.
- **What I decided:**
  - Removed course/assignment/thesis framing from top block; repositioned Purdue affiliation to Contact section as institutional credit (how a quant paper cites a university)
  - Replaced ASCII directory tree with compact module table; moved full tree into `<details>` collapsible
  - Added Reference column to regime detection table (Hamilton 1989, Rabiner 1989, Bandt & Pompe 2002, Bollerslev 1986) and PCA table (Litterman & Scheinkman 1991)
  - Removed "Development Process" section (commit count reads as proving effort)
  - Reduced "Limitations & Next Steps" to 3 factual limitations; moved roadmap items to product-roadmap.md
  - Dashboard section lists 6 analytical pages only; removed About page descriptions
  - Added References section (6 academic citations), Disclaimer, MIT License
  - Renamed "Getting Started" to "Quickstart" (3 commands); Docker/FRED config in collapsible
  - Deleted "Dashboard theme" and "Deployment" rows from Technical Decisions (not research methodology)
  - Removed "premium" from FRED description
  - AI collaboration reduced to one sentence + link
- **What I rejected:** Initial plan called for removing all Purdue references. I overrode this — Purdue Daniels is my institutional affiliation and should be credited professionally, not hidden.
- **Verification:** Final README contains no mentions of "course", "professor", or "assignment". Purdue Daniels appears once in Contact section as professional affiliation. All internal links (LICENSE, ai_collaboration_log.md, product/README.md) are valid.

### Session 13: Equity Spillover Page, PDF Analyst Credit & Bold Fix
- **AI Tool:** Claude Code (Claude Opus 4.6)
- **What I asked:** (1) Build a new Equity Market Spillover page showing JGB/BOJ policy transmission to equity sectors across USA, Japan, India, and China. (2) Add professional analyst attribution ("Heramb S. Patkar, MSF") to all PDF reports. (3) Fix markdown `**bold**` rendering as raw asterisks in HTML `_section_note` blocks.
- **What I decided:**
  - Four equity markets (USA, Japan, India, China) with sector-level indices: 11 SPDR sector ETFs for USA, 5 Japan ETFs, 15 NSE sector indices for India, 5 China ETFs
  - Four spillover analytics tabs mirroring the bond spillover page: rolling correlation, Granger causality, DCC-GARCH, Diebold-Yilmaz — reusing existing spillover modules rather than duplicating code
  - Per-sector pairwise correlation computation instead of all-at-once DataFrame join, to handle cross-calendar date mismatches (Indian and Japanese trading holidays differ significantly)
  - Filter out sectors with <60 data points rather than failing silently
  - PDF analyst credit on title page, report headers, trade ideas summary, page footer, and disclaimer
- **What I rejected:**
  - Initial implementation used a single `dropna()` across the entire equity returns DataFrame, which eliminated all rows when any one sparse sector (e.g., "Financial Services" with 1 data point) had NaN — I required per-sector independent handling
  - Initial Granger column names were assumed (`best_lag`, `F_stat`) but the actual module uses `optimal_lag`, `f_stat` — caught via live testing on Japanese equities
- **Verification:** All 37 tests pass. Import validation clean. Tested all four markets with live yfinance data. India correlation now returns 7 sectors with 3,900+ data points.

### Session 14: Test Coverage for Equity Spillover & README Cleanup
- **AI Tool:** Claude Code (Claude Opus 4.6)
- **What I asked:** Anticipate professor's next criticism based on past feedback patterns and address proactively.
- **Analysis of past feedback patterns:**
  - First feedback (94.5/A): required README, PCA validation, credit ratings, AI collaboration log — all about *documentation and validation evidence*
  - Second feedback: required splitting monolithic app.py, adding model validations — about *code quality and modularity*
  - Pattern: professor values test coverage, validation evidence, and professional documentation quality
- **What I decided:**
  - Created `tests/test_equity_spillover.py` with 17 tests covering config integrity, helper functions, serialization, rolling correlation logic, sparse-sector exclusion, and integration with Granger/Diebold-Yilmaz modules — the new equity spillover page was the only module with zero test coverage
  - Removed "Development Process" section from README — Session 12 explicitly decided to remove it ("commit count reads as proving effort") but it was still present. Institutional-grade repos show results, not process metrics
  - Updated test count across README from 37 to 54
- **Verification:** All 54 tests pass (37 existing + 17 new). README contains no self-congratulatory process descriptions.

### Session 15: Institutional PDF Redesign
- **AI Tool:** Claude Code (Claude Opus 4.6)
- **What I asked:** Redesign the PDF export to match institutional buy/sell-side research note format. Right-side vertical sidebar on summary page, dense full-width layout for trade cards, formal typography, massive disclaimer. Reorder profile buttons: Analyst first, then Trader, then Academic.
- **What I decided:**
  - Removed all decorative elements (gold header bars, gold rules, coloured accents) — institutional research notes use monochrome palettes with minimal hairlines
  - Summary page: right-side sidebar strip (grey background column) containing regime state, total trades, conviction breakdown, categories, and lead trade. Content flows in the left area
  - Trade card pages: dense full-width layout with specification tables, key levels displayed as compact metric rows, payout graphs spanning full content width
  - Full-page disclaimer (9 paragraphs): general disclaimer, not investment advice, model limitations, data sources, AI-assisted development, payout profiles, IP, limitation of liability, conflicts of interest
  - Profile order: Analyst first (balanced coverage), Trader second (action-first), Academic third (methodology-heavy) — reflects institutional norm where the analyst view is the default deliverable
- **What I rejected:**
  - Initial implementation put the sidebar on the left and on every page — I corrected to right-side, summary page only, matching sell-side research note conventions (GS, JPM, MS style)
- **Verification:** All 54 tests pass. Smoke test generates 6-page PDF (98KB) without errors. Import validation clean.

### Session 16: Institutional First-Page Layout, Sidebar Formalization, About Page CSS
- **AI Tool:** Claude Code (Claude Opus 4.6)
- **What I asked:** Redesign PDF first page to match institutional equity research initiation notes (referenced JPM ASOS and student Walmart equity research reports as templates). Formalize sidebar navigation — current buttons looked "kiddish". Fix About page CSS (all classes were undefined).
- **What I decided:**
  - PDF first page now mirrors JPM/GS equity research layout: black header bar (firm left, report type + date right), bold title on left ~62%, gold "REPRICING/SUPPRESSED" recommendation box on right ~38% (like "Recommendation: Buy"), Key Data table with alternating rows, category breakdown, analyst info panel, full-width trade summary table with black header at bottom
  - Fixed all text overlap issues: added explicit `set_x(_MARGIN)` before every PDF element, moved PCA/ML/Ensemble analytical sections to page 2 to prevent overlap with right panel
  - Sidebar navigation: grouped into ANALYTICS (6 pages), STRATEGY (2 pages), DIAGNOSTICS (2 pages), REFERENCE (2 about pages) with terminal-style buttons — no border-radius, 2px gold left-accent on active, muted grey inactive, 0.5rem uppercase section headers
  - About page: added all 30+ missing CSS class definitions (hero typography, experience timeline, education, publication, certifications, interest tags, acknowledgments) — root cause of broken rendering was HTML referencing classes never defined in `_about_page_styles()`
  - Fixed stale stat: 10 → 11 dashboard pages in about_heramb.py
- **Verification:** All 54 tests pass. PDF smoke test: Trade Ideas 5 pages, Full Analysis 7 pages, no text overlaps. All page module imports clean.

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
