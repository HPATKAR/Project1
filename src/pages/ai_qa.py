"""AI Q&A page."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from src.ui.shared import (
    _style_fig, _chart, _page_intro, _section_note,
    _page_footer, _add_boj_events, _PALETTE,
)
from src.pages._data import load_unified
from src.pages.regime import _run_ensemble, _run_markov, _run_entropy, _run_garch, _run_breaks
from src.pages.yield_curve import _run_pca, _run_ns, _run_liquidity
from src.pages.spillover import _run_granger, _run_te, _run_spillover, _run_dcc, _run_carry
from src.pages.trade_ideas import _generate_trades


def _get_args():
    """Retrieve sidebar args from session state."""
    return st.session_state["_app_args"]


def _get_layout_config():
    return st.session_state["_layout_config"]


def _get_alert_notifier():
    return st.session_state.get("_alert_notifier")


def _build_analysis_context(args):
    """Serialize current analysis outputs into a rich text context for the LLM.

    Covers: regime (ensemble + sub-models), PCA (variance + loadings),
    spillover (total + top edges), DCC correlations, GARCH vol, Granger
    causality, entropy, structural breaks, Nelson-Siegel, carry, liquidity,
    latest data snapshot, trade ideas, and BOJ policy timeline.
    """
    parts = []

    # ── 1. Regime state (ensemble + sub-model detail) ──
    try:
        ensemble = _run_ensemble(*args)
        if ensemble is not None and len(ensemble.dropna()) > 0:
            ens_clean = ensemble.dropna()
            prob = float(ens_clean.iloc[-1])
            regime = "REPRICING" if prob > 0.5 else "SUPPRESSED"
            # Trend: compare last 20 obs average to prior 20
            if len(ens_clean) >= 40:
                recent = float(ens_clean.iloc[-20:].mean())
                prior = float(ens_clean.iloc[-40:-20].mean())
                trend = "rising" if recent > prior + 0.03 else "falling" if recent < prior - 0.03 else "stable"
            else:
                trend = "n/a"
            parts.append(
                f"REGIME STATE:\n"
                f"  Ensemble probability = {prob:.2%} ({regime})\n"
                f"  Sample average = {ens_clean.mean():.2%}\n"
                f"  20-day trend = {trend}\n"
                f"  Conviction: {'STRONG' if prob > 0.7 or prob < 0.3 else 'MODERATE' if prob > 0.6 or prob < 0.4 else 'TRANSITION'}"
            )
    except Exception:
        pass

    # Sub-models: Markov
    try:
        markov = _run_markov(*args)
        if markov is not None and len(markov.dropna()) > 0:
            parts.append(f"  Markov-Switching: latest high-vol state prob = {float(markov.dropna().iloc[-1]):.2%}")
    except Exception:
        pass

    # Sub-models: Entropy
    try:
        ent, sig = _run_entropy(*args)
        if ent is not None and len(ent.dropna()) > 0:
            ent_v = float(ent.dropna().iloc[-1])
            sig_v = int(sig.dropna().iloc[-1]) if sig is not None and len(sig.dropna()) > 0 else 0
            parts.append(f"  Permutation Entropy: latest = {ent_v:.3f}, regime signal = {'ELEVATED (early warning)' if sig_v == 1 else 'NORMAL'}")
    except Exception:
        pass

    # Sub-models: GARCH vol
    try:
        vol, vol_breaks = _run_garch(*args)
        if vol is not None and len(vol.dropna()) > 0:
            vol_v = float(vol.dropna().iloc[-1])
            vol_pct = float((vol.dropna() < vol_v).mean() * 100)
            parts.append(f"  GARCH(1,1) Conditional Vol: {vol_v:.2f} bps/day ({vol_pct:.0f}th percentile)")
    except Exception:
        pass

    # ── 2. PCA (variance + loadings) ──
    try:
        pca_res = _run_pca(*args)
        if pca_res is not None:
            ev = pca_res["explained_variance_ratio"]
            loadings = pca_res["loadings"]
            pca_lines = [
                f"PCA YIELD CURVE DECOMPOSITION:",
                f"  PC1 (Level): {ev[0]:.1%} variance explained",
                f"  PC2 (Slope): {ev[1]:.1%} variance explained",
                f"  PC3 (Curvature): {ev[2]:.1%} variance explained",
                f"  Cumulative: {sum(ev):.1%}",
                f"  PC1 loadings: {', '.join(f'{c}={v:+.3f}' for c, v in loadings.iloc[0].items())}",
                f"  PC2 loadings: {', '.join(f'{c}={v:+.3f}' for c, v in loadings.iloc[1].items())}",
                f"  PC3 loadings: {', '.join(f'{c}={v:+.3f}' for c, v in loadings.iloc[2].items())}",
            ]
            parts.append("\n".join(pca_lines))
    except Exception:
        pass

    # ── 3. Nelson-Siegel curve factors ──
    try:
        ns_res = _run_ns(*args)
        if ns_res is not None and "params" in ns_res:
            ns_params = ns_res["params"]
            if not ns_params.empty:
                latest_ns = ns_params.iloc[-1]
                parts.append(
                    f"NELSON-SIEGEL CURVE FACTORS (latest):\n"
                    f"  Beta0 (Level) = {latest_ns.get('beta0', float('nan')):.4f}\n"
                    f"  Beta1 (Slope) = {latest_ns.get('beta1', float('nan')):.4f}\n"
                    f"  Beta2 (Curvature) = {latest_ns.get('beta2', float('nan')):.4f}\n"
                    f"  Interpretation: {'Flattening' if latest_ns.get('beta1', 0) > 0 else 'Steepening'} curve, "
                    f"{'Positive' if latest_ns.get('beta2', 0) > 0 else 'Negative'} belly"
                )
    except Exception:
        pass

    # ── 4. Spillover (total + top directional edges) ──
    try:
        spill = _run_spillover(*args)
        if spill is not None:
            net = spill["net_spillover"]
            mat = spill.get("spillover_matrix")
            spill_lines = [
                f"DIEBOLD-YILMAZ SPILLOVER (VAR(4), 10-step horizon):",
                f"  Total spillover index = {spill['total_spillover']:.1f}%",
                f"  Net transmitters: {', '.join(f'{k}={v:+.1f}%' for k, v in net.sort_values(ascending=False).head(3).items())}",
                f"  Net receivers: {', '.join(f'{k}={v:+.1f}%' for k, v in net.sort_values().head(3).items())}",
            ]
            # Top 5 directional edges from spillover matrix
            if mat is not None:
                edges = []
                for i in mat.index:
                    for j in mat.columns:
                        if i != j:
                            edges.append((i, j, float(mat.loc[i, j])))
                edges.sort(key=lambda x: -x[2])
                top_edges = edges[:5]
                spill_lines.append(f"  Top 5 directional flows:")
                for src, tgt, val in top_edges:
                    spill_lines.append(f"    {src} → {tgt}: {val:.1f}%")
            parts.append("\n".join(spill_lines))
    except Exception:
        pass

    # ── 5. DCC correlations (latest) ──
    try:
        dcc_res = _run_dcc(*args)
        if dcc_res is not None:
            corrs = dcc_res.get("conditional_correlations", {})
            if corrs:
                dcc_lines = ["DCC TIME-VARYING CORRELATIONS (latest):"]
                for pair, series in corrs.items():
                    if len(series.dropna()) > 0:
                        latest_c = float(series.dropna().iloc[-1])
                        avg_c = float(series.dropna().mean())
                        dcc_lines.append(f"  {pair}: current={latest_c:+.3f}, sample avg={avg_c:+.3f}, "
                                         f"{'ELEVATED' if abs(latest_c) > abs(avg_c) + 0.1 else 'NORMAL'}")
                parts.append("\n".join(dcc_lines))
    except Exception:
        pass

    # ── 6. Granger causality (significant pairs only) ──
    try:
        granger_df = _run_granger(*args)
        if granger_df is not None and not granger_df.empty:
            sig_pairs = granger_df[granger_df["significant"] == True].sort_values("p_value")
            if not sig_pairs.empty:
                gc_lines = [f"GRANGER CAUSALITY (significant pairs, p<0.05):"]
                for _, row in sig_pairs.head(8).iterrows():
                    gc_lines.append(
                        f"  {row['cause']} → {row['effect']}: F={row['f_stat']:.2f}, p={row['p_value']:.4f}, lag={int(row['optimal_lag'])}"
                    )
                parts.append("\n".join(gc_lines))
    except Exception:
        pass

    # ── 7. Structural breaks ──
    try:
        changes, bkps = _run_breaks(*args)
        if bkps and changes is not None and len(changes) > 0:
            break_dates = [changes.index[min(b, len(changes) - 1)].strftime("%Y-%m-%d") for b in bkps if b < len(changes)]
            if break_dates:
                parts.append(f"STRUCTURAL BREAKS (PELT on JP_10Y changes): {', '.join(break_dates)}")
    except Exception:
        pass

    # ── 8. FX Carry ──
    try:
        carry = _run_carry(*args)
        if carry is not None and len(carry["carry_to_vol"].dropna()) > 0:
            ctv = float(carry["carry_to_vol"].dropna().iloc[-1])
            carry_raw = float(carry["carry"].dropna().iloc[-1]) if len(carry["carry"].dropna()) > 0 else float("nan")
            parts.append(
                f"FX CARRY ANALYTICS:\n"
                f"  Carry (US-JP rate differential) = {carry_raw:.2f}%\n"
                f"  Carry-to-vol ratio = {ctv:.2f} ({'Attractive — carry exceeds vol risk' if ctv > 1 else 'Marginal' if ctv > 0.5 else 'Unattractive — vol dominates carry'})"
            )
    except Exception:
        pass

    # ── 9. Liquidity ──
    try:
        liq = _run_liquidity(*args)
        if liq is not None and len(liq["composite_index"].dropna()) > 0:
            liq_v = float(liq["composite_index"].dropna().iloc[-1])
            parts.append(f"LIQUIDITY: Composite index = {liq_v:+.2f} z-score. {'Stressed — wider bid-ask, higher impact costs' if liq_v < -1 else 'Healthy' if liq_v > 0 else 'Neutral'}.")
    except Exception:
        pass

    # ── 10. Latest data snapshot ──
    try:
        df = load_unified(*args)
        if not df.empty:
            latest = df.iloc[-1]
            snap = []
            for col in ["JP_10Y", "US_10Y", "DE_10Y", "UK_10Y", "AU_10Y", "USDJPY", "VIX", "NIKKEI"]:
                if col in latest.index and pd.notna(latest[col]):
                    snap.append(f"{col}={latest[col]:.2f}")
            if snap:
                parts.append(f"LATEST DATA ({df.index[-1]:%Y-%m-%d}): {', '.join(snap)}.")
            # Curve slopes if available
            slopes = []
            for short, long, label in [("JP_2Y", "JP_10Y", "JP 2s10s"), ("JP_10Y", "JP_30Y", "JP 10s30s"), ("US_2Y", "US_10Y", "US 2s10s")]:
                if short in latest.index and long in latest.index and pd.notna(latest[short]) and pd.notna(latest[long]):
                    slopes.append(f"{label}={latest[long] - latest[short]:+.2f}%")
            if slopes:
                parts.append(f"CURVE SLOPES: {', '.join(slopes)}")
    except Exception:
        pass

    # ── 11. Trade ideas (all, with failure scenarios) ──
    try:
        cards, rs = _generate_trades(*args)
        if cards:
            top_5 = sorted(cards, key=lambda c: -c.conviction)[:5]
            trade_lines = [f"TOP TRADE IDEAS ({len(cards)} total, top 5 by conviction):"]
            for c in top_5:
                trade_lines.append(
                    f"  - {c.name} ({c.direction.upper()}, {c.conviction:.0%}, {c.category})\n"
                    f"    Entry: {c.entry_signal}\n"
                    f"    Failure: {c.failure_scenario}"
                )
            parts.append("\n".join(trade_lines))
    except Exception:
        pass

    # ── 12. BOJ events ──
    from src.data.config import BOJ_EVENTS as _boj_events
    parts.append("BOJ POLICY TIMELINE:\n" + "\n".join(f"  {d}: {e}" for d, e in _boj_events.items()))

    return "\n\n".join(parts)



def page_ai_qa():
    st.header("AI Q&A")
    _page_intro(
        "Chat interface grounded in the case study. The AI assistant is injected with live framework outputs "
        "including regime ensemble, PCA loadings, Nelson-Siegel curve factors, Diebold-Yilmaz spillover edges, "
        "DCC correlations, Granger causality pairs, GARCH vol, carry analytics, liquidity, structural breaks, "
        "and trade ideas with failure scenarios. Every answer must cite specific metrics from the analysis."
    )

    # --- API key setup (secrets.toml > env var > sidebar) ---
    ai_provider = st.sidebar.selectbox("AI Provider", ["OpenAI (GPT)", "Anthropic (Claude)"], key="ai_provider")
    _secret_key = ""
    if "OpenAI" in ai_provider:
        _secret_key = st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    else:
        _secret_key = st.secrets.get("ANTHROPIC_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
    ai_api_key = st.sidebar.text_input(
        "AI API Key",
        value=_secret_key,
        type="password",
        key="ai_api_key",
        help="Auto-loaded from .streamlit/secrets.toml if set. Or paste here.",
    )

    # --- Args for context builder (lazy, only called when needed) ---
    args = _get_args()

    if not ai_api_key:
        # Show a professional empty state instead of just a bare info box
        st.markdown(
            "<div style='text-align:center;padding:3rem 2rem;'>"
            "<div style='font-size:2.5rem;margin-bottom:0.8rem;opacity:0.15;'>&#x1F4AC;</div>"
            "<p style='font-family:var(--font-sans);font-size:var(--fs-2xl);font-weight:600;"
            "color:#000;margin:0 0 6px 0;'>AI Research Assistant</p>"
            "<p style='font-family:var(--font-sans);font-size:var(--fs-md);color:#2d2d2d;"
            "max-width:480px;margin:0 auto 1.5rem auto;line-height:1.65;'>"
            "Enter your API key in the sidebar to activate the conversational AI assistant. "
            "It has access to live regime state, PCA decomposition, spillover metrics, "
            "and trade ideas from this session.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        # Suggested topics as visual guidance
        st.markdown(
            "<div style='display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-bottom:2rem;'>"
            "<span style='background:#fafaf8;border:1px solid #eceae6;border-radius:8px;"
            "padding:8px 16px;font-size:var(--fs-base);color:#1a1a1a;font-family:var(--font-sans);'>"
            "Regime call with evidence</span>"
            "<span style='background:#fafaf8;border:1px solid #eceae6;border-radius:8px;"
            "padding:8px 16px;font-size:var(--fs-base);color:#1a1a1a;font-family:var(--font-sans);'>"
            "PCA loadings &amp; curve factors</span>"
            "<span style='background:#fafaf8;border:1px solid #eceae6;border-radius:8px;"
            "padding:8px 16px;font-size:var(--fs-base);color:#1a1a1a;font-family:var(--font-sans);'>"
            "Spillover transmission chain</span>"
            "<span style='background:#fafaf8;border:1px solid #eceae6;border-radius:8px;"
            "padding:8px 16px;font-size:var(--fs-base);color:#1a1a1a;font-family:var(--font-sans);'>"
            "Trade thesis &amp; failure scenarios</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        _page_footer()
        return

    # --- Chat state ---
    if "qa_messages" not in st.session_state:
        st.session_state.qa_messages = []

    # Determine if a card was just clicked (pending question to process)
    _pending_card_q = None

    # Show empty state with suggestions when no messages yet
    if not st.session_state.qa_messages:
        st.markdown(
            "<div style='text-align:center;padding:2rem 2rem 1rem 2rem;'>"
            "<p style='font-family:var(--font-sans);font-size:var(--fs-2xl);font-weight:600;"
            "color:#000;margin:0 0 4px 0;'>Ready to assist</p>"
            "<p style='font-family:var(--font-sans);font-size:var(--fs-md);color:#2d2d2d;"
            "margin:0 0 1.2rem 0;'>Ask about JGBs, rates, macro, BOJ policy, "
            "trading strategies, or yield curves.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        # Clickable topic cards
        _topics = [
            ("Explain the current regime call: cite the ensemble probability, sub-model signals, and what is driving the reading.", "Regime Call"),
            ("Break down the PCA loadings: what do PC1/PC2/PC3 represent economically, and what does the variance split tell us about yield dynamics?", "PCA Factors"),
            ("Trace the transmission chain: how are shocks flowing between JGBs, USTs, USDJPY, and VIX using spillover edges, Granger causality, and DCC correlations?", "Spillover Chain"),
            ("What are the top trade ideas, their entry signals, and what would falsify each thesis?", "Trade Thesis"),
        ]
        _cols = st.columns(len(_topics))
        for _col, (_q, _label) in zip(_cols, _topics):
            with _col:
                if st.button(
                    _label,
                    key=f"qa_topic_{_label}",
                    use_container_width=True,
                    type="secondary",
                ):
                    _pending_card_q = _q

    # Display chat history
    for msg in st.session_state.qa_messages:
        _avatar = "\U0001F9D1" if msg["role"] == "user" else "\U0001F916"
        with st.chat_message(msg["role"], avatar=_avatar):
            st.markdown(msg["content"])

    # --- Chat input (inline, not fixed to bottom) ---
    user_input = st.text_input(
        "Ask anything about JGBs, rates, macro, BOJ policy, trading strategies...",
        key="qa_text_input",
        label_visibility="collapsed",
        placeholder="Ask anything about JGBs, rates, macro, BOJ policy, trading strategies...",
    )

    # Use card click or text input
    question = _pending_card_q or (user_input if user_input else None)

    if question:
        st.session_state.qa_messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="\U0001F9D1"):
            st.markdown(question)

        # Build context only when actually needed
        with st.spinner("Building analysis context..."):
            analysis_context = _build_analysis_context(args)

        system_prompt = (
            "You are an AI research assistant for a JGB Repricing Framework, a quantitative case study "
            "analyzing the regime shift from BOJ-suppressed yields to market-driven pricing in the Japanese "
            "Government Bond market.\n\n"
            "THESIS: Japan's decade of yield curve control (YCC) and quantitative easing artificially suppressed "
            "JGB yields. As the BOJ exits these policies (Dec 2022 band widening, Mar 2024 formal YCC exit), "
            "repricing risk propagates through rates, FX, volatility, and cross-asset channels.\n\n"
            "RULES:\n"
            "1. ALWAYS ground your answers in the live framework data below. Cite at least 2 specific metrics "
            "   from the injected context when discussing regime, spillovers, curve, carry, or trade ideas.\n"
            "2. Frame answers through the thesis lens: BOJ suppression, cross-asset spillovers, repricing risk, "
            "   trade implications.\n"
            "3. When referencing PCA, explain what the loadings mean economically (level/slope/curvature).\n"
            "4. When discussing trades, always mention the failure scenario.\n"
            "5. If the data does not support a claim, say so explicitly: 'Not supported by current framework outputs.'\n"
            "6. Do NOT invent data that is not in the context below.\n"
            "7. Be concise, quantitative, and actionable. Use bullet points for structure.\n\n"
            f"=== LIVE FRAMEWORK CONTEXT ===\n{analysis_context}\n=== END CONTEXT ==="
        )

        # Build message history for the API
        api_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.qa_messages]

        with st.chat_message("assistant", avatar="\U0001F916"):
            with st.spinner("Thinking..."):
                try:
                    if "Anthropic" in ai_provider:
                        import anthropic
                        client = anthropic.Anthropic(api_key=ai_api_key)
                        response = client.messages.create(
                            model="claude-sonnet-4-5-20250929",
                            max_tokens=2048,
                            system=system_prompt,
                            messages=api_messages,
                        )
                        assistant_msg = response.content[0].text
                    else:
                        import openai
                        client = openai.OpenAI(api_key=ai_api_key)
                        oai_messages = [{"role": "system", "content": system_prompt}] + api_messages
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=oai_messages,
                            max_tokens=2048,
                        )
                        assistant_msg = response.choices[0].message.content

                    st.markdown(assistant_msg)
                    st.session_state.qa_messages.append({"role": "assistant", "content": assistant_msg})
                except ImportError as e:
                    missing = "anthropic" if "anthropic" in str(e) else "openai"
                    st.error(f"Missing package: `{missing}`. Install with `pip install {missing}`.")
                except Exception as e:
                    st.error(f"API call failed: {e}")

    _page_footer()


