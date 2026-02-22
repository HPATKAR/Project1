"""Early Warning System page."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.ui.shared import (
    _style_fig, _chart, _page_intro, _section_note, _definition_block,
    _takeaway_block, _page_conclusion, _page_footer, _add_boj_events,
    _about_page_styles, _PALETTE,
)
from src.pages._data import load_unified, load_rates, load_market, _safe_col
from src.pages.regime import _run_ensemble


def _get_args():
    """Retrieve sidebar args from session state."""
    return st.session_state["_app_args"]


def _get_layout_config():
    return st.session_state["_layout_config"]


def _get_alert_notifier():
    return st.session_state.get("_alert_notifier")

from src.regime.early_warning import compute_simple_warning_score, generate_warnings
from src.ui.alert_system import AlertDetector, AlertNotifier, AlertThresholds



@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
def _run_warning_score(simulated, start, end, api_key, entropy_window):
    df = load_unified(simulated, start, end, api_key)
    score = compute_simple_warning_score(df, entropy_window=entropy_window)
    return score



@st.cache_data(show_spinner=False, ttl=3600, max_entries=2)
def _run_ml_predictor(simulated, start, end, api_key, entropy_window):
    """Cached ML regime predictor — avoids retraining on every page load."""
    from src.regime.ml_predictor import MLRegimePredictor, compute_regime_features, create_regime_labels

    df = load_unified(simulated, start, end, api_key)
    ensemble = _run_ensemble(simulated, start, end, api_key)
    if ensemble is None or len(ensemble.dropna()) < 100:
        return None, None, None
    features = compute_regime_features(df, entropy_window=entropy_window)
    if len(features) < 100:
        return None, None, None
    labels = create_regime_labels(ensemble)
    predictor = MLRegimePredictor()
    preds, probs, importance = predictor.fit_predict(features, labels)
    return preds, probs, importance



def page_early_warning():
    st.header("Early Warning System")
    _page_intro(
        "Composite early warning score combining entropy divergence, carry stress, and spillover "
        "intensity into a single 0-100 metric. The system monitors for conditions that historically "
        "precede JGB repricing events, providing lead time for position adjustment."
    )
    _definition_block(
        "How the Early Warning System Works",
        "Three independent stress indicators are combined into a single composite score: "
        "<b>Entropy Divergence (40% weight)</b>: When yield changes become unusually complex and "
        "unpredictable (measured by rolling volatility z-score), it signals a breakdown in the "
        "orderly BOJ-suppressed regime. This often fires 1-2 weeks BEFORE other indicators. "
        "<b>Carry Stress (30% weight)</b>: The z-score of the US-Japan rate differential. Rising "
        "carry stress means the gap between US and Japanese rates is widening beyond normal bounds, "
        "creating pressure for JGB yields to catch up. "
        "<b>Spillover Intensity (30% weight)</b>: When JP-US yield correlation exceeds historical "
        "norms, foreign rate moves are transmitting into JGBs more than usual. "
        "The composite score normalizes each to 0-100 and weights them. "
        "<b>Thresholds:</b> 30+ = early signal (INFO), 50+ = elevated (WARNING), 80+ = critical (CRITICAL). "
        "<b>How to read:</b> Watch the score trend, not just the level. A score rising from 20 to 50 "
        "over two weeks is more actionable than a stable 60."
    )

    args = _get_args()

    # --- Load warning score with robust error handling ---
    try:
        with st.spinner("Computing early warning score..."):
            warning_score = _run_warning_score(*args, _get_layout_config().entropy_window)
    except Exception as exc:
        st.error(f"Failed to compute early warning score: {exc}")
        _page_footer()
        return

    if warning_score is None or len(warning_score.dropna()) < 10:
        st.warning("Insufficient data for early warning computation. Ensure the date range covers at least 60 trading days and data sources are configured.")
        _page_footer()
        return

    ws_clean = warning_score.dropna()
    current_score = max(0.0, min(100.0, float(ws_clean.iloc[-1])))

    # --- KPI row ---
    try:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Score", f"{current_score:.0f}/100")
        severity = "CRITICAL" if current_score > 80 else "WARNING" if current_score > 50 else "INFO" if current_score > 30 else "NORMAL"
        c2.metric("Status", severity)
        avg_30 = ws_clean.tail(30).mean() if len(ws_clean) >= 1 else 0.0
        c3.metric("30d Average", f"{avg_30:.0f}")
        c4.metric("Sample Max", f"{ws_clean.max():.0f}")
    except Exception as exc:
        st.warning(f"Could not render KPI metrics: {exc}")

    # --- Score time series ---
    try:
        st.subheader("Composite Warning Score Over Time")
        _section_note(
            f"Current score: <b>{current_score:.0f}/100</b>. "
            f"{'CRITICAL: Multiple stress indicators firing. Prepare for regime shift.' if current_score > 80 else 'WARNING: Elevated stress. Monitor closely.' if current_score > 50 else 'Early signals detected but not yet actionable.' if current_score > 30 else 'All clear. No immediate repricing pressure.'}"
        )
        fig_ws = go.Figure()
        fig_ws.add_trace(go.Scatter(
            x=ws_clean.index, y=ws_clean.values,
            mode="lines", name="Warning Score",
            line=dict(color="#E8413C", width=2),
            fill="tozeroy", fillcolor="rgba(232,65,60,0.08)",
        ))
        fig_ws.add_hline(y=80, line_dash="dash", line_color="#dc2626", annotation_text="CRITICAL (80)")
        fig_ws.add_hline(y=50, line_dash="dash", line_color="#d97706", annotation_text="WARNING (50)")
        fig_ws.add_hline(y=30, line_dash="dot", line_color="#2563eb", annotation_text="INFO (30)")
        fig_ws.update_layout(yaxis_title="Score (0-100)", yaxis_range=[0, 100])
        _add_boj_events(fig_ws)
        _chart(_style_fig(fig_ws, 420))

        _takeaway_block(
            f"Early warning score at <b>{current_score:.0f}/100</b>. "
            f"{'All three components (entropy, carry, spillover) are elevated. This historically precedes repricing by 5-15 trading days. Reduce long JGB duration immediately.' if current_score > 80 else 'Stress is building across multiple indicators. Begin reducing exposure to long-dated JGBs and consider protective hedges.' if current_score > 50 else 'Early stress detected. No action required yet but tighten monitoring frequency and review stop-loss levels.' if current_score > 30 else 'No stress signals. Current positioning can be maintained with standard risk parameters.'}"
        )
    except Exception as exc:
        st.warning(f"Could not render warning score time series: {exc}")

    # --- Score distribution ---
    try:
        st.subheader("Score Distribution")
        _section_note("Histogram of daily warning scores over the full sample period.")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=ws_clean.values, nbinsx=50,
            marker_color="#CFB991", opacity=0.8,
        ))
        fig_hist.add_vline(x=current_score, line_dash="dash", line_color="#E8413C",
                           annotation_text=f"Current: {current_score:.0f}")
        fig_hist.update_layout(xaxis_title="Warning Score", yaxis_title="Frequency")
        _chart(_style_fig(fig_hist, 340))
    except Exception as exc:
        st.warning(f"Could not render score distribution: {exc}")

    # --- Component breakdown ---
    try:
        st.subheader("Component Breakdown")
        _section_note(
            "Each component contributes to the composite score. "
            "Entropy Divergence (40%), Carry Stress (30%), Spillover Intensity (30%)."
        )
        df = load_unified(*_get_args())
        _ew = _get_layout_config().entropy_window
        # Entropy divergence component
        jp10 = df["JP_10Y"].dropna() if "JP_10Y" in df.columns else pd.Series(dtype=float)
        ent_comp = pd.Series(dtype=float)
        carry_comp = pd.Series(dtype=float)
        spill_comp = pd.Series(dtype=float)
        if len(jp10) > _ew:
            roll_vol = jp10.diff().rolling(_ew).std()
            vol_mean = roll_vol.rolling(_ew * 2).mean()
            vol_std = roll_vol.rolling(_ew * 2).std().replace(0, float("nan"))
            z = ((roll_vol - vol_mean) / vol_std).clip(-3, 3)
            ent_comp = ((z + 3) / 6 * 100).dropna()
        # Carry stress component
        us10 = df["US_10Y"].dropna() if "US_10Y" in df.columns else pd.Series(dtype=float)
        if len(jp10) > _ew and len(us10) > _ew:
            spread = (us10 - jp10.reindex(us10.index, method="ffill")).dropna()
            if len(spread) > _ew:
                sp_mean = spread.rolling(_ew * 2).mean()
                sp_std = spread.rolling(_ew * 2).std().replace(0, float("nan"))
                z_carry = ((spread - sp_mean) / sp_std).clip(-3, 3)
                carry_comp = ((z_carry + 3) / 6 * 100).dropna()
        # Spillover intensity component
        if len(jp10) > _ew and len(us10) > _ew:
            corr = jp10.diff().rolling(_ew).corr(us10.diff().reindex(jp10.index, method="ffill").diff())
            spill_comp = (corr.clip(0, 1) * 100).dropna()

        fig_comp = go.Figure()
        if len(ent_comp) > 0:
            fig_comp.add_trace(go.Scatter(x=ent_comp.index, y=ent_comp.values, mode="lines", name="Entropy Divergence (40%)", line=dict(color="#c0392b", width=1.5)))
        if len(carry_comp) > 0:
            fig_comp.add_trace(go.Scatter(x=carry_comp.index, y=carry_comp.values, mode="lines", name="Carry Stress (30%)", line=dict(color="#CFB991", width=1.5)))
        if len(spill_comp) > 0:
            fig_comp.add_trace(go.Scatter(x=spill_comp.index, y=spill_comp.values, mode="lines", name="Spillover Intensity (30%)", line=dict(color="#2e7d32", width=1.5)))
        fig_comp.update_layout(yaxis_title="Component Score (0-100)", yaxis_range=[0, 100])
        _add_boj_events(fig_comp)
        _chart(_style_fig(fig_comp, 380))

        _takeaway_block(
            "The component breakdown reveals <b>which stress channel</b> is driving the composite score. "
            "If entropy divergence leads, the source is domestic (BOJ policy uncertainty). If carry stress "
            "leads, global rate differentials are the driver. If spillover intensity leads, foreign bond "
            "sell-offs are transmitting into JGBs."
        )
    except Exception as exc:
        st.warning(f"Could not render component breakdown: {exc}")

    # --- Rolling statistics ---
    try:
        st.subheader("Rolling Statistics")
        _section_note("30-day rolling mean and standard deviation of the composite warning score.")
        if len(ws_clean) >= 30:
            roll_mean = ws_clean.rolling(30).mean().dropna()
            roll_std = ws_clean.rolling(30).std().dropna()
            fig_roll = go.Figure()
            fig_roll.add_trace(go.Scatter(x=roll_mean.index, y=roll_mean.values, mode="lines", name="30d Mean", line=dict(color="#000000", width=2)))
            fig_roll.add_trace(go.Scatter(x=roll_mean.index, y=(roll_mean + roll_std).values, mode="lines", name="+1 Std Dev", line=dict(color="#CFB991", width=1, dash="dot")))
            fig_roll.add_trace(go.Scatter(x=roll_mean.index, y=(roll_mean - roll_std).clip(lower=0).values, mode="lines", name="-1 Std Dev", line=dict(color="#CFB991", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(207,185,145,0.1)"))
            fig_roll.update_layout(yaxis_title="Score", yaxis_range=[0, 100])
            _add_boj_events(fig_roll)
            _chart(_style_fig(fig_roll, 340))
        else:
            st.info("Need at least 30 observations for rolling statistics.")
    except Exception as exc:
        st.warning(f"Could not render rolling statistics: {exc}")

    # --- Alert generation ---
    try:
        warnings = generate_warnings(ws_clean.tail(60))
        if warnings:
            st.subheader("Recent Warnings")
            _section_note(f"{len(warnings)} warning(s) in the last 60 trading days.")
            for w in warnings[-10:]:
                color = {"CRITICAL": "#dc2626", "WARNING": "#d97706", "INFO": "#2563eb"}.get(w.severity, "#6b7280")
                st.markdown(
                    f"<div style='border-left:4px solid {color};padding:8px 12px;margin-bottom:8px;"
                    f"background:rgba(0,0,0,0.02);border-radius:0 6px 6px 0;'>"
                    f"<b style='color:{color}'>{w.severity}</b> "
                    f"<span style='color:#666;font-size:0.85rem'>{w.timestamp:%Y-%m-%d}</span><br>"
                    f"{w.message}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.subheader("Recent Warnings")
            st.info("No warnings triggered in the last 60 trading days. The market is in a calm state.")
    except Exception as exc:
        st.warning(f"Could not generate warnings: {exc}")

    # --- Detect and persist alerts ---
    try:
        ensemble = _run_ensemble(*args)
        regime_prob = float(ensemble.dropna().iloc[-1]) if ensemble is not None and len(ensemble.dropna()) > 0 else None
    except Exception:
        regime_prob = None

    try:
        detector = AlertDetector()
        alerts = detector.check_all_conditions(
            warning_score=current_score,
            regime_prob=regime_prob,
        )
        if alerts:
            if _get_alert_notifier() is not None:
                _get_alert_notifier().process_alerts(alerts)
            for a in alerts:
                st.toast(f"{a.severity}: {a.title}", icon="🔴" if a.severity == "CRITICAL" else "🟡")
    except Exception as exc:
        st.warning(f"Alert detection encountered an issue: {exc}")

    # --- Data table ---
    try:
        st.subheader("Warning Score Data")
        _section_note("Raw daily warning scores for the selected date range. Download via the CSV button on the Performance Review page.")
        score_df = pd.DataFrame({"Warning Score": ws_clean}).tail(60)
        score_df.index.name = "Date"
        st.dataframe(score_df.style.format({"Warning Score": "{:.1f}"}), use_container_width=True, height=300)
    except Exception as exc:
        st.warning(f"Could not render data table: {exc}")

    # --- Page conclusion (always render, even if sections above failed) ---
    try:
        _verdict_ew = (
            f"Score at {current_score:.0f}: {'CRITICAL — act now' if current_score > 80 else 'WARNING — prepare to act' if current_score > 50 else 'monitoring' if current_score > 30 else 'all clear'}."
        )
        _page_conclusion(
            _verdict_ew,
            f"The composite early warning score integrates entropy divergence, carry stress, and "
            f"spillover intensity with an entropy window of {_get_layout_config().entropy_window} days. "
            f"Current reading: <b>{current_score:.0f}/100</b>.",
        )
    except Exception:
        pass
    _page_footer()


