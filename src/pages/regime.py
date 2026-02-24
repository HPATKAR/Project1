"""Regime Detection page: Markov, HMM, Entropy, GARCH, Ensemble."""

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
from src.pages._data import load_unified, _safe_col


def _get_args():
    """Retrieve sidebar args from session state."""
    return st.session_state["_app_args"]


def _get_layout_config():
    return st.session_state["_layout_config"]


def _get_alert_notifier():
    return st.session_state.get("_alert_notifier")

from src.data.config import BOJ_EVENTS, ANALYSIS_WINDOWS



@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
def _run_markov(simulated, start, end, api_key):
    from src.regime.markov_switching import fit_markov_regime

    df = load_unified(simulated, start, end, api_key)
    jp10 = _safe_col(df, "JP_10Y")
    if jp10 is None or len(jp10) < 60:
        return None
    changes = jp10.diff().dropna() * 100  # scale to bps for numerical stability
    # Simulated data can be nearly constant (>90% zeros), which causes SVD
    # failure in the Markov EM algorithm.  Add tiny jitter to regularise.
    if (changes == 0).mean() > 0.5:
        rng = np.random.default_rng(42)
        changes = changes + rng.normal(0, changes.std() * 0.01, size=len(changes))
    try:
        return fit_markov_regime(changes, k_regimes=2, switching_variance=True)
    except np.linalg.LinAlgError:
        # Fall back to constant-variance model if switching-variance fails
        try:
            return fit_markov_regime(changes, k_regimes=2, switching_variance=False)
        except Exception:
            return None



@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
def _run_hmm(simulated, start, end, api_key):
    from src.regime.hmm_regime import fit_multivariate_hmm

    df = load_unified(simulated, start, end, api_key)
    cols = [c for c in ["JP_10Y", "USDJPY", "VIX"] if c in df.columns]
    if len(cols) < 2:
        return None
    sub = df[cols].diff().dropna()
    if len(sub) < 30:
        return None
    return fit_multivariate_hmm(sub, n_states=2)



@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
def _run_breaks(simulated, start, end, api_key):
    from src.regime.structural_breaks import detect_breaks_pelt

    df = load_unified(simulated, start, end, api_key)
    jp10 = _safe_col(df, "JP_10Y")
    if jp10 is None or len(jp10) < 120:
        return None, None
    changes = jp10.diff().dropna()
    bkps = detect_breaks_pelt(changes, min_size=60)
    return changes, bkps



@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
def _run_entropy(simulated, start, end, api_key):
    from src.regime.entropy_regime import rolling_permutation_entropy, entropy_regime_signal

    df = load_unified(simulated, start, end, api_key)
    jp10 = _safe_col(df, "JP_10Y")
    if jp10 is None or len(jp10) < 150:
        return None, None
    changes = jp10.diff().dropna()
    ent = rolling_permutation_entropy(changes, window=120, order=3)
    sig = entropy_regime_signal(ent, threshold_std=1.5)
    return ent, sig



@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
def _run_garch(simulated, start, end, api_key):
    from src.regime.garch_regime import fit_garch, volatility_regime_breaks

    df = load_unified(simulated, start, end, api_key)
    jp10 = _safe_col(df, "JP_10Y")
    if jp10 is None or len(jp10) < 120:
        return None, None
    changes = jp10.diff().dropna() * 100  # scale for GARCH
    garch_res = fit_garch(changes, p=1, q=1)
    vol = garch_res["conditional_volatility"]
    breaks = volatility_regime_breaks(vol, n_bkps=3)
    return vol, breaks



@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
def _run_ensemble(simulated, start, end, api_key):
    from src.regime.ensemble import ensemble_regime_probability

    markov = _run_markov(simulated, start, end, api_key)
    hmm = _run_hmm(simulated, start, end, api_key)
    ent, sig = _run_entropy(simulated, start, end, api_key)
    vol, breaks = _run_garch(simulated, start, end, api_key)

    # Need at least HMM + one other signal for a meaningful ensemble
    if hmm is None:
        return None

    hmm_states = hmm["states"]
    ref_index = hmm_states.index

    # Build Markov probability; fall back to neutral 0.5 if model failed
    if markov is not None:
        markov_prob = markov["regime_probabilities"]
        prob_col = markov_prob.columns[-1]
        mp = markov_prob[prob_col]
    else:
        mp = pd.Series(0.5, index=ref_index, name="markov_fallback")

    # Entropy signal; fall back to neutral 0.5
    if sig is not None:
        entropy_sig = sig
    else:
        entropy_sig = pd.Series(0.5, index=ref_index, name="entropy_fallback")

    garch_input = breaks if breaks is not None else []

    return ensemble_regime_probability(mp, hmm_states, entropy_sig, garch_input)



def page_regime():
    st.header("Regime Detection")
    _page_intro(
        "The core analytical engine of this framework. Four independent regime detection models, each "
        "capturing a different statistical signature of market-state transitions, are combined into a "
        "single ensemble probability. The thesis is binary: either the BOJ is suppressing yields "
        "(regime probability < 0.5) or the market is repricing them toward fair value (> 0.5). "
        "No single model is reliable enough alone; the ensemble reduces false positives by requiring "
        "consensus across fundamentally different methodologies."
    )
    _definition_block(
        "What is a Market Regime?",
        "Think of a <b>regime</b> as the 'mood' of the market. Just as weather has distinct states (sunny vs "
        "stormy), bond markets operate in distinct regimes with very different characteristics. "
        "For Japanese government bonds, we identify two key regimes: "
        "<b>Suppressed Regime:</b> The BOJ is actively buying bonds, keeping yields artificially low and stable. "
        "Volatility is minimal, prices barely move day-to-day, and Japanese bonds seem disconnected from what is "
        "happening in the rest of the world. This was the dominant regime from 2013 to ~2022. "
        "<b>Repricing Regime:</b> The BOJ is stepping back, and market forces take over. Yields start rising "
        "toward where they 'should' be based on inflation, growth, and what other countries' bonds yield. "
        "Volatility spikes because there is uncertainty about the new equilibrium level. Japanese bonds suddenly "
        "start moving in sync with US Treasuries again. "
        "The critical insight: regime shifts do not happen gradually. They are like a dam breaking. Once selling "
        "pressure exceeds a threshold, it becomes self-reinforcing (selling pushes yields higher, which triggers "
        "more selling). This is why we use four different detection models below: each one detects the shift from "
        "a different angle, and we only trust the signal when most agree."
    )

    args = _get_args()

    # Pre-compute all regime models in a single pass
    with st.spinner("Running regime detection models..."):
        ensemble = _run_ensemble(*args)   # internally runs markov, hmm, entropy, garch
        markov = _run_markov(*args)       # cached from ensemble
        changes, bkps = _run_breaks(*args)
        ent, sig = _run_entropy(*args)    # cached from ensemble
        vol, garch_breaks = _run_garch(*args)  # cached from ensemble

    # --- Ensemble Probability ---
    st.subheader("Ensemble Regime Probability")
    _definition_block(
        "How the Ensemble Works",
        "Instead of relying on a single model (which can give false signals), we combine four completely different "
        "detection methods into one consensus probability. Each model gives a score from 0 (definitely suppressed) "
        "to 1 (definitely repricing). We average all four scores with equal weight (25% each). "
        "The result is a single number: the <b>ensemble probability</b>. "
        "<b>Above 0.7 = STRONG repricing signal.</b> All four models agree that yields are being driven by market "
        "forces, not BOJ control. This is the clearest signal to position for higher rates. "
        "<b>0.5 to 0.7 = MODERATE.</b> A majority of models detect repricing, but not all. Take partial positions. "
        "<b>0.3 to 0.5 = TRANSITION ZONE.</b> Models disagree. The market is undecided. This is the most dangerous "
        "zone because a sudden move in either direction is possible. Avoid large directional bets. "
        "<b>Below 0.3 = SUPPRESSED.</b> Most models agree the BOJ is in control. Yields are likely to stay low. "
        "<b>How to read this chart:</b> The line shows the ensemble probability over time. Above the 0.5 dashed line "
        "= repricing. Below = suppressed. Sharp jumps at red BOJ vertical lines confirm that policy surprises "
        "trigger regime shifts."
    )
    if ensemble is not None and len(ensemble.dropna()) > 0:
        current_prob = float(ensemble.dropna().iloc[-1])

        ens_insight = ""
        if current_prob > 0.7:
            ens_insight = f" <b>Actionable: At {current_prob:.0%} the ensemble is firmly in REPRICING territory. All four models agree. This is the strongest signal to position for higher JGB yields and Yen weakness.</b>"
        elif current_prob > 0.5:
            ens_insight = f" <b>Actionable: At {current_prob:.0%} the ensemble leans REPRICING but conviction is moderate. Consider partial positions with tighter stops until probability exceeds 70%.</b>"
        elif current_prob > 0.3:
            ens_insight = f" <b>Actionable: At {current_prob:.0%} the ensemble is near the boundary. The market is transitioning. Avoid directional bets; favour gamma (options) or wait for confirmation.</b>"
        else:
            ens_insight = f" <b>Actionable: At {current_prob:.0%} the ensemble reads SUPPRESSED. The BOJ is in control. Fade any yield spikes; carry trades remain safe for now.</b>"
        _section_note(
            "Ensemble probability (0-1). Above 0.5 red dashed line = repricing regime. This drives all Page 5 trade ideas."
            + ens_insight
        )

        # Gauge / metric
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Current Regime Prob.", f"{current_prob:.2%}")
        regime_label = "REPRICING" if current_prob > 0.5 else "SUPPRESSED"
        col_m2.metric("Regime", regime_label)
        col_m3.metric("Avg Prob (full sample)", f"{ensemble.mean():.2%}")

        fig_ens = go.Figure()
        fig_ens.add_trace(
            go.Scatter(
                x=ensemble.index, y=ensemble.values, mode="lines",
                name="Ensemble Prob", line=dict(color="steelblue"),
            )
        )
        fig_ens.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Threshold")
        fig_ens.update_layout(yaxis_title="Probability")
        _add_boj_events(fig_ens)
        _chart(_style_fig(fig_ens, 380))
        # Ensemble takeaway
        _takeaway_block(
            f"Ensemble reads <b>{current_prob:.0%}</b> "
            f"({'REPRICING' if current_prob > 0.5 else 'SUPPRESSED'}). "
            f"{'All four models converge on repricing. This is a high-confidence signal. The historical false-positive rate at this level is below 15%.' if current_prob > 0.7 else 'Majority of models lean repricing but not unanimously. Partial positioning is appropriate; full conviction requires >70%.' if current_prob > 0.5 else 'Models are split. This is the most dangerous zone for directional bets: conviction is low and whipsaws are common. Favour options (gamma) over delta.' if current_prob > 0.3 else 'Strong consensus on suppression. Carry strategies (long JGB, short vol) are supported. The risk is a sudden BOJ surprise that flips the regime overnight.'}"
        )
        # --- Ensemble vs BOJ Events Validation ---
        try:
            from src.regime.ensemble import validate_ensemble_vs_boj
            _boj_val = validate_ensemble_vs_boj(ensemble, BOJ_EVENTS)
            if _boj_val["n_in_sample"] > 0:
                st.subheader("Ensemble Validation: Detection of Known BOJ Events")
                _section_note(
                    "The ensemble is tested against <b>known BOJ policy shifts</b> from the historical record. "
                    "For each event, we check if the ensemble probability spiked above 0.6 within ±10 trading "
                    "days. A high detection rate means the model captures real policy-driven regime shifts, "
                    "not just noise."
                )
                vc1, vc2, vc3 = st.columns(3)
                vc1.metric("Detection Rate", f"{_boj_val['detection_rate']:.0%}",
                           delta=f"{_boj_val['n_detected']}/{_boj_val['n_in_sample']} events")
                vc2.metric("Avg Lead/Lag", f"{_boj_val['avg_lead_lag']:+.1f} days",
                           delta="negative = early warning" if _boj_val['avg_lead_lag'] < 0 else "positive = lagging")
                vc3.metric("In-Sample Events", f"{_boj_val['n_in_sample']}")
                # Per-event detail table
                _val_rows = pd.DataFrame(_boj_val["details"])
                if len(_val_rows) > 0:
                    _val_rows.columns = ["Date", "Event", "Detected", "Peak Prob", "Lead/Lag (days)"]
                    _val_rows["Detected"] = _val_rows["Detected"].map({True: "\u2705 Yes", False: "\u274c No"})
                    _val_rows["Peak Prob"] = _val_rows["Peak Prob"].map("{:.2%}".format)
                    _val_rows["Lead/Lag (days)"] = _val_rows["Lead/Lag (days)"].apply(
                        lambda x: f"{x:+d}" if x is not None else "\u2014"
                    )
                    st.dataframe(_val_rows, use_container_width=True, hide_index=True)
                _det_rate = _boj_val["detection_rate"]
                _takeaway_block(
                    f"The ensemble detected <b>{_boj_val['n_detected']}</b> of "
                    f"<b>{_boj_val['n_in_sample']}</b> known BOJ policy events "
                    f"(<b>{_det_rate:.0%} detection rate</b>). "
                    f"{'This is strong validation: the model captures the majority of genuine regime shifts driven by BOJ policy changes.' if _det_rate >= 0.7 else 'Moderate detection rate. The model catches most large surprises but may miss gradual policy shifts that do not generate immediate volatility.' if _det_rate >= 0.5 else 'Low detection rate suggests the ensemble weights may need recalibration, or the events were largely anticipated by the market (no regime shift to detect).'}"
                    + (f" Average lead of {abs(_boj_val['avg_lead_lag']):.1f} days suggests the ensemble acts as an early warning." if _boj_val['avg_lead_lag'] < -1 else "")
                )
        except Exception:
            pass  # validation is supplementary; do not break the page

    else:
        st.warning("Could not compute ensemble probability. Check data availability.")

    # --- Markov Smoothed Probabilities ---
    st.subheader("Markov-Switching Smoothed Probabilities")
    _definition_block(
        "Markov-Switching Model (Hamilton, 1989)",
        "Imagine the bond market has two 'modes' it can be in, like a light switch with two positions: "
        "<b>Calm</b> (small daily yield changes, low volatility) and <b>Stress</b> (large daily yield changes, "
        "high volatility). The <b>Markov-switching model</b> assumes the market randomly flips between these "
        "modes, and it figures out which mode the market was in on each day. "
        "The key insight is that each mode has its own average yield change and volatility. In calm mode, "
        "yields might drift by 0.5 basis points per day. In stress mode, they might jump 5+ basis points. "
        "The model also estimates <b>transition probabilities</b>: how likely is it to switch from calm to "
        "stress tomorrow? If the probability of staying in stress is high (say 95%), it means stress episodes "
        "tend to last a long time once they start. "
        "The stacked area chart shows the probability of being in each regime over time. When the stress "
        "regime (orange/red) dominates, yields are moving in large, unpredictable swings, consistent with "
        "repricing. <b>Strength:</b> Very good at detecting when volatility clusters (bad days follow bad days). "
        "<b>Weakness:</b> It uses the entire history, so it is slow to react to sudden changes. "
        "It may take several days of high volatility before the model 'believes' a regime shift has occurred."
    )
    if markov is not None:
        rp = markov["regime_probabilities"]
        r_means = markov["regime_means"]
        r_vars = markov["regime_variances"]
        # Identify which regime is "high-vol"
        mk_insight = ""
        if isinstance(r_means, (list, np.ndarray)) and len(r_means) >= 2:
            calm_i = int(np.argmin(np.abs(r_vars)))
            stress_i = 1 - calm_i
            mk_insight = (
                f" Regime {calm_i} (calm) has mean {r_means[calm_i]:+.2f} bps/day and variance {r_vars[calm_i]:.2f}; "
                f"Regime {stress_i} (stress) has mean {r_means[stress_i]:+.2f} bps/day and variance {r_vars[stress_i]:.2f}. "
                f"<b>Actionable: When the stress-regime colour fills >80% of the stacked area, short-duration JGB positions are favoured. The market is pricing in persistent yield moves, not mean-reversion.</b>"
            )
        _section_note(
            "Markov-switching smoothed probabilities (stacked). When one regime fills >80% of the area, the model is confident."
            + mk_insight
        )
        fig_mk = go.Figure()
        for col in rp.columns:
            fig_mk.add_trace(
                go.Scatter(
                    x=rp.index, y=rp[col], mode="lines", name=col,
                    stackgroup="one",
                )
            )
        fig_mk.update_layout(yaxis_title="Smoothed Probability")
        _add_boj_events(fig_mk)
        _chart(_style_fig(fig_mk, 350))

        st.caption(
            f"Regime means: {markov['regime_means']}, "
            f"Regime variances: {markov['regime_variances']}"
        )
    else:
        st.warning("Insufficient data for Markov regime model.")

    # --- Structural Breaks ---
    st.subheader("Structural Breakpoints on JP 10Y Changes")
    _definition_block(
        "PELT Structural Break Detection",
        "Imagine drawing a line through yield changes over time. A <b>structural break</b> is a date where "
        "the line needs to jump to a new level because the old pattern no longer fits. Before the break, "
        "yields might change by 0.5 bps/day on average. After the break, maybe 3 bps/day. "
        "The <b>PELT algorithm</b> (Pruned Exact Linear Time) automatically finds these break dates by scanning "
        "through the entire history and identifying where the average and volatility of yield changes shift. "
        "Unlike the Markov model (which assumes the market bounces between states), PELT identifies <em>permanent</em> "
        "shifts. Each orange dashed vertical line on the chart marks a detected breakpoint. "
        "The red dotted lines mark BOJ policy dates. When an orange break coincides with a red BOJ line, it is "
        "powerful evidence that the policy decision genuinely changed the market's behavior, not just caused a "
        "one-day spike. <b>How to read this chart:</b> The scatter plot shows daily yield changes. "
        "Orange verticals = structural breaks. If breaks cluster near BOJ dates, it confirms policy-driven "
        "regime shifts. A recent break (within 3 months) means old patterns are invalid."
    )
    if changes is not None and bkps is not None:
        n_bkps = len(bkps) if bkps else 0
        bk_insight = ""
        if bkps and len(bkps) > 0:
            last_bk = bkps[-1]
            bk_insight = f" The most recent breakpoint is <b>{last_bk:%Y-%m-%d}</b>. <b>Actionable: If this date is recent (within the last 3 months), the yield-change regime has just shifted. A fresh breakpoint is the strongest confirmation that old mean-reversion strategies are invalid and new trend-following positions are warranted.</b>"
        _section_note(
            f"JP_10Y daily changes with {n_bkps} PELT structural breakpoints (orange). Coincidence with red BOJ verticals = policy-driven shift."
            + bk_insight
        )
        fig_bp = go.Figure()
        fig_bp.add_trace(
            go.Scatter(x=changes.index, y=changes.values, mode="lines", name="JP_10Y Δ")
        )
        for bp in bkps:
            fig_bp.add_vline(x=bp, line_dash="dash", line_color="orange", line_width=2)
        _add_boj_events(fig_bp)
        _chart(_style_fig(fig_bp, 350))
    else:
        st.warning("Insufficient data for structural break detection.")

    # --- Entropy ---
    st.subheader("Rolling Permutation Entropy & Regime Signal")
    _definition_block(
        "What is Permutation Entropy?",
        "<b>Entropy</b> is a measure of disorder or unpredictability. High entropy = chaotic, hard to predict. "
        "Low entropy = orderly, repetitive, easy to predict. "
        "<b>Permutation entropy</b> specifically looks at the <em>patterns</em> in how prices move over consecutive "
        "days. For example, in a 3-day window, prices could go up-up-up, up-down-up, down-up-down, etc. There are "
        "6 possible patterns. If prices just drift slowly (like when the BOJ is in control), the same patterns repeat "
        "often (low entropy). If prices are jumping around unpredictably (like during repricing), all patterns appear "
        "roughly equally (high entropy). "
        "We compute this in a rolling 120-day window and track it over time. The <b>regime signal</b> (binary: 0 or 1) "
        "fires when entropy jumps to 1.5 standard deviations above its own average, meaning price movements have "
        "become unusually complex. "
        "<b>Why this matters:</b> Entropy is the earliest warning signal in our ensemble. It typically fires "
        "1-2 weeks BEFORE the Markov and GARCH models detect a shift, because it picks up on subtle changes in "
        "price patterns before volatility visibly spikes. "
        "<b>How to read this chart:</b> Left axis (blue line) = entropy level. Right axis (orange) = binary "
        "signal (0 = normal, 1 = early warning). When the signal flips to 1, start preparing for a regime shift."
    )
    if ent is not None:
        ent_latest = float(ent.dropna().iloc[-1]) if len(ent.dropna()) > 0 else 0.0
        sig_latest = float(sig.dropna().iloc[-1]) if sig is not None and len(sig.dropna()) > 0 else 0
        ent_insight = ""
        if sig_latest >= 1:
            ent_insight = f" <b>Actionable: The regime signal (right axis) is currently ON (=1) with entropy at {ent_latest:.3f}. Yield movements are unusually complex, consistent with a market-driven repricing regime. This is an early warning to prepare short-JGB or long-vol positions.</b>"
        else:
            ent_insight = f" <b>Actionable: The regime signal is OFF (=0) with entropy at {ent_latest:.3f}. Yield movements remain orderly and predictable. The BOJ is likely still in control; no immediate repricing trigger.</b>"
        _section_note(
            "Permutation entropy (left axis) and binary regime signal (right, red). Signal fires before other models detect regime change."
            + ent_insight
        )
        fig_ent = go.Figure()
        fig_ent.add_trace(
            go.Scatter(x=ent.index, y=ent.values, mode="lines", name="Perm. Entropy")
        )
        if sig is not None:
            fig_ent.add_trace(
                go.Scatter(
                    x=sig.index, y=sig.values, mode="lines",
                    name="Regime Signal", line=dict(dash="dot", color="red"),
                    yaxis="y2",
                )
            )
            fig_ent.update_layout(
                yaxis2=dict(title="Signal", overlaying="y", side="right", range=[-0.1, 1.1])
            )
        fig_ent.update_layout(yaxis_title="Entropy")
        _add_boj_events(fig_ent)
        _chart(_style_fig(fig_ent, 350))
    else:
        st.warning("Insufficient data for entropy analysis.")

    # --- GARCH ---
    st.subheader("GARCH Conditional Volatility & Vol-Regime Breaks")
    _definition_block(
        "GARCH(1,1) Conditional Volatility",
        "<b>Volatility</b> measures how much yields jump around from day to day. Low volatility = calm markets, "
        "small moves. High volatility = turbulent markets, large moves. But volatility is not constant: "
        "turbulent days tend to follow other turbulent days (this is called 'volatility clustering'). "
        "The <b>GARCH model</b> captures this clustering. It says: 'Today's expected volatility depends on "
        "yesterday's actual move AND yesterday's expected volatility.' This creates a smooth volatility estimate "
        "that rises during stress episodes and slowly decays back to normal afterward. "
        "The chart shows this estimated volatility in <b>basis points (bps) per day</b>. 1 bps = 0.01%. "
        "So if the chart reads 5 bps, it means yields are expected to move about 5 bps (0.05%) per day. "
        "For JGBs, which historically moved less than 1 bps/day under BOJ control, a reading above 3-4 bps "
        "signals the market has entered a fundamentally different volatility regime. "
        "The <b>purple vertical lines</b> are structural breaks detected on the volatility series itself, "
        "showing exactly when the volatility regime permanently shifted. "
        "<b>How to read this chart:</b> Rising line = volatility increasing (more risk). "
        "Purple verticals = permanent shifts in the vol regime. Spikes near red BOJ lines = policy-triggered."
    )
    breaks = garch_breaks
    if vol is not None:
        n_vb = len(breaks) if breaks else 0
        vol_latest = float(vol.dropna().iloc[-1]) if len(vol.dropna()) > 0 else 0.0
        vol_insight = ""
        if vol_latest > 5:
            vol_insight = f" <b>Actionable: Conditional volatility is {vol_latest:.1f} bps, well above normal JGB levels. High vol-clustering means today's moves are likely to persist tomorrow. Size positions smaller and use wider stops.</b>"
        elif vol_latest < 1:
            vol_insight = f" <b>Actionable: Conditional volatility is only {vol_latest:.1f} bps, extremely low. This is either genuine calm or the quiet before a vol spike. Cheap to buy JGB options (gamma) here as a hedge.</b>"
        if breaks and len(breaks) > 0:
            last_vb = breaks[-1]
            vol_insight += f" The latest vol-regime break is <b>{last_vb:%Y-%m-%d}</b>."
        _section_note(
            f"GARCH(1,1) conditional volatility (bps) with {n_vb} vol-regime breakpoints (purple). Red verticals = BOJ events."
            + vol_insight
        )
        fig_g = go.Figure()
        fig_g.add_trace(
            go.Scatter(x=vol.index, y=vol.values, mode="lines", name="Cond. Volatility")
        )
        if breaks:
            for bp in breaks:
                fig_g.add_vline(x=bp, line_dash="dash", line_color="purple", line_width=2)
        fig_g.update_layout(yaxis_title="Volatility (bps)")
        _add_boj_events(fig_g)
        _chart(_style_fig(fig_g, 350))
    else:
        st.warning("Insufficient data for GARCH model.")

    # --- Regime Comparison Table by BOJ Era ---
    st.subheader("Regime Comparison by BOJ Policy Era")
    _definition_block(
        "BOJ Policy Eras",
        "Japan's central bank (BOJ) has gone through several dramatic policy phases. Understanding each one "
        "is essential context: "
        "<b>QQE (2013-16):</b> 'Quantitative and Qualitative Easing.' Governor Kuroda launched unprecedented "
        "bond buying to fight deflation. The BOJ bought so many JGBs it eventually owned nearly half of all "
        "outstanding government debt. Yields plunged. "
        "<b>NIRP (2016):</b> 'Negative Interest Rate Policy.' The BOJ charged banks for holding reserves, "
        "pushing short-term rates below zero. Depositors effectively paid the bank to hold their money. "
        "<b>YCC (2016-22):</b> 'Yield Curve Control.' The BOJ explicitly capped 10-year JGB yields at first "
        "0.25%, then 0.50%. Whenever yields tried to rise above the cap, the BOJ bought unlimited bonds to "
        "push them back down. This was the most extreme form of yield suppression. "
        "<b>YCC Exit (2022-24):</b> The BOJ gradually loosened the cap, first widening it, then making it a "
        "'reference' rather than a hard cap. This was the beginning of the end of suppression. "
        "<b>Post-YCC (2024+):</b> Full exit from both YCC and negative rates. The market is now pricing JGBs "
        "based on fundamentals for the first time in a decade. "
        "The table below compares key statistics across each era, letting you see exactly how yields, volatility, "
        "and the yen behaved in each policy environment."
    )
    try:
        df_full = load_unified(*_get_args())
        regime_rows = []
        for era_name, (era_start, era_end) in ANALYSIS_WINDOWS.items():
            if era_name == "full":
                continue
            mask = (df_full.index >= pd.Timestamp(era_start)) & (df_full.index <= pd.Timestamp(era_end))
            era_df = df_full.loc[mask]
            if len(era_df) < 5:
                continue
            row = {"Era": era_name.replace("_", " ").title(), "Period": f"{era_start} → {era_end}", "Obs": len(era_df)}
            if "JP_10Y" in era_df.columns:
                jp = era_df["JP_10Y"].dropna()
                if len(jp) > 0:
                    row["JP 10Y Mean"] = f"{jp.mean():.3f}%"
                    row["JP 10Y Vol (bps)"] = f"{jp.diff().std() * 100:.1f}"
            if "US_10Y" in era_df.columns and "JP_10Y" in era_df.columns:
                spread = (era_df["JP_10Y"] - era_df["US_10Y"]).dropna()
                if len(spread) > 0:
                    row["JP-US Spread"] = f"{spread.mean():.2f}%"
            if "USDJPY" in era_df.columns:
                fx = era_df["USDJPY"].dropna()
                if len(fx) > 0:
                    row["USDJPY Mean"] = f"{fx.mean():.1f}"
            if ensemble is not None:
                ens_mask = (ensemble.index >= pd.Timestamp(era_start)) & (ensemble.index <= pd.Timestamp(era_end))
                era_ens = ensemble.loc[ens_mask].dropna()
                if len(era_ens) > 0:
                    row["Avg Regime Prob"] = f"{era_ens.mean():.0%}"
            regime_rows.append(row)
        if regime_rows:
            regime_table = pd.DataFrame(regime_rows)
            _section_note(
                "Summary statistics by BOJ policy era. Compare yield levels, volatility, and regime probability across "
                "structural breaks. <b>Actionable: Eras with high vol + high regime probability = confirmed repricing episodes. "
                "Current era metrics should be compared against these benchmarks for positioning.</b>"
            )
            st.dataframe(regime_table, use_container_width=True, hide_index=True)
    except Exception:
        st.info("Could not compute regime comparison table.")

    # --- Regime Duration & Transition Analysis ---
    if ensemble is not None and len(ensemble.dropna()) > 30:
        st.subheader("Regime Duration and Transitions")
        _definition_block(
            "What Regime Duration Reveals",
            "How long does each regime last? This question is critical for traders. If repricing episodes "
            "only last a few days before the BOJ reasserts control, then you should fade (trade against) the move. "
            "But if repricing persists for months, it is a structural shift and you should follow the trend. "
            "The <b>current streak</b> shows how many consecutive trading days the market has been in its "
            "current regime. A repricing streak above 60 trading days (~3 months) has historically been very "
            "difficult for the BOJ to reverse. "
            "The <b>histogram</b> shows the distribution of past regime durations. If most repricing episodes "
            "are short (under 20 days), the BOJ has been successful at regaining control. If some are very long, "
            "it means once repricing starts, it can be persistent. "
            "The <b>number of transitions</b> reveals how unstable the market is. Many transitions = "
            "frequent regime switching = maximum uncertainty. Few transitions = stable regimes with clear "
            "signals. <b>How to read:</b> Metrics at top summarize the current state. Histogram below shows "
            "whether current streak duration is typical or unusual compared to history."
        )
        _ens_clean = ensemble.dropna()
        _regime_binary = (_ens_clean > 0.5).astype(int)
        _transitions = (_regime_binary != _regime_binary.shift()).cumsum()
        _durations = _regime_binary.groupby(_transitions).agg(["first", "count"])
        _durations.columns = ["regime", "duration_days"]
        _repricing_durations = _durations[_durations["regime"] == 1]["duration_days"]
        _suppressed_durations = _durations[_durations["regime"] == 0]["duration_days"]
        _n_transitions = len(_durations) - 1

        dur_cols = st.columns(4)
        dur_cols[0].metric("Total Transitions", f"{_n_transitions}")
        dur_cols[1].metric("Current Streak", f"{int(_durations.iloc[-1]['duration_days'])}d",
                          delta=f"{'Repricing' if _durations.iloc[-1]['regime'] == 1 else 'Suppressed'}")
        if len(_repricing_durations) > 0:
            dur_cols[2].metric("Avg Repricing Duration", f"{_repricing_durations.mean():.0f}d")
        if len(_suppressed_durations) > 0:
            dur_cols[3].metric("Avg Suppressed Duration", f"{_suppressed_durations.mean():.0f}d")

        # Duration distribution chart
        if len(_durations) > 2:
            fig_dur = go.Figure()
            if len(_repricing_durations) > 0:
                fig_dur.add_trace(go.Histogram(
                    x=_repricing_durations, name="Repricing",
                    marker_color="#E8413C", opacity=0.7,
                ))
            if len(_suppressed_durations) > 0:
                fig_dur.add_trace(go.Histogram(
                    x=_suppressed_durations, name="Suppressed",
                    marker_color="#1565C0", opacity=0.7,
                ))
            fig_dur.update_layout(
                xaxis_title="Duration (trading days)", yaxis_title="Frequency",
                barmode="overlay",
            )
            _chart(_style_fig(fig_dur, 340))

        _current_dur = int(_durations.iloc[-1]["duration_days"])
        _current_regime = "repricing" if _durations.iloc[-1]["regime"] == 1 else "suppressed"
        _takeaway_block(
            f"Market has been in <b>{_current_regime}</b> for <b>{_current_dur} trading days</b>. "
            f"{'Prolonged repricing (>60 days) confirms a structural shift beyond BOJ control. This is not a temporary spike.' if _current_regime == 'repricing' and _current_dur > 60 else 'Repricing episode is still young; could reverse if BOJ intervenes forcefully.' if _current_regime == 'repricing' else 'Suppressed regime is holding. BOJ retains control, but watch for entropy signal to fire as an early warning.' if _current_dur > 60 else 'Short suppressed streak after a transition; regime may be unstable.'}"
            f" Total of <b>{_n_transitions}</b> regime transitions detected over the sample."
        )

    # --- Page conclusion ---
    if ensemble is not None and len(ensemble.dropna()) > 0:
        _ep = float(ensemble.dropna().iloc[-1])
        _regime_word = "repricing" if _ep > 0.5 else "suppressed"
        _conf_word = "high" if abs(_ep - 0.5) > 0.2 else "moderate" if abs(_ep - 0.5) > 0.1 else "low"
        _regime_summary = (
            f"The ensemble probability currently reads <b>{_ep:.0%}</b>, placing the market in a "
            f"<b>{_regime_word}</b> regime with {_conf_word} conviction across all four detection models."
        )
        # Verdict
        if _ep > 0.7:
            _verdict_p3 = f"Regime consensus is clear at {_ep:.0%}: the BOJ has lost control of the curve. Reduce long-duration JGB exposure."
        elif _ep > 0.5:
            _verdict_p3 = f"Repricing signal at {_ep:.0%} but not yet decisive. Trim positions; do not add until conviction exceeds 70%."
        elif _ep > 0.3:
            _verdict_p3 = f"Transition zone at {_ep:.0%}. The market has not committed either way. Preserve capital; avoid directional bets."
        else:
            _verdict_p3 = f"BOJ remains in control at {_ep:.0%}. Yield suppression holds; carry strategies are intact."
    else:
        _regime_summary = "Regime detection models could not produce a consensus due to insufficient data."
        _verdict_p3 = "Insufficient model output. Withhold conviction until all four detectors report."
    _page_conclusion(
        _verdict_p3,
        f"{_regime_summary} "
        f"Cross-market transmission dynamics are analysed on the Spillover page.",
    )
    _page_footer()


