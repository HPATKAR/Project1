"""
JGB Repricing Framework — Streamlit Dashboard

Multi-page dashboard visualising the full JGB repricing pipeline:
data overview, yield curve analytics, regime detection, cross-asset
spillover, and trade ideas.

Launch:  .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import sys
import os
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `src.*` imports resolve
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# src imports (all lazy — only called within their respective pages)
# ---------------------------------------------------------------------------
from src.data.data_store import DataStore
from src.data.config import BOJ_EVENTS, JGB_TENORS, DEFAULT_START, DEFAULT_END

# ---------------------------------------------------------------------------
# Global Streamlit config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="JGB Repricing Dashboard",
    page_icon="🇯🇵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — navigation + global controls
# ---------------------------------------------------------------------------
st.sidebar.title("JGB Repricing Framework")
page = st.sidebar.radio(
    "Navigate",
    [
        "1 — Overview & Data",
        "2 — Yield Curve Analytics",
        "3 — Regime Detection",
        "4 — Spillover & Info Flow",
        "5 — Trade Ideas",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Settings")

use_simulated = st.sidebar.toggle("Simulated data", value=True)
fred_api_key = st.sidebar.text_input("FRED API Key", type="password")
date_range = st.sidebar.date_input(
    "Date range",
    value=(DEFAULT_START, DEFAULT_END),
    min_value=date(2000, 1, 1),
    max_value=date.today(),
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = DEFAULT_START, DEFAULT_END


# ===================================================================
# Cached helpers
# ===================================================================
@st.cache_resource
def get_data_store(simulated: bool) -> DataStore:
    return DataStore(use_simulated=simulated)


@st.cache_data(show_spinner="Loading unified data…")
def load_unified(simulated: bool, start: str, end: str, api_key: str | None):
    store = get_data_store(simulated)
    store.clear_cache()
    return store.get_unified(start=start, end=end, fred_api_key=api_key or None)


@st.cache_data(show_spinner="Loading rates data…")
def load_rates(simulated: bool, start: str, end: str, api_key: str | None):
    store = get_data_store(simulated)
    return store.get_rates(start=start, end=end, fred_api_key=api_key or None)


@st.cache_data(show_spinner="Loading market data…")
def load_market(simulated: bool, start: str, end: str):
    store = get_data_store(simulated)
    return store.get_market(start=start, end=end)


def _safe_col(df: pd.DataFrame, col: str) -> pd.Series | None:
    """Return column as Series if present, else None."""
    if col in df.columns:
        return df[col].dropna()
    return None


def _add_boj_events(fig: go.Figure, y_pos: float | None = None) -> go.Figure:
    """Add vertical BOJ event lines to a plotly figure."""
    for dt_str, label in BOJ_EVENTS.items():
        fig.add_vline(
            x=dt_str, line_dash="dot", line_color="rgba(255,0,0,0.3)", line_width=1
        )
        if y_pos is not None:
            fig.add_annotation(
                x=dt_str,
                y=y_pos,
                text=label,
                showarrow=False,
                textangle=-90,
                font=dict(size=8, color="red"),
                yshift=10,
            )
    return fig


# ===================================================================
# Page 1 — Overview & Data
# ===================================================================
def page_overview():
    st.header("Overview & Data")

    try:
        df = load_unified(
            use_simulated,
            str(start_date),
            str(end_date),
            fred_api_key or None,
        )
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        return

    # --- KPI row ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Date Range", f"{df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}")
    c2.metric("Rows", f"{len(df):,}")
    c3.metric("Sources", "Simulated" if use_simulated else "FRED + yfinance")
    c4.metric("Columns", f"{df.shape[1]}")

    # --- Rates chart ---
    st.subheader("Sovereign Yields & VIX")
    rate_cols = [c for c in ["JP_10Y", "US_10Y", "DE_10Y", "VIX"] if c in df.columns]
    if rate_cols:
        fig = go.Figure()
        for col in rate_cols:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))
        fig.update_layout(
            height=420, legend=dict(orientation="h"), hovermode="x unified"
        )
        _add_boj_events(fig)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No rate columns found in data.")

    # --- Market chart ---
    st.subheader("FX & Equity")
    mkt_cols = [c for c in ["USDJPY", "EURJPY", "NIKKEI"] if c in df.columns]
    if mkt_cols:
        fig2 = go.Figure()
        for col in mkt_cols:
            fig2.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))
        fig2.update_layout(
            height=420, legend=dict(orientation="h"), hovermode="x unified"
        )
        _add_boj_events(fig2)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No market columns found in data.")

    # --- Raw data expander ---
    with st.expander("Raw data (last 20 rows)"):
        st.dataframe(df.tail(20), use_container_width=True)


# ===================================================================
# Page 2 — Yield Curve Analytics
# ===================================================================
@st.cache_data(show_spinner="Running PCA…")
def _run_pca(simulated, start, end, api_key):
    from src.yield_curve.pca import fit_yield_pca

    df = load_unified(simulated, start, end, api_key)
    # Use available yield columns for PCA
    yield_cols = [c for c in df.columns if c.startswith(("JP_", "US_", "DE_")) and "CPI" not in c and "CALL" not in c and "FF" not in c]
    if len(yield_cols) < 2:
        return None
    yield_df = df[yield_cols].dropna()
    changes = yield_df.diff().dropna()
    if len(changes) < 30:
        return None
    return fit_yield_pca(changes, n_components=min(3, len(yield_cols)))


@st.cache_data(show_spinner="Fitting Nelson-Siegel…")
def _run_ns(simulated, start, end, api_key):
    from src.yield_curve.nelson_siegel import fit_ns_timeseries

    df = load_unified(simulated, start, end, api_key)
    # Build yield panel for available JGB tenors
    jgb_cols = [f"JP_{t}Y" for t in JGB_TENORS if f"JP_{t}Y" in df.columns]
    if len(jgb_cols) < 3:
        # Fall back to any yield columns
        jgb_cols = [c for c in df.columns if c.startswith(("JP_", "US_")) and "CPI" not in c and "CALL" not in c and "FF" not in c]
    if len(jgb_cols) < 3:
        return None
    tenors = list(range(1, len(jgb_cols) + 1))
    yield_panel = df[jgb_cols].dropna()
    if len(yield_panel) < 30:
        return None
    # Sub-sample for speed: weekly
    yield_weekly = yield_panel.resample("W").last().dropna()
    if len(yield_weekly) < 10:
        return None
    return fit_ns_timeseries(yield_weekly, tenors=tenors)


@st.cache_data(show_spinner="Computing liquidity metrics…")
def _run_liquidity(simulated, start, end, api_key):
    from src.yield_curve.liquidity import roll_measure, composite_liquidity_index

    df = load_unified(simulated, start, end, api_key)
    jp10 = _safe_col(df, "JP_10Y")
    if jp10 is None or len(jp10) < 30:
        return None
    returns = jp10.diff().dropna()
    roll = roll_measure(returns, window=22)
    metrics = {"roll": roll}
    composite = composite_liquidity_index(metrics, method="z_score")
    return pd.DataFrame({"roll_measure": roll, "composite_index": composite})


def page_yield_curve():
    st.header("Yield Curve Analytics")

    args = (use_simulated, str(start_date), str(end_date), fred_api_key or None)

    # --- PCA ---
    st.subheader("PCA of Yield Changes")
    pca_result = _run_pca(*args)
    if pca_result is None:
        st.warning("Insufficient yield data for PCA.")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Explained Variance**")
            ev = pca_result["explained_variance_ratio"]
            fig_ev = go.Figure(
                go.Bar(
                    x=[f"PC{i+1}" for i in range(len(ev))],
                    y=ev,
                    text=[f"{v:.1%}" for v in ev],
                    textposition="outside",
                )
            )
            fig_ev.update_layout(height=320, yaxis_title="Variance Ratio")
            st.plotly_chart(fig_ev, use_container_width=True)

        with col_b:
            st.markdown("**Loadings Heatmap**")
            loadings = pca_result["loadings"]
            fig_ld = px.imshow(
                loadings.T.values,
                x=loadings.index.astype(str),
                y=[f"PC{i+1}" for i in range(loadings.shape[1])],
                color_continuous_scale="RdBu_r",
                aspect="auto",
            )
            fig_ld.update_layout(height=320)
            st.plotly_chart(fig_ld, use_container_width=True)

        st.markdown("**PCA Scores Over Time**")
        scores = pca_result["scores"]
        fig_sc = go.Figure()
        labels = {0: "PC1 (Level)", 1: "PC2 (Slope)", 2: "PC3 (Curvature)"}
        for i, col in enumerate(scores.columns):
            fig_sc.add_trace(
                go.Scatter(
                    x=scores.index,
                    y=scores[col],
                    mode="lines",
                    name=labels.get(i, col),
                )
            )
        fig_sc.update_layout(height=380, hovermode="x unified")
        _add_boj_events(fig_sc)
        st.plotly_chart(fig_sc, use_container_width=True)

    # --- Liquidity ---
    st.subheader("Liquidity Metrics")
    liq = _run_liquidity(*args)
    if liq is None:
        st.warning("Insufficient data for liquidity metrics.")
    else:
        fig_liq = go.Figure()
        for col in liq.columns:
            fig_liq.add_trace(
                go.Scatter(x=liq.index, y=liq[col], mode="lines", name=col)
            )
        fig_liq.update_layout(height=380, hovermode="x unified")
        _add_boj_events(fig_liq)
        st.plotly_chart(fig_liq, use_container_width=True)

    # --- Nelson-Siegel ---
    st.subheader("Nelson-Siegel Factor Evolution")
    ns_result = _run_ns(*args)
    if ns_result is None:
        st.warning("Insufficient data for Nelson-Siegel fitting.")
    else:
        fig_ns = go.Figure()
        ns_labels = {"beta0": "β0 (Level)", "beta1": "β1 (Slope)", "beta2": "β2 (Curvature)"}
        for col in ["beta0", "beta1", "beta2"]:
            if col in ns_result.columns:
                fig_ns.add_trace(
                    go.Scatter(
                        x=ns_result.index,
                        y=ns_result[col],
                        mode="lines",
                        name=ns_labels.get(col, col),
                    )
                )
        fig_ns.update_layout(height=380, hovermode="x unified")
        _add_boj_events(fig_ns)
        st.plotly_chart(fig_ns, use_container_width=True)


# ===================================================================
# Page 3 — Regime Detection
# ===================================================================
@st.cache_data(show_spinner="Fitting Markov regime model…")
def _run_markov(simulated, start, end, api_key):
    from src.regime.markov_switching import fit_markov_regime

    df = load_unified(simulated, start, end, api_key)
    jp10 = _safe_col(df, "JP_10Y")
    if jp10 is None or len(jp10) < 60:
        return None
    changes = jp10.diff().dropna()
    return fit_markov_regime(changes, k_regimes=2, switching_variance=True)


@st.cache_data(show_spinner="Fitting HMM…")
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


@st.cache_data(show_spinner="Detecting structural breaks…")
def _run_breaks(simulated, start, end, api_key):
    from src.regime.structural_breaks import detect_breaks_pelt

    df = load_unified(simulated, start, end, api_key)
    jp10 = _safe_col(df, "JP_10Y")
    if jp10 is None or len(jp10) < 120:
        return None, None
    changes = jp10.diff().dropna()
    bkps = detect_breaks_pelt(changes, min_size=60)
    return changes, bkps


@st.cache_data(show_spinner="Computing entropy…")
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


@st.cache_data(show_spinner="Fitting GARCH…")
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


@st.cache_data(show_spinner="Computing ensemble probability…")
def _run_ensemble(simulated, start, end, api_key):
    from src.regime.ensemble import ensemble_regime_probability

    markov = _run_markov(simulated, start, end, api_key)
    hmm = _run_hmm(simulated, start, end, api_key)
    ent, sig = _run_entropy(simulated, start, end, api_key)
    vol, breaks = _run_garch(simulated, start, end, api_key)

    if any(x is None for x in [markov, hmm, sig]):
        return None

    markov_prob = markov["regime_probabilities"]
    # Use the column most likely to represent the repricing regime
    prob_col = markov_prob.columns[-1]
    mp = markov_prob[prob_col]

    hmm_states = hmm["states"]
    garch_input = breaks if breaks is not None else []

    return ensemble_regime_probability(mp, hmm_states, sig, garch_input)


def page_regime():
    st.header("Regime Detection")

    args = (use_simulated, str(start_date), str(end_date), fred_api_key or None)

    # --- Ensemble Probability ---
    st.subheader("Ensemble Regime Probability")
    ensemble = _run_ensemble(*args)
    if ensemble is not None and len(ensemble.dropna()) > 0:
        current_prob = float(ensemble.dropna().iloc[-1])

        # Gauge / metric
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Current Regime Prob.", f"{current_prob:.2%}")
        regime_label = "REPRICING" if current_prob > 0.5 else "SUPPRESSED"
        colour = "🔴" if current_prob > 0.5 else "🟢"
        col_m2.metric("Regime", f"{colour} {regime_label}")
        col_m3.metric("Avg Prob (full sample)", f"{ensemble.mean():.2%}")

        fig_ens = go.Figure()
        fig_ens.add_trace(
            go.Scatter(
                x=ensemble.index, y=ensemble.values, mode="lines",
                name="Ensemble Prob", line=dict(color="steelblue"),
            )
        )
        fig_ens.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Threshold")
        fig_ens.update_layout(height=380, yaxis_title="Probability", hovermode="x unified")
        _add_boj_events(fig_ens)
        st.plotly_chart(fig_ens, use_container_width=True)
    else:
        st.warning("Could not compute ensemble probability — check data availability.")

    # --- Markov Smoothed Probabilities ---
    st.subheader("Markov-Switching Smoothed Probabilities")
    markov = _run_markov(*args)
    if markov is not None:
        rp = markov["regime_probabilities"]
        fig_mk = go.Figure()
        for col in rp.columns:
            fig_mk.add_trace(
                go.Scatter(
                    x=rp.index, y=rp[col], mode="lines", name=col,
                    stackgroup="one",
                )
            )
        fig_mk.update_layout(height=350, yaxis_title="Smoothed Probability", hovermode="x unified")
        _add_boj_events(fig_mk)
        st.plotly_chart(fig_mk, use_container_width=True)

        st.caption(
            f"Regime means: {markov['regime_means']}, "
            f"Regime variances: {markov['regime_variances']}"
        )
    else:
        st.warning("Insufficient data for Markov regime model.")

    # --- Structural Breaks ---
    st.subheader("Structural Breakpoints on JP 10Y Changes")
    changes, bkps = _run_breaks(*args)
    if changes is not None and bkps is not None:
        fig_bp = go.Figure()
        fig_bp.add_trace(
            go.Scatter(x=changes.index, y=changes.values, mode="lines", name="JP_10Y Δ")
        )
        for bp in bkps:
            fig_bp.add_vline(x=bp, line_dash="dash", line_color="orange", line_width=2)
        fig_bp.update_layout(height=350, hovermode="x unified")
        _add_boj_events(fig_bp)
        st.plotly_chart(fig_bp, use_container_width=True)
    else:
        st.warning("Insufficient data for structural break detection.")

    # --- Entropy ---
    st.subheader("Rolling Permutation Entropy & Regime Signal")
    ent, sig = _run_entropy(*args)
    if ent is not None:
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
        fig_ent.update_layout(height=350, yaxis_title="Entropy", hovermode="x unified")
        _add_boj_events(fig_ent)
        st.plotly_chart(fig_ent, use_container_width=True)
    else:
        st.warning("Insufficient data for entropy analysis.")

    # --- GARCH ---
    st.subheader("GARCH Conditional Volatility & Vol-Regime Breaks")
    vol, breaks = _run_garch(*args)
    if vol is not None:
        fig_g = go.Figure()
        fig_g.add_trace(
            go.Scatter(x=vol.index, y=vol.values, mode="lines", name="Cond. Volatility")
        )
        if breaks:
            for bp in breaks:
                fig_g.add_vline(x=bp, line_dash="dash", line_color="purple", line_width=2)
        fig_g.update_layout(height=350, yaxis_title="Volatility (bps)", hovermode="x unified")
        _add_boj_events(fig_g)
        st.plotly_chart(fig_g, use_container_width=True)
    else:
        st.warning("Insufficient data for GARCH model.")


# ===================================================================
# Page 4 — Spillover & Information Flow
# ===================================================================
@st.cache_data(show_spinner="Running Granger causality tests…")
def _run_granger(simulated, start, end, api_key):
    from src.spillover.granger import pairwise_granger

    df = load_unified(simulated, start, end, api_key)
    cols = [c for c in ["JP_10Y", "US_10Y", "DE_10Y", "USDJPY", "NIKKEI", "VIX"] if c in df.columns]
    if len(cols) < 2:
        return None
    sub = df[cols].diff().dropna()
    if len(sub) < 30:
        return None
    return pairwise_granger(sub, max_lag=5, significance=0.05)


@st.cache_data(show_spinner="Computing transfer entropy…")
def _run_te(simulated, start, end, api_key):
    from src.spillover.transfer_entropy import pairwise_transfer_entropy

    df = load_unified(simulated, start, end, api_key)
    cols = [c for c in ["JP_10Y", "US_10Y", "DE_10Y", "USDJPY", "NIKKEI", "VIX"] if c in df.columns]
    if len(cols) < 2:
        return None
    sub = df[cols].diff().dropna()
    if len(sub) < 30:
        return None
    return pairwise_transfer_entropy(sub, lag=1, n_bins=3)


@st.cache_data(show_spinner="Computing Diebold-Yilmaz spillover index…")
def _run_spillover(simulated, start, end, api_key):
    from src.spillover.diebold_yilmaz import compute_spillover_index

    df = load_unified(simulated, start, end, api_key)
    cols = [c for c in ["JP_10Y", "US_10Y", "DE_10Y", "USDJPY", "NIKKEI"] if c in df.columns]
    if len(cols) < 2:
        return None
    sub = df[cols].diff().dropna()
    if len(sub) < 50:
        return None
    return compute_spillover_index(sub, var_lags=4, forecast_horizon=10)


@st.cache_data(show_spinner="Computing DCC-GARCH correlations…")
def _run_dcc(simulated, start, end, api_key):
    from src.spillover.dcc_garch import compute_dcc

    df = load_unified(simulated, start, end, api_key)
    cols = [c for c in ["JP_10Y", "US_10Y", "USDJPY", "NIKKEI"] if c in df.columns]
    if len(cols) < 2:
        return None
    sub = df[cols].diff().dropna() * 100  # scale for GARCH
    if len(sub) < 60:
        return None
    return compute_dcc(sub, p=1, q=1)


@st.cache_data(show_spinner="Computing FX carry metrics…")
def _run_carry(simulated, start, end, api_key):
    from src.fx.carry_analytics import compute_carry, carry_to_vol

    df = load_unified(simulated, start, end, api_key)
    jp_rate = _safe_col(df, "JP_CALL_RATE")
    us_rate = _safe_col(df, "US_FF")
    usdjpy = _safe_col(df, "USDJPY")

    if jp_rate is None or us_rate is None or usdjpy is None:
        # Try fallbacks
        jp_rate = _safe_col(df, "JP_10Y")
        us_rate = _safe_col(df, "US_10Y")
        if jp_rate is None or us_rate is None or usdjpy is None:
            return None

    carry = compute_carry(us_rate, jp_rate)
    fx_returns = usdjpy.pct_change().dropna()
    fx_vol = fx_returns.rolling(63).std() * np.sqrt(252)
    fx_vol = fx_vol.dropna()
    # Align
    common_idx = carry.index.intersection(fx_vol.index)
    if len(common_idx) < 30:
        return None
    carry_aligned = carry.loc[common_idx]
    fx_vol_aligned = fx_vol.loc[common_idx]
    ctv = carry_to_vol(carry_aligned, fx_vol_aligned)
    return pd.DataFrame({
        "carry": carry_aligned,
        "realized_vol": fx_vol_aligned,
        "carry_to_vol": ctv,
    })


def page_spillover():
    st.header("Spillover & Information Flow")

    args = (use_simulated, str(start_date), str(end_date), fred_api_key or None)

    # --- Granger Causality ---
    st.subheader("Granger Causality (significant pairs)")
    granger_df = _run_granger(*args)
    if granger_df is not None and not granger_df.empty:
        sig_df = granger_df[granger_df["significant"] == True].reset_index(drop=True)
        if not sig_df.empty:
            st.dataframe(
                sig_df.style.highlight_max(subset=["f_stat"], color="lightyellow"),
                use_container_width=True,
            )
        else:
            st.info("No significant Granger-causal pairs at 5% level.")
        with st.expander("Full Granger results"):
            st.dataframe(granger_df, use_container_width=True)
    else:
        st.warning("Insufficient data for Granger causality.")

    # --- Transfer Entropy Heatmap ---
    st.subheader("Transfer Entropy Heatmap")
    te_df = _run_te(*args)
    if te_df is not None and not te_df.empty:
        sources = te_df["source"].unique()
        targets = te_df["target"].unique()
        all_labels = sorted(set(sources) | set(targets))
        te_matrix = pd.DataFrame(0.0, index=all_labels, columns=all_labels)
        for _, row in te_df.iterrows():
            te_matrix.loc[row["source"], row["target"]] = row["te_value"]

        fig_te = px.imshow(
            te_matrix.values,
            x=te_matrix.columns.tolist(),
            y=te_matrix.index.tolist(),
            color_continuous_scale="Viridis",
            aspect="auto",
            labels=dict(color="TE"),
        )
        fig_te.update_layout(height=450)
        st.plotly_chart(fig_te, use_container_width=True)
    else:
        st.warning("Insufficient data for transfer entropy.")

    # --- Diebold-Yilmaz ---
    st.subheader("Diebold-Yilmaz Spillover")
    spill = _run_spillover(*args)
    if spill is not None:
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            st.metric("Total Spillover Index", f"{spill['total_spillover']:.1f}%")
        with col_s2:
            net = spill["net_spillover"]
            fig_net = go.Figure(
                go.Bar(x=net.index.tolist(), y=net.values, marker_color=["green" if v > 0 else "red" for v in net.values])
            )
            fig_net.update_layout(height=320, title="Net Directional Spillover", yaxis_title="Net (%)")
            st.plotly_chart(fig_net, use_container_width=True)

        with st.expander("Spillover matrix"):
            st.dataframe(spill["spillover_matrix"].round(2), use_container_width=True)
    else:
        st.warning("Insufficient data for spillover analysis.")

    # --- DCC ---
    st.subheader("DCC Time-Varying Correlations")
    dcc = _run_dcc(*args)
    if dcc is not None:
        cond_corr = dcc["conditional_correlations"]
        if cond_corr:
            fig_dcc = go.Figure()
            for pair, series in cond_corr.items():
                fig_dcc.add_trace(
                    go.Scatter(x=series.index, y=series.values, mode="lines", name=pair)
                )
            fig_dcc.update_layout(
                height=380, yaxis_title="Conditional Correlation", hovermode="x unified"
            )
            _add_boj_events(fig_dcc)
            st.plotly_chart(fig_dcc, use_container_width=True)
        else:
            st.info("No correlation pairs computed.")
    else:
        st.warning("Insufficient data for DCC-GARCH.")

    # --- FX Carry ---
    st.subheader("FX Carry Metrics (USD/JPY)")
    carry = _run_carry(*args)
    if carry is not None:
        fig_c = go.Figure()
        fig_c.add_trace(
            go.Scatter(x=carry.index, y=carry["carry"], mode="lines", name="Carry")
        )
        fig_c.add_trace(
            go.Scatter(x=carry.index, y=carry["realized_vol"], mode="lines", name="Realized Vol")
        )
        fig_c.add_trace(
            go.Scatter(
                x=carry.index, y=carry["carry_to_vol"], mode="lines",
                name="Carry / Vol", line=dict(dash="dot"),
                yaxis="y2",
            )
        )
        fig_c.update_layout(
            height=380,
            yaxis_title="Rate / Vol",
            yaxis2=dict(title="Carry-to-Vol Ratio", overlaying="y", side="right"),
            hovermode="x unified",
        )
        _add_boj_events(fig_c)
        st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.warning("Insufficient data for carry analytics.")


# ===================================================================
# Page 5 — Trade Ideas
# ===================================================================
@st.cache_data(show_spinner="Generating trade ideas…")
def _generate_trades(simulated, start, end, api_key):
    from src.strategy.trade_generator import generate_all_trades
    from src.yield_curve.term_premium import estimate_acm_term_premium

    df = load_unified(simulated, start, end, api_key)

    # Gather regime state inputs
    # Regime probability
    ensemble = _run_ensemble(simulated, start, end, api_key)
    regime_prob = float(ensemble.dropna().iloc[-1]) if ensemble is not None and len(ensemble.dropna()) > 0 else 0.5

    # PCA scores
    pca_res = _run_pca(simulated, start, end, api_key)
    pca_scores = pca_res["scores"] if pca_res is not None else pd.DataFrame({"PC1": [0], "PC2": [0], "PC3": [0]})

    # Term premium
    yield_cols = [c for c in df.columns if c.startswith(("JP_", "US_")) and "CPI" not in c and "CALL" not in c and "FF" not in c]
    if len(yield_cols) >= 3:
        try:
            tenors = list(range(1, len(yield_cols) + 1))
            tp_df = estimate_acm_term_premium(df[yield_cols].dropna(), tenors=tenors, n_factors=min(3, len(yield_cols)))
            term_premium = tp_df["term_premium"]
        except Exception:
            term_premium = pd.Series(np.zeros(100), index=pd.date_range("2020-01-01", periods=100, freq="B"))
    else:
        term_premium = pd.Series(np.zeros(100), index=pd.date_range("2020-01-01", periods=100, freq="B"))

    # Liquidity
    liq = _run_liquidity(simulated, start, end, api_key)
    liquidity_index = liq["composite_index"] if liq is not None else pd.Series(np.zeros(100), index=pd.date_range("2020-01-01", periods=100, freq="B"))

    # Carry
    carry_df = _run_carry(simulated, start, end, api_key)
    if carry_df is not None and len(carry_df) > 0:
        ctv_val = float(carry_df["carry_to_vol"].dropna().iloc[-1]) if len(carry_df["carry_to_vol"].dropna()) > 0 else 1.0
    else:
        ctv_val = 1.0

    # USDJPY trend
    usdjpy = _safe_col(df, "USDJPY")
    if usdjpy is not None and len(usdjpy) >= 20:
        usdjpy_trend = float((usdjpy.iloc[-1] / usdjpy.iloc[-20] - 1) * 100)
    else:
        usdjpy_trend = 0.0

    # Entropy signal
    _, sig = _run_entropy(simulated, start, end, api_key)
    if sig is not None and len(sig.dropna()) > 0:
        entropy_signal = float(sig.dropna().iloc[-1])
    else:
        entropy_signal = 0.5

    # GARCH vol
    vol, breaks = _run_garch(simulated, start, end, api_key)
    if vol is not None and len(vol.dropna()) > 0:
        garch_vol = float(vol.dropna().iloc[-1]) / 100  # back to decimal
    else:
        garch_vol = 0.02

    # Spillover
    spill = _run_spillover(simulated, start, end, api_key)
    spillover_index = float(spill["total_spillover"]) if spill is not None else 50.0

    # TE network
    te_df = _run_te(simulated, start, end, api_key)
    te_network = None
    if te_df is not None and not te_df.empty:
        sources = te_df["source"].unique()
        targets = te_df["target"].unique()
        all_labels = sorted(set(sources) | set(targets))
        te_matrix = pd.DataFrame(0.0, index=all_labels, columns=all_labels)
        for _, row in te_df.iterrows():
            te_matrix.loc[row["source"], row["target"]] = row["te_value"]
        te_network = te_matrix

    # DCC
    dcc_res = _run_dcc(simulated, start, end, api_key)
    dcc_correlations = None

    regime_state = {
        "regime_prob": regime_prob,
        "term_premium": term_premium,
        "pca_scores": pca_scores,
        "liquidity_index": liquidity_index,
        "carry_to_vol": ctv_val,
        "usdjpy_trend": usdjpy_trend,
        "positioning": 0.0,  # no live CFTC data in simulated mode
        "garch_vol": garch_vol,
        "entropy_signal": entropy_signal,
        "spillover_index": spillover_index,
        "te_network": te_network,
        "dcc_correlations": dcc_correlations,
    }

    cards = generate_all_trades(regime_state)
    return cards, regime_state


def page_trade_ideas():
    st.header("Trade Ideas")

    args = (use_simulated, str(start_date), str(end_date), fred_api_key or None)

    try:
        cards, regime_state = _generate_trades(*args)
    except Exception as exc:
        st.error(f"Trade generation failed: {exc}")
        import traceback
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        return

    if not cards:
        st.info("No trade ideas generated for the current regime state. Try adjusting date range or data mode.")
        st.json({k: (str(v)[:80] if not isinstance(v, (int, float)) else v) for k, v in regime_state.items()})
        return

    from src.strategy.trade_card import trade_cards_to_dataframe

    # --- Summary metrics ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades", len(cards))
    categories = {}
    for card in cards:
        categories[card.category] = categories.get(card.category, 0) + 1
    c2.metric("Categories", ", ".join(f"{k}: {v}" for k, v in categories.items()))
    avg_conviction = np.mean([c.conviction for c in cards])
    c3.metric("Avg Conviction", f"{avg_conviction:.0%}")
    c4.metric("Regime Prob", f"{regime_state['regime_prob']:.2%}")

    # --- Sidebar filters ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Trade Filters")
    all_cats = sorted(set(c.category for c in cards))
    selected_cats = st.sidebar.multiselect("Categories", all_cats, default=all_cats)
    min_conv, max_conv = st.sidebar.slider("Conviction range", 0.0, 1.0, (0.0, 1.0), 0.05)

    filtered = [
        c for c in cards
        if c.category in selected_cats and min_conv <= c.conviction <= max_conv
    ]

    # --- Conviction bar chart ---
    st.subheader("Conviction Distribution")
    if filtered:
        conv_data = pd.DataFrame({
            "Trade": [c.name for c in filtered],
            "Conviction": [c.conviction for c in filtered],
            "Category": [c.category for c in filtered],
        })
        fig_conv = px.bar(
            conv_data, x="Trade", y="Conviction", color="Category",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_conv.update_layout(height=350, xaxis_tickangle=-45)
        st.plotly_chart(fig_conv, use_container_width=True)

    # --- Trade cards ---
    st.subheader(f"Trade Cards ({len(filtered)} shown)")
    for card in sorted(filtered, key=lambda c: -c.conviction):
        direction_arrow = "⬆️" if card.direction == "long" else "⬇️"
        conv_colour = "🟢" if card.conviction >= 0.7 else ("🟡" if card.conviction >= 0.4 else "🔴")

        with st.expander(
            f"{direction_arrow} **{card.name}** | {conv_colour} {card.conviction:.0%} | {card.category}"
        ):
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.markdown(f"**Direction:** {card.direction.upper()}")
                st.markdown(f"**Instruments:** {', '.join(card.instruments)}")
                st.markdown(f"**Regime Condition:** {card.regime_condition}")
                st.markdown(f"**Edge Source:** {card.edge_source}")
            with col_r:
                st.markdown(f"**Entry Signal:** {card.entry_signal}")
                st.markdown(f"**Exit Signal:** {card.exit_signal}")
                st.markdown(f"**Failure Scenario:** {card.failure_scenario}")
                st.markdown(f"**Sizing:** {card.sizing_method}")

    # --- Export ---
    st.markdown("---")
    if filtered:
        df_export = trade_cards_to_dataframe(filtered)
        csv = df_export.to_csv(index=False)
        st.download_button(
            "Download Trade Cards as CSV",
            data=csv,
            file_name="jgb_trade_cards.csv",
            mime="text/csv",
        )


# ===================================================================
# Router
# ===================================================================
if page.startswith("1"):
    page_overview()
elif page.startswith("2"):
    page_yield_curve()
elif page.startswith("3"):
    page_regime()
elif page.startswith("4"):
    page_spillover()
elif page.startswith("5"):
    page_trade_ideas()
