"""Yield Curve Analytics page: PCA, Nelson-Siegel, Liquidity, Term Premium."""

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
from src.pages._data import load_unified


def _get_args():
    """Retrieve sidebar args from session state."""
    return st.session_state["_app_args"]


def _get_layout_config():
    return st.session_state["_layout_config"]


def _get_alert_notifier():
    return st.session_state.get("_alert_notifier")

from src.data.config import JGB_TENORS



@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
def _run_pca(simulated, start, end, api_key):
    from src.yield_curve.pca import fit_yield_pca, validate_pca_factors

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



@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
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



@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
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
    _page_intro(
        "Three complementary methods decompose the yield curve into interpretable factors. "
        "PCA extracts statistically orthogonal drivers (Level, Slope, Curvature) from daily yield changes. "
        "Nelson-Siegel fits a parametric model to the curve shape, tracking its evolution over time. "
        "The Roll liquidity measure estimates implicit bid-ask spreads from price reversals, revealing "
        "when the market can no longer absorb flow without outsized price impact."
    )

    args = _get_args()

    # Pre-compute all models in a single pass
    with st.spinner("Computing yield curve analytics..."):
        pca_result = _run_pca(*args)
        liq = _run_liquidity(*args)
        ns_result = _run_ns(*args)

    # --- PCA ---
    st.subheader("PCA of Yield Changes")
    _definition_block(
        "What is PCA?",
        "Imagine you have yields for 2-year, 5-year, 10-year, 20-year, and 30-year bonds, all moving every day. "
        "That is a lot of data. <b>Principal Component Analysis (PCA)</b> is a statistical technique that finds "
        "the <em>main patterns</em> hidden in all that movement. It answers: 'What are the 2-3 most important "
        "forces driving the entire yield curve?' "
        "In bond markets worldwide, PCA almost always finds three patterns: "
        "<b>PC1 (Level)</b>: All yields move up or down together. This explains 60-80% of all movement. "
        "Think of it as 'the market collectively decides rates should be higher or lower.' "
        "<b>PC2 (Slope)</b>: Short-term yields move one way while long-term yields move the opposite way. "
        "This is the curve 'twisting'. It explains 10-20% of movement and reflects expectations about "
        "future rate changes (e.g. markets expect rate cuts soon, so short rates fall but long rates hold). "
        "<b>PC3 (Curvature)</b>: The middle of the curve (5-7 year) moves differently from both ends. "
        "This 'butterfly' pattern explains 5-10% and often reflects specific supply/demand at certain maturities. "
        "<b>How to read the charts below:</b> The bar chart shows how much each factor explains (PC1 should "
        "dominate). The heatmap shows which bonds each factor affects most (red = positive, blue = negative). "
        "The line chart over time shows when each factor was active."
    )
    if pca_result is None:
        st.warning("Insufficient yield data for PCA.")
    else:
        ev = pca_result["explained_variance_ratio"]
        pc1_pct = ev[0] if len(ev) > 0 else 0
        _section_note(
            f"Variance decomposition and factor loadings. PC1 explains <b>{pc1_pct:.1%}</b> of all yield movements."
            f" <b>Actionable: If PC1 >{80}%, the entire curve is moving in lockstep, indicating a broad repricing event, not a local kink.</b>"
        )
        col_a, col_b = st.columns(2)

        with col_a:
            _section_note("Explained variance by component")
            fig_ev = go.Figure(
                go.Bar(
                    x=[f"PC{i+1}" for i in range(len(ev))],
                    y=ev,
                    text=[f"{v:.1%}" for v in ev],
                    textposition="outside",
                )
            )
            fig_ev.update_layout(yaxis_title="Variance Ratio")
            _chart(_style_fig(fig_ev, 320))

        with col_b:
            loadings = pca_result["loadings"]
            # loadings shape: (PCs x securities) — rows=PC1/PC2/PC3, cols=JP_10Y/US_10Y/...
            sec_names = list(loadings.columns.astype(str))
            pc_labels = list(loadings.index.astype(str))
            # Build data-driven note from actual securities
            _ld_parts = []
            for ci in range(min(loadings.shape[0], 3)):
                abs_row = loadings.iloc[ci, :].abs()
                top_sec = abs_row.idxmax()
                _ld_parts.append(f"{pc_labels[ci]} loads heaviest on <b>{top_sec}</b>")
            _section_note(
                "; ".join(_ld_parts) + "." if _ld_parts else "Factor loadings across yield series."
            )
            fig_ld = px.imshow(
                loadings.values,
                x=sec_names,
                y=pc_labels,
                color_continuous_scale="RdBu_r",
                aspect="auto",
            )
            _chart(_style_fig(fig_ld, 320))

        # --- PCA Loadings by Security (line chart) ---
        loadings_line = pca_result["loadings"]
        # loadings_line shape: (PCs x securities)
        _sec_names_lc = list(loadings_line.columns.astype(str))
        if len(_sec_names_lc) >= 2 and loadings_line.shape[0] >= 1:
            # Build security-level insight
            _lc_note = f"Beta weights across {', '.join(_sec_names_lc[:5])}{'...' if len(_sec_names_lc) > 5 else ''}. "
            if loadings_line.shape[0] >= 2:
                pc1_row = loadings_line.iloc[0, :]  # PC1 loadings across securities
                pc2_row = loadings_line.iloc[1, :]  # PC2 loadings across securities
                pc1_range = pc1_row.max() - pc1_row.min()
                pc2_range = pc2_row.max() - pc2_row.min()
                if pc1_range < 0.15:
                    _lc_note += "PC1 is near-flat: the level factor moves all securities in lockstep."
                else:
                    _pc1_top = pc1_row.abs().idxmax()
                    _lc_note += f"PC1 tilts toward <b>{_pc1_top}</b>, indicating uneven parallel exposure."
                if pc2_range > 0.2:
                    _pc2_hi = pc2_row.idxmax()
                    _pc2_lo = pc2_row.idxmin()
                    _lc_note += f" PC2 slope runs from <b>{_pc2_lo}</b> to <b>{_pc2_hi}</b>."
            _section_note(_lc_note)
            fig_pcl = go.Figure()
            pc_names = {0: "PC1 (Level)", 1: "PC2 (Slope)", 2: "PC3 (Curvature)"}
            for i in range(loadings_line.shape[0]):  # iterate over PCs (rows)
                fig_pcl.add_trace(
                    go.Scatter(
                        x=_sec_names_lc,
                        y=loadings_line.iloc[i, :].values,
                        mode="lines+markers",
                        name=pc_names.get(i, f"PC{i+1}"),
                        marker=dict(size=6),
                    )
                )
            fig_pcl.update_layout(
                yaxis_title="Loading (beta)",
                xaxis_title="Security",
            )
            _chart(_style_fig(fig_pcl, 340))

        # --- PCA Factor Validation ---
        from src.yield_curve.pca import validate_pca_factors
        pca_validation = validate_pca_factors(pca_result)
        st.markdown("**Factor Validation** (Litterman-Scheinkman 1991)")
        val_cols = st.columns(len(pca_validation["factor_checks"]))
        for vc, (name, passed, detail) in zip(val_cols, pca_validation["factor_checks"]):
            vc.metric(name, "PASS" if passed else "FAIL", help=detail)
        cum_var = pca_validation["cumulative_variance"]
        _section_note(
            f"Cumulative explained variance: <b>{cum_var:.1%}</b>. "
            f"Literature benchmark: PC1-3 should explain >95% of yield curve movements. "
            + (f"<b>Validation confirms standard Level/Slope/Curvature factor structure.</b>" if sum(1 for _, p, _ in pca_validation["factor_checks"] if p) >= 3
               else f"<b>Some factors deviate from classical structure — review loadings for regime-specific dynamics.</b>")
        )

        scores = pca_result["scores"]
        # Compute recent PC1 trend for actionable insight
        _pc1 = scores.iloc[:, 0].dropna() if scores.shape[1] > 0 else pd.Series(dtype=float)
        pc1_insight = ""
        if len(_pc1) >= 20:
            pc1_recent = float(_pc1.iloc[-20:].mean())
            pc1_earlier = float(_pc1.iloc[-60:-20].mean()) if len(_pc1) >= 60 else float(_pc1.iloc[:len(_pc1)//2].mean())
            if pc1_recent > pc1_earlier + 0.5:
                pc1_insight = " <b>Actionable: PC1 (Level) has been trending upward recently. All yields are rising in unison, signalling a broad repricing move. Consider positioning for higher rates.</b>"
            elif pc1_recent < pc1_earlier - 0.5:
                pc1_insight = " <b>Actionable: PC1 (Level) is trending downward. Yields are compressing across maturities, suggesting the BOJ suppression regime is reasserting control.</b>"
        _section_note(
            "PC1 (Level), PC2 (Slope), PC3 (Curvature) factor scores over time. Spikes near red BOJ verticals confirm policy-driven repricing."
            + pc1_insight
        )
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
        _add_boj_events(fig_sc)
        _chart(_style_fig(fig_sc, 380))

        # PCA takeaway
        _takeaway_block(
            f"The first three principal components explain <b>{sum(ev[:3]):.1%}</b> of total yield curve variance. "
            f"{'PC1 dominance (>' + f'{ev[0]:.0%}' + ') means the entire curve is moving as a unit, consistent with a broad repricing event rather than isolated tenor moves.' if ev[0] > 0.75 else 'Variance is spread more evenly across factors, suggesting curve-shape trades (steepeners, butterflies) may be more relevant than outright directional bets.'}"
        )

    # --- Liquidity ---
    st.subheader("Liquidity Metrics")
    _definition_block(
        "What is the Roll Measure?",
        "<b>Liquidity</b> in bond markets means how easily you can buy or sell without moving the price. "
        "In a liquid market, you can trade large amounts and the price barely budges. In an illiquid market, "
        "even a small trade can cause a big price swing. "
        "The <b>bid-ask spread</b> is the gap between the price buyers are willing to pay (bid) and the price "
        "sellers are asking (ask). A wide spread means poor liquidity, because you lose money just entering and "
        "exiting a position. The problem is: JGB bid-ask spreads are not publicly available in real-time. "
        "The <b>Roll measure</b> (Roll, 1984) cleverly <em>estimates</em> the bid-ask spread from price patterns. "
        "When prices bounce back and forth (buy at ask, sell at bid), it creates a negative pattern in consecutive "
        "price changes. The Roll measure detects this pattern and converts it into an estimated spread. "
        "The <b>composite liquidity index</b> converts this into a z-score (standard deviations from average). "
        "Below -1 = liquidity is worse than usual (danger zone). Above +1 = liquidity is better than usual. "
        "<b>Why it matters for JGBs:</b> When the BOJ was buying most JGBs, liquidity was artificially good. "
        "As BOJ buying slows, liquidity deteriorates, and any repricing shock gets amplified because there are "
        "fewer buyers to absorb selling pressure."
    )
    if liq is None:
        st.warning("Insufficient data for liquidity metrics.")
    else:
        _comp_latest = float(liq["composite_index"].dropna().iloc[-1]) if len(liq["composite_index"].dropna()) > 0 else 0.0
        liq_insight = ""
        if _comp_latest < -1:
            liq_insight = f" <b>Actionable: Composite index at {_comp_latest:.2f} (below -1 z-score). Liquidity is deteriorating; expect larger price gaps on any repricing shock. Reduce position sizes or widen stop-losses.</b>"
        elif _comp_latest > 1:
            liq_insight = f" <b>Actionable: Composite index at {_comp_latest:.2f} (above +1 z-score). Liquidity is healthy; market can absorb order flow without outsized price impact.</b>"
        _section_note(
            "Roll measure (implicit bid-ask) and composite liquidity z-score. Spikes at BOJ events = liquidity withdrawal."
            + liq_insight
        )
        fig_liq = go.Figure()
        for col in liq.columns:
            fig_liq.add_trace(
                go.Scatter(x=liq.index, y=liq[col], mode="lines", name=col)
            )
        _add_boj_events(fig_liq)
        _chart(_style_fig(fig_liq, 380))

        # Liquidity takeaway
        _comp = liq["composite_index"].dropna()
        if len(_comp) > 0:
            _c_last = float(_comp.iloc[-1])
            _c_mean = float(_comp.mean())
            _takeaway_block(
                f"Composite liquidity z-score is <b>{_c_last:+.2f}</b> (sample mean: {_c_mean:+.2f}). "
                f"{'Liquidity is thin. During repricing episodes, thin liquidity amplifies price moves and can trigger stop-loss cascades. Reduce position sizes.' if _c_last < -0.5 else 'Liquidity is adequate. The market can absorb reasonable order flow without outsized price impact.' if _c_last > 0.5 else 'Liquidity is neutral. No immediate concerns, but monitor around BOJ meeting dates when depth typically thins.'}"
            )

    # --- Nelson-Siegel ---
    st.subheader("Nelson-Siegel Factor Evolution")
    _definition_block(
        "What is the Nelson-Siegel Model?",
        "While PCA (above) finds patterns statistically, the <b>Nelson-Siegel model</b> fits the yield curve "
        "with a specific mathematical formula that has built-in economic meaning. It describes any yield curve "
        "using just three numbers: "
        "<b>&beta;0 (Level)</b>: Where yields converge in the very long run. Think of it as the market's "
        "answer to 'What should interest rates be 30+ years from now?' If &beta;0 rises over time, it means "
        "the market is structurally repricing its long-run expectations for Japanese rates upward. "
        "<b>&beta;1 (Slope)</b>: The difference between short-term and long-term yields. Negative &beta;1 means "
        "the curve slopes upward (normal: you earn more for lending longer). If &beta;1 becomes more negative, "
        "the curve is steepening, meaning long-term rates are rising faster than short-term ones. "
        "<b>&beta;2 (Curvature)</b>: Whether the middle of the curve (around 5-7 years) is higher or lower "
        "than you would expect from a smooth line between short and long ends. Positive = the belly is 'humped' "
        "(higher than expected), often reflecting specific demand from institutional investors at those maturities. "
        "<b>How to read this chart:</b> Watch &beta;0 over time. If it trends upward, the repricing thesis is "
        "confirmed: the market believes Japanese rates will be permanently higher."
    )
    if ns_result is None:
        st.warning("Insufficient data for Nelson-Siegel fitting.")
    else:
        _b0 = ns_result["beta0"].dropna() if "beta0" in ns_result.columns else pd.Series(dtype=float)
        _b1 = ns_result["beta1"].dropna() if "beta1" in ns_result.columns else pd.Series(dtype=float)
        ns_insight = ""
        if len(_b0) >= 10:
            b0_start, b0_end = float(_b0.iloc[0]), float(_b0.iloc[-1])
            b0_chg = b0_end - b0_start
            if b0_chg > 0.1:
                ns_insight += f" <b>Actionable: β0 (Level) rose {b0_chg:+.2f} over the sample. The market has structurally repriced the long-run yield floor upward. This is the core confirmation of a JGB repricing regime.</b>"
            elif b0_chg < -0.1:
                ns_insight += f" <b>Actionable: β0 (Level) fell {b0_chg:+.2f}. Long-run yield expectations are declining, consistent with continued BOJ suppression.</b>"
        if len(_b1) >= 10:
            b1_end = float(_b1.iloc[-1])
            if b1_end < -0.5:
                ns_insight += f" <b>β1 (Slope) at {b1_end:.2f} indicates a steep curve. Short rates far below long rates; consider steepener trades.</b>"
        _section_note(
            "Nelson-Siegel beta factors (weekly). β0 = long-run floor, β1 = slope, β2 = curvature. Red verticals = BOJ events."
            + ns_insight
        )
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
        _add_boj_events(fig_ns)
        _chart(_style_fig(fig_ns, 380))

    # --- Real Yield Proxy ---
    _df_yc = load_unified(*args)
    _jp10_yc = _df_yc["JP_10Y"].dropna() if "JP_10Y" in _df_yc.columns else pd.Series(dtype=float)
    _be_yc = _df_yc["JP_BREAKEVEN"].dropna() if "JP_BREAKEVEN" in _df_yc.columns else pd.Series(dtype=float)
    if len(_jp10_yc) > 20 and len(_be_yc) > 20:
        st.subheader("Real Yield Proxy")
        _definition_block(
            "What is Japan's Real Yield",
            "If a bond pays you 1% per year but inflation is 2%, you are actually <em>losing</em> 1% of "
            "purchasing power annually. That -1% is your <b>real yield</b>: what you actually earn after "
            "accounting for inflation. Real yield = nominal yield (what the bond pays) minus expected inflation "
            "(what the market thinks prices will rise). "
            "For over a decade, Japanese real yields were deeply negative because the BOJ suppressed nominal "
            "yields near zero while inflation, though low, was still positive. This meant bondholders were "
            "quietly losing money in real terms every year. "
            "Now, as nominal yields rise and inflation expectations shift, real yields are moving toward zero "
            "or positive territory. This is a critical threshold: when real yields turn positive, pension funds "
            "and insurance companies (who need real returns to meet obligations) may finally find JGBs attractive "
            "again, stabilizing prices. But the journey FROM deeply negative TO zero is the most dangerous period "
            "because it means rapid repricing. "
            "<b>How to read this chart:</b> Blue = nominal yield (what the bond pays). Orange = breakeven "
            "inflation (expected inflation). Red = real yield (the difference). When the red line crosses zero "
            "from below, it is a major structural shift."
        )
        _real_aligned = pd.DataFrame({"JP_10Y": _jp10_yc, "JP_BREAKEVEN": _be_yc}).dropna()
        if len(_real_aligned) > 10:
            _real_aligned["REAL_YIELD"] = _real_aligned["JP_10Y"] - _real_aligned["JP_BREAKEVEN"]
            _section_note(
                "Real yield = JP 10Y nominal minus breakeven inflation proxy. "
                "Negative = bondholders losing purchasing power. Crossing zero is a regime signal."
            )
            fig_real = go.Figure()
            fig_real.add_trace(go.Scatter(
                x=_real_aligned.index, y=_real_aligned["JP_10Y"],
                mode="lines", name="Nominal 10Y", line=dict(color="#1565C0"),
            ))
            fig_real.add_trace(go.Scatter(
                x=_real_aligned.index, y=_real_aligned["JP_BREAKEVEN"],
                mode="lines", name="Breakeven", line=dict(color="#FF9800"),
            ))
            fig_real.add_trace(go.Scatter(
                x=_real_aligned.index, y=_real_aligned["REAL_YIELD"],
                mode="lines", name="Real Yield", line=dict(color="#E8413C", width=2),
            ))
            fig_real.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)
            fig_real.update_layout(yaxis_title="Yield (%)")
            _add_boj_events(fig_real)
            _chart(_style_fig(fig_real, 380))
            _real_latest = float(_real_aligned["REAL_YIELD"].iloc[-1])
            _takeaway_block(
                f"Real yield at <b>{_real_latest:.2f}%</b>. "
                f"{'Positive real yield: bondholders are compensated for inflation. This is a structural shift from the BOJ suppression era and supports repricing.' if _real_latest > 0 else 'Real yield remains negative: bondholders are still losing purchasing power. BOJ suppression effects persist, but the gap is narrowing.' if _real_latest > -0.5 else 'Deeply negative real yield: extreme financial repression. This is unsustainable long-term and creates latent repricing pressure.'}"
            )

    # --- Curve Slope and Butterfly ---
    _jp2 = _df_yc["JP_2Y"].dropna() if "JP_2Y" in _df_yc.columns else pd.Series(dtype=float)
    _jp5 = _df_yc["JP_5Y"].dropna() if "JP_5Y" in _df_yc.columns else pd.Series(dtype=float)
    _jp10_s = _df_yc["JP_10Y"].dropna() if "JP_10Y" in _df_yc.columns else pd.Series(dtype=float)
    _jp20 = _df_yc["JP_20Y"].dropna() if "JP_20Y" in _df_yc.columns else pd.Series(dtype=float)
    _jp30 = _df_yc["JP_30Y"].dropna() if "JP_30Y" in _df_yc.columns else pd.Series(dtype=float)
    _slope_data = {}
    if len(_jp2) > 20 and len(_jp10_s) > 20:
        _s_aligned = pd.DataFrame({"JP_2Y": _jp2, "JP_10Y": _jp10_s}).dropna()
        if len(_s_aligned) > 10:
            _slope_data["2s10s"] = _s_aligned["JP_10Y"] - _s_aligned["JP_2Y"]
    if len(_jp10_s) > 20 and len(_jp30) > 20:
        _s_aligned2 = pd.DataFrame({"JP_10Y": _jp10_s, "JP_30Y": _jp30}).dropna()
        if len(_s_aligned2) > 10:
            _slope_data["10s30s"] = _s_aligned2["JP_30Y"] - _s_aligned2["JP_10Y"]
    if len(_jp2) > 20 and len(_jp5) > 20 and len(_jp10_s) > 20:
        _bf_aligned = pd.DataFrame({"JP_2Y": _jp2, "JP_5Y": _jp5, "JP_10Y": _jp10_s}).dropna()
        if len(_bf_aligned) > 10:
            _slope_data["Butterfly (2s5s10s)"] = 2 * _bf_aligned["JP_5Y"] - _bf_aligned["JP_2Y"] - _bf_aligned["JP_10Y"]
    if _slope_data:
        st.subheader("Curve Slopes and Butterfly")
        _definition_block(
            "What are Curve Slopes and Butterfly Spreads",
            "The yield curve is not flat. Normally, longer bonds pay higher yields because you are taking more "
            "risk by lending for longer. The <b>slope</b> measures this difference: "
            "<b>2s10s slope</b> = 10-year yield minus 2-year yield. If this is 0.50%, it means you earn 0.50% "
            "more per year for lending to Japan for 10 years instead of 2. A steep curve (large positive slope) "
            "means the market expects rates to rise. A flat curve (near zero) means the market thinks rates will "
            "stay low. An inverted curve (negative slope) historically signals a coming recession. "
            "<b>10s30s slope</b> = 30-year yield minus 10-year yield. This matters especially in Japan because "
            "life insurers and pension funds are big buyers of 20-30 year JGBs to match their long-term liabilities. "
            "The <b>butterfly spread</b> (2 x 5Y minus 2Y minus 10Y) captures whether the middle of the curve "
            "is expensive or cheap relative to the ends. Positive = the 5-year point is expensive (yields are low), "
            "often because domestic banks concentrate their buying there. Negative = the belly is cheap, "
            "potentially a buying opportunity. "
            "<b>How to read this chart:</b> Lines above zero = upward-sloping curve (normal). "
            "Lines approaching zero = flattening (market expects slower growth). "
            "Sharp steepening at BOJ event dates = repricing of long-end yields."
        )
        _section_note(
            "JGB curve slopes (percentage points) and butterfly spread. "
            "Steepening 2s10s = repricing pressure on the long end. Butterfly captures belly distortion."
        )
        fig_slope = go.Figure()
        slope_colors = {"2s10s": "#1565C0", "10s30s": "#4CAF50", "Butterfly (2s5s10s)": "#E8413C"}
        for label, series in _slope_data.items():
            fig_slope.add_trace(go.Scatter(
                x=series.index, y=series,
                mode="lines", name=label,
                line=dict(color=slope_colors.get(label, "#333")),
            ))
        fig_slope.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)
        fig_slope.update_layout(yaxis_title="Spread (pp)")
        _add_boj_events(fig_slope)
        _chart(_style_fig(fig_slope, 380))
        if "2s10s" in _slope_data:
            _s2s10s = float(_slope_data["2s10s"].iloc[-1])
            _takeaway_block(
                f"JGB 2s10s slope at <b>{_s2s10s:.2f}pp</b>. "
                f"{'Steep curve: the market is pricing in higher long-term rates relative to short rates. Steepener trades are crowded but directionally correct.' if _s2s10s > 0.5 else 'Flat curve: minimal compensation for duration risk. This compresses term premium and makes long-end JGBs vulnerable to repricing.' if _s2s10s < 0.1 else 'Moderate slope: term premium is present but not extreme.'}"
            )

    # --- Yield Change Correlation Heatmap ---
    _yield_cols_corr = [c for c in ["JP_2Y", "JP_5Y", "JP_10Y", "JP_20Y", "JP_30Y", "US_10Y", "DE_10Y", "UK_10Y"] if c in _df_yc.columns]
    if len(_yield_cols_corr) >= 4:
        st.subheader("Yield Change Correlation Matrix")
        _definition_block(
            "What Yield Change Correlations Reveal",
            "This heatmap shows how daily yield <em>changes</em> in different bonds relate to each other. "
            "Each cell shows a correlation from -1 to +1. Dark red (+1) means when one yield goes up, "
            "the other goes up too, every time. Dark blue (-1) means they move in opposite directions. "
            "White (0) means they move independently. "
            "<b>Within Japan:</b> If JP_2Y, JP_5Y, JP_10Y, JP_20Y, JP_30Y all show high correlations "
            "with each other (lots of red), it confirms that the entire JGB curve moves together (PC1 "
            "dominance). This is typical during a broad repricing where all maturities sell off simultaneously. "
            "If some tenors decorrelate (e.g. 2Y stays put while 30Y moves), it suggests targeted repricing "
            "at specific maturities, creating curve trade opportunities. "
            "<b>Across countries:</b> High correlation between JP_10Y and US_10Y means Japan is just following "
            "the US. Low correlation means domestic Japanese factors (BOJ policy, yen, domestic inflation) are "
            "more important. "
            "<b>How to read this heatmap:</b> Look at the JP_10Y row. Which bonds move most with it? "
            "Dark red cells = those bonds move together. The diagonal is always +1 (each bond is perfectly "
            "correlated with itself)."
        )
        _yc_changes = _df_yc[_yield_cols_corr].diff().dropna()
        if len(_yc_changes) > 30:
            _corr_matrix = _yc_changes.corr()
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=_corr_matrix.values,
                x=_corr_matrix.columns.tolist(),
                y=_corr_matrix.index.tolist(),
                colorscale="RdBu",
                zmid=0,
                zmin=-1, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in _corr_matrix.values],
                texttemplate="%{text}",
            ))
            fig_heatmap.update_layout(yaxis_autorange="reversed")
            _chart(_style_fig(fig_heatmap, 420))
            # Find most/least correlated pairs
            _corr_vals = []
            for i in range(len(_corr_matrix)):
                for j in range(i + 1, len(_corr_matrix)):
                    _corr_vals.append((_corr_matrix.index[i], _corr_matrix.columns[j], _corr_matrix.iloc[i, j]))
            _corr_vals.sort(key=lambda x: x[2])
            if _corr_vals:
                _lowest = _corr_vals[0]
                _highest = _corr_vals[-1]
                _takeaway_block(
                    f"Strongest correlation: <b>{_highest[0]}-{_highest[1]}</b> at <b>{_highest[2]:.2f}</b>. "
                    f"Weakest: <b>{_lowest[0]}-{_lowest[1]}</b> at <b>{_lowest[2]:.2f}</b>. "
                    f"{'Low cross-country correlation suggests Japan-specific repricing dynamics.' if _lowest[2] < 0.3 else 'High cross-country correlation confirms global rate transmission is the dominant force.'}"
                )

    # --- ACM Term Premium Validation ---
    try:
        from src.yield_curve.term_premium import estimate_acm_term_premium, validate_term_premium
        _tp_df_full = load_unified(*args)
        _tp_yield_cols = [c for c in _tp_df_full.columns if c.startswith("JP_") and c != "JP_CPI" and c != "JP_CPI_CORE" and c != "JP_CALL_RATE"]
        if len(_tp_yield_cols) >= 3 and len(_tp_df_full.dropna(subset=_tp_yield_cols)) > 30:
            _tp_tenors = []
            for c in _tp_yield_cols:
                # Extract tenor number from column name like "JP_2Y", "JP_10Y"
                _tn = c.replace("JP_", "").replace("Y", "")
                try:
                    _tp_tenors.append(float(_tn))
                except ValueError:
                    _tp_tenors.append(10.0)
            _tp_target = 10.0 if 10.0 in _tp_tenors else _tp_tenors[-1]
            _tp_result = estimate_acm_term_premium(
                _tp_df_full[_tp_yield_cols].dropna(),
                tenors=_tp_tenors,
                n_factors=min(3, len(_tp_yield_cols)),
                target_tenor=_tp_target,
            )
            _tp_val = validate_term_premium(_tp_result, asset_class="JGB")

            st.subheader("ACM Term Premium Validation")
            _section_note(
                "The estimated term premium is validated against <b>published reference ranges</b>. "
                f"For JGB, the expected range is <b>{_tp_val['ref_lo']} to {_tp_val['ref_hi']} bps</b> "
                f"({_tp_val['source']}). A sanity check passes if \u226550% of estimates fall within this range."
            )

            _tp_pass_icon = "\u2705" if _tp_val["sanity_pass"] else "\u26a0\ufe0f"
            _tp_c1, _tp_c2, _tp_c3, _tp_c4 = st.columns(4)
            _tp_c1.metric("Sanity Check", f"{_tp_pass_icon} {'PASS' if _tp_val['sanity_pass'] else 'FAIL'}")
            _tp_c2.metric("Mean TP", f"{_tp_val['mean_bps']:.0f} bps")
            _tp_c3.metric("Std Dev", f"{_tp_val['std_bps']:.0f} bps")
            _tp_c4.metric("In Reference Range", f"{_tp_val['pct_in_range']:.0%}")

            if _tp_val["warnings"]:
                for _w in _tp_val["warnings"]:
                    st.warning(_w)

            _takeaway_block(
                f"Term premium estimates {'pass' if _tp_val['sanity_pass'] else 'fail'} the sanity check: "
                f"<b>{_tp_val['pct_in_range']:.0%}</b> of observations fall within the {_tp_val['ref_lo']} to "
                f"{_tp_val['ref_hi']} bps reference range. "
                f"Mean: <b>{_tp_val['mean_bps']:.0f} bps</b>, range: {_tp_val['min_bps']:.0f} to {_tp_val['max_bps']:.0f} bps. "
                f"{'Estimates are consistent with published ACM values for JGB term premia under BOJ compression.' if _tp_val['sanity_pass'] else 'Some estimates fall outside expected bounds, which may reflect model simplifications or extreme market conditions.'}"
            )
    except Exception:
        pass  # validation is supplementary; do not break the page

    # --- Page conclusion ---
    _yc_parts = []
    if pca_result is not None:
        _ev = pca_result["explained_variance_ratio"]
        _yc_parts.append(f"PC1 explains {_ev[0]:.0%} of yield variance")
    if liq is not None and len(liq["composite_index"].dropna()) > 0:
        _liq_v = float(liq["composite_index"].dropna().iloc[-1])
        _liq_state = "healthy" if _liq_v > 0 else "stressed" if _liq_v < -1 else "neutral"
        _yc_parts.append(f"liquidity is {_liq_state} ({_liq_v:+.2f} z-score)")
    if ns_result is not None and "beta0" in ns_result.columns and len(ns_result["beta0"].dropna()) > 0:
        _b0_v = float(ns_result["beta0"].dropna().iloc[-1])
        _yc_parts.append(f"the Nelson-Siegel level factor stands at {_b0_v:.2f}")
    _yc_summary = "; ".join(_yc_parts) + "." if _yc_parts else "Insufficient data for a complete summary."
    # Verdict
    if pca_result is not None and liq is not None and len(liq["composite_index"].dropna()) > 0:
        _pc1_pct = pca_result["explained_variance_ratio"][0]
        _liq_v2 = float(liq["composite_index"].dropna().iloc[-1])
        if _pc1_pct > 0.8 and _liq_v2 < -0.5:
            _verdict_p2 = f"The entire curve is repricing in unison ({_pc1_pct:.0%} PC1) and liquidity is thin. Broad duration risk is elevated."
        elif _pc1_pct > 0.8:
            _verdict_p2 = f"Parallel shift dominates at {_pc1_pct:.0%}. All maturities are moving together; this is a level story, not a curve story."
        elif _liq_v2 < -1:
            _verdict_p2 = f"Liquidity is deteriorating ({_liq_v2:+.1f} z-score). Expect wider bid-ask spreads and choppy execution on any JGB repositioning."
        else:
            _verdict_p2 = "Yield curve structure is orderly. No unusual concentration in a single factor; standard curve trades apply."
    else:
        _verdict_p2 = "Yield curve analytics require additional data. Expand the date range or switch data sources."
    _page_conclusion(
        _verdict_p2,
        f"{_yc_summary.capitalize()} "
        f"These structural decompositions feed directly into the regime detection models on the next page.",
    )
    _page_footer()


