"""Spillover & Information Flow page: Granger, TE, DY, DCC, Carry."""

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
from src.pages.yield_curve import _run_pca


def _get_args():
    """Retrieve sidebar args from session state."""
    return st.session_state["_app_args"]


def _get_layout_config():
    return st.session_state["_layout_config"]


def _get_alert_notifier():
    return st.session_state.get("_alert_notifier")


@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
def _run_granger(simulated, start, end, api_key):
    from src.spillover.granger import pairwise_granger

    df = load_unified(simulated, start, end, api_key)
    cols = [c for c in ["JP_10Y", "US_10Y", "DE_10Y", "UK_10Y", "AU_10Y", "USDJPY", "NIKKEI", "VIX"] if c in df.columns]
    if len(cols) < 2:
        return None
    sub = df[cols].diff().dropna()
    if len(sub) < 30:
        return None
    return pairwise_granger(sub, max_lag=5, significance=0.05)



@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
def _run_te(simulated, start, end, api_key):
    from src.spillover.transfer_entropy import pairwise_transfer_entropy

    df = load_unified(simulated, start, end, api_key)
    # Keep TE to 6 core variables (56 pairs at 8 vars is slow; 30 pairs at 6 is 2x faster)
    cols = [c for c in ["JP_10Y", "US_10Y", "DE_10Y", "UK_10Y", "USDJPY", "VIX"] if c in df.columns]
    if len(cols) < 2:
        return None
    sub = df[cols].diff().dropna()
    if len(sub) < 30:
        return None
    return pairwise_transfer_entropy(sub, lag=1, n_bins=3)



@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
def _run_spillover(simulated, start, end, api_key):
    from src.spillover.diebold_yilmaz import compute_spillover_index

    df = load_unified(simulated, start, end, api_key)
    cols = [c for c in ["JP_10Y", "US_10Y", "DE_10Y", "UK_10Y", "AU_10Y", "USDJPY", "NIKKEI"] if c in df.columns]
    if len(cols) < 2:
        return None
    sub = df[cols].diff().dropna()
    if len(sub) < 50:
        return None
    return compute_spillover_index(sub, var_lags=4, forecast_horizon=10)



@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
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



@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
def _run_te_pca(simulated, start, end, api_key):
    """Transfer Entropy on PCA factor scores (PC1/PC2/PC3)."""
    from src.spillover.transfer_entropy import pairwise_transfer_entropy

    pca_res = _run_pca(simulated, start, end, api_key)
    if pca_res is None:
        return None
    scores = pca_res["scores"]
    if scores.shape[1] < 2 or len(scores) < 30:
        return None
    return pairwise_transfer_entropy(scores, lag=1, n_bins=3)



@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
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
    _page_intro(
        "How shocks propagate across sovereign bond markets, currencies, and equities. This page measures "
        "the direction, magnitude, and timing of information flow between JGBs and global assets using "
        "four complementary methods. High spillover means JGBs are no longer an isolated market; a shock "
        "in U.S. Treasuries or USDJPY will propagate into JGB yields with quantifiable lag and magnitude."
    )

    args = _get_args()

    # Pre-compute all spillover models in a single pass
    with st.spinner("Computing cross-market spillover analysis..."):
        granger_df = _run_granger(*args)
        te_df = _run_te(*args)
        te_pca_df = _run_te_pca(*args)
        spill = _run_spillover(*args)
        dcc = _run_dcc(*args)
        carry = _run_carry(*args)

    # --- Granger Causality ---
    st.subheader("Granger Causality (significant pairs)")
    _definition_block(
        "What is Granger Causality?",
        "If knowing what happened to US Treasury yields yesterday helps you predict what will happen to "
        "JGB yields today, then US yields 'Granger-cause' JGB yields. This is not about true cause and effect "
        "(it is a statistical test), but it reveals <em>which markets lead and which follow</em>. "
        "The test works by comparing two forecasting models: one that only uses JGB history, and another that "
        "also includes past US yield data. If the second model is significantly better, the F-statistic will "
        "be large and the p-value will be small (below 0.05). "
        "The <b>optimal lag</b> tells you how many days of lead time you get. If US_10Y Granger-causes JP_10Y "
        "at lag 3, it means US yield moves from 3 days ago still predict JGB moves today. This creates a "
        "tradeable window: you can observe the US move and position in JGBs before the Japanese market reacts. "
        "<b>How to read this table:</b> Each row is a significant (p < 0.05) predictive link. "
        "'Cause' column = the leading variable. 'Effect' column = the following variable. "
        "Higher F-stat = stronger prediction. Lower p-value = more statistically reliable. "
        "The full table (in the expander) includes all pairs tested, including non-significant ones."
    )
    if granger_df is not None and not granger_df.empty:
        sig_df = granger_df[granger_df["significant"] == True].reset_index(drop=True)
        n_sig = len(sig_df)
        gc_insight = ""
        if not sig_df.empty:
            top_row = sig_df.loc[sig_df["f_stat"].idxmax()]
            gc_insight = (
                f" <b>Actionable: The strongest link is {top_row['cause']} → {top_row['effect']} "
                f"(F={top_row['f_stat']:.1f}, p={top_row['p_value']:.4f}). Lagged moves in {top_row['cause']} "
                f"statistically predict {top_row['effect']}. Monitor {top_row['cause']} for early signals.</b>"
            )
        _section_note(
            f"Significant Granger-causal pairs at 5% level. {n_sig} of {len(granger_df)} pairs significant. Yellow = strongest F-stat."
            + gc_insight
        )
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
    _definition_block(
        "What is Transfer Entropy?",
        "Granger causality (above) only detects <em>linear</em> relationships. But markets often transmit "
        "information in non-linear ways: a large US move might cause a JGB reaction while a small US move does "
        "not. <b>Transfer entropy</b> captures ALL types of information flow, including non-linear ones. "
        "Technically, it asks: 'How much does knowing X's past reduce my uncertainty about Y's future, "
        "beyond what Y's own past tells me?' "
        "The result is a <b>heatmap</b> where each cell (row=source, column=target) shows how much information "
        "flows from the source to the target. Larger values (brighter colors) = stronger information flow. "
        "Critically, transfer entropy is <b>directional</b>: the flow from US_10Y to JP_10Y can be completely "
        "different from JP_10Y to US_10Y. When one direction is much larger, it reveals a clear leader-follower "
        "relationship. The leader moves first, and the follower reacts. "
        "<b>How to read this heatmap:</b> Read across a ROW to see how much information that variable sends "
        "to others. Read down a COLUMN to see how much information that variable receives. "
        "The variable with the brightest row is the biggest 'transmitter' (information source). "
        "The variable with the brightest column is the biggest 'receiver' (most influenced by others)."
    )
    if te_df is not None and not te_df.empty:
        sources = te_df["source"].unique()
        targets = te_df["target"].unique()
        all_labels = sorted(set(sources) | set(targets))
        te_matrix = pd.DataFrame(0.0, index=all_labels, columns=all_labels)
        for _, row in te_df.iterrows():
            te_matrix.loc[row["source"], row["target"]] = row["te_value"]

        # Analyse off-diagonal flows only (exclude self-to-self)
        n = len(all_labels)
        te_vals = te_matrix.values.copy()
        np.fill_diagonal(te_vals, np.nan)  # mask diagonal

        # Strongest single directional link
        flat_idx = int(np.nanargmax(te_vals))
        te_src = all_labels[flat_idx // n]
        te_tgt = all_labels[flat_idx % n]
        te_val = te_vals[flat_idx // n, flat_idx % n]

        # Most asymmetric pair: largest |A→B minus B→A|
        asym_best, asym_leader, asym_follower, asym_fwd, asym_rev = 0.0, "", "", 0.0, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                fwd = te_matrix.iloc[i, j]
                rev = te_matrix.iloc[j, i]
                diff = abs(fwd - rev)
                if diff > asym_best:
                    asym_best = diff
                    if fwd > rev:
                        asym_leader, asym_follower = all_labels[i], all_labels[j]
                        asym_fwd, asym_rev = fwd, rev
                    else:
                        asym_leader, asym_follower = all_labels[j], all_labels[i]
                        asym_fwd, asym_rev = rev, fwd

        # Net transmitter / receiver (sum of outflows minus inflows, off-diagonal)
        out_flow = np.nansum(te_vals, axis=1)  # row sums = total info sent
        in_flow = np.nansum(te_vals, axis=0)   # col sums = total info received
        net_flow = out_flow - in_flow
        net_transmitter = all_labels[int(np.argmax(net_flow))]
        net_receiver = all_labels[int(np.argmin(net_flow))]

        # Who drives JP_10Y specifically?
        jp_insight = ""
        if "JP_10Y" in all_labels:
            jp_col_idx = all_labels.index("JP_10Y")
            jp_inflows = te_vals[:, jp_col_idx].copy()
            jp_inflows[jp_col_idx] = np.nan  # exclude self
            if not np.all(np.isnan(jp_inflows)):
                top_driver_idx = int(np.nanargmax(jp_inflows))
                top_driver = all_labels[top_driver_idx]
                top_driver_te = jp_inflows[top_driver_idx]
                jp_insight = (
                    f" For JGB-specific positioning, <b>{top_driver}</b> is the single strongest information source "
                    f"into JP_10Y (TE = {top_driver_te:.4f}). Monitor {top_driver} for early signals before JGB moves."
                )

        te_description = (
            f"Transfer entropy heatmap (rows = source, columns = target). Off-diagonal only; Viridis scale. "
        )
        te_description += (
            f"<b>Strongest link:</b> {te_src} → {te_tgt} (TE = {te_val:.4f}). "
            f"<b>Most asymmetric pair:</b> {asym_leader} → {asym_follower} "
            f"(fwd {asym_fwd:.4f}, rev {asym_rev:.4f}). "
            f"<b>Net transmitter:</b> {net_transmitter}. <b>Net receiver:</b> {net_receiver}. "
        )
        te_description += (
            f"<b>Actionable:{jp_insight} "
            f"Lag between {asym_leader} and {asym_follower} represents a tradeable window.</b>"
        )
        _section_note(te_description)
        fig_te = px.imshow(
            te_matrix.values,
            x=te_matrix.columns.tolist(),
            y=te_matrix.index.tolist(),
            color_continuous_scale="Viridis",
            aspect="auto",
            labels=dict(color="TE"),
        )
        _chart(_style_fig(fig_te, 450))
    else:
        st.warning("Insufficient data for transfer entropy.")

    # --- Transfer Entropy on PCA Factor Scores ---
    st.subheader("Transfer Entropy on PCA Factors")
    _definition_block(
        "Why Apply TE to PCA Factors?",
        "Instead of measuring information flow between raw asset returns, this section applies transfer "
        "entropy to the PCA factor scores (Level, Slope, Curvature) from Page 2. This answers a deeper "
        "question: <em>does a broad parallel shift (PC1) lead to subsequent slope changes (PC2)?</em> "
        "If so, a level move today predicts a curve reshape tomorrow, a tradeable lag for steepener or "
        "butterfly positions."
    )
    if te_pca_df is not None and not te_pca_df.empty:
        pca_sources = te_pca_df["source"].unique()
        pca_targets = te_pca_df["target"].unique()
        pca_labels = sorted(set(pca_sources) | set(pca_targets))
        te_pca_matrix = pd.DataFrame(0.0, index=pca_labels, columns=pca_labels)
        for _, row in te_pca_df.iterrows():
            te_pca_matrix.loc[row["source"], row["target"]] = row["te_value"]

        # Find dominant information flow among factors
        te_pca_vals = te_pca_matrix.values.copy()
        np.fill_diagonal(te_pca_vals, np.nan)
        flat_idx = int(np.nanargmax(te_pca_vals))
        n_pca = len(pca_labels)
        pca_src = pca_labels[flat_idx // n_pca]
        pca_tgt = pca_labels[flat_idx % n_pca]
        pca_te_val = te_pca_vals[flat_idx // n_pca, flat_idx % n_pca]

        pca_te_insight = (
            f"Transfer entropy computed on PCA factor scores (PC1=Level, PC2=Slope, PC3=Curvature). "
            f"<b>Strongest factor link:</b> {pca_src} → {pca_tgt} (TE = {pca_te_val:.4f}). "
        )
        if "PC1" in pca_src:
            pca_te_insight += f"<b>Actionable: Level factor drives information to {pca_tgt}. Broad yield moves propagate to curve shape changes with a tradeable lag.</b>"
        elif "PC2" in pca_src:
            pca_te_insight += f"<b>Actionable: Slope factor leads {pca_tgt}. Steepening/flattening signals precede the next factor's move — position the slope first.</b>"
        else:
            pca_te_insight += f"<b>Actionable: Curvature factor ({pca_src}) leads {pca_tgt}. Belly moves are driving the curve; butterfly trades have predictive power.</b>"
        _section_note(pca_te_insight)

        fig_te_pca = px.imshow(
            te_pca_matrix.values,
            x=te_pca_matrix.columns.tolist(),
            y=te_pca_matrix.index.tolist(),
            color_continuous_scale="Viridis",
            aspect="auto",
            labels=dict(color="TE"),
        )
        _chart(_style_fig(fig_te_pca, 350))
    else:
        st.info("Insufficient PCA data for factor-level transfer entropy.")

    # --- Diebold-Yilmaz ---
    st.subheader("Diebold-Yilmaz Spillover")
    _definition_block(
        "Diebold-Yilmaz Spillover Index (2012)",
        "Imagine a shock hits the US Treasury market. How much of that shock eventually shows up in JGB yields, "
        "the yen, and the Nikkei? The <b>Diebold-Yilmaz spillover index</b> answers this precisely. "
        "It works by building a model (VAR) of how all markets interact, then asking: 'If I shock one variable, "
        "what fraction of the resulting movement in another variable is due to that external shock vs its own history?' "
        "The <b>total spillover index</b> (shown as a single %) is the average cross-market influence. Above 30% means "
        "markets are tightly coupled: a shock anywhere quickly spreads everywhere. Below 20% means markets are "
        "relatively independent, and diversification across them works well. "
        "The <b>bar chart</b> shows <em>net directional spillover</em>: green bars are net TRANSMITTERS (they send "
        "more shocks than they receive) and red bars are net RECEIVERS (they absorb shocks from others). "
        "If JP_10Y is a net receiver, it means JGBs are being pushed around by other markets. If it is a net "
        "transmitter, JGB moves are spilling into other assets. "
        "<b>How to read:</b> Big green bar = that market drives others. Big red bar = that market is most "
        "vulnerable to outside shocks. Total % below the chart = overall market interconnectedness."
    )
    if spill is not None:
        total_spill = spill["total_spillover"]
        net = spill["net_spillover"]
        top_transmitter = net.idxmax() if len(net) > 0 else "N/A"
        top_receiver = net.idxmin() if len(net) > 0 else "N/A"
        dy_insight = ""
        if total_spill > 30:
            dy_insight = f" <b>Actionable: Total spillover at {total_spill:.1f}% is elevated. Markets are tightly coupled. A shock in {top_transmitter} (biggest :green[green bar]) will propagate quickly. Diversification across these assets is less effective than usual.</b>"
        else:
            dy_insight = f" <b>Actionable: Total spillover at {total_spill:.1f}% is moderate. Markets are relatively independent. Diversification benefits are intact.</b>"
        _section_note(
            f"Total spillover {total_spill:.1f}%. Green bars = net transmitters, red = net receivers. Top transmitter: {top_transmitter}."
            + dy_insight
        )
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            st.metric("Total Spillover Index", f"{total_spill:.1f}%")
        with col_s2:
            net = spill["net_spillover"]
            fig_net = go.Figure(
                go.Bar(x=net.index.tolist(), y=net.values, marker_color=["green" if v > 0 else "red" for v in net.values])
            )
            fig_net.update_layout(title="Net Directional Spillover", yaxis_title="Net (%)")
            _chart(_style_fig(fig_net, 320))

        with st.expander("Spillover matrix"):
            st.dataframe(spill["spillover_matrix"].round(2))
    else:
        st.warning("Insufficient data for spillover analysis.")

    # --- DCC ---
    st.subheader("DCC Time-Varying Correlations")
    _definition_block(
        "DCC-GARCH Correlations",
        "Markets are not always equally connected. During calm periods, JGBs might move independently of US "
        "Treasuries. But during a crisis, they suddenly start moving in lockstep. "
        "<b>Dynamic Conditional Correlation (DCC)</b> tracks this changing relationship over time. "
        "Unlike a simple rolling average (which treats all past data equally), DCC puts more weight on "
        "recent observations, so it reacts faster when correlations suddenly spike during crises. "
        "Each line on the chart shows the correlation between a pair of assets over time. "
        "A correlation near +1 means they move together. Near 0 means independent. Near -1 means they move "
        "in opposite directions. "
        "<b>Why this matters:</b> If you hold JGBs and want to hedge with US Treasuries, you need the "
        "correlation between them to be stable and positive. If correlation suddenly drops to zero, your "
        "hedge stops working. If correlation spikes to near +1 during stress, it means both are selling off "
        "simultaneously (no diversification benefit). "
        "The critical threshold is 0.6: above this, two assets are so correlated that holding both provides "
        "minimal diversification. Correlation spikes at BOJ event dates reveal policy-driven contagion."
    )
    if dcc is not None:
        cond_corr = dcc["conditional_correlations"]
        if cond_corr:
            n_pairs = len(cond_corr)
            # Find the pair with highest latest correlation
            dcc_latest = {p: float(s.dropna().iloc[-1]) for p, s in cond_corr.items() if len(s.dropna()) > 0}
            dcc_insight = ""
            if dcc_latest:
                max_pair = max(dcc_latest, key=dcc_latest.get)
                max_corr = dcc_latest[max_pair]
                min_pair = min(dcc_latest, key=dcc_latest.get)
                min_corr = dcc_latest[min_pair]
                dcc_insight = (
                    f" Currently, <b>{max_pair}</b> has the highest correlation ({max_corr:.2f}) and <b>{min_pair}</b> "
                    f"the lowest ({min_corr:.2f}). "
                )
                if max_corr > 0.6:
                    dcc_insight += f"<b>Actionable: {max_pair} correlation above 0.6. These assets are moving in lockstep. Hedging one with the other is less effective than usual; look for decorrelated pairs instead.</b>"
                elif max_corr < 0.3:
                    dcc_insight += f"<b>Actionable: All correlations are below 0.3. Markets are relatively decoupled. Cross-asset diversification is working; no contagion signal.</b>"
            _section_note(
                f"{n_pairs} DCC-GARCH conditional correlation pair(s). Unlike rolling windows, DCC captures crisis-driven correlation spikes."
                + dcc_insight
            )
            fig_dcc = go.Figure()
            for pair, series in cond_corr.items():
                fig_dcc.add_trace(
                    go.Scatter(x=series.index, y=series.values, mode="lines", name=pair)
                )
            fig_dcc.update_layout(yaxis_title="Conditional Correlation")
            _add_boj_events(fig_dcc)
            _chart(_style_fig(fig_dcc, 380))
        else:
            st.info("No correlation pairs computed.")
    else:
        st.warning("Insufficient data for DCC-GARCH.")

    # --- FX Carry ---
    st.subheader("FX Carry Metrics (USD/JPY)")
    _definition_block(
        "What is the Carry Trade?",
        "The <b>carry trade</b> is one of the most popular strategies in foreign exchange markets. "
        "It works like this: borrow money in a country with low interest rates (Japan, where rates were "
        "near zero for decades) and invest it in a country with higher rates (the US, where rates are "
        "around 4-5%). You earn the difference in interest rates as profit. "
        "The catch: you are exposed to <b>currency risk</b>. If the yen strengthens against the dollar, "
        "your FX losses can wipe out the interest income. The <b>carry</b> (blue line) shows the rate "
        "differential in percent. The <b>realized volatility</b> (green line) shows how much the USD/JPY "
        "exchange rate actually bounces around. "
        "The <b>carry-to-vol ratio</b> (orange, right axis) is the key metric: it divides the interest "
        "earned by the FX risk taken. Above 1.0 = the interest you earn exceeds the risk of FX losses "
        "(trade is attractive). Below 0.5 = the yen is so volatile that carry profits are being destroyed "
        "(positions are vulnerable to violent unwinds). "
        "<b>Why this matters for JGBs:</b> Trillions of dollars are in yen carry trades globally. If the BOJ "
        "raises rates, the carry shrinks and investors start closing these positions, which means selling dollars "
        "and buying yen. This yen strengthening creates a feedback loop: it makes JGB prices go up temporarily "
        "(haven flows) but signals a fundamental shift in the rate environment."
    )
    if carry is not None:
        latest_ctv = carry["carry_to_vol"].dropna().iloc[-1] if len(carry["carry_to_vol"].dropna()) > 0 else float("nan")
        latest_carry = carry["carry"].dropna().iloc[-1] if len(carry["carry"].dropna()) > 0 else float("nan")
        latest_rvol = carry["realized_vol"].dropna().iloc[-1] if len(carry["realized_vol"].dropna()) > 0 else float("nan")
        ctv_label = f"{latest_ctv:.2f}" if not np.isnan(latest_ctv) else "N/A"
        carry_insight = ""
        if not np.isnan(latest_ctv):
            if latest_ctv > 1.0:
                carry_insight = f" <b>Actionable: Carry-to-Vol at {latest_ctv:.2f} (>1.0). The rate differential more than compensates for FX risk. Carry trades are attractive; long USDJPY is favoured.</b>"
            elif latest_ctv > 0.5:
                carry_insight = f" <b>Actionable: Carry-to-Vol at {latest_ctv:.2f} (0.5-1.0). Marginal; carry exists but FX vol is eating into returns. Only enter with tight stop-losses.</b>"
            else:
                carry_insight = f" <b>Actionable: Carry-to-Vol at {latest_ctv:.2f} (<0.5). Danger zone. FX volatility dwarfs the interest differential. Crowded carry positions are vulnerable to violent unwind; consider closing USDJPY longs.</b>"
        _section_note(
            f"Carry (US-JP rate gap, {latest_carry:.2f}%), realized vol ({latest_rvol:.2f}%), and carry-to-vol ratio ({ctv_label}, right axis)."
            + carry_insight
        )
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
            yaxis_title="Rate / Vol",
            yaxis2=dict(title="Carry-to-Vol Ratio", overlaying="y", side="right"),
        )
        _add_boj_events(fig_c)
        _chart(_style_fig(fig_c, 380))
    else:
        st.warning("Insufficient data for carry analytics.")

    # --- Rolling Spillover (time-varying connectedness) ---
    st.subheader("Rolling Spillover Index")
    _definition_block(
        "Why Time-Varying Spillover Matters",
        "The spillover index above gives you a single number for the entire sample. But market connectedness "
        "is not constant: markets can be independent for months, then suddenly become tightly coupled during "
        "a crisis. This chart shows the spillover index computed on a <b>rolling 120-day window</b>, so you "
        "can see exactly when and how market interconnectedness changes over time. "
        "Rising spillover heading into a BOJ meeting date is a warning sign: it means any surprise decision "
        "will ripple across all markets more than usual. Falling spillover after a crisis means markets are "
        "decoupling and returning to normal. "
        "The shaded area under the line helps you visually compare spillover levels across different periods. "
        "Compare the current reading to the sample average: above average means heightened contagion risk, "
        "below average means markets are relatively insulated from each other."
    )
    try:
        _df_spill = load_unified(*_get_args())
        from src.spillover.diebold_yilmaz import compute_spillover_index
        _spill_cols = [c for c in ["JP_10Y", "US_10Y", "USDJPY", "VIX", "NIKKEI"] if c in _df_spill.columns]
        if len(_spill_cols) >= 3:
            _spill_df = _df_spill[_spill_cols].diff().dropna()
            _window = 120
            if len(_spill_df) > _window + 30:
                _rolling_spill = []
                # Sample every 5 days for speed
                _sample_indices = range(_window, len(_spill_df), 5)
                for i in _sample_indices:
                    _window_data = _spill_df.iloc[i - _window:i]
                    try:
                        _res = compute_spillover_index(_window_data, var_lags=2, horizon=5)
                        _rolling_spill.append({"date": _spill_df.index[i], "spillover": _res["total_spillover"]})
                    except Exception:
                        continue
                if _rolling_spill:
                    _rs_df = pd.DataFrame(_rolling_spill).set_index("date")
                    _section_note(
                        f"120-day rolling DY spillover index (sampled every 5 days). "
                        f"Rising index = markets becoming more interconnected. Spikes often precede or coincide with policy events."
                    )
                    fig_rs = go.Figure()
                    fig_rs.add_trace(go.Scatter(
                        x=_rs_df.index, y=_rs_df["spillover"],
                        mode="lines", name="Rolling Spillover (%)",
                        line=dict(color="#E8413C"),
                        fill="tozeroy", fillcolor="rgba(232,65,60,0.1)",
                    ))
                    fig_rs.update_layout(yaxis_title="Total Spillover (%)")
                    _add_boj_events(fig_rs)
                    _chart(_style_fig(fig_rs, 380))
                    _rs_latest = float(_rs_df["spillover"].iloc[-1])
                    _rs_mean = float(_rs_df["spillover"].mean())
                    _takeaway_block(
                        f"Current rolling spillover: <b>{_rs_latest:.1f}%</b> vs sample average <b>{_rs_mean:.1f}%</b>. "
                        f"{'Spillover is elevated above average: markets are more interconnected than usual. Cross-asset hedges may underperform.' if _rs_latest > _rs_mean * 1.1 else 'Spillover is below average: markets are relatively independent. Diversification benefits are intact.'}"
                    )
    except Exception:
        st.info("Could not compute rolling spillover index.")

    # --- Page conclusion ---
    _sp_parts = []
    _sp_total = None
    _sp_ctv = None
    if spill is not None:
        _sp_total = spill["total_spillover"]
        _sp_parts.append(f"total spillover index at {_sp_total:.1f}%")
    if granger_df is not None and not granger_df.empty:
        _n_gc = int(granger_df["significant"].sum())
        _sp_parts.append(f"{_n_gc} significant Granger-causal link{'s' if _n_gc != 1 else ''}")
    if carry is not None and len(carry["carry_to_vol"].dropna()) > 0:
        _sp_ctv = float(carry["carry_to_vol"].dropna().iloc[-1])
        _carry_state = "attractive" if _sp_ctv > 1.0 else "marginal" if _sp_ctv > 0.5 else "unattractive"
        _sp_parts.append(f"FX carry-to-vol ratio is {_carry_state} at {_sp_ctv:.2f}")
    _sp_summary = "; ".join(_sp_parts) + "." if _sp_parts else "Insufficient data for a complete spillover summary."
    # Verdict
    if _sp_total is not None and _sp_ctv is not None:
        if _sp_total > 40 and _sp_ctv < 0.5:
            _verdict_p4 = f"Contagion risk is high ({_sp_total:.0f}% spillover) and carry no longer compensates for vol. Reduce cross-asset exposure."
        elif _sp_total > 30:
            _verdict_p4 = f"Markets are tightly coupled at {_sp_total:.0f}% spillover. Diversification is less effective than usual; hedge explicitly."
        elif _sp_ctv > 1.0:
            _verdict_p4 = f"Spillover is contained ({_sp_total:.0f}%) and carry-to-vol at {_sp_ctv:.1f}x is compelling. Carry trades are well supported."
        else:
            _verdict_p4 = f"Cross-market linkages are moderate ({_sp_total:.0f}% spillover). No systemic contagion signal; standard risk budgets apply."
    elif _sp_total is not None:
        _verdict_p4 = f"Spillover index at {_sp_total:.0f}%. {'Markets are interconnected; size accordingly.' if _sp_total > 30 else 'No systemic contagion detected.'}"
    else:
        _verdict_p4 = "Spillover analysis requires a broader dataset. Expand the date range for meaningful cross-market inference."
    _page_conclusion(
        _verdict_p4,
        f"{_sp_summary.capitalize()} "
        f"These cross-market dynamics are synthesised into actionable trade ideas on the final page.",
    )
    _page_footer()


