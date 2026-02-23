"""Equity Market Spillover page: JGB/BOJ policy transmission to equity sectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.ui.shared import (
    _style_fig, _chart, _page_intro, _section_note, _definition_block,
    _takeaway_block, _page_conclusion, _page_footer, _add_boj_events,
    _PALETTE,
)
from src.pages._data import load_unified, _safe_col
from src.data.config import EQUITY_SECTOR_TICKERS


# ===================================================================
# Helpers
# ===================================================================

def _get_args():
    """Retrieve sidebar args from session state."""
    return st.session_state["_app_args"]


# ===================================================================
# Data loading
# ===================================================================

@st.cache_data(show_spinner=False, ttl=3600, max_entries=8)
def _load_equity_sectors(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    """Download equity sector data from yfinance. Returns daily close prices."""
    import yfinance as yf

    ticker_dict = dict(tickers)  # (name, symbol) pairs
    frames = {}
    for name, symbol in ticker_dict.items():
        try:
            df = yf.download(symbol, start=start, end=end, progress=False)
            if df is not None and len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    frames[name] = df["Close"].iloc[:, 0]
                else:
                    frames[name] = df["Close"]
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).dropna(how="all")


def _build_ticker_tuple(market: str) -> tuple:
    """Build a hashable tuple of (name, symbol) pairs for a market."""
    config = EQUITY_SECTOR_TICKERS.get(market, {})
    pairs = []
    for key, val in config.items():
        if key == "_broad":
            for bname, bsym in val.items():
                pairs.append((bname, bsym))
        else:
            pairs.append((key, val))
    return tuple(pairs)


def _get_broad_names(market: str) -> list[str]:
    """Get broad index names for a market."""
    config = EQUITY_SECTOR_TICKERS.get(market, {})
    broad = config.get("_broad", {})
    return list(broad.keys())


def _get_sector_names(market: str) -> list[str]:
    """Get sector names (excluding _broad) for a market."""
    config = EQUITY_SECTOR_TICKERS.get(market, {})
    return [k for k in config if k != "_broad"]


# ===================================================================
# Analytics (cached)
# ===================================================================

@st.cache_data(show_spinner=False, ttl=3600, max_entries=8)
def _run_equity_correlation(jgb_changes: tuple, equity_returns: tuple,
                            index_vals: tuple, start: str, end: str,
                            window: int = 60) -> pd.DataFrame | None:
    """Rolling correlation between JGB yield changes and equity sector returns.

    Computes per-sector pairwise correlation independently so that sparse
    sectors (different holiday calendars) don't eliminate each other's rows.
    """
    jgb = pd.Series(dict(jgb_changes))
    jgb.index = pd.to_datetime(jgb.index)

    result = {}
    for name, vals in equity_returns:
        s = pd.Series(dict(vals))
        s.index = pd.to_datetime(s.index)
        pair = pd.DataFrame({"eq": s, "jgb": jgb}).dropna()
        if len(pair) < window + 10:
            continue
        result[name] = pair["eq"].rolling(window).corr(pair["jgb"])

    if not result:
        return None
    return pd.DataFrame(result).dropna(how="all")


@st.cache_data(show_spinner=False, ttl=3600, max_entries=8)
def _run_equity_granger(jgb_col: tuple, equity_cols: tuple,
                        col_names: tuple, start: str, end: str) -> pd.DataFrame | None:
    """Granger causality from JGB changes to equity sector returns."""
    from src.spillover.granger import pairwise_granger

    jgb = pd.Series(dict(jgb_col), name="JGB_10Y")
    jgb.index = pd.to_datetime(jgb.index)

    frames = {"JGB_10Y": jgb}
    for name, vals in zip(col_names, equity_cols):
        s = pd.Series(dict(vals))
        s.index = pd.to_datetime(s.index)
        frames[name] = s

    combined = pd.DataFrame(frames).dropna()
    if len(combined) < 30:
        return None
    return pairwise_granger(combined, max_lag=5, significance=0.05)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=8)
def _run_equity_dcc(jgb_col: tuple, equity_cols: tuple,
                    col_names: tuple, start: str, end: str) -> dict | None:
    """DCC-GARCH between JGB changes and equity sector returns."""
    from src.spillover.dcc_garch import compute_dcc

    jgb = pd.Series(dict(jgb_col), name="JGB_10Y")
    jgb.index = pd.to_datetime(jgb.index)

    frames = {"JGB_10Y": jgb}
    for name, vals in zip(col_names, equity_cols):
        s = pd.Series(dict(vals))
        s.index = pd.to_datetime(s.index)
        frames[name] = s

    combined = pd.DataFrame(frames).dropna() * 100  # scale for GARCH
    if len(combined) < 60:
        return None
    return compute_dcc(combined, p=1, q=1)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=8)
def _run_equity_spillover(jgb_col: tuple, equity_cols: tuple,
                          col_names: tuple, start: str, end: str) -> dict | None:
    """Diebold-Yilmaz spillover index for JGB + equity sectors."""
    from src.spillover.diebold_yilmaz import compute_spillover_index

    jgb = pd.Series(dict(jgb_col), name="JGB_10Y")
    jgb.index = pd.to_datetime(jgb.index)

    frames = {"JGB_10Y": jgb}
    for name, vals in zip(col_names, equity_cols):
        s = pd.Series(dict(vals))
        s.index = pd.to_datetime(s.index)
        frames[name] = s

    combined = pd.DataFrame(frames).dropna()
    if len(combined) < 50:
        return None
    return compute_spillover_index(combined, var_lags=4, forecast_horizon=10)


# ===================================================================
# Serialization helpers (make DataFrames/Series cacheable as tuples)
# ===================================================================

def _series_to_tuple(s: pd.Series) -> tuple:
    """Convert a Series to a tuple of (index_str, value) pairs for caching."""
    return tuple(zip(s.index.astype(str), s.values))


def _df_cols_to_tuples(df: pd.DataFrame, cols: list[str]) -> tuple:
    """Convert specific DataFrame columns to nested tuples for caching."""
    return tuple(_series_to_tuple(df[c].dropna()) for c in cols if c in df.columns)


# ===================================================================
# Page layout
# ===================================================================

def page_equity_spillover():
    """Main page function for Equity Market Spillover."""

    st.title("Equity Market Spillover")

    _page_intro(
        "This page examines how JGB yield movements and BOJ policy shifts transmit into "
        "global equity markets. The analysis covers Japan, USA, India, and China at the "
        "sector level, using rolling correlations, Granger causality, DCC-GARCH dynamic "
        "correlations, and Diebold-Yilmaz spillover indices to quantify transmission "
        "channels from Japanese rates to equity sectors."
    )

    _definition_block(
        "JGB-Equity Transmission",
        "BOJ policy changes affect equity markets through multiple channels: (1) discount rate effects "
        "on equity valuations, (2) currency transmission via yen carry trade unwinds, (3) risk appetite "
        "contagion from JGB volatility, and (4) portfolio rebalancing flows as institutions adjust "
        "allocations. Sector sensitivity varies: financials benefit from steeper curves, while "
        "rate-sensitive sectors (real estate, utilities) face headwinds from rising yields."
    )

    # --- Load JGB data ---
    args = _get_args()
    unified = load_unified(*args)
    jgb_col = _safe_col(unified, "JP_10Y")

    if jgb_col is None or len(jgb_col) < 60:
        st.warning("Insufficient JGB 10Y data. Check data source and date range.")
        _page_footer()
        return

    jgb_changes = jgb_col.diff().dropna()

    # --- Market selector ---
    st.subheader("Market Selection")
    _section_note(
        "Select an equity market to analyze its sector-level sensitivity to JGB yield changes. "
        "Each market uses local sector indices or ETF proxies where direct index data is unavailable."
    )

    markets = list(EQUITY_SECTOR_TICKERS.keys())
    selected_market = st.selectbox("Equity Market", markets, index=0)

    # Sector multiselect
    all_sectors = _get_sector_names(selected_market)
    selected_sectors = st.multiselect(
        "Sectors",
        all_sectors,
        default=all_sectors,
        help="Select sectors to include in the analysis.",
    )

    if not selected_sectors:
        st.info("Select at least one sector to begin analysis.")
        _page_footer()
        return

    # --- Load equity data ---
    ticker_tuple = _build_ticker_tuple(selected_market)
    start_str, end_str = str(args[1]), str(args[2])

    with st.spinner(f"Loading {selected_market} equity sector data..."):
        equity_df = _load_equity_sectors(ticker_tuple, start_str, end_str)

    if equity_df.empty:
        st.warning(f"No equity data available for {selected_market}. yfinance may not support some tickers.")
        _page_footer()
        return

    # Compute returns (per-column; do NOT dropna across the whole DataFrame
    # because sparse columns like "Financial Services" would eliminate all rows)
    equity_returns = equity_df.pct_change()
    broad_names = _get_broad_names(selected_market)
    # Only include sectors that actually have meaningful data (>60 non-null returns)
    available_sectors = [
        s for s in selected_sectors
        if s in equity_returns.columns and equity_returns[s].notna().sum() > 60
    ]
    available_broad = [b for b in broad_names if b in equity_df.columns]

    if not available_sectors and not available_broad:
        st.warning("No sector data could be loaded for the selected market.")
        _page_footer()
        return

    # --- Broad index + JGB overlay ---
    st.subheader("Broad Index vs JGB 10Y Yield")
    _section_note(
        "Overlay of broad equity indices with JGB 10Y yield to visualize co-movement. "
        "Divergences may indicate regime shifts or changing transmission dynamics."
    )

    if available_broad:
        fig = go.Figure()
        for i, bname in enumerate(available_broad):
            fig.add_trace(go.Scatter(
                x=equity_df.index, y=equity_df[bname],
                name=bname, yaxis="y",
                line=dict(color=_PALETTE[i % len(_PALETTE)], width=1.5),
            ))
        fig.add_trace(go.Scatter(
            x=jgb_col.index, y=jgb_col,
            name="JGB 10Y (%)", yaxis="y2",
            line=dict(color="#c0392b", width=1.5, dash="dot"),
        ))
        fig.update_layout(
            yaxis=dict(title="Index Level"),
            yaxis2=dict(title="JGB 10Y (%)", overlaying="y", side="right",
                        showgrid=False, tickformat=".2f"),
        )
        fig = _style_fig(fig, height=400)
        fig = _add_boj_events(fig)
        _chart(fig)
    else:
        st.info("No broad index data available for this market.")

    # --- Analysis tabs ---
    st.subheader("Spillover Analytics")

    # Prepare hashable data for cached functions
    jgb_tuple = _series_to_tuple(jgb_changes)
    eq_return_tuples = tuple(
        (name, _series_to_tuple(equity_returns[name].dropna()))
        for name in available_sectors
        if name in equity_returns.columns
    )
    eq_col_names = tuple(name for name, _ in eq_return_tuples)
    eq_col_data = tuple(vals for _, vals in eq_return_tuples)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Correlation", "Granger Causality", "DCC Correlation", "Spillover Matrix"
    ])

    # ── Tab 1: Rolling Correlation ────────────────────────────────────
    with tab1:
        _section_note(
            "60-day rolling Pearson correlation between daily JGB 10Y yield changes and "
            "equity sector daily returns. Positive correlation means sectors move with yields; "
            "negative means they move inversely."
        )

        corr_df = _run_equity_correlation(
            jgb_tuple, eq_return_tuples, tuple(), start_str, end_str, window=60
        )

        if corr_df is not None and not corr_df.empty:
            # Time series plot
            fig_corr = go.Figure()
            for i, col in enumerate(corr_df.columns):
                fig_corr.add_trace(go.Scatter(
                    x=corr_df.index, y=corr_df[col],
                    name=col,
                    line=dict(color=_PALETTE[i % len(_PALETTE)], width=1.2),
                ))
            fig_corr.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.8)
            fig_corr.update_layout(
                yaxis_title="Rolling Correlation with JGB 10Y",
                yaxis_range=[-1, 1],
            )
            fig_corr = _style_fig(fig_corr, height=420)
            _chart(fig_corr)

            # Latest correlation heatmap
            latest = corr_df.iloc[-1:].T
            latest.columns = ["Correlation"]
            latest = latest.sort_values("Correlation", ascending=False)

            fig_heat = px.bar(
                latest.reset_index(),
                x="Correlation", y="index",
                orientation="h",
                color="Correlation",
                color_continuous_scale=["#c0392b", "#f0eeeb", "#2e7d32"],
                color_continuous_midpoint=0,
            )
            fig_heat.update_layout(
                yaxis_title="", xaxis_title="Current Rolling Correlation",
                showlegend=False, coloraxis_showscale=False,
            )
            fig_heat = _style_fig(fig_heat, height=max(250, len(latest) * 28))
            _chart(fig_heat)

            # Takeaway
            most_pos = latest.idxmax().iloc[0]
            most_neg = latest.idxmin().iloc[0]
            _takeaway_block(
                f"Currently, <b>{most_pos}</b> has the strongest positive correlation with JGB yields "
                f"({latest.loc[most_pos, 'Correlation']:.2f}), while <b>{most_neg}</b> has the most "
                f"negative ({latest.loc[most_neg, 'Correlation']:.2f}). Sectors with strong positive "
                f"correlation are most vulnerable to JGB repricing events."
            )
        else:
            st.info("Insufficient overlapping data for rolling correlation analysis.")

    # ── Tab 2: Granger Causality ──────────────────────────────────────
    with tab2:
        _section_note(
            "Granger causality tests whether past JGB yield changes help predict equity sector "
            "returns beyond their own history. Significant results (p < 0.05) suggest a "
            "lead-lag information flow from JGB rates to equities."
        )

        granger_df = _run_equity_granger(
            jgb_tuple, eq_col_data, eq_col_names, start_str, end_str
        )

        if granger_df is not None and not granger_df.empty:
            # Filter to show JGB -> Equity causality
            jgb_to_eq = granger_df[granger_df["cause"] == "JGB_10Y"].copy()
            if not jgb_to_eq.empty:
                jgb_to_eq = jgb_to_eq.sort_values("p_value")
                st.dataframe(
                    jgb_to_eq[["cause", "effect", "optimal_lag", "f_stat", "p_value", "significant"]]
                    .rename(columns={
                        "cause": "Cause", "effect": "Effect",
                        "optimal_lag": "Lag", "f_stat": "F-stat",
                        "p_value": "p-value", "significant": "Significant"
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                # Bar chart of F-statistics
                fig_gc = px.bar(
                    jgb_to_eq, x="f_stat", y="effect",
                    orientation="h",
                    color="significant",
                    color_discrete_map={True: "#2e7d32", False: "#9D9795"},
                )
                fig_gc.update_layout(
                    yaxis_title="", xaxis_title="F-statistic",
                    showlegend=True,
                    legend_title_text="Significant (p<0.05)",
                )
                fig_gc = _style_fig(fig_gc, height=max(250, len(jgb_to_eq) * 30))
                _chart(fig_gc)

                n_sig = jgb_to_eq["significant"].sum()
                _takeaway_block(
                    f"JGB 10Y yield changes Granger-cause {n_sig} out of "
                    f"{len(jgb_to_eq)} {selected_market} sectors at the 5% level. "
                    f"{'This indicates strong information flow from JGB rates to equities.' if n_sig > len(jgb_to_eq) / 2 else 'Limited causal links suggest equities respond to local factors more than JGB rates.'}"
                )
            else:
                st.info("No Granger causality results from JGB to equity sectors.")

            # Also show reverse (equity -> JGB)
            with st.expander("Reverse causality: Equity sectors -> JGB 10Y"):
                eq_to_jgb = granger_df[granger_df["effect"] == "JGB_10Y"].copy()
                if not eq_to_jgb.empty:
                    st.dataframe(
                        eq_to_jgb[["cause", "effect", "optimal_lag", "f_stat", "p_value", "significant"]]
                        .rename(columns={
                            "cause": "Cause", "effect": "Effect",
                            "optimal_lag": "Lag", "f_stat": "F-stat",
                            "p_value": "p-value", "significant": "Significant"
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No reverse causality results available.")
        else:
            st.info("Insufficient data for Granger causality analysis.")

    # ── Tab 3: DCC-GARCH ─────────────────────────────────────────────
    with tab3:
        _section_note(
            "Dynamic Conditional Correlation (DCC-GARCH) captures time-varying correlations "
            "that account for volatility clustering. Unlike rolling windows, DCC adapts "
            "to changing market regimes and provides smoother correlation estimates."
        )

        # Limit to 4 sectors max for DCC (computational cost)
        dcc_sectors = eq_col_names[:4] if len(eq_col_names) > 4 else eq_col_names
        dcc_data = eq_col_data[:4] if len(eq_col_data) > 4 else eq_col_data

        if len(dcc_sectors) > 4:
            st.info("DCC analysis limited to first 4 sectors for computational efficiency.")

        dcc_result = _run_equity_dcc(
            jgb_tuple, dcc_data, dcc_sectors, start_str, end_str
        )

        if dcc_result is not None:
            correlations = dcc_result.get("correlations", {})
            # Find JGB pairs
            jgb_pairs = {k: v for k, v in correlations.items() if "JGB_10Y" in k}

            if jgb_pairs:
                fig_dcc = go.Figure()
                for i, (pair_name, corr_series) in enumerate(jgb_pairs.items()):
                    other = pair_name.replace("JGB_10Y", "").strip(" -_/")
                    label = other if other else pair_name
                    fig_dcc.add_trace(go.Scatter(
                        x=corr_series.index, y=corr_series.values,
                        name=label,
                        line=dict(color=_PALETTE[i % len(_PALETTE)], width=1.3),
                    ))
                fig_dcc.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.8)
                fig_dcc.update_layout(
                    yaxis_title="DCC Conditional Correlation",
                    yaxis_range=[-1, 1],
                )
                fig_dcc = _style_fig(fig_dcc, height=400)
                fig_dcc = _add_boj_events(fig_dcc)
                _chart(fig_dcc)

                _takeaway_block(
                    "DCC correlations spike during stress events (BOJ surprises, global risk-off). "
                    "Persistently elevated correlations indicate structural linkage; transient spikes "
                    "suggest event-driven contagion that fades as markets digest the policy shift."
                )
            else:
                st.info("No DCC pairs involving JGB found.")
        else:
            st.info(
                "DCC-GARCH computation requires sufficient overlapping data (60+ observations). "
                "Try expanding the date range or selecting different sectors."
            )

    # ── Tab 4: Diebold-Yilmaz Spillover ──────────────────────────────
    with tab4:
        _section_note(
            "The Diebold-Yilmaz spillover index decomposes forecast error variance to "
            "measure directional connectedness. A high total spillover indicates tightly "
            "coupled markets; directional bars show net transmitters vs receivers."
        )

        # Limit to 5 sectors for DY
        dy_sectors = eq_col_names[:5] if len(eq_col_names) > 5 else eq_col_names
        dy_data = eq_col_data[:5] if len(eq_col_data) > 5 else eq_col_data

        spillover_result = _run_equity_spillover(
            jgb_tuple, dy_data, dy_sectors, start_str, end_str
        )

        if spillover_result is not None:
            total_idx = spillover_result.get("total_spillover", 0)
            st.metric("Total Spillover Index", f"{total_idx:.1f}%")

            # Directional spillover
            to_others = spillover_result.get("directional_to")
            from_others = spillover_result.get("directional_from")

            if to_others is not None and from_others is not None:
                net = to_others - from_others
                net_df = pd.DataFrame({"Net Spillover": net}).sort_values("Net Spillover", ascending=True)

                fig_dir = px.bar(
                    net_df.reset_index(),
                    x="Net Spillover", y="index",
                    orientation="h",
                    color="Net Spillover",
                    color_continuous_scale=["#2e7d32", "#f0eeeb", "#c0392b"],
                    color_continuous_midpoint=0,
                )
                fig_dir.update_layout(
                    yaxis_title="",
                    xaxis_title="Net Directional Spillover (%)",
                    showlegend=False,
                    coloraxis_showscale=False,
                )
                fig_dir = _style_fig(fig_dir, height=max(250, len(net_df) * 30))
                _chart(fig_dir)

                # Variance decomposition table
                decomp = spillover_result.get("variance_decomposition")
                if decomp is not None:
                    with st.expander("Variance Decomposition Table"):
                        st.dataframe(
                            decomp.style.format("{:.1f}"),
                            use_container_width=True,
                        )

                if "JGB_10Y" in net.index:
                    jgb_net = net["JGB_10Y"]
                    role = "net transmitter" if jgb_net > 0 else "net receiver"
                    _takeaway_block(
                        f"JGB 10Y is a <b>{role}</b> of shocks (net spillover: {jgb_net:+.1f}%). "
                        f"Total spillover index of {total_idx:.1f}% indicates "
                        f"{'tightly coupled' if total_idx > 30 else 'moderately connected' if total_idx > 15 else 'loosely connected'} "
                        f"markets. During BOJ policy surprises, these linkages typically intensify."
                    )
            else:
                st.info("Directional spillover decomposition not available.")
        else:
            st.info(
                "Spillover index computation requires sufficient data (50+ observations). "
                "Try expanding the date range."
            )

    # ── Conclusion ────────────────────────────────────────────────────
    # Auto-generate verdict based on available results
    verdict_parts = []
    if corr_df is not None and not corr_df.empty:
        avg_abs_corr = corr_df.iloc[-1].abs().mean()
        if avg_abs_corr > 0.3:
            verdict_parts.append(f"strong correlation (avg |r|={avg_abs_corr:.2f})")
        elif avg_abs_corr > 0.15:
            verdict_parts.append(f"moderate correlation (avg |r|={avg_abs_corr:.2f})")
        else:
            verdict_parts.append(f"weak correlation (avg |r|={avg_abs_corr:.2f})")

    if spillover_result is not None:
        total_idx = spillover_result.get("total_spillover", 0)
        verdict_parts.append(f"spillover index {total_idx:.1f}%")

    if verdict_parts:
        verdict = f"JGB-to-{selected_market} equity transmission: {', '.join(verdict_parts)}"
    else:
        verdict = f"JGB-to-{selected_market} equity transmission analysis"

    _page_conclusion(
        verdict,
        f"The analysis examines how JGB 10Y yield changes transmit to {selected_market} equity sectors "
        f"through four complementary lenses: rolling correlation, Granger causality, DCC-GARCH, and "
        f"Diebold-Yilmaz spillover decomposition. Sectors with persistent positive correlation to JGB "
        f"yields are most exposed to repricing risk. Monitor DCC correlations for signs of regime-dependent "
        f"transmission intensification around BOJ policy events."
    )

    _page_footer()
