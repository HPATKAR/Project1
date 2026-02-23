"""Trade Ideas page."""

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
from src.pages.regime import _run_ensemble, _run_markov, _run_entropy, _run_garch
from src.pages.yield_curve import _run_pca, _run_ns, _run_liquidity
from src.pages.spillover import _run_granger, _run_te, _run_spillover, _run_dcc, _run_te_pca, _run_carry
from src.pages.early_warning import _run_warning_score, _run_ml_predictor


def _get_args():
    """Retrieve sidebar args from session state."""
    return st.session_state["_app_args"]


def _get_layout_config():
    return st.session_state["_layout_config"]


def _get_alert_notifier():
    return st.session_state.get("_alert_notifier")

from datetime import datetime
from src.reporting.pdf_export import JGBReportPDF, dataframe_to_csv_bytes



@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
def _generate_trades(simulated, start, end, api_key):
    from src.strategy.trade_generator import generate_all_trades
    from src.yield_curve.term_premium import estimate_acm_term_premium

    df = load_unified(simulated, start, end, api_key)

    # Gather regime state inputs
    # Regime probability
    try:
        ensemble = _run_ensemble(simulated, start, end, api_key)
    except Exception:
        ensemble = None
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

    # Spot levels for concrete trade targets
    jp10_level = float(df["JP_10Y"].dropna().iloc[-1]) if "JP_10Y" in df.columns and len(df["JP_10Y"].dropna()) > 0 else None
    us10_level = float(df["US_10Y"].dropna().iloc[-1]) if "US_10Y" in df.columns and len(df["US_10Y"].dropna()) > 0 else None
    usdjpy_level = float(df["USDJPY"].dropna().iloc[-1]) if "USDJPY" in df.columns and len(df["USDJPY"].dropna()) > 0 else None
    nikkei_level = float(df["NIKKEI"].dropna().iloc[-1]) if "NIKKEI" in df.columns and len(df["NIKKEI"].dropna()) > 0 else None
    jp2y_level = float(df["JP_2Y"].dropna().iloc[-1]) if "JP_2Y" in df.columns and len(df["JP_2Y"].dropna()) > 0 else None

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
        "jp10_level": jp10_level,
        "us10_level": us10_level,
        "usdjpy_level": usdjpy_level,
        "nikkei_level": nikkei_level,
        "jp2y_level": jp2y_level,
    }

    cards = generate_all_trades(regime_state)
    return cards, regime_state



def _build_payout_chart(card) -> "go.Figure | None":
    """Build a Plotly payout profile chart for a trade card. Returns None if unsupported."""
    meta = card.metadata or {}
    _GOLD = "#CFB991"
    _RED = "#c0392b"
    _GREEN = "#2e7d32"
    _BLK = "#000000"

    fig = None

    # --- Straddle ---
    if "straddle_strike" in meta:
        K = meta["straddle_strike"]
        premium = abs(K) * 0.015
        x = np.linspace(K - K * 0.05, K + K * 0.05, 200)
        call_pay = np.maximum(x - K, 0) - premium / 2
        put_pay = np.maximum(K - x, 0) - premium / 2
        total = call_pay + put_pay
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=total, mode="lines", line=dict(color=_BLK, width=2), name="Straddle P&L"))
        fig.add_trace(go.Scatter(x=x, y=np.where(total > 0, total, 0), fill="tozeroy", line=dict(width=0), fillcolor="rgba(207,185,145,0.3)", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=np.where(total < 0, total, 0), fill="tozeroy", line=dict(width=0), fillcolor="rgba(192,57,43,0.15)", showlegend=False))
        fig.add_vline(x=K, line_dash="dot", line_color=_GOLD, annotation_text=f"Strike {K:.1f}")
        fig.add_hline(y=0, line_dash="dash", line_color="#888")
        fig.update_layout(xaxis_title="Underlying Price", yaxis_title="P&L", title=f"{card.name} — Straddle Payout")

    # --- Payer spread ---
    elif "atm_strike" in meta and "otm_strike" in meta:
        K1 = meta["atm_strike"]
        K2 = meta["otm_strike"]
        premium = abs(K2 - K1) * 0.4
        x = np.linspace(K1 - abs(K2 - K1) * 2, K2 + abs(K2 - K1) * 2, 200)
        total = np.maximum(x - K1, 0) - np.maximum(x - K2, 0) - premium
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=total, mode="lines", line=dict(color=_BLK, width=2), name="Payer Spread P&L"))
        fig.add_trace(go.Scatter(x=x, y=np.where(total > 0, total, 0), fill="tozeroy", line=dict(width=0), fillcolor="rgba(207,185,145,0.3)", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=np.where(total < 0, total, 0), fill="tozeroy", line=dict(width=0), fillcolor="rgba(192,57,43,0.15)", showlegend=False))
        fig.add_vline(x=K1, line_dash="dot", line_color=_GOLD, annotation_text=f"Buy {K1:.3f}%")
        fig.add_vline(x=K2, line_dash="dot", line_color=_RED, annotation_text=f"Sell {K2:.3f}%")
        fig.add_hline(y=0, line_dash="dash", line_color="#888")
        fig.update_layout(xaxis_title="Swap Rate (%)", yaxis_title="P&L (bps)", title=f"{card.name} — Payer Spread Payout")

    # --- Short strangle ---
    elif "call_strike" in meta and "put_strike" in meta:
        Kc = meta["call_strike"]
        Kp = meta["put_strike"]
        premium = abs(Kc - Kp) * 0.3
        x = np.linspace(Kp - abs(Kc - Kp), Kc + abs(Kc - Kp), 200)
        total = -np.maximum(x - Kc, 0) - np.maximum(Kp - x, 0) + premium
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=total, mode="lines", line=dict(color=_BLK, width=2), name="Short Strangle P&L"))
        fig.add_trace(go.Scatter(x=x, y=np.where(total > 0, total, 0), fill="tozeroy", line=dict(width=0), fillcolor="rgba(207,185,145,0.3)", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=np.where(total < 0, total, 0), fill="tozeroy", line=dict(width=0), fillcolor="rgba(192,57,43,0.15)", showlegend=False))
        fig.add_vline(x=Kp, line_dash="dot", line_color=_GREEN, annotation_text=f"Put {Kp:.2f}")
        fig.add_vline(x=Kc, line_dash="dot", line_color=_RED, annotation_text=f"Call {Kc:.2f}")
        fig.add_hline(y=0, line_dash="dash", line_color="#888")
        fig.update_layout(xaxis_title="Underlying Price", yaxis_title="P&L", title=f"{card.name} — Short Strangle Payout")

    # --- Single put ---
    elif "put_strike" in meta and "usdjpy_spot" in meta:
        K = meta["put_strike"]
        spot = meta["usdjpy_spot"]
        premium = abs(spot - K) * 0.15
        x = np.linspace(K * 0.94, spot * 1.04, 200)
        total = np.maximum(K - x, 0) - premium
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=total, mode="lines", line=dict(color=_BLK, width=2), name="Long Put P&L"))
        fig.add_trace(go.Scatter(x=x, y=np.where(total > 0, total, 0), fill="tozeroy", line=dict(width=0), fillcolor="rgba(207,185,145,0.3)", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=np.where(total < 0, total, 0), fill="tozeroy", line=dict(width=0), fillcolor="rgba(192,57,43,0.15)", showlegend=False))
        fig.add_vline(x=K, line_dash="dot", line_color=_GOLD, annotation_text=f"Strike {K:.0f}")
        fig.add_vline(x=spot, line_dash="dot", line_color=_BLK, annotation_text=f"Spot {spot:.0f}")
        fig.add_hline(y=0, line_dash="dash", line_color="#888")
        fig.update_layout(xaxis_title="USDJPY", yaxis_title="P&L per unit", title=f"{card.name} — Put Payout (K={K:.0f})")

    # --- Directional yield with target/stop ---
    elif "target_yield" in meta and "stop_yield" in meta:
        entry = meta.get("jp10_level", 1.0)
        target = meta["target_yield"]
        stop = meta["stop_yield"]
        x = np.linspace(min(stop, entry) - 0.1, max(target, entry) + 0.1, 200)
        pnl = (entry - x) * 100 if card.direction == "short" else (x - entry) * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=pnl, mode="lines", line=dict(color=_BLK, width=2), name="P&L (bps)"))
        fig.add_trace(go.Scatter(x=x, y=np.where(pnl > 0, pnl, 0), fill="tozeroy", line=dict(width=0), fillcolor="rgba(207,185,145,0.3)", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=np.where(pnl < 0, pnl, 0), fill="tozeroy", line=dict(width=0), fillcolor="rgba(192,57,43,0.15)", showlegend=False))
        fig.add_vline(x=entry, line_dash="solid", line_color=_BLK, annotation_text=f"Entry {entry:.3f}%")
        fig.add_vline(x=target, line_dash="dash", line_color=_GREEN, annotation_text=f"Target {target:.2f}%")
        fig.add_vline(x=stop, line_dash="dash", line_color=_RED, annotation_text=f"Stop {stop:.2f}%")
        fig.add_hline(y=0, line_dash="dash", line_color="#888")
        fig.update_layout(xaxis_title="Yield (%)", yaxis_title="P&L (bps)", title=f"{card.name} — {card.direction.upper()} P&L Profile")

    # --- USDJPY with target/stop ---
    elif "target" in meta and "stop" in meta and "usdjpy_spot" in meta:
        spot = meta["usdjpy_spot"]
        target = meta["target"]
        stop = meta["stop"]
        x = np.linspace(stop * 0.98, target * 1.02, 200)
        pnl = (x - spot) / spot * 100 if card.direction == "long" else (spot - x) / spot * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=pnl, mode="lines", line=dict(color=_BLK, width=2), name="P&L (%)"))
        fig.add_trace(go.Scatter(x=x, y=np.where(pnl > 0, pnl, 0), fill="tozeroy", line=dict(width=0), fillcolor="rgba(207,185,145,0.3)", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=np.where(pnl < 0, pnl, 0), fill="tozeroy", line=dict(width=0), fillcolor="rgba(192,57,43,0.15)", showlegend=False))
        fig.add_vline(x=spot, line_dash="solid", line_color=_BLK, annotation_text=f"Spot {spot:.2f}")
        fig.add_vline(x=target, line_dash="dash", line_color=_GREEN, annotation_text=f"Target {target:.2f}")
        fig.add_vline(x=stop, line_dash="dash", line_color=_RED, annotation_text=f"Stop {stop:.2f}")
        fig.add_hline(y=0, line_dash="dash", line_color="#888")
        fig.update_layout(xaxis_title="USDJPY", yaxis_title="P&L (%)", title=f"{card.name} — {card.direction.upper()} P&L Profile")

    # --- Spread trade ---
    elif "spread_bps" in meta and "target_spread_bps" in meta:
        entry = meta["spread_bps"]
        target = meta["target_spread_bps"]
        x = np.linspace(entry - 30, max(target, entry) + 20, 200)
        pnl = x - entry if card.direction == "long" else entry - x
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=pnl, mode="lines", line=dict(color=_BLK, width=2), name="Spread P&L (bps)"))
        fig.add_trace(go.Scatter(x=x, y=np.where(pnl > 0, pnl, 0), fill="tozeroy", line=dict(width=0), fillcolor="rgba(207,185,145,0.3)", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=np.where(pnl < 0, pnl, 0), fill="tozeroy", line=dict(width=0), fillcolor="rgba(192,57,43,0.15)", showlegend=False))
        fig.add_vline(x=entry, line_dash="solid", line_color=_BLK, annotation_text=f"Entry {entry:.0f} bps")
        fig.add_vline(x=target, line_dash="dash", line_color=_GREEN, annotation_text=f"Target {target:.0f} bps")
        fig.add_hline(y=0, line_dash="dash", line_color="#888")
        fig.update_layout(xaxis_title="Spread (bps)", yaxis_title="P&L (bps)", title=f"{card.name} — Spread P&L")

    return fig



def page_trade_ideas():
    st.header("Trade Ideas")
    _page_intro(
        "Regime-conditional trade generation synthesising all upstream analytics into actionable "
        "positions. Each trade card specifies instruments (with Bloomberg tickers), direction, "
        "entry/exit signals with concrete levels, DV01-neutral or vol-target sizing, conviction "
        "scores derived from model consensus, and a mandatory failure scenario explaining what "
        "would invalidate the thesis. Filter by category and conviction via sidebar controls."
    )
    _definition_block(
        "How Trades are Generated",
        "A trade idea is a specific, actionable suggestion: buy or sell a particular instrument "
        "because your analysis says the price is likely to move in a certain direction. This page "
        "takes all the quantitative models from earlier pages and converts their outputs into concrete "
        "trade recommendations."
        "<br><br>"
        "<b>Where do these ideas come from?</b> The system reads the current 'regime state' - a snapshot "
        "of every model's output: Is the market repricing JGBs or staying calm? Is volatility rising? "
        "Are foreign markets spilling over into Japan? Is the yen carry trade attractive? Each answer "
        "points toward a different set of trades."
        "<br><br>"
        "<b>Four trade categories:</b>"
        "<br>- <b>Rates:</b> Trades on JGB yields directly - buying/selling JGB futures, betting on the "
        "yield curve steepening or flattening, or exploiting liquidity premiums in less-traded tenors."
        "<br>- <b>FX:</b> Currency trades, especially yen carry (borrowing in low-rate yen to invest in "
        "higher-rate currencies) and momentum-based yen bets."
        "<br>- <b>Volatility:</b> Trades that profit from how much prices swing, not which direction. "
        "Straddles (buying both a call and put option) profit when volatility is higher than expected."
        "<br>- <b>Cross-Asset:</b> Relative value trades across different markets, like going long "
        "Nikkei vs short JGBs when equity-bond correlation breaks down."
        "<br><br>"
        "<b>Conviction score:</b> Each trade gets a score from 0% to 100% measuring how strongly the "
        "models agree. Higher conviction means more signals point the same way. It is calculated as: "
        "base score + (regime probability x weight) + (model signal x weight), capped at 100%. "
        "A trade at 80% conviction has multiple models agreeing; one at 30% has mixed signals."
        "<br><br>"
        "<b>Failure scenario:</b> Every trade must answer 'What would kill this thesis?' This is the "
        "single most important field. Before entering any trade, read the failure scenario first. If "
        "that scenario is already playing out, skip the trade regardless of conviction score."
        "<br><br>"
        "<b>How to read the trade cards:</b> Start with the conviction bar chart to see which ideas "
        "have the strongest signal alignment. Then expand individual cards. Read Failure Scenario before "
        "Entry Signal - understanding what can go wrong matters more than what you hope will go right."
    )

    args = _get_args()

    with st.spinner("Generating trade ideas from all models..."):
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
    st.sidebar.markdown(
        "<div style='border-top:1px solid rgba(255,255,255,0.06); margin:0.4rem 0 0.5rem 0; padding-top:0.5rem;'>"
        "<span style='font-size:0.55rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.14em;color:rgba(255,255,255,0.4);"
        "font-family:var(--font-sans);'>Trade Filters</span></div>",
        unsafe_allow_html=True,
    )
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
        n_high = sum(1 for c in filtered if c.conviction >= 0.7)
        n_med = sum(1 for c in filtered if 0.4 <= c.conviction < 0.7)
        n_low = sum(1 for c in filtered if c.conviction < 0.4)
        top_card = max(filtered, key=lambda c: c.conviction)
        conv_insight = (
            f" <b>Actionable: The highest-conviction idea is \"{top_card.name}\" at {top_card.conviction:.0%} "
            f"({top_card.direction} {', '.join(top_card.instruments[:2])}). "
            f"This is where the most signals align. Focus research and sizing here first.</b>"
        )
        _section_note(
            f"Conviction by trade, colour-coded by category. "
            f"{n_high} high (≥70%), {n_med} medium (40-69%), {n_low} low (<40%)."
            + conv_insight
        )
        conv_data = pd.DataFrame({
            "Trade": [c.name for c in filtered],
            "Conviction": [c.conviction for c in filtered],
            "Category": [c.category for c in filtered],
        })
        fig_conv = px.bar(
            conv_data, x="Trade", y="Conviction", color="Category",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_conv.update_layout(xaxis_tickangle=-45)
        _chart(_style_fig(fig_conv, 350))

    # --- Trade cards ---
    st.subheader(f"Trade Cards ({len(filtered)} shown)")
    _section_note(
        "Sorted by conviction descending. Expand for full trade specification. Read Failure Scenario before Entry Signal."
    )
    for card in sorted(filtered, key=lambda c: -c.conviction):
        direction_tag = "LONG" if card.direction == "long" else "SHORT"
        dir_color = "#16a34a" if card.direction == "long" else "#dc2626"
        if card.conviction >= 0.7:
            conv_tag, conv_color = "HIGH", "#16a34a"
        elif card.conviction >= 0.4:
            conv_tag, conv_color = "MED", "#d97706"
        else:
            conv_tag, conv_color = "LOW", "#dc2626"

        with st.expander(
            f"{direction_tag}  {card.name}  ·  {card.conviction:.0%}  ·  {card.category}"
        ):
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:14px;flex-wrap:wrap;'>"
                f"<span style='background:{dir_color};color:#fff;padding:3px 10px;"
                f"border-radius:20px;font-weight:600;font-size:var(--fs-base);letter-spacing:var(--ls-wide);"
                f"font-family:var(--font-sans);'>{direction_tag}</span>"
                f"<span style='background:{conv_color};color:#fff;padding:3px 10px;"
                f"border-radius:20px;font-weight:600;font-size:var(--fs-base);"
                f"font-family:var(--font-mono);'>{card.conviction:.0%}</span>"
                f"<span style='background:#f7f8fb;color:#3b4259;padding:3px 12px;"
                f"border-radius:20px;font-size:var(--fs-base);font-weight:500;"
                f"border:1px solid #dfe2ec;font-family:var(--font-sans);'>{card.category}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown(
                    f"<div style='font-size:var(--fs-lg);line-height:1.85;color:#3b4259;"
                    f"font-family:var(--font-sans);'>"
                    f"<b style='color:#0b0f19;'>Instruments:</b> {', '.join(card.instruments)}<br>"
                    f"<b style='color:#0b0f19;'>Regime Condition:</b> {card.regime_condition}<br>"
                    f"<b style='color:#0b0f19;'>Edge Source:</b> {card.edge_source}<br>"
                    f"<b style='color:#0b0f19;'>Entry Signal:</b> {card.entry_signal}</div>",
                    unsafe_allow_html=True,
                )
            with col_r:
                st.markdown(
                    f"<div style='font-size:var(--fs-lg);line-height:1.85;color:#3b4259;"
                    f"font-family:var(--font-sans);'>"
                    f"<b style='color:#0b0f19;'>Exit Signal:</b> {card.exit_signal}<br>"
                    f"<b style='color:#0b0f19;'>Sizing:</b> {card.sizing_method}<br>"
                    f"<b style='color:#dc2626;font-weight:600;'>Failure Scenario:</b> "
                    f"<span style='color:#2d2d2d;'>{card.failure_scenario}</span></div>",
                    unsafe_allow_html=True,
                )

            # --- Key Levels ---
            meta = card.metadata or {}
            levels = {}
            for _k in ["jp10_level", "target_yield", "stop_yield", "usdjpy_spot",
                        "target", "stop", "put_strike", "straddle_strike",
                        "payer_strike", "call_strike", "receiver_strike",
                        "atm_strike", "otm_strike", "breakeven",
                        "spread_bps", "target_spread_bps"]:
                if _k in meta and meta[_k] is not None:
                    levels[_k.replace("_", " ").title()] = meta[_k]
            if levels:
                st.markdown(
                    "<div style='margin-top:10px;'>"
                    "<span style='font-size:var(--fs-xs);font-weight:700;text-transform:uppercase;"
                    "letter-spacing:var(--ls-widest);color:#8E6F3E;'>Key Levels &amp; Strike Prices</span></div>",
                    unsafe_allow_html=True,
                )
                _lvl_cols = st.columns(min(len(levels), 5))
                for _i, (_lk, _lv) in enumerate(levels.items()):
                    _fmt = f"{_lv:,.2f}" if isinstance(_lv, float) else str(_lv)
                    _lvl_cols[_i % len(_lvl_cols)].metric(_lk, _fmt)

            # --- Payout Graph ---
            try:
                payout_fig = _build_payout_chart(card)
                if payout_fig is not None:
                    _chart(_style_fig(payout_fig, 300))
            except Exception:
                pass

    # --- Export ---
    if filtered:
        st.subheader("Export Trade Ideas")
        _section_note(
            "Download a <b>profile-tailored PDF</b> — each version is optimised for a different audience. "
            "<b>Trader</b>: action-first with regime calls, key levels, and sizing. "
            "<b>Analyst</b>: balanced coverage of models, trades, and performance. "
            "<b>Academic</b>: full methodology, validation detail, and references."
        )

        # Gather framework-wide context for the full report
        _pdf_args = _get_args()
        _pdf_pca = _run_pca(*_pdf_args)
        _pdf_ensemble_val = None
        try:
            _pdf_ens = _run_ensemble(*_pdf_args)
            if _pdf_ens is not None and len(_pdf_ens.dropna()) > 0:
                _pdf_ensemble_val = float(_pdf_ens.dropna().iloc[-1])
        except Exception:
            pass
        _pdf_warn = None
        try:
            _ws = _run_warning_score(*_pdf_args)
            if _ws is not None and len(_ws.dropna()) > 0:
                _pdf_warn = float(_ws.dropna().iloc[-1])
        except Exception:
            pass
        _pdf_ml_prob = None
        _pdf_ml_imp = None
        try:
            _, _ml_p, _ml_i = _run_ml_predictor(*_pdf_args, _get_layout_config().entropy_window)
            if _ml_p is not None and len(_ml_p.dropna()) > 0:
                _pdf_ml_prob = float(_ml_p.dropna().iloc[-1])
                _pdf_ml_imp = _ml_i
        except Exception:
            pass

        _pdf_kwargs = dict(
            regime_state=regime_state,
            pca_result=_pdf_pca,
            ensemble_prob=_pdf_ensemble_val,
            warning_score=_pdf_warn,
            ml_prob=_pdf_ml_prob,
            ml_importance=_pdf_ml_imp,
            cards=filtered,
        )

        col_t, col_a, col_ac, col_csv = st.columns(4)
        for _col, _prof, _icon, _key in [
            (col_t,  "Trader",   "Trader PDF",   "pdf_trader"),
            (col_a,  "Analyst",  "Analyst PDF",  "pdf_analyst"),
            (col_ac, "Academic", "Academic PDF",  "pdf_academic"),
        ]:
            with _col:
                try:
                    _r = JGBReportPDF()
                    _r.add_title_page(
                        title=f"JGB Report  -  {_prof} View",
                        subtitle=f"{len(filtered)} Trades  |  {datetime.now():%Y-%m-%d %H:%M}",
                    )
                    _r.add_full_analysis_report(_prof, **_pdf_kwargs)
                    st.download_button(
                        _icon,
                        data=_r.to_bytes(),
                        file_name=f"jgb_{_prof.lower()}_{datetime.now():%Y%m%d}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key=_key,
                    )
                except Exception as exc:
                    st.warning(f"Could not generate {_prof} PDF: {exc}")

        with col_csv:
            df_export = trade_cards_to_dataframe(filtered)
            csv = df_export.to_csv(index=False)
            st.download_button(
                "Trade Cards CSV",
                data=csv,
                file_name=f"jgb_trade_cards_{datetime.now():%Y%m%d}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # --- Page conclusion ---
    _n_total = len(cards)
    _n_shown = len(filtered)
    _rp = regime_state.get("regime_prob", 0.5)
    _regime_word = "repricing" if _rp > 0.5 else "suppressed"
    if filtered:
        _top = max(filtered, key=lambda c: c.conviction)
        _n_high_c = sum(1 for c in filtered if c.conviction >= 0.7)
        _trade_summary = (
            f"The framework generated <b>{_n_total}</b> trade idea{'s' if _n_total != 1 else ''} "
            f"(<b>{_n_shown}</b> shown after filters) under a <b>{_regime_word}</b> regime "
            f"(probability {_rp:.0%}), led by \"{_top.name}\" at {_top.conviction:.0%} conviction."
        )
        # Verdict
        _verdict_p5 = (
            f"Lead with \"{_top.name}\" ({_top.direction.upper()}, {_top.conviction:.0%} conviction). "
            f"{_n_high_c} idea{'s' if _n_high_c != 1 else ''} above 70% conviction; "
            f"{'allocate risk budget here first.' if _n_high_c > 0 else 'no high-conviction trades; keep sizing conservative.'}"
        )
    else:
        _trade_summary = (
            f"No trade ideas passed the current filters under a <b>{_regime_word}</b> regime "
            f"(probability {_rp:.0%}). Adjust category or conviction filters in the sidebar."
        )
        _verdict_p5 = "No actionable trades at current filter settings. Widen conviction range or wait for a clearer regime signal."
    _page_conclusion(_verdict_p5, _trade_summary)
    _page_footer()


