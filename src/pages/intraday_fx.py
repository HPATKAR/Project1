"""Intraday FX Event Study page."""

from __future__ import annotations

from pathlib import Path

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
from src.pages.early_warning import _run_ml_predictor
from src.reporting.pdf_export import JGBReportPDF


_BOJ_ANNOUNCE_HOUR_UTC = 3  # ~12:00 JST


def _get_args():
    """Retrieve sidebar args from session state."""
    return st.session_state["_app_args"]


def _get_layout_config():
    return st.session_state["_layout_config"]


def _get_alert_notifier():
    return st.session_state.get("_alert_notifier")

from datetime import datetime
from src.reporting.pdf_export import JGBReportPDF

_LSEG_CSV_PATH = Path(__file__).resolve().parents[2] / "output" / "data" / "lseg" / "usdjpy_boj_intraday.csv"


@st.cache_data(show_spinner=False, ttl=3600)
def _load_lseg_intraday() -> "pd.DataFrame | None":
    """Load LSEG intraday USDJPY CSV."""
    if not _LSEG_CSV_PATH.exists():
        return None
    try:
        df = pd.read_csv(_LSEG_CSV_PATH, parse_dates=["Timestamp"])
        if "Timestamp" not in df.columns and df.index.name == "Timestamp":
            df = df.reset_index()
        if "Timestamp" not in df.columns:
            # Try first column as index
            df = pd.read_csv(_LSEG_CSV_PATH, index_col=0, parse_dates=True).reset_index()
            df.rename(columns={df.columns[0]: "Timestamp"}, inplace=True)
        return df
    except Exception:
        return None



def page_intraday_fx():
    st.header("Intraday FX Event Study")
    _page_intro(
        "Minute-level USDJPY price action across <b>7 BOJ announcement dates</b> from March\u2013December 2025, "
        "sourced from LSEG (Refinitiv) via Purdue University's data subscription. This page examines "
        "how currency markets react in real-time to BOJ decisions - a critical dimension for understanding "
        "the transmission mechanism from policy changes to market repricing."
    )
    _definition_block(
        "What Is an FX Event Study?",
        "An event study isolates the market reaction to a specific event (here, BOJ announcements) by "
        "examining price behavior in a tight window around the event time. By looking at minute-by-minute "
        "data, we can observe:"
        "<br><br>"
        "<b>Price impact:</b> How much does USDJPY move when the BOJ announces? <b>A hawkish surprise "
        "(rate hike, YCC tweak) typically strengthens the yen (USDJPY drops). A dovish hold weakens it "
        "(USDJPY rises).</b> Think of it this way: if Japan raises interest rates, holding yen becomes "
        "more attractive, so investors buy yen and USDJPY falls."
        "<br><br>"
        "<b>Liquidity stress:</b> The bid-ask spread widens sharply during announcements as market makers "
        "pull quotes. <b>In plain terms: when the BOJ is about to announce, banks and dealers become "
        "afraid to quote tight prices because they might get 'picked off' by someone who processes "
        "the news faster. Wider spreads = higher cost to trade = more uncertainty.</b>"
        "<br><br>"
        "<b>Activity surge:</b> Tick counts (number of quote updates per minute) spike during announcements. "
        "<b>A normal minute might have 50-100 quote updates. During a BOJ announcement, this can jump to "
        "500+. That spike represents thousands of traders and algorithms simultaneously repositioning.</b>"
        "<br><br>"
        "<b>Price discovery speed:</b> How quickly does USDJPY find its new equilibrium? <b>Some "
        "announcements resolve in 5 minutes (market quickly agrees on new fair value). Others see "
        "volatility persist for hours (market is genuinely uncertain about policy implications).</b>"
        "<br><br>"
        "<b>Why USDJPY specifically?</b> The yen is the most liquid expression of BOJ policy expectations. "
        "<b>Currency markets react faster than bond markets</b> because FX trades 24 hours with deeper "
        "global liquidity, while JGB trading is concentrated in Tokyo hours. A BOJ rate hike makes yen "
        "deposits more attractive (higher yield), strengthening JPY (USDJPY falls). A dovish hold does "
        "the opposite."
        "<br><br>"
        "<b>Connection to JGB repricing:</b> This page complements the daily-frequency analysis on other "
        "pages. <b>If the Regime Detection page shows a repricing regime, the intraday data here reveals "
        "HOW that repricing actually happens in real time - was it a sudden shock or a gradual grind?</b>"
        "<br><br>"
        "<b>Data source:</b> LSEG (formerly Refinitiv) via Purdue University Daniels School institutional "
        "subscription. 1-minute bid/ask/mid prices for USDJPY spot (RIC: JPY=)."
    )

    df = _load_lseg_intraday()
    if df is None or len(df) == 0:
        st.warning(
            "No LSEG intraday data found. Place `usdjpy_boj_intraday.csv` in "
            "`output/data/lseg/` to enable this page."
        )
        st.code("output/data/lseg/usdjpy_boj_intraday.csv", language="text")
        _page_footer()
        return

    # Parse and clean
    try:
        if "boj_date" not in df.columns:
            df["boj_date"] = df["Timestamp"].dt.date.astype(str)

        boj_dates = sorted(df["boj_date"].unique())
        mid_col = "MID_PRICE" if "MID_PRICE" in df.columns else "BID"

        # --- Summary metrics ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("BOJ Dates Covered", len(boj_dates))
        c2.metric("Total Data Points", f"{len(df):,}")
        c3.metric("Date Range", f"{boj_dates[0]} to {boj_dates[-1]}")
        avg_mid = df[mid_col].mean()
        c4.metric("Avg USDJPY", f"{avg_mid:.2f}")

        _section_note(
            f"<b>What these numbers mean:</b> We have {len(boj_dates)} BOJ meeting days of "
            f"minute-by-minute data, totaling {len(df):,} individual price snapshots. "
            f"The average USDJPY level across all observations is {avg_mid:.2f}. "
            f"<b>Each data point represents one minute of trading activity - the bid price, "
            f"ask price, midpoint, and how many times quotes updated during that minute.</b>"
        )

        # Prominent listing of all BOJ dates covered
        _date_labels = {
            "2025-03-14": "March 2025",
            "2025-05-01": "May 2025",
            "2025-06-17": "June 2025",
            "2025-07-31": "July 2025",
            "2025-09-19": "September 2025",
            "2025-10-31": "October 2025",
            "2025-12-19": "December 2025",
        }
        _listed = " &bull; ".join(
            f"<b>{_date_labels.get(d, d)}</b>" for d in boj_dates
        )
        _takeaway_block(
            f"This dataset spans <b>{len(boj_dates)} BOJ monetary policy announcements</b>: "
            f"{_listed}. "
            f"Cross-event comparison reveals how market reaction patterns evolved as the BOJ "
            f"progressively normalised policy through 2025 \u2014 from cautious holds to active "
            f"rate adjustments."
        )
    except Exception as exc:
        st.error(f"Error processing LSEG data: {exc}")
        _page_footer()
        return

    # =================================================================
    # Section 1: Overlay of all BOJ announcement days
    # =================================================================
    st.subheader("USDJPY Price Action on BOJ Days (Overlay)")
    _section_note(
        "Each colored line represents one BOJ meeting day. The x-axis shows minutes from midnight UTC "
        "(BOJ typically announces around 03:00 UTC / 12:00 JST, marked by the gold band). "
        "All prices are rebased to 0 at the start of each day so we can compare relative moves."
        "<br><br>"
        "<b>Key insight: Look at what happens at the gold band.</b> Before the announcement, "
        "lines should cluster tightly (market is waiting). After the announcement, lines diverge - "
        "days with big policy surprises show large jumps, while routine meetings show minimal movement. "
        "<b>The speed and magnitude of divergence tells you how surprising the BOJ decision was.</b>"
    )

    try:
        fig_overlay = go.Figure()
        day_moves = {}
        for boj_date in boj_dates:
            day_df = df[df["boj_date"] == boj_date].copy()
            if len(day_df) < 10:
                continue
            day_df = day_df.sort_values("Timestamp")
            day_df["minutes"] = (
                day_df["Timestamp"].dt.hour * 60 + day_df["Timestamp"].dt.minute
            )
            base_price = day_df[mid_col].iloc[0]
            if base_price > 0:
                day_df["rebased"] = (day_df[mid_col] / base_price - 1) * 10000  # bps
            else:
                continue

            end_move = day_df["rebased"].iloc[-1]
            day_moves[boj_date] = end_move

            fig_overlay.add_trace(go.Scatter(
                x=day_df["minutes"],
                y=day_df["rebased"],
                mode="lines",
                name=boj_date,
                line=dict(width=1.5),
                hovertemplate=f"<b>{boj_date}</b><br>Min from midnight: %{{x}}<br>Move: %{{y:.1f}} bps<extra></extra>",
            ))

        fig_overlay.add_vrect(
            x0=_BOJ_ANNOUNCE_HOUR_UTC * 60 - 5,
            x1=_BOJ_ANNOUNCE_HOUR_UTC * 60 + 30,
            fillcolor="rgba(207,185,145,0.15)",
            line_width=0,
            annotation_text="BOJ Window",
            annotation_position="top left",
        )
        fig_overlay.add_vline(
            x=_BOJ_ANNOUNCE_HOUR_UTC * 60,
            line_dash="dash", line_color="#CFB991", line_width=2,
            annotation_text="~12:00 JST",
        )
        fig_overlay.update_layout(
            xaxis_title="Minutes from Midnight (UTC)",
            yaxis_title="Price Change (bps from open)",
            hovermode="x unified",
        )
        _chart(_style_fig(fig_overlay, 450))

        # Bold interpretation
        if day_moves:
            biggest_day = max(day_moves, key=lambda k: abs(day_moves[k]))
            biggest_move = day_moves[biggest_day]
            direction_word = "weakened (rose)" if biggest_move > 0 else "strengthened (fell)"
            _section_note(
                f"<b>How to read this chart:</b> Each line starts at 0 (its opening price) and shows "
                f"how USDJPY moved through the day in basis points. Lines going UP mean the yen "
                f"weakened (USDJPY rose = bad for yen). Lines going DOWN mean the yen strengthened "
                f"(USDJPY fell = good for yen)."
                f"<br><br>"
                f"<b>Biggest mover: {biggest_day}</b> - USDJPY {direction_word} by "
                f"{abs(biggest_move):.1f} bps from open to close. "
                f"{'This suggests a significant hawkish surprise from the BOJ that day.' if biggest_move < -20 else 'This suggests a significant dovish surprise or no-change decision.' if biggest_move > 20 else 'This was a relatively contained reaction.'}"
            )
    except Exception as exc:
        st.warning(f"Could not render overlay chart: {exc}")

    # =================================================================
    # Section 2: Individual day deep dive
    # =================================================================
    st.subheader("Single-Day Deep Dive")
    _section_note(
        "Select a BOJ meeting date below to see exactly what happened that day: the full price "
        "path, how wide spreads got (liquidity stress), and how frantic the trading activity was. "
        "<b>This is like putting the market reaction under a microscope - you can see the exact "
        "minute when news hit and how quickly the market digested it.</b>"
    )

    selected_date = st.selectbox("Select BOJ Meeting Date", boj_dates, index=len(boj_dates) - 1)

    try:
        day_df = df[df["boj_date"] == selected_date].copy().sort_values("Timestamp")
        if len(day_df) < 5:
            st.info(f"Insufficient data for {selected_date}.")
        else:
            day_open = day_df[mid_col].iloc[0]
            day_close = day_df[mid_col].iloc[-1]
            day_high = day_df[mid_col].max()
            day_low = day_df[mid_col].min()
            day_range = day_high - day_low
            day_move = day_close - day_open

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Open", f"{day_open:.2f}")
            mc2.metric("Close", f"{day_close:.2f}", delta=f"{day_move:+.2f}")
            mc3.metric("High", f"{day_high:.2f}")
            mc4.metric("Low", f"{day_low:.2f}")
            mc5.metric("Range (pips)", f"{day_range * 100:.0f}")

            # Bold day interpretation
            move_pips = day_move * 100
            range_pips = day_range * 100
            if abs(move_pips) > 50:
                day_verdict = (
                    f"<b>Big day.</b> USDJPY moved {abs(move_pips):.0f} pips "
                    f"({'higher - yen weakened' if move_pips > 0 else 'lower - yen strengthened'}). "
                    f"A {range_pips:.0f}-pip intraday range is well above normal. "
                    f"<b>This was likely a meaningful policy surprise from the BOJ.</b>"
                )
            elif abs(move_pips) > 20:
                day_verdict = (
                    f"<b>Moderate reaction.</b> USDJPY moved {abs(move_pips):.0f} pips net, "
                    f"with a {range_pips:.0f}-pip range. <b>The market adjusted to BOJ guidance but "
                    f"the decision was partially priced in beforehand.</b>"
                )
            else:
                day_verdict = (
                    f"<b>Quiet day.</b> USDJPY moved only {abs(move_pips):.0f} pips net. "
                    f"<b>The BOJ decision was largely expected - no surprise, no major repositioning. "
                    f"This is what a 'priced in' meeting looks like.</b>"
                )
            _section_note(day_verdict)

            # --- Price chart ---
            fig_day = go.Figure()
            if "BID" in day_df.columns and "ASK" in day_df.columns:
                fig_day.add_trace(go.Scatter(
                    x=day_df["Timestamp"], y=day_df["ASK"],
                    mode="lines", name="Ask (what you pay to buy)",
                    line=dict(color="#c0392b", width=0.8),
                    hovertemplate="Ask: %{y:.2f}<extra></extra>",
                ))
                fig_day.add_trace(go.Scatter(
                    x=day_df["Timestamp"], y=day_df["BID"],
                    mode="lines", name="Bid (what you get to sell)",
                    line=dict(color="#2e7d32", width=0.8),
                    fill="tonexty", fillcolor="rgba(207,185,145,0.1)",
                    hovertemplate="Bid: %{y:.2f}<extra></extra>",
                ))
            fig_day.add_trace(go.Scatter(
                x=day_df["Timestamp"], y=day_df[mid_col],
                mode="lines", name="Mid (fair value)",
                line=dict(color="#000", width=2),
                hovertemplate="Mid: %{y:.2f}<extra></extra>",
            ))
            fig_day.update_layout(
                xaxis_title="Time (UTC)", yaxis_title="USDJPY",
                hovermode="x unified",
            )
            _chart(_style_fig(fig_day, 400))

            _section_note(
                "<b>Reading the price chart:</b> The black line is the mid-price (average of bid and ask) - "
                "this is the 'true' price. The shaded area between red (ask) and green (bid) lines is the "
                "spread. <b>When the shaded area gets wider, it means the market is stressed and trading "
                "costs are higher.</b> Look for sharp moves in the black line around the BOJ announcement time."
            )

            # --- Bid-Ask Spread ---
            if "BID" in day_df.columns and "ASK" in day_df.columns:
                st.subheader("Bid-Ask Spread (Liquidity Stress)")
                _section_note(
                    "<b>What is the bid-ask spread?</b> If you want to buy USDJPY right now, you pay the "
                    "Ask price. If you want to sell, you get the Bid price. The difference is the spread - "
                    "it is the market maker's compensation for providing liquidity. "
                    "<b>In normal conditions, USDJPY spread is about 0.5-2 pips. During BOJ announcements, "
                    "it can jump to 5-20+ pips because banks are scared of being on the wrong side of a "
                    "policy surprise.</b>"
                    "<br><br>"
                    "<b>Why does this matter?</b> If you are trading around BOJ announcements, wider spreads "
                    "mean you pay more to enter and exit positions. A spread spike from 1 pip to 10 pips on "
                    "a $10M position costs an extra $10,000 just in execution costs."
                )
                day_df["spread_pips"] = (day_df["ASK"] - day_df["BID"]) * 100

                fig_spread = go.Figure()
                fig_spread.add_trace(go.Scatter(
                    x=day_df["Timestamp"], y=day_df["spread_pips"],
                    mode="lines", name="Spread (pips)",
                    line=dict(color="#CFB991", width=1.5),
                    fill="tozeroy", fillcolor="rgba(207,185,145,0.2)",
                    hovertemplate="Spread: %{y:.1f} pips<extra></extra>",
                ))
                fig_spread.update_layout(
                    xaxis_title="Time (UTC)", yaxis_title="Bid-Ask Spread (pips)",
                    hovermode="x unified",
                )
                _chart(_style_fig(fig_spread, 300))

                avg_spread = day_df["spread_pips"].mean()
                max_spread = day_df["spread_pips"].max()
                max_spread_time = day_df.loc[day_df["spread_pips"].idxmax(), "Timestamp"]
                spread_ratio = max_spread / max(avg_spread, 0.01)
                near_boj = abs(max_spread_time.hour - _BOJ_ANNOUNCE_HOUR_UTC) <= 1
                _section_note(
                    f"<b>Average spread: {avg_spread:.1f} pips</b> (normal trading cost). "
                    f"<b>Maximum spread: {max_spread:.1f} pips</b> at {max_spread_time.strftime('%H:%M UTC')} "
                    f"({spread_ratio:.1f}x the average). "
                    f"<br><br>"
                    f"<b>{'INFERENCE: The max spread occurred near the BOJ announcement window. This confirms ' if near_boj else 'NOTE: The max spread occurred outside the typical BOJ window. This suggests '}"
                    f"{'that liquidity dried up precisely when the policy decision hit the wires - ' if near_boj else 'other factors (perhaps pre-positioning or overseas news) drove the spread widening '}"
                    f"{'market makers pulled their quotes to avoid being caught on the wrong side.' if near_boj else 'rather than the BOJ announcement itself.'}"
                )

            # --- Tick Activity ---
            bid_mov_col = "BID_NUMMOV" if "BID_NUMMOV" in day_df.columns else None
            ask_mov_col = "ASK_NUMMOV" if "ASK_NUMMOV" in day_df.columns else None

            if bid_mov_col or ask_mov_col:
                st.subheader("Market Activity (Tick Count)")
                _section_note(
                    "<b>What are tick counts?</b> Every time a bank or dealer updates their quote (even by "
                    "0.01 yen), that counts as one 'tick'. <b>More ticks per minute = more frantic trading "
                    "activity.</b> During calm markets, you might see 50-100 ticks/minute. During a BOJ "
                    "announcement, this can spike to 300-500+ as algorithms and traders simultaneously "
                    "reprice their positions."
                    "<br><br>"
                    "<b>What to look for:</b> A spike in ticks combined with a spread widening (see above) "
                    "is the classic signature of a market-moving event. <b>High ticks + narrow spread = "
                    "orderly repricing. High ticks + wide spread = panic.</b>"
                )
                fig_ticks = go.Figure()
                if bid_mov_col:
                    fig_ticks.add_trace(go.Bar(
                        x=day_df["Timestamp"], y=day_df[bid_mov_col],
                        name="Bid Updates", marker_color="rgba(46,125,50,0.6)",
                        hovertemplate="Bid ticks: %{y}<extra></extra>",
                    ))
                if ask_mov_col:
                    fig_ticks.add_trace(go.Bar(
                        x=day_df["Timestamp"], y=day_df[ask_mov_col],
                        name="Ask Updates", marker_color="rgba(192,57,43,0.6)",
                        hovertemplate="Ask ticks: %{y}<extra></extra>",
                    ))
                fig_ticks.update_layout(
                    xaxis_title="Time (UTC)", yaxis_title="Quote Updates / Minute",
                    barmode="overlay", hovermode="x unified",
                )
                _chart(_style_fig(fig_ticks, 300))

                if bid_mov_col:
                    total_ticks = day_df[bid_mov_col].sum()
                    peak_ticks = day_df[bid_mov_col].max()
                    avg_ticks = day_df[bid_mov_col].mean()
                    peak_time = day_df.loc[day_df[bid_mov_col].idxmax(), "Timestamp"]
                    tick_ratio = peak_ticks / max(avg_ticks, 1)
                    _section_note(
                        f"<b>Total bid quote updates: {total_ticks:,.0f}</b>. "
                        f"Average: {avg_ticks:.0f} ticks/minute. "
                        f"<b>Peak activity: {peak_ticks:,.0f} ticks/min at {peak_time.strftime('%H:%M UTC')} "
                        f"({tick_ratio:.1f}x the average).</b>"
                        f"<br><br>"
                        f"<b>{'INFERENCE: Peak activity was ' if tick_ratio > 3 else 'NOTE: Activity was '}"
                        f"{tick_ratio:.1f}x the daily average. "
                        f"{'This is a strong signal that this was a genuine market-moving event - traders were ' if tick_ratio > 3 else 'This suggests relatively calm trading - the BOJ decision was largely '}"
                        f"{'scrambling to reposition.' if tick_ratio > 3 else 'anticipated by the market.'}</b>"
                    )

    except Exception as exc:
        st.warning(f"Could not render day detail: {exc}")

    # =================================================================
    # Section 3: Cross-date comparison
    # =================================================================
    st.subheader("Announcement Reaction Summary")
    _section_note(
        "<b>This is the most important table on this page.</b> It compares the FX reaction across "
        "every BOJ meeting date. The 'Reaction' column shows how many pips USDJPY moved from 30 "
        "minutes before to 60 minutes after the approximate announcement time (~03:00 UTC / 12:00 JST). "
        "<b>Positive = yen weakened. Negative = yen strengthened.</b>"
        "<br><br>"
        "<b>Why 30 min before to 60 min after?</b> We start 30 minutes early to capture any pre-announcement "
        "positioning or leaks. We extend 60 minutes after to allow the market to find its new equilibrium. "
        "Most of the reaction happens in the first 5-15 minutes, but the full 60-minute window captures "
        "secondary reactions as analysts publish commentary."
    )

    reactions = []
    try:
        for boj_date in boj_dates:
            day_df = df[df["boj_date"] == boj_date].copy().sort_values("Timestamp")
            if len(day_df) < 10:
                continue
            day_df["minute_of_day"] = day_df["Timestamp"].dt.hour * 60 + day_df["Timestamp"].dt.minute

            pre_mask = (day_df["minute_of_day"] >= (_BOJ_ANNOUNCE_HOUR_UTC * 60 - 30)) & \
                       (day_df["minute_of_day"] < _BOJ_ANNOUNCE_HOUR_UTC * 60)
            post_mask = (day_df["minute_of_day"] >= _BOJ_ANNOUNCE_HOUR_UTC * 60) & \
                        (day_df["minute_of_day"] <= (_BOJ_ANNOUNCE_HOUR_UTC * 60 + 60))

            pre_df = day_df[pre_mask]
            post_df = day_df[post_mask]

            if len(pre_df) == 0 or len(post_df) == 0:
                pre_price = day_df[mid_col].iloc[0]
                post_price = day_df[mid_col].iloc[-1]
            else:
                pre_price = pre_df[mid_col].iloc[0]
                post_price = post_df[mid_col].iloc[-1]

            move_pips = (post_price - pre_price) * 100
            day_range = (day_df[mid_col].max() - day_df[mid_col].min()) * 100

            if "BID" in day_df.columns and "ASK" in day_df.columns:
                avg_spread = ((day_df["ASK"] - day_df["BID"]) * 100).mean()
                max_spread = ((day_df["ASK"] - day_df["BID"]) * 100).max()
            else:
                avg_spread = max_spread = np.nan

            reactions.append({
                "Date": boj_date,
                "Pre-Price": pre_price,
                "Post-Price": post_price,
                "Reaction (pips)": round(move_pips, 1),
                "Day Range (pips)": round(day_range, 1),
                "Avg Spread (pips)": round(avg_spread, 1),
                "Max Spread (pips)": round(max_spread, 1),
            })

        if reactions:
            react_df = pd.DataFrame(reactions)

            fig_react = go.Figure()
            colors = ["#2e7d32" if r >= 0 else "#c0392b" for r in react_df["Reaction (pips)"]]
            fig_react.add_trace(go.Bar(
                x=react_df["Date"],
                y=react_df["Reaction (pips)"],
                marker_color=colors,
                hovertemplate="<b>%{x}</b><br>Reaction: %{y:+.1f} pips<extra></extra>",
            ))
            fig_react.update_layout(
                xaxis_title="BOJ Meeting Date",
                yaxis_title="USDJPY Reaction (pips)",
                xaxis_tickangle=-45,
            )
            fig_react.add_hline(y=0, line_dash="dash", line_color="#888")
            _chart(_style_fig(fig_react, 400))

            _section_note(
                "<b>Reading the bar chart:</b> Green bars = USDJPY rose = yen weakened (dovish or no-change "
                "decision). Red bars = USDJPY fell = yen strengthened (hawkish surprise, rate hike, or "
                "policy tightening signal). <b>The taller the bar, the bigger the surprise relative to what "
                "the market expected.</b> A zero bar means the decision was perfectly priced in."
            )

            st.dataframe(
                react_df.style.format({
                    "Pre-Price": "{:.2f}", "Post-Price": "{:.2f}",
                    "Reaction (pips)": "{:+.1f}", "Day Range (pips)": "{:.1f}",
                    "Avg Spread (pips)": "{:.1f}", "Max Spread (pips)": "{:.1f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            avg_abs_reaction = react_df["Reaction (pips)"].abs().mean()
            max_reaction = react_df.loc[react_df["Reaction (pips)"].abs().idxmax()]
            n_positive = (react_df["Reaction (pips)"] > 0).sum()
            n_negative = (react_df["Reaction (pips)"] < 0).sum()

            _section_note(
                f"<b>KEY FINDINGS:</b>"
                f"<br>- Across <b>{len(reactions)} BOJ meetings</b>, the average absolute FX reaction "
                f"is <b>{avg_abs_reaction:.1f} pips</b>."
                f"<br>- <b>Largest reaction: {max_reaction['Reaction (pips)']:+.1f} pips on "
                f"{max_reaction['Date']}</b> - this was the most surprising BOJ decision in the dataset."
                f"<br>- Yen weakened (USDJPY up) <b>{n_positive} times</b>, strengthened (USDJPY down) "
                f"<b>{n_negative} times</b>."
                f"<br>- <b>{'The yen has strengthened more often, suggesting BOJ has been more hawkish than markets expected.' if n_negative > n_positive else 'The yen has weakened more often, suggesting BOJ has been more dovish than markets expected.' if n_positive > n_negative else 'Reactions are balanced - the market has been equally surprised in both directions.'}</b>"
                f"<br>- Average max spread on BOJ days: <b>{react_df['Max Spread (pips)'].mean():.1f} pips</b> "
                f"(normal is ~1-2 pips). <b>This means liquidity routinely dries up during BOJ announcements.</b>"
            )
    except Exception as exc:
        st.warning(f"Could not compute reactions: {exc}")

    # =================================================================
    # Section 4: Export
    # =================================================================
    st.subheader("Export Intraday Data")
    _section_note(
        "Download profile-tailored PDFs or raw CSV. "
        "**Trader**: reaction summary + key levels. **Analyst**: full analysis. **Academic**: methodology + references."
    )

    try:
        _fx_args = _get_args()
        _fx_kwargs = dict(reactions=reactions if reactions else [])
        # Gather extra context for full report
        try:
            _fx_ens = _run_ensemble(*_fx_args)
            if _fx_ens is not None and len(_fx_ens.dropna()) > 0:
                _fx_kwargs["ensemble_prob"] = float(_fx_ens.dropna().iloc[-1])
        except Exception:
            pass
        try:
            _, _fx_ml_p, _fx_ml_i = _run_ml_predictor(*_fx_args, _get_layout_config().entropy_window)
            if _fx_ml_p is not None and len(_fx_ml_p.dropna()) > 0:
                _fx_kwargs["ml_prob"] = float(_fx_ml_p.dropna().iloc[-1])
                _fx_kwargs["ml_importance"] = _fx_ml_i
        except Exception:
            pass

        col_t, col_a, col_ac, col_csv = st.columns(4)
        for _col, _prof, _key in [
            (col_t,  "Trader",   "fx_pdf_trader"),
            (col_a,  "Analyst",  "fx_pdf_analyst"),
            (col_ac, "Academic", "fx_pdf_academic"),
        ]:
            with _col:
                try:
                    _r = JGBReportPDF()
                    _r.add_title_page(
                        title=f"FX Event Study  -  {_prof}",
                        subtitle=f"USDJPY Around BOJ  |  {datetime.now():%Y-%m-%d %H:%M}",
                    )
                    _r.add_full_analysis_report(_prof, **_fx_kwargs)
                    st.download_button(
                        f"{_prof} PDF",
                        data=_r.to_bytes(),
                        file_name=f"jgb_fx_{_prof.lower()}_{datetime.now():%Y%m%d}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key=_key,
                    )
                except Exception as exc:
                    st.warning(f"Could not generate {_prof} PDF: {exc}")

        with col_csv:
            csv_export = df.to_csv(index=False)
            st.download_button(
                "Intraday CSV",
                data=csv_export,
                file_name="usdjpy_boj_intraday_export.csv",
                mime="text/csv",
                use_container_width=True,
            )
    except Exception:
        pass

    # Conclusion
    try:
        if reactions:
            _best = max(reactions, key=lambda r: abs(r["Reaction (pips)"]))
            _verdict = (
                f"The largest FX reaction was {_best['Reaction (pips)']:+.1f} pips on {_best['Date']}. "
                f"Average absolute reaction across {len(reactions)} meetings is {avg_abs_reaction:.1f} pips. "
                f"{'Hawkish surprises dominate - the BOJ has been tighter than expected.' if n_negative > n_positive else 'Dovish outcomes dominate - markets keep overestimating BOJ hawkishness.' if n_positive > n_negative else 'Reactions are balanced between hawkish and dovish outcomes.'}"
            )
            _summary = (
                f"Intraday USDJPY analysis covers <b>{len(boj_dates)}</b> BOJ meeting dates with "
                f"<b>{len(df):,}</b> minute-level observations from LSEG (Refinitiv). "
                f"The data reveals how currency markets process BOJ policy decisions in real time, "
                f"with clear patterns of liquidity withdrawal, activity spikes, and asymmetric reactions "
                f"depending on the direction of the policy surprise."
            )
            _page_conclusion(_verdict, _summary)
    except Exception:
        pass

    _page_footer()


