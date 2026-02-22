"""Overview & Data page."""

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


def _get_args():
    """Retrieve sidebar args from session state."""
    return st.session_state["_app_args"]


def _get_layout_config():
    return st.session_state["_layout_config"]


def _get_alert_notifier():
    return st.session_state.get("_alert_notifier")

from src.data.config import JAPAN_CREDIT_RATINGS, BOJ_CREDIBILITY_EVENTS



def page_overview():
    st.header("Overview & Data")
    _page_intro(
        "Unified market data pipeline feeding all downstream analytics. This page shows sovereign yields, "
        "FX rates, equity indices, and volatility measures across Japan, the U.S., Europe, and Asia-Pacific. "
        "Toggle between live (FRED + yfinance) and simulated data via sidebar controls. All charts overlay "
        "BOJ policy event dates (red verticals) to ground market moves in the policy timeline."
    )

    with st.spinner("Loading market data..."):
        try:
            df = load_unified(*_get_args())
        except Exception as exc:
            st.error(f"Failed to load data: {exc}")
            return

    # --- KPI row ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Date Range", f"{df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}")
    c2.metric("Rows", f"{len(df):,}")
    c3.metric("Sources", "Simulated" if _get_args()[0] else "FRED + yfinance")
    c4.metric("Columns", f"{df.shape[1]}")

    # --- Rates chart ---
    st.subheader("Sovereign Yields & VIX")
    _definition_block(
        "What are Sovereign Yields?",
        "When a government needs money, it borrows by issuing bonds. The <b>yield</b> is the annual interest "
        "rate investors earn for lending to that government. Think of it like the interest rate on a loan, "
        "but in reverse: you are the lender, and the government is the borrower. "
        "A <b>10-year yield</b> (like JP_10Y) tells you the annual return for lending to Japan for 10 years. "
        "This chart shows yields from five countries: Japan (JP), the US, Germany (DE), the UK, and Australia (AU). "
        "Normally, countries with stronger growth and higher inflation have higher yields. Japan's yields have been "
        "near zero for years because the Bank of Japan (BOJ) artificially held them down through massive bond buying. "
        "Now that the BOJ is stepping back, the key question is: how fast will Japanese yields rise to match global peers? "
        "The <b>VIX</b> (orange line) is the 'fear index' for US stock markets. When VIX spikes above 25, "
        "it means investors are panicking. During panics, money flows into safe assets like government bonds, "
        "temporarily pushing yields down. The red dotted vertical lines on this chart mark BOJ policy announcements "
        "so you can see exactly how yields reacted to each decision."
    )
    rate_cols = [c for c in ["JP_10Y", "US_10Y", "DE_10Y", "UK_10Y", "AU_10Y", "VIX"] if c in df.columns]
    if rate_cols:
        # Compute chart-specific insights
        _jp = df["JP_10Y"].dropna() if "JP_10Y" in df.columns else pd.Series(dtype=float)
        _us = df["US_10Y"].dropna() if "US_10Y" in df.columns else pd.Series(dtype=float)
        _vix = df["VIX"].dropna() if "VIX" in df.columns else pd.Series(dtype=float)
        insight = ""
        if len(_jp) > 0 and len(_us) > 0:
            spread = float(_jp.iloc[-1] - _us.iloc[-1])
            if abs(spread) < 0.5:
                insight += f" <b>Actionable: The JP-US 10Y spread is only {spread:+.2f}%. JGB yields are converging toward US levels, confirming repricing is underway.</b>"
            else:
                insight += f" <b>Actionable: The JP-US 10Y spread sits at {spread:+.2f}%. The BOJ is still suppressing yields well below the US benchmark; watch for catch-up risk.</b>"
        if len(_vix) > 0 and float(_vix.iloc[-1]) > 25:
            insight += f" <b>VIX at {float(_vix.iloc[-1]):.1f} flags elevated market fear. Risk-off spillover into JGBs is likely.</b>"
        _section_note(
            "JP_10Y vs US/DE benchmarks. Red verticals = BOJ policy dates."
            + insight
        )
        fig = go.Figure()
        for col in rate_cols:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))
        _add_boj_events(fig)
        _chart(_style_fig(fig, 420))
        # Takeaway
        if len(_jp) > 0:
            _jp_last = float(_jp.iloc[-1])
            _jp_1y_ago = float(_jp.iloc[-252]) if len(_jp) >= 252 else float(_jp.iloc[0])
            _jp_chg = _jp_last - _jp_1y_ago
            _takeaway_block(
                f"JP 10Y currently at <b>{_jp_last:.3f}%</b>, "
                f"{'up' if _jp_chg > 0 else 'down'} <b>{abs(_jp_chg):.2f}%</b> from a year ago. "
                f"{'The secular rise confirms the repricing thesis: yields are normalising from BOJ-suppressed levels.' if _jp_chg > 0.1 else 'Yields remain anchored. BOJ suppression is intact or the market expects continued accommodation.' if _jp_chg < 0.05 else 'Modest moves. The market is in wait-and-see mode ahead of the next BOJ decision.'}"
            )
    else:
        st.info("No rate columns found in data.")

    # --- Asian Equity Returns Comparison ---
    st.subheader("Asian Equity Returns")
    _definition_block(
        "Why Cross-Asian Equity Comparison Matters for JGBs",
        "Stock markets in Asia are connected. When one country's market falls, others often follow because "
        "global investors tend to move money in and out of Asia as a region. This chart shows <b>cumulative returns</b> "
        "(total gain or loss since the start date, measured in %) for five major Asian stock indices. "
        "All lines start at 0% so you can directly compare performance. "
        "If the Nikkei (Japan) rises while others fall, it means something Japan-specific is attracting money, "
        "like foreign investors buying Japanese stocks. Those foreign investors usually also sell yen (hedging), "
        "which weakens the currency and indirectly pressures the BOJ to raise rates, pushing JGB yields higher. "
        "If all Asian markets fall together, it signals a global risk-off event where investors worldwide are "
        "selling risky assets. In that scenario, money flows into safe-haven bonds like JGBs, pushing yields down. "
        "<b>How to read this chart:</b> Lines going up = positive returns. Lines going down = losses. "
        "The gap between lines shows relative performance."
    )
    asian_eq_cols = [c for c in ["NIKKEI", "SENSEX", "HANGSENG", "SHANGHAI", "KOSPI"] if c in df.columns]
    if len(asian_eq_cols) >= 2:
        asian_eq = df[asian_eq_cols].dropna()
        if len(asian_eq) > 1:
            cum_returns = (asian_eq / asian_eq.iloc[0] - 1) * 100
            _section_note(
                "Cumulative returns (%) from first available date, normalized to 0%. "
                "Divergence between Nikkei and regional peers highlights Japan-specific risk premia."
            )
            fig_asian = go.Figure()
            color_map = {"NIKKEI": "#E8413C", "SENSEX": "#2196F3", "HANGSENG": "#4CAF50", "SHANGHAI": "#FF9800", "KOSPI": "#9C27B0"}
            label_map = {"NIKKEI": "\U0001F1EF\U0001F1F5 Nikkei 225 (Tokyo)", "SENSEX": "\U0001F1EE\U0001F1F3 Sensex (Mumbai)", "HANGSENG": "\U0001F1ED\U0001F1F0 Hang Seng (Hong Kong)", "SHANGHAI": "\U0001F1E8\U0001F1F3 SSE Composite (Shanghai)", "KOSPI": "\U0001F1F0\U0001F1F7 KOSPI (Seoul)"}
            for col in asian_eq_cols:
                fig_asian.add_trace(go.Scatter(
                    x=cum_returns.index, y=cum_returns[col],
                    mode="lines", name=label_map.get(col, col),
                    line=dict(color=color_map.get(col)),
                ))
            fig_asian.update_layout(yaxis_title="Cumulative Return (%)")
            _add_boj_events(fig_asian)
            _chart(_style_fig(fig_asian, 420))
            # Takeaway
            latest = cum_returns.iloc[-1]
            best = latest.idxmax()
            worst = latest.idxmin()
            _takeaway_block(
                f"<b>{best}</b> leads with <b>{latest[best]:+.1f}%</b> cumulative return; "
                f"<b>{worst}</b> trails at <b>{latest[worst]:+.1f}%</b>. "
                f"{'Nikkei outperformance vs Asian peers suggests Japan-specific inflows. Watch for hedging pressure on JGBs.' if best == 'NIKKEI' else 'Nikkei underperformance relative to Asian peers may reflect BOJ tightening fears or yen-driven headwinds.'}"
            )
    else:
        st.info("Insufficient Asian equity data for cross-market comparison.")

    # --- Market chart ---
    st.subheader("FX & Equity")
    _definition_block(
        "Why FX & Equity Matter for JGBs",
        "<b>USDJPY</b> is how many Japanese yen it costs to buy one US dollar. When this number goes up "
        "(e.g. from 140 to 155), the yen is getting <em>weaker</em>. A weak yen makes imports more expensive "
        "for Japan, which pushes up inflation. Higher inflation forces the BOJ to consider raising interest rates, "
        "which means JGB yields go up (and JGB prices go down, hurting bondholders). "
        "<b>EURJPY</b> is the same concept but for the euro. "
        "<b>NIKKEI 225</b> is Japan's main stock market index (similar to the S&P 500 in the US). "
        "When stocks fall AND the yen weakens at the same time, it is a danger signal: it means foreign investors "
        "are pulling money out of Japan entirely (selling both stocks and yen). This 'capital flight' scenario is "
        "the worst case for JGB holders because it can force rapid yield increases. "
        "<b>How to read this chart:</b> USDJPY rising = yen weakening = bearish for JGBs. "
        "Nikkei falling while USDJPY rises = maximum stress. Red verticals mark BOJ policy dates."
    )
    mkt_cols = [c for c in ["USDJPY", "EURJPY", "NIKKEI"] if c in df.columns]
    if mkt_cols:
        _usdjpy = df["USDJPY"].dropna() if "USDJPY" in df.columns else pd.Series(dtype=float)
        _nikkei = df["NIKKEI"].dropna() if "NIKKEI" in df.columns else pd.Series(dtype=float)
        fx_insight = ""
        if len(_usdjpy) >= 20:
            pct_20d = float((_usdjpy.iloc[-1] / _usdjpy.iloc[-20] - 1) * 100)
            if pct_20d > 1:
                fx_insight += f" <b>Actionable: USDJPY rose {pct_20d:+.1f}% over 20 days. Yen weakening accelerating; carry trades look exposed if BOJ tightens.</b>"
            elif pct_20d < -1:
                fx_insight += f" <b>Actionable: USDJPY fell {pct_20d:+.1f}% over 20 days. Yen strengthening suggests carry unwind or safe-haven flows.</b>"
        _section_note(
            "USDJPY (rising = weaker Yen), EURJPY, Nikkei. Simultaneous Yen weakness + equity drop = foreign outflows."
            + fx_insight
        )
        fig2 = go.Figure()
        for col in mkt_cols:
            fig2.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))
        _add_boj_events(fig2)
        _chart(_style_fig(fig2, 420))
        # Takeaway
        if len(_usdjpy) > 0:
            _fx_last = float(_usdjpy.iloc[-1])
            _takeaway_block(
                f"USDJPY at <b>{_fx_last:.1f}</b>. "
                f"{'Above 150: yen is historically weak. BOJ is under political pressure to tighten, which would push JGB yields higher.' if _fx_last > 150 else 'Below 140: yen is strengthening, reducing BOJ urgency to normalise. JGB repricing may stall.' if _fx_last < 140 else 'In the 140-150 range: balanced. FX is not forcing the BOJ hand in either direction.'}"
            )
    else:
        st.info("No market columns found in data.")

    # --- Bond ETF Flows Proxy ---
    st.subheader("Bond ETF Flow Proxy")
    _definition_block(
        "Why Bond ETF Flows Matter for JGBs",
        "An <b>ETF</b> (Exchange-Traded Fund) is a fund you can buy on the stock exchange that holds a basket "
        "of assets. Bond ETFs hold government or corporate bonds, and their price movements tell us what big "
        "institutional investors are doing with their fixed-income portfolios. "
        "<b>TLT</b> holds US Treasury bonds maturing in 20+ years (long-duration, very sensitive to rate changes). "
        "<b>IEF</b> holds 7-10 year Treasuries (medium duration). <b>SHY</b> holds 1-3 year Treasuries (short "
        "duration, barely moves when rates change). <b>BNDX</b> holds international bonds outside the US. "
        "When TLT drops sharply while SHY stays flat, it means investors are selling long-dated bonds and hiding "
        "in short-dated ones. This is called <em>reducing duration</em>, and it signals that the market expects "
        "interest rates to keep rising. Since Japan has the longest-duration government bond market in the world, "
        "global duration reduction directly threatens JGB prices. "
        "<b>How to read this chart:</b> Lines going down = bond prices falling = yields rising. "
        "TLT dropping much more than SHY = the market is bracing for higher rates."
    )
    etf_cols = [c for c in ["TLT", "IEF", "SHY", "BNDX"] if c in df.columns]
    if len(etf_cols) >= 2:
        etf_data = df[etf_cols].dropna()
        if len(etf_data) > 1:
            etf_returns = (etf_data / etf_data.iloc[0] - 1) * 100
            _section_note(
                "Cumulative returns (%) from base date. TLT/IEF divergence signals duration positioning shifts. "
                "Broad selloff across all ETFs indicates global rate repricing."
            )
            fig_etf = go.Figure()
            etf_colors = {"TLT": "#1565C0", "IEF": "#4CAF50", "SHY": "#FF9800", "BNDX": "#9C27B0"}
            etf_labels = {"TLT": "TLT (20+Y UST)", "IEF": "IEF (7-10Y UST)", "SHY": "SHY (1-3Y UST)", "BNDX": "BNDX (Intl Bond)"}
            for col in etf_cols:
                fig_etf.add_trace(go.Scatter(
                    x=etf_returns.index, y=etf_returns[col],
                    mode="lines", name=etf_labels.get(col, col),
                    line=dict(color=etf_colors.get(col)),
                ))
            fig_etf.update_layout(yaxis_title="Cumulative Return (%)")
            _add_boj_events(fig_etf)
            _chart(_style_fig(fig_etf, 380))
            etf_latest = etf_returns.iloc[-1]
            if "TLT" in etf_latest and "SHY" in etf_latest:
                duration_gap = float(etf_latest["TLT"] - etf_latest["SHY"])
                _takeaway_block(
                    f"TLT vs SHY spread: <b>{duration_gap:+.1f}pp</b>. "
                    f"{'Long-duration bonds sharply underperforming short-duration: global rate repricing is active and JGBs are exposed.' if duration_gap < -10 else 'Duration spread is moderate; no extreme positioning signal.' if abs(duration_gap) < 10 else 'Long-duration outperformance signals flight-to-safety; JGBs may benefit temporarily.'}"
                )
    else:
        st.info("Bond ETF data not available.")

    # --- Global Equity Benchmarks ---
    st.subheader("Global Equity Benchmarks")
    _definition_block(
        "Why Global Equities Matter for JGBs",
        "Stock markets around the world tend to move together because global investors allocate money "
        "across regions based on risk appetite. This chart compares the Nikkei (Japan) against four "
        "other major benchmarks: S&P 500 (US), Euro Stoxx 50 (Europe), FTSE 100 (UK), and ASX 200 (Australia). "
        "If ALL markets fall together, it is a <em>global risk-off event</em> where investors sell stocks "
        "everywhere and buy safe-haven bonds, which is actually GOOD for JGB prices (yields fall). "
        "But if ONLY Japan's stock market falls while others hold up, it signals Japan-specific problems "
        "like policy uncertainty or capital outflows, which is BAD for JGBs because it means investors are "
        "leaving Japan entirely. Conversely, if Japan outperforms the world, foreign money is flowing IN, "
        "but those investors often hedge their yen exposure, which can indirectly push JGB yields higher. "
        "<b>How to read this chart:</b> All lines start at 0%. Compare the Nikkei (red) against others. "
        "If it moves differently from the pack, something Japan-specific is driving it."
    )
    global_eq_cols = [c for c in ["NIKKEI", "SPX", "EUROSTOXX", "FTSE", "ASX200"] if c in df.columns]
    if len(global_eq_cols) >= 2:
        global_eq = df[global_eq_cols].dropna()
        if len(global_eq) > 1:
            global_returns = (global_eq / global_eq.iloc[0] - 1) * 100
            _section_note(
                "Cumulative returns (%) normalized to 0%. Nikkei divergence from global peers signals Japan-specific dynamics."
            )
            fig_global = go.Figure()
            geq_colors = {"NIKKEI": "#E8413C", "SPX": "#1565C0", "EUROSTOXX": "#4CAF50", "FTSE": "#FF9800", "ASX200": "#9C27B0"}
            geq_labels = {"NIKKEI": "\U0001F1EF\U0001F1F5 Nikkei 225 (Tokyo)", "SPX": "\U0001F1FA\U0001F1F8 S&P 500 (New York)", "EUROSTOXX": "\U0001F1EA\U0001F1FA Euro Stoxx 50 (EU)", "FTSE": "\U0001F1EC\U0001F1E7 FTSE 100 (London)", "ASX200": "\U0001F1E6\U0001F1FA ASX 200 (Sydney)"}
            for col in global_eq_cols:
                fig_global.add_trace(go.Scatter(
                    x=global_returns.index, y=global_returns[col],
                    mode="lines", name=geq_labels.get(col, col),
                    line=dict(color=geq_colors.get(col)),
                ))
            fig_global.update_layout(yaxis_title="Cumulative Return (%)")
            _add_boj_events(fig_global)
            _chart(_style_fig(fig_global, 380))
            gl = global_returns.iloc[-1]
            nk_ret = float(gl.get("NIKKEI", 0))
            spx_ret = float(gl.get("SPX", 0)) if "SPX" in gl else None
            if spx_ret is not None:
                _takeaway_block(
                    f"Nikkei at <b>{nk_ret:+.1f}%</b> vs S&P 500 at <b>{spx_ret:+.1f}%</b>. "
                    f"{'Nikkei outperformance suggests foreign inflows into Japan, potentially hedging JPY exposure and adding pressure on JGB yields.' if nk_ret > spx_ret + 5 else 'Nikkei underperformance relative to US equities may reflect BOJ tightening fears or structural concerns.' if nk_ret < spx_ret - 5 else 'Broadly aligned performance; equity markets are not signaling Japan-specific stress.'}"
                )
    else:
        st.info("Insufficient global equity data.")

    # --- Rolling JP-US Yield Correlation ---
    st.subheader("Rolling JP-US Yield Correlation")
    _definition_block(
        "Why JP-US Yield Correlation Matters",
        "<b>Correlation</b> measures whether two things move together. A correlation of +1.0 means they move "
        "in perfect lockstep. A correlation of 0 means they move independently. A correlation of -1.0 means they "
        "move in opposite directions. This chart shows the <em>rolling 60-day correlation</em> between Japanese "
        "and US 10-year government bond yields. "
        "When correlation is high (above 0.6), it means Japan's bond market is just following the US. "
        "If US yields rise, Japanese yields rise too. In this regime, you mainly need to watch the US Federal "
        "Reserve to predict JGB movements. "
        "When correlation drops (below 0.2 or goes negative), it means Japan is marching to its own drum. "
        "This usually happens when the BOJ makes a surprise policy move or domestic Japanese inflation data "
        "diverges from the US. These decorrelation episodes are critical for traders because models that assume "
        "JP and US rates move together will suddenly stop working. "
        "<b>How to read this chart:</b> Line near +1 = markets are connected. Line near 0 or negative = "
        "Japan-specific forces are in control. Sudden drops from high to low often coincide with BOJ surprises "
        "(marked by red verticals)."
    )
    _jp10 = df["JP_10Y"].dropna() if "JP_10Y" in df.columns else pd.Series(dtype=float)
    _us10 = df["US_10Y"].dropna() if "US_10Y" in df.columns else pd.Series(dtype=float)
    if len(_jp10) > 60 and len(_us10) > 60:
        _aligned = pd.DataFrame({"JP_10Y": _jp10, "US_10Y": _us10}).dropna()
        if len(_aligned) > 60:
            _rolling_corr = _aligned["JP_10Y"].rolling(60).corr(_aligned["US_10Y"])
            _section_note(
                "60-day rolling correlation between JP 10Y and US 10Y yields. "
                "High correlation = global rate forces dominate. Low/negative = Japan-specific dynamics."
            )
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(
                x=_rolling_corr.index, y=_rolling_corr,
                mode="lines", name="60d Rolling Correlation",
                line=dict(color="#1565C0"),
            ))
            fig_corr.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)
            fig_corr.update_layout(yaxis_title="Correlation", yaxis_range=[-1, 1])
            _add_boj_events(fig_corr)
            _chart(_style_fig(fig_corr, 340))
            _latest_corr = float(_rolling_corr.dropna().iloc[-1])
            _takeaway_block(
                f"Current JP-US 10Y correlation: <b>{_latest_corr:.2f}</b>. "
                f"{'Yields are highly correlated: global rate moves are driving JGBs. US Treasury selloffs will spill directly into JGBs.' if _latest_corr > 0.6 else 'Correlation has broken down: Japan-specific forces are dominating. BOJ policy and domestic factors matter more than US rates.' if _latest_corr < 0.2 else 'Moderate correlation: both global and local factors are at play.'}"
            )
    else:
        st.info("Insufficient yield data for correlation analysis.")

    # --- Raw data expander ---
    with st.expander("Raw data (last 20 rows)"):
        st.dataframe(df.tail(20))

    # --- Sovereign Credit & Trust Metrics ---
    st.subheader("Japan Sovereign Credit Context")
    _definition_block(
        "What are Sovereign Credit Ratings?",
        "Just like individuals have credit scores, countries have <b>credit ratings</b>. Agencies like Moody's, "
        "S&P, and Fitch grade each country's ability to repay its debts. Ratings range from AAA (best, like "
        "Germany or Singapore) down to D (default). Japan is rated around A/A+ which is 'good but not great'. "
        "Why not higher? Japan has the world's highest <b>debt-to-GDP ratio</b> at ~260%, meaning the government "
        "owes 2.6x its entire annual economic output. However, Japan has a unique advantage: over 90% of JGBs are "
        "held by domestic investors (Japanese banks, pension funds, the BOJ itself), and Japan is a net creditor "
        "to the rest of the world. This means Japan is unlikely to face a sudden foreign exodus like emerging markets. "
        "A <b>downgrade</b> (rating cut) or <b>negative outlook</b> change matters because many large institutional "
        "investors have rules that force them to sell bonds below a certain rating. Even a one-notch downgrade can "
        "trigger billions in forced selling, spiking yields overnight. "
        "<b>BOJ credibility events</b> (table below) are policy decisions that shocked the market. When the central "
        "bank repeatedly surprises investors, it erodes trust in its forward guidance, making yields more volatile."
    )
    from src.data.config import JAPAN_CREDIT_RATINGS, BOJ_CREDIBILITY_EVENTS
    _section_note(
        "Credit ratings provide structural context for JGB repricing risk. Japan's A/A+ rating "
        "reflects high debt-to-GDP offset by its net external creditor position and domestic savings base. "
        "<b>Actionable: Rating downgrades or outlook changes can accelerate repricing by forcing institutional rebalancing.</b>"
    )
    cr_cols = st.columns(len(JAPAN_CREDIT_RATINGS))
    for cr_col, (agency, info) in zip(cr_cols, JAPAN_CREDIT_RATINGS.items()):
        cr_col.metric(
            f"{agency}",
            info["rating"],
            delta=f"Outlook: {info['outlook']}",
            help=f"{info['note']} ({info['last_action']})"
        )

    _section_note(
        "BOJ credibility events: policy decisions that surprised markets (>2 std dev moves). "
        "A pattern of hawkish surprises erodes forward guidance credibility and amplifies repricing."
    )
    cred_df = pd.DataFrame(BOJ_CREDIBILITY_EVENTS)
    cred_df = cred_df.rename(columns={"date": "Date", "direction": "Direction", "impact_bps": "Impact (bps)", "description": "Description"})
    st.dataframe(cred_df, use_container_width=True, hide_index=True)

    # --- Page conclusion ---
    _src_label = "simulated" if _get_args()[0] else "live (FRED + yfinance)"
    # Verdict
    _jp = df["JP_10Y"].dropna() if "JP_10Y" in df.columns else pd.Series(dtype=float)
    _us = df["US_10Y"].dropna() if "US_10Y" in df.columns else pd.Series(dtype=float)
    if len(_jp) > 0 and len(_us) > 0:
        _spread_bps = (float(_jp.iloc[-1]) - float(_us.iloc[-1])) * 100
        _verdict_p1 = f"JGB 10Y trades {abs(_spread_bps):.0f} bps {'below' if _spread_bps < 0 else 'above'} the US benchmark. {'Gap remains wide; repricing has room to run.' if _spread_bps < -100 else 'Spread is narrowing; convergence trade is maturing.' if _spread_bps < 0 else 'JGBs have overshot; watch for mean-reversion.'}"
    else:
        _verdict_p1 = "Data pipeline operational. Review yield and FX series above before proceeding."
    _page_conclusion(
        _verdict_p1,
        f"Dataset loaded with <b>{len(df):,}</b> observations across "
        f"<b>{df.shape[1]}</b> variables from {_src_label} sources, spanning "
        f"<b>{df.index.min():%b %Y}</b> to <b>{df.index.max():%b %Y}</b>. "
        f"Proceed to Yield Curve Analytics to decompose these raw series into interpretable factors.",
    )
    _page_footer()


