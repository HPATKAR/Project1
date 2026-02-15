"""
JGB Repricing Framework: Streamlit Dashboard

Multi-page dashboard visualising the full JGB repricing pipeline:
data overview, yield curve analytics, regime detection, cross-asset
spillover, and trade ideas.

Launch:  .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import sys
import os
import base64
from pathlib import Path
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
# src imports (all lazy, only called within their respective pages)
# ---------------------------------------------------------------------------
from src.data.data_store import DataStore
from src.data.config import BOJ_EVENTS, JGB_TENORS, DEFAULT_START, DEFAULT_END, ANALYSIS_WINDOWS

# ---------------------------------------------------------------------------
# Global Streamlit config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="JGB Repricing Framework",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/Purdue_Boilermakers_logo.svg/1200px-Purdue_Boilermakers_logo.svg.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Institutional CSS — Bloomberg / JPM Research aesthetic
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        /* Purdue Daniels School of Business palette */
        --ink:        #000000;          /* Purdue Black */
        --ink-soft:   #555960;          /* Steel */
        --ink-muted:  #6F727B;          /* Cool Gray */
        --ink-faint:  #9D9795;          /* Railway Gray */
        --surface:    #ffffff;
        --surface-1:  #f9f8f6;          /* warm white */
        --surface-2:  #f0eeeb;          /* warm gray */
        --border:     #C4BFC0;          /* Steam */
        --border-light:#e8e5e2;
        --accent:     #000000;          /* Purdue Black */
        --accent-mid: #8E6F3E;          /* Aged */
        --accent-pop: #CFB991;          /* Boilermaker Gold */
        --gold:       #CFB991;          /* Boilermaker Gold */
        --gold-bright:#DAAA00;          /* Rush */
        --gold-field: #DDB945;          /* Field */
        --gold-dust:  #EBD99F;          /* Dust */
        --red:        #c0392b;
        --green:      #2e7d32;
        --amber:      #8E6F3E;          /* Aged as amber */
    }

    /* ---- Reset & Global ---- */
    html, body, .main, [data-testid="stAppViewContainer"] {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        color: var(--ink-soft);
        text-rendering: optimizeLegibility;
    }
    .main .block-container {
        padding: 2.2rem 3rem 0 3rem;
        max-width: 1360px;
        padding-bottom: 0 !important;
    }
    .main {
        padding-bottom: 0 !important;
        margin-bottom: 0 !important;
        overflow: hidden;
    }
    [data-testid="stAppViewContainer"] > section > div {
        padding-bottom: 0 !important;
    }
    /* Smooth transitions on interactive elements */
    button, a, [data-testid="stMetric"], details[data-testid="stExpander"] {
        transition: all 0.18s ease;
    }

    /* ---- Typography ---- */
    .main h1 {
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        color: var(--ink);
        font-size: 1.25rem;
        letter-spacing: -0.025em;
        border-bottom: none;
        padding-bottom: 0;
        margin-bottom: 0.15rem;
    }
    .main h1::after {
        content: '';
        display: block;
        width: 48px;
        height: 3px;
        background: linear-gradient(90deg, var(--gold) 0%, var(--gold-bright) 100%);
        margin-top: 0.5rem;
        margin-bottom: 1.2rem;
        border-radius: 2px;
    }
    .main h2 {
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        color: var(--ink);
        font-size: 0.82rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        border-bottom: none;
        padding-bottom: 0;
        margin-top: 2.2rem;
        margin-bottom: 0.4rem;
    }
    .main h2::before {
        content: '';
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 2px;
        background: var(--gold);
        margin-right: 8px;
        transform: translateY(-1px);
    }
    .main h3 {
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        color: var(--ink-soft);
        font-size: 0.8rem;
        margin-top: 1.2rem;
        margin-bottom: 0.25rem;
    }
    .main p, .main li {
        color: var(--ink-soft);
        line-height: 1.6;
        font-size: 0.8rem;
    }

    /* ---- Metric cards ---- */
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border-light);
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 0 0 1px rgba(0,0,0,0.02);
    }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.06), 0 0 0 1px rgba(207,185,145,0.2);
        border-color: var(--gold-dust);
    }
    [data-testid="stMetricLabel"] {
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.58rem;
        letter-spacing: 0.1em;
        color: var(--ink-muted);
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', 'SF Mono', monospace;
        font-weight: 500;
        color: var(--ink);
        font-size: 1.05rem;
        letter-spacing: -0.01em;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: #000000;
        border-right: 1px solid rgba(207,185,145,0.1);
    }
    /* Text & labels */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {
        color: rgba(255,255,255,0.7) !important;
        font-family: 'DM Sans', sans-serif;
    }
    section[data-testid="stSidebar"] label {
        font-size: 0.65rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.4) !important;
        margin-bottom: 2px;
    }
    section[data-testid="stSidebar"] b,
    section[data-testid="stSidebar"] strong {
        color: rgba(255,255,255,0.85) !important;
        font-weight: 600;
    }
    /* Inputs, selects, date pickers */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] select,
    section[data-testid="stSidebar"] [data-baseweb="input"] input,
    section[data-testid="stSidebar"] [data-baseweb="select"] div,
    section[data-testid="stSidebar"] [data-baseweb="input"],
    section[data-testid="stSidebar"] [data-baseweb="base-input"] {
        background: rgba(255,255,255,0.06) !important;
        color: rgba(255,255,255,0.9) !important;
        border-color: rgba(255,255,255,0.12) !important;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
    }
    section[data-testid="stSidebar"] input:focus,
    section[data-testid="stSidebar"] [data-baseweb="input"]:focus-within {
        border-color: rgba(207,185,145,0.5) !important;
        box-shadow: 0 0 0 2px rgba(207,185,145,0.15);
    }
    /* Date input */
    section[data-testid="stSidebar"] [data-testid="stDateInput"] input {
        background: rgba(255,255,255,0.06) !important;
        color: rgba(255,255,255,0.85) !important;
        border-color: rgba(255,255,255,0.12) !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
    }
    /* Select / dropdown */
    section[data-testid="stSidebar"] [data-baseweb="select"] {
        background: rgba(255,255,255,0.06);
        border-radius: 6px;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p {
        color: rgba(255,255,255,0.85) !important;
        font-size: 0.76rem;
    }
    /* Toggle */
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: rgba(255,255,255,0.6) !important;
        font-size: 0.74rem;
    }
    /* Nav buttons */
    section[data-testid="stSidebar"] .stButton > button {
        text-align: left;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.78rem;
        border-radius: 6px;
        padding: 0.5rem 0.85rem;
        transition: all 0.15s ease;
        font-weight: 500;
        letter-spacing: 0.01em;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background: transparent;
        border: 1px solid rgba(255,255,255,0.08);
        color: rgba(255,255,255,0.55) !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        background: rgba(255,255,255,0.05);
        border-color: rgba(255,255,255,0.18);
        color: rgba(255,255,255,0.9) !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: rgba(207,185,145,0.12);
        border: 1px solid rgba(207,185,145,0.3);
        border-left: 3px solid #CFB991;
        color: #CFB991 !important;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.06);
        margin: 0.8rem 0;
    }
    /* Multiselect / slider chips in sidebar */
    section[data-testid="stSidebar"] [data-baseweb="tag"] {
        background: rgba(207,185,145,0.15) !important;
        color: #CFB991 !important;
        border: none;
    }
    /* ---- Expander ---- */
    .streamlit-expanderHeader {
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 0.8rem;
        color: var(--ink);
    }
    details[data-testid="stExpander"] {
        border: 1px solid var(--border-light);
        border-radius: 10px;
        margin-bottom: 0.6rem;
        background: var(--surface);
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    details[data-testid="stExpander"]:hover {
        border-color: var(--gold-dust);
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    details[data-testid="stExpander"][open] {
        border-color: rgba(207,185,145,0.3);
    }

    /* ---- Chart containers ---- */
    [data-testid="stPlotlyChart"] {
        border: 1px solid var(--border-light);
        border-radius: 10px;
        background: var(--surface);
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }

    /* ---- Dataframes ---- */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-light);
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }

    /* ---- Divider ---- */
    hr {
        border: none;
        border-top: 1px solid var(--border-light);
        margin: 1.2rem 0;
    }

    /* ---- Download button ---- */
    .stDownloadButton > button {
        font-family: 'DM Sans', sans-serif;
        background: #000000;
        color: #CFB991;
        border: 1px solid #CFB991;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.55rem 1.6rem;
        font-size: 0.74rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        transition: all 0.18s ease;
    }
    .stDownloadButton > button:hover {
        background: #CFB991;
        color: #000000;
        box-shadow: 0 4px 16px rgba(207,185,145,0.35);
        transform: translateY(-1px);
    }
    .stDownloadButton > button:active {
        transform: translateY(0);
        box-shadow: 0 1px 4px rgba(207,185,145,0.2);
    }

    /* ---- Alerts ---- */
    [data-testid="stAlert"] {
        border-radius: 10px;
        font-size: 0.8rem;
        border: none;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid var(--border-light);
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        padding: 0.6rem 1.2rem;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 2px solid var(--gold);
        color: var(--ink);
    }

    /* ---- Chat (AI Q&A page) ---- */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        border: 1px solid var(--border-light);
        padding: 1rem 1.2rem;
        font-size: 0.82rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        margin-bottom: 0.5rem;
    }
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: var(--surface-1);
    }
    .stChatInput textarea {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.82rem;
        border-radius: 12px;
        border: 1px solid var(--border-light);
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .stChatInput textarea:focus {
        border-color: rgba(207,185,145,0.5);
        box-shadow: 0 2px 12px rgba(207,185,145,0.1);
    }

    /* ---- Spinner ---- */
    .stSpinner > div {
        font-size: 0.78rem;
        color: var(--ink-muted);
        font-weight: 500;
    }

    /* ---- Focus ring (accessibility + polish) ---- */
    button:focus-visible,
    input:focus-visible,
    textarea:focus-visible,
    select:focus-visible {
        outline: none;
        box-shadow: 0 0 0 3px rgba(207,185,145,0.35);
    }

    /* ---- Scrollbar ---- */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.12); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(0,0,0,0.2); }

    /* ---- Selection highlight ---- */
    ::selection { background: rgba(207,185,145,0.25); color: var(--ink); }

    /* ---- Tabular numerics for finance-grade numbers ---- */
    * { font-variant-numeric: tabular-nums lining-nums; }

    /* ---- Dataframe table polish ---- */
    [data-testid="stDataFrame"] [role="columnheader"] {
        background: #f6f6f4 !important;
        color: var(--ink) !important;
        border-bottom: 1px solid var(--border) !important;
        font-weight: 700;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    [data-testid="stDataFrame"] [role="row"]:hover {
        background: rgba(207,185,145,0.08);
    }
    [data-testid="stDataFrame"] * {
        font-size: 0.78rem;
    }

    /* ---- Hide ALL Streamlit chrome ---- */
    #MainMenu { display: none !important; }
    footer { display: none !important; }
    header[data-testid="stHeader"] { display: none !important; }
    div[data-testid="stToolbar"] { display: none !important; }
    div[data-testid="stDecoration"] { display: none !important; }
    div[data-testid="stStatusWidget"] { display: none !important; }

    /* ---- Sidebar collapse: kill EVERY variant across Streamlit versions ---- */
    button[data-testid="stSidebarCollapseButton"] { display: none !important; }
    div[data-testid="stSidebarCollapseButton"] { display: none !important; }
    header [data-testid="stSidebarCollapseButton"] { display: none !important; }
    div[data-testid="collapsedControl"] { display: none !important; }
    button[kind="header"] { display: none !important; }
    button[kind="headerNoPadding"] { display: none !important; }
    [data-testid="stSidebar"] > div:first-child > button { display: none !important; }

    /* ---- Column gap polish ---- */
    [data-testid="column"] { padding: 0 0.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: navigation + global controls
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    "<div style='padding:1rem 0 1.2rem 0; border-bottom:1px solid rgba(207,185,145,0.15); "
    "margin-bottom:1rem;'>"
    "<div style='font-size:0.5rem; font-weight:700; text-transform:uppercase; "
    "letter-spacing:0.2em; color:#CFB991; margin-bottom:6px;'>Rates Strategy Desk</div>"
    "<div style='font-size:1.3rem; font-weight:800; color:rgba(255,255,255,0.95); "
    "letter-spacing:-0.03em; line-height:1.15; "
    "font-family:DM Sans,sans-serif;'>JGB Repricing</div>"
    "<div style='font-size:0.64rem; font-weight:400; color:rgba(255,255,255,0.3); "
    "margin-top:4px; letter-spacing:0.02em;'>Quantitative Framework</div>"
    "<div style='font-size:0.58rem; font-weight:500; color:rgba(207,185,145,0.5); "
    "margin-top:2px; letter-spacing:0.04em;'>Purdue Daniels School of Business</div></div>",
    unsafe_allow_html=True,
)

# Button-based navigation with session state
_QP_MAP = {
    "about_heramb": "About: Heramb Patkar",
    "about_zhang": "About: Dr. Zhang",
    "overview": "Overview & Data",
    "yield_curve": "Yield Curve Analytics",
    "regime": "Regime Detection",
    "spillover": "Spillover & Info Flow",
    "trades": "Trade Ideas",
    "ai_qa": "AI Q&A",
}
_qp = st.query_params.get("page", "")
if _qp in _QP_MAP:
    st.session_state.current_page = _QP_MAP[_qp]
    # Clear the query param so it doesn't override sidebar button clicks on rerun
    st.query_params.clear()
elif "current_page" not in st.session_state:
    st.session_state.current_page = "Overview & Data"

_NAV_ITEMS = [
    ("Overview & Data", "overview"),
    ("Yield Curve Analytics", "yield_curve"),
    ("Regime Detection", "regime"),
    ("Spillover & Info Flow", "spillover"),
    ("Trade Ideas", "trades"),
    ("AI Q&A", "ai_qa"),
]

for _label, _key in _NAV_ITEMS:
    _active = st.session_state.current_page == _label
    if st.sidebar.button(
        _label,
        key=f"nav_{_key}",
        use_container_width=True,
        type="primary" if _active else "secondary",
    ):
        st.session_state.current_page = _label
        st.rerun()

page = st.session_state.current_page

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='font-size:0.55rem;font-weight:600;text-transform:uppercase;"
    "letter-spacing:0.14em;color:rgba(255,255,255,0.3);margin:0 0 0.5rem 0;"
    "font-family:DM Sans,sans-serif;'>Configuration</p>",
    unsafe_allow_html=True,
)

use_simulated = st.sidebar.toggle("Simulated data", value=False)
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
# Plotly template (institutional palette)
# ===================================================================
_PALETTE = [
    "#000000",  # Purdue Black
    "#CFB991",  # Boilermaker Gold
    "#8E6F3E",  # Aged
    "#c0392b",  # red (for contrast)
    "#2e7d32",  # green (for contrast)
    "#555960",  # Steel
    "#DAAA00",  # Rush
    "#6F727B",  # Cool Gray
]

_PLOTLY_LAYOUT = dict(
    font=dict(family="DM Sans, -apple-system, sans-serif", size=11, color="#555960"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    title_text="",
    title_font=dict(size=11, color="#0b0f19", family="DM Sans, sans-serif"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        font=dict(size=10, color="#6F727B", family="DM Sans, sans-serif"),
        bgcolor="rgba(0,0,0,0)",
        itemsizing="constant",
    ),
    margin=dict(l=48, r=16, t=28, b=36),
    hovermode="x unified",
    hoverlabel=dict(
        font_size=11,
        font_family="DM Sans, sans-serif",
        bgcolor="rgba(0,0,0,0.92)",
        font_color="#ffffff",
        bordercolor="rgba(0,0,0,0)",
    ),
    xaxis=dict(
        gridcolor="rgba(0,0,0,0.04)",
        linecolor="rgba(0,0,0,0.08)",
        zerolinecolor="rgba(0,0,0,0.06)",
        tickfont=dict(size=10, color="#9D9795", family="JetBrains Mono, monospace"),
        showgrid=True,
        gridwidth=1,
    ),
    yaxis=dict(
        gridcolor="rgba(0,0,0,0.04)",
        linecolor="rgba(0,0,0,0.08)",
        zerolinecolor="rgba(0,0,0,0.06)",
        tickfont=dict(size=10, color="#9D9795", family="JetBrains Mono, monospace"),
        showgrid=True,
        gridwidth=1,
    ),
)


def _style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    """Apply the institutional plotly template."""
    fig.update_layout(**_PLOTLY_LAYOUT, height=height)
    for i, trace in enumerate(fig.data):
        if isinstance(trace, go.Scatter):
            has_color = getattr(trace.line, "color", None) or getattr(trace.marker, "color", None)
            if not has_color:
                fig.data[i].update(
                    line=dict(color=_PALETTE[i % len(_PALETTE)], width=1.5),
                )
    return fig


def _page_intro(text: str):
    """Render a page introduction."""
    st.markdown(
        f"<p style='color:#6b7394;font-family:DM Sans,sans-serif;font-size:0.8rem;"
        f"line-height:1.7;margin:0 0 1.5rem 0;padding:0;max-width:740px;"
        f"letter-spacing:0.005em;'>{text}</p>",
        unsafe_allow_html=True,
    )


def _section_note(text: str):
    """Render analytical context below a section header."""
    st.markdown(
        f"<div style='background:#fafaf8;border-left:3px solid #CFB991;padding:10px 16px;"
        f"border-radius:0 8px 8px 0;margin:-0.1rem 0 0.8rem 0;"
        f"box-shadow:0 1px 3px rgba(0,0,0,0.02);'>"
        f"<p style='color:#555960;font-size:0.76rem;line-height:1.65;margin:0;"
        f"font-family:DM Sans,sans-serif;'>{text}</p></div>",
        unsafe_allow_html=True,
    )


def _page_conclusion(verdict: str, summary: str):
    """Render verdict + assessment panel."""
    st.markdown(
        f"<div style='margin-top:2.5rem;border-radius:12px;overflow:hidden;"
        f"border:1px solid rgba(0,0,0,0.06);box-shadow:0 2px 12px rgba(0,0,0,0.06);'>"
        # verdict block
        f"<div style='background:#000000;padding:18px 24px;"
        f"border-top:3px solid #CFB991;'>"
        f"<p style='margin:0 0 2px 0;color:rgba(207,185,145,0.6);font-family:DM Sans,sans-serif;"
        f"font-size:0.55rem;font-weight:700;text-transform:uppercase;letter-spacing:0.16em;'>"
        f"Verdict</p>"
        f"<p style='margin:0;color:#CFB991;font-family:DM Sans,sans-serif;"
        f"font-size:0.88rem;font-weight:600;line-height:1.55;letter-spacing:-0.01em;'>"
        f"{verdict}</p></div>"
        # assessment block
        f"<div style='background:#fafaf8;padding:16px 24px;'>"
        f"<p style='margin:0 0 6px 0;color:#9D9795;font-family:DM Sans,sans-serif;"
        f"font-size:0.58rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;'>"
        f"Assessment</p>"
        f"<p style='margin:0;color:#555960;font-family:DM Sans,sans-serif;"
        f"font-size:0.78rem;line-height:1.7;'>{summary}</p></div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _page_footer():
    """Render full-bleed institutional footer with Daniels School branding."""
    yr = datetime.now().year
    ts = datetime.now().strftime("%B %d, %Y at %H:%M UTC")
    _w = "color:rgba(255,255,255,0.75);text-decoration:none;font-size:0.72rem;font-weight:500;transition:color 0.15s ease;"
    _g = "font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:0.16em;color:#CFB991;margin:0 0 14px 0;padding-bottom:8px;border-bottom:1px solid rgba(207,185,145,0.15);"
    st.markdown(
        "<style>"
        ".main .block-container { padding-bottom: 0 !important; margin-bottom: 0 !important; }"
        ".main { padding-bottom: 0 !important; margin-bottom: 0 !important; }"
        "[data-testid='stAppViewContainer'] { padding-bottom: 0 !important; }"
        "[data-testid='stBottom'] { display: none !important; }"
        "</style>"
        "<div style='margin-top:4rem;font-family:DM Sans,sans-serif;"
        "position:relative;width:100vw;left:50%;right:50%;margin-left:-50vw;margin-right:-50vw;"
        "margin-bottom:-10rem;padding-bottom:0;'>"
        # ── main black section ──
        "<div style='background:#000000;padding:44px 0 40px 0;'>"
        "<div style='display:grid;grid-template-columns:1.6fr 1fr 1fr 1fr 1fr;gap:28px;max-width:1280px;margin:0 auto;padding:0 48px;'>"
        # col 1: branding
        "<div>"
        "<a href='https://business.purdue.edu/' target='_blank'>"
        "<img src='https://business.purdue.edu/includes/img/medsb_h-full-reverse-rgb_1.png' "
        "alt='Purdue Daniels School of Business' "
        "style='height:40px;margin-bottom:16px;display:block;' /></a>"
        "<p style='font-size:0.68rem;color:rgba(255,255,255,0.7);line-height:1.65;margin:0 0 16px 0;max-width:260px;'>"
        "MGMT 69000 &middot; Mastering AI for Finance<br/>"
        "West Lafayette, Indiana</p>"
        # timestamp placed formally under the branding block
        f"<p style='font-size:0.6rem;color:rgba(207,185,145,0.6);margin:0;"
        f"font-weight:600;letter-spacing:0.04em;'>"
        f"Last updated {ts}</p>"
        "</div>"
        # col 2: navigate
        f"<div><p style='{_g}'>Navigate</p>"
        "<ul style='list-style:none;padding:0;margin:0;'>"
        f"<li style='margin-bottom:8px;'><a href='?page=overview' target='_self' style='{_w}'>Overview &amp; Data</a></li>"
        f"<li style='margin-bottom:8px;'><a href='?page=yield_curve' target='_self' style='{_w}'>Yield Curve Analytics</a></li>"
        f"<li style='margin-bottom:8px;'><a href='?page=regime' target='_self' style='{_w}'>Regime Detection</a></li>"
        f"<li style='margin-bottom:8px;'><a href='?page=spillover' target='_self' style='{_w}'>Spillover &amp; Info Flow</a></li>"
        f"<li style='margin-bottom:8px;'><a href='?page=trades' target='_self' style='{_w}'>Trade Ideas</a></li>"
        f"<li style='margin-bottom:8px;'><a href='?page=ai_qa' target='_self' style='{_w}'>AI Q&amp;A</a></li>"
        "</ul></div>"
        # col 3: about
        f"<div><p style='{_g}'>About</p>"
        "<ul style='list-style:none;padding:0;margin:0;'>"
        f"<li style='margin-bottom:8px;'><a href='?page=about_heramb' target='_self' style='{_w}'>Heramb S. Patkar</a></li>"
        f"<li style='margin-bottom:8px;'><a href='?page=about_zhang' target='_self' style='{_w}'>Dr. Cinder Zhang</a></li>"
        f"<li><a href='https://business.purdue.edu/' target='_blank' style='{_w}'>Daniels School of Business</a></li>"
        "</ul></div>"
        # col 4: connect
        f"<div><p style='{_g}'>Connect</p>"
        "<ul style='list-style:none;padding:0;margin:0;'>"
        f"<li style='margin-bottom:8px;'><a href='https://www.linkedin.com/in/heramb-patkar/' target='_blank' style='{_w}'>LinkedIn: Heramb S. Patkar</a></li>"
        f"<li><a href='https://www.linkedin.com/in/cinder-zhang/' target='_blank' style='{_w}'>LinkedIn: Dr. Cinder Zhang</a></li>"
        "</ul></div>"
        # col 5: source code
        f"<div><p style='{_g}'>Source Code</p>"
        "<ul style='list-style:none;padding:0;margin:0;'>"
        f"<li style='margin-bottom:8px;'><a href='https://github.com/HPATKAR' target='_blank' style='{_w}'>GitHub: Heramb S. Patkar</a></li>"
        f"<li style='margin-bottom:8px;'><a href='https://github.com/CinderZhang' target='_blank' style='{_w}'>GitHub: Dr. Cinder Zhang</a></li>"
        f"<li><a href='https://cinderzhang.github.io/' target='_blank' style='{_w}'>DRIVER Framework</a></li>"
        "</ul></div>"
        "</div></div>"
        # ── gold accent bar: copyright only ──
        "<div style='background:#CFB991;padding:10px 48px;text-align:center;'>"
        f"<p style='font-size:0.58rem;color:#000000;margin:0;font-weight:600;letter-spacing:0.04em;"
        f"font-family:DM Sans,sans-serif;'>"
        f"&copy; {yr} Purdue University &middot; For educational purposes only &middot; Not investment advice</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )


# ===================================================================
# Cached helpers
# ===================================================================
@st.cache_resource(ttl=3600, max_entries=2)
def get_data_store(simulated: bool) -> DataStore:
    return DataStore(use_simulated=simulated)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
def load_unified(simulated: bool, start: str, end: str, api_key: str | None):
    store = get_data_store(simulated)
    return store.get_unified(start=start, end=end, fred_api_key=api_key or None)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
def load_rates(simulated: bool, start: str, end: str, api_key: str | None):
    store = get_data_store(simulated)
    return store.get_rates(start=start, end=end, fred_api_key=api_key or None)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
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
# Page 1: Overview & Data
# ===================================================================
def page_overview():
    st.header("Overview & Data")
    _page_intro(
        "Raw market data underlying all downstream analytics. "
        "Toggle data source and date range via sidebar controls."
    )

    with st.spinner("Loading market data..."):
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
        st.plotly_chart(_style_fig(fig, 420), use_container_width=True)
    else:
        st.info("No rate columns found in data.")

    # --- Market chart ---
    st.subheader("FX & Equity")
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
        st.plotly_chart(_style_fig(fig2, 420), use_container_width=True)
    else:
        st.info("No market columns found in data.")

    # --- Raw data expander ---
    with st.expander("Raw data (last 20 rows)"):
        st.dataframe(df.tail(20), use_container_width=True)

    # --- Sovereign Credit & Trust Metrics ---
    st.subheader("Japan Sovereign Credit Context")
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
    _src_label = "simulated" if use_simulated else "live (FRED + yfinance)"
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


# ===================================================================
# Page 2: Yield Curve Analytics
# ===================================================================
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
        "Yield curve decomposition via PCA, Roll liquidity measure, and Nelson-Siegel parametric fit."
    )

    args = (use_simulated, str(start_date), str(end_date), fred_api_key or None)

    # Pre-compute all models in a single pass
    with st.spinner("Computing yield curve analytics..."):
        pca_result = _run_pca(*args)
        liq = _run_liquidity(*args)
        ns_result = _run_ns(*args)

    # --- PCA ---
    st.subheader("PCA of Yield Changes")
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
            st.plotly_chart(_style_fig(fig_ev, 320), use_container_width=True)

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
            st.plotly_chart(_style_fig(fig_ld, 320), use_container_width=True)

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
            st.plotly_chart(_style_fig(fig_pcl, 340), use_container_width=True)

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
        st.plotly_chart(_style_fig(fig_sc, 380), use_container_width=True)

    # --- Liquidity ---
    st.subheader("Liquidity Metrics")
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
        st.plotly_chart(_style_fig(fig_liq, 380), use_container_width=True)

    # --- Nelson-Siegel ---
    st.subheader("Nelson-Siegel Factor Evolution")
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
        st.plotly_chart(_style_fig(fig_ns, 380), use_container_width=True)

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


# ===================================================================
# Page 3: Regime Detection
# ===================================================================
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
        "Four independent regime detection models (Markov, HMM, entropy, GARCH) combined into an ensemble probability."
    )

    args = (use_simulated, str(start_date), str(end_date), fred_api_key or None)

    # Pre-compute all regime models in a single pass
    with st.spinner("Running regime detection models..."):
        ensemble = _run_ensemble(*args)   # internally runs markov, hmm, entropy, garch
        markov = _run_markov(*args)       # cached from ensemble
        changes, bkps = _run_breaks(*args)
        ent, sig = _run_entropy(*args)    # cached from ensemble
        vol, garch_breaks = _run_garch(*args)  # cached from ensemble

    # --- Ensemble Probability ---
    st.subheader("Ensemble Regime Probability")
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
        st.plotly_chart(_style_fig(fig_ens, 380), use_container_width=True)
    else:
        st.warning("Could not compute ensemble probability. Check data availability.")

    # --- Markov Smoothed Probabilities ---
    st.subheader("Markov-Switching Smoothed Probabilities")
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
        st.plotly_chart(_style_fig(fig_mk, 350), use_container_width=True)

        st.caption(
            f"Regime means: {markov['regime_means']}, "
            f"Regime variances: {markov['regime_variances']}"
        )
    else:
        st.warning("Insufficient data for Markov regime model.")

    # --- Structural Breaks ---
    st.subheader("Structural Breakpoints on JP 10Y Changes")
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
        st.plotly_chart(_style_fig(fig_bp, 350), use_container_width=True)
    else:
        st.warning("Insufficient data for structural break detection.")

    # --- Entropy ---
    st.subheader("Rolling Permutation Entropy & Regime Signal")
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
        st.plotly_chart(_style_fig(fig_ent, 350), use_container_width=True)
    else:
        st.warning("Insufficient data for entropy analysis.")

    # --- GARCH ---
    st.subheader("GARCH Conditional Volatility & Vol-Regime Breaks")
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
        st.plotly_chart(_style_fig(fig_g, 350), use_container_width=True)
    else:
        st.warning("Insufficient data for GARCH model.")

    # --- Regime Comparison Table by BOJ Era ---
    st.subheader("Regime Comparison by BOJ Policy Era")
    try:
        df_full = load_unified(use_simulated, str(start_date), str(end_date), fred_api_key or None)
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


# ===================================================================
# Page 4: Spillover & Information Flow
# ===================================================================
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
        "Cross-market transmission analysis via Granger causality, transfer entropy, Diebold-Yilmaz spillover, and DCC-GARCH."
    )

    args = (use_simulated, str(start_date), str(end_date), fred_api_key or None)

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
        st.plotly_chart(_style_fig(fig_te, 450), use_container_width=True)
    else:
        st.warning("Insufficient data for transfer entropy.")

    # --- Transfer Entropy on PCA Factor Scores ---
    st.subheader("Transfer Entropy on PCA Factors")
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
        st.plotly_chart(_style_fig(fig_te_pca, 350), use_container_width=True)
    else:
        st.info("Insufficient PCA data for factor-level transfer entropy.")

    # --- Diebold-Yilmaz ---
    st.subheader("Diebold-Yilmaz Spillover")
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
            st.plotly_chart(_style_fig(fig_net, 320), use_container_width=True)

        with st.expander("Spillover matrix"):
            st.dataframe(spill["spillover_matrix"].round(2), use_container_width=True)
    else:
        st.warning("Insufficient data for spillover analysis.")

    # --- DCC ---
    st.subheader("DCC Time-Varying Correlations")
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
            st.plotly_chart(_style_fig(fig_dcc, 380), use_container_width=True)
        else:
            st.info("No correlation pairs computed.")
    else:
        st.warning("Insufficient data for DCC-GARCH.")

    # --- FX Carry ---
    st.subheader("FX Carry Metrics (USD/JPY)")
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
        st.plotly_chart(_style_fig(fig_c, 380), use_container_width=True)
    else:
        st.warning("Insufficient data for carry analytics.")

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


# ===================================================================
# Page 5: Trade Ideas
# ===================================================================
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


def page_trade_ideas():
    st.header("Trade Ideas")
    _page_intro(
        "Rule-based trade generation from regime state, curve analytics, and cross-asset signals. "
        "Filter by category and conviction via sidebar."
    )

    args = (use_simulated, str(start_date), str(end_date), fred_api_key or None)

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
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<p style='font-size:0.55rem;font-weight:600;text-transform:uppercase;"
        "letter-spacing:0.14em;color:rgba(255,255,255,0.3);margin:0 0 0.5rem 0;"
        "font-family:DM Sans,sans-serif;'>Trade Filters</p>",
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
        st.plotly_chart(_style_fig(fig_conv, 350), use_container_width=True)

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
                f"border-radius:20px;font-weight:600;font-size:0.68rem;letter-spacing:0.06em;"
                f"font-family:DM Sans,sans-serif;'>{direction_tag}</span>"
                f"<span style='background:{conv_color};color:#fff;padding:3px 10px;"
                f"border-radius:20px;font-weight:600;font-size:0.68rem;"
                f"font-family:JetBrains Mono,monospace;'>{card.conviction:.0%}</span>"
                f"<span style='background:#f7f8fb;color:#3b4259;padding:3px 12px;"
                f"border-radius:20px;font-size:0.7rem;font-weight:500;"
                f"border:1px solid #dfe2ec;font-family:DM Sans,sans-serif;'>{card.category}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown(
                    f"<div style='font-size:0.8rem;line-height:1.85;color:#3b4259;"
                    f"font-family:DM Sans,sans-serif;'>"
                    f"<b style='color:#0b0f19;'>Instruments:</b> {', '.join(card.instruments)}<br>"
                    f"<b style='color:#0b0f19;'>Regime Condition:</b> {card.regime_condition}<br>"
                    f"<b style='color:#0b0f19;'>Edge Source:</b> {card.edge_source}<br>"
                    f"<b style='color:#0b0f19;'>Entry Signal:</b> {card.entry_signal}</div>",
                    unsafe_allow_html=True,
                )
            with col_r:
                st.markdown(
                    f"<div style='font-size:0.8rem;line-height:1.85;color:#3b4259;"
                    f"font-family:DM Sans,sans-serif;'>"
                    f"<b style='color:#0b0f19;'>Exit Signal:</b> {card.exit_signal}<br>"
                    f"<b style='color:#0b0f19;'>Sizing:</b> {card.sizing_method}<br>"
                    f"<b style='color:#dc2626;font-weight:600;'>Failure Scenario:</b> "
                    f"<span style='color:#6b7394;'>{card.failure_scenario}</span></div>",
                    unsafe_allow_html=True,
                )

    # --- Export ---
    if filtered:
        df_export = trade_cards_to_dataframe(filtered)
        csv = df_export.to_csv(index=False)
        st.download_button(
            "Download Trade Cards as CSV",
            data=csv,
            file_name="jgb_trade_cards.csv",
            mime="text/csv",
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


# ===================================================================
# Page 6: AI Q&A
# ===================================================================
def _build_analysis_context(args):
    """Serialize current analysis outputs into a text context for the LLM."""
    parts = []

    # Regime state
    try:
        ensemble = _run_ensemble(*args)
        if ensemble is not None and len(ensemble.dropna()) > 0:
            prob = float(ensemble.dropna().iloc[-1])
            regime = "REPRICING" if prob > 0.5 else "SUPPRESSED"
            parts.append(f"REGIME: Ensemble probability = {prob:.2%} ({regime}). Avg over sample = {ensemble.mean():.2%}.")
    except Exception:
        pass

    # PCA
    try:
        pca_res = _run_pca(*args)
        if pca_res is not None:
            ev = pca_res["explained_variance_ratio"]
            parts.append(f"PCA: PC1 explains {ev[0]:.1%}, PC2 {ev[1]:.1%}, PC3 {ev[2]:.1%} of yield variance (cumulative {sum(ev):.1%}).")
    except Exception:
        pass

    # Spillover
    try:
        spill = _run_spillover(*args)
        if spill is not None:
            net = spill["net_spillover"]
            top_t = net.idxmax()
            parts.append(f"SPILLOVER: Total = {spill['total_spillover']:.1f}%. Net transmitter = {top_t}.")
    except Exception:
        pass

    # Carry
    try:
        carry = _run_carry(*args)
        if carry is not None and len(carry["carry_to_vol"].dropna()) > 0:
            ctv = float(carry["carry_to_vol"].dropna().iloc[-1])
            parts.append(f"FX CARRY: Carry-to-vol ratio = {ctv:.2f}. {'Attractive' if ctv > 1 else 'Marginal' if ctv > 0.5 else 'Unattractive'}.")
    except Exception:
        pass

    # Liquidity
    try:
        liq = _run_liquidity(*args)
        if liq is not None and len(liq["composite_index"].dropna()) > 0:
            liq_v = float(liq["composite_index"].dropna().iloc[-1])
            parts.append(f"LIQUIDITY: Composite index = {liq_v:+.2f} z-score. {'Stressed' if liq_v < -1 else 'Healthy' if liq_v > 0 else 'Neutral'}.")
    except Exception:
        pass

    # Latest data snapshot
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
    except Exception:
        pass

    # Trade ideas summary
    try:
        cards, rs = _generate_trades(*args)
        if cards:
            top_3 = sorted(cards, key=lambda c: -c.conviction)[:3]
            trade_lines = [f"  - {c.name} ({c.direction.upper()}, {c.conviction:.0%} conviction, {c.category})" for c in top_3]
            parts.append(f"TOP TRADE IDEAS:\n" + "\n".join(trade_lines))
    except Exception:
        pass

    # BOJ events
    from src.data.config import BOJ_EVENTS as _boj_events
    parts.append("BOJ POLICY DATES: " + "; ".join(f"{d}: {e}" for d, e in _boj_events.items()))

    return "\n\n".join(parts)


def page_ai_qa():
    # Chat-optimised layout — re-show stBottom (hidden by footer on other pages)
    st.markdown(
        "<style>"
        "[data-testid='stBottom'] { display: block !important; }"
        ".main .block-container { padding-bottom: 80px !important; }"
        "</style>",
        unsafe_allow_html=True,
    )
    st.header("AI Q&A")
    _page_intro(
        "Ask questions about the JGB repricing analysis. The AI assistant has access to all model outputs, "
        "regime state, and trade ideas computed in this session. Powered by Claude (Anthropic) or OpenAI."
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

    # --- Build context once ---
    args = (use_simulated, str(start_date), str(end_date), fred_api_key or None)

    if not ai_api_key:
        # Show a professional empty state instead of just a bare info box
        st.markdown(
            "<div style='text-align:center;padding:3rem 2rem;'>"
            "<div style='font-size:2.5rem;margin-bottom:0.8rem;opacity:0.15;'>&#x1F4AC;</div>"
            "<p style='font-family:DM Sans,sans-serif;font-size:0.88rem;font-weight:600;"
            "color:#000;margin:0 0 6px 0;'>AI Research Assistant</p>"
            "<p style='font-family:DM Sans,sans-serif;font-size:0.76rem;color:#6b7394;"
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
            "padding:8px 16px;font-size:0.72rem;color:#555960;font-family:DM Sans,sans-serif;'>"
            "What is the current regime state?</span>"
            "<span style='background:#fafaf8;border:1px solid #eceae6;border-radius:8px;"
            "padding:8px 16px;font-size:0.72rem;color:#555960;font-family:DM Sans,sans-serif;'>"
            "Explain the PCA decomposition</span>"
            "<span style='background:#fafaf8;border:1px solid #eceae6;border-radius:8px;"
            "padding:8px 16px;font-size:0.72rem;color:#555960;font-family:DM Sans,sans-serif;'>"
            "Best trade ideas right now?</span>"
            "<span style='background:#fafaf8;border:1px solid #eceae6;border-radius:8px;"
            "padding:8px 16px;font-size:0.72rem;color:#555960;font-family:DM Sans,sans-serif;'>"
            "BOJ policy outlook</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        # Show live context snapshot as proof the system is working
        with st.expander("Live Analysis Context (preview of what the AI sees)"):
            ctx = _build_analysis_context(args)
            st.code(ctx, language="text")
        _page_footer()
        return

    # --- Chat state ---
    if "qa_messages" not in st.session_state:
        st.session_state.qa_messages = []

    # Show empty state with suggestions when no messages yet
    if not st.session_state.qa_messages:
        st.markdown(
            "<div style='text-align:center;padding:2rem 2rem 1rem 2rem;'>"
            "<p style='font-family:DM Sans,sans-serif;font-size:0.84rem;font-weight:600;"
            "color:#000;margin:0 0 4px 0;'>Ready to assist</p>"
            "<p style='font-family:DM Sans,sans-serif;font-size:0.74rem;color:#6b7394;"
            "margin:0 0 1.2rem 0;'>Ask about JGBs, rates, macro, BOJ policy, "
            "trading strategies, or yield curves.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        # Clickable topic cards
        _topics = [
            ("What is the current regime probability and what does it mean?", "Regime"),
            ("Break down the PCA yield curve factors and explain the variance", "PCA"),
            ("What are the top trade ideas and their conviction levels?", "Trades"),
            ("How are spillovers flowing between JGBs, USTs, and FX?", "Spillover"),
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
                    st.session_state.qa_messages.append({"role": "user", "content": _q})
                    st.rerun()

    # Display chat history
    for msg in st.session_state.qa_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Chat input ---
    user_input = st.chat_input("Ask anything about JGBs, rates, macro, BOJ policy, trading strategies...")

    if user_input:
        st.session_state.qa_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build context
        analysis_context = _build_analysis_context(args)

        system_prompt = (
            "You are a senior rates strategist at a major investment bank, specialising in Japanese Government Bonds (JGBs), "
            "global fixed income, macro trading, and quantitative finance. "
            "You have access to live analysis outputs from a JGB repricing framework (below), but you can also "
            "answer broader questions about bond markets, monetary policy, BOJ, Fed, yield curve theory, "
            "trading strategies, financial concepts, and anything related to rates and macro. "
            "When relevant, ground your answers in the live data and cite specific numbers. "
            "Be concise, professional, and actionable.\n\n"
            f"=== LIVE ANALYSIS CONTEXT ===\n{analysis_context}\n=== END CONTEXT ==="
        )

        # Build message history for the API
        api_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.qa_messages]

        with st.chat_message("assistant"):
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

    # Context transparency in sidebar
    with st.sidebar.expander("Analysis Context"):
        ctx = _build_analysis_context(args)
        st.text(ctx)


# ===================================================================
# Page 7: About — Heramb Patkar
# ===================================================================
def _about_page_styles():
    """Inject shared CSS for About pages."""
    st.markdown(
        "<style>"
        # ── hero banner ──
        ".about-hero{position:relative;width:100vw;left:50%;right:50%;margin-left:-50vw;"
        "margin-right:-50vw;background:linear-gradient(168deg,#000 0%,#0a0a0a 60%,#111 100%);"
        "padding:64px 0 56px 0;margin-top:-1rem;margin-bottom:2.5rem;"
        "border-bottom:3px solid #CFB991;}"
        ".about-hero-inner{max-width:960px;margin:0 auto;padding:0 40px;}"
        ".about-hero .overline{font-size:0.58rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.22em;color:#CFB991;margin:0 0 12px 0;"
        "display:inline-block;padding:4px 14px;border:1px solid rgba(207,185,145,0.25);"
        "border-radius:4px;background:rgba(207,185,145,0.06);}"
        ".about-hero h1{font-size:2rem;font-weight:800;color:#fff;margin:0 0 10px 0;"
        "letter-spacing:-0.03em;line-height:1.12;}"
        ".about-hero .subtitle{font-size:0.78rem;color:#CFB991;font-weight:600;"
        "margin:0 0 12px 0;letter-spacing:0.01em;}"
        ".about-hero .tagline{font-size:0.78rem;color:rgba(255,255,255,0.6);font-weight:400;"
        "margin:0 0 24px 0;line-height:1.6;max-width:640px;}"
        ".about-hero .links{display:flex;gap:10px;flex-wrap:wrap;}"
        ".about-hero .links a{display:inline-flex;align-items:center;gap:6px;"
        "padding:8px 20px;border:1px solid rgba(207,185,145,0.4);"
        "border-radius:6px;color:#CFB991;font-size:0.68rem;font-weight:600;text-decoration:none;"
        "transition:all 0.2s ease;letter-spacing:0.02em;}"
        ".about-hero .links a:hover{background:#CFB991;color:#000;border-color:#CFB991;"
        "transform:translateY(-1px);box-shadow:0 4px 16px rgba(207,185,145,0.25);}"
        # ── cards ──
        ".about-card{background:#fff;border:1px solid #e8e6e1;border-radius:12px;"
        "padding:28px 28px 24px 28px;margin-bottom:18px;"
        "box-shadow:0 1px 3px rgba(0,0,0,0.04);transition:box-shadow 0.2s ease;}"
        ".about-card:hover{box-shadow:0 4px 20px rgba(0,0,0,0.06);}"
        ".about-card-title{font-size:0.6rem;font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.15em;color:#8E6F3E;margin:0 0 18px 0;padding-bottom:12px;"
        "border-bottom:2px solid #f0eeeb;display:flex;align-items:center;gap:8px;}"
        ".about-card-title::before{content:'';display:inline-block;width:3px;height:14px;"
        "background:#CFB991;border-radius:2px;}"
        # ── experience timeline ──
        ".exp-item{position:relative;margin-bottom:20px;padding-bottom:20px;padding-left:16px;"
        "border-left:2px solid #f0eeeb;border-bottom:none;}"
        ".exp-item:last-child{margin-bottom:0;padding-bottom:0;}"
        ".exp-item::before{content:'';position:absolute;left:-5px;top:4px;width:8px;height:8px;"
        "border-radius:50%;background:#CFB991;border:2px solid #fff;}"
        ".exp-role{font-size:0.76rem;font-weight:700;color:#000;margin:0 0 2px 0;line-height:1.35;}"
        ".exp-org{font-size:0.7rem;font-weight:600;color:#8E6F3E;margin:0 0 4px 0;}"
        ".exp-meta{font-size:0.6rem;color:#9D9795;margin:0 0 8px 0;font-weight:500;"
        "letter-spacing:0.01em;}"
        ".exp-desc{font-size:0.68rem;color:#555960;line-height:1.7;margin:0;}"
        # ── education ──
        ".edu-item{margin-bottom:16px;padding-bottom:16px;border-bottom:1px solid #f5f3f0;}"
        ".edu-item:last-child{margin-bottom:0;padding-bottom:0;border-bottom:none;}"
        ".edu-school{font-size:0.74rem;font-weight:700;color:#000;margin:0 0 2px 0;}"
        ".edu-dept{font-size:0.66rem;color:#8E6F3E;margin:0 0 2px 0;font-weight:600;}"
        ".edu-degree{font-size:0.66rem;color:#555960;margin:0 0 2px 0;font-weight:500;}"
        ".edu-year{font-size:0.58rem;color:#9D9795;margin:0;font-weight:500;}"
        # ── certifications ──
        ".cert-item{margin-bottom:12px;padding-bottom:12px;border-bottom:1px solid #f5f3f0;}"
        ".cert-item:last-child{margin-bottom:0;padding-bottom:0;border-bottom:none;}"
        ".cert-name{font-size:0.7rem;font-weight:600;color:#000;margin:0 0 2px 0;}"
        ".cert-issuer{font-size:0.6rem;color:#9D9795;margin:0;font-weight:500;}"
        # ── publication ──
        ".pub-item{margin-bottom:16px;padding:16px 18px;background:#fafaf8;"
        "border-left:3px solid #CFB991;border-radius:0 8px 8px 0;}"
        ".pub-item:last-child{margin-bottom:0;}"
        ".pub-title{font-size:0.74rem;font-weight:700;color:#000;margin:0 0 6px 0;line-height:1.4;}"
        ".pub-authors{font-size:0.64rem;color:#555960;margin:0 0 6px 0;line-height:1.5;}"
        ".pub-authors strong{color:#000;font-weight:700;}"
        ".pub-journal{font-size:0.62rem;color:#8E6F3E;font-weight:600;margin:0 0 4px 0;"
        "font-style:italic;}"
        ".pub-detail{font-size:0.6rem;color:#9D9795;margin:0 0 8px 0;line-height:1.5;}"
        ".pub-link{display:inline-block;padding:4px 12px;border:1px solid rgba(207,185,145,0.4);"
        "border-radius:4px;color:#8E6F3E;font-size:0.6rem;font-weight:600;text-decoration:none;"
        "transition:all 0.2s;}"
        ".pub-link:hover{background:#CFB991;color:#000;border-color:#CFB991;}"
        # ── tags / pills ──
        ".interest-tag{display:inline-block;padding:6px 16px;border-radius:20px;font-size:0.62rem;"
        "font-weight:600;margin:4px 5px 4px 0;transition:transform 0.15s ease;}"
        ".interest-tag:hover{transform:translateY(-1px);}"
        ".interest-gold{background:rgba(207,185,145,0.12);color:#8E6F3E;"
        "border:1px solid rgba(207,185,145,0.2);}"
        ".interest-neutral{background:rgba(0,0,0,0.03);color:#555960;"
        "border:1px solid rgba(0,0,0,0.06);}"
        # ── acknowledgment ──
        ".ack-text{font-size:0.68rem;color:#555960;line-height:1.75;margin:0;}"
        ".ack-text strong{color:#000;}"
        # ── stat row ──
        ".stat-row{display:flex;gap:16px;margin:16px 0 4px 0;}"
        ".stat-item{flex:1;text-align:center;padding:12px 8px;background:#fafaf8;"
        "border-radius:8px;border:1px solid #f0eeeb;}"
        ".stat-num{font-size:1.1rem;font-weight:800;color:#000;margin:0;line-height:1.2;}"
        ".stat-label{font-size:0.54rem;font-weight:600;color:#9D9795;margin:2px 0 0 0;"
        "text-transform:uppercase;letter-spacing:0.08em;}"
        "</style>",
        unsafe_allow_html=True,
    )


def page_about_heramb():
    _about_page_styles()
    _f = "font-family:'DM Sans',sans-serif;"

    # ── load profile image as base64 ──
    _img_path = Path(__file__).parent / "FinDis.jpeg"
    _img_b64 = ""
    if _img_path.exists():
        _img_b64 = base64.b64encode(_img_path.read_bytes()).decode()

    _photo_html = ""
    if _img_b64:
        _photo_html = (
            "<div style='flex-shrink:0;'>"
            f"<img src='data:image/jpeg;base64,{_img_b64}' "
            "alt='Heramb S. Patkar' style='width:160px;height:160px;border-radius:50%;"
            "object-fit:cover;border:3px solid rgba(207,185,145,0.3);"
            "box-shadow:0 10px 36px rgba(0,0,0,0.4);' />"
            "</div>"
        )

    # ── hero banner ──
    st.markdown(
        "<div class='about-hero'><div class='about-hero-inner' "
        "style='display:flex;align-items:center;gap:36px;'>"
        f"{_photo_html}"
        "<div style='flex:1;min-width:0;'>"
        "<p class='overline'>About the Author</p>"
        "<h1>Heramb S. Patkar</h1>"
        "<p class='subtitle'>MSF Candidate, Purdue Daniels School of Business</p>"
        "<p class='tagline'>BITS Pilani engineering graduate with equity research experience "
        "across Indian and U.S. capital markets. NISM XV certified research analyst. "
        "Published in the Bulletin of Materials Science (IAS, 2025).</p>"
        "<div class='links'>"
        "<a href='https://www.linkedin.com/in/heramb-patkar/' target='_blank'>LinkedIn</a>"
        "<a href='https://github.com/HPATKAR' target='_blank'>GitHub</a>"
        "<a href='https://www.ias.ac.in/article/fulltext/boms/048/0028' target='_blank'>Publication</a>"
        "</div></div>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    # ── two-column body ──
    col_main, col_side = st.columns([1.55, 1], gap="large")

    with col_main:
        # bio
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Profile</p>"
            f"<p style='{_f}color:#555960;font-size:0.76rem;line-height:1.75;margin:0 0 10px 0;'>"
            "Driven by curiosity about how businesses create impact and grow stronger. "
            "With a background in engineering and experience in global equity research, "
            "I enjoy analysing industries, building financial models, and uncovering insights "
            "that drive smarter decisions. Excited by opportunities where analytical thinking "
            "and creativity intersect to solve complex problems and deliver meaningful value.</p>"
            f"<p style='{_f}color:#555960;font-size:0.76rem;line-height:1.75;margin:0;'>"
            "Beyond work, I enjoy exploring new places, listening to Carnatic music, "
            "and learning from different cultures and perspectives. Always open to connecting"
            ", feel free to reach out.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        # project
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>This Project</p>"
            f"<p style='{_f}color:#555960;font-size:0.76rem;line-height:1.75;margin:0 0 12px 0;'>"
            "Built this JGB Repricing Framework as Course Project 1 for Prof. Xinde Zhang's "
            "MGMT 69000-119: a quantitative dashboard that detects regime shifts in Japanese "
            "Government Bond markets and generates institutional-grade trade ideas.</p>"
            "<div class='stat-row'>"
            "<div class='stat-item'><p class='stat-num'>4</p><p class='stat-label'>Regime Models</p></div>"
            "<div class='stat-item'><p class='stat-num'>6</p><p class='stat-label'>Tenors</p></div>"
            "<div class='stat-item'><p class='stat-num'>8</p><p class='stat-label'>Dashboard Pages</p></div>"
            "<div class='stat-item'><p class='stat-num'>5</p><p class='stat-label'>Analytical Layers</p></div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        # experience
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Experience</p>"

            "<div class='exp-item'>"
            "<p class='exp-role'>Practicum Analyst</p>"
            "<p class='exp-org'>Fino Advisors LLC</p>"
            "<p class='exp-meta'>Jan 2026 &ndash; Present &middot; Houston, TX (Remote)</p>"
            "<p class='exp-desc'>Build and update a Series A financial model with revenue "
            "assumptions and simple scenarios. Conduct valuation and comparable company research "
            "and help prepare the investor deck and narrative.</p></div>"

            "<div class='exp-item'>"
            "<p class='exp-role'>Equity Research Associate</p>"
            "<p class='exp-org'>Axis Direct</p>"
            "<p class='exp-meta'>Sep 2024 &ndash; Apr 2025 &middot; Mumbai, India &middot; Full-time</p>"
            "<p class='exp-desc'>Collaborated with the lead equity research analyst on Auto and "
            "Auto Ancillary sector coverage across three quarters. Built and maintained detailed "
            "cash flow / PE models with forecasts for 14 listed names (7 OEMs, 7 ancillaries). "
            "Co-authored IPO notes (Hyundai Motor India, Ather Energy), earnings updates, "
            "and industry volume outlooks. Built a comprehensive Indian auto and farming equipment "
            "industry tracker integrating data from FADA, SIAM, and company filings. Converted "
            "internship into a full-time role.</p></div>"

            "<div class='exp-item'>"
            "<p class='exp-role'>Equity Research Intern</p>"
            "<p class='exp-org'>Axis Direct</p>"
            "<p class='exp-meta'>Jul 2024 &ndash; Aug 2024 &middot; Mumbai, India</p>"
            "<p class='exp-desc'>Supported the lead analyst in Pharma and Hospitality sectors "
            "through industry analysis, financial modelling, and co-authoring research reports. "
            "Co-authored two initiating coverage reports on Chalet Hotels and Juniper.</p></div>"

            "<div class='exp-item'>"
            "<p class='exp-role'>Undergraduate Research Assistant</p>"
            "<p class='exp-org'>BITS Pilani</p>"
            "<p class='exp-meta'>Apr 2022 &ndash; May 2024 &middot; Hyderabad, India</p>"
            "<p class='exp-desc'>Co-designed and validated a low-cost stereotaxic device for "
            "rodent brain research. Work published in the Bulletin of Materials Science "
            "(Indian Academy of Sciences, 2025).</p></div>"

            "<div class='exp-item'>"
            "<p class='exp-role'>Manufacturing Engineer Intern</p>"
            "<p class='exp-org'>Divgi TorqTransfer Systems Ltd</p>"
            "<p class='exp-meta'>Jul 2023 &ndash; Dec 2023 &middot; Sirsi, Karnataka, India</p></div>"

            "<div class='exp-item'>"
            "<p class='exp-role'>HVAC Engineer Intern</p>"
            "<p class='exp-org'>Grasim Industries Limited, Pulp &amp; Fibre</p>"
            "<p class='exp-meta'>May 2022 &ndash; Jul 2022 &middot; Nagda, Madhya Pradesh, India</p></div>"

            "</div>",
            unsafe_allow_html=True,
        )

    with col_side:
        # education
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Education</p>"
            "<div class='edu-item'>"
            "<p class='edu-school'>Purdue University</p>"
            "<p class='edu-dept'>Mitchell E. Daniels, Jr. School of Business</p>"
            "<p class='edu-degree'>Master of Science in Finance</p>"
            "<p class='edu-year'>2025 &ndash; 2026</p></div>"
            "<div class='edu-item'>"
            "<p class='edu-school'>BITS Pilani</p>"
            "<p class='edu-dept'>Hyderabad Campus</p>"
            "<p class='edu-degree'>B.E. (Hons.) Mechanical Engineering</p>"
            "<p class='edu-year'>2020 &ndash; 2024</p></div>"
            "</div>",
            unsafe_allow_html=True,
        )

        # publication — compact, not a full card
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Publication</p>"
            "<div class='pub-item'>"
            "<p class='pub-title'>Design, Fabrication and Validation of a Low-Cost "
            "Stereotaxic Device for Brain Research in Rodents</p>"
            "<p class='pub-authors'>A. Wadkar, <strong>H. Patkar</strong>, "
            "S.P. Kommajosyula</p>"
            "<p class='pub-journal'>Bulletin of Materials Science, Vol. 48, "
            "Article 0028</p>"
            "<p class='pub-detail'>Indian Academy of Sciences &middot; February 2025</p>"
            "<a class='pub-link' href='https://www.ias.ac.in/article/fulltext/boms/048/0028' "
            "target='_blank'>View Full Text</a>"
            "</div></div>",
            unsafe_allow_html=True,
        )

        # certifications
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Licenses &amp; Certifications</p>"
            "<div class='cert-item'>"
            "<p class='cert-name'>NISM Series XV: Research Analyst</p>"
            "<p class='cert-issuer'>NISM &middot; Oct 2024 &ndash; Oct 2027</p></div>"
            "<div class='cert-item'>"
            "<p class='cert-name'>Bloomberg Market Concepts</p>"
            "<p class='cert-issuer'>Bloomberg LP &middot; Feb 2024</p></div>"
            "</div>",
            unsafe_allow_html=True,
        )

        # interests
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Interests</p>"
            "<div>"
            "<span class='interest-tag interest-gold'>Investment Banking</span>"
            "<span class='interest-tag interest-neutral'>Corporate Finance</span>"
            "<span class='interest-tag interest-gold'>Valuations</span>"
            "<span class='interest-tag interest-neutral'>Private Equity</span>"
            "<span class='interest-tag interest-gold'>Equity Research</span>"
            "</div></div>",
            unsafe_allow_html=True,
        )

        # acknowledgments
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Acknowledgments</p>"
            "<p class='ack-text'><strong>Prof. Xinde Zhang</strong>, MGMT 69000: "
            "Framework-first thinking behind regime and spillover design</p>"
            "<p class='ack-text' style='margin-top:8px;'><strong>Prof. Adem Atmaz</strong>, "
            "MGMT 511: Fixed income intuition behind PCA decomposition and "
            "Nelson-Siegel approach</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    _page_footer()


# ===================================================================
# Page 8: About — Dr. Cinder Zhang
# ===================================================================
def page_about_zhang():
    _about_page_styles()
    _f = "font-family:'DM Sans',sans-serif;"

    # ── hero banner ──
    st.markdown(
        "<div class='about-hero'><div class='about-hero-inner'>"
        "<p class='overline'>Course Instructor</p>"
        "<h1>Dr. Xinde (Cinder) Zhang, Ph.D.</h1>"
        "<p class='subtitle'>Finance Faculty, Mitchell E. Daniels, Jr. School of Business</p>"
        "<p class='tagline'>Creator of the DRIVER Framework. Award-winning educator "
        "pioneering AI-integrated finance pedagogy at Purdue University.</p>"
        "<div class='links'>"
        "<a href='https://www.linkedin.com/in/cinder-zhang/' target='_blank'>LinkedIn</a>"
        "<a href='https://github.com/CinderZhang' target='_blank'>GitHub</a>"
        "<a href='https://cinderzhang.github.io/' target='_blank'>DRIVER Framework</a>"
        "</div></div></div>",
        unsafe_allow_html=True,
    )

    # ── two-column body ──
    col_main, col_side = st.columns([1.55, 1], gap="large")

    with col_main:
        # bio
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Profile</p>"
            f"<p style='{_f}color:#555960;font-size:0.76rem;line-height:1.75;margin:0 0 10px 0;'>"
            "Dr. Cinder Zhang is an award-winning finance professor and pioneer in AI-integrated "
            "finance education at Purdue University. He is the creator of the "
            "<strong>DRIVER Framework</strong> (Define &amp; Discover, Represent, Implement, "
            "Validate, Evolve, Reflect), a comprehensive methodology that transforms how financial "
            "management is taught by integrating AI as a cognitive amplifier rather than a "
            "replacement tool.</p>"
            f"<p style='{_f}color:#555960;font-size:0.76rem;line-height:1.75;margin:0;'>"
            "Dr. Zhang advocates shifting finance education from traditional analysis toward "
            "practical implementation, teaching students to <em>build financial tools and "
            "solutions</em> rather than compete with AI in analytical tasks. His approach "
            "emphasizes systematic problem-solving, data-driven decision making, and ethical "
            "AI collaboration.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        # impact with stats
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Impact</p>"
            f"<p style='{_f}color:#555960;font-size:0.76rem;line-height:1.75;margin:0 0 12px 0;'>"
            "As founder of the Financial Analytics program, Dr. Zhang has mentored "
            "graduates into roles at Goldman Sachs, JPMorgan, State Street, Mastercard, EY, "
            "Walmart Global Tech, and FinTech startups.</p>"
            "<div class='stat-row'>"
            "<div class='stat-item'><p class='stat-num'>130+</p><p class='stat-label'>Students</p></div>"
            "<div class='stat-item'><p class='stat-num'>3</p><p class='stat-label'>Textbooks</p></div>"
            "<div class='stat-item'><p class='stat-num'>2</p><p class='stat-label'>Teaching Awards</p></div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        # publications / textbooks
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Open-Source Textbooks</p>"

            "<div class='exp-item'>"
            "<p class='exp-role'>DRIVER: Financial Management</p>"
            "<p class='exp-desc'>Comprehensive financial management through the DRIVER lens, "
            "integrating AI tools throughout the learning process.</p></div>"

            "<div class='exp-item'>"
            "<p class='exp-role'>DRIVER: Financial Modeling</p>"
            "<p class='exp-desc'>Hands-on financial modeling with Python, Excel, and AI-assisted "
            "workflows for real-world applications.</p></div>"

            "<div class='exp-item'>"
            "<p class='exp-role'>DRIVER: Essentials of Investment</p>"
            "<p class='exp-desc'>Investment fundamentals reimagined with modern analytics "
            "and AI-driven research methodologies.</p></div>"

            "</div>",
            unsafe_allow_html=True,
        )

    with col_side:
        # education
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Education</p>"
            "<div class='edu-item'>"
            "<p class='edu-school'>University of North Carolina</p>"
            "<p class='edu-degree'>Ph.D., Finance</p></div>"
            "<div class='edu-item'>"
            "<p class='edu-school'>University of Missouri</p>"
            "<p class='edu-degree'>Doctoral Studies, MIS / Computer Science</p></div>"
            "<div class='edu-item'>"
            "<p class='edu-school'>Youngstown State University</p>"
            "<p class='edu-degree'>M.S., Mathematics</p></div>"
            "<div class='edu-item'>"
            "<p class='edu-school'>Jilin University</p>"
            "<p class='edu-degree'>B.S., Computer Science</p></div>"
            "</div>",
            unsafe_allow_html=True,
        )

        # DRIVER framework card
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>The DRIVER Framework</p>"
            f"<p style='{_f}color:#555960;font-size:0.68rem;line-height:1.7;margin:0 0 12px 0;'>"
            "A six-phase methodology for AI-integrated finance education:</p>"
            "<div style='display:flex;flex-direction:column;gap:6px;'>"
            "<div style='display:flex;align-items:center;gap:8px;'>"
            "<span style='width:22px;height:22px;border-radius:50%;background:#000;color:#CFB991;"
            "font-size:0.52rem;font-weight:800;display:flex;align-items:center;justify-content:center;"
            "flex-shrink:0;'>D</span>"
            f"<span style='{_f}font-size:0.66rem;color:#555960;'>Define &amp; Discover</span></div>"
            "<div style='display:flex;align-items:center;gap:8px;'>"
            "<span style='width:22px;height:22px;border-radius:50%;background:#000;color:#CFB991;"
            "font-size:0.52rem;font-weight:800;display:flex;align-items:center;justify-content:center;"
            "flex-shrink:0;'>R</span>"
            f"<span style='{_f}font-size:0.66rem;color:#555960;'>Represent</span></div>"
            "<div style='display:flex;align-items:center;gap:8px;'>"
            "<span style='width:22px;height:22px;border-radius:50%;background:#000;color:#CFB991;"
            "font-size:0.52rem;font-weight:800;display:flex;align-items:center;justify-content:center;"
            "flex-shrink:0;'>I</span>"
            f"<span style='{_f}font-size:0.66rem;color:#555960;'>Implement</span></div>"
            "<div style='display:flex;align-items:center;gap:8px;'>"
            "<span style='width:22px;height:22px;border-radius:50%;background:#000;color:#CFB991;"
            "font-size:0.52rem;font-weight:800;display:flex;align-items:center;justify-content:center;"
            "flex-shrink:0;'>V</span>"
            f"<span style='{_f}font-size:0.66rem;color:#555960;'>Validate</span></div>"
            "<div style='display:flex;align-items:center;gap:8px;'>"
            "<span style='width:22px;height:22px;border-radius:50%;background:#000;color:#CFB991;"
            "font-size:0.52rem;font-weight:800;display:flex;align-items:center;justify-content:center;"
            "flex-shrink:0;'>E</span>"
            f"<span style='{_f}font-size:0.66rem;color:#555960;'>Evolve</span></div>"
            "<div style='display:flex;align-items:center;gap:8px;'>"
            "<span style='width:22px;height:22px;border-radius:50%;background:#000;color:#CFB991;"
            "font-size:0.52rem;font-weight:800;display:flex;align-items:center;justify-content:center;"
            "flex-shrink:0;'>R</span>"
            f"<span style='{_f}font-size:0.66rem;color:#555960;'>Reflect</span></div>"
            "</div></div>",
            unsafe_allow_html=True,
        )

        # awards
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Awards &amp; Recognition</p>"
            "<div class='cert-item'>"
            "<p class='cert-name'>FMA Teaching Innovation Award</p>"
            "<p class='cert-issuer'>Financial Management Association</p></div>"
            "<div class='cert-item'>"
            "<p class='cert-name'>Teaching Innovation Award</p>"
            "<p class='cert-issuer'>University of Arkansas</p></div>"
            "</div>",
            unsafe_allow_html=True,
        )

        # focus areas
        st.markdown(
            "<div class='about-card'>"
            "<p class='about-card-title'>Focus Areas</p>"
            "<div>"
            "<span class='interest-tag interest-gold'>AI in Finance</span>"
            "<span class='interest-tag interest-neutral'>Financial Analytics</span>"
            "<span class='interest-tag interest-gold'>DRIVER Framework</span>"
            "<span class='interest-tag interest-neutral'>Python in Finance</span>"
            "<span class='interest-tag interest-gold'>Pedagogy</span>"
            "</div></div>",
            unsafe_allow_html=True,
        )

    _page_footer()


# ===================================================================
# Cache warming args
# ===================================================================
_args = (use_simulated, str(start_date), str(end_date), fred_api_key)


# ===================================================================
# Router
# ===================================================================
if page == "Overview & Data":
    page_overview()
elif page == "Yield Curve Analytics":
    page_yield_curve()
elif page == "Regime Detection":
    page_regime()
elif page == "Spillover & Info Flow":
    page_spillover()
elif page == "Trade Ideas":
    page_trade_ideas()
elif page == "AI Q&A":
    page_ai_qa()
elif page == "About: Heramb Patkar":
    page_about_heramb()
elif page == "About: Dr. Zhang":
    page_about_zhang()
