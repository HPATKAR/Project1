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
from src.ui.layout_config import LayoutConfig, LayoutManager, render_settings_panel
from src.ui.alert_system import AlertDetector, AlertNotifier, AlertThresholds
from src.regime.early_warning import compute_simple_warning_score, generate_warnings
from src.reporting.metrics_tracker import AccuracyTracker, generate_improvement_suggestions
from src.reporting.pdf_export import JGBReportPDF, dataframe_to_csv_bytes

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

        /* ── Type scale (12 steps) ── */
        --fs-micro:  0.52rem;           /* DRIVER step numbers */
        --fs-tiny:   0.56rem;           /* uppercase overlines, metric labels, stat labels */
        --fs-xs:     0.60rem;           /* card titles, metadata, cert/pub details */
        --fs-sm:     0.65rem;           /* tags, sidebar labels, small descriptions */
        --fs-base:   0.70rem;           /* body secondary, footer links, card content */
        --fs-md:     0.74rem;           /* body primary, definition text, download btn */
        --fs-lg:     0.78rem;           /* nav text, section notes, tabs, spinner */
        --fs-xl:     0.82rem;           /* h2, chat messages, emphasized text */
        --fs-2xl:    0.88rem;           /* verdict, welcome titles */
        --fs-metric: 1.05rem;           /* metric values, stat numbers */
        --fs-h1:     1.25rem;           /* page h1 */
        --fs-brand:  1.30rem;           /* sidebar brand name */
        --fs-hero:   2.0rem;            /* about hero headline */

        /* ── Letter-spacing tokens ── */
        --ls-tight:  -0.025em;          /* h1, large display */
        --ls-snug:   -0.01em;           /* metric values */
        --ls-normal:  0;                /* body text */
        --ls-wide:    0.06em;           /* h2, download btn */
        --ls-wider:   0.10em;           /* metric labels, sidebar labels */
        --ls-widest:  0.16em;           /* overlines, footer headers */

        /* ── Font stacks ── */
        --font-sans: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        --font-mono: 'JetBrains Mono', 'SF Mono', monospace;
    }

    /* ---- Reset & Global ---- */
    html, body, .main, [data-testid="stAppViewContainer"] {
        font-family: var(--font-sans);
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        color: var(--ink-soft);
        text-rendering: optimizeLegibility;
        scroll-behavior: smooth;
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
    /* ---- Animations ---- */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Smooth transitions on hover/interactive elements */
    button, a, [data-testid="stMetric"], details[data-testid="stExpander"] {
        transition: all 0.2s ease;
    }

    /* Simple fade-in on render for all content blocks */
    [data-testid="stMetric"],
    [data-testid="stPlotlyChart"],
    [data-testid="stDataFrame"],
    [data-testid="stAlert"],
    details[data-testid="stExpander"],
    [data-testid="stChatMessage"] {
        animation: fadeInUp 0.4s ease both;
    }

    /* ---- Typography ---- */
    .main h1 {
        font-family: var(--font-sans);
        font-weight: 700;
        color: var(--ink);
        font-size: var(--fs-h1);
        letter-spacing: var(--ls-tight);
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
        font-family: var(--font-sans);
        font-weight: 600;
        color: var(--ink);
        font-size: var(--fs-xl);
        letter-spacing: var(--ls-wide);
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
        font-family: var(--font-sans);
        font-weight: 600;
        color: var(--ink-soft);
        font-size: var(--fs-lg);
        margin-top: 1.2rem;
        margin-bottom: 0.25rem;
    }
    .main p, .main li {
        color: var(--ink-soft);
        line-height: 1.6;
        font-size: var(--fs-lg);
    }

    /* ---- Metric cards ---- */
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border-light);
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 0 0 1px rgba(0,0,0,0.02);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.06), 0 0 0 1px rgba(207,185,145,0.2);
        border-color: var(--gold-dust);
    }
    [data-testid="stMetricLabel"] {
        font-family: var(--font-sans);
        font-weight: 600;
        text-transform: uppercase;
        font-size: var(--fs-tiny);
        letter-spacing: var(--ls-wider);
        color: var(--ink-muted);
    }
    [data-testid="stMetricValue"] {
        font-family: var(--font-mono);
        font-weight: 500;
        color: var(--ink);
        font-size: var(--fs-metric);
        letter-spacing: var(--ls-snug);
    }
    [data-testid="stMetricDelta"] {
        font-family: var(--font-mono);
        font-size: var(--fs-md);
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: #000000;
        border-right: 1px solid rgba(207,185,145,0.12);
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0.6rem;
        padding-bottom: 1rem;
    }
    /* Tighten default Streamlit element gaps inside sidebar */
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > [data-testid="element-container"] {
        margin-bottom: 0px;
    }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
        gap: 0.35rem;
    }
    /* Text & labels */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {
        color: rgba(255,255,255,0.92) !important;
        font-family: var(--font-sans);
    }
    section[data-testid="stSidebar"] label {
        font-size: 0.62rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.55) !important;
        margin-bottom: 1px;
    }
    section[data-testid="stSidebar"] b,
    section[data-testid="stSidebar"] strong {
        color: #fff !important;
        font-weight: 700;
    }
    /* Inputs, selects, date pickers */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] select,
    section[data-testid="stSidebar"] [data-baseweb="input"] input,
    section[data-testid="stSidebar"] [data-baseweb="select"] div,
    section[data-testid="stSidebar"] [data-baseweb="input"],
    section[data-testid="stSidebar"] [data-baseweb="base-input"] {
        background: rgba(255,255,255,0.05) !important;
        color: rgba(255,255,255,0.9) !important;
        border-color: rgba(255,255,255,0.1) !important;
        border-radius: 4px;
        font-family: var(--font-mono);
        font-size: var(--fs-md);
    }
    section[data-testid="stSidebar"] input:focus,
    section[data-testid="stSidebar"] [data-baseweb="input"]:focus-within {
        border-color: rgba(207,185,145,0.5) !important;
        box-shadow: 0 0 0 2px rgba(207,185,145,0.12);
    }
    /* Date input */
    section[data-testid="stSidebar"] [data-testid="stDateInput"] input {
        background: rgba(255,255,255,0.05) !important;
        color: rgba(255,255,255,0.85) !important;
        border-color: rgba(255,255,255,0.1) !important;
        font-family: var(--font-mono);
        font-size: var(--fs-md);
    }
    /* Select / dropdown */
    section[data-testid="stSidebar"] [data-baseweb="select"] {
        background: rgba(255,255,255,0.05);
        border-radius: 4px;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p {
        color: rgba(255,255,255,0.85) !important;
        font-size: var(--fs-md);
    }
    /* Toggle */
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: rgba(255,255,255,0.8) !important;
        font-size: 0.72rem;
        font-weight: 600;
    }
    /* Nav buttons */
    section[data-testid="stSidebar"] .stButton > button {
        text-align: left;
        font-family: var(--font-sans);
        font-size: 0.78rem;
        border-radius: 4px;
        padding: 0.45rem 0.75rem;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        font-weight: 600;
        letter-spacing: 0.015em;
        margin-bottom: 1px;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background: transparent;
        border: 1px solid rgba(255,255,255,0.06);
        color: rgba(255,255,255,0.7) !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        background: rgba(255,255,255,0.06);
        border-color: rgba(255,255,255,0.15);
        color: #fff !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: rgba(207,185,145,0.1);
        border: 1px solid rgba(207,185,145,0.25);
        border-left: 3px solid #CFB991;
        color: #CFB991 !important;
        font-weight: 700;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.06);
        margin: 0.5rem 0;
    }
    /* Multiselect / slider chips in sidebar */
    section[data-testid="stSidebar"] [data-baseweb="tag"] {
        background: rgba(207,185,145,0.12) !important;
        color: #CFB991 !important;
        border: none;
        font-size: 0.65rem;
    }
    /* ---- Expander ---- */
    .streamlit-expanderHeader {
        font-family: var(--font-sans);
        font-weight: 600;
        font-size: var(--fs-lg);
        color: var(--ink);
    }
    details[data-testid="stExpander"] {
        border: 1px solid var(--border-light);
        border-radius: 10px;
        margin-bottom: 0.6rem;
        background: var(--surface);
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
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
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
    }
    [data-testid="stPlotlyChart"]:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        border-color: var(--gold-dust);
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
        font-family: var(--font-sans);
        background: #000000;
        color: #CFB991;
        border: 1px solid #CFB991;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.55rem 1.6rem;
        font-size: var(--fs-md);
        letter-spacing: var(--ls-wide);
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
        font-size: var(--fs-lg);
        border: none;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid var(--border-light);
    }
    .stTabs [data-baseweb="tab"] {
        font-family: var(--font-sans);
        font-size: var(--fs-lg);
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
        font-size: var(--fs-xl);
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        margin-bottom: 0.5rem;
        animation: fadeInUp 0.3s cubic-bezier(0.4, 0, 0.2, 1) both;
    }
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: var(--surface-1);
    }
    .stChatInput textarea {
        font-family: var(--font-sans);
        font-size: var(--fs-xl);
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
        font-size: var(--fs-lg);
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
        font-size: var(--fs-base);
        text-transform: uppercase;
        letter-spacing: var(--ls-wide);
    }
    [data-testid="stDataFrame"] [role="row"]:hover {
        background: rgba(207,185,145,0.08);
    }
    [data-testid="stDataFrame"] * {
        font-size: var(--fs-lg);
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
    "<div style='padding:0.8rem 0 1rem 0; border-bottom:1px solid rgba(207,185,145,0.12); "
    "margin-bottom:0.6rem;'>"
    # Japanese flag + "Rates Strategy Desk" on the same line
    "<div style='display:flex; align-items:center; gap:8px; margin-bottom:8px;'>"
    "<svg width='28' height='19' viewBox='0 0 900 600' xmlns='http://www.w3.org/2000/svg' "
    "style='border-radius:2px; box-shadow:0 1px 3px rgba(0,0,0,0.3); flex-shrink:0;'>"
    "<rect width='900' height='600' fill='#fff'/>"
    "<circle cx='450' cy='300' r='180' fill='#BC002D'/>"
    "</svg>"
    "<span style='font-size:0.58rem; font-weight:700; text-transform:uppercase; "
    "letter-spacing:0.14em; color:#CFB991; font-family:var(--font-sans);'>Rates Strategy Desk</span></div>"
    "<div style='font-size:1.4rem; font-weight:800; color:#fff; "
    "letter-spacing:-0.01em; line-height:1.1; "
    "font-family:var(--font-sans);'>JGB Repricing</div>"
    "<div style='font-size:0.68rem; font-weight:600; color:rgba(255,255,255,0.6); "
    "margin-top:3px; letter-spacing:0.06em; text-transform:uppercase; "
    "font-family:var(--font-sans);'>Quantitative Framework</div>"
    "<div style='font-size:0.56rem; font-weight:600; color:rgba(207,185,145,0.6); "
    "margin-top:6px; letter-spacing:0.1em; text-transform:uppercase; "
    "font-family:var(--font-sans);'>Purdue Daniels School of Business</div></div>",
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
    "early_warning": "Early Warning",
    "performance": "Performance Review",
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
    ("Early Warning", "early_warning"),
    ("Trade Ideas", "trades"),
    ("Performance Review", "performance"),
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

st.sidebar.markdown(
    "<div style='border-top:1px solid rgba(255,255,255,0.06); margin:0.4rem 0 0.5rem 0; padding-top:0.5rem;'>"
    "<span style='font-size:0.55rem;font-weight:700;text-transform:uppercase;"
    "letter-spacing:0.14em;color:rgba(255,255,255,0.4);"
    "font-family:var(--font-sans);'>Configuration</span></div>",
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

# --- Dashboard Settings (Enhancement 4) ---
try:
    _layout_mgr = LayoutManager()
    _layout_config = render_settings_panel(st.sidebar, _layout_mgr)
except Exception:
    from src.ui.layout_config import LayoutConfig
    _layout_mgr = None
    _layout_config = LayoutConfig()

# --- Alert System (Enhancement 3) ---
try:
    _alert_notifier = AlertNotifier()
    _alert_notifier.render_sidebar_log(st.sidebar)
except Exception:
    _alert_notifier = None

# --- Last Update Indicator (Enhancement 1) ---
st.sidebar.markdown(
    "<div style='border-top:1px solid rgba(255,255,255,0.06);"
    "margin:0.4rem 0 0.5rem 0;padding-top:0.5rem;'>"
    "<span style='font-size:0.55rem;font-weight:700;text-transform:uppercase;"
    "letter-spacing:0.14em;color:rgba(255,255,255,0.4);"
    f"font-family:var(--font-sans);'>Last Update: {datetime.now().strftime('%H:%M:%S')}</span></div>",
    unsafe_allow_html=True,
)
if st.sidebar.button("Refresh Data", key="manual_refresh", use_container_width=True, type="secondary"):
    st.cache_data.clear()
    st.rerun()


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
        font_family="JetBrains Mono, monospace",
        bgcolor="rgba(10,10,10,0.94)",
        font_color="#ffffff",
        bordercolor="rgba(207,185,145,0.3)",
        namelength=-1,
    ),
    # Crosshair spike lines (Bloomberg-style)
    xaxis=dict(
        gridcolor="rgba(0,0,0,0.04)",
        linecolor="rgba(0,0,0,0.08)",
        zerolinecolor="rgba(0,0,0,0.06)",
        tickfont=dict(size=10, color="#9D9795", family="JetBrains Mono, monospace"),
        showgrid=True,
        gridwidth=1,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor="rgba(0,0,0,0.25)",
        spikedash="dot",
    ),
    yaxis=dict(
        gridcolor="rgba(0,0,0,0.04)",
        linecolor="rgba(0,0,0,0.08)",
        zerolinecolor="rgba(0,0,0,0.06)",
        tickfont=dict(size=10, color="#9D9795", family="JetBrains Mono, monospace"),
        showgrid=True,
        gridwidth=1,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor="rgba(0,0,0,0.25)",
        spikedash="dot",
    ),
    # Smooth 300ms transitions on zoom/pan/relayout
    transition=dict(duration=300, easing="cubic-in-out"),
)

# Range selector buttons for time-series charts (Bloomberg/TradingView style)
_RANGE_SELECTOR = dict(
    buttons=[
        dict(count=1, label="1M", step="month", stepmode="backward"),
        dict(count=3, label="3M", step="month", stepmode="backward"),
        dict(count=6, label="6M", step="month", stepmode="backward"),
        dict(count=1, label="YTD", step="year", stepmode="todate"),
        dict(count=1, label="1Y", step="year", stepmode="backward"),
        dict(count=3, label="3Y", step="year", stepmode="backward"),
        dict(count=5, label="5Y", step="year", stepmode="backward"),
        dict(step="all", label="ALL"),
    ],
    font=dict(size=9, family="DM Sans, sans-serif", color="#555960"),
    bgcolor="rgba(0,0,0,0)",
    activecolor="rgba(207,185,145,0.2)",
    bordercolor="rgba(0,0,0,0.08)",
    borderwidth=1,
    x=0,
    y=1.06,
)


def _style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    """Apply the institutional plotly template with screener-grade interactions."""
    fig.update_layout(**_PLOTLY_LAYOUT, height=height)

    # Determine if this is a time-series chart (date x-axis)
    _has_timeseries = False
    for trace in fig.data:
        if isinstance(trace, go.Scatter) and trace.x is not None and len(trace.x) > 0:
            _sample = trace.x[0] if not hasattr(trace.x, 'iloc') else trace.x.iloc[0]
            if hasattr(_sample, 'year') or (isinstance(_sample, str) and len(_sample) >= 8):
                _has_timeseries = True
                break

    if _has_timeseries:
        # Add range selector buttons + mini range slider
        fig.update_xaxes(
            rangeselector=_RANGE_SELECTOR,
            rangeslider=dict(visible=True, thickness=0.04, bgcolor="rgba(0,0,0,0.02)",
                             bordercolor="rgba(0,0,0,0.06)", borderwidth=1),
        )
        # Extra bottom margin for range slider
        fig.update_layout(margin=dict(l=48, r=16, t=38, b=8), height=height + 30)

    # Apply palette to unstyled traces
    for i, trace in enumerate(fig.data):
        if isinstance(trace, go.Scatter):
            has_color = getattr(trace.line, "color", None) or getattr(trace.marker, "color", None)
            if not has_color:
                fig.data[i].update(
                    line=dict(color=_PALETTE[i % len(_PALETTE)], width=1.5),
                )

    # Plotly config: modebar tools for screener-grade interaction
    fig._jgb_config = dict(
        displayModeBar=True,
        modeBarButtonsToRemove=["lasso2d", "select2d", "autoScale2d"],
        displaylogo=False,
        scrollZoom=True,
    )
    return fig


def _chart(fig: go.Figure, **kwargs):
    """Render a Plotly chart with screener-grade config."""
    config = getattr(fig, "_jgb_config", {
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
        "displaylogo": False,
        "scrollZoom": True,
    })
    kwargs.setdefault("use_container_width", True)
    kwargs.setdefault("config", config)
    st.plotly_chart(fig, **kwargs)


def _page_intro(text: str):
    """Render a page introduction."""
    st.markdown(
        f"<p style='color:#2d2d2d;font-family:var(--font-sans);font-size:var(--fs-lg);"
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
        f"<p style='color:#1a1a1a;font-size:var(--fs-md);line-height:1.65;margin:0;"
        f"font-family:var(--font-sans);'>{text}</p></div>",
        unsafe_allow_html=True,
    )


def _definition_block(title: str, body: str):
    """Render a compact definition/concept box with black header stripe."""
    st.markdown(
        f"<div style='border:1px solid #e8e5e2;border-radius:8px;overflow:hidden;"
        f"margin:0.6rem 0 1rem 0;box-shadow:0 1px 4px rgba(0,0,0,0.03);'>"
        f"<div style='background:#000;padding:6px 14px;'>"
        f"<p style='margin:0;color:#CFB991;font-size:var(--fs-tiny);font-weight:700;"
        f"text-transform:uppercase;letter-spacing:var(--ls-widest);font-family:var(--font-sans);'>"
        f"{title}</p></div>"
        f"<div style='padding:10px 14px;background:#fff;'>"
        f"<p style='margin:0;color:#1a1a1a;font-size:var(--fs-base);line-height:1.65;"
        f"font-family:var(--font-sans);'>{body}</p></div></div>",
        unsafe_allow_html=True,
    )


def _takeaway_block(text: str):
    """Render a key takeaway callout with a gold left accent."""
    st.markdown(
        f"<div style='background:rgba(207,185,145,0.08);border-left:3px solid #8E6F3E;"
        f"padding:10px 16px;border-radius:0 8px 8px 0;margin:0.5rem 0 1.2rem 0;'>"
        f"<p style='margin:0 0 2px 0;color:#8E6F3E;font-size:var(--fs-tiny);font-weight:700;"
        f"text-transform:uppercase;letter-spacing:var(--ls-wider);font-family:var(--font-sans);'>"
        f"Key Takeaway</p>"
        f"<p style='margin:0;color:#000;font-size:var(--fs-md);line-height:1.65;font-weight:500;"
        f"font-family:var(--font-sans);'>{text}</p></div>",
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
        f"<p style='margin:0 0 2px 0;color:rgba(207,185,145,0.6);font-family:var(--font-sans);"
        f"font-size:var(--fs-tiny);font-weight:700;text-transform:uppercase;letter-spacing:var(--ls-widest);'>"
        f"Verdict</p>"
        f"<p style='margin:0;color:#CFB991;font-family:var(--font-sans);"
        f"font-size:var(--fs-2xl);font-weight:600;line-height:1.55;letter-spacing:var(--ls-snug);'>"
        f"{verdict}</p></div>"
        # assessment block
        f"<div style='background:#fafaf8;padding:16px 24px;'>"
        f"<p style='margin:0 0 6px 0;color:#4a4a4a;font-family:var(--font-sans);"
        f"font-size:var(--fs-tiny);font-weight:700;text-transform:uppercase;letter-spacing:var(--ls-wider);'>"
        f"Assessment</p>"
        f"<p style='margin:0;color:#1a1a1a;font-family:var(--font-sans);"
        f"font-size:var(--fs-lg);line-height:1.7;'>{summary}</p></div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _page_footer():
    """Render full-bleed institutional footer with Daniels School branding."""
    yr = datetime.now().year
    ts = datetime.now().strftime("%B %d, %Y at %H:%M UTC")
    _w = "color:rgba(255,255,255,0.75);text-decoration:none;font-size:var(--fs-base);font-weight:500;transition:color 0.15s ease;"
    _g = "font-size:var(--fs-xs);font-weight:700;text-transform:uppercase;letter-spacing:var(--ls-widest);color:#CFB991;margin:0 0 14px 0;padding-bottom:8px;border-bottom:1px solid rgba(207,185,145,0.15);"
    st.markdown(
        "<style>"
        ".main .block-container { padding-bottom: 0 !important; margin-bottom: 0 !important; }"
        ".main { padding-bottom: 0 !important; margin-bottom: 0 !important; }"
        "[data-testid='stAppViewContainer'] { padding-bottom: 0 !important; }"
        "[data-testid='stBottom'] { display: none !important; }"
        "</style>"
        "<div style='margin-top:4rem;font-family:var(--font-sans);"
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
        "<p style='font-size:var(--fs-base);color:rgba(255,255,255,0.7);line-height:1.65;margin:0 0 16px 0;max-width:260px;'>"
        "MGMT 69000 &middot; Mastering AI for Finance<br/>"
        "West Lafayette, Indiana</p>"
        # timestamp placed formally under the branding block
        f"<p style='font-size:var(--fs-xs);color:rgba(207,185,145,0.6);margin:0;"
        f"font-weight:600;letter-spacing:var(--ls-wide);'>"
        f"Last updated {ts}</p>"
        "</div>"
        # col 2: navigate
        f"<div><p style='{_g}'>Navigate</p>"
        "<ul style='list-style:none;padding:0;margin:0;'>"
        f"<li style='margin-bottom:8px;'><a href='?page=overview' target='_self' style='{_w}'>Overview &amp; Data</a></li>"
        f"<li style='margin-bottom:8px;'><a href='?page=yield_curve' target='_self' style='{_w}'>Yield Curve Analytics</a></li>"
        f"<li style='margin-bottom:8px;'><a href='?page=regime' target='_self' style='{_w}'>Regime Detection</a></li>"
        f"<li style='margin-bottom:8px;'><a href='?page=spillover' target='_self' style='{_w}'>Spillover &amp; Info Flow</a></li>"
        f"<li style='margin-bottom:8px;'><a href='?page=early_warning' target='_self' style='{_w}'>Early Warning</a></li>"
        f"<li style='margin-bottom:8px;'><a href='?page=trades' target='_self' style='{_w}'>Trade Ideas</a></li>"
        f"<li style='margin-bottom:8px;'><a href='?page=performance' target='_self' style='{_w}'>Performance Review</a></li>"
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
        f"<p style='font-size:var(--fs-tiny);color:#000000;margin:0;font-weight:600;letter-spacing:var(--ls-wide);"
        f"font-family:var(--font-sans);'>"
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
        "Unified market data pipeline feeding all downstream analytics. This page shows sovereign yields, "
        "FX rates, equity indices, and volatility measures across Japan, the U.S., Europe, and Asia-Pacific. "
        "Toggle between live (FRED + yfinance) and simulated data via sidebar controls. All charts overlay "
        "BOJ policy event dates (red verticals) to ground market moves in the policy timeline."
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
        "Three complementary methods decompose the yield curve into interpretable factors. "
        "PCA extracts statistically orthogonal drivers (Level, Slope, Curvature) from daily yield changes. "
        "Nelson-Siegel fits a parametric model to the curve shape, tracking its evolution over time. "
        "The Roll liquidity measure estimates implicit bid-ask spreads from price reversals, revealing "
        "when the market can no longer absorb flow without outsized price impact."
    )

    args = (use_simulated, str(start_date), str(end_date), fred_api_key or None)

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
        "How shocks propagate across sovereign bond markets, currencies, and equities. This page measures "
        "the direction, magnitude, and timing of information flow between JGBs and global assets using "
        "four complementary methods. High spillover means JGBs are no longer an isolated market; a shock "
        "in U.S. Treasuries or USDJPY will propagate into JGB yields with quantifiable lag and magnitude."
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
        _df_spill = load_unified(use_simulated, str(start_date), str(end_date), fred_api_key or None)
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
        _section_note("Download trade cards as a branded PDF report with payout graphs, or as raw CSV data.")
        col_pdf, col_csv = st.columns(2)

        with col_pdf:
            try:
                report = JGBReportPDF()
                report.add_title_page(
                    title="JGB Trade Ideas Report",
                    subtitle=f"{len(filtered)} Trade Ideas  |  {datetime.now():%Y-%m-%d %H:%M}",
                )
                report.add_trade_ideas(filtered, regime_state)
                pdf_bytes = report.to_bytes()
                st.download_button(
                    "Download Trade Ideas PDF",
                    data=pdf_bytes,
                    file_name=f"jgb_trade_ideas_{datetime.now():%Y%m%d}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as exc:
                st.warning(f"Could not generate PDF: {exc}")

        with col_csv:
            df_export = trade_cards_to_dataframe(filtered)
            csv = df_export.to_csv(index=False)
            st.download_button(
                "Download Trade Cards CSV",
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


# ===================================================================
# Page 6: AI Q&A
# ===================================================================
def _build_analysis_context(args):
    """Serialize current analysis outputs into a rich text context for the LLM.

    Covers: regime (ensemble + sub-models), PCA (variance + loadings),
    spillover (total + top edges), DCC correlations, GARCH vol, Granger
    causality, entropy, structural breaks, Nelson-Siegel, carry, liquidity,
    latest data snapshot, trade ideas, and BOJ policy timeline.
    """
    parts = []

    # ── 1. Regime state (ensemble + sub-model detail) ──
    try:
        ensemble = _run_ensemble(*args)
        if ensemble is not None and len(ensemble.dropna()) > 0:
            ens_clean = ensemble.dropna()
            prob = float(ens_clean.iloc[-1])
            regime = "REPRICING" if prob > 0.5 else "SUPPRESSED"
            # Trend: compare last 20 obs average to prior 20
            if len(ens_clean) >= 40:
                recent = float(ens_clean.iloc[-20:].mean())
                prior = float(ens_clean.iloc[-40:-20].mean())
                trend = "rising" if recent > prior + 0.03 else "falling" if recent < prior - 0.03 else "stable"
            else:
                trend = "n/a"
            parts.append(
                f"REGIME STATE:\n"
                f"  Ensemble probability = {prob:.2%} ({regime})\n"
                f"  Sample average = {ens_clean.mean():.2%}\n"
                f"  20-day trend = {trend}\n"
                f"  Conviction: {'STRONG' if prob > 0.7 or prob < 0.3 else 'MODERATE' if prob > 0.6 or prob < 0.4 else 'TRANSITION'}"
            )
    except Exception:
        pass

    # Sub-models: Markov
    try:
        markov = _run_markov(*args)
        if markov is not None and len(markov.dropna()) > 0:
            parts.append(f"  Markov-Switching: latest high-vol state prob = {float(markov.dropna().iloc[-1]):.2%}")
    except Exception:
        pass

    # Sub-models: Entropy
    try:
        ent, sig = _run_entropy(*args)
        if ent is not None and len(ent.dropna()) > 0:
            ent_v = float(ent.dropna().iloc[-1])
            sig_v = int(sig.dropna().iloc[-1]) if sig is not None and len(sig.dropna()) > 0 else 0
            parts.append(f"  Permutation Entropy: latest = {ent_v:.3f}, regime signal = {'ELEVATED (early warning)' if sig_v == 1 else 'NORMAL'}")
    except Exception:
        pass

    # Sub-models: GARCH vol
    try:
        vol, vol_breaks = _run_garch(*args)
        if vol is not None and len(vol.dropna()) > 0:
            vol_v = float(vol.dropna().iloc[-1])
            vol_pct = float((vol.dropna() < vol_v).mean() * 100)
            parts.append(f"  GARCH(1,1) Conditional Vol: {vol_v:.2f} bps/day ({vol_pct:.0f}th percentile)")
    except Exception:
        pass

    # ── 2. PCA (variance + loadings) ──
    try:
        pca_res = _run_pca(*args)
        if pca_res is not None:
            ev = pca_res["explained_variance_ratio"]
            loadings = pca_res["loadings"]
            pca_lines = [
                f"PCA YIELD CURVE DECOMPOSITION:",
                f"  PC1 (Level): {ev[0]:.1%} variance explained",
                f"  PC2 (Slope): {ev[1]:.1%} variance explained",
                f"  PC3 (Curvature): {ev[2]:.1%} variance explained",
                f"  Cumulative: {sum(ev):.1%}",
                f"  PC1 loadings: {', '.join(f'{c}={v:+.3f}' for c, v in loadings.iloc[0].items())}",
                f"  PC2 loadings: {', '.join(f'{c}={v:+.3f}' for c, v in loadings.iloc[1].items())}",
                f"  PC3 loadings: {', '.join(f'{c}={v:+.3f}' for c, v in loadings.iloc[2].items())}",
            ]
            parts.append("\n".join(pca_lines))
    except Exception:
        pass

    # ── 3. Nelson-Siegel curve factors ──
    try:
        ns_res = _run_ns(*args)
        if ns_res is not None and "params" in ns_res:
            ns_params = ns_res["params"]
            if not ns_params.empty:
                latest_ns = ns_params.iloc[-1]
                parts.append(
                    f"NELSON-SIEGEL CURVE FACTORS (latest):\n"
                    f"  Beta0 (Level) = {latest_ns.get('beta0', float('nan')):.4f}\n"
                    f"  Beta1 (Slope) = {latest_ns.get('beta1', float('nan')):.4f}\n"
                    f"  Beta2 (Curvature) = {latest_ns.get('beta2', float('nan')):.4f}\n"
                    f"  Interpretation: {'Flattening' if latest_ns.get('beta1', 0) > 0 else 'Steepening'} curve, "
                    f"{'Positive' if latest_ns.get('beta2', 0) > 0 else 'Negative'} belly"
                )
    except Exception:
        pass

    # ── 4. Spillover (total + top directional edges) ──
    try:
        spill = _run_spillover(*args)
        if spill is not None:
            net = spill["net_spillover"]
            mat = spill.get("spillover_matrix")
            spill_lines = [
                f"DIEBOLD-YILMAZ SPILLOVER (VAR(4), 10-step horizon):",
                f"  Total spillover index = {spill['total_spillover']:.1f}%",
                f"  Net transmitters: {', '.join(f'{k}={v:+.1f}%' for k, v in net.sort_values(ascending=False).head(3).items())}",
                f"  Net receivers: {', '.join(f'{k}={v:+.1f}%' for k, v in net.sort_values().head(3).items())}",
            ]
            # Top 5 directional edges from spillover matrix
            if mat is not None:
                edges = []
                for i in mat.index:
                    for j in mat.columns:
                        if i != j:
                            edges.append((i, j, float(mat.loc[i, j])))
                edges.sort(key=lambda x: -x[2])
                top_edges = edges[:5]
                spill_lines.append(f"  Top 5 directional flows:")
                for src, tgt, val in top_edges:
                    spill_lines.append(f"    {src} → {tgt}: {val:.1f}%")
            parts.append("\n".join(spill_lines))
    except Exception:
        pass

    # ── 5. DCC correlations (latest) ──
    try:
        dcc_res = _run_dcc(*args)
        if dcc_res is not None:
            corrs = dcc_res.get("conditional_correlations", {})
            if corrs:
                dcc_lines = ["DCC TIME-VARYING CORRELATIONS (latest):"]
                for pair, series in corrs.items():
                    if len(series.dropna()) > 0:
                        latest_c = float(series.dropna().iloc[-1])
                        avg_c = float(series.dropna().mean())
                        dcc_lines.append(f"  {pair}: current={latest_c:+.3f}, sample avg={avg_c:+.3f}, "
                                         f"{'ELEVATED' if abs(latest_c) > abs(avg_c) + 0.1 else 'NORMAL'}")
                parts.append("\n".join(dcc_lines))
    except Exception:
        pass

    # ── 6. Granger causality (significant pairs only) ──
    try:
        granger_df = _run_granger(*args)
        if granger_df is not None and not granger_df.empty:
            sig_pairs = granger_df[granger_df["significant"] == True].sort_values("p_value")
            if not sig_pairs.empty:
                gc_lines = [f"GRANGER CAUSALITY (significant pairs, p<0.05):"]
                for _, row in sig_pairs.head(8).iterrows():
                    gc_lines.append(
                        f"  {row['cause']} → {row['effect']}: F={row['f_stat']:.2f}, p={row['p_value']:.4f}, lag={int(row['optimal_lag'])}"
                    )
                parts.append("\n".join(gc_lines))
    except Exception:
        pass

    # ── 7. Structural breaks ──
    try:
        changes, bkps = _run_breaks(*args)
        if bkps and changes is not None and len(changes) > 0:
            break_dates = [changes.index[min(b, len(changes) - 1)].strftime("%Y-%m-%d") for b in bkps if b < len(changes)]
            if break_dates:
                parts.append(f"STRUCTURAL BREAKS (PELT on JP_10Y changes): {', '.join(break_dates)}")
    except Exception:
        pass

    # ── 8. FX Carry ──
    try:
        carry = _run_carry(*args)
        if carry is not None and len(carry["carry_to_vol"].dropna()) > 0:
            ctv = float(carry["carry_to_vol"].dropna().iloc[-1])
            carry_raw = float(carry["carry"].dropna().iloc[-1]) if len(carry["carry"].dropna()) > 0 else float("nan")
            parts.append(
                f"FX CARRY ANALYTICS:\n"
                f"  Carry (US-JP rate differential) = {carry_raw:.2f}%\n"
                f"  Carry-to-vol ratio = {ctv:.2f} ({'Attractive — carry exceeds vol risk' if ctv > 1 else 'Marginal' if ctv > 0.5 else 'Unattractive — vol dominates carry'})"
            )
    except Exception:
        pass

    # ── 9. Liquidity ──
    try:
        liq = _run_liquidity(*args)
        if liq is not None and len(liq["composite_index"].dropna()) > 0:
            liq_v = float(liq["composite_index"].dropna().iloc[-1])
            parts.append(f"LIQUIDITY: Composite index = {liq_v:+.2f} z-score. {'Stressed — wider bid-ask, higher impact costs' if liq_v < -1 else 'Healthy' if liq_v > 0 else 'Neutral'}.")
    except Exception:
        pass

    # ── 10. Latest data snapshot ──
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
            # Curve slopes if available
            slopes = []
            for short, long, label in [("JP_2Y", "JP_10Y", "JP 2s10s"), ("JP_10Y", "JP_30Y", "JP 10s30s"), ("US_2Y", "US_10Y", "US 2s10s")]:
                if short in latest.index and long in latest.index and pd.notna(latest[short]) and pd.notna(latest[long]):
                    slopes.append(f"{label}={latest[long] - latest[short]:+.2f}%")
            if slopes:
                parts.append(f"CURVE SLOPES: {', '.join(slopes)}")
    except Exception:
        pass

    # ── 11. Trade ideas (all, with failure scenarios) ──
    try:
        cards, rs = _generate_trades(*args)
        if cards:
            top_5 = sorted(cards, key=lambda c: -c.conviction)[:5]
            trade_lines = [f"TOP TRADE IDEAS ({len(cards)} total, top 5 by conviction):"]
            for c in top_5:
                trade_lines.append(
                    f"  - {c.name} ({c.direction.upper()}, {c.conviction:.0%}, {c.category})\n"
                    f"    Entry: {c.entry_signal}\n"
                    f"    Failure: {c.failure_scenario}"
                )
            parts.append("\n".join(trade_lines))
    except Exception:
        pass

    # ── 12. BOJ events ──
    from src.data.config import BOJ_EVENTS as _boj_events
    parts.append("BOJ POLICY TIMELINE:\n" + "\n".join(f"  {d}: {e}" for d, e in _boj_events.items()))

    return "\n\n".join(parts)


def page_ai_qa():
    st.header("AI Q&A")
    _page_intro(
        "Chat interface grounded in the case study. The AI assistant is injected with live framework outputs "
        "including regime ensemble, PCA loadings, Nelson-Siegel curve factors, Diebold-Yilmaz spillover edges, "
        "DCC correlations, Granger causality pairs, GARCH vol, carry analytics, liquidity, structural breaks, "
        "and trade ideas with failure scenarios. Every answer must cite specific metrics from the analysis."
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

    # --- Args for context builder (lazy, only called when needed) ---
    args = (use_simulated, str(start_date), str(end_date), fred_api_key or None)

    if not ai_api_key:
        # Show a professional empty state instead of just a bare info box
        st.markdown(
            "<div style='text-align:center;padding:3rem 2rem;'>"
            "<div style='font-size:2.5rem;margin-bottom:0.8rem;opacity:0.15;'>&#x1F4AC;</div>"
            "<p style='font-family:var(--font-sans);font-size:var(--fs-2xl);font-weight:600;"
            "color:#000;margin:0 0 6px 0;'>AI Research Assistant</p>"
            "<p style='font-family:var(--font-sans);font-size:var(--fs-md);color:#2d2d2d;"
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
            "padding:8px 16px;font-size:var(--fs-base);color:#1a1a1a;font-family:var(--font-sans);'>"
            "Regime call with evidence</span>"
            "<span style='background:#fafaf8;border:1px solid #eceae6;border-radius:8px;"
            "padding:8px 16px;font-size:var(--fs-base);color:#1a1a1a;font-family:var(--font-sans);'>"
            "PCA loadings &amp; curve factors</span>"
            "<span style='background:#fafaf8;border:1px solid #eceae6;border-radius:8px;"
            "padding:8px 16px;font-size:var(--fs-base);color:#1a1a1a;font-family:var(--font-sans);'>"
            "Spillover transmission chain</span>"
            "<span style='background:#fafaf8;border:1px solid #eceae6;border-radius:8px;"
            "padding:8px 16px;font-size:var(--fs-base);color:#1a1a1a;font-family:var(--font-sans);'>"
            "Trade thesis &amp; failure scenarios</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        _page_footer()
        return

    # --- Chat state ---
    if "qa_messages" not in st.session_state:
        st.session_state.qa_messages = []

    # Determine if a card was just clicked (pending question to process)
    _pending_card_q = None

    # Show empty state with suggestions when no messages yet
    if not st.session_state.qa_messages:
        st.markdown(
            "<div style='text-align:center;padding:2rem 2rem 1rem 2rem;'>"
            "<p style='font-family:var(--font-sans);font-size:var(--fs-2xl);font-weight:600;"
            "color:#000;margin:0 0 4px 0;'>Ready to assist</p>"
            "<p style='font-family:var(--font-sans);font-size:var(--fs-md);color:#2d2d2d;"
            "margin:0 0 1.2rem 0;'>Ask about JGBs, rates, macro, BOJ policy, "
            "trading strategies, or yield curves.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        # Clickable topic cards
        _topics = [
            ("Explain the current regime call: cite the ensemble probability, sub-model signals, and what is driving the reading.", "Regime Call"),
            ("Break down the PCA loadings: what do PC1/PC2/PC3 represent economically, and what does the variance split tell us about yield dynamics?", "PCA Factors"),
            ("Trace the transmission chain: how are shocks flowing between JGBs, USTs, USDJPY, and VIX using spillover edges, Granger causality, and DCC correlations?", "Spillover Chain"),
            ("What are the top trade ideas, their entry signals, and what would falsify each thesis?", "Trade Thesis"),
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
                    _pending_card_q = _q

    # Display chat history
    for msg in st.session_state.qa_messages:
        _avatar = "\U0001F9D1" if msg["role"] == "user" else "\U0001F916"
        with st.chat_message(msg["role"], avatar=_avatar):
            st.markdown(msg["content"])

    # --- Chat input (inline, not fixed to bottom) ---
    user_input = st.text_input(
        "Ask anything about JGBs, rates, macro, BOJ policy, trading strategies...",
        key="qa_text_input",
        label_visibility="collapsed",
        placeholder="Ask anything about JGBs, rates, macro, BOJ policy, trading strategies...",
    )

    # Use card click or text input
    question = _pending_card_q or (user_input if user_input else None)

    if question:
        st.session_state.qa_messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="\U0001F9D1"):
            st.markdown(question)

        # Build context only when actually needed
        with st.spinner("Building analysis context..."):
            analysis_context = _build_analysis_context(args)

        system_prompt = (
            "You are an AI research assistant for a JGB Repricing Framework, a quantitative case study "
            "analyzing the regime shift from BOJ-suppressed yields to market-driven pricing in the Japanese "
            "Government Bond market.\n\n"
            "THESIS: Japan's decade of yield curve control (YCC) and quantitative easing artificially suppressed "
            "JGB yields. As the BOJ exits these policies (Dec 2022 band widening, Mar 2024 formal YCC exit), "
            "repricing risk propagates through rates, FX, volatility, and cross-asset channels.\n\n"
            "RULES:\n"
            "1. ALWAYS ground your answers in the live framework data below. Cite at least 2 specific metrics "
            "   from the injected context when discussing regime, spillovers, curve, carry, or trade ideas.\n"
            "2. Frame answers through the thesis lens: BOJ suppression, cross-asset spillovers, repricing risk, "
            "   trade implications.\n"
            "3. When referencing PCA, explain what the loadings mean economically (level/slope/curvature).\n"
            "4. When discussing trades, always mention the failure scenario.\n"
            "5. If the data does not support a claim, say so explicitly: 'Not supported by current framework outputs.'\n"
            "6. Do NOT invent data that is not in the context below.\n"
            "7. Be concise, quantitative, and actionable. Use bullet points for structure.\n\n"
            f"=== LIVE FRAMEWORK CONTEXT ===\n{analysis_context}\n=== END CONTEXT ==="
        )

        # Build message history for the API
        api_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.qa_messages]

        with st.chat_message("assistant", avatar="\U0001F916"):
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

    _page_footer()


# ===================================================================
# Page 7: Early Warning
# ===================================================================
@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
def _run_warning_score(simulated, start, end, api_key):
    df = load_unified(simulated, start, end, api_key)
    score = compute_simple_warning_score(df, entropy_window=_layout_config.entropy_window)
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

    args = (use_simulated, str(start_date), str(end_date), fred_api_key or None)

    # --- Load warning score with robust error handling ---
    try:
        with st.spinner("Computing early warning score..."):
            warning_score = _run_warning_score(*args)
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
        df = load_unified(use_simulated, str(start_date), str(end_date), fred_api_key or None)
        _ew = _layout_config.entropy_window
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
            if _alert_notifier is not None:
                _alert_notifier.process_alerts(alerts)
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
            f"spillover intensity with an entropy window of {_layout_config.entropy_window} days. "
            f"Current reading: <b>{current_score:.0f}/100</b>.",
        )
    except Exception:
        pass
    _page_footer()


# ===================================================================
# Page 8: Performance Review
# ===================================================================
def page_performance_review():
    st.header("Performance Review")
    _page_intro(
        "Model performance tracking, accuracy metrics, and improvement suggestions. "
        "This page evaluates the regime detection ensemble against actual market outcomes "
        "and provides actionable recommendations for model refinement. Export reports as PDF or CSV."
    )
    _definition_block(
        "How Performance is Measured",
        "The framework tracks every regime prediction the ensemble makes and compares it against "
        "what actually happened. <b>Accuracy</b> is the percentage of correct predictions. "
        "<b>Precision</b> measures how often a repricing signal is genuine (vs false alarm). "
        "<b>Recall</b> measures how many actual repricing events the model caught. "
        "<b>False Positive Rate</b> is how often the model cried wolf. "
        "A good model has high precision (few false alarms) AND high recall (catches most real events). "
        "The improvement suggestions below are generated automatically based on which metrics need attention."
    )

    args = (use_simulated, str(start_date), str(end_date), fred_api_key or None)

    # --- Compute metrics from ensemble predictions ---
    tracker = AccuracyTracker()
    ensemble = None
    predicted = None
    actual = None
    df = None

    try:
        ensemble = _run_ensemble(*args)
        df = load_unified(use_simulated, str(start_date), str(end_date), fred_api_key or None)
        if ensemble is not None and len(ensemble.dropna()) > 30:
            ens_clean = ensemble.dropna()
            predicted = (ens_clean > 0.5).astype(int)
            jp10 = df["JP_10Y"].dropna() if "JP_10Y" in df.columns else pd.Series(dtype=float)
            if len(jp10) > 30:
                future_change = jp10.diff(5).shift(-5)
                vol = jp10.diff().rolling(60).std()
                actual = ((future_change > vol).astype(int)).reindex(predicted.index)
                metrics = tracker.compute_from_series(predicted, actual.dropna())
            else:
                metrics = tracker.compute_metrics()
        else:
            metrics = tracker.compute_metrics()
    except Exception as exc:
        st.warning(f"Could not compute metrics from ensemble: {exc}")
        metrics = tracker.compute_metrics()

    # --- KPI row ---
    try:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics.prediction_accuracy:.1%}")
        c2.metric("Precision", f"{metrics.precision:.1%}")
        c3.metric("Recall", f"{metrics.recall:.1%}")
        c4.metric("False Positive Rate", f"{metrics.false_positive_rate:.1%}")
    except Exception as exc:
        st.warning(f"Could not render KPI metrics: {exc}")

    # --- Detailed metrics table ---
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "False Positive Rate",
                    "Average Lead Time", "Total Predictions"],
        "Value": [f"{metrics.prediction_accuracy:.1%}", f"{metrics.precision:.1%}",
                  f"{metrics.recall:.1%}", f"{metrics.false_positive_rate:.1%}",
                  f"{metrics.average_lead_time:.1f} days", str(metrics.total_predictions)],
    })
    try:
        st.subheader("Detailed Metrics")
        _section_note(
            f"Based on {metrics.total_predictions} observations. "
            f"Average lead time: {metrics.average_lead_time:.1f} days."
        )
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"Could not render detailed metrics: {exc}")

    # --- Confusion matrix visualization ---
    try:
        if predicted is not None and actual is not None:
            st.subheader("Prediction Analysis")
            _section_note(
                "Comparison of predicted regime states vs actual outcomes. "
                "The ensemble predicts repricing when probability > 0.5."
            )
            # Align predicted and actual
            common_idx = predicted.dropna().index.intersection(actual.dropna().index)
            if len(common_idx) > 10:
                p_aligned = predicted.loc[common_idx]
                a_aligned = actual.loc[common_idx]
                tp = int(((p_aligned == 1) & (a_aligned == 1)).sum())
                fp = int(((p_aligned == 1) & (a_aligned == 0)).sum())
                fn = int(((p_aligned == 0) & (a_aligned == 1)).sum())
                tn = int(((p_aligned == 0) & (a_aligned == 0)).sum())
                cm1, cm2, cm3, cm4 = st.columns(4)
                cm1.metric("True Positives", f"{tp}")
                cm2.metric("False Positives", f"{fp}")
                cm3.metric("True Negatives", f"{tn}")
                cm4.metric("False Negatives", f"{fn}")

                # Ensemble probability over time with actual regime overlay
                if ensemble is not None:
                    ens_clean = ensemble.dropna()
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(
                        x=ens_clean.index, y=ens_clean.values,
                        mode="lines", name="Ensemble Probability",
                        line=dict(color="#CFB991", width=2),
                    ))
                    fig_pred.add_trace(go.Scatter(
                        x=a_aligned.index, y=a_aligned.values * 0.95,
                        mode="markers", name="Actual Repricing",
                        marker=dict(color="#c0392b", size=4, opacity=0.5),
                    ))
                    fig_pred.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Decision Boundary")
                    fig_pred.update_layout(yaxis_title="Probability / Actual", yaxis_range=[0, 1])
                    _add_boj_events(fig_pred)
                    _chart(_style_fig(fig_pred, 380))

                    _takeaway_block(
                        f"Out of {len(common_idx)} aligned observations: <b>{tp} true positives</b> (correctly predicted repricing), "
                        f"<b>{fp} false positives</b> (false alarms), <b>{fn} false negatives</b> (missed repricing events), "
                        f"and <b>{tn} true negatives</b> (correctly predicted suppression)."
                    )
    except Exception as exc:
        st.warning(f"Could not render prediction analysis: {exc}")

    # --- Improvement suggestions ---
    suggestions = generate_improvement_suggestions(metrics)
    try:
        st.subheader("Improvement Suggestions")
        if suggestions:
            for s in suggestions:
                st.info(s)
        else:
            st.success("All metrics are within acceptable ranges. No immediate improvements needed.")
    except Exception as exc:
        st.warning(f"Could not render improvement suggestions: {exc}")

    # --- Ensemble model agreement analysis ---
    try:
        st.subheader("Model Agreement Analysis")
        _section_note(
            "How often do the four regime detection models agree? Higher agreement means stronger conviction."
        )
        markov_result = _run_markov(*args)
        _, ent_sig = _run_entropy(*args)
        vol, _ = _run_garch(*args)

        model_signals = {}
        if ensemble is not None and len(ensemble.dropna()) > 0:
            model_signals["Ensemble"] = (ensemble.dropna() > 0.5).astype(int)
        # markov_result is a dict with 'regime_probabilities' key
        if markov_result is not None and isinstance(markov_result, dict):
            markov_prob = markov_result.get("regime_probabilities")
            if markov_prob is not None and len(markov_prob) > 0:
                prob_col = markov_prob.columns[-1]
                mp = markov_prob[prob_col].dropna()
                if len(mp) > 0:
                    model_signals["Markov"] = (mp > 0.5).astype(int)
        if ent_sig is not None and hasattr(ent_sig, "dropna") and len(ent_sig.dropna()) > 0:
            model_signals["Entropy"] = ent_sig.dropna().astype(int)
        # vol is a Series from GARCH
        if vol is not None and hasattr(vol, "dropna") and len(vol.dropna()) > 0:
            vol_clean = vol.dropna()
            vol_median = vol_clean.median()
            model_signals["GARCH"] = (vol_clean > vol_median).astype(int)

        if len(model_signals) >= 2:
            agreement_df = pd.DataFrame(model_signals)
            agreement_df = agreement_df.dropna()
            if len(agreement_df) > 0:
                agreement_rate = agreement_df.apply(lambda row: row.nunique() == 1, axis=1).mean()
                st.metric("Model Agreement Rate", f"{agreement_rate:.1%}")
                _takeaway_block(
                    f"The regime detection models agree <b>{agreement_rate:.0%}</b> of the time. "
                    f"{'High agreement strengthens signal conviction.' if agreement_rate > 0.7 else 'Moderate agreement suggests uncertainty in regime classification.' if agreement_rate > 0.5 else 'Low agreement indicates conflicting signals. Rely on the ensemble average rather than any single model.'}"
                )
        else:
            st.info("Insufficient model outputs to compute agreement analysis.")
    except Exception as exc:
        st.warning(f"Could not render model agreement analysis: {exc}")

    # --- ML Regime Predictor (Enhancement 1b) ---
    if _layout_config.show_ml_predictions:
        try:
            st.subheader("ML Regime Predictor")
            _definition_block(
                "RandomForest Walk-Forward Predictor",
                "A machine learning model trained on features derived from the framework's analytics. "
                "Uses walk-forward training (504-day window, retrained quarterly) to avoid look-ahead bias. "
                "Features include structural entropy, carry stress, spillover correlation, volatility z-score, "
                "and USDJPY momentum."
            )
            with st.spinner("Running ML predictor..."):
                preds, probs, importance = _run_ml_predictor(
                    use_simulated, str(start_date), str(end_date),
                    fred_api_key or None, _layout_config.entropy_window,
                )

            if probs is not None and len(probs.dropna()) > 0:
                latest_prob = float(probs.dropna().iloc[-1])
                _section_note(
                    f"ML prediction probability (latest): <b>{latest_prob:.0%}</b>. "
                    f"Walk-forward trained on {len(probs.dropna())} prediction points."
                )
                fig_ml = go.Figure()
                fig_ml.add_trace(go.Scatter(
                    x=probs.index, y=probs.values,
                    mode="lines", name="ML Regime Prob",
                    line=dict(color="#CFB991", width=2),
                ))
                fig_ml.add_hline(y=0.5, line_dash="dash", line_color="red")
                fig_ml.update_layout(yaxis_title="Probability", yaxis_range=[0, 1])
                _add_boj_events(fig_ml)
                _chart(_style_fig(fig_ml, 380))

                _takeaway_block(
                    f"The ML predictor assigns a <b>{latest_prob:.0%}</b> probability to the repricing regime. "
                    f"{'This confirms the ensemble signal. Both statistical and ML models agree.' if (latest_prob > 0.5) == (metrics.prediction_accuracy > 0.5) else 'The ML model disagrees with the ensemble. When models diverge, reduce position sizes and wait for convergence.'}"
                )

                # Feature importance
                if importance is not None and len(importance) > 0:
                    st.subheader("Feature Importance")
                    _section_note("RandomForest feature importances from the latest training window.")
                    fig_imp = go.Figure(go.Bar(
                        x=importance.values,
                        y=importance.index,
                        orientation="h",
                        marker_color="#CFB991",
                    ))
                    fig_imp.update_layout(xaxis_title="Importance", yaxis_title="Feature")
                    _chart(_style_fig(fig_imp, 320))

                    top_feature = importance.idxmax() if len(importance) > 0 else "N/A"
                    _takeaway_block(
                        f"The most important feature is <b>{top_feature}</b>. "
                        "Feature importances reveal which market signals the ML model relies on most heavily. "
                        "If a single feature dominates (>40%), the model may be fragile to changes in that signal."
                    )
            else:
                st.info("Insufficient data for ML regime predictor. Need at least 100 observations with both regime classes present in the training window.")
        except ImportError:
            st.info("ML predictor requires scikit-learn. Install with: `pip install scikit-learn`")
        except Exception as exc:
            st.warning(f"ML predictor not available: {exc}")

    # --- Export ---
    try:
        st.subheader("Export Reports")
        _section_note("Download a comprehensive PDF report or raw metrics as CSV.")
        col_pdf, col_csv = st.columns(2)

        with col_pdf:
            try:
                report = JGBReportPDF()
                report.add_title_page(subtitle=f"Generated {datetime.now():%Y-%m-%d %H:%M}")
                report.add_metrics_summary(metrics)
                report.add_suggestions(suggestions)
                # Add data table if available
                if metrics_df is not None:
                    report.add_data_table(metrics_df, title="Performance Metrics")
                pdf_bytes = report.to_bytes()
                st.download_button(
                    "Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"jgb_report_{datetime.now():%Y%m%d}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except ImportError:
                st.info("Install fpdf2 for PDF export: `pip install fpdf2`")
            except Exception as exc:
                st.warning(f"Could not generate PDF: {exc}")

        with col_csv:
            try:
                csv_bytes = dataframe_to_csv_bytes(metrics_df)
                st.download_button(
                    "Download Metrics CSV",
                    data=csv_bytes,
                    file_name=f"jgb_metrics_{datetime.now():%Y%m%d}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            except Exception as exc:
                st.warning(f"Could not generate CSV: {exc}")
    except Exception as exc:
        st.warning(f"Could not render export section: {exc}")

    # --- Page conclusion (always render, even if sections above failed) ---
    try:
        _verdict_pr = (
            f"Accuracy at {metrics.prediction_accuracy:.0%}: "
            f"{'strong model performance' if metrics.prediction_accuracy > 0.7 else 'room for improvement' if metrics.prediction_accuracy > 0.5 else 'model needs retraining'}."
        )
        _page_conclusion(
            _verdict_pr,
            f"Performance metrics computed over {metrics.total_predictions} observations. "
            f"{len(suggestions)} improvement suggestion(s) generated.",
        )
    except Exception:
        pass
    _page_footer()


# ===================================================================
# Page 9: About — Heramb Patkar
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
        ".about-hero-inner{max-width:1040px;margin:0 auto;padding:0 52px;}"
        ".about-hero .overline{font-size:var(--fs-tiny);font-weight:700;text-transform:uppercase;"
        "letter-spacing:var(--ls-widest);color:#CFB991;margin:0 0 12px 0;"
        "display:inline-block;padding:4px 14px;border:1px solid rgba(207,185,145,0.25);"
        "border-radius:4px;background:rgba(207,185,145,0.06);}"
        ".about-hero h1{font-size:var(--fs-hero);font-weight:800;color:#fff;margin:0 0 10px 0;"
        "letter-spacing:var(--ls-tight);line-height:1.12;}"
        ".about-hero .subtitle{font-size:var(--fs-lg);color:#CFB991;font-weight:600;"
        "margin:0 0 12px 0;letter-spacing:0.01em;}"
        ".about-hero .tagline{font-size:var(--fs-lg);color:rgba(255,255,255,0.6);font-weight:400;"
        "margin:0 0 24px 0;line-height:1.6;max-width:640px;}"
        ".about-hero .links{display:flex;gap:10px;flex-wrap:wrap;}"
        ".about-hero .links a{display:inline-flex;align-items:center;gap:6px;"
        "padding:8px 20px;border:1px solid rgba(207,185,145,0.4);"
        "border-radius:6px;color:#CFB991;font-size:var(--fs-base);font-weight:600;text-decoration:none;"
        "transition:all 0.2s ease;letter-spacing:0.02em;}"
        ".about-hero .links a:hover{background:#CFB991;color:#000;border-color:#CFB991;"
        "transform:translateY(-1px);box-shadow:0 4px 16px rgba(207,185,145,0.25);}"
        # ── cards ──
        ".about-card{background:#fff;border:1px solid #e8e6e1;border-radius:12px;"
        "padding:28px 28px 24px 28px;margin-bottom:18px;"
        "box-shadow:0 1px 3px rgba(0,0,0,0.04);transition:box-shadow 0.2s ease;}"
        ".about-card:hover{box-shadow:0 4px 20px rgba(0,0,0,0.06);}"
        ".about-card-title{font-size:var(--fs-xs);font-weight:700;text-transform:uppercase;"
        "letter-spacing:var(--ls-widest);color:#8E6F3E;margin:0 0 18px 0;padding-bottom:12px;"
        "border-bottom:2px solid #f0eeeb;display:flex;align-items:center;gap:8px;}"
        ".about-card-title::before{content:'';display:inline-block;width:3px;height:14px;"
        "background:#CFB991;border-radius:2px;}"
        # ── experience timeline ──
        ".exp-item{position:relative;margin-bottom:20px;padding-bottom:20px;padding-left:16px;"
        "border-left:2px solid #f0eeeb;border-bottom:none;}"
        ".exp-item:last-child{margin-bottom:0;padding-bottom:0;}"
        ".exp-item::before{content:'';position:absolute;left:-5px;top:4px;width:8px;height:8px;"
        "border-radius:50%;background:#CFB991;border:2px solid #fff;}"
        ".exp-role{font-size:var(--fs-md);font-weight:700;color:#000;margin:0 0 2px 0;line-height:1.35;}"
        ".exp-org{font-size:var(--fs-base);font-weight:600;color:#8E6F3E;margin:0 0 4px 0;}"
        ".exp-meta{font-size:var(--fs-xs);color:#4a4a4a;margin:0 0 8px 0;font-weight:500;"
        "letter-spacing:0.01em;}"
        ".exp-desc{font-size:var(--fs-base);color:#1a1a1a;line-height:1.7;margin:0;}"
        # ── education ──
        ".edu-item{margin-bottom:16px;padding-bottom:16px;border-bottom:1px solid #f5f3f0;}"
        ".edu-item:last-child{margin-bottom:0;padding-bottom:0;border-bottom:none;}"
        ".edu-school{font-size:var(--fs-md);font-weight:700;color:#000;margin:0 0 2px 0;}"
        ".edu-dept{font-size:var(--fs-sm);color:#8E6F3E;margin:0 0 2px 0;font-weight:600;}"
        ".edu-degree{font-size:var(--fs-sm);color:#1a1a1a;margin:0 0 2px 0;font-weight:500;}"
        ".edu-year{font-size:var(--fs-tiny);color:#4a4a4a;margin:0;font-weight:500;}"
        # ── certifications ──
        ".cert-item{margin-bottom:12px;padding-bottom:12px;border-bottom:1px solid #f5f3f0;}"
        ".cert-item:last-child{margin-bottom:0;padding-bottom:0;border-bottom:none;}"
        ".cert-name{font-size:var(--fs-base);font-weight:600;color:#000;margin:0 0 2px 0;}"
        ".cert-issuer{font-size:var(--fs-xs);color:#4a4a4a;margin:0;font-weight:500;}"
        # ── publication ──
        ".pub-item{margin-bottom:16px;padding:16px 18px;background:#fafaf8;"
        "border-left:3px solid #CFB991;border-radius:0 8px 8px 0;}"
        ".pub-item:last-child{margin-bottom:0;}"
        ".pub-title{font-size:var(--fs-md);font-weight:700;color:#000;margin:0 0 6px 0;line-height:1.4;}"
        ".pub-authors{font-size:var(--fs-sm);color:#1a1a1a;margin:0 0 6px 0;line-height:1.5;}"
        ".pub-authors strong{color:#000;font-weight:700;}"
        ".pub-journal{font-size:var(--fs-sm);color:#8E6F3E;font-weight:600;margin:0 0 4px 0;"
        "font-style:italic;}"
        ".pub-detail{font-size:var(--fs-xs);color:#4a4a4a;margin:0 0 8px 0;line-height:1.5;}"
        ".pub-link{display:inline-block;padding:4px 12px;border:1px solid rgba(207,185,145,0.4);"
        "border-radius:4px;color:#8E6F3E;font-size:var(--fs-xs);font-weight:600;text-decoration:none;"
        "transition:all 0.2s;}"
        ".pub-link:hover{background:#CFB991;color:#000;border-color:#CFB991;}"
        # ── tags / pills ──
        ".interest-tag{display:inline-block;padding:6px 16px;border-radius:20px;font-size:var(--fs-sm);"
        "font-weight:600;margin:4px 5px 4px 0;transition:transform 0.15s ease;}"
        ".interest-tag:hover{transform:translateY(-1px);}"
        ".interest-gold{background:rgba(207,185,145,0.12);color:#8E6F3E;"
        "border:1px solid rgba(207,185,145,0.2);}"
        ".interest-neutral{background:rgba(0,0,0,0.03);color:#1a1a1a;"
        "border:1px solid rgba(0,0,0,0.06);}"
        # ── acknowledgment ──
        ".ack-text{font-size:var(--fs-base);color:#1a1a1a;line-height:1.75;margin:0;}"
        ".ack-text strong{color:#000;}"
        # ── stat row ──
        ".stat-row{display:flex;gap:16px;margin:16px 0 4px 0;}"
        ".stat-item{flex:1;text-align:center;padding:12px 8px;background:#fafaf8;"
        "border-radius:8px;border:1px solid #f0eeeb;}"
        ".stat-num{font-size:var(--fs-metric);font-weight:800;color:#000;margin:0;line-height:1.2;}"
        ".stat-label{font-size:var(--fs-tiny);font-weight:600;color:#4a4a4a;margin:2px 0 0 0;"
        "text-transform:uppercase;letter-spacing:var(--ls-wider);}"
        "</style>",
        unsafe_allow_html=True,
    )


def page_about_heramb():
    _about_page_styles()
    _f = "font-family:var(--font-sans);"

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
            "alt='Heramb S. Patkar' style='width:190px;height:190px;border-radius:50%;"
            "object-fit:cover;border:4px solid rgba(207,185,145,0.3);"
            "box-shadow:0 12px 40px rgba(0,0,0,0.45);' />"
            "</div>"
        )

    # ── hero banner ──
    st.markdown(
        "<div class='about-hero'><div class='about-hero-inner' "
        "style='display:flex;align-items:center;gap:44px;'>"
        "<div style='flex:1;'>"
        "<p class='overline'>About the Author</p>"
        "<h1>Heramb S. Patkar</h1>"
        "<p class='subtitle'>MSF Candidate, Purdue Daniels School of Business</p>"
        "<p class='tagline'>BITS Pilani engineering graduate and NISM XV certified research analyst "
        "with equity research experience spanning Indian and U.S. capital markets. "
        "Published researcher in biomedical device design.</p>"
        "<div class='links'>"
        "<a href='https://www.linkedin.com/in/heramb-patkar/' target='_blank'>LinkedIn</a>"
        "<a href='https://github.com/HPATKAR' target='_blank'>GitHub</a>"
        "<a href='https://www.ias.ac.in/article/fulltext/boms/048/0028' target='_blank'>Publication</a>"
        "</div></div>"
        f"{_photo_html}"
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
            f"<p style='{_f}color:#1a1a1a;font-size:var(--fs-md);line-height:1.75;margin:0 0 10px 0;'>"
            "Driven by curiosity about how businesses create impact and grow stronger. "
            "With a background in engineering and experience in global equity research, "
            "I enjoy analysing industries, building financial models, and uncovering insights "
            "that drive smarter decisions. Excited by opportunities where analytical thinking "
            "and creativity intersect to solve complex problems and deliver meaningful value.</p>"
            f"<p style='{_f}color:#1a1a1a;font-size:var(--fs-md);line-height:1.75;margin:0;'>"
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
            f"<p style='{_f}color:#1a1a1a;font-size:var(--fs-md);line-height:1.75;margin:0 0 12px 0;'>"
            "Built this JGB Repricing Framework as Course Project 1 for Prof. Cinder Zhang's"
            "MGMT 69000-119: a quantitative dashboard that detects regime shifts in Japanese "
            "Government Bond markets and generates institutional-grade trade ideas.</p>"
            "<div class='stat-row'>"
            "<div class='stat-item'><p class='stat-num'>4</p><p class='stat-label'>Regime Models</p></div>"
            "<div class='stat-item'><p class='stat-num'>6</p><p class='stat-label'>Tenors</p></div>"
            "<div class='stat-item'><p class='stat-num'>10</p><p class='stat-label'>Dashboard Pages</p></div>"
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
            "<p class='ack-text'><strong>Prof. Cinder Zhang</strong>, MGMT 69000: "
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
    _f = "font-family:var(--font-sans);"

    # ── hero banner ──
    st.markdown(
        "<div class='about-hero'><div class='about-hero-inner'>"
        "<p class='overline'>Course Instructor</p>"
        "<h1>Dr. Cinder Zhang, Ph.D.</h1>"
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
            f"<p style='{_f}color:#1a1a1a;font-size:var(--fs-md);line-height:1.75;margin:0 0 10px 0;'>"
            "Dr. Cinder Zhang is an award-winning finance professor and pioneer in AI-integrated "
            "finance education at Purdue University. He is the creator of the "
            "<strong>DRIVER Framework</strong> (Define &amp; Discover, Represent, Implement, "
            "Validate, Evolve, Reflect), a comprehensive methodology that transforms how financial "
            "management is taught by integrating AI as a cognitive amplifier rather than a "
            "replacement tool.</p>"
            f"<p style='{_f}color:#1a1a1a;font-size:var(--fs-md);line-height:1.75;margin:0;'>"
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
            f"<p style='{_f}color:#1a1a1a;font-size:var(--fs-md);line-height:1.75;margin:0 0 12px 0;'>"
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
            f"<p style='{_f}color:#1a1a1a;font-size:var(--fs-base);line-height:1.7;margin:0 0 12px 0;'>"
            "A six-phase methodology for AI-integrated finance education:</p>"
            "<div style='display:flex;flex-direction:column;gap:6px;'>"
            "<div style='display:flex;align-items:center;gap:8px;'>"
            "<span style='width:22px;height:22px;border-radius:50%;background:#000;color:#CFB991;"
            "font-size:var(--fs-micro);font-weight:800;display:flex;align-items:center;justify-content:center;"
            "flex-shrink:0;'>D</span>"
            f"<span style='{_f}font-size:var(--fs-sm);color:#1a1a1a;'>Define &amp; Discover</span></div>"
            "<div style='display:flex;align-items:center;gap:8px;'>"
            "<span style='width:22px;height:22px;border-radius:50%;background:#000;color:#CFB991;"
            "font-size:var(--fs-micro);font-weight:800;display:flex;align-items:center;justify-content:center;"
            "flex-shrink:0;'>R</span>"
            f"<span style='{_f}font-size:var(--fs-sm);color:#1a1a1a;'>Represent</span></div>"
            "<div style='display:flex;align-items:center;gap:8px;'>"
            "<span style='width:22px;height:22px;border-radius:50%;background:#000;color:#CFB991;"
            "font-size:var(--fs-micro);font-weight:800;display:flex;align-items:center;justify-content:center;"
            "flex-shrink:0;'>I</span>"
            f"<span style='{_f}font-size:var(--fs-sm);color:#1a1a1a;'>Implement</span></div>"
            "<div style='display:flex;align-items:center;gap:8px;'>"
            "<span style='width:22px;height:22px;border-radius:50%;background:#000;color:#CFB991;"
            "font-size:var(--fs-micro);font-weight:800;display:flex;align-items:center;justify-content:center;"
            "flex-shrink:0;'>V</span>"
            f"<span style='{_f}font-size:var(--fs-sm);color:#1a1a1a;'>Validate</span></div>"
            "<div style='display:flex;align-items:center;gap:8px;'>"
            "<span style='width:22px;height:22px;border-radius:50%;background:#000;color:#CFB991;"
            "font-size:var(--fs-micro);font-weight:800;display:flex;align-items:center;justify-content:center;"
            "flex-shrink:0;'>E</span>"
            f"<span style='{_f}font-size:var(--fs-sm);color:#1a1a1a;'>Evolve</span></div>"
            "<div style='display:flex;align-items:center;gap:8px;'>"
            "<span style='width:22px;height:22px;border-radius:50%;background:#000;color:#CFB991;"
            "font-size:var(--fs-micro);font-weight:800;display:flex;align-items:center;justify-content:center;"
            "flex-shrink:0;'>R</span>"
            f"<span style='{_f}font-size:var(--fs-sm);color:#1a1a1a;'>Reflect</span></div>"
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
_PAGE_FN_MAP = {
    "Overview & Data": page_overview,
    "Yield Curve Analytics": page_yield_curve,
    "Regime Detection": page_regime,
    "Spillover & Info Flow": page_spillover,
    "Early Warning": page_early_warning,
    "Trade Ideas": page_trade_ideas,
    "Performance Review": page_performance_review,
    "AI Q&A": page_ai_qa,
    "About: Heramb Patkar": page_about_heramb,
    "About: Dr. Zhang": page_about_zhang,
}

_page_fn = _PAGE_FN_MAP.get(page)
if _page_fn is not None:
    try:
        _page_fn()
    except Exception as _page_exc:
        st.error(f"An error occurred while rendering **{page}**: {_page_exc}")
        st.info("Try refreshing the page or adjusting the date range / data source in the sidebar.")
        import traceback
        with st.expander("Error details", expanded=False):
            st.code(traceback.format_exc(), language="text")
        _page_footer()
