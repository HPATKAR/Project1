"""
JGB Repricing Framework: Streamlit Dashboard

Multi-page dashboard visualising the full JGB repricing pipeline:
data overview, yield curve analytics, regime detection, cross-asset
spillover, and trade ideas.

Launch:  .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import date, datetime

import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `src.*` imports resolve
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# src imports (sidebar configuration)
# ---------------------------------------------------------------------------
from src.data.config import DEFAULT_START, DEFAULT_END
from src.ui.layout_config import LayoutConfig, LayoutManager, render_settings_panel
from src.ui.alert_system import AlertNotifier

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
        font-size: var(--fs-xs) !important;
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
        font-size: var(--fs-base);
        font-weight: 600;
    }
    /* Nav links — pure HTML, full control */
    .sb-nav { margin: 0; padding: 0; }
    .sb-nav-section {
        font-size: var(--fs-micro);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: rgba(207,185,145,0.55);
        padding: 0.6rem 0 0.25rem 0.85rem;
        font-family: var(--font-sans);
        margin: 0;
        user-select: none;
    }
    .sb-nav a {
        display: block;
        padding: 0.38rem 0.7rem 0.38rem 0.85rem;
        font-family: var(--font-sans);
        font-size: var(--fs-sm);
        font-weight: 500;
        color: rgba(255,255,255,0.65);
        text-decoration: none;
        border-left: 2px solid transparent;
        transition: all 0.12s ease;
        letter-spacing: 0.015em;
        line-height: 1.3;
    }
    .sb-nav a:hover {
        color: rgba(255,255,255,0.92);
        background: rgba(255,255,255,0.04);
        border-left-color: rgba(207,185,145,0.35);
    }
    .sb-nav a.active {
        color: #CFB991;
        font-weight: 600;
        background: rgba(207,185,145,0.07);
        border-left-color: #CFB991;
    }
    .sb-nav-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin: 0.35rem 0.8rem;
    }
    /* Hide Streamlit buttons used only for refresh */
    section[data-testid="stSidebar"] .stButton > button {
        font-family: var(--font-sans);
        font-size: var(--fs-xs);
        border-radius: 3px;
        padding: 0.35rem 0.7rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        border: 1px solid rgba(255,255,255,0.08);
        background: transparent;
        color: rgba(255,255,255,0.5) !important;
        transition: all 0.15s ease;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.05);
        border-color: rgba(255,255,255,0.15);
        color: rgba(255,255,255,0.85) !important;
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
        font-size: var(--fs-sm);
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
    "<div style='padding:0.9rem 0.85rem 1.1rem 0.85rem; "
    "border-bottom:1px solid rgba(207,185,145,0.12); margin-bottom:0.5rem;'>"
    # Row 1: Japanese flag + desk label
    "<div style='display:flex;align-items:center;gap:8px;margin-bottom:10px;'>"
    "<svg width='24' height='16' viewBox='0 0 900 600' xmlns='http://www.w3.org/2000/svg' "
    "style='border-radius:2px;box-shadow:0 1px 3px rgba(0,0,0,0.3);flex-shrink:0;'>"
    "<rect width='900' height='600' fill='#fff'/>"
    "<circle cx='450' cy='300' r='180' fill='#BC002D'/>"
    "</svg>"
    "<span style='font-size:var(--fs-sm);font-weight:800;text-transform:uppercase;"
    "letter-spacing:0.16em;color:rgba(207,185,145,0.82);font-family:var(--font-sans);'>"
    "Rates Strategy Desk</span></div>"
    # Row 2: Title
    "<div style='font-size:var(--fs-hero);font-weight:900;color:#fff;"
    "letter-spacing:-0.02em;line-height:1.05;font-family:var(--font-sans);'>"
    "JGB Repricing</div>"
    # Row 3: Subtitle
    "<div style='font-size:var(--fs-base);font-weight:700;color:rgba(255,255,255,0.72);"
    "margin-top:4px;letter-spacing:0.08em;text-transform:uppercase;"
    "font-family:var(--font-sans);'>Quantitative Framework</div>"
    # Row 4: Institution
    "<div style='font-size:var(--fs-xs);font-weight:700;color:rgba(207,185,145,0.78);"
    "margin-top:5px;letter-spacing:0.12em;text-transform:uppercase;"
    "font-family:var(--font-sans);'>Purdue Daniels School of Business</div>"
    "</div>",
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
    "intraday_fx": "Intraday FX Event Study",
    "early_warning": "Early Warning",
    "performance": "Performance Review",
    "ai_qa": "AI Q&A",
    "equity_spillover": "Equity Spillover",
}
_qp = st.query_params.get("page", "")
if _qp in _QP_MAP:
    st.session_state.current_page = _QP_MAP[_qp]
    # Clear the query param so it doesn't override sidebar button clicks on rerun
    st.query_params.clear()
elif "current_page" not in st.session_state:
    st.session_state.current_page = "Overview & Data"

_NAV_SECTIONS = [
    ("Analytics", [
        ("Overview & Data", "overview"),
        ("Yield Curve Analytics", "yield_curve"),
        ("Regime Detection", "regime"),
        ("Spillover & Info Flow", "spillover"),
        ("Equity Spillover", "equity_spillover"),
        ("Early Warning", "early_warning"),
    ]),
    ("Strategy", [
        ("Trade Ideas", "trades"),
        ("Intraday FX Event Study", "intraday_fx"),
    ]),
    ("Diagnostics", [
        ("Performance Review", "performance"),
        ("AI Q&A", "ai_qa"),
    ]),
    ("Reference", [
        ("About: Heramb Patkar", "about_heramb"),
        ("About: Dr. Zhang", "about_zhang"),
    ]),
]

# Build the entire navigation as one HTML block — no Streamlit buttons
_nav_html_parts = ["<nav class='sb-nav'>"]
for _si, (_section_label, _section_items) in enumerate(_NAV_SECTIONS):
    if _si > 0:
        _nav_html_parts.append("<hr class='sb-nav-divider'/>")
    _nav_html_parts.append(f"<p class='sb-nav-section'>{_section_label}</p>")
    for _label, _key in _section_items:
        _cls = " class='active'" if st.session_state.current_page == _label else ""
        _nav_html_parts.append(
            f"<a href='?page={_key}' target='_self'{_cls}>{_label}</a>"
        )
_nav_html_parts.append("</nav>")
st.sidebar.markdown("".join(_nav_html_parts), unsafe_allow_html=True)

page = st.session_state.current_page

st.sidebar.markdown(
    "<hr class='sb-nav-divider' style='margin:0.5rem 0.4rem;'/>"
    "<p class='sb-nav-section' style='padding-left:0.4rem;'>Configuration</p>",
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
    "<span style='font-size:var(--fs-tiny);font-weight:700;text-transform:uppercase;"
    "letter-spacing:0.14em;color:rgba(255,255,255,0.4);"
    f"font-family:var(--font-sans);'>Last Update: {datetime.now().strftime('%H:%M:%S')}</span></div>",
    unsafe_allow_html=True,
)
if st.sidebar.button("Refresh Data", key="manual_refresh", use_container_width=True, type="secondary"):
    st.cache_data.clear()
    st.rerun()


# ---------------------------------------------------------------------------
# Store sidebar globals in session state for page module access
# ---------------------------------------------------------------------------
st.session_state["_app_args"] = (use_simulated, str(start_date), str(end_date), fred_api_key or None)
st.session_state["_layout_config"] = _layout_config
st.session_state["_alert_notifier"] = _alert_notifier

# ---------------------------------------------------------------------------
# Page module imports
# ---------------------------------------------------------------------------
from src.pages.overview import page_overview
from src.pages.yield_curve import page_yield_curve, _run_pca, _run_ns, _run_liquidity
from src.pages.regime import (
    page_regime, _run_ensemble, _run_markov, _run_hmm,
    _run_entropy, _run_garch, _run_breaks,
)
from src.pages.spillover import (
    page_spillover, _run_granger, _run_te, _run_spillover,
    _run_dcc, _run_te_pca, _run_carry,
)
from src.pages.early_warning import page_early_warning, _run_warning_score, _run_ml_predictor
from src.pages.trade_ideas import page_trade_ideas
from src.pages.ai_qa import page_ai_qa
from src.pages.performance_review import page_performance_review
from src.pages.about_heramb import page_about_heramb
from src.pages.about_zhang import page_about_zhang
from src.pages.intraday_fx import page_intraday_fx
from src.pages.equity_spillover import page_equity_spillover
from src.pages._data import load_unified
from src.ui.shared import _page_footer

# ===================================================================
# Cache pre-warming — run ALL heavy computations once on startup
# so page switches are instant (results served from st.cache_data).
# ===================================================================
_args = (use_simulated, str(start_date), str(end_date), fred_api_key or None)

if "cache_warmed" not in st.session_state or st.session_state.get("_cache_key") != _args:
    with st.spinner("Loading analytics engine — first load may take a moment..."):
        try:
            # Tier 1: data layer (everything depends on this)
            load_unified(*_args)

            # Tier 2: independent model runs (parallel-safe under cache)
            _run_pca(*_args)
            _run_ns(*_args)
            _run_liquidity(*_args)
            _run_markov(*_args)
            _run_hmm(*_args)
            _run_entropy(*_args)
            _run_garch(*_args)
            _run_breaks(*_args)

            # Tier 3: depends on Tier 2
            _run_ensemble(*_args)
            _run_granger(*_args)
            _run_te(*_args)
            _run_spillover(*_args)
            _run_dcc(*_args)
            _run_te_pca(*_args)
            _run_carry(*_args)

            # Tier 4: depends on Tier 3
            _run_warning_score(*_args, _layout_config.entropy_window)
            _run_ml_predictor(*_args, _layout_config.entropy_window)
        except Exception:
            pass  # individual pages handle their own errors gracefully
    st.session_state["cache_warmed"] = True
    st.session_state["_cache_key"] = _args


# ===================================================================
# Router
# ===================================================================
_PAGE_FN_MAP = {
    "Overview & Data": page_overview,
    "Yield Curve Analytics": page_yield_curve,
    "Regime Detection": page_regime,
    "Spillover & Info Flow": page_spillover,
    "Equity Spillover": page_equity_spillover,
    "Early Warning": page_early_warning,
    "Trade Ideas": page_trade_ideas,
    "Intraday FX Event Study": page_intraday_fx,
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
