"""Shared UI helpers and constants for the JGB Repricing Framework dashboard.

All page modules import these functions instead of defining their own.
"""

from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from src.data.config import BOJ_EVENTS


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
    transition=dict(duration=300, easing="cubic-in-out"),
)

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


# ===================================================================
# Chart helpers
# ===================================================================

def _style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    """Apply the institutional plotly template with screener-grade interactions."""
    fig.update_layout(**_PLOTLY_LAYOUT, height=height)

    _has_timeseries = False
    for trace in fig.data:
        if isinstance(trace, go.Scatter) and trace.x is not None and len(trace.x) > 0:
            _sample = trace.x[0] if not hasattr(trace.x, 'iloc') else trace.x.iloc[0]
            if hasattr(_sample, 'year') or (isinstance(_sample, str) and len(_sample) >= 8):
                _has_timeseries = True
                break

    if _has_timeseries:
        fig.update_xaxes(
            rangeselector=_RANGE_SELECTOR,
            rangeslider=dict(visible=True, thickness=0.04, bgcolor="rgba(0,0,0,0.02)",
                             bordercolor="rgba(0,0,0,0.06)", borderwidth=1),
        )
        fig.update_layout(margin=dict(l=48, r=16, t=38, b=8), height=height + 30)

    for i, trace in enumerate(fig.data):
        if isinstance(trace, go.Scatter):
            has_color = getattr(trace.line, "color", None) or getattr(trace.marker, "color", None)
            if not has_color:
                fig.data[i].update(
                    line=dict(color=_PALETTE[i % len(_PALETTE)], width=1.5),
                )

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


# ===================================================================
# Text helpers
# ===================================================================

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
        f"<div style='background:#000000;padding:18px 24px;"
        f"border-top:3px solid #CFB991;'>"
        f"<p style='margin:0 0 2px 0;color:rgba(207,185,145,0.6);font-family:var(--font-sans);"
        f"font-size:var(--fs-tiny);font-weight:700;text-transform:uppercase;letter-spacing:var(--ls-widest);'>"
        f"Verdict</p>"
        f"<p style='margin:0;color:#CFB991;font-family:var(--font-sans);"
        f"font-size:var(--fs-2xl);font-weight:600;line-height:1.55;letter-spacing:var(--ls-snug);'>"
        f"{verdict}</p></div>"
        f"<div style='background:#fafaf8;padding:16px 24px;'>"
        f"<p style='margin:0 0 6px 0;color:#4a4a4a;font-family:var(--font-sans);"
        f"font-size:var(--fs-tiny);font-weight:700;text-transform:uppercase;letter-spacing:var(--ls-wider);'>"
        f"Assessment</p>"
        f"<p style='margin:0;color:#1a1a1a;font-family:var(--font-sans);"
        f"font-size:var(--fs-lg);line-height:1.7;'>{summary}</p></div>"
        f"</div>",
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=86400)
def _footer_logo_b64() -> str:
    """Return base64-encoded reverse Daniels logo for footer embedding."""
    logo_path = Path(__file__).resolve().parent.parent.parent / "assets" / "purdue_daniels_logo_reverse.png"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    return ""


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
        "<div style='background:#000000;padding:44px 0 40px 0;'>"
        "<div style='display:grid;grid-template-columns:1.6fr 1fr 1fr 1fr 1fr;gap:28px;max-width:1280px;margin:0 auto;padding:0 48px;'>"
        "<div>"
        "<a href='https://business.purdue.edu/' target='_blank'>"
        f"<img src='{_footer_logo_b64()}' "
        "alt='Purdue Daniels School of Business' "
        "style='height:40px;margin-bottom:16px;display:block;' /></a>"
        "<p style='font-size:var(--fs-base);color:rgba(255,255,255,0.7);line-height:1.65;margin:0 0 16px 0;max-width:260px;'>"
        "MGMT 69000 &middot; Mastering AI for Finance<br/>"
        "West Lafayette, Indiana</p>"
        f"<p style='font-size:var(--fs-xs);color:rgba(207,185,145,0.6);margin:0;"
        f"font-weight:600;letter-spacing:var(--ls-wide);'>"
        f"Last updated {ts}</p>"
        "</div>"
        f"<div><p style='{_g}'>Navigate</p>"
        "<ul style='list-style:none;padding:0;margin:0;'>"
        f"<li style='margin-bottom:8px;'><a href='?page=overview' target='_self' style='{_w}'>Overview &amp; Data</a></li>"
        f"<li style='margin-bottom:8px;'><a href='?page=yield_curve' target='_self' style='{_w}'>Yield Curve Analytics</a></li>"
        f"<li style='margin-bottom:8px;'><a href='?page=regime' target='_self' style='{_w}'>Regime Detection</a></li>"
        f"<li style='margin-bottom:8px;'><a href='?page=spillover' target='_self' style='{_w}'>Spillover &amp; Info Flow</a></li>"
        f"<li style='margin-bottom:8px;'><a href='?page=equity_spillover' target='_self' style='{_w}'>Equity Spillover</a></li>"
        f"<li style='margin-bottom:8px;'><a href='?page=early_warning' target='_self' style='{_w}'>Early Warning</a></li>"
        f"<li style='margin-bottom:8px;'><a href='?page=trades' target='_self' style='{_w}'>Trade Ideas</a></li>"
        f"<li style='margin-bottom:8px;'><a href='?page=intraday_fx' target='_self' style='{_w}'>Intraday FX Event Study</a></li>"
        f"<li style='margin-bottom:8px;'><a href='?page=performance' target='_self' style='{_w}'>Performance Review</a></li>"
        f"<li style='margin-bottom:8px;'><a href='?page=ai_qa' target='_self' style='{_w}'>AI Q&amp;A</a></li>"
        "</ul></div>"
        f"<div><p style='{_g}'>About</p>"
        "<ul style='list-style:none;padding:0;margin:0;'>"
        f"<li style='margin-bottom:8px;'><a href='?page=about_heramb' target='_self' style='{_w}'>Heramb S. Patkar</a></li>"
        f"<li style='margin-bottom:8px;'><a href='?page=about_zhang' target='_self' style='{_w}'>Dr. Cinder Zhang</a></li>"
        f"<li><a href='https://business.purdue.edu/' target='_blank' style='{_w}'>Daniels School of Business</a></li>"
        "</ul></div>"
        f"<div><p style='{_g}'>Connect</p>"
        "<ul style='list-style:none;padding:0;margin:0;'>"
        f"<li style='margin-bottom:8px;'><a href='https://www.linkedin.com/in/heramb-patkar/' target='_blank' style='{_w}'>LinkedIn: Heramb S. Patkar</a></li>"
        f"<li><a href='https://www.linkedin.com/in/cinder-zhang/' target='_blank' style='{_w}'>LinkedIn: Dr. Cinder Zhang</a></li>"
        "</ul></div>"
        f"<div><p style='{_g}'>Source Code</p>"
        "<ul style='list-style:none;padding:0;margin:0;'>"
        f"<li style='margin-bottom:8px;'><a href='https://github.com/HPATKAR' target='_blank' style='{_w}'>GitHub: Heramb S. Patkar</a></li>"
        f"<li style='margin-bottom:8px;'><a href='https://github.com/CinderZhang' target='_blank' style='{_w}'>GitHub: Dr. Cinder Zhang</a></li>"
        f"<li><a href='https://cinderzhang.github.io/' target='_blank' style='{_w}'>DRIVER Framework</a></li>"
        "</ul></div>"
        "</div></div>"
        "<div style='background:#CFB991;padding:10px 48px;text-align:center;'>"
        f"<p style='font-size:var(--fs-tiny);color:#000000;margin:0;font-weight:600;letter-spacing:var(--ls-wide);"
        f"font-family:var(--font-sans);'>"
        f"&copy; {yr} Purdue University &middot; For educational purposes only &middot; Not investment advice</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )


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


def _about_page_styles():
    """Inject CSS for About pages (hero banner, cards, timelines, etc.)."""
    st.markdown("""<style>
    /* ── Hero banner ───────────────────────────────────── */
    .about-hero {
        background: linear-gradient(135deg, #000000 0%, #1a1a2e 100%);
        border-radius: 16px;
        padding: 2.5rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(207,185,145,0.15);
    }
    .about-hero::before {
        content: '';
        position: absolute;
        top: 0; right: 0;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(207,185,145,0.08) 0%, transparent 70%);
    }
    .about-hero-inner { position: relative; z-index: 1; }
    .about-hero h1 {
        font-family: var(--font-sans, 'DM Sans', sans-serif);
        font-size: var(--fs-hero, 2.0rem);
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.02em;
        line-height: 1.15;
    }
    .about-hero .overline {
        font-family: var(--font-sans, 'DM Sans', sans-serif);
        font-size: var(--fs-base, 0.70rem);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #CFB991;
        margin: 0 0 0.5rem 0;
    }
    .about-hero .subtitle {
        font-family: var(--font-sans, 'DM Sans', sans-serif);
        font-size: var(--fs-2xl, 0.88rem);
        color: rgba(255,255,255,0.7);
        margin: 0 0 0.8rem 0;
        font-weight: 400;
    }
    .about-hero .tagline {
        font-family: var(--font-sans, 'DM Sans', sans-serif);
        font-size: var(--fs-xl, 0.82rem);
        color: rgba(255,255,255,0.55);
        margin: 0 0 1rem 0;
        line-height: 1.6;
    }
    .about-hero .links {
        display: flex;
        gap: 0.8rem;
        flex-wrap: wrap;
    }
    .about-hero .links a {
        font-family: var(--font-sans, 'DM Sans', sans-serif);
        font-size: var(--fs-lg, 0.78rem);
        font-weight: 600;
        color: #CFB991;
        text-decoration: none;
        padding: 0.3rem 0.9rem;
        border: 1px solid rgba(207,185,145,0.35);
        border-radius: 20px;
        transition: all 0.2s ease;
    }
    .about-hero .links a:hover {
        background: rgba(207,185,145,0.15);
        border-color: #CFB991;
    }

    /* ── Cards ─────────────────────────────────────────── */
    .about-card {
        background: #fff;
        border: 1px solid #e8e5e2;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
        transition: all 0.2s ease;
    }
    .about-card:hover {
        border-color: #CFB991;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    .about-card-title {
        font-family: var(--font-sans, 'DM Sans', sans-serif);
        font-size: var(--fs-base, 0.70rem);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #8E6F3E;
        margin: 0 0 0.7rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #f0eeeb;
    }

    /* ── Experience timeline ───────────────────────────── */
    .exp-item {
        border-left: 2px solid #e8e5e2;
        padding-left: 1rem;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        transition: border-color 0.2s ease;
    }
    .exp-item:hover { border-color: #CFB991; }
    .exp-role {
        font-family: var(--font-sans, 'DM Sans', sans-serif);
        font-size: var(--fs-2xl, 0.88rem);
        font-weight: 700;
        color: #1a1a1a;
        margin: 0 0 0.1rem 0;
    }
    .exp-org {
        font-family: var(--font-sans, 'DM Sans', sans-serif);
        font-size: var(--fs-xl, 0.82rem);
        font-weight: 600;
        color: #8E6F3E;
        margin: 0 0 0.15rem 0;
    }
    .exp-meta {
        font-family: var(--font-sans, 'DM Sans', sans-serif);
        font-size: var(--fs-base, 0.70rem);
        color: #888;
        margin: 0 0 0.3rem 0;
    }
    .exp-desc {
        font-family: var(--font-sans, 'DM Sans', sans-serif);
        font-size: var(--fs-lg, 0.78rem);
        color: #444;
        line-height: 1.6;
        margin: 0;
    }

    /* ── Education ─────────────────────────────────────── */
    .edu-item {
        padding: 0.6rem 0;
        border-bottom: 1px solid #f0eeeb;
    }
    .edu-item:last-child { border-bottom: none; }
    .edu-school {
        font-family: var(--font-sans, 'DM Sans', sans-serif);
        font-size: var(--fs-2xl, 0.88rem);
        font-weight: 700;
        color: #1a1a1a;
        margin: 0 0 0.1rem 0;
    }
    .edu-dept {
        font-family: var(--font-sans, 'DM Sans', sans-serif);
        font-size: var(--fs-lg, 0.78rem);
        color: #8E6F3E;
        margin: 0 0 0.1rem 0;
        font-weight: 500;
    }
    .edu-degree {
        font-family: var(--font-sans, 'DM Sans', sans-serif);
        font-size: var(--fs-lg, 0.78rem);
        color: #444;
        margin: 0 0 0.1rem 0;
    }
    .edu-year {
        font-family: var(--font-sans, 'DM Sans', sans-serif);
        font-size: var(--fs-base, 0.70rem);
        color: #999;
        margin: 0;
    }

    /* ── Publication ───────────────────────────────────── */
    .pub-item {
        padding: 0.4rem 0;
        font-size: var(--fs-lg, 0.78rem);
    }
    .pub-title {
        font-family: var(--font-sans, 'DM Sans', sans-serif);
        font-size: var(--fs-xl, 0.82rem);
        font-weight: 600;
        color: #1a1a1a;
        margin: 0 0 0.25rem 0;
        line-height: 1.45;
    }
    .pub-authors {
        font-family: var(--font-sans, 'DM Sans', sans-serif);
        font-size: var(--fs-lg, 0.78rem);
        color: #555;
        margin: 0 0 0.15rem 0;
    }
    .pub-journal {
        font-family: var(--font-sans, 'DM Sans', sans-serif);
        font-size: var(--fs-md, 0.74rem);
        color: #8E6F3E;
        font-style: italic;
        margin: 0 0 0.1rem 0;
    }
    .pub-detail {
        font-family: var(--font-sans, 'DM Sans', sans-serif);
        font-size: var(--fs-base, 0.70rem);
        color: #999;
        margin: 0 0 0.4rem 0;
    }
    .pub-link {
        font-family: var(--font-sans, 'DM Sans', sans-serif);
        font-size: var(--fs-md, 0.74rem);
        font-weight: 600;
        color: #CFB991;
        text-decoration: none;
        border-bottom: 1px solid rgba(207,185,145,0.3);
        transition: border-color 0.2s;
    }
    .pub-link:hover { border-color: #CFB991; }

    /* ── Certifications ────────────────────────────────── */
    .cert-item {
        padding: 0.4rem 0;
        border-bottom: 1px solid #f5f4f2;
    }
    .cert-item:last-child { border-bottom: none; }
    .cert-name {
        font-family: var(--font-sans, 'DM Sans', sans-serif);
        font-size: var(--fs-xl, 0.82rem);
        font-weight: 600;
        color: #1a1a1a;
        margin: 0 0 0.1rem 0;
    }
    .cert-issuer {
        font-family: var(--font-sans, 'DM Sans', sans-serif);
        font-size: var(--fs-base, 0.70rem);
        color: #999;
        margin: 0;
    }

    /* ── Interest tags ─────────────────────────────────── */
    .interest-tag {
        display: inline-block;
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-family: var(--font-sans, 'DM Sans', sans-serif);
        font-size: var(--fs-base, 0.70rem);
        font-weight: 600;
        margin: 0.15rem;
    }
    .interest-gold {
        background: rgba(207,185,145,0.12);
        color: #8E6F3E;
        border: 1px solid rgba(207,185,145,0.25);
    }
    .interest-neutral {
        background: #f5f4f2;
        color: #555;
        border: 1px solid #e8e5e2;
    }

    /* ── Acknowledgments ───────────────────────────────── */
    .ack-text {
        font-family: var(--font-sans, 'DM Sans', sans-serif);
        font-size: var(--fs-lg, 0.78rem);
        color: #555;
        line-height: 1.6;
        margin: 0;
    }
    .ack-text strong { color: #1a1a1a; }

    /* ── Stats row ─────────────────────────────────────── */
    .stat-row {
        display: flex;
        justify-content: space-around;
        padding: 0.6rem 0;
        border-top: 1px solid rgba(207,185,145,0.15);
        margin-top: 0.5rem;
    }
    .stat-item { text-align: center; }
    .stat-num {
        font-size: var(--fs-h1, 1.25rem);
        font-weight: 700;
        color: #CFB991;
        font-family: var(--font-mono, monospace);
        margin: 0;
    }
    .stat-label {
        font-size: var(--fs-xs, 0.60rem);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #888;
        margin: 2px 0 0 0;
        font-family: var(--font-sans, 'DM Sans', sans-serif);
    }
    </style>""", unsafe_allow_html=True)
