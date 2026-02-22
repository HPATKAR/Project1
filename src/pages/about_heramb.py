"""About: Heramb Patkar page."""

from __future__ import annotations

import base64
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


def _get_args():
    """Retrieve sidebar args from session state."""
    return st.session_state["_app_args"]


def _get_layout_config():
    return st.session_state["_layout_config"]


def _get_alert_notifier():
    return st.session_state.get("_alert_notifier")


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


