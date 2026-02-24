"""About: Dr. Zhang page."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from src.ui.shared import (
    _about_page_styles, _page_footer,
)


def _get_args():
    """Retrieve sidebar args from session state."""
    return st.session_state["_app_args"]


def _get_layout_config():
    return st.session_state["_layout_config"]


def _get_alert_notifier():
    return st.session_state.get("_alert_notifier")


def page_about_zhang():
    _about_page_styles()
    _f = "font-family:var(--font-sans);"

    # ── load profile image as base64 ──
    _img_path = Path(__file__).resolve().parent.parent.parent / "assets" / "zhang_profile.png"
    _img_b64 = ""
    if _img_path.exists():
        _img_b64 = base64.b64encode(_img_path.read_bytes()).decode()

    _photo_html = ""
    if _img_b64:
        _photo_html = (
            "<div class='hero-photo'>"
            f"<img src='data:image/png;base64,{_img_b64}' "
            "alt='Dr. Cinder Zhang' />"
            "</div>"
        )

    # ── hero banner ──
    st.markdown(
        "<div class='about-hero'><div class='about-hero-inner'>"
        f"{_photo_html}"
        "<div class='hero-body'>"
        "<p class='overline'>Course Instructor</p>"
        "<h1>Dr. Cinder Zhang, Ph.D.</h1>"
        "<p class='subtitle'>Finance Faculty &middot; Mitchell E. Daniels, Jr. School of Business</p>"
        "<p class='tagline'>Creator of the DRIVER Framework. Award-winning educator "
        "pioneering AI-integrated finance pedagogy at Purdue University.</p>"
        "<div class='links'>"
        "<a href='https://www.linkedin.com/in/cinder-zhang/' target='_blank'>LinkedIn</a>"
        "<a href='https://github.com/CinderZhang' target='_blank'>GitHub</a>"
        "<a href='https://cinderzhang.github.io/' target='_blank'>DRIVER Framework</a>"
        "</div></div></div></div>",
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


