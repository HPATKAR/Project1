"""
PDF and CSV export for JGB Repricing Framework reports.

Institutional buy/sell-side format:
  - Clean serif/sans typography, no decorative elements
  - Left sidebar strip on trade card pages with key metrics
  - Formal tables with minimal horizontal rules
  - Full-page disclaimer
"""
from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

from src.reporting.metrics_tracker import AccuracyMetrics

# Colour palette — institutional monochrome with minimal accent
_BLACK = (0, 0, 0)
_DARK_GREY = (50, 50, 50)
_MID_GREY = (120, 120, 120)
_LIGHT_GREY = (200, 200, 200)
_SIDEBAR_BG = (243, 243, 240)  # very light warm grey for sidebar strip
_ACCENT = (142, 111, 62)       # muted gold for sparing use
_ALERT_BG = (255, 242, 242)    # failure scenario background

_LOGO_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "purdue_daniels_logo.png"

# Layout constants
_LEFT_COL_W = 48   # mm — sidebar strip width
_LEFT_COL_X = 10   # mm — sidebar X start
_RIGHT_COL_X = 62  # mm — content X start
_RIGHT_COL_W = 138 # mm — content width
_PAGE_W = 210
_MARGIN = 10


class JGBReportPDF:
    """Generate institutional-grade PDF reports for the JGB Repricing Framework."""

    def __init__(self):
        if FPDF is None:
            raise ImportError("fpdf2 is required for PDF export. Install with: pip install fpdf2")
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=20)
        self.pdf.set_left_margin(_MARGIN)
        self.pdf.set_right_margin(_MARGIN)

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _safe(text: str) -> str:
        """Sanitize text for Helvetica (Latin-1). Replace unsupported chars."""
        return (
            text.replace("\u2014", "-")   # em dash
                .replace("\u2013", "-")   # en dash
                .replace("\u2018", "'")   # left single quote
                .replace("\u2019", "'")   # right single quote
                .replace("\u201c", '"')   # left double quote
                .replace("\u201d", '"')   # right double quote
                .replace("\u2026", "...")  # ellipsis
                .replace("\u2022", "*")   # bullet
                .replace("\u00b7", "*")   # middle dot
                .encode("latin-1", errors="replace").decode("latin-1")
        )

    def _hairline(self, y: float | None = None, x1: float = _MARGIN, x2: float = _PAGE_W - _MARGIN) -> None:
        """Draw a thin grey horizontal rule."""
        if y is None:
            y = self.pdf.get_y()
        self.pdf.set_draw_color(*_LIGHT_GREY)
        self.pdf.set_line_width(0.3)
        self.pdf.line(x1, y, x2, y)
        self.pdf.set_draw_color(*_BLACK)
        self.pdf.set_line_width(0.2)

    def _add_page_footer(self) -> None:
        """Institutional footer: analyst name, date, page number."""
        self.pdf.set_auto_page_break(auto=False)
        self.pdf.set_y(-16)
        self._hairline()
        self.pdf.ln(2)
        self.pdf.set_font("Helvetica", "", 6.5)
        self.pdf.set_text_color(*_MID_GREY)
        self.pdf.cell(0, 4, "Heramb S. Patkar, MSF  |  Purdue University, Daniels School of Business  |  MGMT 69000", align="L")
        self.pdf.cell(0, 4, f"{datetime.now():%Y-%m-%d %H:%M}  |  Page {self.pdf.page_no()}", align="R", ln=True)
        self.pdf.set_text_color(*_BLACK)
        self.pdf.set_auto_page_break(auto=True, margin=20)

    def _section_header(self, text: str) -> None:
        """Formal uppercase section header."""
        self.pdf.set_font("Helvetica", "B", 9)
        self.pdf.set_text_color(*_DARK_GREY)
        self.pdf.cell(0, 6, text.upper(), ln=True)
        self.pdf.set_text_color(*_BLACK)
        self.pdf.ln(1.5)

    def _body_text(self, text: str, size: float = 8.5) -> None:
        """Standard body paragraph."""
        self.pdf.set_font("Helvetica", "", size)
        self.pdf.set_text_color(*_DARK_GREY)
        self.pdf.multi_cell(0, 4.5, self._safe(text))
        self.pdf.set_text_color(*_BLACK)
        self.pdf.ln(2)

    # ── title page ──────────────────────────────────────────────────────
    def add_title_page(
        self,
        title: str = "JGB Repricing Framework",
        subtitle: str = "",
    ) -> None:
        self.pdf.add_page()

        # Top rule
        self.pdf.set_draw_color(*_BLACK)
        self.pdf.set_line_width(1.2)
        self.pdf.line(_MARGIN, 12, _PAGE_W - _MARGIN, 12)
        self.pdf.set_line_width(0.2)

        # Logo
        self.pdf.set_y(20)
        logo = str(_LOGO_PATH) if _LOGO_PATH.exists() else None
        if logo:
            try:
                self.pdf.image(logo, x=60, w=90)
                self.pdf.ln(10)
            except Exception:
                pass

        self._hairline()
        self.pdf.ln(18)

        # Title
        self.pdf.set_font("Helvetica", "B", 24)
        self.pdf.set_text_color(*_BLACK)
        self.pdf.cell(0, 12, title, ln=True, align="C")
        self.pdf.ln(3)

        if subtitle:
            self.pdf.set_font("Helvetica", "", 12)
            self.pdf.set_text_color(*_MID_GREY)
            self.pdf.cell(0, 8, subtitle, ln=True, align="C")
            self.pdf.set_text_color(*_BLACK)
            self.pdf.ln(6)

        self._hairline(x1=70, x2=140)
        self.pdf.ln(20)

        # Metadata
        self.pdf.set_font("Helvetica", "", 10)
        self.pdf.set_text_color(*_DARK_GREY)
        self.pdf.cell(0, 7, f"Report Date: {datetime.now().strftime('%d %B %Y')}", ln=True, align="C")
        self.pdf.ln(4)
        self.pdf.set_font("Helvetica", "B", 11)
        self.pdf.set_text_color(*_BLACK)
        self.pdf.cell(0, 7, "Heramb S. Patkar, MSF", ln=True, align="C")
        self.pdf.set_font("Helvetica", "", 10)
        self.pdf.set_text_color(*_DARK_GREY)
        self.pdf.cell(0, 7, "Purdue University, Daniels School of Business", ln=True, align="C")
        self.pdf.cell(0, 7, "MGMT 69000 - Mastering AI for Finance", ln=True, align="C")
        self.pdf.set_text_color(*_BLACK)
        self.pdf.ln(16)

        # Confidentiality notice
        self.pdf.set_font("Helvetica", "I", 7.5)
        self.pdf.set_text_color(*_MID_GREY)
        self.pdf.cell(0, 5, "For educational and research purposes only. Not investment advice.", ln=True, align="C")
        self.pdf.set_text_color(*_BLACK)

        self._add_page_footer()

    def add_metrics_summary(self, metrics: AccuracyMetrics) -> None:
        self.pdf.add_page()
        self.pdf.set_y(14)
        self.pdf.set_font("Helvetica", "B", 14)
        self.pdf.cell(0, 10, "Performance Metrics", ln=True)
        self._hairline()
        self.pdf.ln(6)

        rows = [
            ("Prediction Accuracy", f"{metrics.prediction_accuracy:.1%}"),
            ("Average Lead Time", f"{metrics.average_lead_time:.1f} days"),
            ("Precision", f"{metrics.precision:.1%}"),
            ("Recall", f"{metrics.recall:.1%}"),
            ("False Positive Rate", f"{metrics.false_positive_rate:.1%}"),
            ("Total Predictions", str(metrics.total_predictions)),
        ]

        # Clean table with alternating row shading
        self.pdf.set_font("Helvetica", "B", 9)
        self.pdf.set_fill_color(*_SIDEBAR_BG)
        self.pdf.cell(95, 7, "  Metric", border="B", align="L")
        self.pdf.cell(85, 7, "Value", border="B", align="R", ln=True)

        self.pdf.set_font("Helvetica", "", 9)
        for i, (label, value) in enumerate(rows):
            fill = i % 2 == 0
            self.pdf.cell(95, 7, f"  {label}", fill=fill)
            self.pdf.cell(85, 7, value, fill=fill, align="R", ln=True)

        self._add_page_footer()

    def add_suggestions(self, suggestions: List[str]) -> None:
        self.pdf.add_page()
        self.pdf.set_y(14)
        self.pdf.set_font("Helvetica", "B", 14)
        self.pdf.cell(0, 10, "Improvement Recommendations", ln=True)
        self._hairline()
        self.pdf.ln(6)

        self.pdf.set_font("Helvetica", "", 9)
        for i, suggestion in enumerate(suggestions, 1):
            self.pdf.multi_cell(0, 5, f"{i}.  {suggestion}")
            self.pdf.ln(2)
        self._add_page_footer()

    def add_chart_image(self, image_path: str, title: str = "") -> None:
        self.pdf.add_page()
        self.pdf.set_y(14)
        if title:
            self.pdf.set_font("Helvetica", "B", 12)
            self.pdf.cell(0, 8, title, ln=True)
            self._hairline()
            self.pdf.ln(4)
        try:
            self.pdf.image(image_path, x=_MARGIN, w=_PAGE_W - 2 * _MARGIN)
        except Exception:
            self.pdf.set_font("Helvetica", "I", 9)
            self.pdf.cell(0, 8, f"[Chart image not available: {image_path}]", ln=True)
        self._add_page_footer()

    def add_data_table(
        self,
        df: pd.DataFrame,
        title: str = "",
        max_rows: int = 50,
    ) -> None:
        self.pdf.add_page()
        self.pdf.set_y(14)
        if title:
            self.pdf.set_font("Helvetica", "B", 12)
            self.pdf.cell(0, 8, title, ln=True)
            self._hairline()
            self.pdf.ln(4)

        if df.empty:
            self.pdf.set_font("Helvetica", "I", 9)
            self.pdf.cell(0, 7, "No data available.", ln=True)
            return

        display_df = df.head(max_rows)
        n_cols = min(len(display_df.columns), 6)
        col_width = (_PAGE_W - 2 * _MARGIN) / n_cols

        self.pdf.set_font("Helvetica", "B", 7.5)
        for col in display_df.columns[:n_cols]:
            self.pdf.cell(col_width, 6, str(col)[:22], border="B", align="C")
        self.pdf.ln()

        self.pdf.set_font("Helvetica", "", 7)
        self.pdf.set_fill_color(*_SIDEBAR_BG)
        for idx, (_, row) in enumerate(display_df.iterrows()):
            fill = idx % 2 == 0
            for col in display_df.columns[:n_cols]:
                val = str(row[col])[:22]
                self.pdf.cell(col_width, 5.5, val, fill=fill)
            self.pdf.ln()
        self._add_page_footer()

    # ── trade ideas (institutional equity-research format) ───────────
    def add_trade_ideas(self, cards: list, regime_state: dict | None = None) -> None:
        """Add Trade Ideas section modelled on JPM / GS equity research first page.

        Layout mirrors institutional initiation notes:
          - Header bar: firm/course left, report type + date right
          - Left ~62%: bold title, subtitle, dense thesis bullets, regime context
          - Right ~38%: coloured recommendation box, key data table, analyst info
          - Bottom full-width: trade summary table
          - Subsequent pages: dense full-width individual trade cards
        """
        import numpy as np

        if not cards:
            return

        sorted_cards = sorted(cards, key=lambda c: -c.conviction)
        n_high = sum(1 for c in cards if c.conviction >= 0.7)
        n_med = sum(1 for c in cards if 0.4 <= c.conviction < 0.7)
        n_low = sum(1 for c in cards if c.conviction < 0.4)
        categories = sorted(set(c.category for c in cards))
        top = sorted_cards[0] if sorted_cards else None
        rp = regime_state.get("regime_prob", 0.5) if regime_state else 0.5
        regime_word = "REPRICING" if rp > 0.5 else "SUPPRESSED"

        # Accent colour for the recommendation box (muted gold)
        _REC_BG = (207, 185, 145)
        _REC_FG = (0, 0, 0)

        # Column geometry
        _sb_x = 134          # right panel X start
        _sb_w = _PAGE_W - _MARGIN - _sb_x  # right panel width (~66mm)
        _cw = _sb_x - _MARGIN - 3          # left content width (~121mm)

        self.pdf.add_page()

        # ══════════════════════════════════════════════════════════════
        #  HEADER BAR  (like "J.P.Morgan CAZENOVE" ... "Europe Equity Research")
        # ══════════════════════════════════════════════════════════════
        self.pdf.set_fill_color(*_BLACK)
        self.pdf.rect(_MARGIN, 10, _PAGE_W - 2 * _MARGIN, 9, "F")
        self.pdf.set_xy(_MARGIN + 2, 10.5)
        self.pdf.set_font("Helvetica", "B", 8)
        self.pdf.set_text_color(255, 255, 255)
        self.pdf.cell(90, 8, "Purdue Daniels School of Business")
        self.pdf.set_font("Helvetica", "", 7)
        self.pdf.cell(_PAGE_W - 2 * _MARGIN - 94, 8, f"JGB Rates Research  |  {datetime.now():%d %B %Y}", align="R")
        self.pdf.set_text_color(*_BLACK)

        # ══════════════════════════════════════════════════════════════
        #  RIGHT PANEL — Recommendation box + key data + analyst info
        # ══════════════════════════════════════════════════════════════
        # -- Recommendation box (coloured, prominent) --
        _box_y = 22
        self.pdf.set_fill_color(*_REC_BG)
        self.pdf.rect(_sb_x, _box_y, _sb_w, 26, "F")
        self.pdf.set_xy(_sb_x + 2, _box_y + 1)
        self.pdf.set_font("Helvetica", "", 6.5)
        self.pdf.set_text_color(*_REC_FG)
        self.pdf.cell(_sb_w - 4, 4, "Regime Assessment", align="C")
        self.pdf.set_xy(_sb_x + 2, _box_y + 5.5)
        self.pdf.set_font("Helvetica", "B", 16)
        self.pdf.cell(_sb_w - 4, 9, regime_word, align="C")
        self.pdf.set_xy(_sb_x + 2, _box_y + 16)
        self.pdf.set_font("Helvetica", "", 7.5)
        self.pdf.cell(_sb_w - 4, 4, f"Ensemble Prob: {rp:.1%}", align="C")
        self.pdf.set_xy(_sb_x + 2, _box_y + 21)
        self.pdf.set_font("Helvetica", "", 6.5)
        conv_word = "HIGH" if n_high > n_med else "MODERATE" if n_med >= n_low else "LOW"
        self.pdf.cell(_sb_w - 4, 4, f"Conviction: {conv_word}", align="C")

        # -- Key data table (like "Selected Financial Data") --
        _kd_y = _box_y + 29
        self.pdf.set_xy(_sb_x, _kd_y)
        self.pdf.set_font("Helvetica", "B", 7)
        self.pdf.set_text_color(*_BLACK)
        self.pdf.set_fill_color(*_REC_BG)
        self.pdf.cell(_sb_w, 5.5, "  Key Data", fill=True, ln=True)

        _kd_rows = [
            ("Total Trades", str(len(cards))),
            ("High Conviction", str(n_high)),
            ("Medium Conviction", str(n_med)),
            ("Low Conviction", str(n_low)),
            ("Categories", str(len(categories))),
            ("Regime Prob.", f"{rp:.1%}"),
        ]
        if top:
            _kd_rows.append(("Lead Trade", self._safe(top.name[:22])))
            _kd_rows.append(("Lead Direction", top.direction.upper()))
            _kd_rows.append(("Lead Conviction", f"{top.conviction:.0%}"))

        self.pdf.set_fill_color(*_SIDEBAR_BG)
        for i, (label, value) in enumerate(_kd_rows):
            self.pdf.set_x(_sb_x)
            fill = i % 2 == 0
            self.pdf.set_font("Helvetica", "", 6.5)
            self.pdf.set_text_color(*_DARK_GREY)
            self.pdf.cell(_sb_w - 18, 4.5, f"  {label}", fill=fill)
            self.pdf.set_font("Helvetica", "B", 6.5)
            self.pdf.set_text_color(*_BLACK)
            self.pdf.cell(18, 4.5, value, fill=fill, align="R", ln=True)

        # Hairline under key data
        _kd_end = self.pdf.get_y()
        self.pdf.set_draw_color(*_LIGHT_GREY)
        self.pdf.set_line_width(0.3)
        self.pdf.line(_sb_x, _kd_end, _sb_x + _sb_w, _kd_end)

        # -- Category breakdown --
        self.pdf.set_xy(_sb_x, _kd_end + 3)
        self.pdf.set_font("Helvetica", "B", 6.5)
        self.pdf.set_text_color(*_DARK_GREY)
        self.pdf.cell(_sb_w, 4, "  Categories", ln=True)
        for cat in categories:
            self.pdf.set_x(_sb_x)
            cat_count = sum(1 for c in cards if c.category == cat)
            self.pdf.set_font("Helvetica", "", 6.5)
            self.pdf.set_text_color(*_DARK_GREY)
            self.pdf.cell(_sb_w - 14, 4, f"  {cat.replace('_', ' ').title()}")
            self.pdf.set_font("Helvetica", "B", 6.5)
            self.pdf.set_text_color(*_BLACK)
            self.pdf.cell(14, 4, str(cat_count), align="R", ln=True)

        # Hairline
        _cat_end = self.pdf.get_y()
        self.pdf.line(_sb_x, _cat_end, _sb_x + _sb_w, _cat_end)

        # -- Analyst info --
        self.pdf.set_xy(_sb_x, _cat_end + 3)
        self.pdf.set_font("Helvetica", "B", 6.5)
        self.pdf.set_text_color(*_DARK_GREY)
        self.pdf.cell(_sb_w, 4, "  Analyst", ln=True)
        self.pdf.set_x(_sb_x)
        self.pdf.set_font("Helvetica", "", 6.5)
        self.pdf.set_text_color(*_BLACK)
        self.pdf.cell(_sb_w, 3.5, "  Heramb S. Patkar, MSF", ln=True)
        self.pdf.set_x(_sb_x)
        self.pdf.set_text_color(*_MID_GREY)
        self.pdf.cell(_sb_w, 3.5, "  Purdue Daniels School of Business", ln=True)
        self.pdf.set_x(_sb_x)
        self.pdf.cell(_sb_w, 3.5, "  MGMT 69000-119", ln=True)
        self.pdf.set_x(_sb_x)
        self.pdf.cell(_sb_w, 3.5, "  github.com/HPATKAR", ln=True)

        self.pdf.set_draw_color(*_BLACK)
        self.pdf.set_line_width(0.2)
        self.pdf.set_text_color(*_BLACK)

        # ══════════════════════════════════════════════════════════════
        #  LEFT BODY — Title, subtitle, thesis bullets, regime context
        # ══════════════════════════════════════════════════════════════
        self.pdf.set_xy(_MARGIN, 22)
        self.pdf.set_font("Helvetica", "B", 18)
        self.pdf.cell(_cw, 10, "JGB Trade Ideas")
        self.pdf.ln(10)

        # Subtitle / tagline
        self.pdf.set_x(_MARGIN)
        self.pdf.set_font("Helvetica", "", 9)
        self.pdf.set_text_color(*_DARK_GREY)
        subtitle = (
            f"Regime-conditional strategies under {regime_word.lower()} "
            f"with ensemble probability {rp:.0%}"
        )
        self.pdf.multi_cell(_cw, 4.5, self._safe(subtitle))
        self.pdf.set_text_color(*_BLACK)
        self.pdf.ln(3)

        # Dense thesis paragraph
        self.pdf.set_x(_MARGIN)
        self.pdf.set_font("Helvetica", "", 8)
        thesis_text = (
            f"The four-model ensemble regime detector (Markov-Switching, HMM, Permutation "
            f"Entropy, GARCH+PELT) currently signals a {rp:.1%} probability of market-driven "
            f"repricing. The framework has generated {len(cards)} trade "
            f"idea{'s' if len(cards) != 1 else ''} spanning "
            f"{', '.join(c.replace('_', ' ') for c in categories)}. "
            f"Distribution: {n_high} high, {n_med} medium, {n_low} low conviction."
        )
        self.pdf.multi_cell(_cw, 4, self._safe(thesis_text))
        self.pdf.ln(3)

        # Bullet points — compelling reasons (like the Walmart template)
        bullets = []
        if top:
            bullets.append(
                f"Lead trade: {top.name} ({top.direction.upper()}, {top.conviction:.0%}). "
                f"Edge: {top.edge_source}."
            )
        if rp > 0.7:
            bullets.append(
                "Strong repricing signal: directional short-JGB and vol-long trades favoured. "
                "Carry unwind risk elevated."
            )
        elif rp > 0.5:
            bullets.append(
                "Transitional regime: relative value and spread trades may outperform directional bets."
            )
        else:
            bullets.append(
                "Suppressed regime: carry and yield-enhancement favoured. Vol selling can capture "
                "range-bound premium."
            )
        if n_high > 0:
            high_names = [c.name for c in sorted_cards if c.conviction >= 0.7][:3]
            bullets.append(
                f"{n_high} high-conviction trade{'s' if n_high != 1 else ''}: "
                f"{', '.join(high_names)}."
            )
        if top:
            bullets.append(
                f"Entry: {top.entry_signal}. Exit: {top.exit_signal}."
            )
            bullets.append(
                f"Failure scenario: {top.failure_scenario[:120]}."
            )

        self.pdf.set_font("Helvetica", "", 7.5)
        for bullet in bullets:
            self.pdf.set_x(_MARGIN)
            # Bullet character
            self.pdf.set_font("Helvetica", "B", 7.5)
            self.pdf.cell(4, 4, "*")
            self.pdf.set_font("Helvetica", "", 7.5)
            self.pdf.multi_cell(_cw - 4, 4, self._safe(bullet))
            self.pdf.ln(1)

        # Regime context section
        self.pdf.ln(2)
        self.pdf.set_x(_MARGIN)
        self.pdf.set_font("Helvetica", "B", 8)
        self.pdf.set_text_color(*_DARK_GREY)
        self.pdf.cell(_cw, 5, "Market Regime Context", ln=True)
        self.pdf.set_text_color(*_BLACK)
        self.pdf.set_x(_MARGIN)
        self.pdf.set_font("Helvetica", "", 7.5)
        if regime_state:
            ctx_text = f"Ensemble: {rp:.1%} ({regime_word}). "
            ms = regime_state.get("markov_prob")
            hmm = regime_state.get("hmm_prob")
            ent = regime_state.get("entropy_prob")
            garch = regime_state.get("garch_prob")
            parts = []
            if ms is not None:
                parts.append(f"Markov: {ms:.0%}")
            if hmm is not None:
                parts.append(f"HMM: {hmm:.0%}")
            if ent is not None:
                parts.append(f"Entropy: {ent:.0%}")
            if garch is not None:
                parts.append(f"GARCH: {garch:.0%}")
            if parts:
                ctx_text += "Components: " + ", ".join(parts) + "."
            self.pdf.multi_cell(_cw, 3.5, self._safe(ctx_text))
        self.pdf.ln(1)

        # ══════════════════════════════════════════════════════════════
        #  BOTTOM — Full-width trade summary table
        # ══════════════════════════════════════════════════════════════
        _tbl_y = max(self.pdf.get_y(), _cat_end + 20) + 2
        self.pdf.set_xy(_MARGIN, _tbl_y)

        # Table header bar
        self.pdf.set_fill_color(*_BLACK)
        fw = _PAGE_W - 2 * _MARGIN
        tbl_w = [58, 20, 20, 52, 18, 22]
        headers = ["Trade", "Dir.", "Conv.", "Instruments", "Cat.", "Sizing"]
        self.pdf.set_font("Helvetica", "B", 7)
        self.pdf.set_text_color(255, 255, 255)
        for w, h in zip(tbl_w, headers):
            self.pdf.cell(w, 5.5, f"  {h}", fill=True)
        self.pdf.ln()
        self.pdf.set_text_color(*_BLACK)

        # Table rows with alternating shading
        self.pdf.set_fill_color(*_SIDEBAR_BG)
        self.pdf.set_font("Helvetica", "", 7)
        for i, card in enumerate(sorted_cards):
            fill = i % 2 == 0
            self.pdf.set_x(_MARGIN)
            self.pdf.cell(tbl_w[0], 5, self._safe(f"  {card.name[:30]}"), fill=fill)
            self.pdf.cell(tbl_w[1], 5, card.direction.upper(), fill=fill, align="C")
            self.pdf.cell(tbl_w[2], 5, f"{card.conviction:.0%}", fill=fill, align="C")
            self.pdf.cell(tbl_w[3], 5, self._safe(", ".join(card.instruments[:2])[:28]), fill=fill)
            self.pdf.cell(tbl_w[4], 5, self._safe(card.category[:8].title()), fill=fill, align="C")
            self.pdf.cell(tbl_w[5], 5, self._safe(card.sizing_method[:12]), fill=fill, align="C")
            self.pdf.ln()

        # Source line
        self.pdf.ln(2)
        self.pdf.set_font("Helvetica", "I", 6)
        self.pdf.set_text_color(*_MID_GREY)
        self.pdf.cell(0, 3.5, "Source: JGB Repricing Framework ensemble regime detector. Heramb S. Patkar.", ln=True)
        self.pdf.set_text_color(*_BLACK)

        self._add_page_footer()

        # --- Individual trade card pages (dense full-width, no sidebar) ---
        for card in sorted_cards:
            self._add_trade_card_page(card)

    def _add_trade_card_page(self, card) -> None:
        """Render a single trade card as a dense full-width page."""
        import numpy as np

        self.pdf.add_page()
        self.pdf.set_y(14)
        self.pdf.set_x(_MARGIN)
        meta = card.metadata or {}
        fw = _PAGE_W - 2 * _MARGIN  # full width

        # Title line
        dir_label = "LONG" if card.direction == "long" else "SHORT"
        self.pdf.set_font("Helvetica", "B", 12)
        self.pdf.set_x(_MARGIN)
        self.pdf.cell(fw, 8, self._safe(f"{dir_label}  |  {card.name}  |  {card.conviction:.0%}"), ln=True)
        self._hairline()
        self.pdf.ln(4)

        # Trade thesis
        self.pdf.set_x(_MARGIN)
        self._section_header("Trade Thesis")
        conv_word = "high" if card.conviction >= 0.7 else "moderate" if card.conviction >= 0.4 else "low"
        thesis = (
            f"{conv_word.title()}-conviction {card.direction} trade on "
            f"{', '.join(card.instruments[:3])} ({card.category.replace('_', ' ')}). "
            f"Edge: {card.edge_source}. Regime condition: {card.regime_condition}."
        )
        self.pdf.set_x(_MARGIN)
        self._body_text(thesis, size=8)

        # Specification table
        self.pdf.set_x(_MARGIN)
        self._section_header("Trade Specification")
        fields = [
            ("Instruments", ", ".join(card.instruments)),
            ("Regime Condition", card.regime_condition),
            ("Edge Source", card.edge_source),
            ("Entry Signal", card.entry_signal),
            ("Exit Signal", card.exit_signal),
            ("Sizing", card.sizing_method),
        ]
        self.pdf.set_fill_color(*_SIDEBAR_BG)
        for i, (label, value) in enumerate(fields):
            fill = i % 2 == 0
            self.pdf.set_x(_MARGIN)
            self.pdf.set_font("Helvetica", "B", 7.5)
            self.pdf.cell(32, 5, f"  {label}", fill=fill)
            self.pdf.set_font("Helvetica", "", 7.5)
            self.pdf.cell(fw - 32, 5, self._safe(value[:140]), fill=fill, ln=True)
        self.pdf.ln(3)

        # Key levels (inline table if present)
        levels = {}
        for key in ["jp10_level", "target_yield", "stop_yield", "usdjpy_spot",
                     "target", "stop", "put_strike", "straddle_strike",
                     "payer_strike", "call_strike", "receiver_strike",
                     "atm_strike", "otm_strike", "breakeven",
                     "spread_bps", "target_spread_bps"]:
            if key in meta and meta[key] is not None:
                levels[key.replace("_", " ").title()] = meta[key]

        if levels:
            self.pdf.set_x(_MARGIN)
            self._section_header("Key Levels")
            self.pdf.set_fill_color(*_SIDEBAR_BG)
            col_count = min(len(levels), 4)
            lev_w = fw / col_count
            items = list(levels.items())
            # Row of labels
            self.pdf.set_x(_MARGIN)
            self.pdf.set_font("Helvetica", "", 6.5)
            self.pdf.set_text_color(*_MID_GREY)
            for k, _ in items[:col_count]:
                self.pdf.cell(lev_w, 4, k, align="C")
            self.pdf.ln()
            # Row of values
            self.pdf.set_x(_MARGIN)
            self.pdf.set_font("Helvetica", "B", 9)
            self.pdf.set_text_color(*_BLACK)
            for _, v in items[:col_count]:
                fmt = f"{v:,.2f}" if isinstance(v, float) else str(v)
                self.pdf.cell(lev_w, 5, fmt, align="C")
            self.pdf.ln()
            # Second row if >4 levels
            if len(items) > col_count:
                col_count2 = min(len(items) - col_count, 4)
                lev_w2 = fw / col_count2
                self.pdf.set_x(_MARGIN)
                self.pdf.set_font("Helvetica", "", 6.5)
                self.pdf.set_text_color(*_MID_GREY)
                for k, _ in items[col_count:col_count + col_count2]:
                    self.pdf.cell(lev_w2, 4, k, align="C")
                self.pdf.ln()
                self.pdf.set_x(_MARGIN)
                self.pdf.set_font("Helvetica", "B", 9)
                self.pdf.set_text_color(*_BLACK)
                for _, v in items[col_count:col_count + col_count2]:
                    fmt = f"{v:,.2f}" if isinstance(v, float) else str(v)
                    self.pdf.cell(lev_w2, 5, fmt, align="C")
                self.pdf.ln()
            self.pdf.ln(1)
            # Key levels text explanation
            self.pdf.set_x(_MARGIN)
            self.pdf.set_font("Helvetica", "I", 6.5)
            self.pdf.set_text_color(*_MID_GREY)
            self.pdf.multi_cell(fw, 3.5, self._explain_key_levels(card, meta, levels))
            self.pdf.set_text_color(*_BLACK)
            self.pdf.ln(2)

        # Failure scenario
        self.pdf.set_x(_MARGIN)
        self.pdf.set_fill_color(*_ALERT_BG)
        self.pdf.set_font("Helvetica", "B", 8)
        self.pdf.set_text_color(140, 20, 20)
        self.pdf.cell(fw, 5, "  FAILURE SCENARIO", ln=True, fill=True)
        self.pdf.set_x(_MARGIN)
        self.pdf.set_text_color(60, 20, 20)
        self.pdf.set_font("Helvetica", "", 7.5)
        self.pdf.multi_cell(fw, 4.5, self._safe("  " + card.failure_scenario[:500]))
        self.pdf.set_x(_MARGIN)
        self.pdf.set_text_color(*_MID_GREY)
        self.pdf.set_font("Helvetica", "I", 6.5)
        self.pdf.multi_cell(fw, 3.5, "  If this scenario is in play, skip this trade regardless of conviction.")
        self.pdf.set_text_color(*_BLACK)
        self.pdf.ln(3)

        # Payout graph
        payout_path = self._generate_payout_graph(card)
        if payout_path:
            self.pdf.set_x(_MARGIN)
            self._section_header("Estimated Payout Profile")
            self.pdf.set_x(_MARGIN)
            self.pdf.set_font("Helvetica", "", 6.5)
            self.pdf.set_text_color(*_MID_GREY)
            self.pdf.multi_cell(fw, 3.5, self._explain_payout(card, meta))
            self.pdf.set_text_color(*_BLACK)
            self.pdf.ln(2)
            try:
                self.pdf.image(payout_path, x=_MARGIN, w=fw)
            except Exception:
                pass
            try:
                Path(payout_path).unlink(missing_ok=True)
            except Exception:
                pass
            self.pdf.ln(2)
            self.pdf.set_x(_MARGIN)
            self.pdf.set_font("Helvetica", "I", 6)
            self.pdf.set_text_color(*_MID_GREY)
            self.pdf.multi_cell(fw, 3.5,
                "Payout profiles use proxy premium assumptions. Illustrative only. Verify with live pricing."
            )
            self.pdf.set_text_color(*_BLACK)

        self._add_page_footer()

    def _explain_key_levels(self, card, meta: dict, levels: dict) -> str:
        """Generate textual explanation of key levels for the trade."""
        parts = []
        if "target_yield" in meta and "stop_yield" in meta:
            entry = meta.get("jp10_level", meta.get("target_yield", 0))
            target = meta["target_yield"]
            stop = meta["stop_yield"]
            rr = abs(target - entry) / max(abs(stop - entry), 0.001)
            parts.append(
                f"Entry {entry:.3f}%, target {target:.2f}%, stop {stop:.2f}%. "
                f"Risk/reward: {rr:.1f}:1. "
            )
            if rr >= 2:
                parts.append("Favourable (>=2:1). ")
            elif rr >= 1:
                parts.append("Acceptable (>=1:1). ")
            else:
                parts.append("Unfavourable (<1:1) - reduce sizing. ")
        if "usdjpy_spot" in meta:
            parts.append(f"USDJPY spot: {meta['usdjpy_spot']:.2f}. ")
        if "straddle_strike" in meta:
            K = meta["straddle_strike"]
            parts.append(f"Straddle at {K:.1f}. Max loss = total premium. BE approx {K:.1f} +/- premium.")
        if "atm_strike" in meta and "otm_strike" in meta:
            parts.append(
                f"Payer spread: buy ATM {meta['atm_strike']:.3f}%, sell OTM {meta['otm_strike']:.3f}%. "
                f"Max gain = spread width - premium. Max loss = premium."
            )
        if "spread_bps" in meta and "target_spread_bps" in meta:
            parts.append(
                f"Entry spread: {meta['spread_bps']:.0f} bps, target: {meta['target_spread_bps']:.0f} bps."
            )
        return "".join(parts) if parts else "Key levels define entry, target, and risk boundaries."

    def _explain_payout(self, card, meta: dict) -> str:
        """Generate textual interpretation of the payout profile."""
        if "straddle_strike" in meta:
            return (
                "Straddle profits from significant moves in either direction. "
                "Market-neutral; benefits from realised vol exceeding implied vol."
            )
        if "atm_strike" in meta and "otm_strike" in meta:
            return (
                "Payer spread: capped upside, defined max loss. "
                "Directional bet on higher rates with limited downside."
            )
        if "call_strike" in meta and "put_strike" in meta:
            return (
                "Short strangle: collects premium. Profit zone between strikes. "
                "Losses beyond breakeven points. Profits from low realised volatility."
            )
        if "put_strike" in meta and "usdjpy_spot" in meta:
            return (
                "Long put: profits on decline below strike minus premium. "
                "Max loss limited to premium paid."
            )
        if "target_yield" in meta and "stop_yield" in meta:
            return (
                f"Linear {card.direction} yield trade. P&L proportional to yield movement. "
                f"Green: target. Red: stop-loss."
            )
        if "target" in meta and "stop" in meta and "usdjpy_spot" in meta:
            return (
                f"Linear {card.direction} FX trade. P&L proportional to spot movement."
            )
        if "spread_bps" in meta and "target_spread_bps" in meta:
            return "Spread trade: P&L depends on basis-point spread change."
        return "Estimated payout profile for this trade structure."

    def _generate_payout_graph(self, card) -> Optional[str]:
        """Generate a payout diagram as a temp PNG. Returns path or None."""
        import numpy as np
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        meta = card.metadata or {}
        fig, ax = plt.subplots(figsize=(6.5, 2.4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#fafafa")

        generated = False

        # --- Options payout: straddle ---
        if "straddle_strike" in meta:
            K = meta["straddle_strike"]
            premium = abs(K) * 0.015
            x = np.linspace(K - K * 0.05, K + K * 0.05, 200)
            call_payout = np.maximum(x - K, 0) - premium / 2
            put_payout = np.maximum(K - x, 0) - premium / 2
            total = call_payout + put_payout
            ax.plot(x, total, color="#000000", linewidth=1.8, label="Straddle P&L")
            ax.fill_between(x, total, 0, where=total > 0, color="#8E6F3E", alpha=0.15)
            ax.fill_between(x, total, 0, where=total < 0, color="#c0392b", alpha=0.10)
            ax.axhline(0, color="#aaa", linewidth=0.6, linestyle="--")
            ax.axvline(K, color="#8E6F3E", linewidth=0.8, linestyle=":", label=f"Strike {K:.1f}")
            ax.set_xlabel("Underlying Price", fontsize=8)
            ax.set_ylabel("P&L", fontsize=8)
            ax.set_title(f"{card.name} - Straddle Payout", fontsize=9, fontweight="bold")
            generated = True

        # --- Options payout: payer spread ---
        elif "atm_strike" in meta and "otm_strike" in meta:
            K1 = meta["atm_strike"]
            K2 = meta["otm_strike"]
            premium = abs(K2 - K1) * 0.4
            x = np.linspace(K1 - abs(K2 - K1) * 2, K2 + abs(K2 - K1) * 2, 200)
            long_call = np.maximum(x - K1, 0)
            short_call = np.maximum(x - K2, 0)
            total = long_call - short_call - premium
            ax.plot(x, total, color="#000000", linewidth=1.8, label="Payer Spread P&L")
            ax.fill_between(x, total, 0, where=total > 0, color="#8E6F3E", alpha=0.15)
            ax.fill_between(x, total, 0, where=total < 0, color="#c0392b", alpha=0.10)
            ax.axhline(0, color="#aaa", linewidth=0.6, linestyle="--")
            ax.axvline(K1, color="#8E6F3E", linewidth=0.8, linestyle=":", label=f"Buy {K1:.3f}%")
            ax.axvline(K2, color="#c0392b", linewidth=0.8, linestyle=":", label=f"Sell {K2:.3f}%")
            ax.set_xlabel("Swap Rate (%)", fontsize=8)
            ax.set_ylabel("P&L (bps)", fontsize=8)
            ax.set_title(f"{card.name} - Payer Spread", fontsize=9, fontweight="bold")
            generated = True

        # --- Options payout: strangle (short) ---
        elif "call_strike" in meta and "put_strike" in meta:
            Kc = meta["call_strike"]
            Kp = meta["put_strike"]
            premium = abs(Kc - Kp) * 0.3
            x = np.linspace(Kp - abs(Kc - Kp), Kc + abs(Kc - Kp), 200)
            short_call = -np.maximum(x - Kc, 0)
            short_put = -np.maximum(Kp - x, 0)
            total = short_call + short_put + premium
            ax.plot(x, total, color="#000000", linewidth=1.8, label="Short Strangle P&L")
            ax.fill_between(x, total, 0, where=total > 0, color="#8E6F3E", alpha=0.15)
            ax.fill_between(x, total, 0, where=total < 0, color="#c0392b", alpha=0.10)
            ax.axhline(0, color="#aaa", linewidth=0.6, linestyle="--")
            ax.axvline(Kp, color="#2e7d32", linewidth=0.8, linestyle=":", label=f"Put {Kp:.2f}")
            ax.axvline(Kc, color="#c0392b", linewidth=0.8, linestyle=":", label=f"Call {Kc:.2f}")
            ax.set_xlabel("Underlying Price", fontsize=8)
            ax.set_ylabel("P&L", fontsize=8)
            ax.set_title(f"{card.name} - Short Strangle", fontsize=9, fontweight="bold")
            generated = True

        # --- Options payout: single put ---
        elif "put_strike" in meta and "usdjpy_spot" in meta:
            K = meta["put_strike"]
            spot = meta["usdjpy_spot"]
            premium = abs(spot - K) * 0.15
            x = np.linspace(K * 0.94, spot * 1.04, 200)
            total = np.maximum(K - x, 0) - premium
            ax.plot(x, total, color="#000000", linewidth=1.8, label="Long Put P&L")
            ax.fill_between(x, total, 0, where=total > 0, color="#8E6F3E", alpha=0.15)
            ax.fill_between(x, total, 0, where=total < 0, color="#c0392b", alpha=0.10)
            ax.axhline(0, color="#aaa", linewidth=0.6, linestyle="--")
            ax.axvline(K, color="#8E6F3E", linewidth=0.8, linestyle=":", label=f"Strike {K:.0f}")
            ax.axvline(spot, color="#000", linewidth=0.8, alpha=0.4, label=f"Spot {spot:.0f}")
            ax.set_xlabel("USDJPY", fontsize=8)
            ax.set_ylabel("P&L per unit", fontsize=8)
            ax.set_title(f"{card.name} - Put Payout (K={K:.0f})", fontsize=9, fontweight="bold")
            generated = True

        # --- Linear: directional with target/stop ---
        elif "target_yield" in meta and "stop_yield" in meta:
            entry = meta.get("jp10_level", 1.0)
            target = meta["target_yield"]
            stop = meta["stop_yield"]
            x = np.linspace(min(stop, entry) - 0.1, max(target, entry) + 0.1, 200)
            if card.direction == "short":
                pnl = (entry - x) * 100
            else:
                pnl = (x - entry) * 100
            ax.plot(x, pnl, color="#000000", linewidth=1.8, label="P&L (bps)")
            ax.fill_between(x, pnl, 0, where=pnl > 0, color="#8E6F3E", alpha=0.15)
            ax.fill_between(x, pnl, 0, where=pnl < 0, color="#c0392b", alpha=0.10)
            ax.axhline(0, color="#aaa", linewidth=0.6, linestyle="--")
            ax.axvline(entry, color="#000", linewidth=1, label=f"Entry {entry:.3f}%")
            ax.axvline(target, color="#2e7d32", linewidth=1, linestyle="--", label=f"Target {target:.2f}%")
            ax.axvline(stop, color="#c0392b", linewidth=1, linestyle="--", label=f"Stop {stop:.2f}%")
            ax.set_xlabel("Yield (%)", fontsize=8)
            ax.set_ylabel("P&L (bps)", fontsize=8)
            ax.set_title(f"{card.name} - {card.direction.upper()} P&L", fontsize=9, fontweight="bold")
            generated = True

        # --- Linear: USDJPY with target/stop ---
        elif "target" in meta and "stop" in meta and "usdjpy_spot" in meta:
            spot = meta["usdjpy_spot"]
            target = meta["target"]
            stop = meta["stop"]
            x = np.linspace(stop * 0.98, target * 1.02, 200)
            if card.direction == "long":
                pnl = (x - spot) / spot * 100
            else:
                pnl = (spot - x) / spot * 100
            ax.plot(x, pnl, color="#000000", linewidth=1.8, label="P&L (%)")
            ax.fill_between(x, pnl, 0, where=pnl > 0, color="#8E6F3E", alpha=0.15)
            ax.fill_between(x, pnl, 0, where=pnl < 0, color="#c0392b", alpha=0.10)
            ax.axhline(0, color="#aaa", linewidth=0.6, linestyle="--")
            ax.axvline(spot, color="#000", linewidth=1, label=f"Spot {spot:.2f}")
            ax.axvline(target, color="#2e7d32", linewidth=1, linestyle="--", label=f"Target {target:.2f}")
            ax.axvline(stop, color="#c0392b", linewidth=1, linestyle="--", label=f"Stop {stop:.2f}")
            ax.set_xlabel("USDJPY", fontsize=8)
            ax.set_ylabel("P&L (%)", fontsize=8)
            ax.set_title(f"{card.name} - {card.direction.upper()} P&L", fontsize=9, fontweight="bold")
            generated = True

        # --- Spread trade ---
        elif "spread_bps" in meta and "target_spread_bps" in meta:
            entry = meta["spread_bps"]
            target = meta["target_spread_bps"]
            x = np.linspace(entry - 30, max(target, entry) + 20, 200)
            if card.direction == "long":
                pnl = x - entry
            else:
                pnl = entry - x
            ax.plot(x, pnl, color="#000000", linewidth=1.8, label="Spread P&L (bps)")
            ax.fill_between(x, pnl, 0, where=pnl > 0, color="#8E6F3E", alpha=0.15)
            ax.fill_between(x, pnl, 0, where=pnl < 0, color="#c0392b", alpha=0.10)
            ax.axhline(0, color="#aaa", linewidth=0.6, linestyle="--")
            ax.axvline(entry, color="#000", linewidth=1, label=f"Entry {entry:.0f} bps")
            ax.axvline(target, color="#2e7d32", linewidth=1, linestyle="--", label=f"Target {target:.0f} bps")
            ax.set_xlabel("Spread (bps)", fontsize=8)
            ax.set_ylabel("P&L (bps)", fontsize=8)
            ax.set_title(f"{card.name} - Spread P&L", fontsize=9, fontweight="bold")
            generated = True

        if not generated:
            plt.close(fig)
            return None

        ax.legend(fontsize=6.5, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.15, linewidth=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)
        fig.tight_layout()

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(tmp.name, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return tmp.name

    def add_intraday_fx_summary(
        self, df: pd.DataFrame, boj_dates: list, reactions: list
    ) -> None:
        """Add Intraday FX Event Study summary pages."""
        import numpy as np

        self.pdf.add_page()
        self.pdf.set_y(14)
        self.pdf.set_font("Helvetica", "B", 16)
        self.pdf.cell(0, 10, "Intraday FX Event Study", ln=True)
        self.pdf.set_font("Helvetica", "", 8)
        self.pdf.set_text_color(*_MID_GREY)
        self.pdf.cell(0, 5, f"Heramb S. Patkar, MSF  |  {datetime.now():%d %B %Y}", ln=True)
        self.pdf.set_text_color(*_BLACK)
        self._hairline()
        self.pdf.ln(6)

        # Executive summary
        self._section_header("Executive Summary")
        mid_col = "MID_PRICE" if "MID_PRICE" in df.columns else "BID"
        avg_mid = float(df[mid_col].mean()) if mid_col in df.columns and not df.empty else 0

        self._body_text(
            f"Minute-level USDJPY analysis around {len(boj_dates) if boj_dates else len(reactions)} "
            f"BOJ monetary policy announcements. Data source: LSEG (Refinitiv) via Purdue University "
            f"institutional subscription. {len(df):,} price observations at 1-minute frequency."
        )

        self._body_text(
            "Methodology: USDJPY tracked from -30 min to +60 min around approximate announcement "
            "time (12:00 JST / 03:00 UTC). Reaction = pip change over this window. Positive = yen "
            "weakened (dovish). Negative = yen strengthened (hawkish)."
        )

        if reactions:
            self._section_header("Key Findings")
            react_vals = [r["Reaction (pips)"] for r in reactions]
            avg_abs = np.mean([abs(v) for v in react_vals])
            best = max(reactions, key=lambda r: abs(r["Reaction (pips)"]))
            n_pos = sum(1 for v in react_vals if v > 0)
            n_neg = sum(1 for v in react_vals if v < 0)
            avg_max_spread = np.mean([r.get("Max Spread (pips)", 0) for r in reactions])

            findings = [
                f"Average absolute reaction: {avg_abs:.1f} pips across {len(reactions)} meetings.",
                f"Largest: {best['Reaction (pips)']:+.1f} pips on {best['Date']}.",
                f"Direction: yen weakened {n_pos}x, strengthened {n_neg}x.",
                f"Average max spread on BOJ days: {avg_max_spread:.1f} pips (normal: 1-2).",
            ]
            self.pdf.set_font("Helvetica", "", 8.5)
            for f in findings:
                self.pdf.multi_cell(0, 4.5, self._safe(f"  -  {f}"))
                self.pdf.ln(1)
            self.pdf.ln(2)

            # Inference
            self._section_header("Inference")
            if n_neg > n_pos:
                self._body_text(
                    "BOJ has been more hawkish than expected. Markets have consistently "
                    "underestimated the pace of policy normalisation."
                )
            elif n_pos > n_neg:
                self._body_text(
                    "BOJ has been more dovish than expected. Markets have overestimated "
                    "how quickly the BOJ would tighten."
                )
            else:
                self._body_text("Reactions balanced; market reasonably calibrated on average.")

            # Reaction table
            self._section_header("Reaction Table")
            col_w = [28, 24, 24, 28, 28, 24, 24]
            headers = ["Date", "Pre", "Post", "React", "Range", "AvgSprd", "MaxSprd"]
            self.pdf.set_font("Helvetica", "B", 7)
            for w, h in zip(col_w, headers):
                self.pdf.cell(w, 5.5, h, border="B", align="C")
            self.pdf.ln()
            self.pdf.set_font("Helvetica", "", 6.5)
            self.pdf.set_fill_color(*_SIDEBAR_BG)
            for i, r in enumerate(reactions):
                fill = i % 2 == 0
                self.pdf.cell(col_w[0], 5, str(r["Date"]), fill=fill)
                self.pdf.cell(col_w[1], 5, f"{r['Pre-Price']:.2f}", fill=fill, align="R")
                self.pdf.cell(col_w[2], 5, f"{r['Post-Price']:.2f}", fill=fill, align="R")
                self.pdf.cell(col_w[3], 5, f"{r['Reaction (pips)']:+.1f}", fill=fill, align="R")
                self.pdf.cell(col_w[4], 5, f"{r['Day Range (pips)']:.1f}", fill=fill, align="R")
                self.pdf.cell(col_w[5], 5, f"{r['Avg Spread (pips)']:.1f}", fill=fill, align="R")
                self.pdf.cell(col_w[6], 5, f"{r['Max Spread (pips)']:.1f}", fill=fill, align="R")
                self.pdf.ln()

        self.pdf.ln(4)
        self.pdf.set_font("Helvetica", "I", 6.5)
        self.pdf.set_text_color(*_MID_GREY)
        self.pdf.multi_cell(0, 3.5, self._safe(
            "Data: LSEG (Refinitiv), Purdue Daniels institutional subscription. RIC: JPY=. "
            "1-min frequency. All times UTC. For educational purposes only."
        ))
        self.pdf.set_text_color(*_BLACK)
        self._add_page_footer()

    # ── profile-aware full report ──────────────────────────────────────
    def add_full_analysis_report(
        self,
        profile: str,
        *,
        regime_state: dict | None = None,
        pca_result: dict | None = None,
        ensemble_prob: float | None = None,
        warning_score: float | None = None,
        ml_prob: float | None = None,
        ml_importance: "pd.Series | None" = None,
        spillover_pct: float | None = None,
        carry_ratio: float | None = None,
        cards: list | None = None,
        metrics: "AccuracyMetrics | None" = None,
        suggestions: list | None = None,
        reactions: list | None = None,
    ) -> None:
        """Build a profile-tailored PDF in institutional research note format.

        Profiles
        --------
        Analyst  - balanced: all sections, moderate detail.
        Trader   - action-first: regime state, trade ideas, alerts, key levels.
        Academic - methodology-heavy: model descriptions, validation, references.
        """
        import numpy as np

        profile = profile.strip().title()
        rp = (regime_state or {}).get("regime_prob", ensemble_prob or 0.5)
        regime_word = "repricing" if rp > 0.5 else "suppressed"

        # ── Page 1: Regime Overview (institutional equity-research layout) ─
        self.pdf.add_page()

        # Accent colour for recommendation box
        _REC_BG = (207, 185, 145)
        _sb_x = 134
        _sb_w = _PAGE_W - _MARGIN - _sb_x
        _cw = _sb_x - _MARGIN - 3

        # Header bar
        self.pdf.set_fill_color(*_BLACK)
        self.pdf.rect(_MARGIN, 10, _PAGE_W - 2 * _MARGIN, 9, "F")
        self.pdf.set_xy(_MARGIN + 2, 10.5)
        self.pdf.set_font("Helvetica", "B", 8)
        self.pdf.set_text_color(255, 255, 255)
        self.pdf.cell(90, 8, "Purdue Daniels School of Business")
        self.pdf.set_font("Helvetica", "", 7)
        self.pdf.cell(_PAGE_W - 2 * _MARGIN - 94, 8, f"JGB Rates Research  |  {profile} View  |  {datetime.now():%d %B %Y}", align="R")
        self.pdf.set_text_color(*_BLACK)

        # Right panel — recommendation box
        _box_y = 22
        self.pdf.set_fill_color(*_REC_BG)
        self.pdf.rect(_sb_x, _box_y, _sb_w, 22, "F")
        self.pdf.set_xy(_sb_x + 2, _box_y + 1)
        self.pdf.set_font("Helvetica", "", 6.5)
        self.pdf.cell(_sb_w - 4, 4, "Regime Assessment", align="C")
        self.pdf.set_xy(_sb_x + 2, _box_y + 5.5)
        self.pdf.set_font("Helvetica", "B", 14)
        self.pdf.cell(_sb_w - 4, 8, regime_word.upper(), align="C")
        self.pdf.set_xy(_sb_x + 2, _box_y + 15)
        self.pdf.set_font("Helvetica", "", 7)
        self.pdf.cell(_sb_w - 4, 4, f"Prob: {rp:.1%}", align="C")

        # Right panel — key data
        _kd_y = _box_y + 25
        self.pdf.set_fill_color(*_REC_BG)
        self.pdf.set_xy(_sb_x, _kd_y)
        self.pdf.set_font("Helvetica", "B", 7)
        self.pdf.cell(_sb_w, 5.5, "  Key Data", fill=True, ln=True)
        _kd_rows = [("Ensemble Prob.", f"{rp:.1%}")]
        if ml_prob is not None:
            _kd_rows.append(("ML 5d Prob.", f"{ml_prob:.1%}"))
        if warning_score is not None:
            _kd_rows.append(("Warning Score", f"{warning_score:.0f}/100"))
        if spillover_pct is not None:
            _kd_rows.append(("Spillover %", f"{spillover_pct:.1f}%"))
        if carry_ratio is not None:
            _kd_rows.append(("Carry/Vol", f"{carry_ratio:.2f}"))
        if cards:
            _kd_rows.append(("Total Trades", str(len(cards))))
        self.pdf.set_fill_color(*_SIDEBAR_BG)
        for i, (label, value) in enumerate(_kd_rows):
            self.pdf.set_x(_sb_x)
            fill = i % 2 == 0
            self.pdf.set_font("Helvetica", "", 6.5)
            self.pdf.set_text_color(*_DARK_GREY)
            self.pdf.cell(_sb_w - 18, 4.5, f"  {label}", fill=fill)
            self.pdf.set_font("Helvetica", "B", 6.5)
            self.pdf.set_text_color(*_BLACK)
            self.pdf.cell(18, 4.5, value, fill=fill, align="R", ln=True)

        # Right panel — analyst info
        _a_y = self.pdf.get_y() + 3
        self.pdf.set_draw_color(*_LIGHT_GREY)
        self.pdf.line(_sb_x, _a_y - 1, _sb_x + _sb_w, _a_y - 1)
        self.pdf.set_xy(_sb_x, _a_y)
        self.pdf.set_font("Helvetica", "B", 6.5)
        self.pdf.set_text_color(*_DARK_GREY)
        self.pdf.cell(_sb_w, 4, "  Analyst", ln=True)
        self.pdf.set_x(_sb_x)
        self.pdf.set_font("Helvetica", "", 6.5)
        self.pdf.set_text_color(*_BLACK)
        self.pdf.cell(_sb_w, 3.5, "  Heramb S. Patkar, MSF", ln=True)
        self.pdf.set_x(_sb_x)
        self.pdf.set_text_color(*_MID_GREY)
        self.pdf.cell(_sb_w, 3.5, "  Purdue Daniels School of Business", ln=True)
        self.pdf.set_x(_sb_x)
        self.pdf.cell(_sb_w, 3.5, "  MGMT 69000-119", ln=True)
        self.pdf.set_draw_color(*_BLACK)
        self.pdf.set_text_color(*_BLACK)

        # Left body — title + content
        self.pdf.set_xy(_MARGIN, 22)
        self.pdf.set_font("Helvetica", "B", 18)
        self.pdf.cell(_cw, 10, "JGB Repricing Report")
        self.pdf.ln(10)
        self.pdf.set_x(_MARGIN)
        self.pdf.set_font("Helvetica", "", 9)
        self.pdf.set_text_color(*_DARK_GREY)
        self.pdf.multi_cell(_cw, 4.5, self._safe(
            f"{profile} view: regime-conditional analysis under {regime_word.lower()} "
            f"with ensemble probability {rp:.0%}"
        ))
        self.pdf.set_text_color(*_BLACK)
        self.pdf.ln(4)

        # Current regime state (constrained to left column while beside right panel)
        self.pdf.set_x(_MARGIN)
        self.pdf.set_font("Helvetica", "B", 9)
        self.pdf.set_text_color(*_DARK_GREY)
        self.pdf.cell(_cw, 6, "CURRENT REGIME STATE", ln=True)
        self.pdf.set_text_color(*_BLACK)
        self.pdf.ln(1)
        self.pdf.set_x(_MARGIN)
        regime_text = f"Ensemble regime probability: {rp:.1%} ({regime_word.upper()}). "
        if ml_prob is not None:
            regime_text += f"ML 5-day forward probability: {ml_prob:.1%}. "
            if (ml_prob > 0.5) == (rp > 0.5):
                regime_text += "Ensemble and ML model agree: high conviction. "
            else:
                regime_text += "Ensemble and ML diverge: reduce sizing, await convergence. "
        if warning_score is not None:
            regime_text += f"Composite warning score: {warning_score:.0f}/100. "
            if warning_score > 70:
                regime_text += "CRITICAL: multiple stress indicators firing. "
            elif warning_score > 50:
                regime_text += "Elevated: monitor for regime shift. "
        self.pdf.set_font("Helvetica", "", 8)
        self.pdf.set_text_color(*_DARK_GREY)
        self.pdf.multi_cell(_cw, 4, self._safe(regime_text))
        self.pdf.set_text_color(*_BLACK)
        self.pdf.ln(3)

        # Cross-market conditions (still in left column)
        if spillover_pct is not None or carry_ratio is not None:
            self.pdf.set_x(_MARGIN)
            self.pdf.set_font("Helvetica", "B", 9)
            self.pdf.set_text_color(*_DARK_GREY)
            self.pdf.cell(_cw, 6, "CROSS-MARKET CONDITIONS", ln=True)
            self.pdf.set_text_color(*_BLACK)
            self.pdf.ln(1)
            self.pdf.set_x(_MARGIN)
            cm_text = ""
            if spillover_pct is not None:
                cm_text += (
                    f"Total spillover index: {spillover_pct:.1f}%. "
                    f"{'Above 30%: tightly coupled, diversification impaired. ' if spillover_pct > 30 else 'Below 30%: markets relatively independent. '}"
                )
            if carry_ratio is not None:
                cm_text += (
                    f"Carry-to-vol ratio: {carry_ratio:.2f}. "
                    f"{'Below 0.5: carry poorly compensated. ' if carry_ratio < 0.5 else 'Above 1.0: attractive carry. ' if carry_ratio > 1.0 else ''}"
                )
            self.pdf.set_font("Helvetica", "", 8)
            self.pdf.set_text_color(*_DARK_GREY)
            self.pdf.multi_cell(_cw, 4, self._safe(cm_text))
            self.pdf.set_text_color(*_BLACK)
            self.pdf.ln(3)

        self._add_page_footer()

        # ── Page 2: Analytical Detail (full-width, no sidebar) ────────────
        _has_detail = (
            (profile in ("Analyst", "Academic") and pca_result is not None)
            or ml_prob is not None
            or profile == "Academic"
        )
        if _has_detail:
            self.pdf.add_page()
            self.pdf.set_y(14)

            if profile in ("Analyst", "Academic") and pca_result is not None:
                self._section_header("Yield Curve Decomposition (PCA)")
                ev = pca_result.get("explained_variance_ratio", [])
                pca_text = ""
                if len(ev) >= 3:
                    pca_text += (
                        f"PC1 (Level): {ev[0]:.1%}. PC2 (Slope): {ev[1]:.1%}. PC3 (Curvature): {ev[2]:.1%}. "
                    )
                    if ev[0] > 0.8:
                        pca_text += "PC1 dominates (>80%): curve repricing in lockstep. "
                    elif ev[1] > 0.15:
                        pca_text += "Elevated PC2: slope rotation, steepener/flattener opportunity. "
                if profile == "Academic":
                    pca_text += (
                        "Ref: Litterman & Scheinkman (1991). PCA on daily yield changes (not levels). "
                        "Covariance-based scaling preserves bps economic meaning. Rolling 252-day window."
                    )
                self._body_text(pca_text)

            if ml_prob is not None:
                self._section_header("ML Regime Transition Predictor")
                ml_text = f"Walk-forward RandomForest: {ml_prob:.1%} probability of transition within 5 days. "
                if profile == "Trader":
                    if ml_prob > 0.5:
                        ml_text += "ACTION: position for repricing. Full conviction sizing. "
                    else:
                        ml_text += "ACTION: maintain positioning. No imminent shift. "
                if ml_importance is not None and len(ml_importance) > 0:
                    top_feat = ml_importance.index[0]
                    top_val = ml_importance.iloc[0]
                    ml_text += f"Top feature: {top_feat} ({top_val:.2f}). "
                    if top_val > 0.4:
                        ml_text += "Single feature dominance >40%: model may be fragile. "
                if profile == "Academic":
                    ml_text += (
                        "Methodology: RF (50 trees, max depth 5), 504-day window, quarterly retrain. "
                        "Features: structural entropy, entropy delta, carry stress, spillover correlation, "
                        "vol z-score, VIX, USDJPY momentum. Labels: binary forward-looking. No look-ahead bias."
                    )
                self._body_text(ml_text)

            if profile == "Academic":
                self._section_header("Ensemble Regime Detection Methodology")
                self.pdf.set_font("Helvetica", "", 8)
                self.pdf.multi_cell(0, 4.5, self._safe(
                    "Four models, normalised to [0,1], equally weighted (25% each): "
                    "(1) Hamilton (1989) 2-state Markov-Switching: smoothed state probabilities. "
                    "(2) Gaussian HMM (hmmlearn): Viterbi state sequence on multivariate features. "
                    "(3) Bandt & Pompe (2002) permutation entropy (order=3, window=120): z-score threshold. "
                    "(4) Bollerslev (1986) GARCH(1,1) + PELT breakpoints: vol-regime detection. "
                    "Thresholds: >0.7 STRONG, 0.5-0.7 MODERATE, 0.3-0.5 TRANSITION, <0.3 SUPPRESSED."
                ))
                self.pdf.ln(3)

            self._add_page_footer()

        # ── Trade Ideas ──────────────────────────────────────────────────
        if cards:
            self.add_trade_ideas(cards, regime_state)

        # ── Performance Metrics ──────────────────────────────────────────
        if profile in ("Analyst", "Academic") and metrics is not None:
            self.add_metrics_summary(metrics)
            if suggestions:
                self.add_suggestions(suggestions)

        # ── Intraday FX ──────────────────────────────────────────────────
        if reactions:
            self.add_intraday_fx_summary(pd.DataFrame(), [], reactions)

        # ── References (Academic) ────────────────────────────────────────
        if profile == "Academic":
            self.pdf.add_page()
            self.pdf.set_y(14)
            self.pdf.set_font("Helvetica", "B", 14)
            self.pdf.cell(0, 10, "References", ln=True)
            self._hairline()
            self.pdf.ln(6)
            self.pdf.set_font("Helvetica", "", 8)
            refs = [
                "Litterman, R. & Scheinkman, J. (1991). Common factors affecting bond returns. Journal of Fixed Income, 1(1), 54-61.",
                "Hamilton, J.D. (1989). A new approach to the economic analysis of nonstationary time series. Econometrica, 57(2), 357-384.",
                "Diebold, F.X. & Yilmaz, K. (2012). Better to give than to receive: Predictive directional measurement of volatility spillovers. Int. J. of Forecasting, 28(1), 57-66.",
                "Bandt, C. & Pompe, B. (2002). Permutation entropy: A natural complexity measure for time series. Physical Review Letters, 88(17), 174102.",
                "Schreiber, T. (2000). Measuring information transfer. Physical Review Letters, 85(2), 461-464.",
                "Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics, 31(3), 307-327.",
            ]
            for i, ref in enumerate(refs, 1):
                self.pdf.multi_cell(0, 4.5, self._safe(f"[{i}]  {ref}"))
                self.pdf.ln(2)
            self._add_page_footer()

        # ── Disclaimer (full page, dense) ────────────────────────────────
        self._add_full_disclaimer(profile)

    def _add_full_disclaimer(self, profile: str = "") -> None:
        """Full-page institutional disclaimer."""
        self.pdf.add_page()
        self.pdf.set_y(14)

        # Title
        self.pdf.set_font("Helvetica", "B", 16)
        self.pdf.cell(0, 10, "Important Disclosures and Disclaimer", ln=True)
        self._hairline()
        self.pdf.ln(8)

        # Body
        self.pdf.set_font("Helvetica", "", 8)
        self.pdf.set_text_color(*_DARK_GREY)

        paragraphs = [
            (
                "GENERAL DISCLAIMER. This report (the 'Report') has been prepared by Heramb S. Patkar "
                "('the Author') as part of coursework for MGMT 69000 - Mastering AI for Finance at "
                "Purdue University, Daniels School of Business, under the supervision of Prof. Cinder "
                "Zhang. The Report is provided for educational and research purposes only."
            ),
            (
                "NOT INVESTMENT ADVICE. Nothing contained in this Report constitutes investment advice, "
                "a solicitation, or a recommendation to buy, sell, or hold any security, financial "
                "instrument, or investment product. The analysis, trade ideas, and strategies described "
                "herein are hypothetical and have not been executed in live markets. Past performance "
                "of any signal, model, or strategy described in this Report does not guarantee future "
                "results. Any decision to invest based on information in this Report is made at the "
                "reader's sole risk."
            ),
            (
                "MODEL LIMITATIONS. The regime detection models, PCA decomposition, spillover analysis, "
                "and trade generation algorithms described herein are subject to significant estimation "
                "risk, parameter uncertainty, and model specification error. Regime detection is "
                "inherently probabilistic and backward-looking. The ensemble approach mitigates but does "
                "not eliminate false positive and false negative risks. GARCH DCC uses an EWMA proxy "
                "rather than full MLE bivariate optimisation. Transfer entropy uses simple histogram "
                "binning. These simplifications may affect accuracy."
            ),
            (
                "DATA SOURCES AND RELIABILITY. Data is sourced from FRED (Federal Reserve Economic Data), "
                "yfinance (Yahoo Finance), the Japanese Ministry of Finance, and LSEG (Refinitiv) via "
                "Purdue University's institutional subscription. The Author makes no representations "
                "regarding the accuracy, completeness, or timeliness of any data used. Data gaps, "
                "revisions, and feed disruptions may affect results without notice."
            ),
            (
                "AI-ASSISTED DEVELOPMENT. This framework was developed with AI pair-programming tools "
                "(Claude, GPT-4o) used for code generation, debugging, and architecture scaffolding. "
                "All analytical decisions, model selection, parameter choices, thesis formulation, and "
                "trade logic were made by the Author. See the AI collaboration log for full documentation."
            ),
            (
                "PAYOUT PROFILES. Payout diagrams and P&L estimates use proxy premium assumptions and "
                "simplified option pricing. They are illustrative only and do not represent actual market "
                "prices. Actual P&L will depend on execution prices, bid-ask spreads, margin requirements, "
                "liquidity conditions, and volatility at the time of trade entry."
            ),
            (
                "INTELLECTUAL PROPERTY. This Report and the underlying codebase are provided under the "
                "MIT License. The Purdue University name, Daniels School of Business branding, and "
                "associated trademarks are used with permission for academic attribution only."
            ),
            (
                "LIMITATION OF LIABILITY. The Author, Purdue University, and the Daniels School of "
                "Business accept no liability whatsoever for any direct, indirect, incidental, special, "
                "consequential, or exemplary damages arising from or in connection with the use of this "
                "Report or any information contained herein, including but not limited to loss of profits, "
                "loss of data, or trading losses. The reader assumes all risks associated with any "
                "decisions made based on information in this Report."
            ),
            (
                "CONFLICTS OF INTEREST. The Author has no positions in any securities or instruments "
                "discussed in this Report and has received no compensation from any party for the "
                "preparation of this analysis. This Report was prepared solely for academic purposes."
            ),
        ]

        for para in paragraphs:
            self.pdf.multi_cell(0, 4.5, self._safe(para))
            self.pdf.ln(3)

        # Signature block
        self.pdf.ln(6)
        self._hairline()
        self.pdf.ln(4)
        self.pdf.set_font("Helvetica", "I", 7.5)
        self.pdf.set_text_color(*_MID_GREY)
        self.pdf.multi_cell(0, 4, self._safe(
            f"Report generated: {datetime.now():%Y-%m-%d %H:%M:%S}  |  "
            f"Profile: {profile}  |  "
            f"Analyst: Heramb S. Patkar, MSF Candidate  |  "
            f"Purdue University, Daniels School of Business  |  "
            f"MGMT 69000-119: Mastering AI for Finance (Prof. Cinder Zhang)"
        ))
        self.pdf.set_text_color(*_BLACK)
        self._add_page_footer()

    def save(self, path: str | Path) -> str:
        path = str(path)
        self.pdf.output(path)
        return path

    def to_bytes(self) -> bytes:
        result = self.pdf.output()
        return bytes(result) if isinstance(result, bytearray) else result


def export_dataframes_to_csv(
    dataframes: Dict[str, pd.DataFrame],
    output_dir: str | Path | None = None,
) -> Dict[str, str]:
    """Export multiple DataFrames to CSV files."""
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, df in dataframes.items():
        if df is not None and not df.empty:
            path = output_dir / f"{name}.csv"
            df.to_csv(path)
            paths[name] = str(path)
    return paths


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to CSV bytes for Streamlit download."""
    return df.to_csv(index=True).encode("utf-8")
