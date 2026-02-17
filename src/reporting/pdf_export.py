"""
PDF and CSV export for JGB Repricing Framework reports.

Uses fpdf2 for PDF generation. Report structure:
  Title page (with Purdue Daniels logo) -> Metric summary -> Charts -> Suggestions
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

# Purdue brand colours
_PURDUE_BLACK = (0, 0, 0)
_BOILERMAKER_GOLD = (207, 185, 145)
_AGED_GOLD = (142, 111, 62)

_LOGO_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "purdue_daniels_logo.png"


class JGBReportPDF:
    """Generate PDF reports for the JGB Repricing Framework."""

    def __init__(self):
        if FPDF is None:
            raise ImportError("fpdf2 is required for PDF export. Install with: pip install fpdf2")
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)

    # ── helpers ──────────────────────────────────────────────────────────
    def _draw_gold_rule(self, y: float | None = None, width: float = 190) -> None:
        """Draw a horizontal Boilermaker Gold rule."""
        if y is None:
            y = self.pdf.get_y()
        self.pdf.set_draw_color(*_BOILERMAKER_GOLD)
        self.pdf.set_line_width(0.8)
        x = (210 - width) / 2
        self.pdf.line(x, y, x + width, y)
        self.pdf.set_draw_color(0, 0, 0)
        self.pdf.set_line_width(0.2)

    def _draw_header_bar(self) -> None:
        """Draw the black header bar with gold accent at the very top of a page."""
        self.pdf.set_fill_color(*_PURDUE_BLACK)
        self.pdf.rect(0, 0, 210, 8, "F")
        self.pdf.set_fill_color(*_BOILERMAKER_GOLD)
        self.pdf.rect(0, 8, 210, 1.5, "F")

    def _add_page_footer(self) -> None:
        """Add a small footer with course + date to the current page bottom."""
        self.pdf.set_y(-20)
        self._draw_gold_rule(width=170)
        self.pdf.ln(3)
        self.pdf.set_font("Helvetica", "", 7)
        self.pdf.set_text_color(100, 100, 100)
        self.pdf.cell(0, 5, "AI for Finance | MGMT 69000-119 | Purdue University", align="L")
        self.pdf.cell(0, 5, f"Generated {datetime.now():%Y-%m-%d %H:%M}", align="R", ln=True)
        self.pdf.set_text_color(0, 0, 0)

    # ── title page ──────────────────────────────────────────────────────
    def add_title_page(
        self,
        title: str = "JGB Repricing Framework Report",
        subtitle: str = "",
    ) -> None:
        self.pdf.add_page()

        # Black header bar + gold accent
        self._draw_header_bar()

        # Logo
        self.pdf.ln(18)
        logo = str(_LOGO_PATH) if _LOGO_PATH.exists() else None
        if logo:
            try:
                # Centre the logo (image width 90mm)
                self.pdf.image(logo, x=60, w=90)
                self.pdf.ln(8)
            except Exception:
                pass

        # Gold rule under logo
        self._draw_gold_rule()
        self.pdf.ln(12)

        # Title
        self.pdf.set_font("Helvetica", "B", 26)
        self.pdf.set_text_color(*_PURDUE_BLACK)
        self.pdf.cell(0, 14, title, ln=True, align="C")
        self.pdf.ln(2)

        if subtitle:
            self.pdf.set_font("Helvetica", "", 13)
            self.pdf.set_text_color(*_AGED_GOLD)
            self.pdf.cell(0, 9, subtitle, ln=True, align="C")
            self.pdf.set_text_color(*_PURDUE_BLACK)
            self.pdf.ln(4)

        # Gold rule under title
        self._draw_gold_rule(width=120)
        self.pdf.ln(16)

        # Metadata block
        self.pdf.set_font("Helvetica", "", 11)
        self.pdf.set_text_color(80, 80, 80)
        self.pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
        self.pdf.cell(0, 8, "Mastering AI for Finance  |  MGMT 69000-119", ln=True, align="C")
        self.pdf.cell(0, 8, "Mitch Daniels School of Business  |  Purdue University", ln=True, align="C")
        self.pdf.ln(6)
        self.pdf.set_font("Helvetica", "I", 10)
        self.pdf.cell(0, 8, "West Lafayette, Indiana", ln=True, align="C")
        self.pdf.set_text_color(0, 0, 0)

        # Footer
        self._add_page_footer()

    def add_metrics_summary(self, metrics: AccuracyMetrics) -> None:
        self.pdf.add_page()
        self._draw_header_bar()
        self.pdf.ln(14)
        self.pdf.set_font("Helvetica", "B", 16)
        self.pdf.cell(0, 12, "Performance Metrics", ln=True)
        self.pdf.ln(5)

        rows = [
            ("Prediction Accuracy", f"{metrics.prediction_accuracy:.1%}"),
            ("Average Lead Time", f"{metrics.average_lead_time:.1f} days"),
            ("Precision", f"{metrics.precision:.1%}"),
            ("Recall", f"{metrics.recall:.1%}"),
            ("False Positive Rate", f"{metrics.false_positive_rate:.1%}"),
            ("Total Predictions", str(metrics.total_predictions)),
        ]

        self.pdf.set_font("Helvetica", "B", 10)
        self.pdf.cell(90, 8, "Metric", border=1, align="C")
        self.pdf.cell(90, 8, "Value", border=1, align="C", ln=True)

        self.pdf.set_font("Helvetica", "", 10)
        for label, value in rows:
            self.pdf.cell(90, 8, label, border=1)
            self.pdf.cell(90, 8, value, border=1, align="C", ln=True)

    def add_suggestions(self, suggestions: List[str]) -> None:
        self.pdf.add_page()
        self._draw_header_bar()
        self.pdf.ln(14)
        self.pdf.set_font("Helvetica", "B", 16)
        self.pdf.cell(0, 12, "Improvement Suggestions", ln=True)
        self.pdf.ln(5)

        self.pdf.set_font("Helvetica", "", 10)
        for i, suggestion in enumerate(suggestions, 1):
            self.pdf.multi_cell(0, 6, f"{i}. {suggestion}")
            self.pdf.ln(3)

    def add_chart_image(self, image_path: str, title: str = "") -> None:
        self.pdf.add_page()
        self._draw_header_bar()
        self.pdf.ln(14)
        if title:
            self.pdf.set_font("Helvetica", "B", 14)
            self.pdf.cell(0, 10, title, ln=True)
            self.pdf.ln(3)
        try:
            self.pdf.image(image_path, x=10, w=190)
        except Exception:
            self.pdf.set_font("Helvetica", "I", 10)
            self.pdf.cell(0, 10, f"[Chart image not available: {image_path}]", ln=True)

    def add_data_table(
        self,
        df: pd.DataFrame,
        title: str = "",
        max_rows: int = 50,
    ) -> None:
        self.pdf.add_page()
        self._draw_header_bar()
        self.pdf.ln(14)
        if title:
            self.pdf.set_font("Helvetica", "B", 14)
            self.pdf.cell(0, 10, title, ln=True)
            self.pdf.ln(3)

        if df.empty:
            self.pdf.set_font("Helvetica", "I", 10)
            self.pdf.cell(0, 8, "No data available.", ln=True)
            return

        display_df = df.head(max_rows)
        n_cols = min(len(display_df.columns), 6)
        col_width = 180 / n_cols

        self.pdf.set_font("Helvetica", "B", 8)
        for col in display_df.columns[:n_cols]:
            self.pdf.cell(col_width, 7, str(col)[:20], border=1, align="C")
        self.pdf.ln()

        self.pdf.set_font("Helvetica", "", 7)
        for _, row in display_df.iterrows():
            for col in display_df.columns[:n_cols]:
                val = str(row[col])[:20]
                self.pdf.cell(col_width, 6, val, border=1)
            self.pdf.ln()

    def save(self, path: str | Path) -> str:
        path = str(path)
        self.pdf.output(path)
        return path

    def to_bytes(self) -> bytes:
        result = self.pdf.output()
        # fpdf2 returns bytearray; Streamlit needs bytes
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
