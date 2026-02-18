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
        # Temporarily disable auto page break so writing near the bottom
        # doesn't create a spurious blank page.
        self.pdf.set_auto_page_break(auto=False)
        self.pdf.set_y(-20)
        self._draw_gold_rule(width=170)
        self.pdf.ln(3)
        self.pdf.set_font("Helvetica", "", 7)
        self.pdf.set_text_color(100, 100, 100)
        self.pdf.cell(0, 5, "AI for Finance | MGMT 69000-119 | Purdue University", align="L")
        self.pdf.cell(0, 5, f"Generated {datetime.now():%Y-%m-%d %H:%M}", align="R", ln=True)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.set_auto_page_break(auto=True, margin=15)

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

    # ── trade ideas ────────────────────────────────────────────────────
    def add_trade_ideas(self, cards: list, regime_state: dict | None = None) -> None:
        """Add a full Trade Ideas section with payout graphs and detailed textual logic.

        Parameters
        ----------
        cards : list[TradeCard]
            Trade cards from generate_all_trades().
        regime_state : dict, optional
            Regime state snapshot for summary context.
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

        # --- Summary page ---
        self.pdf.add_page()
        self._draw_header_bar()
        self.pdf.ln(14)
        self.pdf.set_font("Helvetica", "B", 18)
        self.pdf.cell(0, 12, "Trade Ideas", ln=True)
        self.pdf.ln(2)
        self._draw_gold_rule(width=60)
        self.pdf.ln(6)

        # Executive summary
        self.pdf.set_font("Helvetica", "B", 10)
        self.pdf.set_text_color(*_AGED_GOLD)
        self.pdf.cell(0, 6, "EXECUTIVE SUMMARY", ln=True)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.ln(2)

        rp = regime_state.get("regime_prob", 0.5) if regime_state else 0.5
        regime_word = "repricing" if rp > 0.5 else "suppressed"
        self.pdf.set_font("Helvetica", "", 9)
        exec_text = (
            f"The framework has generated {len(cards)} trade idea{'s' if len(cards) != 1 else ''} "
            f"under a {regime_word} regime environment (regime probability: {rp:.1%}). "
            f"The ideas span {len(categories)} categor{'ies' if len(categories) != 1 else 'y'}: "
            f"{', '.join(categories)}. "
            f"Of these, {n_high} {'are' if n_high != 1 else 'is'} high conviction (>=70%), "
            f"{n_med} medium (40-69%), and {n_low} low (<40%)."
        )
        self.pdf.multi_cell(0, 5, exec_text)
        self.pdf.ln(2)

        if top:
            self.pdf.set_font("Helvetica", "B", 9)
            self.pdf.multi_cell(0, 5,
                f"Lead Recommendation: {top.name} ({top.direction.upper()}, "
                f"{top.conviction:.0%} conviction) on {', '.join(top.instruments[:3])}."
            )
            self.pdf.ln(1)
            self.pdf.set_font("Helvetica", "", 9)
            self.pdf.multi_cell(0, 5,
                f"Rationale: {top.edge_source}. "
                f"This trade activates when: {top.regime_condition}. "
                f"Entry signal: {top.entry_signal}. "
                f"Exit signal: {top.exit_signal}."
            )
        self.pdf.ln(3)

        # Regime context box
        if regime_state:
            self.pdf.set_font("Helvetica", "B", 10)
            self.pdf.set_text_color(*_AGED_GOLD)
            self.pdf.cell(0, 6, "MARKET REGIME CONTEXT", ln=True)
            self.pdf.set_text_color(0, 0, 0)
            self.pdf.ln(2)
            self.pdf.set_font("Helvetica", "", 9)
            regime_text = (
                f"Current regime probability is {rp:.1%}, indicating a "
                f"{'strong repricing' if rp > 0.7 else 'moderate repricing' if rp > 0.5 else 'suppressed'} "
                f"environment. "
            )
            if rp > 0.7:
                regime_text += (
                    "In a strong repricing regime, directional short-JGB and volatility-long trades "
                    "tend to perform best. Carry trades face elevated risk of sharp unwind. "
                    "Position sizes should reflect the heightened conviction from aligned signals."
                )
            elif rp > 0.5:
                regime_text += (
                    "A moderate repricing regime suggests transitional conditions. Relative value "
                    "and spread trades may outperform outright directional bets. Consider hedged "
                    "positions and asymmetric option strategies to manage regime uncertainty."
                )
            else:
                regime_text += (
                    "A suppressed regime favors carry and yield-enhancement strategies. "
                    "Volatility selling (short straddles/strangles) can capture range-bound premium. "
                    "Watch for early warning signals of regime shift before adding to carry exposure."
                )
            self.pdf.multi_cell(0, 5, regime_text)
            self.pdf.ln(3)

        # Risk allocation guidance
        self.pdf.set_font("Helvetica", "B", 10)
        self.pdf.set_text_color(*_AGED_GOLD)
        self.pdf.cell(0, 6, "RISK ALLOCATION GUIDANCE", ln=True)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.ln(2)
        self.pdf.set_font("Helvetica", "", 9)
        risk_text = (
            f"With {n_high} high-conviction trade{'s' if n_high != 1 else ''}, "
            f"allocate the majority of risk budget to these ideas first. "
        )
        if n_high >= 2:
            risk_text += (
                "Multiple high-conviction signals suggest strong model consensus - "
                "consider concentrating 60-70% of risk budget on these positions. "
            )
        elif n_high == 1:
            risk_text += (
                "A single high-conviction trade suggests focused opportunity - "
                "allocate up to 40% of risk budget here with the remainder spread across medium-conviction ideas. "
            )
        else:
            risk_text += (
                "No high-conviction trades are present. This is a signal for conservative positioning - "
                "keep individual position sizes small and focus on relative-value structures. "
            )
        risk_text += (
            "Medium-conviction ideas (40-69%) should carry standard position sizes. "
            "Low-conviction ideas (<40%) should be monitored but not actively traded until signals strengthen. "
            "Always read the failure scenario for each trade before sizing."
        )
        self.pdf.multi_cell(0, 5, risk_text)
        self.pdf.ln(4)

        # Summary table
        self.pdf.set_font("Helvetica", "B", 10)
        self.pdf.set_text_color(*_AGED_GOLD)
        self.pdf.cell(0, 6, "TRADE SUMMARY TABLE", ln=True)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.ln(2)
        self.pdf.set_font("Helvetica", "B", 9)
        col_w = [55, 22, 22, 50, 41]
        headers = ["Trade", "Dir", "Conv.", "Instruments", "Category"]
        for w, h in zip(col_w, headers):
            self.pdf.cell(w, 7, h, border=1, align="C")
        self.pdf.ln()
        self.pdf.set_font("Helvetica", "", 8)
        for card in sorted_cards:
            self.pdf.cell(col_w[0], 6, card.name[:28], border=1)
            self.pdf.cell(col_w[1], 6, card.direction.upper(), border=1, align="C")
            self.pdf.cell(col_w[2], 6, f"{card.conviction:.0%}", border=1, align="C")
            self.pdf.cell(col_w[3], 6, ", ".join(card.instruments[:2])[:26], border=1)
            self.pdf.cell(col_w[4], 6, card.category, border=1, align="C")
            self.pdf.ln()
        self._add_page_footer()

        # --- Individual trade card pages ---
        for card in sorted_cards:
            self._add_trade_card_page(card)

    def _add_trade_card_page(self, card) -> None:
        """Render a single trade card as a full PDF page with payout graph and textual analysis."""
        import numpy as np

        self.pdf.add_page()
        self._draw_header_bar()
        self.pdf.ln(14)

        # Title bar
        dir_label = "LONG" if card.direction == "long" else "SHORT"
        self.pdf.set_font("Helvetica", "B", 14)
        self.pdf.cell(0, 10, f"{dir_label}  |  {card.name}  |  {card.conviction:.0%} Conviction", ln=True)
        self.pdf.ln(2)
        self._draw_gold_rule()
        self.pdf.ln(4)

        # --- Trade thesis narrative ---
        self.pdf.set_font("Helvetica", "B", 9)
        self.pdf.set_text_color(*_AGED_GOLD)
        self.pdf.cell(0, 6, "TRADE THESIS", ln=True)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.ln(1)
        self.pdf.set_font("Helvetica", "", 8)
        conv_word = "high" if card.conviction >= 0.7 else "moderate" if card.conviction >= 0.4 else "low"
        thesis = (
            f"This is a {conv_word}-conviction {card.direction} trade on "
            f"{', '.join(card.instruments[:3])} in the {card.category.replace('_', ' ')} space. "
            f"The trade is driven by: {card.edge_source}. "
            f"It activates under the following regime condition: {card.regime_condition}. "
        )
        if card.conviction >= 0.7:
            thesis += (
                "At this conviction level, multiple quantitative signals are aligned. "
                "This represents a strong opportunity where model consensus is high and the "
                "risk/reward profile favors aggressive sizing relative to other ideas in the portfolio."
            )
        elif card.conviction >= 0.4:
            thesis += (
                "At moderate conviction, some signals agree but not all models confirm the thesis. "
                "Standard position sizing is appropriate. Monitor the regime condition closely "
                "for strengthening or weakening signals before adjusting exposure."
            )
        else:
            thesis += (
                "Low conviction indicates mixed or weak signals. This idea should be on a watchlist "
                "rather than actively traded. Wait for additional confirming signals before deploying capital."
            )
        self.pdf.multi_cell(0, 4.5, thesis)
        self.pdf.ln(3)

        # --- Trade specification ---
        self.pdf.set_font("Helvetica", "B", 9)
        self.pdf.set_text_color(*_AGED_GOLD)
        self.pdf.cell(0, 6, "TRADE SPECIFICATION", ln=True)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.ln(2)

        fields = [
            ("Category", card.category.replace("_", " ").title()),
            ("Instruments", ", ".join(card.instruments)),
            ("Regime Condition", card.regime_condition),
            ("Edge Source", card.edge_source),
            ("Entry Signal", card.entry_signal),
            ("Exit Signal", card.exit_signal),
            ("Sizing Method", card.sizing_method),
        ]
        for label, value in fields:
            self.pdf.set_font("Helvetica", "B", 8)
            self.pdf.cell(35, 5, label + ":", align="R")
            self.pdf.set_font("Helvetica", "", 8)
            self.pdf.cell(3, 5, "")  # spacer
            x_start = self.pdf.get_x()
            y_start = self.pdf.get_y()
            self.pdf.multi_cell(152, 5, value[:200])
            self.pdf.ln(1)

        # --- Entry/exit logic explanation ---
        self.pdf.ln(2)
        self.pdf.set_font("Helvetica", "B", 9)
        self.pdf.set_text_color(*_AGED_GOLD)
        self.pdf.cell(0, 6, "EXECUTION LOGIC", ln=True)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.ln(1)
        self.pdf.set_font("Helvetica", "", 8)
        exec_logic = (
            f"Entry: {card.entry_signal}. Once the entry condition is met, initiate the "
            f"{card.direction} position using {card.sizing_method} sizing methodology. "
            f"Exit: {card.exit_signal}. "
            f"The position should be unwound when the exit signal triggers, regardless of P&L at that point. "
            f"Discipline in following pre-defined exit rules is critical to risk management."
        )
        self.pdf.multi_cell(0, 4.5, exec_logic)
        self.pdf.ln(2)

        # --- Failure scenario (red highlight) ---
        self.pdf.set_fill_color(255, 240, 240)
        self.pdf.set_font("Helvetica", "B", 8)
        self.pdf.set_text_color(180, 30, 30)
        self.pdf.cell(0, 6, "  FAILURE SCENARIO  —  READ BEFORE ENTERING TRADE", ln=True, fill=True)
        self.pdf.set_text_color(60, 20, 20)
        self.pdf.set_font("Helvetica", "", 8)
        self.pdf.multi_cell(0, 5, "  " + card.failure_scenario[:400])
        self.pdf.set_text_color(80, 80, 80)
        self.pdf.set_font("Helvetica", "I", 7)
        self.pdf.multi_cell(0, 4,
            "  If this failure scenario is already playing out or is likely imminent, "
            "skip this trade regardless of conviction score. The failure scenario is the "
            "single most important risk check before deployment."
        )
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.ln(3)

        # Strike prices / key levels from metadata
        meta = card.metadata or {}
        levels = {}
        for key in ["jp10_level", "target_yield", "stop_yield", "usdjpy_spot",
                     "target", "stop", "put_strike", "straddle_strike",
                     "payer_strike", "call_strike", "receiver_strike",
                     "atm_strike", "otm_strike", "breakeven",
                     "spread_bps", "target_spread_bps"]:
            if key in meta and meta[key] is not None:
                nice_key = key.replace("_", " ").title()
                levels[nice_key] = meta[key]

        if levels:
            self.pdf.set_font("Helvetica", "B", 9)
            self.pdf.set_text_color(*_AGED_GOLD)
            self.pdf.cell(0, 6, "KEY LEVELS & STRIKE PRICES", ln=True)
            self.pdf.set_text_color(0, 0, 0)
            self.pdf.ln(1)
            self.pdf.set_font("Helvetica", "B", 8)
            for k, v in levels.items():
                self.pdf.cell(50, 5, k, border="B")
                self.pdf.set_font("Helvetica", "", 8)
                fmt = f"{v:,.2f}" if isinstance(v, float) else str(v)
                self.pdf.cell(40, 5, fmt, border="B", ln=True)
                self.pdf.set_font("Helvetica", "B", 8)
            self.pdf.ln(2)

            # Textual explanation of key levels
            self.pdf.set_font("Helvetica", "I", 7)
            self.pdf.set_text_color(80, 80, 80)
            level_text = self._explain_key_levels(card, meta, levels)
            self.pdf.multi_cell(0, 4, level_text)
            self.pdf.set_text_color(0, 0, 0)
            self.pdf.ln(3)

        # Payout graph
        payout_path = self._generate_payout_graph(card)
        if payout_path:
            self.pdf.set_font("Helvetica", "B", 9)
            self.pdf.set_text_color(*_AGED_GOLD)
            self.pdf.cell(0, 6, "ESTIMATED PAYOUT PROFILE", ln=True)
            self.pdf.set_text_color(0, 0, 0)
            self.pdf.ln(1)

            # Payout interpretation text
            self.pdf.set_font("Helvetica", "", 7)
            self.pdf.set_text_color(80, 80, 80)
            payout_text = self._explain_payout(card, meta)
            self.pdf.multi_cell(0, 4, payout_text)
            self.pdf.set_text_color(0, 0, 0)
            self.pdf.ln(2)

            try:
                self.pdf.image(payout_path, x=15, w=180)
            except Exception:
                pass
            try:
                Path(payout_path).unlink(missing_ok=True)
            except Exception:
                pass
            self.pdf.ln(2)
            self.pdf.set_font("Helvetica", "I", 7)
            self.pdf.set_text_color(120, 120, 120)
            self.pdf.multi_cell(0, 4,
                "Note: Payout profiles are estimated using proxy premium assumptions and are for "
                "illustrative purposes only. Actual P&L will depend on market conditions, execution "
                "prices, and volatility at the time of trade entry. Always verify with live pricing."
            )
            self.pdf.set_text_color(0, 0, 0)

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
                f"The entry level is {entry:.3f}%, targeting {target:.2f}% "
                f"with a stop at {stop:.2f}%. This gives an approximate risk/reward ratio "
                f"of {rr:.1f}:1. "
            )
            if rr >= 2:
                parts.append("The risk/reward is favorable (>=2:1). ")
            elif rr >= 1:
                parts.append("The risk/reward is acceptable (>=1:1) but not exceptional. ")
            else:
                parts.append("The risk/reward is unfavorable (<1:1) - consider tighter sizing. ")
        if "usdjpy_spot" in meta:
            parts.append(f"USDJPY spot reference: {meta['usdjpy_spot']:.2f}. ")
        if "straddle_strike" in meta:
            K = meta["straddle_strike"]
            parts.append(
                f"Straddle struck at {K:.1f}. Maximum loss is the total premium paid. "
                f"Breakeven points are approximately {K:.1f} +/- premium on each side."
            )
        if "atm_strike" in meta and "otm_strike" in meta:
            parts.append(
                f"Payer spread: buy at ATM strike {meta['atm_strike']:.3f}%, "
                f"sell at OTM strike {meta['otm_strike']:.3f}%. "
                f"Max gain capped at the spread width minus net premium. Max loss is net premium paid."
            )
        if "spread_bps" in meta and "target_spread_bps" in meta:
            parts.append(
                f"Entry spread: {meta['spread_bps']:.0f} bps, targeting {meta['target_spread_bps']:.0f} bps. "
                f"P&L is linear in spread movement."
            )
        return "".join(parts) if parts else "Key levels define the trade's entry, target, and risk boundaries."

    def _explain_payout(self, card, meta: dict) -> str:
        """Generate textual interpretation of the payout profile."""
        if "straddle_strike" in meta:
            return (
                "This straddle profits when the underlying moves significantly in either direction. "
                "The gold-shaded region shows profit zones; the red-shaded region shows the loss zone around "
                "the strike where the combined premium cost exceeds the intrinsic value. The trade is "
                "market-neutral and benefits from realized volatility exceeding implied volatility."
            )
        if "atm_strike" in meta and "otm_strike" in meta:
            return (
                "This payer spread has a capped upside (max gain at or above the OTM strike) and a defined "
                "max loss (net premium paid). The gold region shows profits when rates rise above the lower "
                "strike plus premium. This is a directional bet on higher rates with limited downside."
            )
        if "call_strike" in meta and "put_strike" in meta:
            return (
                "This short strangle collects premium by selling both a call and a put. The gold region "
                "between the strikes shows the profit zone where neither option is exercised. Losses are "
                "theoretically unlimited beyond the breakeven points. This trade profits from low volatility."
            )
        if "put_strike" in meta and "usdjpy_spot" in meta:
            return (
                "This long put profits when the underlying falls below the strike price minus the premium paid. "
                "Maximum loss is limited to the premium. The gold region shows the profit zone on a decline; "
                "the red region shows the fixed premium cost if the underlying stays above the strike."
            )
        if "target_yield" in meta and "stop_yield" in meta:
            return (
                f"This is a linear {'short' if card.direction == 'short' else 'long'} yield trade. "
                f"P&L scales linearly with yield movement. The green dashed line marks the target; "
                f"the red dashed line marks the stop-loss level. The gold-shaded area represents the "
                f"profitable zone; the red-shaded area represents the loss zone."
            )
        if "target" in meta and "stop" in meta and "usdjpy_spot" in meta:
            return (
                f"This is a linear {card.direction} FX trade on USDJPY. P&L is proportional to the "
                f"percentage move from spot. Target and stop levels define the expected range. "
                f"Gold shading shows profit, red shows loss."
            )
        if "spread_bps" in meta and "target_spread_bps" in meta:
            return (
                "This is a spread trade where P&L depends on the change in the basis-point spread. "
                "The trade profits when the spread moves toward the target level."
            )
        return "The chart below shows the estimated payout profile for this trade structure."

    def _generate_payout_graph(self, card) -> Optional[str]:
        """Generate a payout diagram as a temp PNG. Returns path or None."""
        import numpy as np
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker
        except ImportError:
            return None

        meta = card.metadata or {}
        fig, ax = plt.subplots(figsize=(7, 2.8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#fafaf8")

        generated = False

        # --- Options payout: straddle ---
        if "straddle_strike" in meta:
            K = meta["straddle_strike"]
            premium = abs(K) * 0.015  # ~1.5% of strike as proxy
            x = np.linspace(K - K * 0.05, K + K * 0.05, 200)
            call_payout = np.maximum(x - K, 0) - premium / 2
            put_payout = np.maximum(K - x, 0) - premium / 2
            total = call_payout + put_payout
            ax.plot(x, total, color="#000000", linewidth=2, label="Straddle P&L")
            ax.fill_between(x, total, 0, where=total > 0, color="#CFB991", alpha=0.3)
            ax.fill_between(x, total, 0, where=total < 0, color="#c0392b", alpha=0.15)
            ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
            ax.axvline(K, color="#CFB991", linewidth=1, linestyle=":", label=f"Strike {K:.1f}")
            ax.set_xlabel("Underlying Price")
            ax.set_ylabel("P&L")
            ax.set_title(f"{card.name} — Straddle Payout at Strike {K:.1f}", fontsize=10, fontweight="bold")
            generated = True

        # --- Options payout: payer spread ---
        elif "atm_strike" in meta and "otm_strike" in meta:
            K1 = meta["atm_strike"]
            K2 = meta["otm_strike"]
            premium = abs(K2 - K1) * 0.4  # proxy net debit
            x = np.linspace(K1 - abs(K2 - K1) * 2, K2 + abs(K2 - K1) * 2, 200)
            long_call = np.maximum(x - K1, 0)
            short_call = np.maximum(x - K2, 0)
            total = long_call - short_call - premium
            ax.plot(x, total, color="#000000", linewidth=2, label="Payer Spread P&L")
            ax.fill_between(x, total, 0, where=total > 0, color="#CFB991", alpha=0.3)
            ax.fill_between(x, total, 0, where=total < 0, color="#c0392b", alpha=0.15)
            ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
            ax.axvline(K1, color="#CFB991", linewidth=1, linestyle=":", label=f"Buy {K1:.3f}%")
            ax.axvline(K2, color="#c0392b", linewidth=1, linestyle=":", label=f"Sell {K2:.3f}%")
            ax.set_xlabel("Swap Rate (%)")
            ax.set_ylabel("P&L (bps)")
            ax.set_title(f"{card.name} — Payer Spread Payout", fontsize=10, fontweight="bold")
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
            ax.plot(x, total, color="#000000", linewidth=2, label="Short Strangle P&L")
            ax.fill_between(x, total, 0, where=total > 0, color="#CFB991", alpha=0.3)
            ax.fill_between(x, total, 0, where=total < 0, color="#c0392b", alpha=0.15)
            ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
            ax.axvline(Kp, color="#2e7d32", linewidth=1, linestyle=":", label=f"Put {Kp:.2f}")
            ax.axvline(Kc, color="#c0392b", linewidth=1, linestyle=":", label=f"Call {Kc:.2f}")
            ax.set_xlabel("Underlying Price")
            ax.set_ylabel("P&L")
            ax.set_title(f"{card.name} — Short Strangle Payout", fontsize=10, fontweight="bold")
            generated = True

        # --- Options payout: single put ---
        elif "put_strike" in meta and "usdjpy_spot" in meta:
            K = meta["put_strike"]
            spot = meta["usdjpy_spot"]
            premium = abs(spot - K) * 0.15
            x = np.linspace(K * 0.94, spot * 1.04, 200)
            total = np.maximum(K - x, 0) - premium
            ax.plot(x, total, color="#000000", linewidth=2, label="Long Put P&L")
            ax.fill_between(x, total, 0, where=total > 0, color="#CFB991", alpha=0.3)
            ax.fill_between(x, total, 0, where=total < 0, color="#c0392b", alpha=0.15)
            ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
            ax.axvline(K, color="#CFB991", linewidth=1, linestyle=":", label=f"Strike {K:.0f}")
            ax.axvline(spot, color="#000", linewidth=1, linestyle="-", alpha=0.4, label=f"Spot {spot:.0f}")
            ax.set_xlabel("USDJPY")
            ax.set_ylabel("P&L per unit")
            ax.set_title(f"{card.name} — Put Payout (K={K:.0f})", fontsize=10, fontweight="bold")
            generated = True

        # --- Linear: directional with target/stop ---
        elif "target_yield" in meta and "stop_yield" in meta:
            entry = meta.get("jp10_level", 1.0)
            target = meta["target_yield"]
            stop = meta["stop_yield"]
            x = np.linspace(min(stop, entry) - 0.1, max(target, entry) + 0.1, 200)
            if card.direction == "short":
                pnl = (entry - x) * 100  # bps
            else:
                pnl = (x - entry) * 100
            ax.plot(x, pnl, color="#000000", linewidth=2, label="P&L (bps)")
            ax.fill_between(x, pnl, 0, where=pnl > 0, color="#CFB991", alpha=0.3)
            ax.fill_between(x, pnl, 0, where=pnl < 0, color="#c0392b", alpha=0.15)
            ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
            ax.axvline(entry, color="#000", linewidth=1.2, label=f"Entry {entry:.3f}%")
            ax.axvline(target, color="#2e7d32", linewidth=1.2, linestyle="--", label=f"Target {target:.2f}%")
            ax.axvline(stop, color="#c0392b", linewidth=1.2, linestyle="--", label=f"Stop {stop:.2f}%")
            ax.set_xlabel("Yield (%)")
            ax.set_ylabel("P&L (bps)")
            ax.set_title(f"{card.name} — {card.direction.upper()} P&L Profile", fontsize=10, fontweight="bold")
            generated = True

        # --- Linear: USDJPY with target/stop ---
        elif "target" in meta and "stop" in meta and "usdjpy_spot" in meta:
            spot = meta["usdjpy_spot"]
            target = meta["target"]
            stop = meta["stop"]
            x = np.linspace(stop * 0.98, target * 1.02, 200)
            if card.direction == "long":
                pnl = (x - spot) / spot * 100  # % P&L
            else:
                pnl = (spot - x) / spot * 100
            ax.plot(x, pnl, color="#000000", linewidth=2, label="P&L (%)")
            ax.fill_between(x, pnl, 0, where=pnl > 0, color="#CFB991", alpha=0.3)
            ax.fill_between(x, pnl, 0, where=pnl < 0, color="#c0392b", alpha=0.15)
            ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
            ax.axvline(spot, color="#000", linewidth=1.2, label=f"Spot {spot:.2f}")
            ax.axvline(target, color="#2e7d32", linewidth=1.2, linestyle="--", label=f"Target {target:.2f}")
            ax.axvline(stop, color="#c0392b", linewidth=1.2, linestyle="--", label=f"Stop {stop:.2f}")
            ax.set_xlabel("USDJPY")
            ax.set_ylabel("P&L (%)")
            ax.set_title(f"{card.name} — {card.direction.upper()} P&L Profile", fontsize=10, fontweight="bold")
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
            ax.plot(x, pnl, color="#000000", linewidth=2, label="Spread P&L (bps)")
            ax.fill_between(x, pnl, 0, where=pnl > 0, color="#CFB991", alpha=0.3)
            ax.fill_between(x, pnl, 0, where=pnl < 0, color="#c0392b", alpha=0.15)
            ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
            ax.axvline(entry, color="#000", linewidth=1.2, label=f"Entry {entry:.0f} bps")
            ax.axvline(target, color="#2e7d32", linewidth=1.2, linestyle="--", label=f"Target {target:.0f} bps")
            ax.set_xlabel("Spread (bps)")
            ax.set_ylabel("P&L (bps)")
            ax.set_title(f"{card.name} — Spread P&L", fontsize=10, fontweight="bold")
            generated = True

        if not generated:
            plt.close(fig)
            return None

        ax.legend(fontsize=7, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(tmp.name, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return tmp.name

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
