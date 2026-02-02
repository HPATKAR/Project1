"""Structured trade output format for the JGB repricing framework.

Every trade idea produced by the strategy layer is represented as a
``TradeCard`` dataclass.  Cards carry the full context needed to evaluate,
size, and execute a trade -- from the regime condition that triggers entry
to the failure scenario that would invalidate the thesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import pandas as pd


# ------------------------------------------------------------------
# Valid category / direction literals
# ------------------------------------------------------------------
VALID_CATEGORIES = {"rates", "fx", "volatility", "cross_asset"}
VALID_DIRECTIONS = {"long", "short"}


# ------------------------------------------------------------------
# TradeCard dataclass
# ------------------------------------------------------------------
@dataclass
class TradeCard:
    """Structured representation of a single trade idea.

    Parameters
    ----------
    name : str
        Short human-readable name (e.g. "JGB 10Y Short").
    category : str
        One of ``"rates"``, ``"fx"``, ``"volatility"``, ``"cross_asset"``.
    direction : str
        ``"long"`` or ``"short"``.
    instruments : list[str]
        Tradable instrument identifiers (e.g. ``["JGB 10Y Future"]``).
    regime_condition : str
        Plain-text description of the regime state that triggers this
        trade (e.g. ``"regime_prob > 0.7 and entropy_rising"``).
    edge_source : str
        What gives this trade an informational or structural edge.
    entry_signal : str
        Concrete entry trigger description.
    exit_signal : str
        Concrete exit / take-profit / stop-loss description.
    failure_scenario : str
        What makes this trade lose -- the key risk to monitor.
    sizing_method : str
        Name of the sizing approach (e.g. ``"vol_target"``).
    conviction : float
        Conviction score in [0, 1].
    metadata : dict
        Optional bag of additional data (timestamps, model outputs, etc.).
    """

    name: str
    category: str
    direction: str
    instruments: list[str]
    regime_condition: str
    edge_source: str
    entry_signal: str
    exit_signal: str
    failure_scenario: str
    sizing_method: str
    conviction: float
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category not in VALID_CATEGORIES:
            raise ValueError(
                f"category must be one of {VALID_CATEGORIES}, got '{self.category}'"
            )
        if self.direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {VALID_DIRECTIONS}, got '{self.direction}'"
            )
        if not 0.0 <= self.conviction <= 1.0:
            raise ValueError(
                f"conviction must be in [0, 1], got {self.conviction}"
            )


# ------------------------------------------------------------------
# Pretty-print
# ------------------------------------------------------------------
def format_trade_card(card: TradeCard) -> str:
    """Return a human-readable, multi-line summary of a TradeCard.

    Parameters
    ----------
    card : TradeCard
        The trade card to format.

    Returns
    -------
    str
        Formatted text block suitable for logging or display.
    """
    bar = "=" * 60
    instruments_str = ", ".join(card.instruments)
    conviction_pct = f"{card.conviction:.0%}"

    lines = [
        bar,
        f"  TRADE CARD: {card.name}",
        bar,
        f"  Category       : {card.category}",
        f"  Direction      : {card.direction.upper()}",
        f"  Instruments    : {instruments_str}",
        f"  Conviction     : {conviction_pct}",
        f"  Sizing Method  : {card.sizing_method}",
        "-" * 60,
        f"  Regime Cond.   : {card.regime_condition}",
        f"  Edge Source    : {card.edge_source}",
        f"  Entry Signal   : {card.entry_signal}",
        f"  Exit Signal    : {card.exit_signal}",
        f"  Failure Scen.  : {card.failure_scenario}",
    ]

    if card.metadata:
        lines.append("-" * 60)
        for key, value in card.metadata.items():
            lines.append(f"  {key}: {value}")

    lines.append(bar)
    return "\n".join(lines)


# ------------------------------------------------------------------
# Batch conversion to DataFrame
# ------------------------------------------------------------------
def trade_cards_to_dataframe(cards: list[TradeCard]) -> pd.DataFrame:
    """Convert a list of TradeCard objects into a pandas DataFrame.

    Each row corresponds to one trade card.  The ``instruments`` list is
    stored as a semicolon-separated string for easier tabular display,
    and ``metadata`` is kept as a dict column.

    Parameters
    ----------
    cards : list[TradeCard]
        Trade cards to convert.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per trade card.  Columns mirror the
        TradeCard fields.
    """
    if not cards:
        return pd.DataFrame()

    records = []
    for card in cards:
        row = asdict(card)
        # Flatten instruments list into a readable string
        row["instruments"] = "; ".join(row["instruments"])
        records.append(row)

    df = pd.DataFrame(records)
    return df
