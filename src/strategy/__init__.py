"""Strategy module for JGB repricing framework.

Provides regime-conditional trade generation, structured trade output,
position sizing, and vectorized backtesting for systematic macro trading
around Japanese Government Bond repricing events.

Sub-modules
-----------
trade_card : Structured trade output format (TradeCard dataclass).
trade_generator : Regime-conditional trade idea generation across rates,
    FX, volatility, and cross-asset spaces.
sizing : Position sizing utilities (vol-targeting, Kelly, regime scaling).
backtester : Vectorized signal backtesting and event-study analysis.
"""
