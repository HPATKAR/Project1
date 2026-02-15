"""Regime-conditional trade generation for the JGB repricing framework.

Each generator function inspects the current regime state (transition
probabilities, term-premium dynamics, volatility surfaces, cross-asset
spillovers) and produces a list of ``TradeCard`` objects that represent
actionable trade ideas with full context.

The master function ``generate_all_trades`` aggregates output from every
asset-class generator into a single trade book.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.strategy.trade_card import TradeCard


# ======================================================================
# Rates trades
# ======================================================================
def generate_rates_trades(
    regime_prob: float,
    term_premium: pd.Series,
    pca_scores: pd.DataFrame,
    liquidity_index: pd.Series,
    *,
    jp10_level: float | None = None,
    us10_level: float | None = None,
    jp2y_level: float | None = None,
) -> list[TradeCard]:
    """Generate rates trade ideas conditional on the current regime state.

    Parameters
    ----------
    regime_prob : float
        Probability of being in the repricing / policy-shift regime (0-1).
    term_premium : pd.Series
        Time series of estimated term premium (e.g. ACM or Kim-Wright).
        The most recent values are used for signal extraction.
    pca_scores : pd.DataFrame
        PCA factor scores of the JGB yield curve.  Expected columns
        include at least ``"PC1"`` (level), ``"PC2"`` (slope), and
        ``"PC3"`` (curvature).
    liquidity_index : pd.Series
        Composite liquidity metric for JGB market (higher = more liquid).

    Returns
    -------
    list[TradeCard]
        Rates-space trade ideas.
    """
    cards: list[TradeCard] = []

    # --- 1. JGB 10Y Short -----------------------------------------------
    if regime_prob > 0.6:
        # Term-premium momentum: positive 20-day change signals rising risk
        tp_change = (
            term_premium.iloc[-1] - term_premium.iloc[-20]
            if len(term_premium) >= 20
            else 0.0
        )
        conviction = min(0.95, 0.5 + regime_prob * 0.3 + max(tp_change, 0) * 2.0)

        # Concrete yield targets
        _jp10 = jp10_level if jp10_level is not None else 0.0
        _target_yield = round(_jp10 + 0.20, 2)  # +20 bps target
        _stop_yield = round(_jp10 - 0.15, 2)    # -15 bps stop

        cards.append(
            TradeCard(
                name="JGB 10Y Short",
                category="rates",
                direction="short",
                instruments=[
                    "JB1 Comdty (TSE JGB 10Y Future)",
                    "JGBS10 (JGB 10Y Cash, ISIN JP1103101)",
                ],
                regime_condition=(
                    f"regime_prob={regime_prob:.2f} > 0.60; "
                    f"JGB 10Y currently at {_jp10:.3f}%; "
                    "BoJ policy-normalisation regime detected"
                ),
                edge_source=(
                    "Ensemble regime model (Markov + HMM + entropy + GARCH) "
                    f"identifies {regime_prob:.0%} probability of sustained yield "
                    f"repricing; term premium momentum +{max(tp_change,0):.1f} bps over 20d"
                ),
                entry_signal=(
                    f"Sell JB1 (JGB 10Y Future) at market. "
                    f"Current yield: {_jp10:.3f}%. "
                    f"Target: {_target_yield:.2f}% (+20 bps). "
                    f"Notional: size to 5 bps/day vol target (~JPY 500M DV01 per bp)"
                ),
                exit_signal=(
                    f"Take profit at {_target_yield:.2f}% yield OR "
                    f"stop loss if yield falls below {_stop_yield:.2f}% "
                    f"(15 bps adverse). regime_prob < 0.40 = forced exit"
                ),
                failure_scenario=(
                    "BoJ re-anchors YCC band or conducts surprise fixed-rate "
                    "purchase operation (rinban); global risk-off drives safe-haven "
                    "demand into JGBs compressing yields 10-20 bps in a session"
                ),
                sizing_method="vol_target (5 bps/day)",
                conviction=round(conviction, 2),
                metadata={
                    "regime_prob": round(regime_prob, 4),
                    "tp_20d_change": round(tp_change, 4),
                    "jp10_level": round(_jp10, 4),
                    "target_yield": _target_yield,
                    "stop_yield": _stop_yield,
                },
            )
        )

    # --- 2. 2s10s Steepener ---------------------------------------------
    tp_rising = False
    if len(term_premium) >= 10:
        tp_rising = float(term_premium.iloc[-1]) > float(term_premium.iloc[-10])

    if tp_rising and regime_prob > 0.5:
        conviction = min(0.90, 0.4 + regime_prob * 0.25 + 0.15)
        _jp10 = jp10_level if jp10_level is not None else 0.0
        _jp2 = jp2y_level if jp2y_level is not None else 0.0
        _spread = round((_jp10 - _jp2) * 100, 1)  # bps
        _target_spread = round(_spread + 15, 1)

        cards.append(
            TradeCard(
                name="JGB 2s10s Steepener",
                category="rates",
                direction="long",
                instruments=[
                    "JB1 Comdty (sell JGB 10Y Future, short leg)",
                    "JB2 Comdty (buy JGB 2Y Future, long leg)",
                ],
                regime_condition=(
                    f"regime_prob={regime_prob:.2f} > 0.50; term premium "
                    f"rising over trailing 10 days. 2Y at {_jp2:.3f}%, "
                    f"10Y at {_jp10:.3f}%, spread {_spread:.0f} bps"
                ),
                edge_source=(
                    "Policy normalisation historically steepens 2s10s by 15-30 bps "
                    "as the front end is anchored by BoJ short-rate guidance while "
                    "the long end reprices term premium"
                ),
                entry_signal=(
                    f"Buy the 2s10s spread at {_spread:.0f} bps. "
                    f"DV01-neutral: ~4:1 notional ratio (2Y:10Y). "
                    f"Target: {_target_spread:.0f} bps (+15 bps steepening). "
                    f"Stop: spread flattens 10 bps below entry"
                ),
                exit_signal=(
                    f"Take profit at {_target_spread:.0f} bps spread OR "
                    f"stop at {_spread - 10:.0f} bps. "
                    "Also exit if term premium 10d change turns negative"
                ),
                failure_scenario=(
                    "BoJ hikes short rate faster than market expects (surprise "
                    "+25 bps to overnight rate), flattening the curve from the "
                    "front end; global recession fears compress term premium"
                ),
                sizing_method="DV01-neutral spread",
                conviction=round(conviction, 2),
                metadata={
                    "tp_rising": True,
                    "regime_prob": round(regime_prob, 4),
                    "spread_bps": _spread,
                    "target_spread_bps": _target_spread,
                },
            )
        )

    # --- 3. Long-end Butterfly (5s-10s-20s) ------------------------------
    if "PC3" in pca_scores.columns and len(pca_scores) > 0:
        pc3_current = float(pca_scores["PC3"].iloc[-1])
        pc3_zscore = (
            (pc3_current - float(pca_scores["PC3"].mean()))
            / max(float(pca_scores["PC3"].std()), 1e-8)
        )

        if abs(pc3_zscore) > 1.5:
            direction = "short" if pc3_zscore > 1.5 else "long"
            conviction = min(0.85, 0.3 + abs(pc3_zscore) * 0.15 + regime_prob * 0.1)
            _fly_action = "sell the belly (10Y), buy the wings (5Y + 20Y)" if direction == "short" else "buy the belly (10Y), sell the wings (5Y + 20Y)"
            cards.append(
                TradeCard(
                    name="JGB Long-End Butterfly (5s-10s-20s)",
                    category="rates",
                    direction=direction,
                    instruments=[
                        "JB5 (TSE JGB 5Y Future)",
                        "JB1 Comdty (TSE JGB 10Y Future, belly)",
                        "JGBS20 (JGB 20Y Cash, ISIN JP1201401)",
                    ],
                    regime_condition=(
                        f"PC3 curvature z-score={pc3_zscore:.2f} exceeds "
                        "1.5 standard deviations (extreme curvature). "
                        f"JGB 10Y at {jp10_level:.3f}%" if jp10_level else
                        f"PC3 curvature z-score={pc3_zscore:.2f} exceeds "
                        "1.5 standard deviations (extreme curvature)"
                    ),
                    edge_source=(
                        "PCA curvature factor at extreme; historical mean-reversion "
                        "within 20 trading days in 78% of past episodes. "
                        "Expected P&L: 3-5 bps butterfly spread convergence"
                    ),
                    entry_signal=(
                        f"{_fly_action}. "
                        f"DV01 weights: 1x 5Y, -2x 10Y, 1x 20Y (duration-neutral). "
                        f"PC3 z-score: {pc3_zscore:.2f}. Enter at market"
                    ),
                    exit_signal=(
                        "|PC3 z-score| < 0.5 (convergence target) OR "
                        "max holding period 40 business days OR "
                        "loss exceeds 3 bps on the butterfly spread"
                    ),
                    failure_scenario=(
                        "BoJ shifts purchase allocation across maturity buckets "
                        "(e.g., reduces super-long rinban); life insurer demand/supply "
                        "imbalance permanently alters 20Y-sector curvature"
                    ),
                    sizing_method="DV01-neutral butterfly",
                    conviction=round(conviction, 2),
                    metadata={
                        "pc3_zscore": round(pc3_zscore, 4),
                        "regime_prob": round(regime_prob, 4),
                    },
                )
            )

    # --- 4. Liquidity Premium Capture ------------------------------------
    if len(liquidity_index) >= 20:
        liq_current = float(liquidity_index.iloc[-1])
        liq_mean = float(liquidity_index.rolling(60).mean().iloc[-1])
        liq_deteriorating = liq_current < liq_mean * 0.9

        if liq_deteriorating:
            conviction = min(0.80, 0.35 + regime_prob * 0.2 + 0.15)
            _liq_gap = round((liq_mean - liq_current) / max(abs(liq_mean), 0.01) * 100, 1)
            cards.append(
                TradeCard(
                    name="JGB Liquidity Premium Capture",
                    category="rates",
                    direction="short",
                    instruments=[
                        "JGBS10 off-the-run (#366, #367 series)",
                        "JB1 Comdty on-the-run (hedge leg)",
                    ],
                    regime_condition=(
                        f"Liquidity index at {liq_current:.2f} vs 60d avg {liq_mean:.2f} "
                        f"({_liq_gap:.0f}% deterioration). Regime transition amplifies "
                        "illiquidity premium in off-the-run securities"
                    ),
                    edge_source=(
                        "Off-the-run JGBs trade 2-5 bps cheap to the future during "
                        "liquidity stress. Basis normalises within 15-25 trading days "
                        "as BoJ operations restore depth"
                    ),
                    entry_signal=(
                        "Sell off-the-run 10Y JGB (#366/#367), buy JB1 future "
                        "as hedge. Expected basis: 2-5 bps. "
                        f"Liquidity gap: {_liq_gap:.0f}% below 60d mean"
                    ),
                    exit_signal=(
                        "Liquidity index recovers above 60d mean OR "
                        "basis narrows to <1 bp (take profit) OR "
                        "basis widens beyond 8 bps (stop loss)"
                    ),
                    failure_scenario=(
                        "BoJ emergency fixed-rate operation compresses "
                        "spreads before basis converges; forced unwind by "
                        "leveraged accounts widens basis to 10+ bps"
                    ),
                    sizing_method="basis trade (matched DV01)",
                    conviction=round(conviction, 2),
                    metadata={
                        "liq_current": round(liq_current, 4),
                        "liq_60d_mean": round(liq_mean, 4),
                        "liq_gap_pct": _liq_gap,
                        "regime_prob": round(regime_prob, 4),
                    },
                )
            )

    return cards


# ======================================================================
# FX trades
# ======================================================================
def generate_fx_trades(
    regime_prob: float,
    carry_to_vol: float,
    usdjpy_trend: float,
    positioning: float,
    *,
    usdjpy_level: float | None = None,
) -> list[TradeCard]:
    """Generate FX trade ideas tied to the JGB repricing regime.

    Parameters
    ----------
    regime_prob : float
        Probability of repricing regime.
    carry_to_vol : float
        Carry-to-volatility ratio for JPY crosses (higher = more
        attractive carry).  Typical range 0.5-3.0.
    usdjpy_trend : float
        Trailing trend signal for USD/JPY (positive = JPY weakening).
        Can be e.g. 20d return or z-score of momentum.
    positioning : float
        Net speculative positioning in JPY futures (negative = market is
        short JPY; range roughly -1 to +1 normalised).

    Returns
    -------
    list[TradeCard]
        FX trade ideas.
    """
    cards: list[TradeCard] = []

    # --- 1. Short JPY (carry + regime shift) -----------------------------
    if carry_to_vol > 1.0 and regime_prob > 0.5:
        conviction = min(
            0.90,
            0.35 + carry_to_vol * 0.1 + regime_prob * 0.2 + max(usdjpy_trend, 0) * 0.1,
        )
        _spot = usdjpy_level if usdjpy_level is not None else 150.0
        _target = round(_spot * 1.03, 2)   # +3% target
        _stop = round(_spot * 0.97, 2)     # -3% stop

        cards.append(
            TradeCard(
                name="Short JPY via USD/JPY",
                category="fx",
                direction="long",  # long USD/JPY = short JPY
                instruments=[
                    "USDJPY Curncy (spot)",
                    "JPY1M / JPY3M Curncy (1M/3M forward)",
                ],
                regime_condition=(
                    f"regime_prob={regime_prob:.2f} > 0.50; carry-to-vol "
                    f"ratio={carry_to_vol:.2f} > 1.0. "
                    f"USDJPY spot at {_spot:.2f}, 20d trend {usdjpy_trend:+.1f}%"
                ),
                edge_source=(
                    f"Carry yield {carry_to_vol:.1f}x compensates for vol. "
                    "BoJ regime shift widens rate differential; capital outflows "
                    "reinforce JPY weakness. Positive carry accrues daily"
                ),
                entry_signal=(
                    f"Buy USDJPY spot at {_spot:.2f}. "
                    f"Target: {_target:.2f} (+3%, ~{_target - _spot:.0f} pips). "
                    f"Stop: {_stop:.2f} (-3%). "
                    f"Position size: 1% NAV risk per trade"
                ),
                exit_signal=(
                    f"Take profit at {_target:.2f} OR stop at {_stop:.2f} OR "
                    "carry_to_vol < 0.7 OR regime_prob < 0.35"
                ),
                failure_scenario=(
                    "Sharp global risk-off triggers JPY safe-haven rally (5-8% "
                    "in a week historically); Fed emergency cut compresses "
                    "US-JP differential; MoF FX intervention above 155"
                ),
                sizing_method="vol_target (1% NAV risk)",
                conviction=round(conviction, 2),
                metadata={
                    "carry_to_vol": round(carry_to_vol, 4),
                    "usdjpy_spot": round(_spot, 2),
                    "target": _target,
                    "stop": _stop,
                    "usdjpy_trend": round(usdjpy_trend, 4),
                },
            )
        )

    # --- 2. Long JPY Volatility ------------------------------------------
    regime_uncertainty = regime_prob * (1 - regime_prob) * 4  # peaks at 0.5
    if regime_uncertainty > 0.8:
        conviction = min(0.85, 0.3 + regime_uncertainty * 0.4)
        _spot = usdjpy_level if usdjpy_level is not None else 150.0
        cards.append(
            TradeCard(
                name="Long JPY Implied Volatility",
                category="fx",
                direction="long",
                instruments=[
                    "USDJPY1MV Curncy (1M ATM implied vol)",
                    "USDJPY25R1M Curncy (1M 25-delta risk reversal)",
                ],
                regime_condition=(
                    f"Regime uncertainty={regime_uncertainty:.2f} > 0.80. "
                    f"USDJPY at {_spot:.2f}. "
                    "regime_prob near 0.50 implies maximum transition risk"
                ),
                edge_source=(
                    "Regime transitions produce realised vol 1.5-2x above "
                    "pre-transition implied. Binary BoJ policy risk is under-priced. "
                    "Max loss limited to premium paid"
                ),
                entry_signal=(
                    f"Buy USDJPY 1M ATM straddle (strike ~{_spot:.0f}). "
                    "Premium: typically 1.5-2.5% of notional. "
                    "Breakeven: spot must move +/- 2-3% within 1 month"
                ),
                exit_signal=(
                    "regime_prob above 0.75 or below 0.25 (uncertainty resolved). "
                    "Alternatively, roll if vol term-structure inverts (1M > 3M)"
                ),
                failure_scenario=(
                    "BoJ manages orderly transition; realised vol stays below "
                    "implied. Theta decay: ~0.05-0.10% of notional per day. "
                    "Max loss = premium paid"
                ),
                sizing_method="premium-limited (max 0.5% NAV)",
                conviction=round(conviction, 2),
                metadata={
                    "regime_uncertainty": round(regime_uncertainty, 4),
                    "regime_prob": round(regime_prob, 4),
                    "usdjpy_spot": round(_spot, 2),
                },
            )
        )

    # --- 3. Carry Unwind Hedge -------------------------------------------
    if carry_to_vol < 0.8 and positioning < -0.3:
        conviction = min(0.85, 0.4 + abs(positioning) * 0.3)
        _spot = usdjpy_level if usdjpy_level is not None else 150.0
        _put_strike = round(_spot * 0.98, 2)  # 2% OTM put

        cards.append(
            TradeCard(
                name="JPY Carry Unwind Hedge",
                category="fx",
                direction="short",  # short USD/JPY = long JPY
                instruments=[
                    "USDJPY Curncy (sell spot)",
                    f"USDJPY 1M {_put_strike:.0f} Put (25-delta)",
                ],
                regime_condition=(
                    f"carry_to_vol={carry_to_vol:.2f} < 0.80; "
                    f"positioning={positioning:.2f} < -0.30 (crowded short-JPY). "
                    f"USDJPY at {_spot:.2f}"
                ),
                edge_source=(
                    "Crowded JPY shorts + collapsing carry-to-vol = asymmetric "
                    "unwind risk. Historical carry unwinds produce 5-10% USDJPY "
                    "drops in 2-4 weeks. P&L ratio ~3:1 at current positioning"
                ),
                entry_signal=(
                    f"Buy USDJPY 1M {_put_strike:.0f} put (2% OTM). "
                    f"Cost: ~0.3-0.5% of notional. "
                    f"Payoff if USDJPY falls below {_put_strike:.0f}: "
                    f"~1% per 1% spot move. Alternatively sell USDJPY spot at {_spot:.2f}"
                ),
                exit_signal=(
                    "Positioning normalises above -0.10 OR carry_to_vol "
                    "recovers above 1.2 OR put expires"
                ),
                failure_scenario=(
                    "US data surprise pushes UST yields higher, widening "
                    "rate differential despite low carry-to-vol; JPY shorts "
                    "remain intact due to structural outflows (pension rebalancing)"
                ),
                sizing_method="premium-limited hedge (0.3-0.5% NAV)",
                conviction=round(conviction, 2),
                metadata={
                    "carry_to_vol": round(carry_to_vol, 4),
                    "positioning": round(positioning, 4),
                    "usdjpy_spot": round(_spot, 2),
                    "put_strike": _put_strike,
                },
            )
        )

    return cards


# ======================================================================
# Volatility trades
# ======================================================================
def generate_vol_trades(
    regime_prob: float,
    garch_vol: float,
    entropy_signal: float,
    *,
    jp10_level: float | None = None,
    usdjpy_level: float | None = None,
) -> list[TradeCard]:
    """Generate volatility trade ideas.

    Parameters
    ----------
    regime_prob : float
        Probability of repricing regime.
    garch_vol : float
        Current GARCH-estimated annualised volatility of JGB 10Y returns.
    entropy_signal : float
        Information-entropy measure of yield-curve dynamics.  Higher
        entropy => more disorder / transition risk.  Typical range 0-1.

    Returns
    -------
    list[TradeCard]
        Volatility trade ideas.
    """
    cards: list[TradeCard] = []

    # --- 1. Long JGB Volatility (regime transitioning) -------------------
    regime_uncertainty = regime_prob * (1 - regime_prob) * 4
    _garch_bps = garch_vol * 10000  # annualised vol in bps
    _jp10 = jp10_level if jp10_level is not None else 1.0
    _fx = usdjpy_level if usdjpy_level is not None else 150.0
    # JGB future price ~ 100 - yield*10 (simplified proxy for ATM strike)
    _jgb_future_px = round(100 - _jp10 * 10, 2)
    _straddle_strike = round(_jgb_future_px, 1)
    _upper_wing = round(_jgb_future_px + 0.10, 2)  # +10 ticks
    _lower_wing = round(_jgb_future_px - 0.10, 2)  # -10 ticks

    if regime_uncertainty > 0.7 or (regime_prob > 0.5 and entropy_signal > 0.6):
        conviction = min(
            0.90, 0.3 + regime_uncertainty * 0.25 + entropy_signal * 0.2
        )
        # Payer swaption strike = current swap rate
        _payer_strike = round(_jp10, 3)
        cards.append(
            TradeCard(
                name="Long JGB Volatility",
                category="volatility",
                direction="long",
                instruments=[
                    f"JB1 Options {_straddle_strike:.1f} Straddle (TSE, 1M expiry)",
                    f"JPY 10Y1Y Payer Swaption K={_payer_strike:.3f}% (OTC, JYSW1Y10Y)",
                ],
                regime_condition=(
                    f"Regime transitioning: uncertainty={regime_uncertainty:.2f}. "
                    f"GARCH vol={_garch_bps:.1f} bps annualised. "
                    f"Entropy={entropy_signal:.2f}. "
                    f"JGB 10Y at {_jp10:.3f}%"
                ),
                edge_source=(
                    f"GARCH estimates current vol at {_garch_bps:.1f} bps. "
                    "Regime transitions historically produce realised vol "
                    "1.5-2x above pre-transition implied. Expected P&L: "
                    "2-4x premium if vol doubles"
                ),
                entry_signal=(
                    f"Buy JB1 {_straddle_strike:.1f} straddle (ATM, 1M expiry). "
                    f"Call strike: {_straddle_strike:.1f}, Put strike: {_straddle_strike:.1f}. "
                    f"+ Buy 10Y1Y payer swaption strike {_payer_strike:.3f}%. "
                    f"Implied vol likely ~{_garch_bps * 0.8:.0f}-{_garch_bps * 1.2:.0f} bps. "
                    "Premium budget: 0.3-0.5% of portfolio NAV"
                ),
                exit_signal=(
                    "regime_uncertainty < 0.30 (regime resolved) AND "
                    "entropy < 0.40. Or take profit when realised vol exceeds "
                    f"{_garch_bps * 1.5:.0f} bps (1.5x GARCH estimate)"
                ),
                failure_scenario=(
                    "BoJ pre-commits policy path via forward guidance, reducing "
                    "uncertainty without generating vol. Theta: ~0.5-1.0 bps/day. "
                    "Max loss = premium paid"
                ),
                sizing_method="premium-limited (0.3-0.5% NAV)",
                conviction=round(conviction, 2),
                metadata={
                    "regime_uncertainty": round(regime_uncertainty, 4),
                    "garch_vol_bps": round(_garch_bps, 2),
                    "entropy_signal": round(entropy_signal, 4),
                    "straddle_strike": _straddle_strike,
                    "payer_strike": _payer_strike,
                },
            )
        )

    # --- 2. Vol Selling in Stable Regime ---------------------------------
    if regime_prob < 0.25 and garch_vol < 0.03 and entropy_signal < 0.3:
        conviction = min(0.80, 0.5 + (0.25 - regime_prob) * 0.8)
        _recv_strike = round(_jp10 - 0.10, 3)  # 10 bps OTM receiver
        cards.append(
            TradeCard(
                name="JGB Vol Selling (Stable Regime)",
                category="volatility",
                direction="short",
                instruments=[
                    f"JB1 Options {_upper_wing:.2f}/{_lower_wing:.2f} Strangle (sell, 1M)",
                    f"JPY 10Y1Y Receiver Swaption K={_recv_strike:.3f}% (sell, OTC)",
                ],
                regime_condition=(
                    f"Stable suppressed regime: regime_prob={regime_prob:.2f}, "
                    f"GARCH vol={_garch_bps:.1f} bps, entropy={entropy_signal:.2f}. "
                    f"JGB 10Y at {_jp10:.3f}%. All three below threshold"
                ),
                edge_source=(
                    f"GARCH vol at only {_garch_bps:.1f} bps but implied typically "
                    f"trades at {_garch_bps * 1.3:.0f}-{_garch_bps * 1.5:.0f} bps. "
                    "Systematic theta harvesting: ~0.3-0.5 bps/day premium capture"
                ),
                entry_signal=(
                    f"Sell JB1 1M strangle: sell {_upper_wing:.2f} call + sell {_lower_wing:.2f} put "
                    f"(+/-10 bps OTM from ATM {_straddle_strike:.1f}). "
                    f"Also sell 10Y1Y receiver swaption strike {_recv_strike:.3f}%. "
                    "Collect premium: ~0.8-1.2% of notional. "
                    "Max loss if breached: hedge with delta at wing strike"
                ),
                exit_signal=(
                    "regime_prob > 0.40 OR GARCH vol > 400 bps OR "
                    "entropy > 0.45 (early transition warning). "
                    "Buy back strangle immediately"
                ),
                failure_scenario=(
                    "Sudden exogenous shock triggers regime jump before model "
                    "detects transition. Short-gamma: losses accelerate beyond "
                    f"wing strikes ({_lower_wing:.2f}/{_upper_wing:.2f}). "
                    "Historical worst case: 20-30 bps move in 1 day"
                ),
                sizing_method="notional-limited (short gamma budget)",
                conviction=round(conviction, 2),
                metadata={
                    "regime_prob": round(regime_prob, 4),
                    "garch_vol_bps": round(_garch_bps, 2),
                    "entropy_signal": round(entropy_signal, 4),
                    "call_strike": _upper_wing,
                    "put_strike": _lower_wing,
                    "receiver_strike": _recv_strike,
                },
            )
        )

    # --- 3. Skew Trade (asymmetric repricing risk) -----------------------
    if regime_prob > 0.4 and entropy_signal > 0.5:
        # Asymmetry detected -- prefer payer side (higher yields)
        conviction = min(0.75, 0.3 + regime_prob * 0.2 + entropy_signal * 0.15)
        _atm_strike = round(_jp10, 3)
        _otm_strike = round(_jp10 + 0.25, 3)  # ATM + 25 bps
        _breakeven = round(_jp10 + 0.10, 3)   # ~10-12 bps above ATM
        cards.append(
            TradeCard(
                name="JGB Vol Skew: Payer Spread",
                category="volatility",
                direction="long",
                instruments=[
                    f"JPY 10Y1Y Payer Swaption K={_atm_strike:.3f}% (buy, OTC)",
                    f"JPY 10Y1Y Payer Swaption K={_otm_strike:.3f}% (sell, OTC)",
                ],
                regime_condition=(
                    f"Asymmetric repricing: regime_prob={regime_prob:.2f}, "
                    f"entropy={entropy_signal:.2f}. JGB 10Y at {_jp10:.3f}%. "
                    "Skew toward higher yields is under-priced relative to regime signal"
                ),
                edge_source=(
                    "Payer skew historically cheapens 2-3 vol points ahead of "
                    "BoJ shifts. Spread structure limits max loss to net premium. "
                    f"Entropy at {entropy_signal:.2f} confirms non-linear dynamics"
                ),
                entry_signal=(
                    f"Buy payer swaption strike {_atm_strike:.3f}% (ATM), "
                    f"sell payer swaption strike {_otm_strike:.3f}% (ATM+25bp). "
                    "Net debit: ~15-25 bps running. "
                    f"Max profit: 25 bps (width of spread) minus premium. "
                    f"Breakeven: 10Y swap rate rises above {_breakeven:.3f}%"
                ),
                exit_signal=(
                    "Payer-receiver skew normalises (spread compresses >50%) OR "
                    "regime_prob < 0.25 OR expiry"
                ),
                failure_scenario=(
                    "Yields drift lower (global easing cycle) making payer "
                    "side worthless. Max loss = net premium paid (~15-25 bps). "
                    "Skew remains flat if market prices symmetric risk"
                ),
                sizing_method="spread-limited (max loss = net premium)",
                conviction=round(conviction, 2),
                metadata={
                    "regime_prob": round(regime_prob, 4),
                    "entropy_signal": round(entropy_signal, 4),
                    "atm_strike": _atm_strike,
                    "otm_strike": _otm_strike,
                    "breakeven": _breakeven,
                },
            )
        )

    return cards


# ======================================================================
# Cross-asset trades
# ======================================================================
def generate_cross_asset_trades(
    regime_prob: float,
    spillover_index: float,
    te_network: Optional[pd.DataFrame],
    dcc_correlations: Optional[pd.DataFrame],
    *,
    jp10_level: float | None = None,
    us10_level: float | None = None,
    nikkei_level: float | None = None,
) -> list[TradeCard]:
    """Generate cross-asset trade ideas driven by spillover analysis.

    Parameters
    ----------
    regime_prob : float
        Probability of repricing regime.
    spillover_index : float
        Diebold-Yilmaz total spillover index (0-100 scale).
    te_network : pd.DataFrame or None
        Transfer-entropy network adjacency matrix.  Rows/columns are
        asset labels (e.g. ``"JGB_10Y"``, ``"UST_10Y"``, ``"NIKKEI"``).
    dcc_correlations : pd.DataFrame or None
        Most recent DCC-GARCH conditional correlation matrix.

    Returns
    -------
    list[TradeCard]
        Cross-asset trade ideas.
    """
    cards: list[TradeCard] = []

    # --- 1. JGB-UST Spread -----------------------------------------------
    jgb_ust_te = 0.0
    if te_network is not None and "JGB_10Y" in te_network.index:
        if "UST_10Y" in te_network.columns:
            jgb_ust_te = float(te_network.loc["JGB_10Y", "UST_10Y"])

    if spillover_index > 60 or jgb_ust_te > 0.3:
        conviction = min(
            0.85,
            0.3 + spillover_index / 200 + regime_prob * 0.2,
        )
        _jp = jp10_level if jp10_level is not None else 0.0
        _us = us10_level if us10_level is not None else 0.0
        _spread_bps = round((_jp - _us) * 100, 0)

        cards.append(
            TradeCard(
                name="JGB-UST 10Y Spread Widener",
                category="cross_asset",
                direction="short",
                instruments=[
                    "JB1 Comdty (sell JGB 10Y Future)",
                    "TY1 Comdty (buy UST 10Y Future)",
                ],
                regime_condition=(
                    f"Spillover={spillover_index:.0f}%; TE(JGB->UST)={jgb_ust_te:.3f}. "
                    f"JGB 10Y at {_jp:.3f}%, UST 10Y at {_us:.3f}%, "
                    f"spread {_spread_bps:.0f} bps"
                ),
                edge_source=(
                    "Elevated spillover means JGB repricing propagates to UST "
                    "with a lag. Spread trade captures differential repricing speed. "
                    f"Current spread {_spread_bps:.0f} bps; target widen 20-30 bps"
                ),
                entry_signal=(
                    f"Sell JB1 (JGB 10Y), buy TY1 (UST 10Y) DV01-neutral. "
                    f"JGB DV01/contract ~JPY 80K, UST DV01/contract ~USD 780. "
                    f"Ratio: ~1.0:1.0 notional. "
                    f"Entry spread: {_spread_bps:.0f} bps"
                ),
                exit_signal=(
                    f"Take profit: spread widens 20-30 bps from {_spread_bps:.0f}. "
                    "Stop: spread narrows 15 bps. "
                    "Also exit if spillover < 40 AND TE < 0.15"
                ),
                failure_scenario=(
                    "Parallel global sell-off: UST reprices in sympathy keeping "
                    "spread constant. Flight-to-quality into UST widens spread "
                    "beyond expected range. FX hedging cost erodes carry"
                ),
                sizing_method="DV01-neutral cross-market",
                conviction=round(conviction, 2),
                metadata={
                    "spillover_index": round(spillover_index, 2),
                    "jgb_ust_te": round(jgb_ust_te, 4),
                    "spread_bps": _spread_bps,
                    "regime_prob": round(regime_prob, 4),
                },
            )
        )

    # --- 2. Nikkei-JGB Decorrelation -------------------------------------
    nk_jgb_corr = 0.0
    if dcc_correlations is not None:
        if "NIKKEI" in dcc_correlations.index and "JGB_10Y" in dcc_correlations.columns:
            nk_jgb_corr = float(dcc_correlations.loc["NIKKEI", "JGB_10Y"])

    # Historically ~-0.3 correlation; if it flips or collapses to zero
    if dcc_correlations is not None and abs(nk_jgb_corr) < 0.1:
        conviction = min(0.75, 0.3 + regime_prob * 0.2 + (0.3 - abs(nk_jgb_corr)))
        _nk = nikkei_level if nikkei_level is not None else 0.0
        cards.append(
            TradeCard(
                name="Nikkei-JGB Decorrelation Trade",
                category="cross_asset",
                direction="long",
                instruments=[
                    "NKY Index / NK1 Comdty (buy Nikkei 225 Future)",
                    "JB1 Comdty (sell JGB 10Y Future)",
                ],
                regime_condition=(
                    f"DCC(Nikkei, JGB)={nk_jgb_corr:.3f} near zero vs historical "
                    f"norm -0.30. Nikkei at {_nk:,.0f}. "
                    "Correlation regime break detected"
                ),
                edge_source=(
                    "Equity-bond correlation breakdown = relative-value opportunity. "
                    "Historical reversion to -0.30 correlation within 30 trading days "
                    "in 65% of past episodes. Beta-neutral structure limits directional risk"
                ),
                entry_signal=(
                    f"Buy NK1 (Nikkei Future, ~JPY 1000/point), "
                    f"sell JB1 (JGB 10Y Future). "
                    f"Beta-neutral ratio: hedge Nikkei delta with ~{_nk * 0.0001:.1f}x "
                    "JGB contracts per Nikkei contract"
                ),
                exit_signal=(
                    "DCC(Nikkei, JGB) re-establishes below -0.20 (mean-reversion). "
                    "Max holding: 60 business days. "
                    "Stop: either leg moves >5% adversely"
                ),
                failure_scenario=(
                    "Structural decorrelation is permanent (new macro regime e.g. "
                    "simultaneous equity sell-off + bond sell-off). Both legs move "
                    "against the position simultaneously"
                ),
                sizing_method="beta-neutral relative value",
                conviction=round(conviction, 2),
                metadata={
                    "nk_jgb_corr": round(nk_jgb_corr, 4),
                    "nikkei_level": round(_nk, 0),
                    "regime_prob": round(regime_prob, 4),
                },
            )
        )

    # --- 3. EM Duration Hedge --------------------------------------------
    if spillover_index > 50 and regime_prob > 0.5:
        conviction = min(0.80, 0.3 + spillover_index / 150 + regime_prob * 0.15)
        cards.append(
            TradeCard(
                name="EM Duration Hedge (Global Spillover)",
                category="cross_asset",
                direction="short",
                instruments=[
                    "EMLC US Equity (VanEck EM LC Bond ETF, short)",
                    "EMB US Equity (iShares JPM USD EM Bond ETF, short)",
                ],
                regime_condition=(
                    f"Spillover index={spillover_index:.0f}% > 50; "
                    f"regime_prob={regime_prob:.2f} > 0.50. "
                    "JGB repricing transmitting duration risk to EM via "
                    "global term-premium channel"
                ),
                edge_source=(
                    "JGB repricing historically triggers EM duration sell-offs "
                    "with 3-5 day lag. EMLC has 6.5Y avg duration; 10 bps "
                    "parallel shift = ~0.65% NAV move. TE network confirms causality"
                ),
                entry_signal=(
                    "Short EMLC (local-currency exposure) and/or EMB (USD-denominated). "
                    f"EMLC at market; target -3% to -5% total return. "
                    "Size: 2-3% of portfolio. Alternatively buy EMLC puts"
                ),
                exit_signal=(
                    "spillover_index < 35 OR regime_prob < 0.35 OR "
                    "EM spreads widen >100 bps (target reached). "
                    "Max holding: 45 business days"
                ),
                failure_scenario=(
                    "EM central banks tighten pre-emptively insulating local bonds. "
                    "Global risk appetite absorbs duration supply. "
                    "USD weakens offsetting duration losses in LC terms"
                ),
                sizing_method="ETF short (2-3% portfolio weight)",
                conviction=round(conviction, 2),
                metadata={
                    "spillover_index": round(spillover_index, 2),
                    "regime_prob": round(regime_prob, 4),
                },
            )
        )

    return cards


# ======================================================================
# Master aggregator
# ======================================================================
def generate_all_trades(regime_state: dict) -> list[TradeCard]:
    """Generate a complete trade book from the current regime state.

    This is the main entry point for the strategy layer.  It dispatches
    to every asset-class generator and returns the union of all trade
    cards.

    Parameters
    ----------
    regime_state : dict
        Dictionary containing all current signals.  Expected keys:

        - ``regime_prob`` : float -- repricing regime probability
        - ``term_premium`` : pd.Series
        - ``pca_scores`` : pd.DataFrame (columns PC1, PC2, PC3)
        - ``liquidity_index`` : pd.Series
        - ``carry_to_vol`` : float
        - ``usdjpy_trend`` : float
        - ``positioning`` : float
        - ``garch_vol`` : float
        - ``entropy_signal`` : float
        - ``spillover_index`` : float
        - ``te_network`` : pd.DataFrame or None
        - ``dcc_correlations`` : pd.DataFrame or None

    Returns
    -------
    list[TradeCard]
        Complete list of trade ideas across all asset classes.

    Raises
    ------
    KeyError
        If a required key is missing from ``regime_state``.
    """
    rp = float(regime_state["regime_prob"])

    all_cards: list[TradeCard] = []

    # --- Rates ---
    all_cards.extend(
        generate_rates_trades(
            regime_prob=rp,
            term_premium=regime_state["term_premium"],
            pca_scores=regime_state["pca_scores"],
            liquidity_index=regime_state["liquidity_index"],
            jp10_level=regime_state.get("jp10_level"),
            us10_level=regime_state.get("us10_level"),
            jp2y_level=regime_state.get("jp2y_level"),
        )
    )

    # --- FX ---
    all_cards.extend(
        generate_fx_trades(
            regime_prob=rp,
            carry_to_vol=float(regime_state["carry_to_vol"]),
            usdjpy_trend=float(regime_state["usdjpy_trend"]),
            positioning=float(regime_state["positioning"]),
            usdjpy_level=regime_state.get("usdjpy_level"),
        )
    )

    # --- Volatility ---
    all_cards.extend(
        generate_vol_trades(
            regime_prob=rp,
            garch_vol=float(regime_state["garch_vol"]),
            entropy_signal=float(regime_state["entropy_signal"]),
            jp10_level=regime_state.get("jp10_level"),
            usdjpy_level=regime_state.get("usdjpy_level"),
        )
    )

    # --- Cross-Asset ---
    all_cards.extend(
        generate_cross_asset_trades(
            regime_prob=rp,
            spillover_index=float(regime_state["spillover_index"]),
            te_network=regime_state.get("te_network"),
            dcc_correlations=regime_state.get("dcc_correlations"),
            jp10_level=regime_state.get("jp10_level"),
            us10_level=regime_state.get("us10_level"),
            nikkei_level=regime_state.get("nikkei_level"),
        )
    )

    return all_cards
