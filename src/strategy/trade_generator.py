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

        cards.append(
            TradeCard(
                name="JGB 10Y Short",
                category="rates",
                direction="short",
                instruments=["JGB 10Y Future", "JGB 10Y Cash"],
                regime_condition=(
                    f"regime_prob={regime_prob:.2f} > 0.60; "
                    "BoJ policy-normalisation regime detected"
                ),
                edge_source=(
                    "Hidden-Markov regime model identifies elevated probability "
                    "of sustained yield repricing; term premium trending higher"
                ),
                entry_signal=(
                    "regime_prob crosses above 0.60 with positive term-premium "
                    "momentum (20d change > 0)"
                ),
                exit_signal=(
                    "regime_prob falls below 0.40 OR 10Y yield mean-reverts "
                    "more than 15 bps from local high"
                ),
                failure_scenario=(
                    "BoJ re-anchors YCC band or conducts surprise fixed-rate "
                    "purchase operation; global risk-off drives safe-haven "
                    "demand into JGBs"
                ),
                sizing_method="vol_target",
                conviction=round(conviction, 2),
                metadata={
                    "regime_prob": round(regime_prob, 4),
                    "tp_20d_change": round(tp_change, 4),
                },
            )
        )

    # --- 2. 2s10s Steepener ---------------------------------------------
    tp_rising = False
    if len(term_premium) >= 10:
        tp_rising = float(term_premium.iloc[-1]) > float(term_premium.iloc[-10])

    if tp_rising and regime_prob > 0.5:
        conviction = min(0.90, 0.4 + regime_prob * 0.25 + 0.15)
        cards.append(
            TradeCard(
                name="JGB 2s10s Steepener",
                category="rates",
                direction="long",
                instruments=[
                    "JGB 2Y Future (short leg)",
                    "JGB 10Y Future (long leg)",
                ],
                regime_condition=(
                    f"regime_prob={regime_prob:.2f} > 0.50 AND term premium "
                    "rising over trailing 10 days"
                ),
                edge_source=(
                    "Policy normalisation historically steepens 2s10s as the "
                    "front end is anchored by BoJ short-rate guidance while "
                    "the long end reprices term premium"
                ),
                entry_signal=(
                    "term_premium 10d change > 0 AND regime_prob > 0.50; "
                    "enter long 10Y / short 2Y duration-neutral"
                ),
                exit_signal=(
                    "term premium reverses (10d change < 0) OR 2s10s spread "
                    "exceeds 80th percentile of 2-year range"
                ),
                failure_scenario=(
                    "BoJ hikes short rate faster than market expects, "
                    "flattening the curve from the front end; global "
                    "recession fears compress term premium"
                ),
                sizing_method="vol_target",
                conviction=round(conviction, 2),
                metadata={"tp_rising": True, "regime_prob": round(regime_prob, 4)},
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
            cards.append(
                TradeCard(
                    name="JGB Long-End Butterfly (5s-10s-20s)",
                    category="rates",
                    direction=direction,
                    instruments=[
                        "JGB 5Y Future",
                        "JGB 10Y Future",
                        "JGB 20Y Cash",
                    ],
                    regime_condition=(
                        f"PC3 curvature z-score={pc3_zscore:.2f} exceeds "
                        "1.5 standard deviations (extreme curvature)"
                    ),
                    edge_source=(
                        "PCA-based curvature factor at extreme levels; "
                        "mean-reversion tendency in PC3 provides "
                        "statistical edge for butterfly convergence"
                    ),
                    entry_signal=(
                        f"|PC3 z-score| > 1.5 (current: {pc3_zscore:.2f}); "
                        "enter butterfly weighted by DV01"
                    ),
                    exit_signal=(
                        "|PC3 z-score| < 0.5 OR holding period exceeds "
                        "40 business days"
                    ),
                    failure_scenario=(
                        "Structural shift in BoJ purchase allocation across "
                        "maturity buckets permanently alters curvature; "
                        "super-long demand/supply imbalance from lifers"
                    ),
                    sizing_method="vol_target",
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
            cards.append(
                TradeCard(
                    name="JGB Liquidity Premium Capture",
                    category="rates",
                    direction="short",
                    instruments=[
                        "JGB 10Y Cash (off-the-run)",
                        "JGB 10Y Future (on-the-run hedge)",
                    ],
                    regime_condition=(
                        "Liquidity index has deteriorated >10% below its "
                        "60-day moving average; regime transition amplifies "
                        "illiquidity premium"
                    ),
                    edge_source=(
                        "Widening bid-ask spreads and reduced market depth "
                        "create a liquidity premium that can be harvested "
                        "via on-the-run / off-the-run basis trades"
                    ),
                    entry_signal=(
                        "liquidity_index < 0.9 * liquidity_index.rolling(60).mean(); "
                        "sell off-the-run, buy on-the-run"
                    ),
                    exit_signal=(
                        "liquidity_index recovers above 60-day mean OR "
                        "spread narrows to within 1 bp of 6-month average"
                    ),
                    failure_scenario=(
                        "BoJ emergency liquidity operation compresses "
                        "spreads abruptly; forced position unwind by "
                        "leveraged accounts widens basis further"
                    ),
                    sizing_method="regime_adjusted",
                    conviction=round(conviction, 2),
                    metadata={
                        "liq_current": round(liq_current, 4),
                        "liq_60d_mean": round(liq_mean, 4),
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
        cards.append(
            TradeCard(
                name="Short JPY via USD/JPY",
                category="fx",
                direction="long",  # long USD/JPY = short JPY
                instruments=["USD/JPY Spot", "USD/JPY 3M Forward"],
                regime_condition=(
                    f"regime_prob={regime_prob:.2f} > 0.50 AND carry-to-vol "
                    f"ratio={carry_to_vol:.2f} > 1.0"
                ),
                edge_source=(
                    "Positive carry with favourable risk-reward; BoJ regime "
                    "shift widens rate differential further, reinforcing JPY "
                    "weakness as capital outflows accelerate"
                ),
                entry_signal=(
                    "carry_to_vol > 1.0 AND regime_prob > 0.50 AND "
                    "USD/JPY trending higher (20d momentum positive)"
                ),
                exit_signal=(
                    "carry_to_vol falls below 0.7 OR regime_prob < 0.35 OR "
                    "USD/JPY drops 3% from entry"
                ),
                failure_scenario=(
                    "Sharp global risk-off triggers JPY safe-haven rally; "
                    "Fed cuts aggressively compressing US-JP rate differential; "
                    "MoF FX intervention on extreme JPY weakness"
                ),
                sizing_method="vol_target",
                conviction=round(conviction, 2),
                metadata={
                    "carry_to_vol": round(carry_to_vol, 4),
                    "usdjpy_trend": round(usdjpy_trend, 4),
                    "positioning": round(positioning, 4),
                },
            )
        )

    # --- 2. Long JPY Volatility ------------------------------------------
    regime_uncertainty = regime_prob * (1 - regime_prob) * 4  # peaks at 0.5
    if regime_uncertainty > 0.8:
        conviction = min(0.85, 0.3 + regime_uncertainty * 0.4)
        cards.append(
            TradeCard(
                name="Long JPY Implied Volatility",
                category="fx",
                direction="long",
                instruments=[
                    "USD/JPY 1M ATM Straddle",
                    "USD/JPY 3M 25-delta Risk Reversal",
                ],
                regime_condition=(
                    f"Regime uncertainty={regime_uncertainty:.2f} > 0.80 "
                    "(regime_prob near 0.5 implies maximum transition risk)"
                ),
                edge_source=(
                    "Regime transition periods produce realised volatility "
                    "that exceeds implied; binary BoJ policy risk is "
                    "under-priced in options markets"
                ),
                entry_signal=(
                    "regime_uncertainty > 0.80 (regime_prob near 0.50); "
                    "buy ATM straddles and JPY-put risk reversals"
                ),
                exit_signal=(
                    "regime_prob moves decisively above 0.75 or below 0.25 "
                    "(uncertainty resolved) OR vol term-structure inverts"
                ),
                failure_scenario=(
                    "Regime uncertainty persists but realised vol stays "
                    "suppressed (BoJ manages orderly transition); "
                    "theta decay erodes position before move materialises"
                ),
                sizing_method="vol_target",
                conviction=round(conviction, 2),
                metadata={
                    "regime_uncertainty": round(regime_uncertainty, 4),
                    "regime_prob": round(regime_prob, 4),
                },
            )
        )

    # --- 3. Carry Unwind Hedge -------------------------------------------
    if carry_to_vol < 0.8 and positioning < -0.3:
        conviction = min(0.85, 0.4 + abs(positioning) * 0.3)
        cards.append(
            TradeCard(
                name="JPY Carry Unwind Hedge",
                category="fx",
                direction="short",  # short USD/JPY = long JPY
                instruments=[
                    "USD/JPY Spot",
                    "USD/JPY 1M 25-delta Put",
                ],
                regime_condition=(
                    f"carry_to_vol={carry_to_vol:.2f} < 0.80 (carry "
                    f"deteriorating) AND positioning={positioning:.2f} < -0.30 "
                    "(crowded short-JPY)"
                ),
                edge_source=(
                    "Crowded JPY shorts with collapsing carry-to-vol ratio "
                    "create asymmetric unwind risk; positioning data shows "
                    "vulnerability to forced covering"
                ),
                entry_signal=(
                    "carry_to_vol < 0.80 AND net speculative positioning "
                    "< -0.30 (normalised); buy USD/JPY puts or go short spot"
                ),
                exit_signal=(
                    "positioning normalises above -0.10 OR carry_to_vol "
                    "recovers above 1.2"
                ),
                failure_scenario=(
                    "Rate differential widens further despite low carry-to-vol "
                    "(e.g. US data surprise pushes UST yields higher); "
                    "JPY shorts remain intact due to structural outflows"
                ),
                sizing_method="regime_adjusted",
                conviction=round(conviction, 2),
                metadata={
                    "carry_to_vol": round(carry_to_vol, 4),
                    "positioning": round(positioning, 4),
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
    if regime_uncertainty > 0.7 or (regime_prob > 0.5 and entropy_signal > 0.6):
        conviction = min(
            0.90, 0.3 + regime_uncertainty * 0.25 + entropy_signal * 0.2
        )
        cards.append(
            TradeCard(
                name="Long JGB Volatility",
                category="volatility",
                direction="long",
                instruments=[
                    "JGB 10Y Future Options (ATM Straddle)",
                    "JGB Swaption 10Y1Y",
                ],
                regime_condition=(
                    f"Regime transitioning: uncertainty={regime_uncertainty:.2f} "
                    f"> 0.70 OR (regime_prob={regime_prob:.2f} > 0.50 AND "
                    f"entropy={entropy_signal:.2f} > 0.60)"
                ),
                edge_source=(
                    "Regime transitions historically produce realised vol "
                    "1.5-2x above pre-transition implied levels; entropy "
                    "rising confirms increasing yield-curve disorder"
                ),
                entry_signal=(
                    "regime_uncertainty > 0.70 OR (regime_prob > 0.50 AND "
                    "entropy_signal > 0.60); buy straddles / payer swaptions"
                ),
                exit_signal=(
                    "regime_uncertainty < 0.30 (regime resolved) AND "
                    "entropy_signal < 0.40"
                ),
                failure_scenario=(
                    "BoJ communication effectively pre-commits policy path, "
                    "reducing uncertainty without generating vol; implied "
                    "vol already elevated and mean-reverts"
                ),
                sizing_method="vol_target",
                conviction=round(conviction, 2),
                metadata={
                    "regime_uncertainty": round(regime_uncertainty, 4),
                    "garch_vol": round(garch_vol, 6),
                    "entropy_signal": round(entropy_signal, 4),
                },
            )
        )

    # --- 2. Vol Selling in Stable Regime ---------------------------------
    if regime_prob < 0.25 and garch_vol < 0.03 and entropy_signal < 0.3:
        conviction = min(0.80, 0.5 + (0.25 - regime_prob) * 0.8)
        cards.append(
            TradeCard(
                name="JGB Vol Selling (Stable Regime)",
                category="volatility",
                direction="short",
                instruments=[
                    "JGB 10Y Future Options (OTM Strangle Sell)",
                    "JGB Swaption 10Y1Y (Receiver)",
                ],
                regime_condition=(
                    f"Stable suppressed regime confirmed: regime_prob="
                    f"{regime_prob:.2f} < 0.25, GARCH vol={garch_vol:.4f} "
                    f"< 3%, entropy={entropy_signal:.2f} < 0.30"
                ),
                edge_source=(
                    "In confirmed suppressed-volatility regime, implied vol "
                    "tends to overstate realised; systematic theta harvesting "
                    "generates positive carry"
                ),
                entry_signal=(
                    "regime_prob < 0.25 AND garch_vol < 0.03 AND "
                    "entropy_signal < 0.30; sell OTM strangles"
                ),
                exit_signal=(
                    "regime_prob rises above 0.40 OR garch_vol > 0.04 OR "
                    "entropy_signal > 0.45 (early transition warning)"
                ),
                failure_scenario=(
                    "Sudden exogenous shock (geopolitical event, global "
                    "rate spike) triggers regime jump before model detects "
                    "transition; short-gamma exposure amplifies losses"
                ),
                sizing_method="regime_adjusted",
                conviction=round(conviction, 2),
                metadata={
                    "regime_prob": round(regime_prob, 4),
                    "garch_vol": round(garch_vol, 6),
                    "entropy_signal": round(entropy_signal, 4),
                },
            )
        )

    # --- 3. Skew Trade (asymmetric repricing risk) -----------------------
    if regime_prob > 0.4 and entropy_signal > 0.5:
        # Asymmetry detected -- prefer payer side (higher yields)
        conviction = min(0.75, 0.3 + regime_prob * 0.2 + entropy_signal * 0.15)
        cards.append(
            TradeCard(
                name="JGB Vol Skew -- Payer Spread",
                category="volatility",
                direction="long",
                instruments=[
                    "JGB Swaption 10Y1Y Payer (ATM)",
                    "JGB Swaption 10Y1Y Payer (OTM, +25bp strike)",
                ],
                regime_condition=(
                    f"Asymmetric repricing detected: regime_prob="
                    f"{regime_prob:.2f} > 0.40 AND entropy={entropy_signal:.2f} "
                    "> 0.50 implies skew towards higher yields"
                ),
                edge_source=(
                    "Payer skew is historically cheap ahead of BoJ policy "
                    "shifts; entropy signal captures non-linear yield curve "
                    "dynamics before they appear in implied vol surface"
                ),
                entry_signal=(
                    "regime_prob > 0.40 AND entropy > 0.50; buy ATM payer "
                    "swaption, sell OTM payer swaption 25bp higher as spread"
                ),
                exit_signal=(
                    "Skew normalises (payer-receiver spread compresses) OR "
                    "regime_prob falls below 0.25"
                ),
                failure_scenario=(
                    "Yields drift lower (global easing cycle) making payer "
                    "side worthless; skew remains flat as market prices "
                    "symmetric risk"
                ),
                sizing_method="vol_target",
                conviction=round(conviction, 2),
                metadata={
                    "regime_prob": round(regime_prob, 4),
                    "entropy_signal": round(entropy_signal, 4),
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
        cards.append(
            TradeCard(
                name="JGB-UST 10Y Spread Widener",
                category="cross_asset",
                direction="short",
                instruments=[
                    "JGB 10Y Future (short)",
                    "UST 10Y Future (long)",
                ],
                regime_condition=(
                    f"Spillover index={spillover_index:.1f} > 60 OR "
                    f"JGB->UST transfer entropy={jgb_ust_te:.3f} > 0.30; "
                    "cross-border rate transmission intensifying"
                ),
                edge_source=(
                    "Elevated spillover transmission means JGB repricing "
                    "will propagate to UST; spread trade captures the "
                    "differential in repricing speed"
                ),
                entry_signal=(
                    "spillover_index > 60 OR te(JGB->UST) > 0.30; "
                    "short JGB 10Y, long UST 10Y duration-neutral"
                ),
                exit_signal=(
                    "spillover_index falls below 40 AND te(JGB->UST) < 0.15 "
                    "OR spread reaches 2-sigma wide vs. 1-year history"
                ),
                failure_scenario=(
                    "UST reprices in sympathy (parallel global sell-off) "
                    "so spread stays constant; flight-to-quality into UST "
                    "widens spread beyond expected range"
                ),
                sizing_method="vol_target",
                conviction=round(conviction, 2),
                metadata={
                    "spillover_index": round(spillover_index, 2),
                    "jgb_ust_te": round(jgb_ust_te, 4),
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
        cards.append(
            TradeCard(
                name="Nikkei-JGB Decorrelation Trade",
                category="cross_asset",
                direction="long",
                instruments=[
                    "Nikkei 225 Future (long)",
                    "JGB 10Y Future (short)",
                ],
                regime_condition=(
                    f"DCC Nikkei-JGB correlation={nk_jgb_corr:.3f} near zero "
                    "(historical norm ~ -0.30); correlation regime break"
                ),
                edge_source=(
                    "When equity-bond correlation breaks from its negative "
                    "norm, relative-value mean-reversion tendency creates "
                    "an exploitable dislocation"
                ),
                entry_signal=(
                    "|DCC(Nikkei, JGB)| < 0.10; long Nikkei vs. short JGB "
                    "in a beta-neutral ratio"
                ),
                exit_signal=(
                    "DCC(Nikkei, JGB) re-establishes below -0.20 OR "
                    "holding period exceeds 60 business days"
                ),
                failure_scenario=(
                    "Structural decorrelation is permanent (new macro regime "
                    "invalidates historical relationship); both legs move "
                    "against the position simultaneously"
                ),
                sizing_method="regime_adjusted",
                conviction=round(conviction, 2),
                metadata={
                    "nk_jgb_corr": round(nk_jgb_corr, 4),
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
                    "EM Local-Currency Bond ETF (short)",
                    "EMB USD-Denominated Bond ETF (short)",
                ],
                regime_condition=(
                    f"Global spillover index={spillover_index:.1f} > 50 AND "
                    f"regime_prob={regime_prob:.2f} > 0.50; JGB repricing "
                    "transmitting duration risk to EM"
                ),
                edge_source=(
                    "JGB repricing events historically trigger EM duration "
                    "sell-offs through the global term-premium channel; "
                    "transfer-entropy network confirms directional causality"
                ),
                entry_signal=(
                    "spillover_index > 50 AND regime_prob > 0.50; "
                    "short EM duration via ETFs or CDS"
                ),
                exit_signal=(
                    "spillover_index < 35 OR regime_prob < 0.35 OR "
                    "EM spreads widen > 100 bps (target reached)"
                ),
                failure_scenario=(
                    "EM central banks tighten pre-emptively insulating local "
                    "bonds; global risk appetite remains strong enough to "
                    "absorb duration supply; USD weakens offsetting "
                    "duration losses"
                ),
                sizing_method="vol_target",
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
        )
    )

    # --- FX ---
    all_cards.extend(
        generate_fx_trades(
            regime_prob=rp,
            carry_to_vol=float(regime_state["carry_to_vol"]),
            usdjpy_trend=float(regime_state["usdjpy_trend"]),
            positioning=float(regime_state["positioning"]),
        )
    )

    # --- Volatility ---
    all_cards.extend(
        generate_vol_trades(
            regime_prob=rp,
            garch_vol=float(regime_state["garch_vol"]),
            entropy_signal=float(regime_state["entropy_signal"]),
        )
    )

    # --- Cross-Asset ---
    all_cards.extend(
        generate_cross_asset_trades(
            regime_prob=rp,
            spillover_index=float(regime_state["spillover_index"]),
            te_network=regime_state.get("te_network"),
            dcc_correlations=regime_state.get("dcc_correlations"),
        )
    )

    return all_cards
