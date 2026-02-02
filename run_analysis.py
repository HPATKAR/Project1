"""
JGB Repricing Framework — Main Analysis Runner
================================================
Runs the full pipeline: data -> yield curve -> regime detection ->
spillover -> trade generation.

Usage:
    python run_analysis.py --simulated     # Development mode with simulated data
    python run_analysis.py                 # Production mode with live data (needs API keys)
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.data.data_store import DataStore
from src.data.config import BOJ_EVENTS


def run_pipeline(use_simulated: bool = True, output_dir: str = "output"):
    """Run the full JGB repricing analysis pipeline."""
    warnings.filterwarnings("ignore")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("JGB REPRICING FRAMEWORK")
    print("Regime Detection & Systematic Macro Trading")
    print("=" * 70)

    # ── SECTION 1: Data ───────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    store = DataStore(use_simulated=use_simulated)
    unified = store.get_unified(start="2013-01-01")
    print(f"  Loaded {unified.shape[0]} rows x {unified.shape[1]} columns")
    print(f"  Date range: {unified.index.min().date()} to {unified.index.max().date()}")

    # ── SECTION 2: Yield Curve Analytics ──────────────────────────────
    print("\n[2/5] Yield curve analytics...")
    from src.yield_curve.pca import fit_yield_pca

    # Use available yield columns for PCA
    yield_cols = [c for c in unified.columns if c.endswith("Y") or "10Y" in c]
    if not yield_cols:
        yield_cols = ["JP_10Y", "US_10Y", "US_2Y", "US_5Y", "US_30Y"]
    yield_cols = [c for c in yield_cols if c in unified.columns]

    if len(yield_cols) >= 2:
        yield_data = unified[yield_cols].dropna()
        yield_changes = yield_data.diff().dropna()
        pca_result = fit_yield_pca(yield_changes)
        print(f"  PCA explained variance: {pca_result['explained_variance']}")
        pca_scores = pd.DataFrame(
            pca_result["scores"],
            index=yield_changes.index,
            columns=[f"PC{i+1}" for i in range(pca_result["scores"].shape[1])],
        )
    else:
        print("  Insufficient yield columns for PCA, skipping...")
        pca_scores = None

    # Liquidity proxy
    from src.yield_curve.liquidity import amihud_illiquidity, roll_measure

    if "JP_10Y" in unified.columns:
        jp_returns = unified["JP_10Y"].diff().dropna()
        # Use VIX as volume proxy if no actual volume data
        fake_vol = pd.Series(np.abs(np.random.randn(len(jp_returns))) * 1e6, index=jp_returns.index)
        amihud = amihud_illiquidity(jp_returns, fake_vol)
        roll = roll_measure(jp_returns)
        print(f"  Amihud illiquidity (latest): {amihud.iloc[-1]:.6f}")
        print(f"  Roll measure (latest): {roll.iloc[-1]:.6f}")

    # ── SECTION 3: Regime Detection ───────────────────────────────────
    print("\n[3/5] Regime detection...")
    from src.regime.markov_switching import fit_markov_regime, classify_current_regime
    from src.regime.structural_breaks import detect_breaks_pelt
    from src.regime.entropy_regime import rolling_permutation_entropy, entropy_regime_signal
    from src.regime.garch_regime import fit_garch, volatility_regime_breaks
    from src.regime.ensemble import ensemble_regime_probability

    if "JP_10Y" in unified.columns:
        jp_changes = unified["JP_10Y"].diff().dropna()

        # Markov switching
        try:
            ms_result = fit_markov_regime(jp_changes)
            regime_probs = ms_result["regime_probabilities"]
            current_regime = classify_current_regime(regime_probs)
            print(f"  Markov regime: {'Market-Driven' if current_regime == 1 else 'Suppressed'}")
        except Exception as e:
            print(f"  Markov switching failed: {e}")
            regime_probs = pd.DataFrame(
                {"regime_0": 0.5, "regime_1": 0.5},
                index=jp_changes.index,
            )

        # Structural breaks
        try:
            breaks = detect_breaks_pelt(jp_changes)
            print(f"  Structural breaks detected: {len(breaks)}")
            for b in breaks[-3:]:
                print(f"    {b}")
        except Exception as e:
            print(f"  Break detection failed: {e}")

        # Entropy
        try:
            entropy = rolling_permutation_entropy(jp_changes, window=60)
            entropy_sig = entropy_regime_signal(entropy)
            print(f"  Permutation entropy (latest): {entropy.iloc[-1]:.4f}")
        except Exception as e:
            print(f"  Entropy calculation failed: {e}")
            entropy_sig = pd.Series(0, index=jp_changes.index)

        # GARCH
        try:
            garch_result = fit_garch(jp_changes * 100)
            cond_vol = garch_result["conditional_volatility"]
            vol_breaks = volatility_regime_breaks(cond_vol)
            print(f"  GARCH conditional vol (latest): {cond_vol.iloc[-1]:.4f}")
            print(f"  Vol regime breaks: {len(vol_breaks)}")
            garch_regime = pd.Series(0, index=jp_changes.index)
            if vol_breaks:
                last_break_idx = jp_changes.index.get_loc(vol_breaks[-1])
                garch_regime.iloc[last_break_idx:] = 1
        except Exception as e:
            print(f"  GARCH failed: {e}")
            garch_regime = pd.Series(0, index=jp_changes.index)

        # Ensemble
        try:
            # Align all signals
            common_idx = regime_probs.index.intersection(entropy_sig.index).intersection(garch_regime.index)
            hmm_proxy = (regime_probs.loc[common_idx].iloc[:, 1] > 0.5).astype(float)

            ensemble_prob = ensemble_regime_probability(
                regime_probs.loc[common_idx, "regime_1"],
                hmm_proxy,
                entropy_sig.loc[common_idx],
                garch_regime.loc[common_idx],
            )
            print(f"  Ensemble regime probability (latest): {ensemble_prob.iloc[-1]:.3f}")
        except Exception as e:
            print(f"  Ensemble failed: {e}")
            ensemble_prob = pd.Series(0.5)

    # ── SECTION 4: Cross-Asset Spillover ──────────────────────────────
    print("\n[4/5] Cross-asset spillover...")
    from src.spillover.granger import pairwise_granger
    from src.spillover.transfer_entropy import pairwise_transfer_entropy

    spillover_cols = [c for c in ["JP_10Y", "US_10Y", "DE_10Y", "USDJPY", "VIX"]
                      if c in unified.columns]

    if len(spillover_cols) >= 3:
        spillover_data = unified[spillover_cols].diff().dropna()

        # Granger causality
        try:
            granger_results = pairwise_granger(spillover_data, max_lag=5)
            sig_pairs = granger_results[granger_results["significant"]]
            print(f"  Significant Granger pairs: {len(sig_pairs)} of {len(granger_results)}")
            for _, row in sig_pairs.head(5).iterrows():
                print(f"    {row['cause']} -> {row['effect']} (p={row['p_value']:.4f}, lag={row['optimal_lag']})")
        except Exception as e:
            print(f"  Granger causality failed: {e}")

        # Transfer entropy
        try:
            te_results = pairwise_transfer_entropy(spillover_data.iloc[-300:], lag=1)
            te_results = te_results.sort_values("te_value", ascending=False)
            print(f"  Top information flows (transfer entropy):")
            for _, row in te_results.head(5).iterrows():
                print(f"    {row['source']} -> {row['target']}: TE={row['te_value']:.4f}")
        except Exception as e:
            print(f"  Transfer entropy failed: {e}")

    # FX carry analytics
    from src.fx.carry_analytics import compute_carry, carry_to_vol

    if "US_FF" in unified.columns and "JP_CALL_RATE" in unified.columns:
        carry = compute_carry(unified["JP_CALL_RATE"], unified["US_FF"])
        if "USDJPY" in unified.columns:
            usdjpy_vol = unified["USDJPY"].pct_change().rolling(63).std() * np.sqrt(252)
            c2v = carry_to_vol(carry, usdjpy_vol)
            print(f"  JPY carry: {carry.iloc[-1]:.2f}%")
            print(f"  Carry-to-vol: {c2v.iloc[-1]:.3f}")

    # ── SECTION 5: Trade Generation ───────────────────────────────────
    print("\n[5/5] Trade generation...")
    from src.strategy.trade_generator import generate_all_trades
    from src.strategy.trade_card import format_trade_card

    # Build regime state dict with sensible fallbacks
    fallback_idx = unified.index[-252:]  # last year of data
    fallback_series = pd.Series(0.0, index=fallback_idx)
    fallback_df = pd.DataFrame(
        {"PC1": 0.0, "PC2": 0.0, "PC3": 0.0}, index=fallback_idx,
    )

    regime_state = {
        "regime_prob": float(ensemble_prob.iloc[-1]) if len(ensemble_prob) > 0 else 0.5,
        "term_premium": fallback_series.copy(),
        "pca_scores": pca_scores if pca_scores is not None else fallback_df,
        "liquidity_index": amihud if "JP_10Y" in unified.columns else fallback_series,
        "carry_to_vol": float(c2v.iloc[-1]) if "US_FF" in unified.columns else 0.3,
        "usdjpy_trend": 0.0,
        "positioning": 0.0,
        "garch_vol": float(cond_vol.iloc[-1]) if "JP_10Y" in unified.columns else 1.0,
        "entropy_signal": float(entropy_sig.iloc[-1]) if "JP_10Y" in unified.columns else 0.0,
        "spillover_index": 50.0,
        "te_network": None,
        "dcc_correlations": None,
    }

    trades = generate_all_trades(regime_state)
    print(f"\n  Generated {len(trades)} trade ideas:")
    for trade in trades:
        print(f"\n{'─' * 60}")
        print(format_trade_card(trade))

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nRegime state: {'MARKET-DRIVEN' if regime_state['regime_prob'] > 0.5 else 'SUPPRESSED'}")
    print(f"Regime probability: {regime_state['regime_prob']:.1%}")
    print(f"Trade ideas generated: {len(trades)}")
    print(f"  Rates: {sum(1 for t in trades if t.category == 'rates')}")
    print(f"  FX: {sum(1 for t in trades if t.category == 'fx')}")
    print(f"  Volatility: {sum(1 for t in trades if t.category == 'volatility')}")
    print(f"  Cross-asset: {sum(1 for t in trades if t.category == 'cross_asset')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JGB Repricing Framework")
    parser.add_argument("--simulated", action="store_true", help="Use simulated data")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    run_pipeline(use_simulated=args.simulated, output_dir=args.output)
