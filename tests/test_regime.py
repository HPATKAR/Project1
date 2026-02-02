"""Tests for regime detection engine."""

import pytest
import pandas as pd
import numpy as np


def _make_regime_data(n=500):
    """Generate data with a clear regime shift at the midpoint."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=n)

    # Regime 1: low vol, near zero mean (suppressed)
    r1 = np.random.normal(0.0, 0.02, n // 2)
    # Regime 2: higher vol, positive mean (market-driven)
    r2 = np.random.normal(0.05, 0.08, n - n // 2)

    series = pd.Series(np.concatenate([r1, r2]), index=dates, name="jgb_10y_chg")
    return series


class TestMarkovSwitching:
    """Test Markov regime-switching model."""

    def test_fit_returns_probabilities(self):
        from src.regime.markov_switching import fit_markov_regime

        data = _make_regime_data()
        result = fit_markov_regime(data, k_regimes=2)
        assert "regime_probabilities" in result
        assert result["regime_probabilities"].shape[1] == 2

    def test_regime_probs_sum_to_one(self):
        from src.regime.markov_switching import fit_markov_regime

        data = _make_regime_data()
        result = fit_markov_regime(data, k_regimes=2)
        row_sums = result["regime_probabilities"].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.01)

    def test_classify_regime(self):
        from src.regime.markov_switching import fit_markov_regime, classify_current_regime

        data = _make_regime_data()
        result = fit_markov_regime(data, k_regimes=2)
        regime = classify_current_regime(result["regime_probabilities"])
        assert regime in [0, 1]


class TestStructuralBreaks:
    """Test structural break detection."""

    def test_pelt_detects_break(self):
        from src.regime.structural_breaks import detect_breaks_pelt

        data = _make_regime_data()
        breaks = detect_breaks_pelt(data)
        # Should detect at least one break near the midpoint
        assert len(breaks) >= 1

    def test_binseg_returns_requested_breaks(self):
        from src.regime.structural_breaks import detect_breaks_binseg

        data = _make_regime_data()
        breaks = detect_breaks_binseg(data, n_bkps=3)
        assert len(breaks) == 3


class TestEntropyRegime:
    """Test entropy-based regime detection."""

    def test_rolling_perm_entropy_length(self):
        from src.regime.entropy_regime import rolling_permutation_entropy

        data = _make_regime_data()
        result = rolling_permutation_entropy(data, window=60)
        assert len(result) == len(data)

    def test_entropy_regime_signal_binary(self):
        from src.regime.entropy_regime import (
            rolling_permutation_entropy,
            entropy_regime_signal,
        )

        data = _make_regime_data()
        entropy = rolling_permutation_entropy(data, window=60)
        signal = entropy_regime_signal(entropy)
        # Signal should be 0 or 1
        unique_vals = set(signal.dropna().unique())
        assert unique_vals.issubset({0, 1, 0.0, 1.0})


class TestGARCH:
    """Test GARCH volatility regime."""

    def test_fit_garch_returns_vol(self):
        from src.regime.garch_regime import fit_garch

        data = _make_regime_data() * 100  # Scale for GARCH
        result = fit_garch(data)
        assert "conditional_volatility" in result
        assert len(result["conditional_volatility"]) == len(data)

    def test_conditional_vol_positive(self):
        from src.regime.garch_regime import fit_garch

        data = _make_regime_data() * 100
        result = fit_garch(data)
        assert (result["conditional_volatility"].dropna() > 0).all()


class TestEnsemble:
    """Test ensemble regime probability."""

    def test_ensemble_output_range(self):
        from src.regime.ensemble import ensemble_regime_probability

        n = 100
        np.random.seed(42)
        markov = pd.DataFrame(
            {"regime_0": np.random.rand(n), "regime_1": np.random.rand(n)}
        )
        markov = markov.div(markov.sum(axis=1), axis=0)
        hmm = pd.Series(np.random.choice([0, 1], n))
        entropy = pd.Series(np.random.choice([0, 1], n).astype(float))
        garch_vol = pd.Series(np.random.choice([0, 1], n).astype(float))

        result = ensemble_regime_probability(markov["regime_1"], hmm, entropy, garch_vol)
        assert (result >= 0).all() and (result <= 1).all()
