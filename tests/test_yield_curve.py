"""Tests for yield curve analytics."""

import pytest
import pandas as pd
import numpy as np


def _make_yield_data(n_dates=500, tenors=None):
    """Generate simulated JGB yield data for testing."""
    tenors = tenors or [2, 5, 7, 10, 20, 30]
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)

    # Simulated yields with realistic curve shape
    base_yields = {2: 0.0, 5: 0.05, 7: 0.1, 10: 0.15, 20: 0.5, 30: 0.8}
    data = {}
    for t in tenors:
        level = base_yields.get(t, 0.1)
        data[f"{t}Y"] = level + np.cumsum(np.random.randn(n_dates) * 0.005)

    return pd.DataFrame(data, index=dates)


class TestPCA:
    """Test PCA decomposition."""

    def test_fit_pca_shapes(self):
        from src.yield_curve.pca import fit_yield_pca

        yields = _make_yield_data()
        changes = yields.diff().dropna()
        result = fit_yield_pca(changes, n_components=3)

        assert result["scores"].shape == (len(changes), 3)
        assert result["loadings"].shape == (3, changes.shape[1])
        assert len(result["explained_variance"]) == 3

    def test_pca_explained_variance_sums_to_at_most_1(self):
        from src.yield_curve.pca import fit_yield_pca

        yields = _make_yield_data()
        changes = yields.diff().dropna()
        result = fit_yield_pca(changes, n_components=3)
        assert sum(result["explained_variance"]) <= 1.01

    def test_pca_first_component_dominates(self):
        from src.yield_curve.pca import fit_yield_pca

        yields = _make_yield_data()
        changes = yields.diff().dropna()
        result = fit_yield_pca(changes, n_components=3)
        # PC1 should explain more than PC2
        assert result["explained_variance"][0] > result["explained_variance"][1]


class TestNelsonSiegel:
    """Test Nelson-Siegel fitting."""

    def test_fit_ns_returns_params(self):
        from src.yield_curve.nelson_siegel import fit_ns

        tenors = [2, 5, 7, 10, 20, 30]
        yields = pd.Series([0.0, 0.05, 0.1, 0.15, 0.5, 0.8], index=tenors)
        result = fit_ns(yields, tenors)
        assert "beta0" in result
        assert "beta1" in result
        assert "beta2" in result


class TestLiquidity:
    """Test liquidity metrics."""

    def test_amihud_output_length(self):
        from src.yield_curve.liquidity import amihud_illiquidity

        np.random.seed(42)
        returns = pd.Series(np.random.randn(250) * 0.01)
        volume = pd.Series(np.random.uniform(1e6, 1e7, 250))
        result = amihud_illiquidity(returns, volume, window=22)
        assert len(result) == len(returns)

    def test_roll_measure_non_negative(self):
        from src.yield_curve.liquidity import roll_measure

        np.random.seed(42)
        returns = pd.Series(np.random.randn(250) * 0.01)
        result = roll_measure(returns, window=22)
        # Roll measure should be non-negative (by construction)
        assert (result.dropna() >= 0).all()

    def test_composite_liquidity_index(self):
        from src.yield_curve.liquidity import composite_liquidity_index

        np.random.seed(42)
        n = 250
        metrics = {
            "amihud": pd.Series(np.random.randn(n)),
            "roll": pd.Series(np.random.randn(n)),
        }
        result = composite_liquidity_index(metrics)
        assert len(result) == n
