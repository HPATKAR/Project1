"""Tests for cross-asset spillover and FX modules."""

import pytest
import pandas as pd
import numpy as np


def _make_multivariate_data(n=500, k=4):
    """Generate correlated multivariate returns."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=n)

    # Create correlated returns
    cov = np.eye(k) * 0.01
    for i in range(k):
        for j in range(k):
            if i != j:
                cov[i, j] = 0.003

    returns = np.random.multivariate_normal(np.zeros(k), cov, n)
    columns = ["JGB_10Y", "UST_10Y", "USDJPY", "VIX"][:k]
    return pd.DataFrame(returns, index=dates, columns=columns)


class TestGranger:
    """Test Granger causality."""

    def test_pairwise_granger_output(self):
        from src.spillover.granger import pairwise_granger

        data = _make_multivariate_data()
        result = pairwise_granger(data, max_lag=5)
        assert isinstance(result, pd.DataFrame)
        assert "cause" in result.columns
        assert "effect" in result.columns
        assert "p_value" in result.columns

    def test_granger_no_self_pairs(self):
        from src.spillover.granger import pairwise_granger

        data = _make_multivariate_data()
        result = pairwise_granger(data, max_lag=5)
        same_pairs = result[result["cause"] == result["effect"]]
        assert len(same_pairs) == 0


class TestDieboldYilmaz:
    """Test Diebold-Yilmaz spillover index."""

    def test_spillover_index_scalar(self):
        from src.spillover.diebold_yilmaz import compute_spillover_index

        data = _make_multivariate_data()
        result = compute_spillover_index(data, var_lags=2, forecast_horizon=5)
        assert isinstance(result["total_spillover"], float)
        assert 0 <= result["total_spillover"] <= 100

    def test_spillover_matrix_shape(self):
        from src.spillover.diebold_yilmaz import compute_spillover_index

        data = _make_multivariate_data(k=3)
        result = compute_spillover_index(data, var_lags=2, forecast_horizon=5)
        assert result["spillover_matrix"].shape == (3, 3)


class TestTransferEntropy:
    """Test transfer entropy computation."""

    def test_discretize_series(self):
        from src.spillover.transfer_entropy import discretize_series

        np.random.seed(42)
        s = pd.Series(np.random.randn(200))
        d = discretize_series(s, n_bins=3)
        assert set(d.unique()).issubset({0, 1, 2})

    def test_transfer_entropy_non_negative(self):
        from src.spillover.transfer_entropy import compute_transfer_entropy

        np.random.seed(42)
        source = pd.Series(np.random.randn(500))
        target = pd.Series(np.random.randn(500))
        te = compute_transfer_entropy(source, target, lag=1)
        # TE should be non-negative (or very close to zero)
        assert te >= -0.01

    def test_pairwise_te_output(self):
        from src.spillover.transfer_entropy import pairwise_transfer_entropy

        data = _make_multivariate_data(n=300, k=3)
        result = pairwise_transfer_entropy(data, lag=1)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 6  # 3 * 2 directed pairs


class TestCarryAnalytics:
    """Test FX carry trade analytics."""

    def test_carry_to_vol_ratio(self):
        from src.fx.carry_analytics import carry_to_vol

        np.random.seed(42)
        carry = pd.Series(np.full(100, 0.03))  # 3% carry
        vol = pd.Series(np.full(100, 0.10))     # 10% vol
        ratio = carry_to_vol(carry, vol)
        np.testing.assert_allclose(ratio, 0.3, atol=0.01)


class TestPositioning:
    """Test CTA positioning proxy."""

    def test_trend_signal_bounded(self):
        from src.fx.positioning import trend_signal

        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(300) * 0.5))
        signal = trend_signal(prices)
        assert (signal.dropna() >= -1).all()
        assert (signal.dropna() <= 1).all()
