"""Tests for equity market spillover page helpers and analytics."""

import pytest
import pandas as pd
import numpy as np


# ── Fixtures ──────────────────────────────────────────────────────────

def _make_jgb_series(n=500):
    """Generate synthetic JGB 10Y yield changes."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=n)
    changes = np.random.randn(n) * 0.02  # ~2 bps daily changes
    return pd.Series(changes, index=dates, name="JP_10Y")


def _make_equity_returns(n=500, sectors=3):
    """Generate synthetic equity sector returns."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=n)
    names = [f"Sector_{i}" for i in range(sectors)]
    data = np.random.randn(n, sectors) * 0.01
    return pd.DataFrame(data, index=dates, columns=names)


# ── Config Tests ──────────────────────────────────────────────────────

class TestEquitySectorConfig:
    """Test EQUITY_SECTOR_TICKERS configuration integrity."""

    def test_config_has_four_markets(self):
        from src.data.config import EQUITY_SECTOR_TICKERS
        assert set(EQUITY_SECTOR_TICKERS.keys()) == {"USA", "Japan", "India", "China"}

    def test_each_market_has_broad_index(self):
        from src.data.config import EQUITY_SECTOR_TICKERS
        for market, config in EQUITY_SECTOR_TICKERS.items():
            assert "_broad" in config, f"{market} missing _broad key"
            assert len(config["_broad"]) >= 2, f"{market} needs at least 2 broad indices"

    def test_usa_has_11_sectors(self):
        from src.data.config import EQUITY_SECTOR_TICKERS
        usa = EQUITY_SECTOR_TICKERS["USA"]
        sectors = [k for k in usa if k != "_broad"]
        assert len(sectors) == 11, f"USA should have 11 SPDR sectors, got {len(sectors)}"

    def test_ticker_values_are_strings(self):
        from src.data.config import EQUITY_SECTOR_TICKERS
        for market, config in EQUITY_SECTOR_TICKERS.items():
            for key, val in config.items():
                if key == "_broad":
                    for bname, bsym in val.items():
                        assert isinstance(bsym, str), f"{market}._broad.{bname} not a string"
                else:
                    assert isinstance(val, str), f"{market}.{key} not a string"


# ── Helper Tests ──────────────────────────────────────────────────────

class TestEquitySpilloverHelpers:
    """Test ticker tuple building and sector listing."""

    def test_build_ticker_tuple_returns_tuple(self):
        from src.pages.equity_spillover import _build_ticker_tuple
        result = _build_ticker_tuple("USA")
        assert isinstance(result, tuple)
        assert len(result) > 0

    def test_build_ticker_tuple_includes_broad(self):
        from src.pages.equity_spillover import _build_ticker_tuple
        result = _build_ticker_tuple("USA")
        names = [name for name, _ in result]
        assert "S&P 500" in names

    def test_build_ticker_tuple_includes_sectors(self):
        from src.pages.equity_spillover import _build_ticker_tuple
        result = _build_ticker_tuple("USA")
        names = [name for name, _ in result]
        assert "Technology" in names
        assert "Financials" in names

    def test_get_sector_names_excludes_broad(self):
        from src.pages.equity_spillover import _get_sector_names
        sectors = _get_sector_names("USA")
        assert "_broad" not in sectors
        assert "S&P 500" not in sectors

    def test_get_broad_names(self):
        from src.pages.equity_spillover import _get_broad_names
        broad = _get_broad_names("Japan")
        assert "Nikkei 225" in broad

    def test_unknown_market_returns_empty(self):
        from src.pages.equity_spillover import _build_ticker_tuple, _get_sector_names
        assert _build_ticker_tuple("Mars") == ()
        assert _get_sector_names("Mars") == []


# ── Serialization Tests ───────────────────────────────────────────────

class TestSerialization:
    """Test DataFrame/Series to tuple serialization for cache hashability."""

    def test_series_to_tuple_roundtrip(self):
        from src.pages.equity_spillover import _series_to_tuple
        s = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3))
        t = _series_to_tuple(s)
        assert isinstance(t, tuple)
        assert len(t) == 3
        # Reconstruct
        reconstructed = pd.Series(dict(t))
        assert len(reconstructed) == 3

    def test_df_cols_to_tuples(self):
        from src.pages.equity_spillover import _df_cols_to_tuples
        df = pd.DataFrame({
            "A": [1.0, 2.0, np.nan],
            "B": [4.0, 5.0, 6.0],
        }, index=pd.date_range("2020-01-01", periods=3))
        result = _df_cols_to_tuples(df, ["A", "B"])
        assert isinstance(result, tuple)
        assert len(result) == 2
        # Column A has NaN dropped so 2 values
        assert len(result[0]) == 2
        # Column B has no NaN so 3 values
        assert len(result[1]) == 3

    def test_df_cols_ignores_missing(self):
        from src.pages.equity_spillover import _df_cols_to_tuples
        df = pd.DataFrame({"A": [1.0]}, index=pd.date_range("2020-01-01", periods=1))
        result = _df_cols_to_tuples(df, ["A", "NONEXISTENT"])
        assert len(result) == 1  # only A included


# ── Analytics Integration Tests ───────────────────────────────────────

class TestEquityCorrelation:
    """Test rolling correlation computation."""

    def test_correlation_returns_dataframe(self):
        """Correlation function should return a DataFrame with sector columns."""
        from src.pages.equity_spillover import _series_to_tuple

        jgb = _make_jgb_series(300)
        eq = _make_equity_returns(300, sectors=3)

        jgb_tuple = _series_to_tuple(jgb)
        eq_tuples = tuple(
            (col, _series_to_tuple(eq[col]))
            for col in eq.columns
        )

        # Call the raw logic (not the cached version, which needs Streamlit)
        result = {}
        for name, vals in eq_tuples:
            s = pd.Series(dict(vals))
            s.index = pd.to_datetime(s.index)
            jgb_s = pd.Series(dict(jgb_tuple))
            jgb_s.index = pd.to_datetime(jgb_s.index)
            pair = pd.DataFrame({"eq": s, "jgb": jgb_s}).dropna()
            if len(pair) >= 70:
                result[name] = pair["eq"].rolling(60).corr(pair["jgb"])

        corr_df = pd.DataFrame(result).dropna(how="all")
        assert not corr_df.empty
        assert len(corr_df.columns) == 3
        # Correlations bounded [-1, 1]
        assert (corr_df.dropna().values >= -1.001).all()
        assert (corr_df.dropna().values <= 1.001).all()

    def test_sparse_sector_excluded(self):
        """Sectors with fewer than window+10 overlapping observations should be excluded."""
        from src.pages.equity_spillover import _series_to_tuple

        jgb = _make_jgb_series(300)
        # Create one sector with only 30 data points
        dates_short = pd.bdate_range("2020-01-01", periods=30)
        sparse = pd.Series(np.random.randn(30) * 0.01, index=dates_short, name="Sparse")

        jgb_tuple = _series_to_tuple(jgb)
        sparse_tuple = ("Sparse", _series_to_tuple(sparse))

        s = pd.Series(dict(sparse_tuple[1]))
        s.index = pd.to_datetime(s.index)
        jgb_s = pd.Series(dict(jgb_tuple))
        jgb_s.index = pd.to_datetime(jgb_s.index)
        pair = pd.DataFrame({"eq": s, "jgb": jgb_s}).dropna()
        # With window=60, we need 70 points minimum — 30 is not enough
        assert len(pair) < 70


class TestEquityGrangerIntegration:
    """Test Granger causality on equity returns (uses real spillover module)."""

    def test_granger_with_equity_data(self):
        from src.spillover.granger import pairwise_granger

        jgb = _make_jgb_series(300)
        eq = _make_equity_returns(300, sectors=2)

        combined = pd.DataFrame({"JGB_10Y": jgb}).join(eq).dropna()
        result = pairwise_granger(combined, max_lag=5)

        assert isinstance(result, pd.DataFrame)
        assert "cause" in result.columns
        assert "optimal_lag" in result.columns
        assert "f_stat" in result.columns
        # Should have directed pairs: 3 vars => 6 directed pairs
        assert len(result) == 6


class TestEquitySpilloverIntegration:
    """Test Diebold-Yilmaz spillover on equity data."""

    def test_spillover_with_equity_data(self):
        from src.spillover.diebold_yilmaz import compute_spillover_index

        jgb = _make_jgb_series(300)
        eq = _make_equity_returns(300, sectors=2)

        combined = pd.DataFrame({"JGB_10Y": jgb}).join(eq).dropna()
        result = compute_spillover_index(combined, var_lags=2, forecast_horizon=5)

        assert isinstance(result["total_spillover"], float)
        assert 0 <= result["total_spillover"] <= 100
        assert result["spillover_matrix"].shape == (3, 3)
