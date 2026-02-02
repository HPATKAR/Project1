"""Tests for data infrastructure."""

import pytest
import pandas as pd
import numpy as np

from src.data.data_store import DataStore
from src.data.fred_client import generate_simulated_fred
from src.data.market_data import generate_simulated_market
from src.data.config import BOJ_EVENTS, FRED_SERIES, JGB_TENORS


class TestSimulatedData:
    """Test simulated data generation."""

    def test_simulated_fred_shape(self):
        df = generate_simulated_fred("2020-01-01", "2023-12-31")
        assert len(df) > 500
        assert "JP_10Y" in df.columns
        assert "US_10Y" in df.columns
        assert "VIX" in df.columns

    def test_simulated_fred_no_nans(self):
        df = generate_simulated_fred("2020-01-01", "2023-12-31")
        assert df.isna().sum().sum() == 0

    def test_simulated_market_shape(self):
        df = generate_simulated_market("2020-01-01", "2023-12-31")
        assert len(df) > 500
        assert "USDJPY" in df.columns
        assert "NIKKEI" in df.columns
        assert "TLT" in df.columns

    def test_simulated_market_positive_prices(self):
        df = generate_simulated_market("2020-01-01", "2023-12-31")
        assert (df["USDJPY"] > 0).all()
        assert (df["NIKKEI"] > 0).all()

    def test_simulated_fred_realistic_ranges(self):
        df = generate_simulated_fred("2020-01-01", "2023-12-31")
        # Japan 10Y should be near zero to low single digits
        assert df["JP_10Y"].max() < 5.0
        assert df["JP_10Y"].min() > -1.0
        # VIX should be positive
        assert (df["VIX"] > 0).all()


class TestDataStore:
    """Test unified data store."""

    def test_data_store_simulated(self, tmp_path):
        store = DataStore(data_dir=str(tmp_path / "data"), use_simulated=True)
        df = store.get_unified(start="2020-01-01", end="2023-12-31")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert df.index.name == "date"

    def test_data_store_caching(self, tmp_path):
        store = DataStore(data_dir=str(tmp_path / "data"), use_simulated=True)
        df1 = store.get_unified(start="2020-01-01", end="2023-12-31")
        df2 = store.get_unified(start="2020-01-01", end="2023-12-31")
        pd.testing.assert_frame_equal(df1, df2)

    def test_data_store_clear_cache(self, tmp_path):
        store = DataStore(data_dir=str(tmp_path / "data"), use_simulated=True)
        store.get_unified(start="2020-01-01", end="2023-12-31")
        store.clear_cache()
        assert len(store._cache) == 0


class TestConfig:
    """Test configuration values."""

    def test_boj_events_dates_parseable(self):
        for date_str in BOJ_EVENTS.keys():
            pd.Timestamp(date_str)  # Should not raise

    def test_jgb_tenors_sorted(self):
        assert JGB_TENORS == sorted(JGB_TENORS)

    def test_fred_series_not_empty(self):
        assert len(FRED_SERIES) > 5
