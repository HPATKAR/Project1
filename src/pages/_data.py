"""Cached data loaders shared across all page modules."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data.data_store import DataStore


@st.cache_resource(ttl=3600, max_entries=2)
def get_data_store(simulated: bool) -> DataStore:
    return DataStore(use_simulated=simulated)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
def load_unified(simulated: bool, start: str, end: str, api_key: str | None):
    store = get_data_store(simulated)
    return store.get_unified(start=start, end=end, fred_api_key=api_key or None)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
def load_rates(simulated: bool, start: str, end: str, api_key: str | None):
    store = get_data_store(simulated)
    return store.get_rates(start=start, end=end, fred_api_key=api_key or None)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=4)
def load_market(simulated: bool, start: str, end: str):
    store = get_data_store(simulated)
    return store.get_market(start=start, end=end)


def _safe_col(df: pd.DataFrame, col: str) -> pd.Series | None:
    """Return column as Series if present, else None."""
    if col in df.columns:
        return df[col].dropna()
    return None
