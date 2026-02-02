"""Spillover and contagion analysis for JGB repricing framework.

Implements cross-market spillover measurement using:
- Granger causality tests (pairwise, lag-optimized)
- Diebold-Yilmaz spillover index (VAR-based FEVD)
- Dynamic conditional correlation (DCC-GARCH)
- Transfer entropy (information-theoretic directional flow)
"""
