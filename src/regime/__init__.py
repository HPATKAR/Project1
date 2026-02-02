"""
src.regime -- Regime detection and classification for JGB repricing framework.

This module provides multiple complementary approaches to detect shifts in the
Japanese Government Bond (JGB) market regime:

    * **markov_switching** -- Hamilton Markov-switching regression models that
      estimate smoothed regime probabilities from univariate yield-change series.
    * **hmm_regime** -- Multivariate Hidden Markov Models (Gaussian HMM) that
      jointly model JGB yield changes, USD/JPY returns, and volatility indices
      to infer latent market states.
    * **structural_breaks** -- Change-point detection via PELT and binary
      segmentation algorithms (ruptures) for identifying discrete structural
      shifts in yield levels or spreads.
    * **entropy_regime** -- Information-theoretic regime detection using rolling
      permutation entropy and sample entropy to capture changes in the
      complexity / predictability of yield dynamics.
    * **garch_regime** -- GARCH and EGARCH volatility modelling with structural
      break detection on conditional volatility to identify volatility regime
      transitions.
    * **ensemble** -- Ensemble combiner that fuses signals from all five
      methods into a single regime probability (0 = BoJ-suppressed,
      1 = market-driven repricing).

Typical usage::

    from src.regime.markov_switching import fit_markov_regime
    from src.regime.hmm_regime import fit_multivariate_hmm
    from src.regime.structural_breaks import detect_breaks_pelt
    from src.regime.entropy_regime import rolling_permutation_entropy
    from src.regime.garch_regime import fit_garch
    from src.regime.ensemble import ensemble_regime_probability
"""

from src.regime.markov_switching import fit_markov_regime, classify_current_regime
from src.regime.hmm_regime import fit_multivariate_hmm, predict_regime
from src.regime.structural_breaks import (
    detect_breaks_pelt,
    detect_breaks_binseg,
    plot_breaks,
)
from src.regime.entropy_regime import (
    rolling_permutation_entropy,
    rolling_sample_entropy,
    entropy_regime_signal,
)
from src.regime.garch_regime import fit_garch, fit_egarch, volatility_regime_breaks
from src.regime.ensemble import ensemble_regime_probability

__all__ = [
    "fit_markov_regime",
    "classify_current_regime",
    "fit_multivariate_hmm",
    "predict_regime",
    "detect_breaks_pelt",
    "detect_breaks_binseg",
    "plot_breaks",
    "rolling_permutation_entropy",
    "rolling_sample_entropy",
    "entropy_regime_signal",
    "fit_garch",
    "fit_egarch",
    "volatility_regime_breaks",
    "ensemble_regime_probability",
]
