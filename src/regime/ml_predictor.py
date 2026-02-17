"""
ML Regime Transition Predictor.

Uses RandomForest with walk-forward training to predict regime shifts
from features derived from the ensemble regime models.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "data" / "cache"


# ── Feature Engineering ──────────────────────────────────────────────────

def compute_regime_features(
    df: pd.DataFrame,
    entropy_window: int = 60,
) -> pd.DataFrame:
    """Compute features for the regime predictor.

    Expected columns in *df*: JP_10Y, US_10Y, USDJPY, VIX (others optional).
    Returns a DataFrame with one row per date and feature columns.
    """
    features = pd.DataFrame(index=df.index)

    # 1. Structural entropy proxy (rolling std of JP_10Y changes)
    jp10 = df["JP_10Y"] if "JP_10Y" in df.columns else pd.Series(dtype=float)
    if len(jp10.dropna()) > entropy_window:
        changes = jp10.diff()
        features["structural_entropy"] = changes.rolling(entropy_window).std()

    # 2. Entropy delta (30d change in entropy)
    if "structural_entropy" in features.columns:
        features["entropy_delta_30d"] = features["structural_entropy"].diff(30)

    # 3. Carry stress (US_10Y - JP_10Y spread, rolling z-score)
    us10 = df["US_10Y"] if "US_10Y" in df.columns else pd.Series(dtype=float)
    if len(jp10.dropna()) > 60 and len(us10.dropna()) > 60:
        spread = us10 - jp10
        spread_mean = spread.rolling(252).mean()
        spread_std = spread.rolling(252).std()
        features["carry_stress"] = (spread - spread_mean) / spread_std.replace(0, np.nan)

    # 4. Max spillover TE proxy (rolling correlation JP_10Y vs US_10Y)
    if len(jp10.dropna()) > 60 and len(us10.dropna()) > 60:
        features["max_spillover_te"] = jp10.rolling(60).corr(us10)

    # 5. Vol z-score (GARCH proxy: rolling realized vol of JP_10Y changes)
    if len(jp10.dropna()) > 60:
        realized_vol = jp10.diff().rolling(21).std()
        vol_mean = realized_vol.rolling(252).mean()
        vol_std = realized_vol.rolling(252).std()
        features["vol_zscore"] = (realized_vol - vol_mean) / vol_std.replace(0, np.nan)

    # 6. VIX level
    if "VIX" in df.columns:
        features["vix_level"] = df["VIX"]

    # 7. USDJPY momentum (20d change)
    if "USDJPY" in df.columns:
        features["usdjpy_momentum"] = df["USDJPY"].pct_change(20)

    return features.dropna()


def create_regime_labels(
    ensemble_prob: pd.Series,
    threshold: float = 0.5,
    forward_days: int = 5,
) -> pd.Series:
    """Create binary labels: 1 if regime prob > threshold within next forward_days.

    This creates a *forward-looking* label for supervised training.
    """
    future_max = ensemble_prob.rolling(forward_days, min_periods=1).max().shift(-forward_days)
    labels = (future_max > threshold).astype(int)
    return labels.dropna()


# ── ML Regime Predictor ──────────────────────────────────────────────────

class MLRegimePredictor:
    """Walk-forward RandomForest regime predictor."""

    def __init__(
        self,
        train_window: int = 504,
        retrain_freq: int = 63,
        n_estimators: int = 50,
        max_depth: int = 5,
    ):
        self.train_window = train_window
        self.retrain_freq = retrain_freq  # ~quarterly retraining
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self._model_path = _CACHE_DIR / "regime_model.pkl"

    def fit_predict(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> Tuple[pd.Series, pd.Series, Optional[pd.Series]]:
        """Walk-forward fit and predict.

        Retrains quarterly (every retrain_freq days) and batch-predicts
        all points until the next retrain, making it much faster than
        point-by-point prediction.

        Returns (predictions, probabilities, feature_importance).
        """
        common_idx = features.index.intersection(labels.index)
        if len(common_idx) < self.train_window + 20:
            return pd.Series(dtype=float), pd.Series(dtype=float), None

        X = features.loc[common_idx]
        y = labels.loc[common_idx]
        n = len(common_idx)

        predictions = pd.Series(np.nan, index=common_idx, dtype=float)
        probabilities = pd.Series(np.nan, index=common_idx, dtype=float)
        importance = None

        # Build list of retrain points
        retrain_points = list(range(self.train_window, n, self.retrain_freq))
        if not retrain_points:
            return pd.Series(dtype=float), pd.Series(dtype=float), None

        for idx, train_end in enumerate(retrain_points):
            train_start = max(0, train_end - self.train_window)
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]

            # Need both classes for meaningful prediction
            if y_train.nunique() < 2:
                continue

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_train)

            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1,
            )
            self.model.fit(X_scaled, y_train)
            importance = pd.Series(
                self.model.feature_importances_,
                index=X.columns,
            ).sort_values(ascending=False)

            # Batch-predict from train_end to next retrain point (or end)
            pred_end = retrain_points[idx + 1] if idx + 1 < len(retrain_points) else n
            X_pred = X.iloc[train_end:pred_end]
            if len(X_pred) == 0:
                continue
            X_pred_scaled = self.scaler.transform(X_pred)

            preds = self.model.predict(X_pred_scaled)
            proba = self.model.predict_proba(X_pred_scaled)

            predictions.iloc[train_end:pred_end] = preds
            # predict_proba may return 1 column if only one class was seen
            if proba.shape[1] >= 2:
                probabilities.iloc[train_end:pred_end] = proba[:, 1]
            else:
                col_class = self.model.classes_[0]
                probabilities.iloc[train_end:pred_end] = proba[:, 0] if col_class == 1 else 1.0 - proba[:, 0]

        # Trim to valid predictions
        valid = probabilities.dropna()
        return (
            predictions.loc[valid.index],
            probabilities.loc[valid.index],
            importance,
        )

    def save_model(self) -> None:
        """Persist model to disk."""
        if self.model is not None:
            self._model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._model_path, "wb") as f:
                pickle.dump({"model": self.model, "scaler": self.scaler}, f)

    def load_model(self) -> bool:
        """Load persisted model. Returns True if successful."""
        if self._model_path.exists():
            with open(self._model_path, "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.scaler = data["scaler"]
                return True
        return False
