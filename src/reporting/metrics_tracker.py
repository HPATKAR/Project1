"""
Accuracy tracking for regime predictions.

Records predicted vs actual regime shifts in SQLite,
computes accuracy/precision/recall/false positive rate,
and generates improvement suggestions.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


_DB_PATH = Path(__file__).resolve().parent.parent.parent / "output" / "data" / "alerts.db"


def _init_metrics_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Initialize the metrics table in the shared database."""
    path = db_path or _DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS regime_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            predicted_regime TEXT,
            actual_regime TEXT,
            probability REAL,
            lead_time_days INTEGER
        )
    """)
    conn.commit()
    return conn


def record_prediction(
    predicted: str,
    actual: str,
    probability: float,
    lead_time_days: int = 0,
    db_path: Optional[Path] = None,
) -> None:
    """Record a single prediction vs actual outcome."""
    conn = _init_metrics_db(db_path)
    conn.execute(
        "INSERT INTO regime_predictions (timestamp, predicted_regime, actual_regime, probability, lead_time_days) "
        "VALUES (datetime('now'), ?, ?, ?, ?)",
        (predicted, actual, probability, lead_time_days),
    )
    conn.commit()
    conn.close()


def get_all_predictions(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Retrieve all recorded predictions."""
    conn = _init_metrics_db(db_path)
    df = pd.read_sql("SELECT * FROM regime_predictions ORDER BY id", conn)
    conn.close()
    return df


@dataclass
class AccuracyMetrics:
    prediction_accuracy: float = 0.0
    average_lead_time: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    false_positive_rate: float = 0.0
    total_predictions: int = 0


class AccuracyTracker:
    """Computes accuracy metrics from recorded predictions."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path

    def compute_metrics(self) -> AccuracyMetrics:
        """Compute all accuracy metrics from the database."""
        df = get_all_predictions(self.db_path)

        if df.empty:
            return AccuracyMetrics()

        total = len(df)
        correct = (df["predicted_regime"] == df["actual_regime"]).sum()
        accuracy = correct / total if total > 0 else 0.0

        avg_lead = df["lead_time_days"].mean() if "lead_time_days" in df.columns else 0.0

        tp = ((df["predicted_regime"] == "repricing") & (df["actual_regime"] == "repricing")).sum()
        fp = ((df["predicted_regime"] == "repricing") & (df["actual_regime"] != "repricing")).sum()
        fn = ((df["predicted_regime"] != "repricing") & (df["actual_regime"] == "repricing")).sum()
        tn = ((df["predicted_regime"] != "repricing") & (df["actual_regime"] != "repricing")).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return AccuracyMetrics(
            prediction_accuracy=accuracy,
            average_lead_time=float(avg_lead) if not np.isnan(avg_lead) else 0.0,
            precision=precision,
            recall=recall,
            false_positive_rate=fpr,
            total_predictions=total,
        )

    def compute_from_series(
        self,
        predictions: pd.Series,
        actuals: pd.Series,
    ) -> AccuracyMetrics:
        """Compute metrics from prediction and actual series directly."""
        common = predictions.index.intersection(actuals.index)
        if len(common) == 0:
            return AccuracyMetrics()

        pred = predictions.loc[common]
        act = actuals.loc[common]

        total = len(common)
        correct = (pred == act).sum()
        accuracy = correct / total

        tp = ((pred == 1) & (act == 1)).sum()
        fp = ((pred == 1) & (act == 0)).sum()
        fn = ((pred == 0) & (act == 1)).sum()
        tn = ((pred == 0) & (act == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return AccuracyMetrics(
            prediction_accuracy=accuracy,
            average_lead_time=0.0,
            precision=precision,
            recall=recall,
            false_positive_rate=fpr,
            total_predictions=total,
        )


def generate_improvement_suggestions(metrics: AccuracyMetrics) -> List[str]:
    """Generate rule-based improvement suggestions."""
    suggestions = []

    if metrics.total_predictions == 0:
        suggestions.append(
            "No predictions recorded yet. Run the ML regime predictor to start tracking accuracy."
        )
        return suggestions

    if metrics.prediction_accuracy < 0.6:
        suggestions.append(
            "Low accuracy ({:.1%}). Consider: (1) increasing training window, "
            "(2) adding more features, (3) tuning RandomForest hyperparameters.".format(
                metrics.prediction_accuracy
            )
        )

    if metrics.false_positive_rate > 0.3:
        suggestions.append(
            "High false positive rate ({:.1%}). Consider increasing alert thresholds "
            "or adding confirmation signals before triggering alerts.".format(
                metrics.false_positive_rate
            )
        )

    if metrics.recall < 0.5:
        suggestions.append(
            "Low recall ({:.1%}). The model is missing actual regime shifts. "
            "Consider: (1) lowering detection thresholds, (2) adding more sensitive "
            "features like carry stress or spillover intensity.".format(
                metrics.recall
            )
        )

    if metrics.precision < 0.5 and metrics.precision > 0:
        suggestions.append(
            "Low precision ({:.1%}). Many predicted shifts did not materialize. "
            "Consider: (1) requiring multiple confirming indicators, "
            "(2) increasing the probability threshold for regime shift alerts.".format(
                metrics.precision
            )
        )

    if metrics.average_lead_time < 2 and metrics.total_predictions > 0:
        suggestions.append(
            "Short average lead time ({:.1f} days). The model is detecting shifts "
            "too late. Consider: (1) using longer-horizon features, "
            "(2) monitoring entropy trends rather than levels.".format(
                metrics.average_lead_time
            )
        )

    if metrics.prediction_accuracy >= 0.8 and metrics.precision >= 0.7:
        suggestions.append(
            "Strong performance (accuracy: {:.1%}, precision: {:.1%}). "
            "Model is performing well. Consider expanding to additional asset classes.".format(
                metrics.prediction_accuracy, metrics.precision
            )
        )

    if not suggestions:
        suggestions.append(
            "Model performance is moderate. Continue monitoring and consider "
            "retraining with more recent data."
        )

    return suggestions
