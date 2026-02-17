"""
Alert System for JGB Repricing Framework.

SQLite-backed persistence, threshold-based detection, and Streamlit
toast notifications for real-time regime and market alerts.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


_DB_PATH = Path(__file__).resolve().parent.parent.parent / "output" / "data" / "alerts.db"


# ── Data Classes ─────────────────────────────────────────────────────────

@dataclass
class Alert:
    """A single alert event."""
    timestamp: str
    alert_type: str
    severity: str  # INFO, WARNING, CRITICAL
    title: str
    description: str
    value: float = 0.0
    threshold: float = 0.0


@dataclass
class AlertThresholds:
    """Configurable alert thresholds."""
    correlation_anomaly: float = 0.5
    spillover_surge_zscore: float = 2.0
    warning_score_info: float = 30.0
    warning_score_warning: float = 50.0
    warning_score_critical: float = 80.0
    regime_shift_prob: float = 0.5


# ── SQLite Persistence ───────────────────────────────────────────────────

def _init_alert_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Initialize the alerts table."""
    path = db_path or _DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            alert_type TEXT,
            severity TEXT,
            title TEXT,
            description TEXT,
            value REAL,
            threshold REAL
        )
    """)
    conn.commit()
    return conn


def save_alert(alert: Alert, db_path: Optional[Path] = None) -> None:
    """Persist an alert to SQLite."""
    conn = _init_alert_db(db_path)
    conn.execute(
        "INSERT INTO alerts (timestamp, alert_type, severity, title, description, value, threshold) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (alert.timestamp, alert.alert_type, alert.severity, alert.title,
         alert.description, alert.value, alert.threshold),
    )
    conn.commit()
    conn.close()


def get_recent_alerts(
    n: int = 20,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Retrieve the most recent alerts."""
    conn = _init_alert_db(db_path)
    df = pd.read_sql(
        f"SELECT * FROM alerts ORDER BY id DESC LIMIT {n}", conn,
    )
    conn.close()
    return df


# ── Alert Detector ───────────────────────────────────────────────────────

class AlertDetector:
    """Checks market conditions against configurable thresholds."""

    def __init__(self, thresholds: Optional[AlertThresholds] = None):
        self.thresholds = thresholds or AlertThresholds()

    def check_all_conditions(
        self,
        jp_us_correlation: Optional[float] = None,
        spillover_zscore: Optional[float] = None,
        warning_score: Optional[float] = None,
        regime_prob: Optional[float] = None,
    ) -> List[Alert]:
        """Check all alert conditions and return triggered alerts."""
        alerts = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Correlation anomaly
        if jp_us_correlation is not None and abs(jp_us_correlation) > self.thresholds.correlation_anomaly:
            alerts.append(Alert(
                timestamp=now,
                alert_type="correlation_anomaly",
                severity="WARNING",
                title="JP-US Yield Correlation Anomaly",
                description=f"JP_10Y vs US_10Y correlation at {jp_us_correlation:.2f} "
                            f"exceeds threshold of {self.thresholds.correlation_anomaly:.2f}. "
                            f"Markets are unusually coupled; diversification may be impaired.",
                value=jp_us_correlation,
                threshold=self.thresholds.correlation_anomaly,
            ))

        # 2. Spillover surge
        if spillover_zscore is not None and spillover_zscore > self.thresholds.spillover_surge_zscore:
            severity = "CRITICAL" if spillover_zscore > 3.0 else "WARNING"
            alerts.append(Alert(
                timestamp=now,
                alert_type="spillover_surge",
                severity=severity,
                title="Spillover Surge Detected",
                description=f"Cross-market spillover z-score at {spillover_zscore:.1f} "
                            f"(threshold: {self.thresholds.spillover_surge_zscore:.1f}). "
                            f"Contagion risk is elevated.",
                value=spillover_zscore,
                threshold=self.thresholds.spillover_surge_zscore,
            ))

        # 3. Warning score breach
        if warning_score is not None:
            t = self.thresholds
            if warning_score > t.warning_score_critical:
                alerts.append(Alert(
                    timestamp=now,
                    alert_type="warning_score",
                    severity="CRITICAL",
                    title="Critical Warning Score",
                    description=f"Composite warning score at {warning_score:.0f}/100. "
                                f"Multiple stress indicators firing simultaneously.",
                    value=warning_score,
                    threshold=t.warning_score_critical,
                ))
            elif warning_score > t.warning_score_warning:
                alerts.append(Alert(
                    timestamp=now,
                    alert_type="warning_score",
                    severity="WARNING",
                    title="Elevated Warning Score",
                    description=f"Warning score at {warning_score:.0f}/100. "
                                f"Monitor for potential regime shift.",
                    value=warning_score,
                    threshold=t.warning_score_warning,
                ))

        # 4. Regime shift probability
        if regime_prob is not None and regime_prob > self.thresholds.regime_shift_prob:
            severity = "CRITICAL" if regime_prob > 0.7 else "WARNING"
            alerts.append(Alert(
                timestamp=now,
                alert_type="regime_shift",
                severity=severity,
                title="Regime Shift Signal",
                description=f"Ensemble regime probability at {regime_prob:.0%}. "
                            f"{'Strong repricing signal from all models.' if regime_prob > 0.7 else 'Moderate repricing signal.'}",
                value=regime_prob,
                threshold=self.thresholds.regime_shift_prob,
            ))

        return alerts


# ── Streamlit Notification Helper ────────────────────────────────────────

class AlertNotifier:
    """Manage alerts in a Streamlit session."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path

    def process_alerts(self, alerts: List[Alert]) -> List[Alert]:
        """Save alerts to DB and return them for display."""
        new_alerts = []
        for alert in alerts:
            save_alert(alert, self.db_path)
            new_alerts.append(alert)
        return new_alerts

    def render_sidebar_log(self, st_module) -> None:
        """Render recent alerts in a sidebar expander."""
        recent = get_recent_alerts(20, self.db_path)
        if recent.empty:
            return

        with st_module.expander(f"Alert Log ({len(recent)} recent)", expanded=False):
            for _, row in recent.iterrows():
                severity_color = {
                    "CRITICAL": "#dc2626",
                    "WARNING": "#d97706",
                    "INFO": "#2563eb",
                }.get(row.get("severity", "INFO"), "#6b7280")
                st_module.markdown(
                    f"<div style='border-left:3px solid {severity_color};"
                    f"padding:4px 8px;margin-bottom:6px;font-size:0.75rem;'>"
                    f"<b style='color:{severity_color}'>{row.get('severity', 'INFO')}</b> "
                    f"<span style='color:#666'>{row.get('timestamp', '')[:16]}</span><br>"
                    f"{row.get('title', '')}</div>",
                    unsafe_allow_html=True,
                )
