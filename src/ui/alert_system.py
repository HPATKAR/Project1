"""
Alert System for JGB Repricing Framework.

SQLite-backed persistence, threshold-based detection, and Streamlit
toast notifications for real-time regime and market alerts.
"""
from __future__ import annotations

import re
import smtplib
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
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


# ── Email Subscriber Persistence ─────────────────────────────────────────

def _init_subscriber_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Initialize the subscribers table."""
    path = db_path or _DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS subscribers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            subscribed_at TEXT NOT NULL,
            active INTEGER DEFAULT 1,
            min_severity TEXT DEFAULT 'WARNING'
        )
    """)
    conn.commit()
    return conn


def subscribe_email(email: str, min_severity: str = "WARNING", db_path: Optional[Path] = None) -> bool:
    """Add an email subscriber. Returns True if new, False if already exists."""
    email = email.strip().lower()
    if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email):
        raise ValueError(f"Invalid email: {email}")
    conn = _init_subscriber_db(db_path)
    try:
        conn.execute(
            "INSERT INTO subscribers (email, subscribed_at, active, min_severity) VALUES (?, ?, 1, ?)",
            (email, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), min_severity),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        conn.execute(
            "UPDATE subscribers SET active = 1, min_severity = ? WHERE email = ?",
            (min_severity, email),
        )
        conn.commit()
        return False
    finally:
        conn.close()


def unsubscribe_email(email: str, db_path: Optional[Path] = None) -> bool:
    """Deactivate an email subscriber. Returns True if found."""
    email = email.strip().lower()
    conn = _init_subscriber_db(db_path)
    cursor = conn.execute("UPDATE subscribers SET active = 0 WHERE email = ?", (email,))
    conn.commit()
    found = cursor.rowcount > 0
    conn.close()
    return found


def get_active_subscribers(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Retrieve all active subscribers."""
    conn = _init_subscriber_db(db_path)
    df = pd.read_sql("SELECT * FROM subscribers WHERE active = 1 ORDER BY subscribed_at DESC", conn)
    conn.close()
    return df


def get_subscriber_count(db_path: Optional[Path] = None) -> int:
    """Return count of active subscribers."""
    conn = _init_subscriber_db(db_path)
    count = conn.execute("SELECT COUNT(*) FROM subscribers WHERE active = 1").fetchone()[0]
    conn.close()
    return count


_SEVERITY_ORDER = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}


def send_alert_emails(
    alerts: List[Alert],
    smtp_host: str = "",
    smtp_port: int = 587,
    smtp_user: str = "",
    smtp_pass: str = "",
    from_addr: str = "",
    db_path: Optional[Path] = None,
) -> int:
    """Send alert emails to all active subscribers whose severity threshold is met.

    Returns the number of emails sent. If SMTP is not configured, returns 0
    silently (alerts are still persisted to DB for dashboard display).
    """
    if not smtp_host or not smtp_user:
        return 0

    subscribers = get_active_subscribers(db_path)
    if subscribers.empty:
        return 0

    sent = 0
    for _, sub in subscribers.iterrows():
        sub_min = _SEVERITY_ORDER.get(sub["min_severity"], 1)
        relevant = [a for a in alerts if _SEVERITY_ORDER.get(a.severity, 0) >= sub_min]
        if not relevant:
            continue

        body_lines = []
        for a in relevant:
            body_lines.append(f"[{a.severity}] {a.title}\n{a.description}\n")

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"JGB Alert: {relevant[0].severity} — {relevant[0].title}"
        msg["From"] = from_addr or smtp_user
        msg["To"] = sub["email"]

        plain = "JGB Repricing Framework — Early Warning Alert\n\n" + "\n".join(body_lines)
        html = _build_alert_email_html(relevant)
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html, "html"))

        try:
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
            sent += 1
        except Exception:
            pass  # fail silently per subscriber — don't block dashboard

    return sent


def _build_alert_email_html(alerts: List[Alert]) -> str:
    """Build a styled HTML email body for alert notifications."""
    _sev_color = {"CRITICAL": "#dc2626", "WARNING": "#d97706", "INFO": "#2563eb"}
    rows = ""
    for a in alerts:
        color = _sev_color.get(a.severity, "#6b7280")
        rows += (
            f"<tr><td style='border-left:4px solid {color};padding:12px 16px;'>"
            f"<strong style='color:{color};'>{a.severity}</strong><br>"
            f"<span style='font-size:16px;font-weight:600;'>{a.title}</span><br>"
            f"<span style='color:#555;'>{a.description}</span>"
            f"</td></tr>"
        )
    return (
        "<html><body style='font-family:Arial,sans-serif;max-width:600px;margin:0 auto;'>"
        "<div style='background:#000;color:#CFB991;padding:20px;text-align:center;'>"
        "<h2 style='margin:0;'>JGB Repricing Framework</h2>"
        "<p style='margin:4px 0 0 0;font-size:12px;letter-spacing:0.1em;'>EARLY WARNING SYSTEM</p></div>"
        f"<table style='width:100%;border-collapse:collapse;margin-top:16px;'>{rows}</table>"
        "<p style='color:#999;font-size:11px;margin-top:24px;text-align:center;'>"
        "Purdue Daniels School of Business — Rates Strategy Desk<br>"
        "To unsubscribe, visit the dashboard and remove your email.</p>"
        "</body></html>"
    )


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

    def __init__(self, db_path: Optional[Path] = None, smtp_config: Optional[Dict] = None):
        self.db_path = db_path
        self.smtp_config = smtp_config or {}

    def process_alerts(self, alerts: List[Alert]) -> List[Alert]:
        """Save alerts to DB, send email notifications, and return for display."""
        new_alerts = []
        for alert in alerts:
            save_alert(alert, self.db_path)
            new_alerts.append(alert)
        if new_alerts and self.smtp_config.get("smtp_host"):
            try:
                send_alert_emails(new_alerts, db_path=self.db_path, **self.smtp_config)
            except Exception:
                pass
        return new_alerts

    def render_sidebar_log(self, st_module) -> None:
        """Render recent alerts in a clean, compact sidebar expander."""
        recent = get_recent_alerts(20, self.db_path)
        if recent.empty:
            return

        # Count by severity for badge
        sev_counts = recent["severity"].value_counts().to_dict() if "severity" in recent.columns else {}
        crit = sev_counts.get("CRITICAL", 0)
        warn = sev_counts.get("WARNING", 0)

        badge = ""
        if crit:
            badge += f"<span style='background:#dc2626;color:#fff;border-radius:3px;padding:1px 5px;font-size:var(--fs-xs);font-weight:700;margin-right:3px;'>{crit} CRIT</span>"
        if warn:
            badge += f"<span style='background:#d97706;color:#fff;border-radius:3px;padding:1px 5px;font-size:var(--fs-xs);font-weight:700;'>{warn} WARN</span>"

        st_module.markdown(
            "<div style='border-top:1px solid rgba(255,255,255,0.06);"
            "margin:0.4rem 0 0.3rem 0;padding-top:0.5rem;'>"
            "<span style='font-size:var(--fs-tiny);font-weight:700;text-transform:uppercase;"
            "letter-spacing:0.14em;color:rgba(255,255,255,0.4);"
            f"font-family:var(--font-sans);'>Alerts</span> {badge}</div>",
            unsafe_allow_html=True,
        )

        with st_module.expander(f"{len(recent)} recent alerts", expanded=False):
            _sev_icon = {"CRITICAL": "\u25cf", "WARNING": "\u25b2", "INFO": "\u25cb"}
            _sev_color = {"CRITICAL": "#dc2626", "WARNING": "#d97706", "INFO": "#2563eb"}
            _sev_bg = {"CRITICAL": "rgba(220,38,38,0.08)", "WARNING": "rgba(217,119,6,0.06)", "INFO": "rgba(37,99,235,0.05)"}

            for _, row in recent.iterrows():
                sev = row.get("severity", "INFO")
                color = _sev_color.get(sev, "#6b7280")
                bg = _sev_bg.get(sev, "transparent")
                icon = _sev_icon.get(sev, "\u25cb")
                ts = str(row.get("timestamp", ""))[:16]
                title = row.get("title", "")

                st_module.markdown(
                    f"<div style='background:{bg};border-left:3px solid {color};"
                    f"border-radius:0 4px 4px 0;padding:5px 8px;margin-bottom:4px;"
                    f"font-family:var(--font-sans);'>"
                    f"<div style='display:flex;align-items:center;gap:5px;'>"
                    f"<span style='color:{color};font-size:var(--fs-xs);'>{icon}</span>"
                    f"<span style='color:{color};font-size:var(--fs-xs);font-weight:700;letter-spacing:0.08em;'>{sev}</span>"
                    f"<span style='color:rgba(255,255,255,0.35);font-size:var(--fs-tiny);margin-left:auto;'>{ts}</span>"
                    f"</div>"
                    f"<div style='color:rgba(255,255,255,0.75);font-size:var(--fs-base);margin-top:2px;line-height:1.3;'>{title}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
