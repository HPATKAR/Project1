"""
Customizable Dashboard Layout Configuration.

Provides user profiles (Trader/Analyst/Academic), persistent settings,
and a sidebar settings panel for the Streamlit app.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
_SETTINGS_PATH = _CONFIG_DIR / "user_settings.json"
_DEFAULTS_PATH = _CONFIG_DIR / "default_settings.json"


@dataclass
class LayoutConfig:
    """Dashboard layout configuration."""
    profile: str = "Analyst"
    enabled_tabs: List[str] = field(default_factory=lambda: [
        "Overview & Data", "Yield Curve Analytics", "Regime Detection",
        "Spillover & Info Flow", "Trade Ideas", "Intraday FX Event Study",
        "Early Warning", "Performance Review", "AI Q&A",
        "About: Heramb Patkar", "About: Dr. Zhang",
    ])
    default_tab: str = "Regime Detection"
    chart_theme: str = "plotly_dark"
    refresh_interval: int = 60  # minutes
    risk_profile: str = "moderate"  # conservative, moderate, aggressive
    alert_severity_threshold: str = "WARNING"  # INFO, WARNING, CRITICAL
    show_ml_predictions: bool = True
    entropy_window: int = 60


# ── Built-in Profiles ────────────────────────────────────────────────────

PROFILES: Dict[str, LayoutConfig] = {
    "Trader": LayoutConfig(
        profile="Trader",
        default_tab="Trade Ideas",
        refresh_interval=15,
        risk_profile="aggressive",
        alert_severity_threshold="INFO",
        show_ml_predictions=True,
        entropy_window=30,
    ),
    "Analyst": LayoutConfig(
        profile="Analyst",
        default_tab="Regime Detection",
        refresh_interval=60,
        risk_profile="moderate",
        alert_severity_threshold="WARNING",
        show_ml_predictions=True,
        entropy_window=60,
    ),
    "Academic": LayoutConfig(
        profile="Academic",
        default_tab="Yield Curve Analytics",
        refresh_interval=1440,  # daily
        risk_profile="conservative",
        alert_severity_threshold="CRITICAL",
        show_ml_predictions=False,
        entropy_window=120,
    ),
}


# ── Layout Manager ───────────────────────────────────────────────────────

class LayoutManager:
    """Load, save, and manage dashboard layout settings."""

    def __init__(self, settings_path: Optional[Path] = None):
        self.settings_path = settings_path or _SETTINGS_PATH

    def load(self) -> LayoutConfig:
        """Load settings from JSON. Falls back to default profile."""
        if self.settings_path.exists():
            try:
                data = json.loads(self.settings_path.read_text())
                return LayoutConfig(**data)
            except (json.JSONDecodeError, TypeError):
                pass

        # Try defaults
        if _DEFAULTS_PATH.exists():
            try:
                data = json.loads(_DEFAULTS_PATH.read_text())
                return LayoutConfig(**data)
            except (json.JSONDecodeError, TypeError):
                pass

        return LayoutConfig()

    def save(self, config: LayoutConfig) -> None:
        """Persist settings to JSON."""
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings_path.write_text(json.dumps(asdict(config), indent=2))

    def apply_profile(self, profile_name: str) -> LayoutConfig:
        """Apply a built-in profile and save it."""
        config = PROFILES.get(profile_name, LayoutConfig())
        self.save(config)
        return config


def render_settings_panel(st_module, layout_mgr: LayoutManager) -> LayoutConfig:
    """Render settings controls in the sidebar and return the active config."""
    config = layout_mgr.load()

    st_module.markdown(
        "<div style='border-top:1px solid rgba(255,255,255,0.06);"
        "margin:0.4rem 0 0.5rem 0;padding-top:0.5rem;'>"
        "<span style='font-size:var(--fs-tiny);font-weight:700;text-transform:uppercase;"
        "letter-spacing:0.14em;color:rgba(255,255,255,0.4);"
        "font-family:var(--font-sans);'>Dashboard Settings</span></div>",
        unsafe_allow_html=True,
    )

    # Profile selector
    profile_names = list(PROFILES.keys())
    current_idx = profile_names.index(config.profile) if config.profile in profile_names else 1
    selected = st_module.selectbox(
        "Profile",
        profile_names,
        index=current_idx,
        key="settings_profile",
    )

    if selected != config.profile:
        config = layout_mgr.apply_profile(selected)

    # ML predictions toggle
    config.show_ml_predictions = st_module.toggle(
        "Show ML Predictions",
        value=config.show_ml_predictions,
        key="settings_ml",
    )

    # Entropy window
    config.entropy_window = st_module.select_slider(
        "Entropy Window",
        options=[30, 60, 90, 120],
        value=config.entropy_window,
        key="settings_entropy_window",
    )

    # Alert severity
    config.alert_severity_threshold = st_module.selectbox(
        "Alert Threshold",
        ["INFO", "WARNING", "CRITICAL"],
        index=["INFO", "WARNING", "CRITICAL"].index(config.alert_severity_threshold),
        key="settings_alert_severity",
    )

    # Save
    layout_mgr.save(config)
    return config
