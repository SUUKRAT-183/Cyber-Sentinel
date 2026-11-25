"""
engines/events.py
Turns engine outputs into entries for ONE combined "Security events" feed.

IMPORTANT (be upfront about this in the demo and the slides):
severity here is a documented set of RULES, not a learned risk model.
The network and email datasets share no users or events, so a statistically
validated combined risk score is not possible.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from . import common as C

SEVERITY_LABEL = {"High": "High", "Medium": "Medium", "Low": "Low"}

RULES = [
    ("Network", "Known attack flows present", "High"),
    ("Network", "Suspected zero-day flows present (anomaly only)", "Medium"),
    ("Network", "Borderline anomalies (just under the anomaly threshold)", "Low"),
    ("Email", "Phishing probability >= 0.98", "High"),
    ("Email", "Phishing probability above the tuned threshold", "Medium"),
]


def _event(source: str, severity: str, title: str, detail: str) -> dict:
    return {
        "time": datetime.now().strftime("%H:%M:%S"),
        "source": source,
        "severity": severity,
        "title": title,
        "detail": detail,
    }


def _top_ports(raw: pd.DataFrame | None, mask, port_col: str = "Destination Port", k: int = 3) -> str:
    if raw is None or port_col not in raw.columns:
        return ""
    ports = raw.loc[mask, port_col].dropna()
    if ports.empty:
        return ""
    top = ports.astype(int).value_counts().head(k)
    return " Most targeted ports: " + ", ".join(f"{p} ({n:,})" for p, n in top.items()) + "."


def network_events(res: pd.DataFrame, raw: pd.DataFrame | None = None, source_name: str = "scan") -> list[dict]:
    """Summarise one network scan into at most three events."""
    events: list[dict] = []
    n = len(res)
    known = (res["tier"] == C.TIER_KNOWN).to_numpy()
    suspect = (res["tier"] == C.TIER_SUSPECT).to_numpy()
    border = res["borderline"].to_numpy()

    if known.any():
        events.append(_event(
            "Network", "High", "Known attack traffic detected",
            f"{known.sum():,} of {n:,} flows in {source_name} matched known attack behaviour."
            + _top_ports(raw, known)))
    if suspect.any():
        events.append(_event(
            "Network", "Medium", "Suspected zero-day traffic (needs analyst review)",
            f"{suspect.sum():,} flows look unusual but do not match a known attack."
            + _top_ports(raw, suspect)))
    if border.any():
        events.append(_event(
            "Network", "Low", "Borderline anomalies",
            f"{border.sum():,} flows scored just below the anomaly threshold."))
    return events


def email_event(result: dict, preview: str = "") -> dict | None:
    """One email check -> an event, or None when the email looks legitimate."""
    p = result["probability"]
    if not result["is_phishing"]:
        return None
    sev = "High" if p >= 0.98 else "Medium"
    snippet = (preview[:70] + "...") if len(preview) > 70 else preview
    return _event("Email", sev, "Phishing / spam email flagged",
                  f"Probability {p:.1%}. Starts: \"{snippet}\"")
