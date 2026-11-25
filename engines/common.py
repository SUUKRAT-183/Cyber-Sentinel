"""
engines/common.py
Shared helpers used by BOTH the notebooks and the Streamlit app.

Keeping them in one module matters: the saved models reference
`signed_log1p` by name, so the app must be able to import it from here.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# --------------------------------------------------------------------------
# Network data (CICIDS2017)
# --------------------------------------------------------------------------
LABEL_COL = "Label"
BENIGN = "BENIGN"

# Columns that are identifiers / shortcuts, never used as model features.
# 'Destination Port' is kept in the data only for display in the dashboard.
NOT_FEATURES = {
    "Label", "Family", "Day", "Flow ID", "Source IP", "Source Port",
    "Destination IP", "Destination Port", "Protocol", "Timestamp",
}
# Exact duplicate of 'Fwd Header Length' in CICIDS2017.
DUPLICATE_COLUMNS = ["Fwd Header Length.1"]

TIER_KNOWN = "Known attack"
TIER_SUSPECT = "Suspected zero-day"
TIER_BENIGN = "Benign"


def clean_label(s: pd.Series) -> pd.Series:
    """Fix the garbled 'Web Attack <bad char> XSS' labels."""
    s = s.astype(str).str.strip()
    return s.str.replace(
        r"^Web Attack.*?(Brute Force|XSS|Sql Injection)$",
        lambda m: "Web Attack - " + ("SQL Injection" if m.group(1) == "Sql Injection" else m.group(1)),
        regex=True,
    )


def family_of(label: str) -> str:
    """Group the 15 raw labels into attack families (used for the zero-day test)."""
    if label == BENIGN:
        return BENIGN
    if label.startswith("DoS"):
        return "DoS"
    if label == "DDoS":
        return "DDoS"
    if label == "PortScan":
        return "PortScan"
    if label.endswith("Patator"):
        return "Patator"
    if label == "Bot":
        return "Bot"
    if label.startswith("Web Attack"):
        return "WebAttack"
    return "Rare"  # Infiltration, Heartbleed: too few rows to evaluate alone


def load_cicids(data_dir, verbose: bool = True) -> pd.DataFrame:
    """
    Read every CSV in data_dir and return one clean float32 DataFrame.
    Cleaning (all of it is label-independent, so it cannot leak):
      * strip column names, repair labels
      * drop exact duplicate rows (done per file, before anything else)
      * drop the duplicate column 'Fwd Header Length.1'
      * drop rows with inf / NaN and rows with negative Flow Duration
    """
    files = sorted(Path(data_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    parts = []
    for f in files:
        df = pd.read_csv(f, low_memory=False, encoding="latin1")
        df.columns = df.columns.str.strip()
        df = df.drop(columns=[c for c in ["Day"] if c in df.columns])
        df[LABEL_COL] = clean_label(df[LABEL_COL])
        n0 = len(df)
        df = df.drop_duplicates()
        num_cols = [c for c in df.columns if c != LABEL_COL]
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").astype("float32")
        if verbose:
            print(f"{f.name:55s} rows {n0:>8,} -> {len(df):>8,} after removing duplicates")
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df[LABEL_COL] = df[LABEL_COL].astype(str).astype("category")
    df = df.drop(columns=[c for c in DUPLICATE_COLUMNS if c in df.columns])

    num_cols = [c for c in df.columns if c != LABEL_COL]
    vals = df[num_cols].to_numpy()
    bad = ~np.isfinite(vals).all(axis=1)
    if "Flow Duration" in df.columns:
        bad |= (df["Flow Duration"].to_numpy() < 0)
    if verbose:
        print(f"Dropping {int(bad.sum()):,} rows with inf/NaN/negative duration")
    df = df.loc[~bad].reset_index(drop=True)
    return df


def get_features(df: pd.DataFrame) -> list[str]:
    """Numeric columns that are not identifiers and are not constant."""
    cols = [c for c in df.columns if c not in NOT_FEATURES]
    nun = df[cols].nunique()
    return [c for c in cols if nun[c] > 1]


# --------------------------------------------------------------------------
# Models (shared definitions so notebook + app agree)
# --------------------------------------------------------------------------
def signed_log1p(X):
    """Squash heavy-tailed network features while keeping the sign."""
    X = np.asarray(X, dtype=np.float32)
    return np.sign(X) * np.log1p(np.abs(X))


def make_supervised(n_estimators=100, max_depth=14, percentile=60, n_jobs=-1, seed=42):
    """Supervised layer: log transform -> keep top features -> Random Forest."""
    return Pipeline([
        ("log", FunctionTransformer(signed_log1p)),
        ("select", SelectPercentile(f_classif, percentile=percentile)),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=3,
            class_weight="balanced_subsample", n_jobs=n_jobs, random_state=seed)),
    ])


def make_anomaly(n_estimators=200, max_samples=65536, n_jobs=-1, seed=42):
    """Anomaly layer: trained on BENIGN flows only."""
    return Pipeline([
        ("log", FunctionTransformer(signed_log1p)),
        ("scale", StandardScaler()),
        ("iso", IsolationForest(
            n_estimators=n_estimators, max_samples=max_samples,
            n_jobs=n_jobs, random_state=seed)),
    ])


def anomaly_scores(pipe, X) -> np.ndarray:
    """Higher = more unusual."""
    return -pipe.score_samples(X)


def assign_tiers(p, a, t1: float, t2: float) -> np.ndarray:
    """
    Decision fusion.
      p  : supervised attack probability     (t1 = its threshold)
      a  : anomaly score                      (t2 = its threshold)
    """
    tier = np.full(len(p), TIER_BENIGN, dtype=object)
    tier[np.asarray(a) > t2] = TIER_SUSPECT
    tier[np.asarray(p) > t1] = TIER_KNOWN
    return tier


# --------------------------------------------------------------------------
# Phishing / spam text
# --------------------------------------------------------------------------
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_EMAIL = re.compile(r"\S+@\S+")
_NONALNUM = re.compile(r"[^a-z0-9]+")
MAX_CHARS = 5000


def clean_text(text: str) -> str:
    t = str(text)[:MAX_CHARS * 2].lower()
    t = _URL.sub(" urltoken ", t)
    t = _EMAIL.sub(" emailtoken ", t)
    t = _NONALNUM.sub(" ", t).strip()
    return t[:MAX_CHARS]
