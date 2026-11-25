"""
engines/network.py
Loads the saved network models and scores flows into three tiers:
    Known attack / Suspected zero-day / Benign
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import common as C


class ModelsNotFound(RuntimeError):
    pass


class NetworkEngine:
    def __init__(self, models_dir):
        models_dir = Path(models_dir)
        meta_path = models_dir / "ids_meta.json"
        if not meta_path.exists():
            raise ModelsNotFound(
                "Network models not found. Run notebooks/01_network_ids.ipynb first.")
        self.meta = json.loads(meta_path.read_text())
        self.features: list[str] = self.meta["features"]
        self.sup = joblib.load(models_dir / "ids_supervised.joblib")
        self.ano = joblib.load(models_dir / "ids_anomaly.joblib")
        demo_path = models_dir / "ids_supervised_demo.joblib"
        self.sup_demo = joblib.load(demo_path) if demo_path.exists() else None

    # ---- convenient read-only properties -------------------------------
    @property
    def demo_family(self) -> str:
        return self.meta.get("demo_holdout_family", "")

    @property
    def has_demo_model(self) -> bool:
        return self.sup_demo is not None

    def thresholds(self, use_demo_model: bool = False) -> tuple[float, float]:
        if use_demo_model and self.has_demo_model:
            return float(self.meta["t1_demo"]), float(self.meta["t2_demo"])
        return float(self.meta["t1"]), float(self.meta["t2"])

    # ---- input handling -------------------------------------------------
    def prepare(self, df: pd.DataFrame) -> tuple[np.ndarray, int]:
        """Select the model's features (by name) and repair bad values.
        Returns (matrix, number_of_rows_that_needed_repair)."""
        df = df.copy()
        df.columns = df.columns.astype(str).str.strip()
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(
                f"The file is missing {len(missing)} required column(s), e.g. {missing[:5]}. "
                "Use CICIDS2017-style flow CSVs (CICFlowMeter output).")
        X = df[self.features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        bad = ~np.isfinite(X)
        n_bad_rows = int(bad.any(axis=1).sum())
        if n_bad_rows:
            X = np.where(bad, 0.0, X)
        return X, n_bad_rows

    # ---- scoring --------------------------------------------------------
    def predict(self, df: pd.DataFrame, use_demo_model: bool = False,
                chunk: int = 100_000, progress=None) -> pd.DataFrame:
        """Return a DataFrame (same row order as df) with:
        p_attack, anomaly_score, tier, borderline."""
        X, n_fixed = self.prepare(df)
        sup = self.sup_demo if (use_demo_model and self.has_demo_model) else self.sup
        t1, t2 = self.thresholds(use_demo_model)

        p = np.empty(len(X), dtype=np.float32)
        a = np.empty(len(X), dtype=np.float32)
        for start in range(0, len(X), chunk):
            sl = slice(start, start + chunk)
            p[sl] = sup.predict_proba(X[sl])[:, 1]
            a[sl] = C.anomaly_scores(self.ano, X[sl])
            if progress:
                progress(min(start + chunk, len(X)) / len(X))

        tier = C.assign_tiers(p, a, t1, t2)
        out = pd.DataFrame({
            "p_attack": p, "anomaly_score": a, "tier": tier,
            # benign-tier flows that came close to the anomaly threshold
            "borderline": (tier == C.TIER_BENIGN) & (a > 0.95 * t2),
        }, index=df.index)
        out.attrs["repaired_rows"] = n_fixed
        out.attrs["t1"], out.attrs["t2"] = t1, t2
        return out
