"""
engines/phishing.py
predict_email(text) -> verdict, probability and the words that drove the decision.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from . import common as C


class PhishingModelNotFound(RuntimeError):
    pass


class PhishingEngine:
    def __init__(self, models_dir):
        models_dir = Path(models_dir)
        meta_path = models_dir / "phishing_meta.json"
        if not meta_path.exists():
            raise PhishingModelNotFound(
                "Phishing model not found. Run notebooks/02_phishing_nlp.ipynb first.")
        self.meta = json.loads(meta_path.read_text())
        self.threshold = float(self.meta["threshold"])
        self.pipe = joblib.load(models_dir / "phishing_pipeline.joblib")
        self._vec = self.pipe.named_steps["tfidf"]
        self._clf = self.pipe.named_steps["clf"]
        self._names = np.array(self._vec.get_feature_names_out())

    def predict(self, text: str, top_k: int = 8) -> dict:
        text = (text or "").strip()
        if len(text) < 3:
            raise ValueError("Please paste some email text first.")
        cleaned = C.clean_text(text)
        x = self._vec.transform([cleaned])
        prob = float(self._clf.predict_proba(x)[0, 1])

        # per-word contribution = tf-idf value * model weight
        coo = x.tocoo()
        contrib = coo.data * self._clf.coef_[0][coo.col]
        words = self._names[coo.col]
        order = np.argsort(contrib)
        toward_phish = [(words[i], float(contrib[i])) for i in order[::-1][:top_k] if contrib[i] > 0]
        toward_legit = [(words[i], float(contrib[i])) for i in order[:top_k] if contrib[i] < 0]

        return {
            "probability": prob,
            "threshold": self.threshold,
            "is_phishing": prob > self.threshold,
            "verdict": "Phishing / spam" if prob > self.threshold else "Looks legitimate",
            "words_toward_phishing": toward_phish,
            "words_toward_legitimate": toward_legit,
        }
