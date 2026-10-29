"""Logique de prédiction, séparée de FastAPI pour pouvoir la tester simplement."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import COLUMNS, NEGATIVE_LABEL, POSITIVE_LABEL, TARGET


class Predictor:
    """Charge le modèle entraîné et son seuil de décision, puis prédit pour une ou plusieurs personnes."""

    def __init__(self, model, meta: dict | None = None):
        self.model = model
        self.meta = meta or {}
        self.threshold = float(self.meta.get("seuil", 0.5))

    @classmethod
    def from_files(cls, model_path: Path | str, meta_path: Path | str | None = None) -> "Predictor":
        model = joblib.load(model_path)
        meta = {}
        if meta_path and Path(meta_path).exists():
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        return cls(model, meta)

    @staticmethod
    def to_frame(records: list[dict]) -> pd.DataFrame:
        """Met les données au format attendu par le pipeline (mêmes colonnes qu'à l'entraînement).

        Les colonnes non utilisées par le modèle (fnlwgt, education) sont remplies avec NaN.
        """
        cols = [c for c in COLUMNS if c != TARGET]
        df = pd.DataFrame(records).reindex(columns=cols)
        return df.replace({None: np.nan})

    def predict(self, records: list[dict]) -> list[dict]:
        """Renvoie, pour chaque personne, la probabilité de >50K et la classe prédite."""
        proba = self.model.predict_proba(self.to_frame(records))[:, 1]
        return [
            {
                "probabilite_50k": round(float(p), 4),
                "prediction": POSITIVE_LABEL if p >= self.threshold else NEGATIVE_LABEL,
                "seuil": round(self.threshold, 4),
            }
            for p in proba
        ]
