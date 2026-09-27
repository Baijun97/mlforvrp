from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    import joblib
except Exception:
    joblib = None


class ModelScorer:
    """
    Loads an XGBoost/sklearn-style model and returns one selection score per
    feasible candidate row.

    Recommended teammate export:
        model.save_model("xgboost_model.json")
    """

    def __init__(
        self,
        model_path: Path | str,
        feature_columns: Sequence[str],
        model_type: str = "auto",
    ):
        self.model_path = Path(model_path)
        self.feature_columns = list(feature_columns)
        self.model_type = model_type
        self.model = self._load_model()

    def _load_model(self):
        suffix = self.model_path.suffix.lower()

        if self.model_type not in {"auto", "classifier", "booster", "pickle"}:
            raise ValueError(
                "model_type must be one of: auto, classifier, booster, pickle"
            )

        if self.model_type == "pickle" or (
            self.model_type == "auto" and suffix in {".pkl", ".pickle", ".joblib"}
        ):
            if suffix == ".joblib" and joblib is not None:
                return joblib.load(self.model_path)
            with self.model_path.open("rb") as f:
                return pickle.load(f)

        if self.model_type in {"auto", "classifier"}:
            try:
                model = xgb.XGBClassifier()
                model.load_model(self.model_path)
                self.model_type = "classifier"
                return model
            except Exception:
                if self.model_type == "classifier":
                    raise

        booster = xgb.Booster()
        booster.load_model(self.model_path)
        self.model_type = "booster"
        return booster

    def predict_scores(self, feature_rows: List[Dict[str, float]]) -> np.ndarray:
        if not feature_rows:
            return np.array([], dtype=float)

        frame = pd.DataFrame(feature_rows, columns=self.feature_columns)
        missing = [col for col in self.feature_columns if col not in frame.columns]
        if missing:
            raise ValueError(f"Missing model features: {missing}")

        if self.model_type == "booster":
            dmatrix = xgb.DMatrix(
                frame,
                feature_names=self.feature_columns,
                enable_categorical=False,
            )
            scores = np.asarray(self.model.predict(dmatrix), dtype=float).reshape(-1)
            return scores

        if hasattr(self.model, "predict_proba"):
            proba = np.asarray(self.model.predict_proba(frame))
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1].astype(float)
            return proba.reshape(-1).astype(float)

        scores = np.asarray(self.model.predict(frame), dtype=float).reshape(-1)
        return scores
