"""Segundo plugin de referência: baseline simples e rápido, útil para
validar o pipeline inteiro sem custo computacional relevante."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from models.base import BaseModel


class LogRegPlugin(BaseModel):
    name = "logreg_baseline"

    def __init__(self, C: float, penalty: str, max_iter: int, seed: int = 42):
        self.C = C
        self.penalty = penalty
        self.max_iter = max_iter
        self.seed = seed
        self._model: Optional[LogisticRegression] = None

    @classmethod
    def create_model(cls, hyperparameters: Dict[str, Any]) -> "LogRegPlugin":
        return cls(
            C=float(hyperparameters.get("C", 1.0)),
            penalty=hyperparameters.get("penalty", "l2"),
            max_iter=int(hyperparameters.get("max_iter", 500)),
            seed=int(hyperparameters.get("seed", 42)),
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> None:
        self._model = LogisticRegression(
            C=self.C, penalty=self.penalty, max_iter=self.max_iter,
            random_state=self.seed, solver="lbfgs" if self.penalty == "l2" else "saga",
        )
        self._model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        return self._model.predict_proba(X)

    def save(self, path: str) -> None:
        import joblib

        joblib.dump(self._model, path)

    @classmethod
    def load(cls, path: str) -> "LogRegPlugin":
        import joblib

        instance = cls(C=1.0, penalty="l2", max_iter=500)
        instance._model = joblib.load(path)
        return instance

    @staticmethod
    def get_search_space() -> Dict[str, Any]:
        return {
            "C": {"type": "loguniform", "low": 1e-3, "high": 1e2},
            "penalty": {"type": "choice", "values": ["l2"]},
            "max_iter": {"type": "choice", "values": [500]},
        }
