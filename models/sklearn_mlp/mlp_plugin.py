"""
Plugin de referência para modelos de rede neural. Implementado com
sklearn.neural_network.MLPClassifier como stand-in, já que este ambiente
não tem torch/tensorflow disponíveis — mas a INTERFACE é idêntica à que
uma CNN real em PyTorch teria: create_model(hyperparameters) monta a
arquitetura, fit()/predict() escondem o framework por trás.

Para trocar por uma CNN real em PyTorch, troque apenas o conteúdo de
create_model/fit/predict/save/load dentro desta classe — nada fora de
models/cnn/ precisa mudar.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.neural_network import MLPClassifier

from models.base import BaseModel


class MLPPlugin(BaseModel):
    name = "cnn"

    def __init__(self, hidden_layer_sizes, learning_rate_init, alpha, max_iter, seed=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.alpha = alpha
        self.max_iter = max_iter
        self.seed = seed
        self._model: Optional[MLPClassifier] = None

    @classmethod
    def create_model(cls, hyperparameters: Dict[str, Any]) -> "MLPPlugin":
        return cls(
            hidden_layer_sizes=tuple(hyperparameters.get("hidden_layer_sizes", (64,))),
            learning_rate_init=float(hyperparameters.get("learning_rate", 1e-3)),
            alpha=float(hyperparameters.get("dropout", 1e-4)),
            max_iter=int(hyperparameters.get("epochs", 200)),
            seed=int(hyperparameters.get("seed", 42)),
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> None:
        self._model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            learning_rate_init=self.learning_rate_init,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.seed,
        )
        self._model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)
        return None

    def save(self, path: str) -> None:
        import joblib

        joblib.dump(self._model, path)

    @classmethod
    def load(cls, path: str) -> "MLPPlugin":
        import joblib

        instance = cls(hidden_layer_sizes=(64,), learning_rate_init=1e-3, alpha=1e-4, max_iter=200)
        instance._model = joblib.load(path)
        return instance

    @staticmethod
    def get_search_space() -> Dict[str, Any]:
        return {
            "learning_rate": {"type": "loguniform", "low": 1e-4, "high": 1e-1},
            "hidden_layer_sizes": {"type": "choice", "values": [(32,), (64,), (64, 32)]},
            "dropout": {"type": "loguniform", "low": 1e-6, "high": 1e-2},
            "epochs": {"type": "randint", "low": 50, "high": 300},
        }
