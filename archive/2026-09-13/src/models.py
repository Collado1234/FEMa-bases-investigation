"""Modelos disponíveis no framework: 'fema' (foco do projeto) e dois
baselines simples ('logreg', 'mlp'). Trocar de modelo é só usar outro nome
na config YAML — não existe plugin/registry, é um dict direto (MODELS).
Para adicionar um modelo novo: escrever a classe (fit/predict/predict_proba/
save/load) e acrescentar uma linha em MODELS e outra em SEARCH_SPACES.
"""
from __future__ import annotations

import pickle
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from core import EuclideanDistance, BruteForceSearch, ShepardBasis, RadialBasis, FEMaClassifier

_BASIS = {"shepard": ShepardBasis, "radial": RadialBasis}


class FEMaModel:
    """Wrapper fino sobre core.FEMaClassifier. k/z são hiperparâmetros de
    inferência do FEMa; aqui viram hiperparâmetros normais do modelo."""

    def __init__(self, basis_function: str = "shepard", k: int = 5, z: float = 2.0):
        self.basis_function = basis_function
        self.k = k
        self.z = z
        self._model: Optional[FEMaClassifier] = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> None:
        search = BruteForceSearch(metric=EuclideanDistance())
        basis = _BASIS[self.basis_function](search=search)
        self._model = FEMaClassifier(basis=basis, search=search)
        self._model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        labels, _ = self._model.predict(X, k=self.k, z=self.z)
        return labels

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, probs = self._model.predict(X, k=self.k, z=self.z)
        return probs

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str) -> "FEMaModel":
        with open(path, "rb") as fh:
            return pickle.load(fh)


class LogRegModel:
    """Baseline simples e rápido para validar o pipeline."""

    def __init__(self, C: float = 1.0, penalty: str = "l2", max_iter: int = 500, seed: int = 42):
        self.C, self.penalty, self.max_iter, self.seed = C, penalty, max_iter, seed
        self._model: Optional[LogisticRegression] = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> None:
        self._model = LogisticRegression(
            C=self.C, penalty=self.penalty, max_iter=self.max_iter, random_state=self.seed,
            solver="lbfgs" if self.penalty == "l2" else "saga")
        self._model.fit(X_train, y_train)

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def save(self, path: str) -> None:
        joblib.dump(self._model, path)

    @staticmethod
    def load(path: str) -> "LogRegModel":
        instance = LogRegModel()
        instance._model = joblib.load(path)
        return instance


class MLPModel:
    """Baseline de rede neural (stand-in para uma CNN real, mesma interface)."""

    def __init__(self, hidden_layer_sizes=(64,), learning_rate=1e-3, dropout=1e-4, epochs=200, seed=42):
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.epochs = epochs
        self.seed = seed
        self._model: Optional[MLPClassifier] = None

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> None:
        self._model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes, learning_rate_init=self.learning_rate,
            alpha=self.dropout, max_iter=self.epochs, random_state=self.seed)
        self._model.fit(X_train, y_train)

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X) if hasattr(self._model, "predict_proba") else None

    def save(self, path: str) -> None:
        joblib.dump(self._model, path)

    @staticmethod
    def load(path: str) -> "MLPModel":
        instance = MLPModel()
        instance._model = joblib.load(path)
        return instance


MODELS = {"fema": FEMaModel, "logreg": LogRegModel, "mlp": MLPModel}

# espaço de busca declarativo por modelo:
#   escolha discreta -> lista de valores
#   distribuição contínua -> ("uniform"|"loguniform", low, high) ou ("randint", low, high)
SEARCH_SPACES: Dict[str, Dict[str, Any]] = {
    "fema": {
        "basis_function": ["shepard", "radial"],
        "k": ("randint", 3, 30),
        "z": ("uniform", 0.5, 4.0),
    },
    "logreg": {
        "C": ("loguniform", 1e-3, 1e2),
        "penalty": ["l2"],
        "max_iter": [500],
    },
    "mlp": {
        "learning_rate": ("loguniform", 1e-4, 1e-1),
        "hidden_layer_sizes": [(32,), (64,), (64, 32)],
        "dropout": ("loguniform", 1e-6, 1e-2),
        "epochs": ("randint", 50, 300),
    },
}


def create_model(name: str, hyperparameters: Dict[str, Any]):
    if name not in MODELS:
        raise KeyError(f"Modelo '{name}' desconhecido. Disponíveis: {sorted(MODELS)}")
    return MODELS[name](**hyperparameters)


def get_search_space(name: str) -> Dict[str, Any]:
    if name not in SEARCH_SPACES:
        raise KeyError(f"Modelo '{name}' desconhecido. Disponíveis: {sorted(SEARCH_SPACES)}")
    return SEARCH_SPACES[name]
