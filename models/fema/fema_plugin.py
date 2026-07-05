"""
Adapter (Adapter pattern) entre a interface pública BaseModel e o
core.models.fema_classifier.FEMaClassifier já existente no projeto.

Por que um Adapter e não usar o FEMaClassifier direto no pipeline:
o predict() real do FEMa exige (X, k, z) — dois hiperparâmetros de
inferência que o resto do pipeline não deveria conhecer. Aqui, k e z
são recebidos em create_model() como hiperparâmetros normais e guardados
como atributos de instância; o predict(X) público apenas os repassa
internamente. O pipeline nunca sabe que k/z existem.

IMPORTANTE: este adapter depende de core/, que no estado atual do
repositório tem bugs de import (ver README.md do projeto, seção
"Bugs conhecidos em core/"). Os métodos aqui levantam um erro claro e
acionável caso o import falhe, em vez de um traceback opaco.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from models.base import BaseModel


class FEMaCoreUnavailable(RuntimeError):
    """Levantado quando core/ não pôde ser importado por bugs pré-existentes."""


_KNOWN_CORE_ISSUES = """\
Não foi possível usar core.models.fema_classifier. Isso é causado por
problemas pré-existentes em core/ (listados em README.md > "Bugs conhecidos
em core/"; nenhum deles foi corrigido por este framework):
  1) core/__init__.py importa 'core.algebra' (pasta inexistente; a real é
     'core/math').
  2) core/models/__init__.py importa '.CLASSIFIER' e '.REGRESSOR' (arquivos
     inexistentes; os reais são 'fema_classifier.py' e 'fema_regressor.py').
  3) core/models/fema_classifier.py, fema_regressor.py e base_model.py
     importam de 'core.algebra...' (deveria ser 'core.math...') e
     fema_classifier.py importa 'from models.base_model import
     FEMaBaseModel' (pacote absoluto inexistente; deveria ser
     'from .base_model import FEMaBaseModel').
  4) core/math/basis/__init__.py importa '.basi_model' (arquivo inexistente).
  5) core/math/basis/basis.py: o método estático Basis.get() tem apenas o
     docstring, sem corpo/return — a fábrica de bases não está implementada.
     Este adapter contorna isso instanciando SheppardBasis/RadialBasis
     diretamente (ver _build_core_model abaixo).
  6) core/math/distances/euclidean_distance.py: a condição em compute()
     que decide broadcasting (`x1.ndim == 1 and x2.ndim == 2`) está
     invertida para o caso real de uso (X_train 2D vs. amostra 1D),
     fazendo compute() devolver um único escalar em vez de um vetor de
     distâncias por vizinho — isso quebra BruteForceSearch.query() mesmo
     depois de corrigidos os itens 1-4.
Corrija os itens 1-4 e 6 dentro de core/ para habilitar o plugin FEMa
por completo. Erro original: {exc_type}: {exc}
"""


def _import_fema_core():
    try:
        from core.math.basis.sheppard import SheppardBasis
        from core.math.basis.radial import RadialBasis
        from core.math.neighboor_search.brute_force import BruteForceSearch
        from core.math.distances.euclidean_distance import EuclideanDistance
        from core.models.fema_classifier import FEMaClassifier

        basis_classes = {"shepard": SheppardBasis, "radial": RadialBasis}
        return basis_classes, BruteForceSearch, EuclideanDistance, FEMaClassifier
    except Exception as exc:  # noqa: BLE001
        raise FEMaCoreUnavailable(
            _KNOWN_CORE_ISSUES.format(exc_type=type(exc).__name__, exc=exc)
        ) from exc


class FEMaPlugin(BaseModel):
    """Wrapper de core.models.fema_classifier.FEMaClassifier."""

    name = "fema"

    def __init__(self, basis_function: str, k: int, z: float, distance: str = "euclidean"):
        self.basis_function = basis_function
        self.k = k
        self.z = z
        self.distance = distance
        self._model = None  # instanciado sob demanda em fit(), após checar core

    @classmethod
    def create_model(cls, hyperparameters: Dict[str, Any]) -> "FEMaPlugin":
        return cls(
            basis_function=hyperparameters.get("basis_function", "shepard"),
            k=int(hyperparameters.get("k", 5)),
            z=float(hyperparameters.get("z", 2.0)),
            distance=hyperparameters.get("distance", "euclidean"),
        )

    def _build_core_model(self):
        basis_classes, BruteForceSearch, EuclideanDistance, FEMaClassifier = _import_fema_core()
        if self.basis_function not in basis_classes:
            raise ValueError(
                f"basis_function '{self.basis_function}' desconhecida. "
                f"Disponíveis: {sorted(basis_classes)}"
            )
        search = BruteForceSearch(metric=EuclideanDistance())
        basis = basis_classes[self.basis_function](search=search)
        return FEMaClassifier(basis=basis, search=search)

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> None:
        self._model = self._build_core_model()
        self._model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Chame fit() antes de predict().")
        labels, _probs = self._model.predict(X, k=self.k, z=self.z)
        return labels

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self._model is None:
            raise RuntimeError("Chame fit() antes de predict_proba().")
        _labels, probs = self._model.predict(X, k=self.k, z=self.z)
        return probs

    def save(self, path: str) -> None:
        import pickle

        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "basis_function": self.basis_function,
                    "k": self.k,
                    "z": self.z,
                    "distance": self.distance,
                    "model": self._model,
                },
                fh,
            )

    @classmethod
    def load(cls, path: str) -> "FEMaPlugin":
        import pickle

        with open(path, "rb") as fh:
            state = pickle.load(fh)
        instance = cls(
            basis_function=state["basis_function"],
            k=state["k"],
            z=state["z"],
            distance=state["distance"],
        )
        instance._model = state["model"]
        return instance

    @staticmethod
    def get_search_space() -> Dict[str, Any]:
        return {
            "basis_function": {"type": "choice", "values": ["shepard", "radial"]},
            "k": {"type": "randint", "low": 3, "high": 30},
            "z": {"type": "uniform", "low": 0.5, "high": 4.0},
        }
