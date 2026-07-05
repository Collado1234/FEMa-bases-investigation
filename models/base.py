"""
Contrato único que TODO modelo (FEMa, CNN, ResNet, EfficientNet, ...) precisa
implementar. O restante do pipeline (training, tuning, evaluation) só
conhece esta interface — nunca a implementação concreta de um modelo.

Métodos extras específicos de um modelo (ex.: FEMa poderia expor
`explain_neighbors()`) devem ficar encapsulados na subclasse e NUNCA ser
chamados pelo pipeline genérico.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseModel(ABC):
    """Interface pública que todo plugin de modelo deve implementar."""

    #: nome curto usado no registry e nas configs YAML (ex.: "fema", "cnn")
    name: str = "base"

    @classmethod
    @abstractmethod
    def create_model(cls, hyperparameters: Dict[str, Any]) -> "BaseModel":
        """
        Factory: constrói uma instância CONFIGURADA (mas ainda não treinada)
        a partir de um dicionário de hiperparâmetros vindo do search space.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Treina o modelo. X_val/y_val são opcionais (ex.: early stopping)."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retorna rótulos preditos, shape (n_samples,)."""
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Retorna probabilidades por classe, shape (n_samples, n_classes).
        Modelos que não suportam probabilidade devem retornar None
        (o restante do pipeline trata isso, ex.: métricas que dependem de
        AUC simplesmente não são calculadas para esse modelo).
        """
        return None

    @abstractmethod
    def save(self, path: str) -> None:
        """Persiste o modelo treinado em disco."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseModel":
        """Carrega um modelo previamente salvo com save()."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_search_space() -> Dict[str, Any]:
        """
        Espaço de hiperparâmetros declarativo do modelo. Formato:
            {"param_name": {"type": "choice", "values": [...]}}
            {"param_name": {"type": "uniform", "low": ..., "high": ...}}
            {"param_name": {"type": "loguniform", "low": ..., "high": ...}}
            {"param_name": {"type": "randint", "low": ..., "high": ...}}
        O pipeline (tuning/) só entende esse formato — nenhuma lógica
        específica de modelo deve existir fora do próprio modelo.
        """
        raise NotImplementedError
