from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseSearch(ABC):
    """
    Classe base abstrata para métodos de busca de vizinhos.

    Responsabilidades:
        - Indexar X_train na estrutura adequada (build)
        - Calcular distâncias e retornar índices + distâncias (query)

    A métrica de distância é responsabilidade de cada implementação.
    """

    @abstractmethod
    def build(self, X: np.ndarray) -> None:
        """
        Indexa os dados de treino na estrutura espacial.

        Args:
            X: Features de treino (n_samples, n_features)
        """
        pass

    @abstractmethod
    def query(self, sample: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retorna os k vizinhos mais próximos de sample.

        Args:
            sample: Amostra de teste (n_features,)
            k:      Número de vizinhos (0 = todos)

        Returns:
            indices: Índices dos k vizinhos em X_train (k,)
            dists:   Distâncias dos k vizinhos (k,)
        """
        pass