from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from ..distances import BaseDistance
class BaseSearch(ABC):
    """
    Classe base abstrata para métodos de busca de vizinhos.

    Responsabilidades:
        - Indexar X_train na estrutura adequada (build)
        - Calcular distâncias e retornar índices + distâncias (query)

    A métrica de distância é responsabilidade de cada implementação.
    """
    def __init__(self, metric: BaseDistance):
        """
        Inicializa a estrutura de busca com a métrica de distância.

        Args:
            metric (BaseDistance): Instância responsável por calcular a distância
            entre dois vetores. Deve implementar a interface BaseDistance.
        """
        self.metric = metric

    @abstractmethod
    def build(self, X:np.ndarray) -> None:
        """
        Indexa os dados de treino na estrutura espacial.

        Args:
            X: Features de treino (n_samples, n_features)
        """
        pass

    @abstractmethod
    def query(self, sample: np.ndarray, k: int, ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retorna os k vizinhos mais próximos de sample.

        Args:
            sample (np.ndarray): Amostra de teste (n_features)
            k (int): Número de vizinhos (0 = todos)

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
            Índices: Índices dos k vizinhos em X_train (k,)
            dists: Distâncias dos k vizinhos (k,)
        """
        pass