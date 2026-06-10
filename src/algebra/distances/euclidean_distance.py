# ...existing code...
import numpy as np
from .base_distance import BaseDistance


class EuclideanDistance(BaseDistance):
    """
    Implementa a distância euclidiana entre dois pontos representados por vetores NumPy.

    A distância euclidiana mede o comprimento da linha reta entre dois pontos
    no espaço n-dimensional e é amplamente usada em algoritmos de ML, clustering
    e análise de similaridade.

    Herda da classe BaseDistance e fornece a implementação concreta da métrica.
    """

    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calcula a distância euclidiana entre dois vetores.

        Args:
            x1 (np.ndarray): Primeiro vetor de entrada.
            x2 (np.ndarray): Segundo vetor de entrada.

        Returns:
            float: O valor da distância euclidiana entre x1 e x2.

        Example:
            >>> EuclideanDistance().compute(np.array([0, 0]), np.array([3, 4]))
            5.0
        """
        return np.linalg.norm(x1 - x2)
# ...existing code...