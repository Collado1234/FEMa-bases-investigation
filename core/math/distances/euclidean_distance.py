import numpy as np
from .base_distance import BaseDistance


class EuclideanDistance(BaseDistance):
    """
    Distância euclidiana.

    Casos suportados:
        - (D,) x (D,)       -> escalar
        - (N, D) x (D,)     -> vetor (N,)
        - (D,) x (N, D)     -> vetor (N,)
    """

    def compute(self, x1: np.ndarray, x2: np.ndarray):
        diff = x1 - x2

        # Se o resultado é uma matriz (N, D),
        # calcula a norma de cada linha.
        if diff.ndim == 2:
            return np.linalg.norm(diff, axis=1)

        # Caso contrário, calcula uma única distância.
        return np.linalg.norm(diff)