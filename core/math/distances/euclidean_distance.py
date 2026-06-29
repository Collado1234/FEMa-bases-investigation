import numpy as np
from .base_distance import BaseDistance

class EuclideanDistance(BaseDistance):
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Calcula distância euclidiana.
        Se x1 é (D,) e x2 é (N, D), retorna (N,).
        """
        # Garante que x1 tenha uma dimensão extra para broadcasting se necessário
        if x1.ndim == 1 and x2.ndim == 2:
            return np.linalg.norm(x1 - x2, axis=1)
        return np.linalg.norm(x1 - x2)
