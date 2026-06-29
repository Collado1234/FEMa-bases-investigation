from abc import ABC, abstractmethod
import numpy as np

class BaseDistance(ABC):
    """
    Interface abstrata para métricas de distância.
    """
    @abstractmethod
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Calcula a distância entre x1 e x2. 
        Suporta computação vetorizada: x1 (1, D) e x2 (N, D) -> dists (N,)
        """
        pass
