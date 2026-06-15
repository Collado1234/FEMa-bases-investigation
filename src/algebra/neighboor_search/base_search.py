from abc import ABC, abstractmethod
import numpy as np

class BaseSearch(ABC):
    """
    Interface abstrata para métodos de busca de vizinhos.
    """
    @abstractmethod
    def search(self, dists: np.ndarray, k: int) -> np.ndarray:
        """
        Recebe um vetor de distâncias e retorna os índices dos k vizinhos mais próximos.
        """
        pass
