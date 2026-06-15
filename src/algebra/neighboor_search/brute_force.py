import numpy as np
from .base_search import BaseSearch

class BruteForceSearch(BaseSearch):
    def search(self, dists: np.ndarray, k: int) -> np.ndarray:
        if k < 0:
            raise ValueError("k deve ser >= 0")
        
        # Se k=0 ou k >= total, retorna todos ordenados
        if k == 0 or k >= len(dists):
            return np.argsort(dists)
            
        return np.argpartition(dists, k)[:k]
