import numpy as np
from typing import Tuple
from .base_search import BaseSearch

class BruteForceSearch(BaseSearch):
    def __init__(self, metric):
        super().__init__(metric)
        self.X_train = None

    def build(self, X:np.ndarray) -> None:
        """
        Indexa os dados de treino na estrutura espacial.

        Args:
            X: Features de treino (n_samples, n_features)
        """
        self.X_train = X
    
    def query(self, sample: np.ndarray, k: int, ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula distância para todos os pontos e retorna os k menores (distância definida na criação da classe).

        Args:
            sample: Amostra de teste (n_features,)
            k:      Número de vizinhos (0 = todos)

        Returns:
            indices: Índices dos k vizinhos (k,)
            dists:   Distâncias euclidianas dos k vizinhos (k,)
        """
        if self.X_train is None:
            raise RuntimeError("Chame .build() antes de query ()")
        
        distances = self.metric.compute(self.X_train, sample)

        if k == 0 or k >= len(self.X_train): # todos os vizinhos se k=0 ou k >= n_samples
            indices = np.arange(len(self.X_train))
        else:
            # print("X_train.shape:", self.X_train.shape)
            # print("sample.shape:", sample.shape)
            # print("distances.shape:", distances.shape)
            # print("k:", k)
            indices = np.argpartition(distances, k)[:k] 
        
        return indices, distances[indices]
