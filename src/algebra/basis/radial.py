from abc import ABC, abstractmethod

from algebra.distances.base_distance import BaseDistance
from algebra.distances.euclidean_distance import EuclideanDistance
from algebra.neighboor_search.base_search import BaseSearch
from algebra.neighboor_search.brute_force import BruteForceSearch
from .basi_model import BasisModel
import numpy as np

class RadialBasis(BasisModel):
    """
    Base radial com função-mãe sino gaussiano:
        Ψ(r) = exp(-0.5 * (r / r0)²)

    Onde:
        r  = distância entre x e xi
        r0 = raio efetivo (parâmetro z)
    """
    def __init__(self,
                distance: BaseDistance=EuclideanDistance(),
                search: BaseSearch =BruteForceSearch()):
        super().__init__(distance, search)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X_train = X
        self.y_train = y

    def _mother_function(self, r: np.ndarray, r0: float) -> np.ndarray:
        """
        Função-mãe sino gaussiano.
        Considerada nula para r >= 4*r0 na prática.

        Args:
            r:  vetor de distâncias
            r0: raio efetivo (> 0)
        Returns:
            vetor com Ψ(r) para cada distância
        """
        return np.exp(-0.5 * (r / r0) ** 2)

    def predict(self, X: np.ndarray, r0: float = 1.0) -> np.ndarray:
        dists = self.distance.compute(X, self.X_train)
        neighbors_idx = self.search.search(dists, len(self.X_train))

        dists = dists[neighbors_idx]
        train_y_k_nearest_neighbors = self.y_train[neighbors_idx]

        weights = self._mother_function(dists, r0=r0)
        weights = weights / np.sum(weights)  # Partição de unidade
        
        predicted = np.sum(weights * train_y_k_nearest_neighbors, axis=0)

        if np.isnan(predicted):
            predicted = np.mean(self.y_train)
        
        return predicted