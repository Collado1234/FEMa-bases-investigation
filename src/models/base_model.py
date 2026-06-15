from abc import ABC, abstractmethod
import numpy as np
import algebra.neighboor_search.base_search as base_search
import algebra.neighboor_search.brute_force as brute_force
import algebra.basis.basis as basis
import algebra.basis.shepard as shepard
import algebra.distances.base_distance as base_distance
import algebra.distances.euclidean_distance as euclidean_distance

class FEMaBaseModel(ABC):
    def __init__(self,
                 distance: base_distance.BaseDistance=euclidean_distance.EuclideanDistance(), search: base_search.BaseSearch=brute_force.BruteForce(),
                 basis: basis.BaseModel = shepard.SheppardBasis()):
        self.distance = distance
        self.search = search
        self.basis = basis

    @abstractmethod
    def fit(self, X : np.ndarray, y : np.ndarray) -> None: 
        """
        Ajusta o modelo aos dados de treinamento.

        Args:
            X: A matriz de características dos dados de treinamento.
            y: O vetor de rótulos ou valores-alvo dos dados de treinamento.

        Returns:
            None

        Notes:
            Este método deve ser implementado por subclasses concretas, como classificadores ou regressões.
        """
        pass

    @abstractmethod
    def predict(self, X:np.ndarray) -> np.ndarray:
        """
        Faz previsões usando o modelo treinado.

        Args:
            X: O conjunto de dados para o qual as previsões serão feitas (features).

        Returns:
            As previsões do modelo para os dados de entrada.

        Notes:
            Este método deve ser implementado por subclasses concretas, como classificadores ou regressões.
        """
        pass