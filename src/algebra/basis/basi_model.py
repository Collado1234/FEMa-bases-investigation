from abc import ABC, abstractmethod
from distances.base_distance import BaseDistance
from neighboor_search.base_search import BaseSearch
from neighboor_search.brute_force import BruteForceSearch
from distances.euclidean_distance import EuclideanDistance
import numpy as np

class BaseModel(ABC):
    """
    Classe base abstrata para modelos de aprendizado de máquina que utilizam uma métrica de distância e um método de busca de vizinhos.
    """
    def __init__(self, distance: BaseDistance=EuclideanDistance(), search: BaseSearch = BruteForceSearch()):
        self.distance = distance
        self.search = search

    def fit(self, X:np.array, y:np.array) -> None:
        """
        Ajusta o modelo aos dados de treinamento (Os armazena para uso posterior na interpolação).

        Args:
            X (np.ndarray): Matriz de características dos dados de treinamento.
            y (np.ndarray): Vetor de rótulos ou valores-alvo dos dados de treinamento.

        Returns:
            None

        Notes:
            Este método armazena os dados de treinamento para uso posterior na interpolação.
        """
        self.X_train = X
        self.y_train = y

    @abstractmethod
    def predict(self, X):
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