import numpy as np
from .basi_model import BasisModel
from distances.base_distance import BaseDistance
from neighboor_search.base_search import BaseSearch
from distances.euclidean_distance import EuclideanDistance
from neighboor_search.brute_force import BruteForceSearch
import math

class SheppardBasis(BasisModel):
    """
    Implementa a base de Sheppard para interpolação de dados.

    A base de Sheppard é uma técnica de interpolação que utiliza pesos inversamente proporcionais à distância
    entre os pontos de dados e o ponto de consulta. É especialmente útil para interpolação em espaços multidimensionais.

    Herda da classe BasisModel e fornece a implementação concreta da base de Sheppard.
    """
    def __init__(self,
                distance: BaseDistance=EuclideanDistance(),
                search: BaseSearch =BruteForceSearch()):
        super().__init__(distance, search)
    
    def predict(self, X:np.array, k:int, z:int) -> np.array:
        """
        Faz previsões usando a base de Sheppard para os dados de entrada.

        Args:
            X (np.ndarray): Matriz de características dos dados para os quais as previsões serão feitas.
            k (int): O número de vizinhos mais próximos a serem considerados na interpolação.
            z (int): O parâmetro de suavização para evitar singularidades.

        Returns:
            np.ndarray: As previsões do modelo para os dados de entrada.

        Notes:
            Este método calcula os pesos com base nas distâncias entre os pontos de consulta e os pontos de treinamento,
            e retorna as previsões interpoladas usando a base de Sheppard.
        """
        dists = self.distance.compute(X, self.X_train)
        neighbors_idx = self.search.search(dists,k)

        dists = dists[neighbors_idx]
        train_y_k_nearest_neighbors = self.y_train[neighbors_idx]

        dists = np.where(dists == 0, 1e-10, dists)  # Evita divisão por zero
        weights = 1 / (dists**z)
        weights = weights / np.sum(weights)  # Partição de unidade

        predicted = np.sum(weights * train_y_k_nearest_neighbors, axis=0)

        if math.isnan(predicted):
            predicted = np.mean(self.y_train)
        
        return predicted