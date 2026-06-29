from abc import ABC, abstractmethod
import numpy as np
from ..algebra.basis.base_basis import BaseBasis
from ..algebra.neighboor_search import BaseSearch
from ..algebra.neighboor_search.brute_force import BruteForceSearch

class FEMaBaseModel(ABC):
    """
    Classe base abstrata para os modelos FEMa.

    O model é quem:
        - Guarda X_train e y_train
        - Manda o search indexar X no fit
        - Orquestra search + basis no predict
        - Faz o produto interno final

    Uso:
        model = FEMaClassifier(basis=Basis.get('shepard'))
        model.fit(X_train, y_train)
        result = model.predict(X_test, k=5, z=2)
    """
    def __init__(self, basis: BaseBasis, search: BaseSearch = BruteForceSearch):
        self.basis = basis
        self.search = search
        self.X_train = None
        self.y_train = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Ajusta o modelo aos dados de treinamento.

        Args:
            X: Features de treino (n_samples, n_features)
            y: Targets de treino (n_samples,)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, k: int, z: float) -> np.ndarray:
        """
        Faz previsões para o conjunto de teste.

        Args:
            X: Features de teste (n_samples, n_features)
            k: Número de vizinhos (0 = todos)
            z: Parâmetro da base de interpolação

        Returns:
            Vetor de previsões (n_samples,)
        """
        pass

