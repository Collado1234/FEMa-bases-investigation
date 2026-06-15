from abc import ABC, abstractmethod
import numpy as np
from algebra.neighboor_search.base_search import BaseSearch


class BaseBasis(ABC):
    """
    Classe base abstrata para bases de interpolação.

    Basis é puramente matemática — não guarda dados de treino.
    Recebe distâncias prontas e devolve pesos normalizados.
    """

    def __init__(self, search: BaseSearch):
        self.search = search

    @abstractmethod
    def compute_weights(self, dists: np.ndarray, z: float) -> np.ndarray:
        """
        Calcula os pesos a partir das distâncias dos k vizinhos.

        Args:
            dists: Distâncias dos k vizinhos (k,)
            z:     Parâmetro da base

        Returns:
            weights: Pesos normalizados (k,)
        """
        pass