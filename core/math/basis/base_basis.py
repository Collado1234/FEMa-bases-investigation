from abc import ABC, abstractmethod
import numpy as np
from ..neighboor_search import BaseSearch

DELTA = 1e-10
class BaseBasis(ABC):
    """
    Classe base abstrata para bases de interpolação do FEMa.

    Basis é puramente matemática — não guarda dados de treino.
    Recebe distâncias prontas e devolve pesos normalizados.

    A busca de vizinhos e o cálculo de distâncias são delegados
    ao Search, que é injetado no construtor.
    """

    def __init__(self, search: BaseSearch):
        self.search = search
    
    @abstractmethod
    def compute_weights(self, dists: np.ndarray, z:float) -> np.ndarray:
        """
        Calcula os pesos a partir das distâncias dos k vizinhos.

        Args:
            dists: Distâncias dos k vizinhos (k,)
            z:     Parâmetro da base

        Returns:
            weights: Pesos normalizados (k,)
        """
        pass

    


