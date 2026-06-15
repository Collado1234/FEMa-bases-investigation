from abc import ABC, abstractmethod
import numpy as np
from ..algebra.basis.basi_model import BasisModel

class FEMaBaseModel(ABC):
    """
    Classe base para modelos FEMA.
    Encapsula uma base de interpolação.
    """
    def __init__(self, basis: BasisModel):
        self.basis = basis

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Delega o treinamento para a base.
        """
        self.basis.fit(X, y)

    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Interface de predição.
        """
        pass
