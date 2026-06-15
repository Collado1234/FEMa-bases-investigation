import numpy as np
from .base_model import FEMaBaseModel

class FEMaRegressor(FEMaBaseModel):
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Retorna os valores contínuos preditos.
        """
        return self.basis.predict(X, **kwargs)
