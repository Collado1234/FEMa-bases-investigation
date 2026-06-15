import numpy as np
from .base_model import FEMaBaseModel

class FEMaClassifier(FEMaBaseModel):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Garante que y seja 2D para processamento uniforme
        if y.ndim == 1:
            y = y[:, np.newaxis]
            
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # One-hot encoding para as probabilidades das classes
        y_one_hot = (y == self.classes_).astype(float)
        
        # Treina a base com os dados one-hot
        super().fit(X, y_one_hot)

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Retorna as classes preditas.
        """
        probs = self.predict_proba(X, **kwargs)
        return self.classes_[np.argmax(probs, axis=1)]

    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Retorna as probabilidades para cada classe.
        """
        return self.basis.predict(X, **kwargs)
