import numpy as np
from .basi_model import BasisModel

class RadialBasis(BasisModel):
    def predict(self, X: np.ndarray, r0: float = 1.0) -> np.ndarray:
        predictions = []
        for x_sample in X:
            dists = self.distance.compute(x_sample, self.X_train)
            
            # Função Gaussiana simples
            weights = np.exp(-(dists / r0) ** 2)
            weights /= np.sum(weights)
            
            pred = np.sum(weights[:, np.newaxis] * self.y_train, axis=0)
            predictions.append(pred)
            
        return np.array(predictions)
