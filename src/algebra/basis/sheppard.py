import numpy as np
from .basi_model import BasisModel

class SheppardBasis(BasisModel):
    def predict(self, X: np.ndarray, k: int = 0, z: int = 2) -> np.ndarray:
        """
        X: (M, D) matriz de pontos de teste.
        Retorna: (M, L) predições, onde L é a dimensão de y.
        """
        predictions = []
        for x_sample in X:
            dists = self.distance.compute(x_sample, self.X_train)
            indices = self.search.search(dists, k)
            
            # Seleciona vizinhos
            d_k = dists[indices]
            y_k = self.y_train[indices]
            
            # Evita divisão por zero
            d_k = np.where(d_k == 0, 1e-10, d_k)
            weights = 1.0 / (d_k ** z)
            weights /= np.sum(weights)
            
            # Média ponderada (funciona para vetores y também)
            pred = np.sum(weights[:, np.newaxis] * y_k, axis=0)
            predictions.append(pred)
            
        return np.array(predictions)
