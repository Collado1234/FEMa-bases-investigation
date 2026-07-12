import numpy as np
from .base_model import FEMaBaseModel
from ..math.basis.base_basis import BaseBasis
from ..math.neighboor_search.base_search import BaseSearch

class FEMaRegressor(FEMaBaseModel):
    """
    Regressor FEMa por interpolação direta dos valores alvo.

    Para cada amostra de teste:
        1. Search retorna índices e distâncias dos k vizinhos
        2. Basis calcula os pesos a partir das distâncias
        3. Model faz o produto interno entre pesos e y_train[indices]

    Uso:
        model = FEMaRegressor(basis=Basis.get('radial'))
        model.fit(X_train, y_train)
        predictions = model.predict(X_test, k=5, z=2)
    """

    def __init__(self, basis: BaseBasis, search: BaseSearch):
        super().__init__(basis, search)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Indexa X no search e armazena os dados de treino.

        Args:
            X: Features de treino (n_samples, n_features)
            y: Targets de treino (n_samples,)
        """
        self.X_train = X
        self.y_train = y
        self.search.build(X)
    
    def predict(self, X:np.ndarray, k: int, z: float) -> np.ndarray:
        """
        Interpola o valor para cada amostra de teste

        Args:
            X: Features de teste (n_samples, n_features)
            k: Número de vizinhos (0 = todos)
            z: Parâmetros de base para interpolação

        Returns:
            Vetor de previsões (n_samples,)
        """
        if self.X_train is None:
            raise RuntimeError("Chame fit() antes de predict().")
        
        predictions = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            indices, dists = self.search.query(X[i],k)
            weights = self.basis.compute_weights(dists, z)
            predicted = np.dot(weights, self.y_train[indices])

            if np.isnan(predicted):
                predicted = float(np.mean(self.y_train))
            
            predictions[i] = predicted
        
        return predictions
