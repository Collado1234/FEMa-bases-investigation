from typing import Tuple
import numpy as np
from ..math.basis.base_basis import BaseBasis
from ..math.neighboor_search.base_search import BaseSearch
from .base_model import FEMaBaseModel

class FEMaClassifier(FEMaBaseModel):
    """
    Classificador FEMa por interpolação de probabilidades por classe.

    Para cada amostra de teste:
        1. Search retorna índices e distâncias dos k vizinhos (uma vez)
        2. Basis calcula os pesos a partir das distâncias
        3. Model interpola a probabilidade de cada classe com os mesmos pesos

    Uso:
        model = FEMaClassifier(basis=Basis.get('shepard'))
        model.fit(X_train, y_train)
        labels, probs = model.predict(X_test, k=5, z=2)
    """

    def __init__(self, basis:BaseBasis, search: BaseSearch):
        super().__init__(basis, search)
        self.num_classes = None
        self.probability_classes = None #shape (num_classes, n_train_samples)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Indexa X no search e monta a matriz de probabilidades por classe.

        Args:
            X: Features de treino (n_samples, n_features)
            y: Labels de treino (n_samples,)
        """

        # Search indexa X -> Constrói estrutura de dados espacial
        # self.basis.search.build(X)
        self.search.build(X)

        self.num_classes = len(np.unique(y))

        #uma linha por classe: 1.0 quando y == c, 0.0 caso contrário
        #shape: (num_classes, n_train_samples)
        self.probability_classes = np.array(
            [(y == c).astype(float) for c in range(self.num_classes)]
        )

    def predict(self, X: np.ndarray, k: int, z: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Para cada amostra: busca vizinhos uma vez, interpola para cada classe.

        Args:
            X: Features de teste (n_samples, n_features)
            k: Número de vizinhos (0 = todos)
            z: Parâmetro da base de interpolação

        Returns:
            labels: Classes preditas (n_samples,)
            probs:  Probabilidades por classe (n_samples, num_classes)
        """
        if self.probability_classes is None:
            raise RuntimeError("Chame fit() antes de predict()")
        
        num_test_samples = X.shape[0]
        probs = np.zeros((num_test_samples, self.num_classes))

        for i in range(num_test_samples):
            indices, dists = self.search.query(X[i], k)

            weights = self.basis.compute_weights(dists,z)

            #interpola a probabilidade de cada classe com os mesmos pesos
            for c in range(self.num_classes):
                probs[i, c] = np.dot(weights, self.probability_classes[c][indices])

        labels = np.argmax(probs, axis=1)   

        return labels, probs 
        