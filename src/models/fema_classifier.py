from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import algebra.neighboor_search.base_search as base_search
import algebra.neighboor_search.brute_force as brute_force
from algebra.basis.basis import BaseModel
from algebra.basis.shepard import SheppardBasis
from algebra.distances.base_distance import BaseDistance
from algebra.distances.euclidean_distance import EuclideanDistance
from models.base_model import FEMaBaseModel 
class FEMaClassifier(FEMaBaseModel):
    def __init__(self,
                distance: base_distance.BaseDistance= EuclideanDistance(),
                search: base_search.BaseSearch=brute_force.BruteForceSearch(),
                basis: basi_model.BaseModel = sheppard.SheppardBasis()):
        super().__init__(distance, search, basis)

    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        self.X_train = X
        self.y_train = y
        self.basis.fit(X, y)
        self.num_train_samples = len(y)
        self.num_features = X.shape[1]
        self.num_classes = len(np.unique(y[:, 0]))
        self.probabilities_classes = np.zeros((self.num_train_samples, self.num_classes))

        for i in range(self.num_classes):
            self.probabilities_classes[i, :] = (self.y_train[:, 0] == i).astype(float)
    
    
    def predict(self, test_x: np.array, *args) -> Tuple[np.array, np.array]:
        """
        Faz previsões usando o modelo treinado.
        Args:
            test_x: O conjunto de dados para o qual as previsões serão feitas (features).
        Returns:
            Uma tupla contendo:
                - As previsões do modelo para os dados de entrada (rótulos previstos).
                - As probabilidades associadas a cada classe para os dados de entrada.
        """
        num_test_samples = test_x.shape[0]
        labels = np.zeros(num_test_samples)
        predicted_probabilities = np.zeros((num_test_samples, self.num_classes))

        for i in range(num_test_samples):
            predicted_probabilities[i,:] = [self.basis.predict(train_x=self.train_x,
                                                            train_y=self.probabilities_classes[c],
                                                            test_one_sample=test_x[i], k=self.k, z=args[0]) 
                                                            for c in range(self.num_classes)]
            labels[i] = np.argmax(predicted_probabilities[i, :])

        return labels, predicted_probabilities

