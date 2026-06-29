from algebra.basis.fem_basis import Basis
from typing import Tuple
import numpy as np


class FEMaClassifier:
    """
    Class responsible to perform the classification using FEMa approach.
    """

    def __init__(self, k: int = 2, z: int = 2, basis=Basis.shepardBasis) -> None:
        """
        Args:
            k:     Número de vizinhos
            z:     Parâmetro da base de interpolação
            basis: Função de base (Basis.shepardBasis ou Basis.radialBasis)
        """
        self.train_x = None
        self.train_y = None
        self.num_train_samples = 0
        self.num_features = 0
        self.num_classes = 0
        self.k = k
        self.z = z
        self.basis = basis
        self.probability_classes = None

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """
        Armazena os dados e monta a matriz de probabilidades por classe.

        Args:
            train_x: Features de treino (n_samples, n_features)
            train_y: Labels de treino (n_samples,)
        """
        self.train_x = train_x
        self.train_y = train_y
        self.num_train_samples = len(train_y)
        self.num_features = train_x.shape[1]
        self.num_classes = len(np.unique(train_y))

        # shape: (num_classes, n_train_samples)
        self.probability_classes = np.array(
            [(train_y == c).astype(float) for c in range(self.num_classes)]
        )

    def predict(self, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpola a probabilidade de cada classe para cada amostra de teste.

        Args:
            test_x: Features de teste (n_samples, n_features)

        Returns:
            labels:           Classes preditas (n_samples,)
            confidence_level: Probabilidades por classe (n_samples, num_classes)
        """
        num_test_samples = len(test_x)
        labels = np.zeros(num_test_samples)
        confidence_level = np.zeros((num_test_samples, self.num_classes))

        for i in range(num_test_samples):
            for c in range(self.num_classes):
                confidence_level[i, c] = self.basis(
                    X_train=self.train_x,
                    X_sample=test_x[i],
                    y_train=self.probability_classes[c],
                    k_neighboors=self.k,
                    z=self.z
                )
            labels[i] = np.argmax(confidence_level[i, :])

        return labels, confidence_level