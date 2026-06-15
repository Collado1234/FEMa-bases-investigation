from algebra.basis.fem_basis import Basis
import numpy as np


class FEMaRegressor:
    """
    Class responsible to perform the regression using FEMa approach.
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
        self.k = k
        self.z = z
        self.basis = basis

    def fit(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """
        Armazena os dados de treinamento.

        Args:
            train_x: Features de treino (n_samples, n_features)
            train_y: Targets de treino (n_samples,)
        """
        self.train_x = train_x
        self.train_y = train_y
        self.num_train_samples = len(train_x)
        self.num_features = train_x.shape[1]

    def predict(self, test_x: np.ndarray) -> np.ndarray:
        """
        Interpola o valor para cada amostra de teste.

        Args:
            test_x: Features de teste (n_samples, n_features)

        Returns:
            Vetor de previsões (n_samples,)
        """
        predicted = [
            self.basis(
                X_train=self.train_x,
                X_sample=test_x[i],
                y_train=self.train_y,
                k_neighboors=self.k,
                z=self.z
            )
            for i in range(len(test_x))
        ]

        return np.array(predicted)