import sys
import os
from typing import Tuple
import numpy as np
import math 
from multiprocessing import Pool, cpu_count
from fem_basis import Basis  # Certifique-se de que este módulo está corretamente importado

class FEMaClassifier:
    """
    Class responsible to perform the classification using FEMa approach
    """
    def __init__(self, k: int = 2, basis=Basis.shepardBasis) -> None:
        """Constructor that receives the parameter to run the FEMa

        Args:
            k (int): Define the number of neighbors used to interpolate
            basis = The finite element basis
        """
        self.train_x = None
        self.train_y = None
        self.num_train_samples = 0
        self.num_features = 0
        self.num_classes = 0
        self.k = k
        self.basis = basis
        self.probability_classes = None

    def fit(self, train_x: np.array, train_y: np.array) -> None:
        """Method responsible to create the manifold learning probabilities

        Args:
            train_x (np.array): Features of the training set
            train_y (np.array): Class of the training set
        """
        print("Dentro do fit:")
        print(train_y.shape)
        print(train_y.ndim)

        self.train_x = train_x
        self.train_y = train_y
        self.num_train_samples = len(train_y)
        self.num_features = self.train_x.shape[1]
        self.num_classes = len(np.unique(train_y[:, 0]))

        self.probability_classes = np.zeros((self.num_classes, self.num_train_samples))

        for i in range(self.num_classes):
            self.probability_classes[i, :] = (train_y[:, 0] == i).astype(float)

    def predict(self, test_x: np.array, *args) -> Tuple[np.array, np.array]:
        """Returns the prediction of the test set and store the

        Args:
            test_x (np.array): Features of the test set
            *args : List with the specific parameter of the Basis

        Returns:
            np.array: Prediction of the test set
        """    
        num_test_samples = len(test_x)
        labels = np.zeros(num_test_samples)
        confidence_level = np.zeros((num_test_samples, self.num_classes))

        for i in range(num_test_samples):
            confidence_level[i, :] = [self.basis(train_x=self.train_x, train_y=self.probability_classes[c], test_one_sample=test_x[i], k=self.k, z=args[0]) for c in range(self.num_classes)]
            labels[i] = np.argmax(confidence_level[i, :])


        return labels, confidence_level

    def FEMaRelax(train_x: np.array, train_y: np.array, test_x: np.array, k_relax: int, num_repeats: int) -> np.array:
        train_x_cp = train_x.copy()
        test_x_cp = test_x.copy()

        for r in range(num_repeats):
            for i in range(train_x_cp.shape[0]):
                dist_train = np.linalg.norm(train_x_cp - train_x_cp[i], axis=1)
                index_k_relax_train = np.argsort(dist_train)[1:k_relax + 1]
                signal_train = np.array([1 if train_y[i] == train_y[idx] else -1 for idx in index_k_relax_train])
                train_x_cp[i] = np.sum(train_x_cp[index_k_relax_train] * signal_train[:, np.newaxis], axis=0) / len(index_k_relax_train)

            for i in range(test_x_cp.shape[0]):
                dist_test = np.linalg.norm(train_x_cp - test_x_cp[i], axis=1)
                index_k_relax_test = np.argsort(dist_test)[1:k_relax + 1]
                test_x_cp[i] = np.mean(train_x_cp[index_k_relax_test], axis=0)

        return train_x_cp, test_x_cp
