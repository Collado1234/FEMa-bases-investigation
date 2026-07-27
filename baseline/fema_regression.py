import sys
from pathlib import Path

import numpy as np
import math

# pasta onde este arquivo está
src_dir = Path(__file__).resolve().parent

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from fem_basis import Basis

class FEMaRegressor:
    """
    Class responsible to perform the regression using FEMa
    approach
    """
    def __init__(self):
        self.train_x = None 
        self.train_y = None
        self.num_train_samples = 0
        self.num_features = 0
        self.k = 1
        self.basis = None
        

    def __init__(self, k:int=2, basis=Basis.shepardBasis) -> None:
        """Constructor that receives the parameter to run the FEMa

        Args:
            k (int): Define the number of neighboor used to interpolate
            basis = The finite element basis
        """
        self.train_x = None
        self.train_y = None
        self.num_train_samples = 0
        self.num_features = 0
        self.k = k
        self.basis = basis


    def fit(self, train_x:np.array, train_y:np.array) -> None:
        """Create the train_x and train_y inside the object and fill 
        the parameters

        Args:
            train_x (np.array): The feature of the training set
            train_y (np.array): The target of the training set           
        """
        self.train_x = train_x
        self.train_y = train_y
        self.num_train_samples = len(train_x)
        self.num_features = train_x.shape[1]

    def predict(self, test_x:np.array, *args) -> np.array:
        """Returns the prediction of the test set

        Args:
            test_x (np.array): Features of the test set
            *args : List with the specific paramter of the Basis

        Returns:
            np.array: Prediction of the test set
        """
        predicted_list = [self.basis(train_x=self.train_x, train_y=self.train_y, test_one_sample=test_x[i], k=self.k, z=args[0]) for i in range(len(test_x))]

        return predicted_list


