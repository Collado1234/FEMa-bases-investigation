from algebra.basis.fem_basis import Basis
from typing import Tuple
import numpy as np


class FEMaRegressor:
    """
    Class responsible to perform the regression using FEMa approach.
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
    