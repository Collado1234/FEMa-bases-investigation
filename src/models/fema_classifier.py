from abc import ABC, abstractmethod
import numpy as np
import algebra.neighboor_search.base_search as base_search
import algebra.neighboor_search.brute_force as brute_force
import algebra.basis.basi_model as basi_model
import algebra.basis.sheppard as sheppard
import algebra.distances.base_distance as base_distance
import algebra.distances.euclidean_distance as euclidean_distance
from models.base_model import FEMaBaseModel
class FEMaClassifier(FEMaBaseModel):
    def __init__(self,
                distance=base_distance.EuclideanDistance(),
                search=brute_force.BruteForce()):
        super().__init__(distance, search)
    
    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        self.X_train = X
        self.y_train = y
        
