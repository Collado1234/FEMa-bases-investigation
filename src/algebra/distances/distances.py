import numpy as np

class Distance:
    def __init__(self):
        pass

    @staticmethod
    def euclidean_distance(X:np.ndarray, sample:np.ndarray) -> np.ndarray
        return np.linalg.norm(X-sample,axis=1)
    
    