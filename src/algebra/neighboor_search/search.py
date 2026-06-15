import numpy as np

class Search:
    def __init__(self):
        pass

    @staticmethod
    def find_k_neighboors(distances, len_train_x, k):
        if k != 0:
            k_nearest_indices = np.argpartition(distances, k)[:k]
        else:
            k_nearest_indices = np.arange(len_train_x)
        
        return k_nearest_indices