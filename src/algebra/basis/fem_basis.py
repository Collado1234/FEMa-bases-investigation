import numpy as np
from algebra.distances.distances import Distance
from algebra.neighboor_search.search import Search
import math

class Basis:
    def __init__(self) -> None:
        pass

    @staticmethod
    def shepardBasis(X_train:np.ndarray,
                     X_sample:np.ndarray,
                     y_train:np.ndarray,
                     k_neighboors:int,
                     z:int) -> float:
        
        distances = Distance.euclidean_distance(X_train, X_sample)
        k_nearest_indices = Search.find_k_neighboors(distances=distances,
                                                     len_train_x=len(X_train),
                                                     k=k_neighboors
                                                     )
        
        distances = distances[k_nearest_indices]
        y_k_train = y_train[k_nearest_indices]

        distances = np.where(distances == 0, 1e-10, distances)  # Evita divisão por zero
        weights = 1.0 / (distances**z)
        weights = weights/np.sum(weights) #Partição de Unidade

        predicted = np.sum(weights*y_k_train)

        if math.isnan(predicted):
            predicted = np.mean(y_train)

        return predicted
    
    @staticmethod
    def radialBasis(X_train:np.ndarray,
                     X_sample:np.ndarray,
                     y_train:np.ndarray,
                     k_neighboors:int,
                     z:int) -> float:
        distances = Distance.euclidean_distance(X_train, X_sample)
        k_nearest_indices = Search.find_k_neighboors(distances=distances,
                                                     len_train_x=len(X_train),
                                                     k=k_neighboors
                                                     )
        
        distances = distances[k_nearest_indices]
        y_k_train = y_train[k_nearest_indices]

        weights = np.exp(-0.5*(distances/z)**2)
        weights = weights/np.sum(weights) #partição de unidade

        predicted = np.sum(weights*y_k_train)

        if math.isnan(predicted):
            predicted = np.mean(y_train)
        
        return predicted


        