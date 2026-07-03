from abc import ABC, abstractmethod
import numpy as np


class Transform(ABC):
    """
    Interface base para qualquer transformação de preprocessing.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        pass

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)