import numpy as np
from .base import Transform


class StandardScaler(Transform):
    """
    Normalização padrão: (x - mean) / std
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X, y=None):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        return (X - self.mean) / self.std