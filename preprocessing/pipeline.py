import numpy as np


class Pipeline:
    """
    Encadeia múltiplas transformações de preprocessing.
    """

    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        for step in self.steps:
            if hasattr(step, "fit"):
                step.fit(X, y)
        return self

    def transform(self, X, y=None):
        data = X

        for step in self.steps:
            if isinstance(data, tuple):
                data = step.transform(data)
            else:
                data = step.transform(data)

        return data

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)