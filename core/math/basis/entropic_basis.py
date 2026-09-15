import numpy as np
from .base_basis import BaseBasis
from .parameters import BasisParameters


class EntropicBasis(BaseBasis):
    """phi_i(d) = exp(-beta * d_i^2) / sum_j exp(-beta * d_j^2)."""
    PARAMS = ("beta",)

    def __init__(self, search):
        super().__init__(search)

    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        beta = self._require(params)["beta"]

        dists = np.asarray(dists, dtype=float)

        was_1d = dists.ndim == 1

        if was_1d:
            dists = dists.reshape(1, -1)

        weights = np.exp(-beta * dists**2)
        soma = np.sum(weights, axis=1, keepdims=True)

        result = weights / soma

        if was_1d:
            return result[0]

        return result