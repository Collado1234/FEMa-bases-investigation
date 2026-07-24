import numpy as np
from .base_basis import BaseBasis

class CompactSupportBasis(BaseBasis):

    #: h é opcional (None -> raio automático = max(dists)), então não
    #: entra em PARAMS/self._require — mas ainda é tunável, então fica
    #: aqui para quem monta grade de busca de hiperparâmetros saber disso.
    OPTIONAL_PARAMS = ("h",)

    def _normalized_distance(
        self,
        dists: np.ndarray,
        h: float | None = None
    ):
        if h is None:
            h = np.max(dists)

        return dists / h