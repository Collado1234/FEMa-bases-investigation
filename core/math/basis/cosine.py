import numpy as np
from .parameters import BasisParameters
from ._compactSupport import CompactSupportBasis

class CosineBasis(CompactSupportBasis):
    """
    Raised-cosine, suporte compacto: w[r] = cos(pi*r/2)^2, r = d/h, 0 caso r > 1.
    (variante ao quadrado do "cosine kernel" de KDE; cf. Silverman,
    Density Estimation for Statistics and Data Analysis, 1986)

    NOTA: o parâmetro `alpha` existia na assinatura antiga mas nunca era
    usado na fórmula — bug de parâmetro morto. Removido de PARAMS até
    que se decida se a base deveria incorporá-lo (ex. como expoente do
    cosseno) ou se `alpha` deve simplesmente deixar de existir para esta
    base. h é opcional (None -> usa max(dists) como raio de suporte),
    por isso não passa por self._require.
    """
    PARAMS = ()

    def __init__(self, search):
        super().__init__(search)

    def compute_weights(self, dists:np.ndarray, params: BasisParameters) -> np.ndarray:

        r = self._normalized_distance(dists, params.h)

        weights = np.where(
            r <= 1,
            np.cos(np.pi * r / 2)**2,
            0.0
        )

        total = np.sum(weights)
        
        if total > 0:
            weights /= total

        return weights