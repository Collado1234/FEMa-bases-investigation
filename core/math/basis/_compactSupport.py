import numpy as np
from .base_basis import BaseBasis

#: Menor raio aceito; evita divisao por zero quando todos os k vizinhos coincidem
#: com o ponto de consulta (dist. 0). Nesse caso r = 0 e todos recebem phi(0) (pesos iguais).
_MIN_RADIUS = 1e-12


class CompactSupportBasis(BaseBasis):
    """Base de suporte compacto com raio ADAPTATIVO a densidade local.

    PROBLEMA QUE ISTO RESOLVE. Antes, o raio de suporte `h` era um valor ABSOLUTO
    (busca em [0.5, 5]). Com features padronizadas, a escala das distancias cresce com
    sqrt(n_features) e muda com a densidade do dataset: em alta dimensao o vizinho mais
    proximo ja esta bem alem de h = 5, a soma dos pesos da 0 e o codigo caia em pesos
    uniformes (kNN simples) - as quatro bases compactas viravam o MESMO classificador.

    SOLUCAO. O raio e' definido por consulta, em unidades da escala local:

        raio = h_rel * r_k ,   r_k = max(dists) = distancia ao k-esimo vizinho

    r_k e' o estimador kNN de densidade (p(x) ~ k / (n * V_d * r_k^d)): onde a densidade
    e' alta r_k e' pequeno e o suporte encolhe; onde e' baixa o suporte cresce. h_rel e'
    adimensional e comparavel entre datasets e dimensoes:
      - h_rel = 1  : o k-esimo vizinho fica exatamente na borda (peso 0);
      - h_rel > 1  : todos os k vizinhos tem peso > 0 (sem fallback), e quanto maior,
                     mais uniformes os pesos (h_rel -> inf recupera o kNN uniforme);
      - h_rel None : equivale a 1 (comportamento anterior de `h=None`).

    `h` (absoluto) continua aceito em BasisParameters, com prioridade sobre h_rel, so'
    para curvas/diagnosticos; NAO e' mais um hiperparametro tunado.
    """

    OPTIONAL_PARAMS = ("h_rel",)

    def _support_radius(self,
            dists: np.ndarray,
            h: float | None = None,
            h_rel: float | None = None
            ) -> float:
        if h is not None:
            return max(float(h), _MIN_RADIUS)
        r_k = float(np.max(dists) if np.size(dists) else 0.0)
        scale = 1.0 if h_rel is None else float(h_rel)

        return max(scale * r_k, _MIN_RADIUS)

    def _normalized_distance(self, dists: np.ndarray, h: float | None = None, h_rel: float | None = None):
        return dists / self._support_radius(dists, h, h_rel)
