from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from ..neighboor_search import BaseSearch
from .parameters import BasisParameters

DELTA = 1e-10


@dataclass
class NeighborhoodContext:
    """
    Contexto opcional de uma chamada de compute_weights, além do vetor
    `dists` em si.

    MOTIVAÇÃO (ponto de extensão arquitetural): hoje toda base concreta
    calcula phi(d) só a partir da distância escalar já resolvida pelo
    Search. Isso é suficiente para as ~20 bases atuais (todas
    radialmente simétricas: phi depende só de d = ||x - x_i||), mas é
    insuficiente para famílias de base que a literatura de FEMa sugere
    como próximos passos e que a revisão do pacote identificou como
    "fora do contrato atual":

      - bases ANISOTRÓPICAS (ex.: Mahalanobis/ARD Gaussian) — precisam
        do vetor de diferença completo (x - x_i), não só da norma;
      - bases DEPENDENTES DE DENSIDADE LOCAL (ex.: bandwidth adaptativo
        estilo Silverman/Abramson, k-NN adaptive bandwidth) — precisam
        saber QUAIS pontos de treino são os vizinhos (índices) e/ou o
        próprio ponto de consulta, não só a distância já reduzida a
        escalar;
      - bases MANIFOLD-AWARE (diffusion maps, geodésicas estilo Isomap)
        — precisam da estrutura do grafo de vizinhança, não de uma
        distância par-a-par isolada.

    Em vez de forçar uma mudança de assinatura em todas as 20 bases já
    existentes (que quebraria compatibilidade e não agrega nada a elas,
    já que são puramente radiais), este objeto é passado como parâmetro
    OPCIONAL a compute_weights. Bases que não precisam dele simplesmente
    o ignoram (o comportamento atual — chamar compute_weights(dists,
    params) sem context — continua funcionando exatamente como antes,
    inclusive para código externo/testes que não conhece essa classe).

    Bases futuras que precisarem de mais informação podem:
      1. ler `self._context` (setado por compute_weights logo antes de
         chamar evaluate) dentro do próprio evaluate(); e
      2. combinar isso com `self.search` (já injetado no construtor de
         toda BaseBasis) para reconstruir o que for necessário — ex.:
         `self.search.X_train[self._context.indices]` dá os vetores
         brutos dos vizinhos, e `self._context.query_point - vizinhos`
         dá as diferenças completas para uma base anisotrópica.

    Nenhum campo aqui é obrigatório; todos são Optional e o objeto
    inteiro pode ser None (comportamento padrão, equivalente a "sem
    contexto extra").
    """

    #: Índices, em X_train, dos pontos correspondentes a cada entrada
    #: de `dists` (mesma ordem). Combinado com `self.search.X_train`,
    #: dá acesso aos vetores brutos dos vizinhos.
    indices: Optional[np.ndarray] = None

    #: Ponto de consulta (n_features,) que originou esses vizinhos.
    query_point: Optional[np.ndarray] = None

    #: k efetivo usado nessa busca (0 = todos os pontos de treino).
    k: Optional[int] = None


class BaseBasis(ABC):
    """
    Classe base abstrata para bases de interpolação do FEMa.

    Duas camadas, propositalmente separadas:

    - evaluate(dists, params): valor CRU (não normalizado) da função de
      base phi(d), aplicado a um vetor 1D das distâncias dos k vizinhos
      de UM ponto de consulta (é o único shape que o pipeline real —
      FEMaClassifier/FEMaRegressor — de fato produz; ver
      core/models/fema_classifier.py e fema_regressor.py). Cada
      subclasse implementa só isso — é a fórmula da base em si, sem se
      preocupar com normalização. Uma subclasse NUNCA deve normalizar
      dentro de evaluate() (ver NOTA DE BUG CORRIGIDO abaixo) nem supor
      um shape 2D par-a-par: esse uso não existe em nenhum caller atual
      do projeto, e código pronto para ele especulativamente só criava
      bugs (era a causa raiz de AttentionQuadraticBasis e
      SoftmaxRadialBasis quebrarem em runtime com AxisError).

    - compute_weights(dists, params, context=None): phi normalizado
      (partição da unidade) — w = phi(d) / sum(phi(d)) — usado pelo
      FEMa como peso de vizinhos. Implementado UMA ÚNICA VEZ aqui na
      classe base, em cima de evaluate(). Nenhuma subclasse deve
      reimplementar ou antecipar essa normalização dentro de evaluate():
      é justamente onde viviam os bugs de dupla normalização e de
      auto-divisão (`weights /= weights`) encontrados em revisões
      anteriores — centralizar elimina essa classe inteira de bug por
      construção. `context` é o ponto de extensão para bases futuras
      que precisem de mais do que a distância escalar (ver
      NeighborhoodContext acima); é inteiramente opcional e não afeta
      nenhuma base atual.

    NOTA DE BUG CORRIGIDO: AttentionQuadraticBasis e SoftmaxRadialBasis
    chegaram a normalizar dentro do próprio evaluate() (com
    `np.sum(..., axis=1)`), o que (a) duplicava a normalização já feita
    aqui em compute_weights, e (b) quebrava em runtime, já que `dists`
    é sempre 1D nesse pipeline — `axis=1` não existe em um vetor 1D.
    Regra geral daqui pra frente: evaluate() SEMPRE devolve phi(d) cru;
    normalização é 100% responsabilidade de compute_weights.

    Busca de vizinhos e cálculo de distâncias continuam delegados ao
    Search, injetado no construtor. Como o mesmo objeto Search é
    compartilhado entre a base e o modelo (FEMaClassifier/FEMaRegressor
    passam a MESMA instância para os dois — ver models/fema.py), toda
    base já tem acesso a `self.search.X_train` (os dados de treino
    inteiros) sem precisar de nenhuma mudança de interface — combinado
    com `NeighborhoodContext.indices`, isso é o bastante para bases
    anisotrópicas e density-aware sem quebrar as 20 bases radiais
    existentes.
    """

    #: Nomes dos campos de BasisParameters que esta base efetivamente usa.
    #: Toda subclasse concreta deve sobrescrever isso. Permite validar,
    #: em uma única chamada (self._require), que os hiperparâmetros
    #: necessários foram fornecidos.
    PARAMS: Tuple[str, ...] = ()

    #: Campos que a base usa mas que são OPCIONAIS (None é um valor válido
    #: com significado próprio, ex.: h=None em bases de suporte compacto
    #: significa "raio automático" — não passam por self._require). Existe
    #: separado de PARAMS para que ferramentas externas (ex.: grade de
    #: busca de hiperparâmetros em models/fema_plugin.py) saibam que esses
    #: campos ainda valem a pena tunar, mesmo não sendo obrigatórios.
    OPTIONAL_PARAMS: Tuple[str, ...] = ()

    def __init__(self, search: BaseSearch):
        self.search = search
        #: Setado por compute_weights logo antes de chamar evaluate();
        #: ponto de extensão para bases futuras (ver NeighborhoodContext).
        #: Bases atuais não precisam ler isto.
        self._context: Optional[NeighborhoodContext] = None

    @abstractmethod
    def evaluate(self, dists: np.ndarray, params: BasisParameters) -> np.ndarray:
        """
        Valor cru de phi(d), aplicado elemento a elemento em `dists`
        (vetor 1D das distâncias dos k vizinhos de um ponto de
        consulta). NUNCA normalize aqui — isso é responsabilidade
        exclusiva de compute_weights, herdado da classe base.
        """
        raise NotImplementedError

    def compute_weights(
        self,
        dists: np.ndarray,
        params: BasisParameters,
        context: Optional[NeighborhoodContext] = None,
    ) -> np.ndarray:
        """
        Pesos normalizados (partição da unidade): w = phi(d) / sum(phi(d)).

        Se sum(phi(d)) == 0 (ex.: base de suporte compacto onde nenhum
        vizinho cai dentro do raio), cai para pesos uniformes em vez de
        propagar um NaN silencioso.

        `context` é opcional (default None) e existe só para bases
        futuras que precisem de mais do que `dists` — ver
        NeighborhoodContext. Chamadas existentes (`compute_weights(dists,
        params)`, sem o terceiro argumento) continuam funcionando
        exatamente como antes.
        """
        self._context = context
        phi = np.asarray(self.evaluate(dists, params), dtype=float)
        total = phi.sum()

        if total == 0:
            return np.full_like(phi, 1.0 / phi.size)

        return phi / total

    def _require(self, params: BasisParameters) -> Dict[str, float]:
        """
        Extrai de `params` os campos declarados em self.PARAMS e levanta
        um ValueError claro se algum estiver None (não configurado).
        """
        values: Dict[str, float] = {}
        missing = []

        for name in self.PARAMS:
            value = getattr(params, name)
            if value is None:
                missing.append(name)
            values[name] = value

        if missing:
            raise ValueError(
                f"{type(self).__name__} requer os parâmetros {self.PARAMS}, "
                f"mas {missing} não foram configurados em BasisParameters "
                f"(estão None)."
            )

        return values
