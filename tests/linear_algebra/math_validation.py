"""
validation.py

Funções para validar propriedades matemáticas de bases de interpolação
utilizadas pelo FEMa.

Propriedades verificadas:

- Sanidade estrutural (compute_weights não deve levantar exceção nem
  retornar None/NaN — pega bugs de implementação antes dos matemáticos)
- Partição da unidade
- Não negatividade
- Igualdade dos pesos para distâncias iguais
- Monotonicidade
- Maior peso para o ponto mais próximo
- Vazamento de interpolação ("leakage", ver seção abaixo)

SOBRE INTERPOLAÇÃO E "LEAKAGE"
--------------------------------
O que importa para a qualidade da interpolação NÃO é o vetor `weights`
em si (comparado componente a componente a um "vetor ideal" [1,0,0,...]),
e sim a predição final `weights @ y` para os valores-alvo `y` dos
vizinhos. Comparar o vetor de pesos bruto pune bases que têm pesos
pequenos, porém não-nulos, longe do vizinho mais próximo, mesmo quando
isso tem impacto desprezível na predição.

A métrica certa é o quanto a predição pode se afastar do valor
verdadeiro no pior caso. Com partição da unidade (Σ wi = 1) e o vizinho
de distância zero no índice j:

    predição - y_j = Σ_i wi * (y_i - y_j)

logo, pela desigualdade triangular:

    |predição - y_j|  <=  (1 - w_j) * max_i |y_i - y_j|

O termo (1 - w_j) — a soma dos pesos em TODOS os vizinhos exceto o de
distância zero — é chamado aqui de "leakage" (vazamento). Ele é, ao
mesmo tempo, o limite superior do erro de interpolação e um valor
atingível (é o erro de fato quando os demais pontos têm o valor mais
distante possível de y_j). Ou seja: leakage é exatamente o número que
determina o quão longe a interpolação pode ficar, para QUALQUER y — não
é uma aproximação, é o valor exato do pior caso.

Por isso classificamos a proximidade da interpolação exata pelo leakage
(não mais pela norma do vetor de pesos), em três faixas, cada uma com
tolerância configurável:

    - "exata":              leakage <= INTERPOLATION_STRICT_TOL
    - "quase_interpoladora": leakage <= INTERPOLATION_NEAR_TOL
    - "nao_interpoladora":   leakage >  INTERPOLATION_NEAR_TOL

INTERPOLATION_STRICT_TOL e INTERPOLATION_NEAR_TOL abaixo são os valores
padrão; ambos podem ser sobrescritos por chamada em validate_basis(...).

IMPORTANTE: leakage depende da relação entre a escala do parâmetro da
base (epsilon, z, c...) e o espaçamento real dos dados — a mesma base
pode ter leakage ~0 para dados bem espaçados e leakage alto para dados
densos, com o mesmo parâmetro. O harness usa um único cenário de teste
fixo (ver TEST_DISTS) só para comparação entre bases; não é uma
propriedade absoluta da base.

Bases são descobertas automaticamente via Basis.available() (reflete o
dicionário _CLASSES de factory_basis.py — cadastrar uma base nova lá é
suficiente para que ela apareça aqui). Como toda base usa a mesma
assinatura compute_weights(dists, params: BasisParameters), um único
objeto de parâmetros "genéricos" (TEST_PARAMS) serve para testar
qualquer base — cada uma lê, via self.PARAMS/self._require, só os
campos que sua fórmula usa; os demais são ignorados sem problema.

No git bash: python -m tests.linear_algebra.math_validation
"""

import numpy as np

from core.math.basis import BaseBasis, BasisParameters


DEFAULT_TOL = 1e-6

# Cenário de distâncias usado para medir o leakage de interpolação.
# O primeiro ponto (distância 0) é o "vizinho coincidente"; os demais
# são os "vizinhos que vazam" peso para fora dele.
TEST_DISTS = np.array([0.0, 1.0, 2.0, 3.0])

# Tolerâncias para classificar a proximidade da interpolação exata,
# em termos de leakage = 1 - peso no vizinho de distância zero.
# Ajuste esses dois valores (ou passe outros via validate_basis(...))
# para calibrar o que conta como "próxima o suficiente" no seu estudo.
INTERPOLATION_STRICT_TOL = 1e-6   # leakage até aqui: considerada "exata"
INTERPOLATION_NEAR_TOL = 0.05     # leakage até aqui: considerada "quase interpoladora"

# Valores de teste para TODOS os campos de BasisParameters. Cada base usa
# só o subconjunto declarado em basis.PARAMS; os demais são ignorados.
# Se um campo novo for adicionado a BasisParameters, basta uma linha aqui.
TEST_PARAMS = BasisParameters(
    z=1.0,
    epsilon=1.0,
    c=1.0,
    alpha=1.0,
    beta=1.0,
    nu=3.0,
    h=3.0,
    l=1.0,
    p=2.0,
)


def _safe_compute(basis: BaseBasis, dists: np.ndarray, params: BasisParameters):
    """
    Chama compute_weights protegendo contra os bugs estruturais mais
    comuns: exceção na chamada (incluindo ValueError de parâmetro
    faltando, ver BaseBasis._require), retorno None (falta de `return`)
    e NaN/Inf no resultado (ex.: divisão por zero não tratada).

    Returns
    -------
    (weights, error): weights é None se algo deu errado; error traz uma
    mensagem curta explicando o motivo.
    """
    try:
        weights = basis.compute_weights(dists, params)
    except Exception as exc:  # noqa: BLE001 — queremos capturar qualquer falha aqui
        return None, f"{type(exc).__name__}: {exc}"

    if weights is None:
        return None, "compute_weights retornou None (falta 'return'?)"

    weights = np.asarray(weights, dtype=float)

    if weights.shape != dists.shape:
        return None, f"shape inesperado: {weights.shape} != {dists.shape}"

    if not np.all(np.isfinite(weights)):
        return None, "pesos contêm NaN/Inf (ex.: divisão por zero não tratada)"

    return weights, None


def check_partition_of_unity(weights: np.ndarray, tol: float = DEFAULT_TOL) -> bool:
    """Verifica se os pesos formam uma partição da unidade: Σ wi = 1."""
    return bool(np.isclose(np.sum(weights), 1.0, atol=tol))


def check_non_negative_weights(weights: np.ndarray) -> bool:
    """Verifica se todos os pesos são não negativos."""
    return bool(np.all(weights >= 0))


def check_equal_distances(basis: BaseBasis, params: BasisParameters, n_points: int, tol: float = DEFAULT_TOL):
    """Se todas as distâncias forem iguais, todos os pesos devem ser iguais."""
    dists = np.ones(n_points)
    weights, error = _safe_compute(basis, dists, params)
    if error:
        return None
    expected = np.full(n_points, 1.0 / n_points)
    return bool(np.allclose(weights, expected, atol=tol))


def check_monotonicity(weights: np.ndarray) -> bool:
    """
    Verifica se os pesos diminuem conforme a distância aumenta.
    Assume que `dists` usado para gerar `weights` está em ordem crescente.
    """
    return bool(np.all(np.diff(weights) <= 0))


def check_closest_point_has_highest_weight(dists: np.ndarray, weights: np.ndarray) -> bool:
    """O ponto mais próximo deve possuir o maior peso."""
    return bool(np.argmax(weights) == np.argmin(dists))


def compute_interpolation_leakage(basis: BaseBasis, params: BasisParameters, dists: np.ndarray = TEST_DISTS):
    """
    Mede o leakage = 1 - peso no vizinho de distância zero, que é o
    limite exato (superior E atingível) do erro de interpolação
    |weights @ y - y_j| para qualquer y — ver docstring do módulo.

    Returns
    -------
    (dict, error). dict tem "leakage" e "weights"; error é None em caso
    de sucesso, ou uma mensagem curta se compute_weights falhou
    estruturalmente para esse cenário.
    """
    weights, error = _safe_compute(basis, dists, params)
    if error:
        return None, error

    zero_idx = np.argmin(dists)
    leakage = float(1.0 - weights[zero_idx])

    return {"leakage": leakage, "weights": weights}, None


def classify_interpolation(
    leakage: float,
    strict_tol: float = INTERPOLATION_STRICT_TOL,
    near_tol: float = INTERPOLATION_NEAR_TOL,
) -> str:
    """Classifica o leakage em três faixas (ver docstring do módulo)."""
    if leakage <= strict_tol:
        return "exata"
    if leakage <= near_tol:
        return "quase_interpoladora"
    return "nao_interpoladora"


def validate_basis(
    basis: BaseBasis,
    params: BasisParameters = TEST_PARAMS,
    tol: float = DEFAULT_TOL,
    interpolation_strict_tol: float = INTERPOLATION_STRICT_TOL,
    interpolation_near_tol: float = INTERPOLATION_NEAR_TOL,
) -> dict:
    """
    Executa todas as validações matemáticas da base.

    Returns
    -------
    dict com o resultado de cada teste. Um valor None indica que o teste
    não pôde ser avaliado porque compute_weights falhou estruturalmente
    para aquele cenário (ver campo "structural_error").
    """
    dists = np.array([1.0, 2.0, 3.0, 4.0])

    weights, error = _safe_compute(basis, dists, params)

    if error:
        return {
            "structural_ok": False,
            "structural_error": error,
            "partition_of_unity": None,
            "non_negative_weights": None,
            "equal_distances": None,
            "monotonicity": None,
            "closest_point_highest_weight": None,
            "interpolation_leakage": None,
            "interpolation_class": None,
            "all": False,
        }

    leak, leak_error = compute_interpolation_leakage(basis, params)

    results = {
        "structural_ok": True,
        "structural_error": None,
        "partition_of_unity": check_partition_of_unity(weights, tol),
        "non_negative_weights": check_non_negative_weights(weights),
        "equal_distances": check_equal_distances(basis, params, 4, tol),
        "monotonicity": check_monotonicity(weights),
        "closest_point_highest_weight": check_closest_point_has_highest_weight(dists, weights),
        "interpolation_leakage": leak["leakage"] if leak else None,
        "interpolation_class": (
            classify_interpolation(leak["leakage"], interpolation_strict_tol, interpolation_near_tol)
            if leak else None
        ),
        "interpolation_error_detail": leak_error,
    }

    core_checks = [
        results["partition_of_unity"],
        results["non_negative_weights"],
        results["equal_distances"],
        results["monotonicity"],
        results["closest_point_highest_weight"],
    ]
    # "quase interpoladora" conta como aprovada no resumo geral; só
    # "nao_interpoladora" reprova. Se o teste de interpolação nem pôde
    # ser avaliado (None), não penaliza o resumo geral.
    interpolation_ok = results["interpolation_class"] != "nao_interpoladora"

    results["all"] = all(v for v in core_checks if v is not None) and interpolation_ok

    return results


if __name__ == "__main__":

    from core.math.basis import Basis
    from core.math.distances import EuclideanDistance
    from core.math.neighboor_search import BruteForceSearch

    search = BruteForceSearch(EuclideanDistance())

    basis_names = Basis.available()  # auto-descoberta via factory_basis._CLASSES

    print("=" * 70)
    print("VALIDACAO DAS BASES DO FEMa")
    print(f"({len(basis_names)} bases registradas na fábrica)")
    print(f"tolerancias de interpolacao (leakage = 1 - peso no vizinho d=0): "
          f"exata <= {INTERPOLATION_STRICT_TOL}, quase <= {INTERPOLATION_NEAR_TOL}")
    print("=" * 70)

    ranking = []  # (nome, leakage) para o resumo comparativo no final

    for name in basis_names:

        print(f"\nBase: {name}")

        basis = Basis.get(name, search)

        results = validate_basis(basis)

        if not results["structural_ok"]:
            print(f"  [ERRO ESTRUTURAL] {results['structural_error']}")
            continue

        for property_name in (
            "partition_of_unity",
            "non_negative_weights",
            "equal_distances",
            "monotonicity",
            "closest_point_highest_weight",
        ):
            passed = results[property_name]
            status = "[OK]" if passed else ("[N/A]" if passed is None else "[FALHA]")
            print(f"  {property_name:<35} {status}")

        leakage = results["interpolation_leakage"]
        if leakage is None:
            print(f"  {'interpolacao_exata':<35} [N/A] ({results['interpolation_error_detail']})")
        else:
            classe = results["interpolation_class"]
            print(f"  {'interpolacao_exata':<35} [{classe}]  leakage={leakage:.4g}")
            ranking.append((name, leakage))

        overall = "[OK]" if results["all"] else "[NEM TODAS PASSARAM]"
        print(f"  {'resumo':<35} {overall}")

    print("\n" + "=" * 70)
    print("RANKING: proximidade da interpolacao exata (leakage, menor = mais proximo)")
    print("=" * 70)
    for name, leakage in sorted(ranking, key=lambda item: item[1]):
        classe = classify_interpolation(leakage)
        print(f"  {name:<25} leakage={leakage:.6g}   [{classe}]")

    print("\nFinalizado.")