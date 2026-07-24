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
    z=2.0,
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


def check_leakage_matches_worst_case_prediction(basis: BaseBasis, params: BasisParameters, dists: np.ndarray = TEST_DISTS, tol: float = DEFAULT_TOL):
    """
    Verificação empírica (não só algébrica) de que leakage é de fato o
    erro de `weights @ y` no pior caso: monta y_j=0 no vizinho de
    distância zero e y_i=1 em todos os outros (máxima diferença possível
    dentro de [0, 1]), calcula `weights @ y` DE VERDADE, e confere que
    o erro resultante bate com `leakage` calculado pela fórmula fechada.

    Existe para não depender só da dedução algébrica (predição - y_j =
    Σ wi(yi - y_j)) sem nunca ter sido conferida contra o cálculo direto.

    Returns
    -------
    bool, ou None se compute_weights falhou estruturalmente.
    """
    weights, error = _safe_compute(basis, dists, params)
    if error:
        return None

    zero_idx = np.argmin(dists)
    y = np.ones_like(dists)
    y[zero_idx] = 0.0

    prediction_error = abs(float(weights @ y))  # weights @ y, de fato, no pior caso
    leak, _ = compute_interpolation_leakage(basis, params, dists=dists)

    return bool(np.isclose(prediction_error, leak["leakage"], atol=tol))


def compute_interpolation_leakage(basis: BaseBasis, params: BasisParameters, dists: np.ndarray = TEST_DISTS):
    """
    Mede o leakage = 1 - peso no vizinho de distância zero, que é o
    limite exato (superior E atingível) do erro de interpolação
    |weights @ y - y_j| para qualquer y — ver docstring do módulo.

    Essa equivalência é conferida empiricamente por
    check_leakage_matches_worst_case_prediction, não é só assumida.

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
        "leakage_matches_prediction": check_leakage_matches_worst_case_prediction(basis, params),
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


# Para cada base, qual campo de BasisParameters é o "parâmetro de escala"
# (o que controla o quão concentrada/peaked a função fica perto de d=0) e
# em que faixa varrê-lo. Direção do efeito varia por base (às vezes
# aumentar concentra, às vezes diminuir concentra) — por isso a varredura
# testa uma faixa ampla em vez de assumir a direção.
#
# None como parâmetro de escala significa "esta base não tem, hoje, um
# parâmetro que controle a concentração perto de d=0" — não é omissão,
# é uma limitação real da fórmula atual (ver notas em lorentzian_basis.py
# e attention_quadratica.py: ambas fixas, sem parâmetro de escala).
SWEEP_CONFIG = {
    "shepard": ("z", np.geomspace(0.5, 8.0, 25)),
    "radial": ("z", np.geomspace(0.01, 5.0, 25)),
    "rbf_gaussian": ("epsilon", np.geomspace(0.1, 50.0, 25)),
    "multiquadratic": ("c", np.geomspace(1e-3, 5.0, 25)),  # esperado: continua nao_interpoladora p/ qualquer c (cresce com d por construcao)
    "inverse_multiquadratic": ("c", np.geomspace(1e-3, 5.0, 25)),
    "wendland_c2": ("h", np.geomspace(0.1, 5.0, 25)),
    "cubic_spline": ("h", np.geomspace(0.1, 5.0, 25)),
    "quartic_spline": ("h", np.geomspace(0.1, 5.0, 25)),
    "cosine": ("h", np.geomspace(0.1, 5.0, 25)),
    "gen_exponential": ("epsilon", np.geomspace(0.1, 50.0, 25)),
    "softmax_radial": ("beta", np.geomspace(0.1, 50.0, 25)),
    "attention": (None, None),  # sem parametro de escala hoje
    "logarithmic": ("c", np.geomspace(1e-3, 5.0, 25)),
    "harmonic": ("nu", np.geomspace(1e-6, 50.0, 33)),
    "laplacian": ("epsilon", np.geomspace(0.1, 50.0, 25)),
    "cauchy": ("epsilon", np.geomspace(0.1, 50.0, 25)),
    "student_t": ("nu", np.geomspace(1e-6, 50.0, 33)),
    "sigmoidal": ("alpha", np.geomspace(0.1, 50.0, 25)),
    "lorentzian": (None, None),  # sem parametro de escala hoje
    "entropic": ("beta", np.geomspace(0.1, 50.0, 25)),
    "rational_quadratic": ("l", np.geomspace(1e-3, 5.0, 25)),
}


def sweep_leakage(basis: BaseBasis, base_params: BasisParameters, param_name: str, values: np.ndarray, dists: np.ndarray = TEST_DISTS):
    """
    Varre um único campo de BasisParameters (mantendo os demais fixos em
    base_params) e mede o leakage para cada valor.

    Returns
    -------
    list[(valor, leakage_ou_None)] — leakage None quando compute_weights
    falhou estruturalmente para aquele valor específico (ex.: parâmetro
    fora de domínio válido).
    """
    results = []
    for value in values:
        params = replace_field(base_params, param_name, float(value))
        leak, error = compute_interpolation_leakage(basis, params, dists=dists)
        results.append((float(value), leak["leakage"] if leak else None))
    return results


def replace_field(params: BasisParameters, field: str, value: float) -> BasisParameters:
    """Copia BasisParameters trocando um único campo (dataclasses.replace)."""
    from dataclasses import replace
    return replace(params, **{field: value})


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
            confirmado = results["leakage_matches_prediction"]
            check = "confirmado por weights@y" if confirmado else "[AVISO] weights@y NAO bateu com a formula!"
            print(f"  {'interpolacao_exata':<35} [{classe}]  leakage={leakage:.4g}  ({check})")
            ranking.append((name, leakage))

        overall = "[OK]" if results["all"] else "[NEM TODAS PASSARAM]"
        print(f"  {'resumo':<35} {overall}")

    print("\n" + "=" * 70)
    print("RANKING: proximidade da interpolacao exata (leakage, menor = mais proximo)")
    print("=" * 70)
    for name, leakage in sorted(ranking, key=lambda item: item[1]):
        classe = classify_interpolation(leakage)
        print(f"  {name:<25} leakage={leakage:.6g}   [{classe}]")

    print("\n" + "=" * 70)
    print("VARREDURA DE PARAMETRO: melhor leakage possivel por base")
    print("(TEST_PARAMS fixo é só UM ponto do espaço de parâmetros — aqui")
    print(" variamos o parâmetro de escala de cada base para achar o quão")
    print(" perto de interpoladora ela CONSEGUE ficar, e com qual valor)")
    print("=" * 70)

    sweep_results = []  # (nome, melhor_leakage, melhor_valor, param_name)

    for name in basis_names:
        param_name, values = SWEEP_CONFIG.get(name, (None, None))
        basis = Basis.get(name, search)

        if param_name is None:
            print(f"  {name:<25} sem parâmetro de escala disponível hoje")
            continue

        sweep = sweep_leakage(basis, TEST_PARAMS, param_name, values)
        valid = [(v, l) for v, l in sweep if l is not None]

        if not valid:
            print(f"  {name:<25} varredura falhou estruturalmente para todos os valores testados")
            continue

        best_value, best_leakage = min(valid, key=lambda item: item[1])
        classe = classify_interpolation(best_leakage)
        print(f"  {name:<25} melhor leakage={best_leakage:.4g}  em {param_name}={best_value:.4g}   [{classe}]")
        sweep_results.append((name, best_leakage, best_value, param_name))

    print("\nFinalizado.")