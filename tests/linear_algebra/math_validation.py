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
- Interpolação exata (distância zero)
- Maior peso para o ponto mais próximo

IMPORTANTE: nem toda base deve passar em todas as propriedades — isso é
esperado e documentado no projeto (ex.: Gaussian normalizada não é
interpoladora; Multiquadratic não é monotonicamente decrescente). O
objetivo deste script não é "zerar erros", e sim registrar o perfil de
propriedades de cada base para análise posterior.

Bases e parâmetros são descobertos automaticamente:
    - os nomes das bases vêm de Basis.available(), que reflete
      diretamente o dicionário _CLASSES da fábrica (core/math/basis/
      factory_basis.py). Cadastrar uma base nova lá é suficiente para
      que ela apareça aqui — não é necessário editar este arquivo.
    - os parâmetros de compute_weights (z, epsilon, c, alpha, h, ...)
      são descobertos via inspect.signature e preenchidos a partir de
      DEFAULT_PARAM_VALUES. Se uma base introduzir um nome de parâmetro
      novo (que não tenha valor default na própria assinatura), este
      script levanta um erro claro pedindo para cadastrar um valor de
      teste em DEFAULT_PARAM_VALUES — a única edição manual que ainda
      pode ser necessária, e é intencional (evita adivinhar um valor
      sem sentido matemático para um parâmetro desconhecido).

No git bash: python -m tests.linear_algebra.math_validation
"""

import inspect

import numpy as np

from core.math.basis import BaseBasis


DEFAULT_TOL = 1e-6

# Valor de teste padrão para cada NOME de parâmetro que pode aparecer em
# alguma assinatura de compute_weights (além de `dists`). Ao adicionar uma
# base com um parâmetro de nome novo, inclua uma entrada aqui.
DEFAULT_PARAM_VALUES = {
    "z": 2.0,
    "epsilon": 1.0,
    "c": 1.0,
    "alpha": 1.0,
    "beta": 1.0,
    "nu": 3.0,
    "h": 3.0,
    "l": 1.0,
    "p": 2.0,
}


def build_kwargs(basis: BaseBasis) -> dict:
    """
    Inspeciona compute_weights e monta os kwargs de teste automaticamente,
    na ordem/nomes declarados pela própria base — permite que cada base
    tenha uma assinatura diferente sem que o harness precise conhecê-la
    de antemão.
    """
    sig = inspect.signature(basis.compute_weights)
    kwargs = {}

    for name, param in sig.parameters.items():
        if name in ("self", "dists"):
            continue

        if name in DEFAULT_PARAM_VALUES:
            kwargs[name] = DEFAULT_PARAM_VALUES[name]
        elif param.default is not inspect.Parameter.empty:
            kwargs[name] = param.default
        else:
            raise ValueError(
                f"Parâmetro '{name}' de {type(basis).__name__}.compute_weights "
                f"não possui valor de teste em DEFAULT_PARAM_VALUES nem "
                f"default na assinatura. Adicione uma entrada para ele."
            )

    return kwargs


def _safe_compute(basis: BaseBasis, dists: np.ndarray, kwargs: dict):
    """
    Chama compute_weights protegendo contra os bugs estruturais mais
    comuns: exceção na chamada, retorno None (falta de `return`) e
    NaN/Inf no resultado (ex.: divisão por zero não tratada).

    Returns
    -------
    (weights, error): weights é None se algo deu errado; error traz uma
    mensagem curta explicando o motivo.
    """
    try:
        weights = basis.compute_weights(dists, **kwargs)
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


def check_equal_distances(basis: BaseBasis, kwargs: dict, n_points: int, tol: float = DEFAULT_TOL):
    """Se todas as distâncias forem iguais, todos os pesos devem ser iguais."""
    dists = np.ones(n_points)
    weights, error = _safe_compute(basis, dists, kwargs)
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


def check_exact_interpolation(basis: BaseBasis, kwargs: dict, tol: float = DEFAULT_TOL):
    """
    Propriedade interpoladora: quando existe distância exatamente zero,
    o peso correspondente deve ser 1 e todos os demais 0.
    """
    dists = np.array([0.0, 1.0, 2.0, 3.0])
    weights, error = _safe_compute(basis, dists, kwargs)
    if error:
        return None
    expected = np.array([1.0, 0.0, 0.0, 0.0])
    return bool(np.allclose(weights, expected, atol=tol))


def check_closest_point_has_highest_weight(dists: np.ndarray, weights: np.ndarray) -> bool:
    """O ponto mais próximo deve possuir o maior peso."""
    return bool(np.argmax(weights) == np.argmin(dists))


def validate_basis(basis: BaseBasis, tol: float = DEFAULT_TOL) -> dict:
    """
    Executa todas as validações matemáticas da base.

    Returns
    -------
    dict com o resultado de cada teste. Um valor None indica que o teste
    não pôde ser avaliado porque compute_weights falhou estruturalmente
    para aquele cenário (ver campo "structural_error").
    """
    dists = np.array([1.0, 2.0, 3.0, 4.0])

    kwargs = build_kwargs(basis)
    weights, error = _safe_compute(basis, dists, kwargs)

    if error:
        return {
            "structural_ok": False,
            "structural_error": error,
            "params_used": kwargs,
            "partition_of_unity": None,
            "non_negative_weights": None,
            "equal_distances": None,
            "monotonicity": None,
            "exact_interpolation": None,
            "closest_point_highest_weight": None,
            "all": False,
        }

    results = {
        "structural_ok": True,
        "structural_error": None,
        "params_used": kwargs,
        "partition_of_unity": check_partition_of_unity(weights, tol),
        "non_negative_weights": check_non_negative_weights(weights),
        "equal_distances": check_equal_distances(basis, kwargs, 4, tol),
        "monotonicity": check_monotonicity(weights),
        "exact_interpolation": check_exact_interpolation(basis, kwargs, tol),
        "closest_point_highest_weight": check_closest_point_has_highest_weight(dists, weights),
    }

    # "all" ignora testes que não puderam ser avaliados (None) — eles já
    # aparecem sinalizados individualmente, não precisam derrubar o
    # resumo geral.
    results["all"] = all(
        v for k, v in results.items()
        if k not in ("structural_error", "params_used") and v is not None
    )

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
    print("=" * 70)

    for name in basis_names:

        print(f"\nBase: {name}")

        basis = Basis.get(name, search)

        results = validate_basis(basis)

        if not results["structural_ok"]:
            print(f"  [ERRO ESTRUTURAL] {results['structural_error']}")
            continue

        print(f"  parâmetros de teste: {results['params_used']}")

        for property_name, passed in results.items():
            if property_name in ("structural_ok", "structural_error", "params_used", "all"):
                continue
            status = "[OK]" if passed else ("[N/A]" if passed is None else "[FALHA]")
            print(f"  {property_name:<35} {status}")

        overall = "[OK]" if results["all"] else "[NEM TODAS PASSARAM]"
        print(f"  {'resumo':<35} {overall}")

    print("\nFinalizado.")