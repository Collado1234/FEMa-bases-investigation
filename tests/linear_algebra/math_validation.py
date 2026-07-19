"""
validation.py

Funções para validar propriedades matemáticas de bases de interpolação
utilizadas pelo FEMa.

Propriedades verificadas:

- Partição da unidade
- Não negatividade
- Igualdade dos pesos para distâncias iguais
- Monotonicidade
- Interpolação exata (distância zero)


No git bash: python -m tests.linear_algebra.math_validation
"""

import numpy as np

from core.math.basis import BaseBasis


DEFAULT_TOL = 1e-6


def check_partition_of_unity(
    basis: BaseBasis,
    dists: np.ndarray,
    z: float,
    tol: float = DEFAULT_TOL,
) -> bool:
    """
    Verifica se os pesos formam uma partição da unidade.

    Σ wi = 1
    """
    weights = basis.compute_weights(dists, z)

    return bool(np.isclose(np.sum(weights), 1.0, atol=tol))


def check_non_negative_weights(
    basis: BaseBasis,
    dists: np.ndarray,
    z: float,
) -> bool:
    """
    Verifica se todos os pesos são não negativos.
    """
    weights = basis.compute_weights(dists, z)

    return bool(np.all(weights >= 0))


def check_equal_distances(
    basis: BaseBasis,
    n_points: int,
    z: float,
    tol: float = DEFAULT_TOL,
) -> bool:
    """
    Se todas as distâncias forem iguais,
    todos os pesos devem ser iguais.
    """
    dists = np.ones(n_points)

    weights = basis.compute_weights(dists, z)

    expected = np.full(n_points, 1.0 / n_points)

    return bool(np.allclose(weights, expected, atol=tol))


def check_monotonicity(
    basis: BaseBasis,
    dists: np.ndarray,
    z: float,
) -> bool:
    """
    Verifica se pesos diminuem conforme a distância aumenta.

    Assume que dists está ordenado em ordem crescente.
    """
    weights = basis.compute_weights(dists, z)

    return bool(np.all(np.diff(weights) <= 0))


def check_exact_interpolation(  # Ajustar para avaliar percentual de erro
    basis: BaseBasis,
    z: float,
    tol: float = DEFAULT_TOL,
) -> bool:
    """
    Verifica a propriedade interpoladora.

    Quando existe uma distância exatamente igual a zero,
    o peso correspondente deve ser 1 e todos os demais 0.
    """
    dists = np.array([0.0, 1.0, 2.0, 3.0])

    weights = basis.compute_weights(dists, z)

    expected = np.array([1.0, 0.0, 0.0, 0.0])

    return bool(np.allclose(weights, expected, atol=tol))


def check_closest_point_has_highest_weight(
    basis: BaseBasis,
    dists: np.ndarray,
    z: float,
) -> bool:
    """
    O ponto mais próximo deve possuir o maior peso.
    """
    weights = basis.compute_weights(dists, z)

    return bool(np.argmax(weights) == np.argmin(dists))


def validate_basis(
    basis: BaseBasis,
    z: float = 2.0,
    tol: float = DEFAULT_TOL,
) -> dict:
    """
    Executa todas as validações matemáticas da base.

    Returns
    -------
    dict
        Dicionário contendo o resultado de cada teste.
    """

    dists = np.array([1.0, 2.0, 3.0, 4.0])

    results = {
        "partition_of_unity": check_partition_of_unity(
            basis, dists, z, tol
        ),
        "non_negative_weights": check_non_negative_weights(
            basis, dists, z
        ),
        "equal_distances": check_equal_distances(
            basis, 4, z, tol
        ),
        "monotonicity": check_monotonicity(
            basis, dists, z
        ),
        "exact_interpolation": check_exact_interpolation(
            basis, z, tol
        ),
        "closest_point_highest_weight": check_closest_point_has_highest_weight(
            basis, dists, z
        ),
    }

    results["all"] = all(results.values())

    return results


if __name__ == "__main__":

    from core.math.basis import Basis
    from core.math.distances import EuclideanDistance
    from core.math.neighboor_search import BruteForceSearch

    search = BruteForceSearch(EuclideanDistance())

    basis_names = [
        "shepard",
        "radial",
        "rbf_gaussian",
        "multiquadratic",
        "inverse_multiquadratic",
        "wendland_c2",
        "cubic_spline",
        "quartic_spline",
        "gen_exponential",
        "softmax_radial",
    ]

    z = 2.0

    print("=" * 70)
    print("VALIDACAO DAS BASES DO FEMa")
    print("=" * 70)

    for name in basis_names:

        print(f"\nBase: {name}")

        basis = Basis.get(name, search)

        results = validate_basis(basis, z)

        for property_name, passed in results.items():

            status = "[OK]" if passed else "[ERRO]"

            print(f"  {property_name:<35} {status}")

    print("\nFinalizado.")