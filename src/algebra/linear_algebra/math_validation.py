import numpy as np
from algebra.basis.basis import BaseInterpolation


def is_interpolating(
    basis: BaseInterpolation,
    train_x: np.ndarray,
    train_y: np.ndarray,
    k: int,
    z: float,
    tol: float = 1e-6
) -> bool:
    """
    Verifica se a base é interpoladora, ou seja, se reproduz exatamente
    os valores nos pontos de treino.

    Args:
        basis: Instância de uma interpolação (Shepard, Radial, etc.)
        train_x: Features do conjunto de treino (n_samples, n_features)
        train_y: Valores alvo do conjunto de treino (n_samples,)
        k: Número de vizinhos
        z: Parâmetro da base
        tol: Tolerância para comparação de floats

    Returns:
        True se for interpoladora, False caso contrário
    """
    for i in range(len(train_x)):
        predicted = basis.predict(
            train_x=train_x,
            train_y=train_y,
            sample=train_x[i],
            k=k,
            z=z
        )
        if not np.isclose(predicted, train_y[i], atol=tol):
            return False
    return True


def check_partition_of_unity(
    weights: np.ndarray,
    tol: float = 1e-6
) -> bool:
    """
    Verifica se um vetor de pesos é partição de unidade (soma = 1).

    Args:
        weights: Vetor de pesos
        tol: Tolerância para comparação de floats

    Returns:
        True se os pesos somam 1, False caso contrário
    """
    return bool(np.isclose(np.sum(weights), 1.0, atol=tol))


def check_non_negative_weights(weights: np.ndarray) -> bool:
    """
    Verifica se todos os pesos são não-negativos.

    Args:
        weights: Vetor de pesos

    Returns:
        True se todos os pesos são >= 0, False caso contrário
    """
    return bool(np.all(weights >= 0))


def validate_inputs(
    train_x: np.ndarray,
    train_y: np.ndarray,
    sample: np.ndarray,
    k: int
) -> None:
    """
    Valida os inputs antes de rodar a interpolação.
    Lança ValueError com mensagem clara se algo estiver errado.

    Args:
        train_x: Features do conjunto de treino (n_samples, n_features)
        train_y: Valores alvo do conjunto de treino (n_samples,)
        sample:  Amostra de teste (n_features,)
        k:       Número de vizinhos
    """
    if train_x.ndim != 2:
        raise ValueError(
            f"train_x deve ser 2D (n_samples, n_features), "
            f"mas tem shape {train_x.shape}"
        )

    if train_y.ndim != 1:
        raise ValueError(
            f"train_y deve ser 1D (n_samples,), "
            f"mas tem shape {train_y.shape}"
        )

    if len(train_x) != len(train_y):
        raise ValueError(
            f"train_x e train_y devem ter o mesmo número de amostras, "
            f"mas têm {len(train_x)} e {len(train_y)}"
        )

    if sample.ndim != 1:
        raise ValueError(
            f"sample deve ser 1D (n_features,), "
            f"mas tem shape {sample.shape}"
        )

    if sample.shape[0] != train_x.shape[1]:
        raise ValueError(
            f"sample deve ter {train_x.shape[1]} features "
            f"(igual a train_x), mas tem {sample.shape[0]}"
        )

    if k < 0:
        raise ValueError(
            f"k deve ser >= 0 (0 = usar todos os pontos), mas é {k}"
        )

    if k >= len(train_x):
        raise ValueError(
            f"k ({k}) deve ser menor que o número de amostras de treino "
            f"({len(train_x)})"
        )