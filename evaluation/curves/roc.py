from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .common import _normalize, _check_binary


def compute_roc_curve(
    y_true: NDArray,
    y_score: NDArray,
) -> dict:
    """
    Calcula ROC curve manualmente.
    Retorna pontos da curva.
    """

    y_true = _normalize(y_true).astype(int)
    y_score = _normalize(y_score).astype(float)

    _check_binary(y_true)

    # ordenar por score desc
    order = np.argsort(-y_score)

    y_true = y_true[order]

    tp = 0
    fp = 0

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    fpr = [0.0]
    tpr = [0.0]
    thresholds = [float("inf")]

    for i in range(len(y_true)):

        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1

        fpr.append(fp / N if N else 0.0)
        tpr.append(tp / P if P else 0.0)
        thresholds.append(y_score[order[i]])

    fpr.append(1.0)
    tpr.append(1.0)
    thresholds.append(float("-inf"))

    return {
        "fpr": np.array(fpr),
        "tpr": np.array(tpr),
        "thresholds": np.array(thresholds),
    }