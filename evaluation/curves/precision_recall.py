from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .common import _normalize, _check_binary


def compute_pr_curve(
    y_true: NDArray,
    y_score: NDArray,
) -> dict:
    """
    Precision-Recall curve manual.
    """

    y_true = _normalize(y_true).astype(int)
    y_score = _normalize(y_score).astype(float)

    _check_binary(y_true)

    order = np.argsort(-y_score)
    y_true = y_true[order]

    tp = 0
    fp = 0

    P = np.sum(y_true == 1)

    precision = []
    recall = []
    thresholds = []

    for i in range(len(y_true)):

        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / P if P else 0.0

        precision.append(prec)
        recall.append(rec)
        thresholds.append(y_score[order[i]])

    return {
        "precision": np.array(precision),
        "recall": np.array(recall),
        "thresholds": np.array(thresholds),
    }