"""
evaluation/curves/_test_all.py
"""

import numpy as np

try:
    from .roc import compute_roc_curve
    from .precision_recall import compute_pr_curve
except ImportError:
    from evaluation.curves.roc import compute_roc_curve
    from evaluation.curves.precision_recall import compute_pr_curve

def test_all_curves():
    """
    Teste rápido de sanity check do módulo curves.
    """

    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_score = np.array([0.1, 0.4, 0.35, 0.8, 0.9, 0.2])

    roc = compute_roc_curve(y_true, y_score)
    pr = compute_pr_curve(y_true, y_score)

    assert "fpr" in roc
    assert "tpr" in roc

    assert "precision" in pr
    assert "recall" in pr

    print("OK: curves module working")


if __name__ == "__main__":
    test_all_curves()