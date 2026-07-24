"""Gráficos de avaliação, opcionais — o pipeline funciona sem chamá-los.
Um arquivo só, sem classes de estilo separadas; a aparência é só um dict.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STYLE = dict(figsize=(7, 5), dpi=130, grid_alpha=0.3)


def _figure():
    fig, ax = plt.subplots(figsize=STYLE["figsize"], dpi=STYLE["dpi"])
    ax.grid(True, alpha=STYLE["grid_alpha"], linestyle="--")
    return fig, ax


def _finish(fig, filename, show):
    fig.tight_layout()
    if filename:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def confusion_matrix(cm, labels, filename=None, show=True):
    fig, ax = _figure()
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    _finish(fig, filename, show)


def roc_curve(fpr, tpr, auc=None, filename=None, show=True):
    fig, ax = _figure()
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.4f})" if auc is not None else "ROC")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_title("ROC Curve"); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend()
    _finish(fig, filename, show)


def precision_recall_curve(recall, precision, filename=None, show=True):
    fig, ax = _figure()
    ax.plot(recall, precision)
    ax.set_title("Precision-Recall Curve"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    _finish(fig, filename, show)


def actual_vs_predicted(y_true, y_pred, filename=None, show=True):
    fig, ax = _figure()
    ax.scatter(y_true, y_pred, alpha=0.75)
    lo, hi = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], "--", color="gray", label="Ideal")
    ax.set_title("Actual vs Predicted"); ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.legend()
    _finish(fig, filename, show)
