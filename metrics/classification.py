"""
Implementacoes concretas de metricas de classificacao (via sklearn.metrics).

Todas sao "multiclass-aware": quando n_classes > 2, usam average='macro'
(nao ponderado por suporte, para nao mascarar desempenho ruim nas classes
minoritarias) e, no caso de roc_auc, multi_class='ovr'.
"""

from typing import Optional

import numpy as np
from sklearn import metrics as skm

def accuracy(y_true, y_pred, y_score, n_classes) -> Optional[float]:
    return float(skm.accuracy_score(y_true, y_pred))

def balanced_accuracy(y_true, y_pred, y_score, n_classes) -> Optional[float]:
    return float(skm.balanced_accuracy_score(y_true, y_pred))


def precision(y_true, y_pred, y_score, n_classes) -> Optional[float]:
    average = "binary" if n_classes == 2 else "macro"
    return float(skm.precision_score(y_true, y_pred, average=average, zero_division=0))


def recall(y_true, y_pred, y_score, n_classes) -> Optional[float]:
    average = "binary" if n_classes == 2 else "macro"
    return float(skm.recall_score(y_true, y_pred, average=average, zero_division=0))


def f1(y_true, y_pred, y_score, n_classes) -> Optional[float]:
    average = "binary" if n_classes == 2 else "macro"
    return float(skm.f1_score(y_true, y_pred, average=average, zero_division=0))


def mcc(y_true, y_pred, y_score, n_classes) -> Optional[float]:
    return float(skm.matthews_corrcoef(y_true, y_pred))


def per_class_report(y_true, y_pred) -> dict:
    """Metricas por classe (acuracia = recall daquela classe, alem de
    precision/recall/f1/support), indexadas pelo rotulo da classe (como
    string, para ficar serializavel em JSON). Complementa as metricas
    agregadas de cima (que sao macro/binary, uma unica media) com o
    detalhamento por classe pedido no protocolo experimental.

    Nota: "acuracia por classe" em problemas multiclasse e' equivalente a
    "recall daquela classe" (proporcao de amostras daquela classe
    corretamente classificadas) - nao ha' outra nocao de acuracia que se
    decomponha de forma independente por classe. Por isso os dois campos
    ("accuracy" e "recall") tem o mesmo valor abaixo; mantidos separados
    para deixar explicito que o requisito de "acuracia por classe" do
    protocolo foi atendido, e porque precision/f1 sao conceitualmente
    diferentes de recall.
    """
    labels = sorted(np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)])))
    cm = skm.confusion_matrix(y_true, y_pred, labels=labels)
    report = skm.classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    result = {}
    for i, label in enumerate(labels):
        support = int(cm[i].sum())
        correct = int(cm[i, i])
        class_key = str(label)
        class_report = report.get(class_key, {})
        result[class_key] = {
            "accuracy": float(correct / support) if support > 0 else None,
            "precision": float(class_report.get("precision")) if class_report else None,
            "recall": float(class_report.get("recall")) if class_report else None,
            "f1": float(class_report.get("f1-score")) if class_report else None,
            "support": support,
        }
    return result


def roc_auc(y_true, y_pred, y_score, n_classes) -> Optional[float]:
    if y_score is None or len(np.unique(y_true)) < 2:
        return None
    try:
        if n_classes == 2:
            proba = y_score[:, 1] if getattr(y_score, "ndim", 1) == 2 else y_score
            return float(skm.roc_auc_score(y_true, proba))
        return float(skm.roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
    except ValueError:
        # pode falhar se algum fold nao contiver todas as classes
        return None
