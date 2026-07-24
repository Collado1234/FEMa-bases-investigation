"""
Registro de metricas.

Para adicionar uma nova metrica: escreva uma funcao em
metrics/classification.py ou metrics/regression.py seguindo o contrato de
metrics/contracts.py, e registre-a aqui com um nome (e a direcao de
otimizacao em HIGHER_IS_BETTER). Nenhum outro modulo precisa ser alterado.
"""

from metrics import classification as clf
from metrics import regression as reg

_METRICS = {
    "accuracy": clf.accuracy,
    "balanced_accuracy": clf.balanced_accuracy,
    "precision": clf.precision,
    "recall": clf.recall,
    "f1": clf.f1,
    "roc_auc": clf.roc_auc,
    "mcc": clf.mcc,
    "mae": reg.mae,
    "mse": reg.mse,
    "rmse": reg.rmse,
    "r2": reg.r2,
    "mape": reg.mape,
}

# usado por persistence/summary_builder.py para saber se, ao rankear pela
# metrica primaria, um valor maior e melhor (ex: f1) ou pior (ex: mae).
HIGHER_IS_BETTER = {
    "accuracy": True,
    "balanced_accuracy": True,
    "precision": True,
    "recall": True,
    "f1": True,
    "roc_auc": True,
    "mcc": True,
    "mae": False,
    "mse": False,
    "rmse": False,
    "r2": True,
    "mape": False,
}


# Subconjuntos de _METRICS por tipo de tarefa - fonte unica de verdade
# para "todas as metricas registradas" de classificacao/regressao. Usado
# por pipeline/run_model.py (defaults de metric_names por contexto) e por
# reporting/compare_bases.py + tests/ (para garantir que a comparacao e os
# testes cobrem TODAS as metricas relevantes, nao um subconjunto arbitrario
# escolhido a mao em cada lugar que precisa da lista).
CLASSIFICATION_METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "mcc"]
REGRESSION_METRICS = ["mae", "mse", "rmse", "r2", "mape"]


def metrics_for_context(context: str) -> list:
    if context == "classifier":
        return list(CLASSIFICATION_METRICS)
    if context == "regressor":
        return list(REGRESSION_METRICS)
    raise ValueError(f"Contexto '{context}' desconhecido. Use 'classifier' ou 'regressor'.")


def get_metric_fn(metric_name: str):
    if metric_name not in _METRICS:
        raise ValueError(f"Metrica '{metric_name}' desconhecida. Disponiveis: {list(_METRICS.keys())}")
    return _METRICS[metric_name]


def is_higher_better(metric_name: str) -> bool:
    return HIGHER_IS_BETTER.get(metric_name, True)


def compute_all(metric_names, y_true, y_pred, y_score, n_classes) -> dict:
    """Calcula todas as metricas solicitadas, retornando um dict
    {nome: valor|None}. Uma metrica que falhar (ex: excecao inesperada)
    grava None e nao interrompe as demais."""
    results = {}
    for name in metric_names:
        fn = get_metric_fn(name)
        try:
            results[name] = fn(y_true, y_pred, y_score, n_classes)
        except Exception:
            results[name] = None
    return results


def available_metrics():
    return list(_METRICS.keys())