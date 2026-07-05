"""Seleciona a melhor combinação de hiperparâmetros a partir do
summary.json já construído — responsabilidade separada de
persistence/summary_builder.py (que só agrega e calcula estatísticas)."""
from __future__ import annotations

from typing import Any, Dict

from persistence.summary_builder import build_summary


def select_best_config(
    results_root: str,
    model: str,
    experiment_name: str,
    ranking_metric: str,
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    summary = build_summary(results_root, model, experiment_name, ranking_metric, higher_is_better)
    if summary["best_config"] is None:
        raise RuntimeError("Nenhuma combinação de hiperparâmetros produziu métricas válidas.")
    return summary["best_config"]
