"""Agrega todos os run_*.json de um experimento e escreve summary.json com
o ranking completo, médias, desvios-padrão e a configuração vencedora."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

from utils.hashing import stable_hash


def build_summary(
    results_root: str,
    model: str,
    experiment_name: str,
    ranking_metric: str,
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    directory = Path(results_root) / model / experiment_name
    runs: List[Dict[str, Any]] = []
    for run_file in sorted(directory.glob("run_*.json")):
        with open(run_file, "r", encoding="utf-8") as fh:
            runs.append(json.load(fh))

    if not runs:
        raise RuntimeError(f"Nenhum run encontrado em {directory}")

    # agrupa por combinação de hiperparâmetros (ignorando fold/repetição)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for run in runs:
        key = stable_hash({"hyperparameters": run["hyperparameters"]})
        groups.setdefault(key, []).append(run)

    ranking = []
    for key, group_runs in groups.items():
        metric_values = [
            r["metrics"][ranking_metric]
            for r in group_runs
            if r["metrics"].get(ranking_metric) is not None
        ]
        times = [r["execution_time_seconds"] for r in group_runs]
        if not metric_values:
            continue
        ranking.append({
            "hyperparameters": group_runs[0]["hyperparameters"],
            "n_folds_evaluated": len(group_runs),
            f"mean_{ranking_metric}": round(statistics.mean(metric_values), 6),
            f"std_{ranking_metric}": round(statistics.pstdev(metric_values), 6) if len(metric_values) > 1 else 0.0,
            "mean_execution_time_seconds": round(statistics.mean(times), 4),
        })

    ranking.sort(key=lambda r: r[f"mean_{ranking_metric}"], reverse=higher_is_better)

    summary = {
        "model": model,
        "experiment": experiment_name,
        "ranking_metric": ranking_metric,
        "n_total_runs": len(runs),
        "n_hyperparameter_combinations": len(ranking),
        "ranking": ranking,
        "best_config": ranking[0] if ranking else None,
    }

    with open(directory / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)

    return summary
