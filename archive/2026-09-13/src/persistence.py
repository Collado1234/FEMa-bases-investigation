"""Persistência dos resultados em disco (JSON), como uma pequena "base de
dados" de arquivos: cada execução (combinação x fold x repetição) vira um
results/<model>/<experiment>/run_XXXX.json; o próprio sistema de arquivos é
o checkpoint (um run com o mesmo combo_hash já presente é pulado); e
summary.json agrega tudo com médias/desvios e escolhe a melhor combinação.
"""
from __future__ import annotations

import json
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set


def experiment_dir(results_root: str, model: str, experiment_name: str) -> Path:
    path = Path(results_root) / model / experiment_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_completed_hashes(results_root: str, model: str, experiment_name: str) -> Set[str]:
    """Escaneia os run_*.json já existentes para saber o que pular (checkpoint)."""
    directory = Path(results_root) / model / experiment_name
    if not directory.exists():
        return set()
    completed = set()
    for run_file in directory.glob("run_*.json"):
        try:
            with open(run_file, "r", encoding="utf-8") as fh:
                combo_hash = json.load(fh).get("combo_hash")
            if combo_hash:
                completed.add(combo_hash)
        except (json.JSONDecodeError, OSError):
            continue  # arquivo corrompido/parcial de uma interrupção anterior
    return completed


def write_run(results_root: str, model: str, experiment_name: str, record: Dict[str, Any]) -> Path:
    """Escreve run_XXXX.json de forma atômica (write + rename)."""
    directory = experiment_dir(results_root, model, experiment_name)
    existing = sorted(directory.glob("run_*.json"))
    seq = int(existing[-1].stem.split("_")[1]) + 1 if existing else 1
    final_path = directory / f"run_{seq:04d}.json"

    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp_path, final_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return final_path


def build_summary(results_root: str, model: str, experiment_name: str,
                   ranking_metric: str, higher_is_better: bool = True) -> Dict[str, Any]:
    """Agrega os runs de tuning por combinação de hiperparâmetros e ranqueia."""
    directory = Path(results_root) / model / experiment_name
    runs: List[Dict[str, Any]] = [
        json.loads(f.read_text(encoding="utf-8")) for f in sorted(directory.glob("run_*.json"))
    ]
    if not runs:
        raise RuntimeError(f"Nenhum run encontrado em {directory}")

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for run in runs:
        key = json.dumps(run["hyperparameters"], sort_keys=True, default=str)
        groups.setdefault(key, []).append(run)

    ranking = []
    for group_runs in groups.values():
        values = [r["metrics"].get(ranking_metric) for r in group_runs if r["metrics"].get(ranking_metric) is not None]
        if not values:
            continue
        ranking.append({
            "hyperparameters": group_runs[0]["hyperparameters"],
            "n_folds_evaluated": len(group_runs),
            f"mean_{ranking_metric}": round(statistics.mean(values), 6),
            f"std_{ranking_metric}": round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
            "mean_execution_time_seconds": round(statistics.mean(r["execution_time_seconds"] for r in group_runs), 4),
        })
    ranking.sort(key=lambda r: r[f"mean_{ranking_metric}"], reverse=higher_is_better)

    summary = {
        "model": model, "experiment": experiment_name, "ranking_metric": ranking_metric,
        "n_total_runs": len(runs), "n_hyperparameter_combinations": len(ranking),
        "ranking": ranking, "best_config": ranking[0] if ranking else None,
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return summary


def select_best_config(results_root: str, model: str, experiment_name: str,
                        ranking_metric: str, higher_is_better: bool = True) -> Dict[str, Any]:
    summary = build_summary(results_root, model, experiment_name, ranking_metric, higher_is_better)
    if summary["best_config"] is None:
        raise RuntimeError("Nenhuma combinação de hiperparâmetros produziu métricas válidas.")
    return summary["best_config"]


def write_test_results(results_root: str, model: str, experiment_name: str, record: Dict[str, Any]) -> Path:
    directory = experiment_dir(results_root, model, experiment_name)
    path = directory / "test_results.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
