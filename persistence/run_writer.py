"""
Escrita atomica dos resultados de cada execucao (run).

Cada combinacao de hiperparametros x fold x repeticao gera IMEDIATAMENTE
um arquivo run_XXXX.json dentro de results/<model>/<experiment_name>/. A
escrita e atomica: grava-se em um arquivo temporario no mesmo diretorio e
depois faz-se os.replace (rename atomico no nivel do SO), garantindo que
nunca exista um JSON parcialmente escrito no disco - mesmo que o processo
seja interrompido no meio da escrita.

Mantem o mesmo layout de diretorios ja usado pelos resultados existentes
(results/<model>/<experiment_name>/) para nao invalidar checkpoints
anteriores.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _run_dir(model_name: str, experiment_name: str) -> Path:
    d = RESULTS_DIR / model_name / experiment_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def next_run_path(model_name: str, experiment_name: str) -> Path:
    d = _run_dir(model_name, experiment_name)
    existing = sorted(d.glob("run_*.json"))
    next_idx = len(existing) + 1
    return d / f"run_{next_idx:04d}.json"


def write_run_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Grava o payload em `path` de forma atomica."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)  # rename atomico
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_all_runs(model_name: str, experiment_name: str) -> list:
    d = _run_dir(model_name, experiment_name)
    runs = []
    for p in sorted(d.glob("run_*.json")):
        with open(p, "r", encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


def write_summary(model_name: str, experiment_name: str, summary: Dict[str, Any]) -> Path:
    d = _run_dir(model_name, experiment_name)
    path = d / "summary.json"
    write_run_atomic(path, summary)
    return path


def write_test_results(model_name: str, experiment_name: str, results: Dict[str, Any]) -> Path:
    d = _run_dir(model_name, experiment_name)
    path = d / "test_results.json"
    write_run_atomic(path, results)
    return path
