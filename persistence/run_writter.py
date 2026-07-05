"""
Grava cada execução (uma combinação de hiperparâmetros x um fold x uma
repetição) IMEDIATAMENTE em disco, em results/<model>/<experiment>/run_XXXX.json.
Nunca sobrescreve: o nome do arquivo é sequencial e o conteúdo carrega o
hash de identidade do run (combo_hash), que é a chave usada pelo
checkpoint para saber o que já foi feito.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict


def experiment_dir(results_root: str, model: str, experiment_name: str) -> Path:
    path = Path(results_root) / model / experiment_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _next_sequence_number(directory: Path) -> int:
    existing = sorted(directory.glob("run_*.json"))
    if not existing:
        return 1
    last = existing[-1].stem  # "run_0007"
    return int(last.split("_")[1]) + 1


def write_run(results_root: str, model: str, experiment_name: str, record: Dict[str, Any]) -> Path:
    """Escreve um run_XXXX.json de forma atômica (write + rename), para que
    uma interrupção no meio da escrita nunca deixe um JSON corrompido."""
    directory = experiment_dir(results_root, model, experiment_name)
    seq = _next_sequence_number(directory)
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
