"""
Sistema de checkpoint: ao iniciar um experimento, escaneia
results/<model>/<experiment>/*.json já existentes, extrai o campo
"combo_hash" de cada um, e monta um índice em memória. Qualquer combinação
(hiperparâmetros + fold + repetição) cujo hash já esteja no índice é
pulada. O sistema de arquivos JÁ É o checkpoint — não há estado paralelo
que possa dessincronizar dos resultados reais.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Set


def build_completed_index(results_root: str, model: str, experiment_name: str) -> Set[str]:
    directory = Path(results_root) / model / experiment_name
    if not directory.exists():
        return set()

    completed: Set[str] = set()
    for run_file in directory.glob("run_*.json"):
        try:
            with open(run_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            combo_hash = data.get("combo_hash")
            if combo_hash:
                completed.add(combo_hash)
        except (json.JSONDecodeError, OSError):
            # arquivo corrompido/parcial (ex.: interrupção durante escrita
            # não-atômica de uma versão anterior) — ignora e deixa o
            # próximo run recriar essa combinação.
            continue
    return completed
