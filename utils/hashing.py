"""
Geração de hashes determinísticos para configs, combinações de hiperparâmetros
e chaves de execução (run). O hash é a base do sistema de checkpoint: uma
combinação já executada tem hash já presente nos resultados e é pulada.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def _canonical(obj: Any) -> Any:
    """Converte para uma forma serializável e ordenada de forma determinística."""
    if isinstance(obj, Mapping):
        return {str(k): _canonical(obj[k]) for k in sorted(obj.keys(), key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, float):
        # evita que 0.1 vs 0.10000000000000001 gerem hashes diferentes
        return round(obj, 10)
    return obj


def stable_hash(payload: Mapping[str, Any], length: int = 16) -> str:
    """
    Gera um hash SHA-256 truncado, determinístico e estável entre execuções,
    a partir de um dicionário (config, hiperparâmetros, seed, fold, etc).
    """
    canonical = _canonical(payload)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return digest[:length]


def run_identity(
    model: str,
    dataset: str,
    hyperparameters: Mapping[str, Any],
    seed: int,
    fold: int,
    repetition: int,
) -> str:
    """Hash único que identifica uma execução específica (usado no checkpoint)."""
    return stable_hash(
        {
            "model": model,
            "dataset": dataset,
            "hyperparameters": hyperparameters,
            "seed": seed,
            "fold": fold,
            "repetition": repetition,
        }
    )
