"""Utilitários pequenos e sem estado compartilhados pelo pipeline:
logging, controle de seed e hash determinístico (usado no checkpoint).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
from typing import Any, Mapping

import numpy as np


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_global_seed(seed: int) -> None:
    """Fixa a seed nas fontes de aleatoriedade conhecidas. Chamar antes de
    cada fit(), não só uma vez no início, para o resultado de cada run não
    depender da ordem de execução."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _canonical(obj: Any) -> Any:
    """Forma serializável e ordenada de forma determinística (base do hash)."""
    if isinstance(obj, Mapping):
        return {str(k): _canonical(obj[k]) for k in sorted(obj.keys(), key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 10)  # evita 0.1 vs 0.10000000000000001
    return obj


def stable_hash(payload: Mapping[str, Any]) -> str:
    """Hash SHA-256 truncado e determinístico a partir de um dict."""
    encoded = json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def run_identity(model: str, dataset: str, hyperparameters: Mapping[str, Any],
                  seed: int, fold: int, repetition: int) -> str:
    """Hash único de uma execução específica — chave do checkpoint."""
    return stable_hash({
        "model": model, "dataset": dataset, "hyperparameters": hyperparameters,
        "seed": seed, "fold": fold, "repetition": repetition,
    })
