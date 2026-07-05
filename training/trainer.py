"""
Template Method: run_tuning_loop define o esqueleto fixo (para cada
combinação de hiperparâmetros, para cada fold: checar checkpoint, treinar,
avaliar, persistir), delegando a parte variável (como o modelo treina)
inteiramente para BaseModel. Nenhum código aqui conhece FEMa, CNN, etc.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Type

import numpy as np

from cross_validation.strategies import get_cv_splitter, iterate_folds
from metrics.registry import compute_all
from models.base import BaseModel
from persistence.checkpoint_manager import build_completed_index
from ..persistence.run_writter import write_run
from reproducibility.seed_manager import set_global_seed
from utils.hardware_info import get_code_version, get_hardware_info
from utils.hashing import run_identity
from utils.logging_config import get_logger

logger = get_logger("training.trainer")


def run_tuning_loop(
    model_cls: Type[BaseModel],
    model_name: str,
    dataset_name: str,
    experiment_name: str,
    combinations: List[Dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    metrics_names: List[str],
    cv_strategy: str,
    n_splits: int,
    n_repeats: int,
    seed: int,
    results_root: str,
) -> None:
    completed = build_completed_index(results_root, model_name, experiment_name)
    logger.info(
        "Experimento '%s/%s': %d combinações, %d runs já concluídos (checkpoint)",
        model_name, experiment_name, len(combinations), len(completed),
    )

    splitter = get_cv_splitter(cv_strategy, n_splits=n_splits, seed=seed, n_repeats=n_repeats)
    hardware_info = get_hardware_info()
    code_version = get_code_version()

    for hyperparameters in combinations:
        for fold, repetition, X_tr, y_tr, X_va, y_va in iterate_folds(splitter, X, y, n_splits):
            combo_hash = run_identity(model_name, dataset_name, hyperparameters, seed, fold, repetition)
            if combo_hash in completed:
                continue

            set_global_seed(seed)
            start = time.perf_counter()
            model = model_cls.create_model(hyperparameters)
            model.fit(X_tr, y_tr, X_va, y_va)

            y_pred = model.predict(X_va)
            y_prob = model.predict_proba(X_va)
            elapsed = time.perf_counter() - start

            metric_values = compute_all(metrics_names, y_va, y_pred, y_prob)

            record = {
                "run_type": "tuning",
                "combo_hash": combo_hash,
                "model": model_name,
                "dataset": dataset_name,
                "experiment": experiment_name,
                "hyperparameters": hyperparameters,
                "seed": seed,
                "fold": fold,
                "repetition": repetition,
                "metrics": metric_values,
                "execution_time_seconds": round(elapsed, 4),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "hardware": hardware_info,
                "code_version": code_version,
            }
            path = write_run(results_root, model_name, experiment_name, record)
            completed.add(combo_hash)
            logger.info(
                "run salvo: %s | fold=%d rep=%d | %s",
                path.name, fold, repetition,
                {k: v for k, v in metric_values.items() if v is not None},
            )
