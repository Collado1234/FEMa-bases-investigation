"""Ponto de entrada público do framework: run_experiment(config_path) roda
o experimento inteiro (dados -> tuning com CV -> checkpoint -> summary.json
-> avaliação final em teste -> test_results.json).

Uso:
    from src.pipeline import run_experiment
    run_experiment("configs/fema.yaml")
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict

from .config import ExperimentConfig, load_experiment_config
from .cross_validation import iterate_folds
from .datasets import get_dataset
from .metrics import compute_metrics
from .models import create_model, get_search_space
from .persistence import load_completed_hashes, select_best_config, write_run, write_test_results
from .tuning import generate_combinations
from .utils import get_logger, run_identity, set_global_seed

logger = get_logger("src.pipeline")


def run_experiment(config_path: str) -> Dict[str, Any]:
    config = load_experiment_config(config_path)
    return _run(config)


def _run(config: ExperimentConfig) -> Dict[str, Any]:
    logger.info("=== Iniciando experimento: model=%s dataset=%s experiment=%s ===",
                config.model, config.dataset, config.experiment_name)

    data = get_dataset(config.dataset)

    combinations = (
        [config.fixed_hyperparameters] if config.fixed_hyperparameters is not None
        else generate_combinations(config.tuning_strategy, get_search_space(config.model),
                                    seed=config.seed, n_iter=config.tuning_n_iter)
    )

    _run_tuning_loop(config, combinations, data.X_train, data.y_train)

    best = select_best_config(config.output_dir, config.model, config.experiment_name,
                               config.ranking_metric, config.higher_is_better)
    logger.info("Melhor configuração: %s", best["hyperparameters"])

    if data.X_test is None or data.y_test is None:
        logger.warning("Dataset '%s' não possui X_test/y_test — avaliação final pulada.", config.dataset)
        return {"summary_best": best, "test_results": None}

    test_results = _evaluate_on_test(config, best["hyperparameters"], data)
    logger.info("=== Experimento concluído: %s ===",
                {k: v for k, v in test_results["metrics"].items() if v is not None})
    return {"summary_best": best, "test_results": test_results}


def _run_tuning_loop(config: ExperimentConfig, combinations, X, y) -> None:
    completed = load_completed_hashes(config.output_dir, config.model, config.experiment_name)
    logger.info("Experimento '%s/%s': %d combinações, %d runs já concluídos (checkpoint)",
                config.model, config.experiment_name, len(combinations), len(completed))

    for hyperparameters in combinations:
        for fold, repetition, X_tr, y_tr, X_va, y_va in iterate_folds(
                X, y, config.cv_strategy, config.n_splits, config.seed, config.n_repeats):
            combo_hash = run_identity(config.model, config.dataset, hyperparameters, config.seed, fold, repetition)
            if combo_hash in completed:
                continue

            set_global_seed(config.seed)
            start = time.perf_counter()
            model = create_model(config.model, hyperparameters)
            model.fit(X_tr, y_tr, X_va, y_va)
            y_pred = model.predict(X_va)
            y_score = _predict_score(model, X_va)
            elapsed = time.perf_counter() - start

            metric_values = compute_metrics(config.metrics, y_va, y_pred, y_score)
            record = {
                "run_type": "tuning", "combo_hash": combo_hash, "model": config.model,
                "dataset": config.dataset, "experiment": config.experiment_name,
                "hyperparameters": hyperparameters, "seed": config.seed,
                "fold": fold, "repetition": repetition, "metrics": metric_values,
                "execution_time_seconds": round(elapsed, 4),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            path = write_run(config.output_dir, config.model, config.experiment_name, record)
            completed.add(combo_hash)
            logger.info("run salvo: %s | fold=%d rep=%d | %s", path.name, fold, repetition,
                        {k: v for k, v in metric_values.items() if v is not None})


def _evaluate_on_test(config: ExperimentConfig, hyperparameters: Dict[str, Any], data) -> Dict[str, Any]:
    set_global_seed(config.seed)
    model = create_model(config.model, hyperparameters)
    model.fit(data.X_train, data.y_train, data.X_val, data.y_val)

    y_pred = model.predict(data.X_test)
    y_score = _predict_score(model, data.X_test)
    metric_values = compute_metrics(config.metrics, data.y_test, y_pred, y_score)

    record = {
        "run_type": "test", "model": config.model, "dataset": config.dataset,
        "experiment": config.experiment_name, "hyperparameters": hyperparameters,
        "seed": config.seed, "metrics": metric_values,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_test_results(config.output_dir, config.model, config.experiment_name, record)
    return record


def _predict_score(model, X):
    """Probabilidade prevista, se o modelo suportar: coluna da classe
    positiva em binário, matriz completa em multiclasse (auc_roc ovr)."""
    if not hasattr(model, "predict_proba"):
        return None
    try:
        proba = model.predict_proba(X)
        if proba is None:
            return None
        return proba[:, 1] if proba.shape[1] == 2 else proba
    except Exception:
        return None
