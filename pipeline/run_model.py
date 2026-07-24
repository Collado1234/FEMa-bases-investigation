"""
Pipeline unico.

Este e o unico ponto de entrada para rodar um experimento completo:
carregar dados -> checkpoint -> tuning (grid/random search) + CV -> summary
-> retreino final -> avaliacao no teste. Nao conhece nenhum algoritmo
especifico - so fala com plugins atraves do contrato definido em
models/base.py. Equivalente ao antigo src/pipeline.py, mas dividido nas
camadas datasets/models/metrics/tuning/persistence/training/evaluation,
no mesmo espirito do icd-project.

Uso:
    from pipeline.run_model import run_experiment
    run_experiment("configs/experiments/fema_baseline.yaml")

    # ou diretamente, sem arquivo de config:
    from pipeline.run_model import run_model
    run_model(model_name="fema", dataset="fetal_health")
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from datasets.loader import load_dataset
from evaluation.cv_evaluator import evaluate_fold
from evaluation.test_evaluator import evaluate_on_test
from models.registry import get_model_plugin
from persistence.checkpoint import get_completed_keys, is_done
from persistence.run_writer import next_run_path, write_run_atomic, write_test_results
from persistence.summary_builder import build_summary
from training.final_fit import fit_final_model
from tuning.cv_strategy import build_cv_splits
from tuning.grid_search import combo_id
from tuning.param_space import generate_combinations
from utils.logging_config import get_logger
from utils.seeding import derive_seed, set_global_seed
from utils.timing import timer

logger = get_logger("pipeline.run_model")

BASE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "base.yaml"

DEFAULT_METRICS = ["accuracy", "f1", "balanced_accuracy", "mcc"]


def _load_yaml(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_model(
    model_name: str,
    dataset: str,
    experiment_name: str = "baseline",
    metric_names: Optional[list] = None,
    ranking_metric: str = "f1",
    higher_is_better: Optional[bool] = None,
    cv_strategy: str = "repeated_stratified_kfold",
    n_splits: int = 5,
    n_repeats: int = 3,
    tuning_strategy: str = "random_search",
    tuning_n_iter: int = 20,
    master_seed: int = 42,
    fixed_hyperparameters: Optional[Dict[str, Any]] = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    run_final_test: bool = True,
) -> Dict[str, Any]:
    """Executa (ou continua) um experimento completo para um modelo.

    Idempotente: pode ser chamado repetidamente com os mesmos argumentos -
    combinacoes ja executadas (presentes em results/) sao puladas.
    """
    metric_names = metric_names or DEFAULT_METRICS
    set_global_seed(master_seed)

    logger.info(
        f"=== run_model: model={model_name} dataset={dataset} experiment={experiment_name} "
        f"tuning={tuning_strategy} n_splits={n_splits} n_repeats={n_repeats} ==="
    )

    plugin = get_model_plugin(model_name)

    data = load_dataset(dataset, seed=master_seed, val_ratio=val_ratio, test_ratio=test_ratio)
    X_train, y_train = data.X_train, data.y_train

    param_combinations = generate_combinations(
        plugin,
        strategy=tuning_strategy,
        seed=master_seed,
        n_iter=tuning_n_iter,
        fixed_hyperparameters=fixed_hyperparameters,
    )
    logger.info(f"{len(param_combinations)} combinacoes de hiperparametros a avaliar.")

    completed_keys = get_completed_keys(model_name, experiment_name)
    logger.info(f"{len(completed_keys)} runs ja existentes (checkpoint).")

    n_executed = 0
    n_skipped = 0

    for params in param_combinations:
        cid = combo_id(params)

        for repeat_idx, fold_idx, train_idx, val_idx in build_cv_splits(
            X_train, y_train, strategy=cv_strategy, n_splits=n_splits, n_repeats=n_repeats, master_seed=master_seed
        ):
            if is_done(completed_keys, cid, repeat_idx, fold_idx):
                n_skipped += 1
                continue

            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]

            fold_seed = derive_seed(master_seed, model_name, cid, f"repeat={repeat_idx}", f"fold={fold_idx}")

            with timer() as t:
                metric_values = evaluate_fold(
                    plugin, params, X_tr, y_tr, X_val, y_val, metric_names, random_state=fold_seed
                )

            payload = {
                "model": model_name,
                "dataset": dataset,
                "experiment": experiment_name,
                "combo_id": cid,
                "hyperparameters": params,
                "repeat_idx": repeat_idx,
                "fold_idx": fold_idx,
                "seed": fold_seed,
                "n_train_fold": len(train_idx),
                "n_val_fold": len(val_idx),
                "metrics": metric_values,
                "execution_time_seconds": t.elapsed_seconds,
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            }

            run_path = next_run_path(model_name, experiment_name)
            write_run_atomic(run_path, payload)
            n_executed += 1

    logger.info(f"Tuning concluido: {n_executed} runs novos, {n_skipped} pulados (ja existiam).")

    summary = build_summary(model_name, experiment_name, primary_metric=ranking_metric, higher_is_better=higher_is_better)
    logger.info(f"Melhor combinacao ({ranking_metric}): {summary['best_configuration']['combo_id']}")

    result = {"summary": summary, "test_results": None}

    if run_final_test:
        if data.X_test is None or data.y_test is None:
            logger.warning(f"Dataset '{dataset}' nao possui X_test/y_test - avaliacao final pulada.")
            return result

        best_params = summary["best_configuration"]["hyperparameters"]
        final_estimator = fit_final_model(
            plugin, best_params, X_train, y_train, master_seed=master_seed, X_val=data.X_val, y_val=data.y_val
        )

        test_metrics = evaluate_on_test(
            plugin, final_estimator, data.X_test, data.y_test, y_train_reference=y_train, metric_names=metric_names
        )

        test_payload = {
            "model": model_name,
            "dataset": dataset,
            "experiment": experiment_name,
            "best_hyperparameters": best_params,
            "n_train_total": len(X_train),
            "n_test": len(data.X_test),
            "metrics": test_metrics,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        write_test_results(model_name, experiment_name, test_payload)
        logger.info(f"Avaliacao final no teste concluida: {test_metrics}")
        result["test_results"] = test_payload

    return result


def run_from_experiment_file(experiment_path) -> Dict[str, Any]:
    """Le configs/base.yaml para os defaults e o YAML do experimento para os
    parametros especificos, e chama run_model()."""
    base_cfg = _load_yaml(BASE_CONFIG_PATH)
    exp_cfg = _load_yaml(experiment_path)

    cv_cfg = base_cfg.get("cross_validation", {})
    tuning_cfg = base_cfg.get("tuning", {})

    return run_model(
        model_name=exp_cfg["model"],
        dataset=exp_cfg["dataset"],
        experiment_name=exp_cfg.get("experiment_name", "baseline"),
        metric_names=exp_cfg.get("metrics", base_cfg.get("metrics")),
        ranking_metric=exp_cfg.get("ranking_metric", base_cfg.get("ranking_metric", "f1")),
        higher_is_better=exp_cfg.get("higher_is_better", base_cfg.get("higher_is_better")),
        cv_strategy=exp_cfg.get("cv_strategy", cv_cfg.get("strategy", "repeated_stratified_kfold")),
        n_splits=exp_cfg.get("n_splits", cv_cfg.get("n_splits", 5)),
        n_repeats=exp_cfg.get("n_repeats", cv_cfg.get("n_repeats", 3)),
        tuning_strategy=exp_cfg.get("tuning_strategy", tuning_cfg.get("strategy", "random_search")),
        tuning_n_iter=exp_cfg.get("tuning_n_iter", tuning_cfg.get("n_iter", 20)),
        master_seed=exp_cfg.get("seed", base_cfg.get("master_seed", 42)),
        fixed_hyperparameters=exp_cfg.get("fixed_hyperparameters"),
        run_final_test=exp_cfg.get("run_final_test", base_cfg.get("run_final_test", True)),
    )
