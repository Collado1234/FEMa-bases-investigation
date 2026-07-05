"""
Facade: ÚNICO ponto de entrada público do framework. Orquestra dados,
modelo, tuning, treinamento, seleção e avaliação final — mas não
implementa nenhuma dessas responsabilidades, apenas as encadeia.

Uso:
    from experiments.run_experiment import run_experiment
    run_experiment(config_path="configs/fema.yaml")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..io.datasets import get_dataset
from evaluation.evaluator import evaluate_on_test
from experiments.config_schema import ExperimentConfig, load_experiment_config
from models.registry import get_model
from selection.selector import select_best_config
from training.trainer import run_tuning_loop
from tuning.registry import get_tuning_strategy
from utils.logging_config import get_logger

logger = get_logger("experiments.run_experiment")


def run_experiment(config_path: str) -> Dict[str, Any]:
    config = load_experiment_config(config_path)
    return _run(config)


def run_experiment_by_name(model: str, dataset: str, config: str = "baseline", configs_dir: str = "configs") -> Dict[str, Any]:
    """Conveniência equivalente a run_experiment(model=..., dataset=..., config=...)
    pedida na especificação: resolve configs/<model>.yaml e sobrepõe
    model/dataset/experiment_name explicitamente."""
    path = Path(configs_dir) / f"{model}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config não encontrada: {path}")
    experiment_config = load_experiment_config(str(path))
    experiment_config = _override(experiment_config, model=model, dataset=dataset, experiment_name=config)
    return _run(experiment_config)


def _override(config: ExperimentConfig, **overrides) -> ExperimentConfig:
    from dataclasses import replace

    return replace(config, **overrides)


def _run(config: ExperimentConfig) -> Dict[str, Any]:
    logger.info("=== Iniciando experimento: model=%s dataset=%s experiment=%s ===",
                config.model, config.dataset, config.experiment_name)

    # Fase 1-2: dados e modelo (via registry — pipeline não conhece implementação)
    data = get_dataset(config.dataset)
    model_cls = get_model(config.model)

    # Fase 3: tuning + validação cruzada, com checkpoint automático
    if config.fixed_hyperparameters is not None:
        combinations = [config.fixed_hyperparameters]
    else:
        strategy = get_tuning_strategy(config.tuning_strategy, n_iter=config.tuning_n_iter)
        search_space = model_cls.get_search_space()
        combinations = strategy.generate_combinations(search_space, seed=config.seed)

    run_tuning_loop(
        model_cls=model_cls,
        model_name=config.model,
        dataset_name=config.dataset,
        experiment_name=config.experiment_name,
        combinations=combinations,
        X=data.X_train, y=data.y_train,
        metrics_names=config.metrics,
        cv_strategy=config.cv_strategy,
        n_splits=config.n_splits,
        n_repeats=config.n_repeats,
        seed=config.seed,
        results_root=config.output_dir,
    )

    # Fase 4: seleção do melhor modelo -> summary.json
    best = select_best_config(
        results_root=config.output_dir,
        model=config.model,
        experiment_name=config.experiment_name,
        ranking_metric=config.ranking_metric,
        higher_is_better=config.higher_is_better,
    )
    logger.info("Melhor configuração: %s", best["hyperparameters"])

    # Fase 5: avaliação final única em X_test -> test_results.json
    if data.X_test is None or data.y_test is None:
        logger.warning("Dataset '%s' não possui X_test/y_test — avaliação final pulada.", config.dataset)
        return {"summary_best": best, "test_results": None}

    test_results = evaluate_on_test(
        model_cls=model_cls,
        model_name=config.model,
        dataset_name=config.dataset,
        experiment_name=config.experiment_name,
        best_hyperparameters=best["hyperparameters"],
        X_train=data.X_train, y_train=data.y_train,
        X_val=data.X_val, y_val=data.y_val,
        X_test=data.X_test, y_test=data.y_test,
        metrics_names=config.metrics,
        seed=config.seed,
        results_root=config.output_dir,
        include_val_in_final_train=config.include_val_in_final_train,
    )

    logger.info("=== Experimento concluído: %s ===", {k: v for k, v in test_results["metrics"].items() if v is not None})
    return {"summary_best": best, "test_results": test_results}


# alias exigido pela especificação original: run_experiment(model=, dataset=, config=)
def run_experiment_flexible(model: str = None, dataset: str = None, config: str = "baseline", config_path: str = None):
    if config_path:
        return run_experiment(config_path)
    if model and dataset:
        return run_experiment_by_name(model=model, dataset=dataset, config=config)
    raise ValueError("Informe config_path OU (model e dataset).")
