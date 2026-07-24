"""
Pipeline unico.

MUDANCA ARQUITETURAL: existiam um unico ponto de entrada (run_model) que
tratava "modelo" (fema/logreg/mlp) como eixo estrutural. Agora existem
DOIS pontos de entrada, refletindo que o projeto tem dois tipos de
experimento com objetivos completamente diferentes:

- run_basis_experiment(context, basis, dataset, ...): roda o FEMa com o
  contexto (classifier|regressor) e a base de interpolacao FIXOS. Este e'
  o pipeline central da pesquisa - cada chamada corresponde a UMA celula
  da matriz (contexto x base) sendo comparada. Persiste em
  results/<context>/<basis>/<dataset>/<experiment_name>/.

- run_all_bases(context, dataset, ...): roda run_basis_experiment para
  TODAS as bases registradas em core.Basis.available() - e' o modo de uso
  mais comum do projeto (comparar todas as bases de uma vez).

- run_baseline_experiment(model_name, dataset, ...): roda um baseline
  EXTERNO (logreg, knn) - NAO faz parte da comparacao de bases, serve so'
  de referencia metodologica (ex: "o FEMa com a melhor base e' competitivo
  com um baseline simples?"). Persiste, isolado, em
  results/external_baselines/<model_name>/<dataset>/<experiment_name>/.

Os dois compartilham o mesmo motor (_execute): carregar dados ->
checkpoint -> tuning (grid/random search) + CV -> summary -> retreino
final -> avaliacao no teste. O motor nao sabe se esta' rodando FEMa ou
baseline - so' fala com plugins atraves do contrato definido em
models/base.py (ModelPlugin), que continua existindo e sendo usado por
ambos os tipos de experimento.

Uso:
    from pipeline.run_model import run_basis_experiment, run_all_bases, run_baseline_experiment

    run_basis_experiment(context="classifier", basis="shepard", dataset="fetal_health")
    run_all_bases(context="classifier", dataset="fetal_health")
    run_baseline_experiment(model_name="knn", dataset="fetal_health")

    # ou via arquivo de config:
    from pipeline.run_model import run_from_experiment_file
    run_from_experiment_file("configs/experiments/fema_baseline.yaml")
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from datasets.loader import load_dataset
from evaluation.cv_evaluator import evaluate_fold
from evaluation.test_evaluator import evaluate_on_test
from models.base import ModelPlugin
from models.fema import FEMaPlugin, available_bases
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

DEFAULT_CLASSIFICATION_METRICS = ["accuracy", "f1", "balanced_accuracy", "mcc"]
DEFAULT_REGRESSION_METRICS = ["mae", "rmse", "r2"]


def _load_yaml(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _execute(
    scope: Tuple[str, ...],
    plugin: ModelPlugin,
    dataset: str,
    metric_names: list,
    ranking_metric: str,
    higher_is_better: Optional[bool],
    cv_strategy: str,
    n_splits: int,
    n_repeats: int,
    tuning_strategy: str,
    tuning_n_iter: int,
    master_seed: int,
    fixed_hyperparameters: Optional[Dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
    run_final_test: bool,
    summary_extra_fields: Dict[str, Any],
) -> Dict[str, Any]:
    """Motor comum de tuning + CV + summary + retreino final + avaliacao no
    teste. Nao sabe se `plugin` e' FEMa ou baseline externo - so' fala com
    o contrato ModelPlugin. `scope` e' o caminho de diretorios sob
    results/ onde tudo deste experimento sera' persistido (ver
    persistence/run_writer.py). Idempotente: pode ser chamado
    repetidamente com os mesmos argumentos - combinacoes ja executadas
    (presentes em results/<scope...>/) sao puladas."""
    set_global_seed(master_seed)

    logger.info(
        f"=== run: scope=results/{'/'.join(scope)} dataset={dataset} tuning={tuning_strategy} "
        f"n_splits={n_splits} n_repeats={n_repeats} ==="
    )

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

    completed_keys = get_completed_keys(*scope)
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

            fold_seed = derive_seed(master_seed, *scope, cid, f"repeat={repeat_idx}", f"fold={fold_idx}")

            with timer() as t:
                metric_values = evaluate_fold(
                    plugin, params, X_tr, y_tr, X_val, y_val, metric_names, random_state=fold_seed
                )

            payload = {
                **summary_extra_fields,
                "dataset": dataset,
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

            run_path = next_run_path(*scope)
            write_run_atomic(run_path, payload)
            n_executed += 1

    logger.info(f"Tuning concluido: {n_executed} runs novos, {n_skipped} pulados (ja existiam).")

    summary = build_summary(
        *scope,
        primary_metric=ranking_metric,
        higher_is_better=higher_is_better,
        extra_fields={**summary_extra_fields, "dataset": dataset},
    )
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
            **summary_extra_fields,
            "dataset": dataset,
            "best_hyperparameters": best_params,
            "n_train_total": len(X_train),
            "n_test": len(data.X_test),
            "metrics": test_metrics,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        write_test_results(*scope, results=test_payload)
        logger.info(f"Avaliacao final no teste concluida: {test_metrics}")
        result["test_results"] = test_payload

    return result


def run_basis_experiment(
    context: str,
    basis: str,
    dataset: str,
    experiment_name: str = "baseline",
    metric_names: Optional[list] = None,
    ranking_metric: Optional[str] = None,
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
    """Roda (ou continua) o FEMa com um contexto e uma base fixos.
    Resultados: results/<context>/<basis>/<dataset>/<experiment_name>/.

    metric_names/ranking_metric usam defaults de classificacao ou
    regressao de acordo com `context`, se nao forem informados
    explicitamente."""
    plugin = FEMaPlugin(context=context, basis=basis)

    is_classifier = context == "classifier"
    default_metrics = DEFAULT_CLASSIFICATION_METRICS if is_classifier else DEFAULT_REGRESSION_METRICS
    default_ranking = "f1" if is_classifier else "rmse"

    return _execute(
        scope=(context, basis, dataset, experiment_name),
        plugin=plugin,
        dataset=dataset,
        metric_names=metric_names or default_metrics,
        ranking_metric=ranking_metric or default_ranking,
        higher_is_better=higher_is_better,
        cv_strategy=cv_strategy,
        n_splits=n_splits,
        n_repeats=n_repeats,
        tuning_strategy=tuning_strategy,
        tuning_n_iter=tuning_n_iter,
        master_seed=master_seed,
        fixed_hyperparameters=fixed_hyperparameters,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        run_final_test=run_final_test,
        summary_extra_fields={"context": context, "basis": basis, "experiment": experiment_name},
    )


def run_all_bases(context: str, dataset: str, bases: Optional[list] = None, **kwargs) -> Dict[str, Dict[str, Any]]:
    """Roda run_basis_experiment para cada base em `bases` (por padrao,
    TODAS as registradas em core.Basis.available()). E' o modo de uso mais
    comum do projeto: comparar todas as bases de uma vez para um mesmo
    contexto/dataset. Retorna {basis: resultado}."""
    bases = bases or available_bases()
    results = {}
    for basis in bases:
        logger.info(f"--- Iniciando base '{basis}' ({context}/{dataset}) ---")
        results[basis] = run_basis_experiment(context=context, basis=basis, dataset=dataset, **kwargs)
    return results


def run_baseline_experiment(
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
    """Roda um baseline externo (logreg, knn) - NAO faz parte da
    comparacao de bases, serve so' de referencia metodologica. Persiste,
    isolado, em results/external_baselines/<model_name>/<dataset>/<experiment_name>/."""
    plugin = get_model_plugin(model_name)

    return _execute(
        scope=("external_baselines", model_name, dataset, experiment_name),
        plugin=plugin,
        dataset=dataset,
        metric_names=metric_names or DEFAULT_CLASSIFICATION_METRICS,
        ranking_metric=ranking_metric,
        higher_is_better=higher_is_better,
        cv_strategy=cv_strategy,
        n_splits=n_splits,
        n_repeats=n_repeats,
        tuning_strategy=tuning_strategy,
        tuning_n_iter=tuning_n_iter,
        master_seed=master_seed,
        fixed_hyperparameters=fixed_hyperparameters,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        run_final_test=run_final_test,
        summary_extra_fields={"model": model_name, "experiment": experiment_name},
    )


def run_from_experiment_file(experiment_path) -> Dict[str, Any]:
    """Le configs/base.yaml para os defaults e o YAML do experimento para
    os parametros especificos.

    Roteamento pelo schema do YAML:
      - contem 'basis' (uma base) ou 'all_bases: true' -> experimento do
        FEMa (run_basis_experiment / run_all_bases). 'context' e' opcional,
        default 'classifier'.
      - contem 'model' -> baseline externo (run_baseline_experiment).
    """
    base_cfg = _load_yaml(BASE_CONFIG_PATH)
    exp_cfg = _load_yaml(experiment_path)

    cv_cfg = base_cfg.get("cross_validation", {})
    tuning_cfg = base_cfg.get("tuning", {})

    common = dict(
        dataset=exp_cfg["dataset"],
        experiment_name=exp_cfg.get("experiment_name", "baseline"),
        metric_names=exp_cfg.get("metrics", base_cfg.get("metrics")),
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
    ranking_metric = exp_cfg.get("ranking_metric", base_cfg.get("ranking_metric"))

    if "basis" in exp_cfg or exp_cfg.get("all_bases"):
        context = exp_cfg.get("context", "classifier")
        if exp_cfg.get("all_bases"):
            return run_all_bases(context=context, ranking_metric=ranking_metric, **common)
        return run_basis_experiment(context=context, basis=exp_cfg["basis"], ranking_metric=ranking_metric, **common)

    if "model" in exp_cfg:
        return run_baseline_experiment(
            model_name=exp_cfg["model"], ranking_metric=ranking_metric or "f1", **common
        )

    raise ValueError(
        f"Config '{experiment_path}' precisa definir 'basis' (ou 'all_bases: true') "
        "para experimentos do FEMa, ou 'model' para baselines externos."
    )