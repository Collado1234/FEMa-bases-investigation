"""Smoke test do pipeline completo, usando o dataset sintetico (rapido, nao
depende de CSVs externos). Valida que os resultados sao persistidos em
disco (run_*.json, summary.json, test_results.json), no caminho correto
(results/<context>/<basis>/... para o FEMa, results/external_baselines/...
para baselines), e que o checkpoint nao duplica execucoes ja concluidas.

Rodar com: python3 -m pytest tests/ -v
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from core import Basis
from datasets.registry import available_datasets
from models.fema import FEMaPlugin
from models.registry import available_models
from pipeline.run_model import run_baseline_experiment, run_basis_experiment

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def test_baselines_and_datasets_available():
    assert "logreg" in available_models()
    assert "knn" in available_models()
    assert "synthetic_demo" in available_datasets()


def test_shepard_basis_registered():
    assert "shepard" in Basis.available()


def test_fema_classifier_end_to_end():
    """FEMa e' o algoritmo principal do projeto - valida que ele treina/
    prediz de fato (nao so' que o pipeline generico roda), para o contexto
    classifier com a base shepard."""
    plugin = FEMaPlugin(context="classifier", basis="shepard")

    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [10, 10], [11, 11]], dtype=float)
    y_train = np.array([0, 0, 0, 1, 1, 1])
    X_test = np.array([[1.5, 2.5], [10.5, 10.5]])

    estimator = plugin.create_model({"k": 3, "z": 2.0}, random_state=42)
    estimator.fit(X_train, y_train)
    labels = plugin.predict(estimator, X_test)
    assert list(labels) == [0, 1]


def test_basis_experiment_persists_to_scoped_path():
    """Valida a reorganizacao de results/: cada base tem seu proprio
    diretorio, dentro do contexto, dentro do dataset."""
    experiment_name = "_test_basis_run"
    run_dir = RESULTS_DIR / "classifier" / "shepard" / "synthetic_demo" / experiment_name
    shutil.rmtree(run_dir, ignore_errors=True)

    result = run_basis_experiment(
        context="classifier",
        basis="shepard",
        dataset="synthetic_demo",
        experiment_name=experiment_name,
        metric_names=["accuracy", "f1"],
        cv_strategy="stratified_kfold",
        n_splits=3,
        n_repeats=1,
        tuning_strategy="random_search",
        tuning_n_iter=3,
    )

    assert result["test_results"] is not None
    assert "accuracy" in result["test_results"]["metrics"]
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "test_results.json").exists()
    assert len(list(run_dir.glob("run_*.json"))) > 0

    shutil.rmtree(run_dir, ignore_errors=True)


def test_baseline_experiment_persists_to_isolated_path():
    """Valida que baselines externos (logreg, knn) ficam FORA da arvore de
    comparacao de bases, em results/external_baselines/."""
    experiment_name = "_test_run"
    run_dir = RESULTS_DIR / "external_baselines" / "logreg" / "synthetic_demo" / experiment_name
    shutil.rmtree(run_dir, ignore_errors=True)

    result = run_baseline_experiment(
        model_name="logreg",
        dataset="synthetic_demo",
        experiment_name=experiment_name,
        metric_names=["accuracy", "f1"],
        cv_strategy="stratified_kfold",
        n_splits=3,
        n_repeats=1,
        tuning_strategy="random_search",
        tuning_n_iter=3,
    )

    assert result["test_results"] is not None
    assert "accuracy" in result["test_results"]["metrics"]
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "test_results.json").exists()
    assert len(list(run_dir.glob("run_*.json"))) > 0

    shutil.rmtree(run_dir, ignore_errors=True)


def test_checkpoint_skips_completed_runs():
    experiment_name = "_test_checkpoint"
    run_dir = RESULTS_DIR / "external_baselines" / "logreg" / "synthetic_demo" / experiment_name
    shutil.rmtree(run_dir, ignore_errors=True)

    kwargs = dict(
        model_name="logreg",
        dataset="synthetic_demo",
        experiment_name=experiment_name,
        metric_names=["accuracy", "f1"],
        cv_strategy="stratified_kfold",
        n_splits=2,
        n_repeats=1,
        tuning_strategy="random_search",
        tuning_n_iter=2,
    )

    run_baseline_experiment(**kwargs)
    n_runs_first = len(list(run_dir.glob("run_*.json")))

    run_baseline_experiment(**kwargs)  # segunda execucao: tudo deve ser pulado via checkpoint
    n_runs_second = len(list(run_dir.glob("run_*.json")))

    assert n_runs_first == n_runs_second, "checkpoint falhou: gerou runs duplicados"
    shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    test_baselines_and_datasets_available()
    test_shepard_basis_registered()
    test_fema_classifier_end_to_end()
    test_basis_experiment_persists_to_scoped_path()
    test_baseline_experiment_persists_to_isolated_path()
    test_checkpoint_skips_completed_runs()
    print("Todos os testes passaram.")