"""Smoke test do pipeline completo, usando o dataset sintetico (rapido, nao
depende de CSVs externos). Valida que os resultados sao persistidos em
disco (run_*.json, summary.json, test_results.json) e que o checkpoint nao
duplica execucoes ja concluidas.

Rodar com: python3 -m pytest tests/ -v
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from datasets.registry import available_datasets
from models.registry import available_models, get_model_plugin
from pipeline.run_model import run_model

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def test_models_and_datasets_available():
    assert "fema" in available_models()
    assert "logreg" in available_models()
    assert "mlp" in available_models()
    assert "synthetic_demo" in available_datasets()


def test_fema_model_end_to_end():
    """FEMa e o modelo principal do projeto - valida que ele treina/prediz
    de fato (nao so que o pipeline generico roda)."""
    plugin = get_model_plugin("fema")

    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [10, 10], [11, 11]], dtype=float)
    y_train = np.array([0, 0, 0, 1, 1, 1])
    X_test = np.array([[1.5, 2.5], [10.5, 10.5]])

    estimator = plugin.create_model({"basis_function": "shepard", "k": 3, "z": 2.0}, random_state=42)
    estimator.fit(X_train, y_train)
    labels = plugin.predict(estimator, X_test)
    assert list(labels) == [0, 1]


def test_full_pipeline_persists_results():
    experiment_name = "_test_run"
    run_dir = RESULTS_DIR / "logreg" / experiment_name
    shutil.rmtree(run_dir, ignore_errors=True)

    result = run_model(
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
    run_dir = RESULTS_DIR / "logreg" / experiment_name
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

    run_model(**kwargs)
    n_runs_first = len(list(run_dir.glob("run_*.json")))

    run_model(**kwargs)  # segunda execucao: tudo deve ser pulado via checkpoint
    n_runs_second = len(list(run_dir.glob("run_*.json")))

    assert n_runs_first == n_runs_second, "checkpoint falhou: gerou runs duplicados"
    shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    test_models_and_datasets_available()
    test_fema_model_end_to_end()
    test_full_pipeline_persists_results()
    test_checkpoint_skips_completed_runs()
    print("Todos os testes passaram.")
