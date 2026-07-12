"""Smoke test do pipeline completo, usando o dataset sintético (rápido, não
depende de CSVs externos). Valida que os resultados são persistidos em
disco (run_*.json, summary.json, test_results.json) e que o checkpoint não
duplica execuções já concluídas.

Rodar com: python3 -m pytest tests/ -v
"""
from __future__ import annotations

import shutil
from pathlib import Path

from ..src import _run, override, load_experiment_config, MODELS, DATASETS

CONFIG_PATH = "configs/smoke_test.yaml"


def test_models_and_datasets_available():
    assert "fema" in MODELS
    assert "logreg" in MODELS
    assert "mlp" in MODELS
    assert "synthetic_demo" in DATASETS


def test_full_pipeline_persists_results():
    output_dir = "results/_test_run"
    shutil.rmtree(output_dir, ignore_errors=True)

    config = override(load_experiment_config(CONFIG_PATH), output_dir=output_dir)
    result = _run(config)

    assert result["test_results"] is not None
    assert "accuracy" in result["test_results"]["metrics"]

    run_dir = Path(output_dir) / config.model / config.experiment_name
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "test_results.json").exists()
    assert len(list(run_dir.glob("run_*.json"))) > 0

    shutil.rmtree(output_dir, ignore_errors=True)


def test_checkpoint_skips_completed_runs():
    output_dir = "results/_test_checkpoint"
    shutil.rmtree(output_dir, ignore_errors=True)

    config = override(load_experiment_config(CONFIG_PATH), output_dir=output_dir, n_splits=2, tuning_n_iter=2)
    _run(config)
    run_dir = Path(output_dir) / config.model / config.experiment_name
    n_runs_first = len(list(run_dir.glob("run_*.json")))

    _run(config)  # segunda execução: tudo deve ser pulado via checkpoint
    n_runs_second = len(list(run_dir.glob("run_*.json")))

    assert n_runs_first == n_runs_second, "checkpoint falhou: gerou runs duplicados"
    shutil.rmtree(output_dir, ignore_errors=True)


def test_fema_model_end_to_end():
    """FEMa é o modelo principal do projeto — valida que ele treina/prediz
    de fato (não só que o pipeline genérico roda)."""
    import numpy as np
    from src.models import create_model

    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [10, 10], [11, 11]], dtype=float)
    y_train = np.array([0, 0, 0, 1, 1, 1])
    X_test = np.array([[1.5, 2.5], [10.5, 10.5]])

    model = create_model("fema", {"basis_function": "shepard", "k": 3, "z": 2.0})
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    assert list(labels) == [0, 1]


if __name__ == "__main__":
    test_models_and_datasets_available()
    test_fema_model_end_to_end()
    test_full_pipeline_persists_results()
    test_checkpoint_skips_completed_runs()
    print("Todos os testes passaram.")
