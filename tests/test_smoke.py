"""
Smoke test: roda o experimento inteiro (config -> tuning -> checkpoint ->
summary -> avaliação final) contra o baseline sintético, que não depende
de core/ nem de nenhum CSV específico. Serve para validar a infraestrutura
isoladamente da qualidade dos modelos.

Rodar com: python3 -m pytest tests/ -v   (ou python3 tests/test_smoke.py)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.config_schema import load_experiment_config
from experiments.run_experiment import _override, _run
from models.registry import list_models
from io.datasets import list_datasets


def test_registries_populated():
    assert "fema" in list_models()
    assert "cnn" in list_models()
    assert "logreg_baseline" in list_models()
    assert "synthetic_demo" in list_datasets()


def test_full_pipeline_end_to_end(tmp_output="results/_test_run"):
    shutil.rmtree(tmp_output, ignore_errors=True)
    cfg = load_experiment_config("configs/logreg_baseline.yaml")
    cfg = _override(cfg, output_dir=tmp_output, tuning_n_iter=3)
    result = _run(cfg)

    assert result["test_results"] is not None
    assert "accuracy" in result["test_results"]["metrics"]

    run_dir = Path(tmp_output) / "logreg_baseline" / "smoke_test"
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "test_results.json").exists()
    assert len(list(run_dir.glob("run_*.json"))) > 0

    shutil.rmtree(tmp_output, ignore_errors=True)
    print("OK: pipeline completo (config -> tuning -> summary -> teste final)")


def test_checkpoint_skips_completed_runs(tmp_output="results/_test_checkpoint"):
    shutil.rmtree(tmp_output, ignore_errors=True)
    cfg = load_experiment_config("configs/logreg_baseline.yaml")
    cfg = _override(cfg, output_dir=tmp_output, tuning_n_iter=2, n_splits=2, n_repeats=1)

    _run(cfg)
    run_dir = Path(tmp_output) / "logreg_baseline" / "smoke_test"
    n_runs_first = len(list(run_dir.glob("run_*.json")))

    _run(cfg)  # segunda execução: tudo deve ser pulado via checkpoint
    n_runs_second = len(list(run_dir.glob("run_*.json")))

    assert n_runs_first == n_runs_second, "checkpoint falhou: gerou runs duplicados"
    shutil.rmtree(tmp_output, ignore_errors=True)
    print("OK: checkpoint não duplica execuções já concluídas")


if __name__ == "__main__":
    test_registries_populated()
    print("OK: registries de modelos e datasets carregados")
    test_full_pipeline_end_to_end()
    test_checkpoint_skips_completed_runs()
    print("\nTodos os smoke tests passaram.")
