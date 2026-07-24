"""Testes do modulo de comparacao entre bases (reporting/compare_bases.py).

Roda run_all_bases com um pequeno subconjunto de bases sobre o dataset
sintetico (rapido) usando TODAS as metricas registradas para o contexto
(metrics.registry.metrics_for_context) - nao um subconjunto reduzido -
para garantir que a comparacao, a tabela, o JSON e os plots funcionam
para cada metrica registrada, nao so' para accuracy/f1.

Rodar com: python3 -m pytest tests/test_reporting.py -v
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from metrics.registry import metrics_for_context
from pipeline.run_model import run_all_bases
from reporting.compare_bases import compare, consolidate, run_full_comparison, save_comparison_json, save_comparison_table

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

CONTEXT = "classifier"
DATASET = "synthetic_demo"
EXPERIMENT_NAME = "_test_compare"
BASES = ["shepard", "radial", "rbf_gaussian"]


def _cleanup():
    for basis in BASES:
        shutil.rmtree(RESULTS_DIR / CONTEXT / basis / DATASET / EXPERIMENT_NAME, ignore_errors=True)
    shutil.rmtree(REPORTS_DIR / "_test_basis_comparison", ignore_errors=True)


def _run_bases():
    """Roda as bases de teste com TODAS as metricas de classificacao
    registradas (nao passa metric_names -> pega o default do pipeline,
    que agora e' metrics_for_context('classifier'))."""
    run_all_bases(
        context=CONTEXT,
        dataset=DATASET,
        bases=BASES,
        experiment_name=EXPERIMENT_NAME,
        cv_strategy="stratified_kfold",
        n_splits=3,
        n_repeats=1,
        tuning_strategy="random_search",
        tuning_n_iter=3,
    )


def setup_module(module):
    _cleanup()
    _run_bases()


def teardown_module(module):
    _cleanup()


def test_compare_covers_all_registered_classification_metrics():
    """A linha de cada base deve trazer TODAS as metricas de
    classificacao registradas em metrics.registry, nao um subconjunto."""
    rows = compare(context=CONTEXT, dataset=DATASET, experiment_name=EXPERIMENT_NAME, bases=BASES)
    expected_metrics = set(metrics_for_context(CONTEXT))

    assert len(rows) == len(BASES)
    for row in rows:
        assert row["status"] == "ok", row
        assert row["test_metrics"] is not None
        present = set(row["test_metrics"].keys())
        assert present == expected_metrics, f"base={row['basis']}: esperado {expected_metrics}, obtido {present}"


def test_consolidate_and_save_json_json_contains_all_metrics():
    rows = compare(context=CONTEXT, dataset=DATASET, experiment_name=EXPERIMENT_NAME, bases=BASES)
    consolidated = consolidate(rows, context=CONTEXT, dataset=DATASET, experiment_name=EXPERIMENT_NAME)

    assert set(consolidated["metrics_compared"]) == set(metrics_for_context(CONTEXT))
    assert set(consolidated["ranking_per_metric"].keys()) == set(metrics_for_context(CONTEXT))
    for metric_name, ranking in consolidated["ranking_per_metric"].items():
        assert len(ranking) == len(BASES), metric_name
        # ranking deve estar ordenado (melhor primeiro) - checagem fraca:
        # so' garante que existe um "value" numerico em cada entrada.
        assert all(isinstance(e["value"], (int, float)) for e in ranking)

    json_path = save_comparison_json(consolidated, str(REPORTS_DIR / "_test_basis_comparison" / "comparison.json"))
    assert json_path.exists()
    reloaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(reloaded["metrics_compared"]) == set(metrics_for_context(CONTEXT))


def test_save_comparison_table_csv_has_all_metric_columns():
    rows = compare(context=CONTEXT, dataset=DATASET, experiment_name=EXPERIMENT_NAME, bases=BASES)
    csv_path = save_comparison_table(
        rows, str(REPORTS_DIR / "_test_basis_comparison" / "comparison_table.csv"), context=CONTEXT
    )
    assert csv_path.exists()

    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    for metric_name in metrics_for_context(CONTEXT):
        assert metric_name in header, f"coluna '{metric_name}' ausente no CSV"


def test_run_full_comparison_generates_all_artifacts():
    """Ponto de entrada unico: deve gerar CSV, JSON e um plot de barra +
    uma curva de k PARA CADA metrica registrada do contexto."""
    output_dir = REPORTS_DIR / "_test_basis_comparison" / "full"
    result = run_full_comparison(
        context=CONTEXT, dataset=DATASET, experiment_name=EXPERIMENT_NAME, bases=BASES, output_dir=str(output_dir)
    )

    assert result["csv_path"].exists()
    assert result["json_path"].exists()

    expected_metrics = set(metrics_for_context(CONTEXT))
    bar_metric_names = {p.stem.replace("bar_", "") for p in result["bar_plot_paths"]}
    curve_metric_names = {p.stem.replace("curve_", "").replace("_vs_k", "") for p in result["curve_plot_paths"]}

    assert bar_metric_names == expected_metrics, bar_metric_names
    assert curve_metric_names == expected_metrics, curve_metric_names

    for p in result["bar_plot_paths"] + result["curve_plot_paths"]:
        assert p.exists() and p.stat().st_size > 0


if __name__ == "__main__":
    setup_module(None)
    try:
        test_compare_covers_all_registered_classification_metrics()
        test_consolidate_and_save_json_json_contains_all_metrics()
        test_save_comparison_table_csv_has_all_metric_columns()
        test_run_full_comparison_generates_all_artifacts()
        print("Todos os testes de reporting passaram.")
    finally:
        teardown_module(None)