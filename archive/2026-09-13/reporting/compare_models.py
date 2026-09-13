"""
Comparacao entre bases de interpolacao do FEMa.

Esta e' a substituicao de reporting/compare_models.py (removido): aquele
modulo comparava MODELOS (fema vs logreg vs mlp), o que nao e' o objeto de
estudo do projeto. Este compara BASES entre si, para um mesmo contexto
(classifier|regressor) e dataset - a tabela central do projeto de IC.

Le' o summary.json e o test_results.json de cada base ja' rodada via
pipeline.run_model.run_basis_experiment / run_all_bases (em
results/<context>/<basis>/<dataset>/<experiment_name>/) e monta uma tabela
comparativa (metricas de teste + a melhor combinacao de hiperparametros de
cada base) e, opcionalmente, um grafico de barras.

Uso:
    from reporting.compare_bases import compare, save_comparison_table

    rows = compare(context="classifier", dataset="fetal_health", experiment_name="baseline")
    save_comparison_table(rows, "reports/basis_comparison/classifier/fetal_health/comparison_table.csv")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from core import Basis

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare(
    context: str,
    dataset: str,
    experiment_name: str = "baseline",
    bases: Optional[List[str]] = None,
) -> List[Dict]:
    """bases: lista de bases a comparar; por padrao, todas as registradas
    em core.Basis.available()."""
    bases = bases or Basis.available()
    rows = []
    for basis in bases:
        exp_dir = RESULTS_DIR / context / basis / dataset / experiment_name
        summary = _load_json(exp_dir / "summary.json")
        test_results = _load_json(exp_dir / "test_results.json")

        if summary is None:
            rows.append(
                {"context": context, "basis": basis, "dataset": dataset, "status": "sem summary.json"}
            )
            continue

        row = {
            "context": context,
            "basis": basis,
            "dataset": dataset,
            "status": "ok",
            "best_hyperparameters": summary["best_configuration"]["hyperparameters"],
        }
        if test_results is not None:
            row["test_metrics"] = test_results["metrics"]
        else:
            row["test_metrics"] = None
            row["status"] = "sem test_results.json (run_final_test=False?)"
        rows.append(row)
    return rows


def save_comparison_table(rows: List[Dict], output_path: str) -> Path:
    import csv

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metric_names = set()
    for row in rows:
        if row.get("test_metrics"):
            metric_names.update(row["test_metrics"].keys())
    metric_names = sorted(metric_names)

    fieldnames = ["context", "basis", "dataset", "status"] + metric_names + ["best_hyperparameters"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {
                "context": row["context"],
                "basis": row["basis"],
                "dataset": row["dataset"],
                "status": row["status"],
                "best_hyperparameters": json.dumps(row.get("best_hyperparameters"), ensure_ascii=False),
            }
            test_metrics = row.get("test_metrics") or {}
            for m in metric_names:
                flat[m] = test_metrics.get(m)
            writer.writerow(flat)

    return path


def plot_metric_comparison(
    rows: List[Dict], metric_name: str, filename: Optional[str] = None, show: bool = False
):
    """Grafico de barras comparando `metric_name` entre as bases (um bar
    por base, nao mais por modelo)."""
    import matplotlib.pyplot as plt

    labels, values = [], []
    for row in rows:
        test_metrics = row.get("test_metrics") or {}
        value = test_metrics.get(metric_name)
        if value is None:
            continue
        labels.append(row["basis"])
        values.append(value)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=130)
    ax.bar(labels, values)
    ax.set_ylabel(metric_name)
    ax.set_title(f"Comparacao de bases de interpolacao: {metric_name}")
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()

    if filename:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)