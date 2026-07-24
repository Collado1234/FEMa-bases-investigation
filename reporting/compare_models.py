"""
Comparacao de modelos.

Le o summary.json e o test_results.json de uma lista de experimentos
(pares model/experiment_name) ja rodados via pipeline.run_model, monta uma
tabela comparativa (metricas de teste + a melhor combinacao de
hiperparametros de cada um) e, opcionalmente, um grafico de barras.

Versao enxuta do reporting/compare_models.py do icd-project - aqui nao
existe a dimensao task/subgroup, entao a comparacao e simplesmente por
(model, experiment_name).

Uso:
    from reporting.compare_models import compare, save_comparison_table

    rows = compare([("fema", "baseline"), ("logreg", "baseline"), ("mlp", "baseline")])
    save_comparison_table(rows, "reports/model_comparison/baseline/comparison_table.csv")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare(experiments: List[Tuple[str, str]]) -> List[Dict]:
    """experiments: lista de (model_name, experiment_name) ja executados.
    Retorna uma lista de dicts, um por experimento, com as metricas de
    teste e os melhores hiperparametros encontrados no tuning."""
    rows = []
    for model_name, experiment_name in experiments:
        exp_dir = RESULTS_DIR / model_name / experiment_name
        summary = _load_json(exp_dir / "summary.json")
        test_results = _load_json(exp_dir / "test_results.json")

        if summary is None:
            row = {"model": model_name, "experiment": experiment_name, "status": "sem summary.json"}
            rows.append(row)
            continue

        row = {
            "model": model_name,
            "experiment": experiment_name,
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

    fieldnames = ["model", "experiment", "status"] + metric_names + ["best_hyperparameters"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {
                "model": row["model"],
                "experiment": row["experiment"],
                "status": row["status"],
                "best_hyperparameters": json.dumps(row.get("best_hyperparameters"), ensure_ascii=False),
            }
            test_metrics = row.get("test_metrics") or {}
            for m in metric_names:
                flat[m] = test_metrics.get(m)
            writer.writerow(flat)

    return path


def plot_metric_comparison(rows: List[Dict], metric_name: str, filename: Optional[str] = None, show: bool = False):
    """Grafico de barras simples comparando `metric_name` entre os
    experimentos (usa reporting/plots.py como base de estilo)."""
    import matplotlib.pyplot as plt

    labels, values = [], []
    for row in rows:
        test_metrics = row.get("test_metrics") or {}
        value = test_metrics.get(metric_name)
        if value is None:
            continue
        labels.append(f"{row['model']}/{row['experiment']}")
        values.append(value)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=130)
    ax.bar(labels, values)
    ax.set_ylabel(metric_name)
    ax.set_title(f"Comparacao de modelos: {metric_name}")
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()

    if filename:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
