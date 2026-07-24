"""
Comparacao entre bases de interpolacao do FEMa.

Substitui reporting/compare_models.py (removido): aquele modulo comparava
MODELOS (fema vs logreg vs mlp), o que nao e' o objeto de estudo do
projeto. Este compara BASES entre si, para um mesmo contexto
(classifier|regressor) e dataset - a tabela/figura central do projeto de
IC.

Le' o summary.json e o test_results.json de cada base ja' rodada via
pipeline.run_model.run_basis_experiment / run_all_bases (em
results/<context>/<basis>/<dataset>/<experiment_name>/) e produz:

  - uma tabela comparativa (CSV) e um JSON consolidado, com TODAS as
    metricas de teste calculadas (ver metrics.registry.metrics_for_context)
    + a melhor combinacao de hiperparametros de cada base;
  - um grafico de barras por metrica (todas as bases lado a lado);
  - uma curva de sensibilidade a k por metrica (uma linha por base),
    construida a partir do ranking de CV de cada summary.json (nao do
    teste - o teste so' tem UM ponto por base, nao da' pra' fazer curva).

Ponto de entrada unico (o que efetivamente "aciona" a comparacao):

    from reporting.compare_bases import run_full_comparison

    run_full_comparison(
        context="classifier", dataset="fetal_health", experiment_name="baseline",
        output_dir="reports/basis_comparison/classifier/fetal_health",
    )

Isso gera, dentro de output_dir/:
    comparison_table.csv
    comparison.json
    plots/bar_<metrica>.png            (uma por metrica de teste disponivel)
    plots/curve_<metrica>_vs_k.png     (uma por metrica de teste disponivel)
"""
from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from core import Basis
from metrics.registry import metrics_for_context

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
) -> List[Dict[str, Any]]:
    """Monta uma linha por base com o resultado de teste + a melhor
    combinacao de hiperparametros. `bases`: por padrao, todas as
    registradas em core.Basis.available()."""
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

        row: Dict[str, Any] = {
            "context": context,
            "basis": basis,
            "dataset": dataset,
            "status": "ok",
            "best_hyperparameters": summary["best_configuration"]["hyperparameters"],
            "n_combinations_evaluated": summary["n_combinations_evaluated"],
        }
        if test_results is not None:
            row["test_metrics"] = test_results["metrics"]
        else:
            row["test_metrics"] = None
            row["status"] = "sem test_results.json (run_final_test=False?)"
        rows.append(row)
    return rows


def _all_metric_names(rows: List[Dict[str, Any]], context: str) -> List[str]:
    """Todas as metricas registradas para o contexto (ver
    metrics.registry.metrics_for_context) que de fato aparecem em ALGUMA
    linha - preserva a ordem canonica do registry em vez de depender da
    ordem de insercao dos dicts."""
    present = set()
    for row in rows:
        if row.get("test_metrics"):
            present.update(row["test_metrics"].keys())
    return [m for m in metrics_for_context(context) if m in present]


def consolidate(
    rows: List[Dict[str, Any]],
    context: str,
    dataset: str,
    experiment_name: str = "baseline",
) -> Dict[str, Any]:
    """Empacota `rows` num unico dict serializavel em JSON, pronto para
    save_comparison_json. Inclui a lista de metricas efetivamente
    comparadas e um ranking por metrica (bases ordenadas da melhor pra'
    pior, respeitando se a metrica e' higher_is_better)."""
    from metrics.registry import is_higher_better

    metric_names = _all_metric_names(rows, context)

    ranking_per_metric: Dict[str, List[Dict[str, Any]]] = {}
    for metric_name in metric_names:
        higher_is_better = is_higher_better(metric_name)
        entries = [
            {"basis": row["basis"], "value": row["test_metrics"].get(metric_name)}
            for row in rows
            if row.get("test_metrics") and row["test_metrics"].get(metric_name) is not None
        ]
        entries.sort(key=lambda e: e["value"], reverse=higher_is_better)
        ranking_per_metric[metric_name] = entries

    return {
        "context": context,
        "dataset": dataset,
        "experiment": experiment_name,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "metrics_compared": metric_names,
        "bases": rows,
        "ranking_per_metric": ranking_per_metric,
    }


def save_comparison_json(consolidated: Dict[str, Any], output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2, ensure_ascii=False, default=str)
    return path


def save_comparison_table(rows: List[Dict[str, Any]], output_path: str, context: Optional[str] = None) -> Path:
    """context: se informado, ordena as colunas de metrica na ordem
    canonica de metrics.registry.metrics_for_context(context); senao, usa
    a ordem alfabetica das metricas encontradas em `rows`."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if context:
        metric_names = _all_metric_names(rows, context)
    else:
        metric_names = set()
        for row in rows:
            if row.get("test_metrics"):
                metric_names.update(row["test_metrics"].keys())
        metric_names = sorted(metric_names)

    fieldnames = ["context", "basis", "dataset", "status"] + metric_names + [
        "best_hyperparameters", "n_combinations_evaluated",
    ]

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
                "n_combinations_evaluated": row.get("n_combinations_evaluated"),
            }
            test_metrics = row.get("test_metrics") or {}
            for m in metric_names:
                flat[m] = test_metrics.get(m)
            writer.writerow(flat)

    return path


def plot_metric_comparison(
    rows: List[Dict[str, Any]], metric_name: str, filename: Optional[str] = None, show: bool = False
) -> Optional[Path]:
    """Grafico de barras comparando `metric_name` entre as bases (um bar
    por base). Retorna None (sem gerar figura) se nenhuma base tiver essa
    metrica calculada."""
    import matplotlib.pyplot as plt

    labels, values = [], []
    for row in rows:
        test_metrics = row.get("test_metrics") or {}
        value = test_metrics.get(metric_name)
        if value is None:
            continue
        labels.append(row["basis"])
        values.append(value)

    if not values:
        return None

    fig, ax = plt.subplots(figsize=(9, 5), dpi=130)
    ax.bar(labels, values)
    ax.set_ylabel(metric_name)
    ax.set_title(f"Comparacao de bases de interpolacao: {metric_name}")
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()

    out_path = None
    if filename:
        out_path = Path(filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_all_metrics(
    rows: List[Dict[str, Any]], context: str, output_dir: str, prefix: str = "bar"
) -> List[Path]:
    """Chama plot_metric_comparison para TODA metrica registrada em
    metrics.registry.metrics_for_context(context) que apareca em `rows`.
    Retorna a lista de arquivos gerados (metricas ausentes/None em todas
    as bases sao puladas, sem erro)."""
    metric_names = _all_metric_names(rows, context)
    generated = []
    for metric_name in metric_names:
        path = plot_metric_comparison(
            rows, metric_name, filename=f"{output_dir}/{prefix}_{metric_name}.png"
        )
        if path is not None:
            generated.append(path)
    return generated


def _load_basis_ranking(context: str, basis: str, dataset: str, experiment_name: str) -> Optional[List[dict]]:
    summary = _load_json(RESULTS_DIR / context / basis / dataset / experiment_name / "summary.json")
    return summary["ranking"] if summary else None


def plot_k_curve(
    context: str,
    dataset: str,
    metric_name: str,
    experiment_name: str = "baseline",
    bases: Optional[List[str]] = None,
    filename: Optional[str] = None,
    show: bool = False,
) -> Optional[Path]:
    """Curva de sensibilidade a k por base: para cada base, plota (k,
    metrica media de VALIDACAO/CV - nao teste) de cada combinacao avaliada
    no grid/random search, uma linha por base.

    IMPORTANTE (limitacao conhecida): como o random_search tambem varia o
    parametro proprio de cada base (z, epsilon, c, ...) simultaneamente a
    k, os pontos de uma mesma base podem nao formar uma curva monotonica
    "limpa" em k - o valor no eixo Y para um dado k e' de UMA combinacao
    especifica sorteada com aquele k, nao uma media marginal sobre esse
    parametro. Ainda assim e' util para visualizar a faixa de desempenho
    de cada base ao longo de k. Para uma curva estritamente marginal em k,
    rode um experimento com fixed_hyperparameters variando so' k."""
    import matplotlib.pyplot as plt

    bases = bases or Basis.available()

    fig, ax = plt.subplots(figsize=(9, 5), dpi=130)
    any_data = False

    for basis in bases:
        ranking = _load_basis_ranking(context, basis, dataset, experiment_name)
        if not ranking:
            continue

        points = [
            (entry["hyperparameters"].get("k"), entry["metrics"].get(metric_name, {}).get("mean"))
            for entry in ranking
            if entry["metrics"].get(metric_name, {}).get("mean") is not None
        ]
        points = sorted(p for p in points if p[0] is not None)
        if not points:
            continue

        ks, values = zip(*points)
        ax.plot(ks, values, marker="o", markersize=4, linewidth=1, label=basis)
        any_data = True

    if not any_data:
        plt.close(fig)
        return None

    ax.set_xlabel("k")
    ax.set_ylabel(f"{metric_name} (media de CV)")
    ax.set_title(f"Sensibilidade a k por base: {metric_name}")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    out_path = None
    if filename:
        out_path = Path(filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_all_k_curves(
    context: str, dataset: str, output_dir: str, experiment_name: str = "baseline",
    bases: Optional[List[str]] = None, prefix: str = "curve",
) -> List[Path]:
    """Chama plot_k_curve para toda metrica registrada em
    metrics.registry.metrics_for_context(context)."""
    generated = []
    for metric_name in metrics_for_context(context):
        path = plot_k_curve(
            context, dataset, metric_name, experiment_name=experiment_name, bases=bases,
            filename=f"{output_dir}/{prefix}_{metric_name}_vs_k.png",
        )
        if path is not None:
            generated.append(path)
    return generated


def run_full_comparison(
    context: str,
    dataset: str,
    experiment_name: str = "baseline",
    bases: Optional[List[str]] = None,
    output_dir: str = "reports/basis_comparison",
) -> Dict[str, Any]:
    """Ponto de entrada unico: le' os resultados ja' persistidos (ver
    pipeline.run_model.run_all_bases), monta a comparacao, e grava tabela
    (CSV), JSON consolidado e todos os plots (barras + curvas de k) para
    TODAS as metricas registradas do contexto - tudo dentro de
    `output_dir`. NAO roda experimentos - assume que
    run_basis_experiment/run_all_bases ja' foram executados.

    Retorna um dict com os caminhos de tudo o que foi gerado, para uso
    programatico (ex: anexar num relatorio) ou so' inspecao no terminal."""
    output_dir_path = Path(output_dir)
    plots_dir = output_dir_path / "plots"

    rows = compare(context=context, dataset=dataset, experiment_name=experiment_name, bases=bases)
    consolidated = consolidate(rows, context=context, dataset=dataset, experiment_name=experiment_name)

    csv_path = save_comparison_table(rows, str(output_dir_path / "comparison_table.csv"), context=context)
    json_path = save_comparison_json(consolidated, str(output_dir_path / "comparison.json"))

    bar_paths = plot_all_metrics(rows, context=context, output_dir=str(plots_dir), prefix="bar")
    curve_paths = plot_all_k_curves(
        context, dataset, output_dir=str(plots_dir), experiment_name=experiment_name, bases=bases, prefix="curve"
    )

    return {
        "rows": rows,
        "consolidated": consolidated,
        "csv_path": csv_path,
        "json_path": json_path,
        "bar_plot_paths": bar_paths,
        "curve_plot_paths": curve_paths,
    }