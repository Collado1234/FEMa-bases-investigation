"""
Construcao do resumo final (summary.json).

Le todos os run_*.json de results/<scope...>/, agrupa por combinacao de
hiperparametros (combo_id), calcula media e desvio padrao de cada metrica
entre folds/repeticoes, rankeia as combinacoes pela metrica "primaria"
definida na config (respeitando se maior ou menor e melhor - ver
metrics/registry.py::HIGHER_IS_BETTER) e grava summary.json.

MUDANCA ARQUITETURAL: como `*scope` agora e' generico (persistence/
run_writer.py) e cada diretorio de resultados passou a corresponder a UMA
UNICA base do FEMa (ou a UM UNICO baseline externo), o ranking aqui dentro
e' sempre homogeneo - so' varia k e os parametros PROPRIOS daquela base.
Isso e' a correcao do problema original: antes, o ranking podia misturar
combinacoes de bases DIFERENTES (ex: shepard e radial) na mesma lista, e
o "best_configuration" global nao respondia a pergunta de pesquisa do
projeto ("qual e' o melhor desempenho DE CADA base", nao "qual e' a melhor
combinacao entre todas misturadas"). Ver reporting/compare_bases.py para
a comparacao ENTRE bases, que le' um summary.json por base.

`extra_fields` e' injetado no summary tal como veio de quem chamou (ex:
{"context": "classifier", "basis": "shepard", "experiment": "baseline"}
ou {"model": "logreg", "experiment": "baseline"}) - summary_builder nao
precisa saber o que esses campos significam.
"""

import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional

from metrics.registry import is_higher_better
from persistence.run_writer import load_all_runs, write_summary


def build_summary(
    *scope: str,
    primary_metric: str,
    higher_is_better: Optional[bool] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    runs = load_all_runs(*scope)
    if not runs:
        raise RuntimeError(
            f"Nenhum run encontrado para results/{'/'.join(scope)}. "
            "Rode o grid/random search antes de gerar o summary."
        )

    if higher_is_better is None:
        higher_is_better = is_higher_better(primary_metric)

    grouped: Dict[str, List[dict]] = defaultdict(list)
    for run in runs:
        grouped[run["combo_id"]].append(run)

    ranking = []
    for combo_id, group_runs in grouped.items():
        metric_names = list(group_runs[0]["metrics"].keys())
        agg_metrics = {}
        for m in metric_names:
            values = [r["metrics"][m] for r in group_runs if r["metrics"].get(m) is not None]
            if values:
                agg_metrics[m] = {
                    "mean": statistics.fmean(values),
                    "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
            else:
                agg_metrics[m] = {"mean": None, "std": None, "n": 0}

        times = [r["execution_time_seconds"] for r in group_runs]

        ranking.append(
            {
                "combo_id": combo_id,
                "hyperparameters": group_runs[0]["hyperparameters"],
                "n_runs": len(group_runs),
                "metrics": agg_metrics,
                "mean_execution_time_seconds": statistics.fmean(times),
            }
        )

    def sort_key(entry):
        mean_val = entry["metrics"].get(primary_metric, {}).get("mean")
        if mean_val is None:
            return (True, 0.0)
        return (False, -mean_val if higher_is_better else mean_val)

    ranking.sort(key=sort_key)

    summary = {
        **(extra_fields or {}),
        "primary_metric": primary_metric,
        "higher_is_better": higher_is_better,
        "n_combinations_evaluated": len(ranking),
        "total_runs": len(runs),
        "best_configuration": ranking[0] if ranking else None,
        "ranking": ranking,
    }

    write_summary(*scope, summary=summary)
    return summary