"""
analysis/ranking_report.py

Coloque este arquivo dentro da pasta `analysis/` do seu projeto, ao lado
de `statistical_comparison.py` (importa as funcoes de la -- reaproveita
100% da logica estatistica ja validada: Friedman + post-hoc Wilcoxon com
Holm, seguindo Demsar 2006).

MODO SIMPLES (recomendado): auto-descobre TODAS as basis e TODOS os
datasets que tem resultados para o --experiment-name informado, sem
precisar listar nada manualmente:

    python -m analysis.ranking_report \
        --context classifier \
        --experiment-name experiments_v2 \
        --metric f1 \
        --csv-prefix reports/experiments_v2

Isso gera:
  - reports/experiments_v2_ranking.csv   (1 linha por dataset x basis: posicao
    no ranking, media, desvio, resultado do Friedman naquele dataset)
  - reports/experiments_v2_pairwise.csv  (1 linha por dataset x par de basis
    testado no post-hoc: p-valor, p-valor Holm, significativo?, effect size)
  - reports/experiments_v2.md            (mesma coisa em Markdown, para colar
    direto no relatorio/apresentacao)

Tambem imprime tudo no console, dataset por dataset, com o mesmo veredito.

Se algum dataset nao tiver TODAS as basis rodadas com esse experiment_name,
ele e' automaticamente EXCLUIDO da comparacao (Friedman exige matriz
completa) e um aviso e' impresso dizendo exatamente o que falta.

Uso avancado (fixar manualmente datasets/basis em vez de auto-descobrir):
    python -m analysis.ranking_report \
        --context classifier --datasets letter iris --bases attention lorentzian \
        --metric f1 --experiment-name experiments_v2 --csv-prefix reports/manual
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple

from analysis.statistical_comparison import (
    ALPHA_DEFAULT,
    RESULTS_ROOT,
    REPORTS_ROOT,
    DataAvailabilityError,
    analyze_dataset_metric,
)


# --------------------------------------------------------------------------
# Auto-descoberta de basis/datasets a partir da pasta results/
# --------------------------------------------------------------------------

def discover_bases_and_datasets(
    context: str, experiment_name: str
) -> Tuple[List[str], List[str], Dict[str, Set[str]]]:
    """Escaneia results/<context>/<basis>/<dataset>/<experiment_name>/summary.json.
    Retorna (bases_encontradas, datasets_com_TODAS_as_basis, datasets_por_basis)."""
    context_dir = RESULTS_ROOT / context
    if not context_dir.exists():
        raise SystemExit(f"Pasta nao encontrada: {context_dir}")

    bases = sorted(p.name for p in context_dir.iterdir() if p.is_dir())
    if not bases:
        raise SystemExit(f"Nenhuma basis encontrada em {context_dir}")

    datasets_por_basis: Dict[str, Set[str]] = {}
    for basis in bases:
        basis_dir = context_dir / basis
        datasets_por_basis[basis] = {
            p.name for p in basis_dir.iterdir()
            if p.is_dir() and (p / experiment_name / "summary.json").exists()
        }

    all_datasets = sorted(set.union(*datasets_por_basis.values())) if datasets_por_basis else []
    common_datasets = sorted(set.intersection(*datasets_por_basis.values())) if datasets_por_basis else []

    incomplete = sorted(set(all_datasets) - set(common_datasets))
    for dataset in incomplete:
        faltando = [b for b in bases if dataset not in datasets_por_basis[b]]
        print(f"  [aviso] dataset '{dataset}' EXCLUIDO da comparacao -- faltam resultados de: "
              f"{', '.join(faltando)} (experiment_name='{experiment_name}')")

    return bases, common_datasets, datasets_por_basis


# --------------------------------------------------------------------------
# Formatacao / apresentacao
# --------------------------------------------------------------------------

def _ranked_bases(res: dict) -> List[str]:
    return sorted(res["bases"], key=lambda b: res["descriptive"][b]["mean"], reverse=True)


def _holm_pairs_lookup(res: dict) -> dict:
    return {frozenset(ph["pair"]): ph for ph in res["posthoc"]}


def print_dataset_ranking(res: dict) -> None:
    ranking = _ranked_bases(res)
    print(f"\nDataset: {res['dataset']}  (metrica: {res['metric']}, "
          f"n_folds pareados={res['n_folds_paired']})")
    print("  Ranking:")
    for pos, basis in enumerate(ranking, start=1):
        d = res["descriptive"][basis]
        print(f"    {pos}. {basis:28s} media={d['mean']:.4f}  desvio={d['std']:.4f}")

    fr = res["friedman"]
    if not fr["significant"]:
        print(f"  Friedman: p={fr['p_value']:.4g}  -> NAO significativo "
              f"(sem evidencia de diferenca entre as bases nesse dataset).")
        return

    print(f"  Friedman: p={fr['p_value']:.4g}  -> SIGNIFICATIVO "
          f"(Kendall's W={res['kendalls_w']:.3f})")

    lookup = _holm_pairs_lookup(res)
    sig_lines, nonsig_lines = [], []
    for i in range(len(ranking)):
        for j in range(i + 1, len(ranking)):
            b_better, b_worse = ranking[i], ranking[j]
            ph = lookup.get(frozenset({b_better, b_worse}))
            if ph is None:
                continue
            diff = ph["mean_diff"] if ph["pair"][0] == b_better else -ph["mean_diff"]
            line = (f"{b_better} > {b_worse}  (p_holm={ph['p_value_holm']:.4g}, "
                    f"diff={diff:+.4f}, r={ph['effect_size_rank_biserial_r']:+.3f})")
            if ph["significant_after_holm"]:
                sig_lines.append(line)
            else:
                nonsig_lines.append(
                    f"{b_better} ~ {b_worse}  (p_holm={ph['p_value_holm']:.4g}, sem diferenca significativa)"
                )

    if sig_lines:
        print("  Diferencas SIGNIFICATIVAS (Holm):")
        for line in sig_lines:
            print(f"    - {line}")
    if nonsig_lines:
        print("  Sem diferenca significativa:")
        for line in nonsig_lines:
            print(f"    - {line}")


def build_win_count_summary(all_results: List[dict]) -> str:
    wins = {}
    for res in all_results:
        ranking = _ranked_bases(res)
        first = ranking[0]
        second = ranking[1] if len(ranking) > 1 else None
        wins.setdefault(first, {"n_primeiro_lugar": 0, "n_primeiro_sig": 0, "datasets": []})
        wins[first]["n_primeiro_lugar"] += 1
        wins[first]["datasets"].append(res["dataset"])
        if second is not None:
            lookup = _holm_pairs_lookup(res)
            ph = lookup.get(frozenset({first, second}))
            if res["friedman"]["significant"] and ph is not None and ph["significant_after_holm"]:
                wins[first]["n_primeiro_sig"] += 1

    lines = ["| Basis | 1o lugar (n datasets) | ...dos quais significativo vs 2o colocado | Datasets |",
             "|---|---|---|---|"]
    for basis, info in sorted(wins.items(), key=lambda kv: kv[1]["n_primeiro_lugar"], reverse=True):
        lines.append(f"| {basis} | {info['n_primeiro_lugar']} | {info['n_primeiro_sig']} | "
                      f"{', '.join(info['datasets'])} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Exportacao CSV
# --------------------------------------------------------------------------

def export_csv(all_results: List[dict], csv_prefix: str) -> Tuple[Path, Path]:
    ranking_path = Path(f"{csv_prefix}_ranking.csv")
    pairwise_path = Path(f"{csv_prefix}_pairwise.csv")
    ranking_path.parent.mkdir(parents=True, exist_ok=True)

    ranking_fields = ["dataset", "metric", "posicao", "basis", "mean", "std",
                       "n_folds_paired", "friedman_statistic", "friedman_p",
                       "friedman_significant", "kendalls_w"]
    with open(ranking_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ranking_fields)
        writer.writeheader()
        for res in all_results:
            ranking = _ranked_bases(res)
            for pos, basis in enumerate(ranking, start=1):
                d = res["descriptive"][basis]
                writer.writerow({
                    "dataset": res["dataset"], "metric": res["metric"], "posicao": pos,
                    "basis": basis, "mean": d["mean"], "std": d["std"],
                    "n_folds_paired": res["n_folds_paired"],
                    "friedman_statistic": res["friedman"]["statistic"],
                    "friedman_p": res["friedman"]["p_value"],
                    "friedman_significant": res["friedman"]["significant"],
                    "kendalls_w": res["kendalls_w"],
                })

    pairwise_fields = ["dataset", "metric", "basis_1", "basis_2", "p_value", "p_value_holm",
                        "significant_after_holm", "effect_size_rank_biserial_r", "mean_diff_1_minus_2"]
    with open(pairwise_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=pairwise_fields)
        writer.writeheader()
        for res in all_results:
            for ph in res["posthoc"]:
                writer.writerow({
                    "dataset": res["dataset"], "metric": res["metric"],
                    "basis_1": ph["pair"][0], "basis_2": ph["pair"][1],
                    "p_value": ph["p_value"], "p_value_holm": ph["p_value_holm"],
                    "significant_after_holm": ph["significant_after_holm"],
                    "effect_size_rank_biserial_r": ph["effect_size_rank_biserial_r"],
                    "mean_diff_1_minus_2": ph["mean_diff"],
                })

    return ranking_path, pairwise_path


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--context", type=str, default="classifier")
    parser.add_argument("--datasets", nargs="+", default=None,
                         help="Se omitido, auto-descobre todos os datasets com resultados completos "
                              "(todas as basis) para o --experiment-name informado.")
    parser.add_argument("--bases", nargs="+", default=None,
                         help="Se omitido, auto-descobre todas as basis presentes em results/<context>/.")
    parser.add_argument("--metric", type=str, default="f1")
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    parser.add_argument("--csv-prefix", type=str, default=None,
                         help="Prefixo dos CSVs de saida (gera <prefixo>_ranking.csv e "
                              "<prefixo>_pairwise.csv). Default: reports/ranking_report")
    parser.add_argument("--md-output", type=str, default=None)
    args = parser.parse_args()

    if args.bases is None or args.datasets is None:
        print("Auto-descobrindo basis e datasets em "
              f"{RESULTS_ROOT / args.context} (experiment_name='{args.experiment_name}')...")
        auto_bases, auto_datasets, _ = discover_bases_and_datasets(args.context, args.experiment_name)
        bases = args.bases or auto_bases
        datasets = args.datasets or auto_datasets
    else:
        bases, datasets = args.bases, args.datasets

    print(f"\nBasis a comparar ({len(bases)}): {', '.join(bases)}")
    print(f"Datasets a comparar ({len(datasets)}): {', '.join(datasets)}\n")

    print("=" * 90)
    print(f"RANKING E SIGNIFICANCIA POR DATASET  (metrica={args.metric}, alpha={args.alpha})")
    print("=" * 90)

    all_results = []
    for dataset in datasets:
        try:
            res = analyze_dataset_metric(bases, dataset, args.metric, args.experiment_name, args.context, args.alpha)
        except DataAvailabilityError as e:
            print(f"\nDataset: {dataset}  [DADOS INSUFICIENTES] {e}")
            continue
        all_results.append(res)
        print_dataset_ranking(res)

    if not all_results:
        print("\nNenhum dataset pode ser analisado -- ver mensagens acima.")
        return

    n_sig_datasets = sum(1 for r in all_results if r["friedman"]["significant"])
    print(f"\n{'=' * 90}\nRESUMO GERAL\n{'=' * 90}")
    print(f"Em {n_sig_datasets}/{len(all_results)} datasets, o Friedman indicou diferenca "
          f"significativa entre as bases testadas.")
    sig_datasets = [r["dataset"] for r in all_results if r["friedman"]["significant"]]
    nonsig_datasets = [r["dataset"] for r in all_results if not r["friedman"]["significant"]]
    print(f"  Com diferenca significativa: {', '.join(sig_datasets) if sig_datasets else '-'}")
    print(f"  Sem diferenca significativa: {', '.join(nonsig_datasets) if nonsig_datasets else '-'}")

    print(f"\n{'=' * 90}\nTABELA DE 1o LUGAR POR BASIS (Markdown)\n{'=' * 90}\n")
    table_md = build_win_count_summary(all_results)
    print(table_md)

    csv_prefix = args.csv_prefix or str(REPORTS_ROOT / "ranking_report")
    ranking_path, pairwise_path = export_csv(all_results, csv_prefix)
    print(f"\n[salvo] {ranking_path}")
    print(f"[salvo] {pairwise_path}")

    md_path = Path(args.md_output) if args.md_output else Path(f"{csv_prefix}.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Ranking e significancia por dataset (metrica: {args.metric})\n\n")
        for res in all_results:
            ranking = _ranked_bases(res)
            fh.write(f"## {res['dataset']}\n\n")
            fh.write("| Posicao | Basis | Media | Desvio |\n|---|---|---|---|\n")
            for pos, basis in enumerate(ranking, start=1):
                d = res["descriptive"][basis]
                fh.write(f"| {pos} | {basis} | {d['mean']:.4f} | {d['std']:.4f} |\n")
            fr = res["friedman"]
            fh.write(f"\nFriedman: p={fr['p_value']:.4g} "
                      f"({'significativo' if fr['significant'] else 'nao significativo'})\n\n")
        fh.write("## Resumo -- 1o lugar por basis\n\n")
        fh.write(table_md + "\n")
    print(f"[salvo] {md_path}")


if __name__ == "__main__":
    main()