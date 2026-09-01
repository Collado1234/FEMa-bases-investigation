#!/usr/bin/env python3
"""
run_all.py

Roda a comparacao estatistica (Wilcoxon + t pareado corrigido de
Nadeau-Bengio) para TODOS os datasets presentes no consolidated_runs.csv
de uma vez, e resume tudo em uma tabela final com correcao de
multiplas comparacoes (Holm-Bonferroni).

Reaproveita a funcao compare_pair() de compare_basis.py -- os dois
arquivos precisam estar na mesma pasta.

MODOS DE USO:

  1) Basis fixas, todos os datasets (o caso mais comum: "X e melhor que Y?"):

     python run_all.py --csv consolidated_runs.csv \
         --basis-x attention --basis-y lorentzian \
         --out resultados_todos_datasets.csv

  2) Todas as combinacoes de basis, em todos os datasets (exploratorio):

     python run_all.py --csv consolidated_runs.csv --all-pairs \
         --out resultados_todos_pares.csv

Em ambos os casos, ao final e impressa uma tabela resumo com o p-valor do
Wilcoxon ORIGINAL e o p-valor AJUSTADO pela correcao de Holm-Bonferroni
(alpha default = 0.05), alem de indicar quais comparacoes permanecem
significativas depois da correcao.
"""

import argparse
import itertools
import sys

import numpy as np
import pandas as pd

from compare_basis import compare_pair


def holm_bonferroni(p_values, alpha=0.05):
    """Implementacao simples da correcao de Holm-Bonferroni (mais poderosa
    que Bonferroni simples, controla o FWER). Retorna listas (p_ajustado,
    rejeitado) na ordem original de p_values. NaNs sao ignorados (mantidos
    como NaN, nao rejeitados)."""
    n = len(p_values)
    indexed = [(p, i) for i, p in enumerate(p_values) if pd.notna(p)]
    indexed.sort(key=lambda x: x[0])

    adjusted = [float("nan")] * n
    rejected = [False] * n
    m = len(indexed)
    max_adj_so_far = 0.0
    for rank, (p, orig_idx) in enumerate(indexed):
        factor = m - rank
        adj_p = min(1.0, p * factor)
        adj_p = max(adj_p, max_adj_so_far)  # garante monotonicidade
        max_adj_so_far = adj_p
        adjusted[orig_idx] = adj_p
        rejected[orig_idx] = adj_p < alpha
    return adjusted, rejected


def summarize_by_pair(results_df, alpha=0.05):
    """Agrupa por (basis_x, basis_y) e resume: em quantos datasets a
    diferenca foi significativa, com e sem correcao de Holm-Bonferroni,
    e lista explicitamente quais datasets caem em cada grupo -- pronto
    para apresentar tipo 'em X datasets teve diferenca significativa,
    em Y nao'."""
    rows = []
    for (basis_x, basis_y), group in results_df.groupby(["basis_x", "basis_y"]):
        group = group.sort_values("dataset")
        sig_raw_mask = group["wilcoxon_p"] < alpha
        sig_holm_mask = group["significativo_apos_correcao"]

        # lado que "ganhou" em cada dataset significativo (baseado na diferenca de medias)
        vencedor = np.where(group["mean_diff_x_minus_y"] > 0, basis_x, basis_y)

        datasets_sig_holm = group.loc[sig_holm_mask, "dataset"].tolist()
        datasets_nao_sig_holm = group.loc[~sig_holm_mask, "dataset"].tolist()
        vencedor_sig = pd.Series(vencedor, index=group.index)[sig_holm_mask]

        rows.append({
            "basis_x": basis_x,
            "basis_y": basis_y,
            "n_datasets_testados": len(group),
            "significativos_sem_correcao": int(sig_raw_mask.sum()),
            "significativos_com_holm": int(sig_holm_mask.sum()),
            "datasets_significativos_holm": ", ".join(datasets_sig_holm) if datasets_sig_holm else "-",
            "vencedor_por_dataset_significativo": ", ".join(
                f"{d}:{v}" for d, v in zip(datasets_sig_holm, vencedor_sig)
            ) if datasets_sig_holm else "-",
            "datasets_nao_significativos": ", ".join(datasets_nao_sig_holm) if datasets_nao_sig_holm else "-",
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Roda comparacoes de basis para todos os datasets e resume com Holm-Bonferroni.")
    parser.add_argument("--csv", required=True, help="Caminho do consolidated_runs.csv")
    parser.add_argument("--basis-x", default=None, help="Basis X fixa (ignora --all-pairs)")
    parser.add_argument("--basis-y", default=None, help="Basis Y fixa (ignora --all-pairs)")
    parser.add_argument("--all-pairs", action="store_true",
                         help="Testa todas as combinacoes de basis presentes em cada dataset, "
                              "em vez de um par fixo")
    parser.add_argument("--metric", default="f1")
    parser.add_argument("--alpha", type=float, default=0.05, help="Nivel de significancia (default 0.05)")
    parser.add_argument("--out", default="resumo_comparacoes.csv", help="CSV final com todos os resultados")
    args = parser.parse_args()

    if not args.all_pairs and (not args.basis_x or not args.basis_y):
        sys.exit("Informe --basis-x e --basis-y, ou use --all-pairs para testar todas as combinacoes.")

    df_all = pd.read_csv(args.csv)
    datasets = sorted(df_all["dataset"].unique())
    print(f"{len(datasets)} datasets encontrados: {datasets}\n")

    results = []
    for dataset in datasets:
        df_ds = df_all[df_all["dataset"] == dataset].copy()
        basis_available = sorted(df_ds["basis"].unique())

        if args.all_pairs:
            pairs = list(itertools.combinations(basis_available, 2))
        else:
            pairs = [(args.basis_x, args.basis_y)]

        for basis_x, basis_y in pairs:
            print(f"--- dataset={dataset} | {basis_x} vs {basis_y} ---")
            result = compare_pair(df_ds, dataset, basis_x, basis_y, metric=args.metric, verbose=True)
            if result is not None:
                results.append(result)
            print()

    if not results:
        sys.exit("Nenhuma comparacao valida foi produzida. Verifique os nomes das basis e o CSV.")

    results_df = pd.DataFrame(results)

    # Holm-Bonferroni sobre os p-valores do Wilcoxon (metodo principal recomendado por Demsar 2006)
    adj_p, rejected = holm_bonferroni(results_df["wilcoxon_p"].tolist(), alpha=args.alpha)
    results_df["wilcoxon_p_holm"] = adj_p
    results_df["significativo_apos_correcao"] = rejected

    results_df.to_csv(args.out, index=False)

    print("=" * 100)
    print(f"RESUMO FINAL ({len(results_df)} comparacoes, alpha={args.alpha}, correcao Holm-Bonferroni)")
    print("=" * 100)
    display_cols = ["dataset", "basis_x", "combo_x", "basis_y", "combo_y",
                     "mean_x", "mean_y", "n_paired", "wilcoxon_p", "wilcoxon_p_holm",
                     "significativo_apos_correcao"]
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(results_df[display_cols].to_string(index=False))

    n_sig = int(results_df["significativo_apos_correcao"].sum())
    print(f"\n{n_sig}/{len(results_df)} comparacoes permanecem significativas apos correcao de Holm-Bonferroni.")
    print(f"\nTabela completa salva em: {args.out}")

    # Resumo por par de basis -- pronto para apresentar (ex: "em X datasets houve
    # diferenca significativa, em Y nao")
    pair_summary = summarize_by_pair(results_df, alpha=args.alpha)
    pair_summary_path = args.out.rsplit(".", 1)[0] + "_resumo_por_par.csv"
    pair_summary.to_csv(pair_summary_path, index=False)

    print("\n" + "=" * 100)
    print("RESUMO POR PAR DE BASIS (pronto para apresentar)")
    print("=" * 100)
    for _, row in pair_summary.iterrows():
        print(f"\n{row['basis_x']} vs {row['basis_y']}:")
        print(f"  - {row['significativos_com_holm']}/{row['n_datasets_testados']} datasets com diferenca "
              f"SIGNIFICATIVA (apos Holm-Bonferroni)")
        print(f"    -> {row['datasets_significativos_holm']}")
        if row['datasets_significativos_holm'] != "-":
            print(f"    -> vencedor em cada um: {row['vencedor_por_dataset_significativo']}")
        print(f"  - {row['n_datasets_testados'] - row['significativos_com_holm']}/{row['n_datasets_testados']} "
              f"datasets SEM diferenca significativa")
        print(f"    -> {row['datasets_nao_significativos']}")
        print(f"  (sem correcao, seriam {row['significativos_sem_correcao']}/{row['n_datasets_testados']} "
              f"significativos -- Holm-Bonferroni e mais conservador)")

    print(f"\nResumo por par salvo em: {pair_summary_path}")


if __name__ == "__main__":
    main()