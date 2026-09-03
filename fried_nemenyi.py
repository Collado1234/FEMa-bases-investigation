#!/usr/bin/env python3
"""
friedman_nemenyi.py

Implementa o procedimento ORIGINAL de Demsar (2006) para comparar
MULTIPLAS funcoes de base (classificadores) em MULTIPLOS datasets:

  1. Para cada dataset, pega o score agregado (media da metrica escolhida)
     da MELHOR config de cada funcao de base -> monta uma matriz
     dataset x basis (uma linha por dataset, uma coluna por basis).
  2. Friedman test (nao-parametrico, blocado por dataset) testa se ha
     diferenca significativa entre as funcoes de base.
  3. Se o Friedman rejeitar H0, aplica post-hoc de Nemenyi (todos-contra-
     todos), reportando o Critical Difference (CD) e quais pares diferem.

Isso e DIFERENTE do run_all.py --all-pairs: aquele faz testes pareados por
FOLD (Wilcoxon) dentro de cada dataset e corrige com Holm-Bonferroni --
uma analise mais fina (usa a variancia entre folds, que o Demsar nem
assume disponivel), mas NAO e o procedimento que o artigo recomenda
especificamente para "N classificadores x M datasets". Este script segue
a receita literal do artigo. O ideal e rodar os dois e comparar
conclusoes -- se convergirem, o resultado fica mais robusto.

Uso:
    python friedman_nemenyi.py --csv consolidated_runs.csv --metric f1
"""

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy import stats


def build_performance_matrix(df, metric):
    """Retorna (matrix, long_df). matrix: DataFrame dataset x basis com a
    media da metrica na MELHOR config de cada basis, dentro de cada
    dataset. long_df: mesma info em formato longo (com combo_id usado)."""
    records = []
    for dataset, df_ds in df.groupby("dataset"):
        for basis, df_b in df_ds.groupby("basis"):
            means = df_b.groupby("combo_id")[metric].mean()
            best_combo = means.idxmax()
            best_mean = means.max()
            records.append({"dataset": dataset, "basis": basis,
                             "combo_id": best_combo, "score": best_mean})
    long_df = pd.DataFrame(records)
    matrix = long_df.pivot(index="dataset", columns="basis", values="score")
    return matrix, long_df


def nemenyi_posthoc(matrix, alpha=0.05):
    """Nemenyi post-hoc a partir da matriz dataset x basis (sem NaN).
    Retorna (avg_ranks, cd, pairs_df)."""
    n, k = matrix.shape  # n = datasets, k = numero de basis functions
    ranks = matrix.rank(axis=1, ascending=False)  # rank 1 = melhor
    avg_ranks = ranks.mean(axis=0).sort_values()

    q_alpha = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))

    pairs = []
    for b1, b2 in itertools.combinations(avg_ranks.index, 2):
        diff = abs(avg_ranks[b1] - avg_ranks[b2])
        se = np.sqrt(k * (k + 1) / (6.0 * n))
        z = diff / se
        p = 2 * (1 - stats.norm.cdf(z))
        pairs.append({"basis_1": b1, "basis_2": b2,
                       "rank_diff": diff, "z": z, "p_value": p,
                       "significativo_cd": diff > cd})
    pairs_df = pd.DataFrame(pairs).sort_values("p_value").reset_index(drop=True)
    return avg_ranks, cd, pairs_df


def main():
    parser = argparse.ArgumentParser(description="Friedman + Nemenyi (Demsar 2006) para N basis x M datasets.")
    parser.add_argument("--csv", required=True, help="Caminho do consolidated_runs.csv")
    parser.add_argument("--metric", default="f1")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out-prefix", default="friedman_nemenyi")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    matrix, long_df = build_performance_matrix(df, args.metric)

    before_shape = matrix.shape
    matrix = matrix.dropna(axis=0, how="any")  # remove datasets sem alguma basis
    dropped_datasets = before_shape[0] - matrix.shape[0]
    if dropped_datasets > 0:
        print(f"AVISO: {dropped_datasets} dataset(s) removidos por nao terem todas as "
              f"basis functions avaliadas (matriz precisa ser completa para o Friedman).")

    n_datasets, n_basis = matrix.shape
    print(f"Matriz final: {n_datasets} datasets x {n_basis} basis functions.\n")
    print(matrix.round(4).to_string())

    if n_basis < 3:
        print("\nAVISO: com menos de 3 classificadores, Friedman/Nemenyi nao sao o teste indicado "
              "-- use o Wilcoxon pareado por fold (run_all.py) entre as duas basis diretamente.")
        return
    if n_datasets < 3:
        print("\nAVISO: com menos de 3 datasets, o poder do Friedman fica muito baixo -- "
              "resultado deve ser interpretado com cautela.")

    long_df.to_csv(f"{args.out_prefix}_best_combo_per_dataset_basis.csv", index=False)
    matrix.to_csv(f"{args.out_prefix}_matrix.csv")

    stat, p_value = stats.friedmanchisquare(*[matrix[col] for col in matrix.columns])
    print(f"\nFriedman test: statistic={stat:.4f}, p-value={p_value:.6f}")

    if p_value >= args.alpha:
        print(f"\nH0 NAO rejeitada (p >= {args.alpha}): nao ha evidencia de diferenca "
              f"significativa entre as funcoes de base nesse conjunto de datasets. "
              f"Post-hoc de Nemenyi nao e necessario nesse caso.")
        return

    print(f"\nH0 rejeitada (p < {args.alpha}): ha diferenca entre as funcoes de base. "
          f"Rodando post-hoc de Nemenyi...\n")

    avg_ranks, cd, pairs_df = nemenyi_posthoc(matrix, alpha=args.alpha)
    print("Ranks medios (menor = melhor):")
    print(avg_ranks.round(3).to_string())
    print(f"\nCritical Difference (CD) a alpha={args.alpha}: {cd:.4f}")
    print("\nComparacoes par-a-par (diferenca de rank > CD == significativo):")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(pairs_df.to_string(index=False))

    pairs_df.to_csv(f"{args.out_prefix}_posthoc.csv", index=False)
    print(f"\nArquivos salvos: {args.out_prefix}_matrix.csv, "
          f"{args.out_prefix}_posthoc.csv, {args.out_prefix}_best_combo_per_dataset_basis.csv")


if __name__ == "__main__":
    main()