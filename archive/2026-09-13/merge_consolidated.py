#!/usr/bin/env python3
"""
merge_consolidated.py

Junta varios consolidated_runs.csv (um gerado por PC, via audit_runs.py)
num unico CSV master, removendo duplicatas (caso a mesma run tenha sido
copiada/rodada em mais de um PC por engano) e mostrando quantas runs
UNICAS existem agora por (dataset, basis, combo_id) -- util especialmente
para confirmar que um dataset que estava dividido entre PCs (ex: Isolet)
ficou completo depois da juncao.

Uso:
    python merge_consolidated.py --csv consolidated_lab.csv --csv consolidated_meu.csv \
        --out consolidated_runs_master.csv

Opcional: se voce passar --expected-n, ele avisa quais combinacoes
(dataset, basis, combo_id) ainda tem MENOS runs que esse numero depois da
juncao (util quando voce sabe, por exemplo, que todo combo deveria ter 90
runs e quer achar rapido o que ainda esta faltando).
"""

import argparse
import sys

import pandas as pd


DEDUPE_KEYS = ["context", "basis", "experiment", "dataset", "combo_id",
               "repeat_idx", "fold_idx", "seed"]


def main():
    parser = argparse.ArgumentParser(description="Junta CSVs consolidados de multiplos PCs.")
    parser.add_argument("--csv", action="append", required=True,
                         help="Caminho de um consolidated_runs.csv. Pode repetir --csv varias vezes.")
    parser.add_argument("--out", default="consolidated_runs_master.csv")
    parser.add_argument("--expected-n", type=int, default=None,
                         help="Se informado, avisa quais (dataset,basis,combo_id) tem menos runs "
                              "que esse numero depois da juncao (ex: --expected-n 90).")
    args = parser.parse_args()

    dfs = []
    for path in args.csv:
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            sys.exit(f"Arquivo nao encontrado: {path}")
        df["__source_csv__"] = path
        dfs.append(df)
        print(f"  {path}: {len(df)} linhas")

    missing_keys = [k for k in DEDUPE_KEYS if k not in dfs[0].columns]
    if missing_keys:
        sys.exit(f"Colunas esperadas ausentes no CSV: {missing_keys}")

    merged = pd.concat(dfs, ignore_index=True)
    n_before = len(merged)

    duplicated_mask = merged.duplicated(subset=DEDUPE_KEYS, keep=False)
    n_duplicated_rows = int(duplicated_mask.sum())
    if n_duplicated_rows > 0:
        print(f"\n{n_duplicated_rows} linha(s) duplicada(s) encontradas (mesma run presente em mais "
              f"de um CSV) -- mantendo apenas uma copia de cada.")

    merged = merged.drop_duplicates(subset=DEDUPE_KEYS, keep="first")
    n_after = len(merged)
    print(f"\nTotal apos merge + dedupe: {n_after} runs unicas (eram {n_before} somando os CSVs brutos).")

    merged.drop(columns=["__source_csv__"], errors="ignore").to_csv(args.out, index=False)
    print(f"Salvo em: {args.out}")

    print("\nContagem final por (dataset, basis, combo_id):")
    counts = merged.groupby(["dataset", "basis", "combo_id"]).size().sort_values(ascending=False)
    print(counts.to_string())

    if args.expected_n is not None:
        below = counts[counts < args.expected_n]
        print(f"\n{'=' * 70}")
        if below.empty:
            print(f"Nenhuma combinacao abaixo de {args.expected_n} runs -- tudo completo!")
        else:
            print(f"AINDA FALTANDO (menos de {args.expected_n} runs) apos a juncao:")
            print(below.to_string())


if __name__ == "__main__":
    main()