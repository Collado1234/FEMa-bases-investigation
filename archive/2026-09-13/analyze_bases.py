"""
Analisa o CSV de comparação de "bases" (kernels) de um classificador,
com foco em:
  1. Detectar grupos de bases com métricas IDÊNTICAS (possível bug de kernel)
  2. Verificar se esses empates coincidem com o mesmo valor de k
  3. Rankear as bases por métrica
  4. Gerar um resumo (desbalanceamento accuracy x balanced_accuracy/mcc)

Uso:
    python analyze_bases.py caminho_para_o.csv
"""

import sys
import json
import csv
from collections import defaultdict

METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc"]
# nº de casas decimais usado para agrupar "resultados iguais" (evita falso negativo
# por erro de ponto flutuante, mas ainda é bem rigoroso)
ROUND_DIGITS = 9


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("status") != "ok":
                continue
            r["hyperparams"] = json.loads(r["best_hyperparameters"])
            r["k"] = r["hyperparams"].get("k")
            for m in METRICS:
                r[m] = float(r[m])
            rows.append(r)
    return rows


def find_duplicate_groups(rows):
    """Agrupa bases cujas métricas batem exatamente (até ROUND_DIGITS)."""
    groups = defaultdict(list)
    for r in rows:
        key = tuple(round(r[m], ROUND_DIGITS) for m in METRICS)
        groups[key].append(r)
    # só interessa quem tem mais de 1 basis no grupo
    return [g for g in groups.values() if len(g) > 1]


def print_duplicate_groups(dup_groups):
    print("=" * 78)
    print("GRUPOS DE BASES COM MÉTRICAS IDÊNTICAS")
    print("=" * 78)
    if not dup_groups:
        print("Nenhum grupo duplicado encontrado.")
        return

    all_same_k_count = 0
    for i, group in enumerate(sorted(dup_groups, key=lambda g: -len(g)), 1):
        bases = [r["basis"] for r in group]
        ks = {r["k"] for r in group}
        same_k = len(ks) == 1
        all_same_k_count += same_k

        print(f"\nGrupo {i}: {len(group)} bases empatadas -> {bases}")
        print(f"  k usados: {sorted(ks)}  {'(TODOS IGUAIS -> suspeito!)' if same_k else '(k diferentes)'}")
        for r in group:
            extra = {k: v for k, v in r["hyperparams"].items() if k != "k"}
            print(f"    - {r['basis']:<22} k={r['k']:<4} outros_params={extra}")
        print(f"  mcc={group[0]['mcc']:.6f}  accuracy={group[0]['accuracy']:.6f}")

    print(f"\n>> {all_same_k_count}/{len(dup_groups)} grupos duplicados têm exatamente o MESMO k")
    print("   entre bases com hiperparâmetros próprios (epsilon/beta/c/z/nu) DIFERENTES.")
    print("   Isso é forte indício de que o parâmetro específico do kernel não está")
    print("   entrando de fato no cálculo dos pesos (só o k estaria influenciando o resultado).")


def print_rankings(rows):
    print("\n" + "=" * 78)
    print("RANKING POR MÉTRICA (do melhor para o pior)")
    print("=" * 78)
    for m in METRICS:
        ranked = sorted(rows, key=lambda r: r[m], reverse=True)
        print(f"\n-- {m} --")
        for pos, r in enumerate(ranked, 1):
            print(f"  {pos:>2}. {r['basis']:<22} {r[m]:.6f}")


def print_imbalance_summary(rows):
    print("\n" + "=" * 78)
    print("ALERTA DE DESBALANCEAMENTO (accuracy vs balanced_accuracy/mcc)")
    print("=" * 78)
    print(f"{'basis':<22}{'accuracy':>10}{'bal_acc':>10}{'mcc':>10}{'gap(acc-bal)':>15}")
    for r in sorted(rows, key=lambda r: r["accuracy"] - r["balanced_accuracy"], reverse=True):
        gap = r["accuracy"] - r["balanced_accuracy"]
        print(f"{r['basis']:<22}{r['accuracy']:>10.4f}{r['balanced_accuracy']:>10.4f}{r['mcc']:>10.4f}{gap:>15.4f}")


def print_best_overall(rows):
    print("\n" + "=" * 78)
    print("MELHOR BASIS POR MÉTRICA (resumo rápido)")
    print("=" * 78)
    for m in METRICS:
        best = max(rows, key=lambda r: r[m])
        print(f"  {m:<20} -> {best['basis']:<22} ({best[m]:.6f})")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    rows = load_rows(csv_path)
    print(f"Carregadas {len(rows)} bases com status 'ok' de {csv_path}\n")

    dup_groups = find_duplicate_groups(rows)
    print_duplicate_groups(dup_groups)
    print_rankings(rows)
    print_imbalance_summary(rows)
    print_best_overall(rows)


if __name__ == "__main__":
    main()