#!/usr/bin/env python3
"""
audit_runs.py

Verifica quais combinacoes (context/basis/experiment/dataset/combo_id) ja
possuem runs individuais salvas localmente, comparando com o total esperado
(via summary.json), e consolida as runs encontradas em um unico CSV para uso
em testes estatisticos pareados (Wilcoxon, Nadeau-Bengio, McNemar etc).

Uso:
    python audit_runs.py --root /caminho/para/pasta/com/jsons \
        --out consolidated_runs.csv

O script varre recursivamente `--root` procurando arquivos .json e
classifica cada um como:
  - "summary"    -> tem chaves "ranking" e "best_configuration"
  - "individual" -> tem chaves "repeat_idx", "fold_idx", "seed", "metrics"
  - "unknown"    -> nao bate com nenhum dos padroes acima (ex: test_results
                    com outro formato -- ajuste a funcao classify() se quiser
                    reconhecer esse schema tambem)

Depois:
  1. Agrupa summaries por (context, basis, experiment, dataset) e extrai o
     n_runs esperado por combo_id.
  2. Agrupa runs individuais pelas mesmas chaves + combo_id, deduplicando
     por (repeat_idx, fold_idx, seed) -- util se voce copiou a mesma run
     de mais de um PC sem perceber.
  3. Imprime um relatorio comparando esperado vs encontrado, por combo_id.
  4. Salva um CSV consolidado com todas as runs individuais encontradas,
     ja com uma coluna por hiperparametro e por metrica, pronto para
     pivotar por fold e rodar os testes estatisticos.
"""

import json
import argparse
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv


def classify(data):
    if isinstance(data, dict) and "ranking" in data and "best_configuration" in data:
        return "summary"
    if isinstance(data, dict) and {"repeat_idx", "fold_idx", "seed", "metrics"}.issubset(data.keys()):
        return "individual"
    return "unknown"


def scan_paths(root):
    """os.walk + string check e mais rapido que Path.rglob em arvores grandes,
    especialmente em pastas sincronizadas (OneDrive/Google Drive)."""
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".json"):
                yield os.path.join(dirpath, fn)


def read_and_classify(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return path, "error", None
    return path, classify(data), data


def load_json_files(root, workers=32, progress_every=2000):
    paths = list(scan_paths(root))
    total = len(paths)
    print(f"Encontrados {total} arquivos .json. Lendo com {workers} threads em paralelo...\n"
          f"(se estiver em pasta OneDrive/Drive, a primeira leitura pode ser lenta ate os "
          f"arquivos serem baixados localmente)\n")

    summaries, individuals, unknown = [], [], []
    processed = 0
    errors = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(read_and_classify, p): p for p in paths}
        for fut in as_completed(futures):
            path, kind, data = fut.result()
            processed += 1
            if kind == "summary":
                summaries.append((path, data))
            elif kind == "individual":
                individuals.append((path, data))
            elif kind == "error":
                errors += 1
            else:
                unknown.append(path)

            if processed % progress_every == 0 or processed == total:
                elapsed = time.time() - start
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = total - processed
                eta_min = (remaining / rate / 60) if rate > 0 else float("inf")
                print(f"  {processed}/{total} processados "
                      f"({rate:.1f} arquivos/s, ETA ~{eta_min:.1f} min, "
                      f"{errors} erros de leitura ate agora)")

    return summaries, individuals, unknown


def key_exp(d):
    return (d.get("context"), d.get("basis"), d.get("experiment"), d.get("dataset"))


def build_expected(summaries):
    expected = {}
    for _, s in summaries:
        exp_key = key_exp(s)
        combos = expected.setdefault(exp_key, {})
        for combo in s.get("ranking", []):
            combos[combo["combo_id"]] = combo["n_runs"]
    return expected


def build_found(individuals):
    # found[exp_key][combo_id] = set of (repeat_idx, fold_idx, seed) -> dedupe automatico
    found = defaultdict(lambda: defaultdict(set))
    for _, r in individuals:
        exp_key = key_exp(r)
        combo_id = r.get("combo_id")
        fold_key = (r.get("repeat_idx"), r.get("fold_idx"), r.get("seed"))
        found[exp_key][combo_id].add(fold_key)
    return found


def print_report(expected, found):
    print("=" * 92)
    print("RELATORIO DE COBERTURA DE RUNS INDIVIDUAIS")
    print("=" * 92)
    all_keys = set(expected.keys()) | set(found.keys())
    for exp_key in sorted(all_keys, key=lambda k: [str(x) for x in k]):
        context, basis, experiment, dataset = exp_key
        print(f"\n[context={context} | basis={basis} | experiment={experiment} | dataset={dataset}]")
        exp_combos = expected.get(exp_key, {})
        found_combos = found.get(exp_key, {})
        combo_ids = sorted(
            set(exp_combos.keys()) | set(found_combos.keys()),
            key=lambda c: exp_combos.get(c, -1),
            reverse=True,
        )
        if not combo_ids:
            print("  (nenhuma informacao encontrada)")
            continue
        print(f"  {'combo_id':<12}{'esperado':>10}{'encontrado':>12}{'faltando':>10}  status")
        for combo_id in combo_ids:
            n_expected = exp_combos.get(combo_id)
            n_found = len(found_combos.get(combo_id, set()))
            if isinstance(n_expected, int):
                n_missing = max(n_expected - n_found, 0)
                if n_missing == 0:
                    status = "OK"
                elif n_found > 0:
                    status = "PARCIAL"
                else:
                    status = "FALTA TUDO"
                n_exp_str, n_miss_str = str(n_expected), str(n_missing)
            else:
                status = "SEM SUMMARY"
                n_exp_str, n_miss_str = "?", "?"
            print(f"  {combo_id:<12}{n_exp_str:>10}{n_found:>12}{n_miss_str:>10}  {status}")


def export_csv(individuals, out_path):
    if not individuals:
        print("\nNenhuma run individual encontrada para exportar.")
        return

    seen = set()
    rows = []
    for path, r in individuals:
        dedupe_key = (
            r.get("context"), r.get("basis"), r.get("experiment"), r.get("dataset"),
            r.get("combo_id"), r.get("repeat_idx"), r.get("fold_idx"), r.get("seed"),
        )
        if dedupe_key in seen:
            continue  # mesma run copiada de outro PC
        seen.add(dedupe_key)

        row = {
            "context": r.get("context"),
            "basis": r.get("basis"),
            "experiment": r.get("experiment"),
            "dataset": r.get("dataset"),
            "combo_id": r.get("combo_id"),
            "repeat_idx": r.get("repeat_idx"),
            "fold_idx": r.get("fold_idx"),
            "seed": r.get("seed"),
            "n_train_fold": r.get("n_train_fold"),
            "n_val_fold": r.get("n_val_fold"),
            "execution_time_seconds": r.get("execution_time_seconds"),
            "timestamp": r.get("timestamp"),
            "source_file": str(path),
        }
        for hp_name, hp_val in (r.get("hyperparameters") or {}).items():
            row[f"hp_{hp_name}"] = hp_val
        for metric_name, metric_val in (r.get("metrics") or {}).items():
            row[metric_name] = metric_val
        rows.append(row)

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} runs individuais unicas consolidadas em: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Audita e consolida runs individuais de experimentos.")
    parser.add_argument("--root", required=True, help="Pasta raiz para varrer recursivamente por .json")
    parser.add_argument("--out", default="consolidated_runs.csv", help="Caminho do CSV consolidado de saida")
    parser.add_argument("--workers", type=int, default=32,
                         help="Threads paralelas para leitura (aumente se estiver em rede/OneDrive)")
    parser.add_argument("--progress-every", type=int, default=2000,
                         help="Intervalo (em arquivos) para imprimir progresso")
    args = parser.parse_args()

    summaries, individuals, unknown = load_json_files(
        args.root, workers=args.workers, progress_every=args.progress_every
    )
    print(f"\nArquivos encontrados: {len(summaries)} summaries, "
          f"{len(individuals)} runs individuais, {len(unknown)} nao reconhecidos.\n")

    expected = build_expected(summaries)
    found = build_found(individuals)
    print_report(expected, found)
    export_csv(individuals, args.out)


if __name__ == "__main__":
    main()