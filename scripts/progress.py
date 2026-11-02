"""Mostra o progresso do v3 (somente leitura; pode rodar enquanto o run_parallel roda).

    python scripts/progress.py --workers 6
    python scripts/progress.py --workers 6 --watch 60     # atualiza a cada 60 s

Unidade de progresso = ESCOPO (base x dataset x seed) concluido = tem test_results.json.
Por dataset x seed ha (n_bases + 2 baselines) escopos. ETA = trabalho restante estimado
com a duracao media por tamanho de dataset (logs/v3_progress.csv) / workers.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from run_v3 import TIERS  # noqa: E402
from models.fema import available_bases  # noqa: E402


def scope_dirs(results, bases, ds, exp):
    for b in bases:
        yield results / "classifier" / b / ds / exp
    for m in ("knn", "logreg"):
        yield results / "external_baselines" / m / ds / exp


def snapshot(a):
    results = Path(a.results)
    bases = available_bases()
    per_scope = len(bases) + 2
    size_of = {ds: sz for sz, lst in TIERS.items() for ds in lst}
    tasks = []
    for seed in sorted(a.seeds):
        for sz in ("large", "medium", "small"):
            if sz == "large" and seed not in a.large_seeds:
                continue
            tasks += [(seed, ds, sz) for ds in TIERS[sz]]

    done_scopes = partial = 0
    task_done, by_seed = {}, defaultdict(lambda: [0, 0])
    for seed, ds, sz in tasks:
        n_done = 0
        for d in scope_dirs(results, bases, ds, f"{a.name}_s{seed}"):
            if (d / "test_results.json").exists():
                n_done += 1
            elif d.exists():
                partial += 1
        task_done[(seed, ds)] = n_done
        done_scopes += n_done
        by_seed[seed][0] += n_done
        by_seed[seed][1] += per_scope
    total_scopes = len(tasks) * per_scope

    # duracao por tarefa (maior valor registrado: retomadas rapidas nao distorcem)
    dur = {}
    csv_path = ROOT / "logs" / "v3_progress.csv"
    if csv_path.exists():
        for row in csv.reader(open(csv_path, encoding="utf-8")):
            if len(row) >= 6 and row[4] == "ok":
                k = (int(row[1]), row[2])
                dur[k] = max(dur.get(k, 0), float(row[5]))
    avg = {}
    for sz in ("small", "medium", "large"):
        v = [t for (s, d), t in dur.items() if size_of.get(d) == sz]
        avg[sz] = sum(v) / len(v) if v else None

    remaining = 0.0
    unknown = False
    for seed, ds, sz in tasks:
        frac_left = 1 - task_done[(seed, ds)] / per_scope
        if frac_left > 0:
            if avg[sz] is None:
                unknown = True
            else:
                remaining += avg[sz] * frac_left

    pct = 100 * done_scopes / total_scopes if total_scopes else 0
    bar = "#" * int(pct / 2.5) + "-" * (40 - int(pct / 2.5))
    print(f"\n[{time.strftime('%H:%M:%S')}] [{bar}] {pct:5.1f}%  ({done_scopes}/{total_scopes} escopos concluidos, {partial} em andamento)")
    for seed in sorted(by_seed):
        d, t = by_seed[seed]
        print(f"  seed {seed}: {100*d/t:5.1f}%  ({d}/{t})")
    tasks_done = sum(1 for v in task_done.values() if v == per_scope)
    print(f"  datasets x seeds completos: {tasks_done}/{len(tasks)}")
    if remaining > 0:
        eta = remaining / max(1, a.workers)
        h = eta / 3600
        print(f"  ETA aprox.: {h:.1f} h ({'estimativa parcial: sem dados de algum tamanho' if unknown else f'{a.workers} workers'})")
    elif done_scopes == total_scopes:
        print("  CONCLUIDO.")
    elif not dur:
        print("  ETA: ainda sem tarefas concluidas para calibrar.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--large-seeds", nargs="*", type=int, default=[42])
    ap.add_argument("--name", default="v3")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--watch", type=int, default=0, help="atualiza a cada N segundos")
    a = ap.parse_args()
    while True:
        snapshot(a)
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()