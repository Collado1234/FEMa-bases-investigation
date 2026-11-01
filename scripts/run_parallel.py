"""Roda o v3 inteiro em paralelo (1 processo por dataset x seed), no terminal do VS Code.

    python scripts/run_parallel.py                 # tudo, com defaults do protocolo
    python scripts/run_parallel.py --workers 6
    python scripts/run_parallel.py --dry-run       # so' lista as tarefas

Protocolo (identico para TODOS os datasets/bases -> comparacao justa):
  random_search, n_iter=20 | CV estratificada repetida 10x3 | ranking por f1
  seeds 42..46 para small/medium; large so' com --large-seeds (default: 42)
Ordem: seed 42 completa primeiro (large antes: sao as mais longas), depois as outras seeds.
Idempotente: se cair, rode o MESMO comando de novo (checkpoint pula o que ja foi feito).
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_v3 import TIERS  # noqa: E402

LOGS = ROOT / "logs"
_procs = []


def run_task(task, passthrough, env):
    seed, ds, size = task
    LOGS.mkdir(exist_ok=True)
    log = LOGS / f"v3_s{seed}_{ds}.out"
    cmd = [sys.executable, "-u", str(ROOT / "scripts" / "run_v3.py"),
           "--datasets", ds, "--seeds", str(seed), *passthrough]
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
        _procs.append(p)
        rc = p.wait()
    dt = time.time() - t0
    failed = rc != 0 or "FALHA em" in log.read_text(encoding="utf-8", errors="ignore")
    status = "FALHOU" if failed else "ok"
    print(f"[{time.strftime('%H:%M:%S')}] {status:6} seed={seed} {ds:34} {dt/60:6.1f} min  (log: {log.name})", flush=True)
    with open(LOGS / "v3_progress.csv", "a", newline="") as f:
        csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"), seed, ds, size, status, round(dt)])
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--large-seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--tiers", nargs="+", default=["small", "medium", "large"], choices=list(TIERS))
    ap.add_argument("--name", default="v3")
    ap.add_argument("--n-iter", default="20")
    ap.add_argument("--n-splits", default="10")
    ap.add_argument("--n-repeats", default="3")
    ap.add_argument("--ranking-metric", default="f1")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    tasks = []
    for seed in sorted(a.seeds):
        for size in ("large", "medium", "small"):          # mais longas primeiro
            if size not in a.tiers:
                continue
            if size == "large" and seed not in a.large_seeds:
                continue
            tasks += [(seed, ds, size) for ds in TIERS[size]]

    passthrough = ["--name", a.name, "--n-iter", a.n_iter, "--n-splits", a.n_splits,
                   "--n-repeats", a.n_repeats, "--ranking-metric", a.ranking_metric]
    print(f"{len(tasks)} tarefas, {a.workers} em paralelo. Protocolo: random_search n_iter={a.n_iter}, "
          f"CV {a.n_splits}x{a.n_repeats}, ranking={a.ranking_metric}")
    if a.dry_run:
        for t in tasks:
            print(t)
        return

    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "PYTHONHASHSEED": "0"}
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            results = list(ex.map(lambda t: run_task(t, passthrough, env), tasks))
    except KeyboardInterrupt:
        print("Interrompido: encerrando processos filhos (rode de novo para retomar).")
        for p in _procs:
            if p.poll() is None:
                p.terminate()
        sys.exit(130)
    bad = [t for t, r in zip(tasks, results) if r != "ok"]
    print(f"FIM em {(time.time()-t0)/3600:.2f} h. Falhas: {bad or 'nenhuma'}")
    print("Proximo passo: python scripts/bundle_results.py --experiment-prefix v3 --out bundle_v3.json --strict")


if __name__ == "__main__":
    main()
