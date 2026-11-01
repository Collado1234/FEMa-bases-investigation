"""Agrega results/**/ em UM unico JSON (bundle.json), sem perder rastreabilidade.

    python scripts/bundle_results.py --experiment-prefix v3 --out bundle_v3.json [--strict]

Para cada escopo (diretorio com run_*.json) grava: summary, test_results e TODAS as runs em
formato compacto (sem timestamp; hiperparametros guardados uma vez por combo_id).
Valida: runs duplicadas, runs faltando (esperado = combos x n_splits x n_repeats) e config_hash
misto. Problemas vao para meta.problems; com --strict o script sai com codigo 1.
Ordem deterministica => o proprio bundle e' reprodutivel (compare com diff/hash).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
DROP_RUN_KEYS = {"timestamp", "hyperparameters", "context", "basis", "model", "experiment", "dataset", "env"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--experiment-prefix", default="", help="so' escopos cujo ultimo segmento comeca com isto")
    ap.add_argument("--out", default="bundle.json")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    root = Path(args.results)
    problems, experiments, envs = [], [], {}
    for d in sorted({p.parent for p in root.rglob("run_*.json")}):
        scope = list(d.relative_to(root).parts)
        if not scope[-1].startswith(args.experiment_prefix):
            continue
        runs = [json.load(open(f, encoding="utf-8")) for f in sorted(d.glob("run_*.json"))]
        keys = [(r["combo_id"], r["repeat_idx"], r["fold_idx"]) for r in runs]
        if len(keys) != len(set(keys)):
            problems.append({"scope": scope, "problem": f"{len(keys) - len(set(keys))} runs duplicadas"})
        hashes = {r.get("config_hash") for r in runs}
        if len(hashes) > 1:
            problems.append({"scope": scope, "problem": f"config_hash misto: {sorted(map(str, hashes))}"})
        for r in runs:
            if r.get("env"):
                envs[json.dumps(r["env"], sort_keys=True)] = r["env"]

        summary = json.load(open(d / "summary.json", encoding="utf-8")) if (d / "summary.json").exists() else None
        test = json.load(open(d / "test_results.json", encoding="utf-8")) if (d / "test_results.json").exists() else None
        if summary is None:
            problems.append({"scope": scope, "problem": "sem summary.json"})
        else:
            cfg = summary.get("config") or {}
            if cfg:
                expected = summary["n_combinations_evaluated"] * cfg["n_splits"] * (
                    cfg["n_repeats"] if cfg["cv_strategy"] == "repeated_stratified_kfold" else 1)
                if len(set(keys)) != expected:
                    problems.append({"scope": scope, "problem": f"runs distintas {len(set(keys))} != esperado {expected}"})

        combos = {r["combo_id"]: r["hyperparameters"] for r in runs}
        compact = sorted(
            ({k: v for k, v in r.items() if k not in DROP_RUN_KEYS} for r in runs),
            key=lambda r: (r["combo_id"], r["repeat_idx"], r["fold_idx"]),
        )
        if test:
            test = {k: v for k, v in test.items() if k not in ("timestamp", "env")}
        experiments.append({"scope": scope, "summary": summary, "test_results": test,
                            "combos": dict(sorted(combos.items())), "runs": compact})

    bundle = {"meta": {"n_experiments": len(experiments), "n_runs": sum(len(e["runs"]) for e in experiments),
                       "environments": list(envs.values()), "problems": problems},
              "experiments": experiments}
    text = json.dumps(bundle, ensure_ascii=False, sort_keys=True, default=str)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"{args.out}: {bundle['meta']['n_experiments']} experimentos, {bundle['meta']['n_runs']} runs, "
          f"{len(text)/1e6:.1f} MB, sha256={hashlib.sha256(text.encode()).hexdigest()[:16]}, problemas={len(problems)}")
    for p in problems:
        print("  PROBLEMA:", p)
    if args.strict and problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
