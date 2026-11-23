"""Gerador de results/ SINTETICO com verdade plantada, para testar a bateria estatistica, os relatorios
por dataset e a analise de k sem rodar experimentos reais.

Estrutura gravada (igual a do pipeline):
    <root>/classifier/<base>/<dataset>/v3_s<seed>/{summary.json, run_XXXX.json...}

Verdade plantada (aditiva, em todas as metricas):
  - laplacian > wendland_c2 > radial == rbf_gaussian > shepard > cauchy > cubic_spline >> multiquadratic
  - cada base tem um k preferido (compactas preferem k maior); o score cai com a distancia log(k/k_pref)
  - fold dificil/facil e' COMUM as bases (pareamento), ruido proprio da base e' pequeno
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tuning.grid_search import combo_id

DEFAULT_BASES = ["wendland_c2", "multiquadratic", "radial", "rbf_gaussian", "shepard", "laplacian", "cauchy", "cubic_spline"]
DEFAULT_DATASETS = ["iris", "wine", "seeds", "diabetes", "segment", "letter", "isolet", "zoo"]
EFFECT = {"laplacian": 0.012, "wendland_c2": 0.010, "radial": 0.008, "rbf_gaussian": 0.008, "shepard": 0.004,
          "cauchy": 0.0, "cubic_spline": -0.004, "multiquadratic": -0.06}
K_PREF = {"wendland_c2": 40, "cubic_spline": 40, "radial": 12, "rbf_gaussian": 12, "shepard": 8, "laplacian": 10,
          "cauchy": 9, "multiquadratic": 5}
METRIC_SHIFT = {"balanced_accuracy": 0.0, "f1": -0.02, "precision": 0.01, "recall": -0.005}
COMPACT = {"wendland_c2", "cubic_spline", "quartic_spline", "cosine"}


def make(root, bases=None, datasets=None, seeds=(42, 43), n_splits=10, n_repeats=3, n_combos=5, n_train_fold=378,
         k_max=60, noise=0.004, seed=0, effects=None, k_pref=None):
    root = Path(root)
    bases = bases or DEFAULT_BASES
    datasets = datasets or DEFAULT_DATASETS
    effects = effects or EFFECT
    k_pref = k_pref or K_PREF
    rng = np.random.default_rng(seed)
    n_val_fold = n_train_fold // (n_splits - 1)
    level = {d: rng.uniform(0.6, 0.93) for d in datasets}
    blocks = {(d, s, r, f): rng.normal(0, 0.03) for d in datasets for s in seeds for r in range(n_repeats) for f in range(n_splits)}
    for d in datasets:
        for s in seeds:
            for b in bases:
                scope = root / "classifier" / b / d / f"v3_s{s}"
                scope.mkdir(parents=True, exist_ok=True)
                combos = []
                for _ in range(n_combos):
                    k = int(round(math.exp(rng.uniform(math.log(3), math.log(k_max)))))
                    hp = {"k": k}
                    if b in COMPACT:
                        hp["h_rel"] = float(round(math.exp(rng.uniform(math.log(1.05), math.log(3.0))), 4))
                    elif b in ("radial",):
                        hp["z"] = float(round(rng.uniform(0.1, 3.0), 4))
                    combos.append(hp)
                seen, uniq = set(), []
                for hp in combos:                                  # dedupe como o pipeline
                    c = combo_id(hp)
                    if c not in seen:
                        seen.add(c); uniq.append(hp)
                idx, per_combo = 0, []
                for hp in uniq:
                    pen = 0.012 * abs(math.log(hp["k"] / k_pref[b]))
                    combo_shift = rng.normal(0, 0.002)
                    vals = {m: [] for m in METRIC_SHIFT}
                    for r in range(n_repeats):
                        for f in range(n_splits):
                            base = level[d] + effects[b] - pen + combo_shift + blocks[(d, s, r, f)] + rng.normal(0, noise)
                            met = {m: float(np.clip(base + sh + rng.normal(0, 0.002), 0, 1)) for m, sh in METRIC_SHIFT.items()}
                            met.update(accuracy=met["balanced_accuracy"], roc_auc=min(1.0, met["balanced_accuracy"] + 0.05),
                                       mcc=met["balanced_accuracy"] - 0.1)
                            idx += 1
                            for m in vals:
                                vals[m].append(met[m])
                            payload = {"context": "classifier", "basis": b, "experiment": f"v3_s{s}", "dataset": d,
                                       "combo_id": combo_id(hp), "hyperparameters": hp, "repeat_idx": r, "fold_idx": f,
                                       "seed": int(rng.integers(1 << 30)), "n_train_fold": n_train_fold,
                                       "n_val_fold": n_val_fold, "metrics": met,
                                       "diagnostics": {"frac_uniform_fallback": 0.0}, "execution_time_seconds": 0.01}
                            (scope / f"run_{idx:04d}.json").write_text(json.dumps(payload), encoding="utf-8")
                    entry = {"combo_id": combo_id(hp), "hyperparameters": hp, "n_runs": n_splits * n_repeats,
                             "metrics": {m: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "n": len(v)}
                                         for m, v in vals.items()},
                             "mean_execution_time_seconds": 0.01,
                             "diagnostics": {"frac_uniform_fallback": {"mean": 0.0}}}
                    per_combo.append(entry)
                per_combo.sort(key=lambda e: e["metrics"]["f1"]["mean"], reverse=True)
                summary = {"context": "classifier", "basis": b, "experiment": f"v3_s{s}", "dataset": d,
                           "primary_metric": "f1", "higher_is_better": True, "std_ddof": 1,
                           "n_combinations_evaluated": len(per_combo), "total_runs": len(per_combo) * n_splits * n_repeats,
                           "best_configuration": per_combo[0], "ranking": per_combo,
                           "config": {"n_splits": n_splits, "n_repeats": n_repeats, "master_seed": s,
                                      "search_space": {"k": ["logint", 3, k_max]}}}
                (scope / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return root
