"""Leitura de results/ (um run_XXXX.json por combo x repeat x fold) para a tabela longa
   dataset, basis, seed, repeat, fold, score, n_train, n_val
usando APENAS a configuracao vencedora do random search de cada (base, dataset, seed)
(summary.json -> best_configuration.combo_id).  Cacheia em CSV: reler ~10^5 JSONs e' lento."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SEED_RE = re.compile(r"_s(\d+)$")

FAMILIES = {
    "potencia/algebrica singular": ["shepard", "harmonic", "inverse_multiquadratic", "logarithmic"],
    "cauda pesada (racional)": ["cauchy", "lorentzian", "student_t", "rational_quadratic", "attention"],
    "exponencial/gaussiana": ["rbf_gaussian", "radial", "laplacian", "gen_exponential", "softmax_radial", "entropic"],
    "suporte compacto": ["cosine", "cubic_spline", "quartic_spline", "wendland_c2"],
    "nao-decrescente/sigmoide": ["multiquadratic", "sigmoidal"],
}

BASIS_TO_FAMILY = {b: f for f, bs in FAMILIES.items() for b in bs}

def _read_scope(args):
    d, metric = args
    try:
        summ = json.load(open(d / "summary.json", encoding="utf-8"))
    except FileNotFoundError:
        return None
    best = (summ.get("best_configuration") or {}).get("combo_id")
    if best is None:
        return None
    rows = []
    for f in d.glob("run_*.json"):
        r = json.load(open(f, encoding="utf-8"))
        if r.get("combo_id") != best:
            continue
        v = (r.get("metrics") or {}).get(metric)
        if v is None:
            continue
        rows.append((r["repeat_idx"], r["fold_idx"], float(v), r.get("n_train_fold"), r.get("n_val_fold")))
    rows = sorted(set(rows))     # dedupe defensivo (mesmo criterio do summary_builder)
    rk = [(c.get("metrics", {}).get(metric, {}).get("mean")) for c in summ.get("ranking", [])]
    rk = [x for x in rk if x is not None]
    tuning = dict(n_combos=len(rk), best_mean=max(rk) if rk else np.nan,
                  median_mean=float(np.median(rk)) if rk else np.nan)
    return d, rows, tuning