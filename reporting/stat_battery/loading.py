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

def load_fold_scores(results_root, prefix="v3", metric="balanced_accuracy", context="classifier",
                     cache: str | None = None, workers: int = 8):
    """-> (folds DataFrame, tuning DataFrame).  Com `cache`, reaproveita/gera <cache>_folds.csv e _tuning.csv."""
    if cache and Path(f"{cache}_folds.csv").exists():
        return pd.read_csv(f"{cache}_folds.csv"), pd.read_csv(f"{cache}_tuning.csv")
    root = Path(results_root) / context
    scopes = [p for p in root.glob("*/*/*") if p.is_dir() and p.name.startswith(prefix) and SEED_RE.search(p.name)]
    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(_read_scope, [(s, metric) for s in scopes]))
    frows, trows = [], []
    for res in out:
        if res is None:
            continue
        d, rows, tun = res
        basis, ds, exp = d.parts[-3], d.parts[-2], d.parts[-1]
        seed = int(SEED_RE.search(exp).group(1))
        frows += [dict(dataset=ds, basis=basis, seed=seed, repeat=r, fold=f, score=v, n_train=nt, n_val=nv)
                  for r, f, v, nt, nv in rows]
        trows.append(dict(dataset=ds, basis=basis, seed=seed, **tun))
    folds, tuning = pd.DataFrame(frows), pd.DataFrame(trows)
    if cache:
        folds.to_csv(f"{cache}_folds.csv", index=False)
        tuning.to_csv(f"{cache}_tuning.csv", index=False)
    return folds, tuning

def complete_blocks(folds: pd.DataFrame):
    """Mantem, por dataset, so' os blocos (seed, repeat, fold) presentes em TODAS as bases
    (necessario para o pareamento) e so' as bases presentes em todos os blocos. Retorna
    (folds filtrado, lista de problemas)."""
    problems, keep = [], []
    all_bases = sorted(folds["basis"].unique())
    for ds, g in folds.groupby("dataset"):
        piv = g.pivot_table(index=["seed", "repeat", "fold"], columns="basis", values="score")
        missing = [b for b in all_bases if b not in piv.columns]
        if missing:
            problems.append((ds, f"bases ausentes: {missing[:5]}"))
            continue
        full = piv.dropna()
        if len(full) < len(piv):
            problems.append((ds, f"{len(piv) - len(full)} blocos incompletos descartados"))
        keep.append(full.reset_index().assign(dataset=ds))
    if not keep:
        return pd.DataFrame(), problems, pd.DataFrame()
    wide = pd.concat(keep, ignore_index=True)
    meta = folds.groupby("dataset")[["n_train", "n_val"]].mean()
    return wide, problems, meta


def load_groups(path):
    g = pd.read_csv(path, encoding="utf-8-sig").set_index("dataset")
    g["log_amostras"] = np.log10(g["n_amostras"])
    g["log_features"] = np.log10(g["n_features"])
    g["log_desbal"] = np.log10(g["indice_desbalanceamento"])
    return g