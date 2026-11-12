"""Camada C - teste de DeLong (1988) para comparar AUCs CORRELACIONADAS no mesmo conjunto de teste.
Implementacao vetorizada de Sun & Xu (2014).  Binario: AUC direta.  Multiclasse: um-contra-resto por classe,
Holm entre classes; o p combinado = menor p ajustado (H1: difere em ao menos uma classe)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from . import primitives as P


def _delong(y: np.ndarray, S: np.ndarray):
    """y in {0,1}; S (k x n). -> aucs (k,), cov (k x k)."""
    pos, neg = S[:, y == 1], S[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    if m < 2 or n < 2:
        raise ValueError("DeLong exige >= 2 positivos e >= 2 negativos")
    tx = np.apply_along_axis(stats.rankdata, 1, pos)
    ty = np.apply_along_axis(stats.rankdata, 1, neg)
    tz = np.apply_along_axis(stats.rankdata, 1, np.concatenate([pos, neg], axis=1))
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def delong_pairs(y_true: np.ndarray, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    """scores[base] = (n,) [binario, escore da classe positiva] ou (n, C) [probabilidades]."""
    bases = list(scores)
    y_true = np.asarray(y_true)
    classes = np.unique(y_true)
    binary = len(classes) == 2
    per_class = []
    for ci, c in enumerate(classes):
        if binary and ci == 0:
            continue                                        # binario: so' a classe positiva (a maior)
        yb = (y_true == c).astype(int)
        S = np.stack([(scores[b] if scores[b].ndim == 1 else scores[b][:, ci]) for b in bases])
        try:
            aucs, cov = _delong(yb, S)
        except ValueError:
            continue
        per_class.append((c, aucs, cov))
    rows = []
    for i, j in itertools.combinations(range(len(bases)), 2):
        zs, ps, dA = [], [], []
        for c, aucs, cov in per_class:
            var = cov[i, i] + cov[j, j] - 2 * cov[i, j]
            diff = aucs[i] - aucs[j]
            z = 0.0 if (var <= 1e-18 and abs(diff) < 1e-12) else diff / np.sqrt(max(var, 1e-18))
            zs.append(z); ps.append(float(2 * stats.norm.sf(abs(z)))); dA.append(diff)
        if not ps:
            continue
        p_adj = P.holm(np.array(ps))
        rows.append(dict(a=bases[i], b=bases[j], auc_diff_macro=float(np.mean(dA)), n_classes_testadas=len(ps),
                         p_min_holm=float(p_adj.min()), classe_mais_discrepante=str(per_class[int(np.argmin(ps))][0]),
                         auc_a_macro=float(np.mean([pc[1][i] for pc in per_class])),
                         auc_b_macro=float(np.mean([pc[1][j] for pc in per_class]))))
    out = pd.DataFrame(rows)
    out["p_holm_pares"] = P.holm(out["p_min_holm"].values)
    return out
