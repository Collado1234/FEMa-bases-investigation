"""Camada B - avaliacao por GRUPO de datasets. Unidade de analise = dataset (Demsar 2006):
M e' uma matriz datasets x bases (media do escore por dataset). Nao ha pseudo-replicacao."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from . import primitives as P
from .loading import FAMILIES

CRITERIA = {"n_features": "cat_features", "n_amostras": "cat_amostras", "n_classes": "cat_classes",
            "desbalanceamento": "cat_desbalanceamento", "IHDD": "cat_IHDD"}
COVARIATES = {"log_amostras": "log10(n amostras)", "log_features": "log10(n features)",
              "n_classes": "n classes", "log_desbal": "log10(indice desbal.)"}


def rank_matrix(M: pd.DataFrame, higher_is_better=True) -> pd.DataFrame:
    return M.apply(lambda r: pd.Series(stats.rankdata(-r.values if higher_is_better else r.values), index=r.index), axis=1)


def family_matrix(M: pd.DataFrame) -> pd.DataFrame:
    """Escore de cada familia = media das bases membros (nao 'melhor da familia', que favoreceria familias grandes)."""
    cols = {f: [b for b in bs if b in M.columns] for f, bs in FAMILIES.items()}
    return pd.DataFrame({f: M[bs].mean(axis=1) for f, bs in cols.items() if bs})


def nemenyi_cd(k: int, N: int, alpha=0.05) -> float:
    q = stats.studentized_range.ppf(1 - alpha, k, 10**6) / math.sqrt(2)
    return float(q * math.sqrt(k * (k + 1) / (6 * N)))


def per_group_friedman(M: pd.DataFrame, groups: pd.DataFrame, alpha=0.05):
    """Para cada criterio e nivel: Friedman+Iman-Davenport entre as bases DENTRO do grupo, postos medios, CD."""
    summ, long = [], []
    for crit, col in CRITERIA.items():
        for lvl, idx in groups.loc[M.index].groupby(col).groups.items():
            sub = M.loc[idx]
            n, k = sub.shape
            ranks = pd.Series(P.avg_ranks(sub.to_numpy()), index=sub.columns)
            fr = P.friedman_id(sub.to_numpy()) if n >= 3 else dict(F=np.nan, p_F=np.nan)
            summ.append(dict(criterio=crit, nivel=lvl, n_datasets=n, k=k, F=fr["F"], p_F=fr["p_F"],
                             CD_nemenyi=nemenyi_cd(k, n, alpha) if n >= 2 else np.nan,
                             top3=", ".join(ranks.nsmallest(3).index), pior=ranks.idxmax(),
                             datasets=", ".join(sorted(idx))))
            long += [dict(criterio=crit, nivel=lvl, basis=b, posto_medio=r, n_datasets=n) for b, r in ranks.items()]
    s = pd.DataFrame(summ)
    s["p_F_holm"] = np.nan
    ok = s["p_F"].notna()
    s.loc[ok, "p_F_holm"] = P.holm(s.loc[ok, "p_F"].values)
    return s, pd.DataFrame(long)


def rank_dependence_permutation(R: pd.DataFrame, labels: pd.Series, n_perm=9999, seed=0) -> dict:
    """H0: o PERFIL de postos das bases nao depende do grupo do dataset.
    T = sum_g n_g * || rbar_g - rbar ||^2  (dispersao entre grupos dos vetores de posto medio).
    p-valor: permuta rotulos de grupo entre datasets (tamanhos fixos). Nao-parametrico, sem hipotese
    de normalidade, valido com n pequeno; e' um TESTE GLOBAL (todas as bases de uma vez)."""
    X = R.to_numpy()
    lab = labels.loc[R.index].to_numpy()
    levels = np.unique(lab)
    grand = X.mean(axis=0)

    def T(l):
        t = 0.0
        for g in levels:
            m = l == g
            t += m.sum() * ((X[m].mean(axis=0) - grand) ** 2).sum()
        return t

    obs = T(lab)
    rng = np.random.default_rng(seed)
    null = np.array([T(rng.permutation(lab)) for _ in range(n_perm)])
    return dict(T=float(obs), p=float((1 + (null >= obs - 1e-12).sum()) / (n_perm + 1)),
                n_groups=len(levels), min_group=int(min((lab == g).sum() for g in levels)))


def per_basis_kruskal(R: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    lab = labels.loc[R.index]
    rows = []
    for b in R.columns:
        samples = [R.loc[lab == g, b].values for g in lab.unique()]
        samples = [s for s in samples if len(s) > 0]
        try:
            H, p = stats.kruskal(*samples)
        except ValueError:      # todos os postos iguais
            H, p = 0.0, 1.0
        means = R[b].groupby(lab).mean()
        rows.append(dict(basis=b, H=H, p=p, melhor_grupo=means.idxmin(), pior_grupo=means.idxmax(),
                         posto_melhor=means.min(), posto_pior=means.max()))
    out = pd.DataFrame(rows)
    out["p_holm"] = P.holm(out["p"].values)
    return out.sort_values("p")


def spearman_covariates(R: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b in R.columns:
        for c, nome in COVARIATES.items():
            x = groups.loc[R.index, c]
            rho, p = stats.spearmanr(x, R[b]) if R[b].nunique() > 1 else (0.0, 1.0)
            rows.append(dict(basis=b, covariavel=nome, rho_spearman=rho, p=p))
    out = pd.DataFrame(rows)
    # postos: rho<0 = quanto MAIOR a covariavel, MELHOR (menor posto) a base
    out["p_bh"] = P.benjamini_hochberg(out["p"].fillna(1.0).values)
    return out.sort_values("p")


def run_group_analysis(M: pd.DataFrame, groups: pd.DataFrame, alpha=0.05, n_perm=9999):
    """M: datasets x bases. Roda tudo no nivel de base E no nivel de familia."""
    out = {}
    for nivel, X in (("base", M), ("familia", family_matrix(M))):
        R = rank_matrix(X)
        s, long = per_group_friedman(X, groups, alpha)
        dep = pd.DataFrame([dict(criterio=c, **rank_dependence_permutation(R, groups.loc[R.index, col], n_perm))
                            for c, col in CRITERIA.items()])
        dep["p_holm"] = P.holm(dep["p"].values)
        kw = pd.concat([per_basis_kruskal(R, groups.loc[R.index, col]).assign(criterio=c)
                        for c, col in CRITERIA.items()], ignore_index=True)
        kw["p_holm_global"] = P.holm(kw["p"].values)
        out[nivel] = dict(friedman=s, ranks=long, dependencia=dep, kruskal=kw,
                          spearman=spearman_covariates(R, groups), rank_matrix=R)
    return out
