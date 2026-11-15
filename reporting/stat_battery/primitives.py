"""Primitivas estatisticas"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy import stats

def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    if m == 0:
        return p

    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0

    for i, idx in enumerate(order):
        run = max(run, (m-i) * p[idx])
        adj[idx] = min(1.0, run)
    return adj

def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    run = 1.0
    for rank_from_top, idx in enumerate(order):
        i = m - rank_from_top
        run = min(run, p[idx] * m / i)
        adj[idx] = run
    return adj

def nb_rho(n_train: float, n_val: float) -> float:
    """Termo de correlacao de Nadeau & Bengio (2003): n_teste / n_treino."""
    return float(n_val) / float(n_train)


def corrected_t(d: np.ndarray, rho: float) -> dict:
    """t pareado com variancia corrigida (Nadeau & Bengio 2003):
    Var(media) = s^2 * (1/n + rho).  Sem correcao ele seria anticonservador, porque os
    treinos dos folds se sobrepoem e as diferencas por fold NAO sao independentes."""
    d = np.asarray(d, float)
    n = len(d)
    mean = float(d.mean())
    s2 = float(d.var(ddof=1)) if n > 1 else 0.0
    se = math.sqrt(s2 * (1.0 / n + rho))
    if se == 0.0:
        t = 0.0 if mean == 0.0 else math.copysign(math.inf, mean)
        p = 1.0 if mean == 0.0 else 0.0
    else:
        t = mean / se
        p = float(2 * stats.t.sf(abs(t), n - 1))
    crit = stats.t.ppf(0.975, n - 1)
    return dict(mean_diff=mean, se=se, t=t, df=n - 1, p=p, ci_low=mean - crit * se, ci_high=mean + crit * se)

def tost_corrected(d: np.ndarray, rho: float, delta: float) -> dict:
    """TOST (Schuirmann 1987) com o mesmo erro-padrao corrigido. H0: |mu| >= delta.
    p_tost = max(p_inferior, p_superior); equivalencia pratica se p_tost < alpha."""
    d = np.asarray(d, float)
    n = len(d)
    mean = float(d.mean())
    se = math.sqrt(float(d.var(ddof=1)) * (1.0 / n + rho)) if n > 1 else 0.0
    if se == 0.0:
        return dict(p_tost=0.0 if abs(mean) < delta else 1.0)
    p_low = stats.t.sf((mean + delta) / se, n - 1)      # H0: mu <= -delta
    p_up = stats.t.cdf((mean - delta) / se, n - 1)      # H0: mu >= +delta
    return dict(p_tost=float(max(p_low, p_up)))


def wilcoxon_paired(d: np.ndarray) -> float:
    """Wilcoxon signed-rank (Pratt). Verificacao de robustez: IGNORA a dependencia entre folds."""
    d = np.asarray(d, float)
    if not np.any(d != 0):
        return 1.0
    return float(stats.wilcoxon(d, zero_method="pratt", alternative="two-sided").pvalue)


def prob_superiority(d: np.ndarray) -> float:
    """P(A > B) + 0.5 P(A = B) sobre as diferencas pareadas. 0.5 = sem efeito."""
    d = np.asarray(d, float)
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / len(d))


# ------------------------------------------------------------------ EDF de duas amostras (KS / CvM / AD)
def _edf_stats(labels: np.ndarray, grp_size: np.ndarray, last: np.ndarray, n: int, m: int):
    """Estatisticas EDF para varias permutacoes de uma vez, tie-aware.
    labels: (P, N) 1 = amostra A, na ordem dos valores ordenados do pool.
    last: mascara das posicoes que encerram um grupo de empates; grp_size: tamanho do grupo
    (apenas nas posicoes `last`)."""
    N = n + m
    cA = np.cumsum(labels, axis=1)[:, last]                   # A com valor <= z_g
    pos = (np.arange(N) + 1)[last]                            # B_g (pool <= z_g)
    cB = pos[None, :] - cA
    diff = cA / n - cB / m
    D = np.abs(diff).max(axis=1)
    w = grp_size[last]
    omega2 = (n * m / N**2) * (w[None, :] * diff**2).sum(axis=1)       # Cramer-von Mises (Anderson 1962)
    # Anderson-Darling k=2 com empates (Scholz & Stephens 1987, versao "midrank")
    h = w.astype(float)
    Bj = pos - h / 2.0
    # M_Aj = (#A < z_j) + f_Aj/2 ; f_Aj = A em z_j
    fA = np.diff(np.concatenate([np.zeros((labels.shape[0], 1)), cA], axis=1), axis=1)
    MA = (cA - fA) + fA / 2.0
    MB = ((pos - h)[None, :] - (cA - fA)) + (h[None, :] - fA) / 2.0
    denom = Bj * (N - Bj) - N * h / 4.0
    ok = denom > 0
    termA = (N * MA - n * Bj[None, :]) ** 2
    termB = (N * MB - m * Bj[None, :]) ** 2
    A2 = ((N - 1) / N**2) * ((h[None, :] * (termA / n + termB / m) / np.where(ok, denom, 1))[:, ok]).sum(axis=1)
    return D, omega2, A2


def edf_two_sample(a: np.ndarray, b: np.ndarray, deff: float = 1.0, n_perm: int = 999, seed: int = 0) -> dict:
    """KS, Cramer-von Mises e Anderson-Darling de duas amostras, com p-valor de PERMUTACAO
    (exato sob H0 de amostras iid, valido com empates) e correcao por efeito de desenho:

      deff = n / n_eff = 1 + n*rho   (Nadeau-Bengio; deff=1 -> iid)

    Os escores dos folds nao sao iid, entao o p-valor iid e' otimista. A correcao ENCOLHE a
    estatistica observada: D -> D/sqrt(deff); omega^2, A^2 -> /deff  (todas escalam com o
    tamanho efetivo de amostra), antes de compara-la com a distribuicao nula de permutacao.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n, m = len(a), len(b)
    pool = np.concatenate([a, b])
    order = np.argsort(pool, kind="mergesort")
    zs = pool[order]
    lab_obs = np.concatenate([np.ones(n), np.zeros(m)])[order][None, :]
    N = n + m
    last = np.ones(N, bool)
    last[:-1] = zs[1:] != zs[:-1]
    grp = np.zeros(N)
    starts = np.flatnonzero(np.concatenate([[True], zs[1:] != zs[:-1]]))
    ends = np.flatnonzero(last)
    grp[ends] = ends - starts + 1
    D0, W0, A0 = (x[0] for x in _edf_stats(lab_obs, grp, last, n, m))
    if D0 == 0 and np.all(zs == zs[0]):
        return dict(ks_D=0.0, ks_p=1.0, cvm_w2=0.0, cvm_p=1.0, ad_A2=0.0, ad_p=1.0)
    rng = np.random.default_rng(seed)
    perm = np.argsort(rng.random((n_perm, N)), axis=1)
    labels = (perm < n).astype(float)                       # posicoes 'A' em cada permutacao
    D, W, A = _edf_stats(labels, grp, last, n, m)
    sd, sdf = math.sqrt(deff), deff
    pv = lambda null, obs: float((1 + (null >= obs - 1e-12).sum()) / (n_perm + 1))
    return dict(ks_D=float(D0), ks_p=pv(D, D0 / sd), cvm_w2=float(W0), cvm_p=pv(W, W0 / sdf),
                ad_A2=float(A0), ad_p=pv(A, A0 / sdf))


def edf_paired_swap(a: np.ndarray, b: np.ndarray, deff: float = 1.0, n_perm: int = 999, seed: int = 0) -> dict:
    """KS / Cramer-von Mises / Anderson-Darling entre as EDFs de A e B, com p-valor de PERMUTACAO PAREADA
    (troca A<->B dentro de cada bloco com prob. 1/2). Preserva o pareamento: a dificuldade do fold e' comum as duas
    bases, entao a permutacao NAO pareada compararia amostras 'mais parecidas do que independentes' e seria
    conservadora demais (medido em tests/calibration_null.py).  `deff` encolhe a estatistica observada como em
    edf_two_sample (deff=1: sem correcao; deff=1+n*rho: tamanho efetivo de Nadeau-Bengio)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = len(a)
    pool = np.concatenate([a, b])
    order = np.argsort(pool, kind="mergesort")
    zs = pool[order]
    N = 2 * n
    inv = np.empty(N, int); inv[order] = np.arange(N)          # posicao ordenada de cada observacao
    pa, pb = inv[:n], inv[n:]
    last = np.ones(N, bool); last[:-1] = zs[1:] != zs[:-1]
    starts = np.flatnonzero(np.concatenate([[True], zs[1:] != zs[:-1]])); ends = np.flatnonzero(last)
    grp = np.zeros(N); grp[ends] = ends - starts + 1
    if np.all(zs == zs[0]):
        return dict(ks_D=0.0, ks_p=1.0, cvm_w2=0.0, cvm_p=1.0, ad_A2=0.0, ad_p=1.0)
    rng = np.random.default_rng(seed)
    sw = (rng.random((n_perm + 1, n)) < 0.5).astype(float)
    sw[0] = 0.0                                                # linha 0 = observado
    labels = np.zeros((n_perm + 1, N))
    labels[:, pa] = 1.0 - sw
    labels[:, pb] = sw
    D, W, A = _edf_stats(labels, grp, last, n, n)
    pv = lambda s, shrink: float((1 + (s[1:] >= s[0] / shrink - 1e-12).sum()) / (n_perm + 1))
    return dict(ks_D=float(D[0]), ks_p=pv(D, math.sqrt(deff)), cvm_w2=float(W[0]), cvm_p=pv(W, deff),
                ad_A2=float(A[0]), ad_p=pv(A, deff))


# ------------------------------------------------------------------ normalidade (1 amostra) com Monte Carlo
@lru_cache(maxsize=256)
def _normal_null(n: int, n_mc: int = 2000, seed: int = 123):
    rng = np.random.default_rng(seed + n)
    x = rng.standard_normal((n_mc, n))
    return _norm_stats(x)


def _norm_stats(x: np.ndarray):
    x = np.sort(x, axis=1)
    n = x.shape[1]
    sd = x.std(axis=1, ddof=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    z = (x - x.mean(axis=1, keepdims=True)) / sd
    F = np.clip(stats.norm.cdf(z), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    ks = np.maximum((i / n - F).max(axis=1), (F - (i - 1) / n).max(axis=1))            # Lilliefors
    cvm = 1 / (12 * n) + ((F - (2 * i - 1) / (2 * n)) ** 2).sum(axis=1)
    ad = -n - ((2 * i - 1) * (np.log(F) + np.log(1 - F[:, ::-1]))).sum(axis=1) / n
    return ks, cvm, ad


def normality_gof(d: np.ndarray) -> dict:
    """KS (Lilliefors), CvM e AD de aderencia a Normal com parametros ESTIMADOS.
    Os p-valores vem de Monte Carlo (a distribuicao nula depende so' de n); usar a tabela
    do KS com media/desvio estimados seria conservador demais."""
    d = np.asarray(d, float)
    n = len(d)
    if n < 8 or np.ptp(d) == 0:
        return dict(norm_ks_p=np.nan, norm_cvm_p=np.nan, norm_ad_p=np.nan)
    obs = _norm_stats(d[None, :])
    null = _normal_null(n)
    out = {}
    for name, o, nl in zip(("ks", "cvm", "ad"), obs, null):
        out[f"norm_{name}_p"] = float((1 + (nl >= o[0]).sum()) / (len(nl) + 1))
    return out


# ------------------------------------------------------------------ omnibus Friedman (+ Iman-Davenport)
def friedman_id(M: np.ndarray) -> dict:
    """M: (blocos x tratamentos). Friedman com correcao de empates + Iman-Davenport."""
    N, k = M.shape
    if np.all(np.ptp(M, axis=1) == 0):                      # todos empatados em todos os blocos: sem informacao
        return dict(N=N, k=k, chi2=np.nan, p_chi2=np.nan, F=np.nan, p_F=np.nan)
    with np.errstate(all="ignore"):
        chi2, p_chi = stats.friedmanchisquare(*[M[:, j] for j in range(k)])
    denom = N * (k - 1) - chi2
    if not np.isfinite(chi2) or denom <= 0:
        return dict(N=N, k=k, chi2=float(chi2), p_chi2=float(p_chi), F=np.nan, p_F=np.nan)
    F = (N - 1) * chi2 / denom
    return dict(N=N, k=k, chi2=float(chi2), p_chi2=float(p_chi), F=float(F),
                p_F=float(stats.f.sf(F, k - 1, (k - 1) * (N - 1))))


def avg_ranks(M: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    R = np.apply_along_axis(lambda r: stats.rankdata(-r if higher_is_better else r), 1, M)
    return R.mean(axis=0)
