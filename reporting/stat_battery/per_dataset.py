"""Camada A - analise INDIVIDUAL por dataset, sobre os escores por fold da configuracao
vencedora de cada base (10 folds x 3 repeticoes [x seeds]), pareados por bloco (seed, repeat, fold)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from . import primitives as P


def analyze_dataset(M: np.ndarray, bases: list[str], rho: float, alpha=0.05, delta=0.01,
                    n_perm=999, seed=0, equiv_adjust="bh", edf_deff=False):
    """M: (blocos x bases). Retorna (omnibus dict, pairs DataFrame, summary DataFrame)."""
    n, k = M.shape
    deff = 1.0 + n * rho                         # n / n_eff, Nadeau-Bengio
    # ---- omnibus: Friedman + Iman-Davenport, ingenuo e com tamanho efetivo (N -> N/deff)
    fr = P.friedman_id(M)
    chi2_eff = fr["chi2"] / deff
    n_eff = n / deff
    den = n_eff * (k - 1) - chi2_eff
    F_eff = (n_eff - 1) * chi2_eff / den if den > 0 and n_eff > 1 else np.nan
    p_eff = float(stats.f.sf(F_eff, k - 1, (k - 1) * (n_eff - 1))) if np.isfinite(F_eff) else np.nan
    omni = dict(n_blocos=n, k=k, deff=deff, n_eff=n_eff, friedman_chi2=fr["chi2"], p_friedman_naive=fr["p_F"],
                p_friedman_deff=p_eff)

    # ---- todos os pares
    rows = []
    for i, j in itertools.combinations(range(k), 2):
        d = M[:, i] - M[:, j]
        ct = P.corrected_t(d, rho)
        row = dict(a=bases[i], b=bases[j], **ct,
                   p_tost=P.tost_corrected(d, rho, delta)["p_tost"],
                   p_wilcoxon_naive=P.wilcoxon_paired(d),
                   prob_sup=P.prob_superiority(d),
                   **P.normality_gof(d),
                   **P.edf_paired_swap(M[:, i], M[:, j], deff=deff if edf_deff else 1.0, n_perm=n_perm,
                                      seed=seed + i * 997 + j))
        rows.append(row)
    pr = pd.DataFrame(rows)
    for col in ("p", "p_wilcoxon_naive", "ks_p", "cvm_p", "ad_p"):
        pr[f"{col}_holm"] = P.holm(pr[col].values)          # afirmar diferenca: controle FWER (Holm)
    # afirmar equivalencia: familia propria. Holm sobre 210 pares torna a equivalencia inalcancavel; o padrao e'
    # BH (FDR) - ver METODOLOGIA.md. 'holm' e 'none' ficam disponiveis para analise de sensibilidade.
    adj = {"holm": P.holm, "bh": P.benjamini_hochberg, "none": lambda x: np.asarray(x, float)}[equiv_adjust]
    pr["p_tost_adj"] = adj(pr["p_tost"].values)
    # ---- desfecho de 3 vias: equivalente (TOST) / A melhor / A pior / inconclusivo
    def outcome(r):
        if r["p_tost_adj"] < alpha:
            return "equivalente"
        if r["p_holm"] < alpha:
            return "a_melhor" if r["mean_diff"] > 0 else "b_melhor"
        return "inconclusivo"
    pr["desfecho"] = pr.apply(outcome, axis=1)
    # nivel 'indicativo': mesmo t pareado SEM a correcao N-B (rho=0) + Holm. Mais poder, menos garantia.
    pr["p_naive"] = [P.corrected_t(M[:, bases.index(x)] - M[:, bases.index(y)], 0.0)["p"] for x, y in zip(pr.a, pr.b)]
    pr["p_naive_holm"] = P.holm(pr["p_naive"].values)
    pr["desfecho_indicativo"] = np.where(pr["p_naive_holm"] < alpha,
                                         np.where(pr["mean_diff"] > 0, "a_melhor", "b_melhor"), "inconclusivo")
    pr["nivel_evidencia"] = np.select(
        [pr["desfecho"].isin(["a_melhor", "b_melhor"]), pr["desfecho_indicativo"] != "inconclusivo"],
        ["robusto", "indicativo"], default="nenhum")
    pr["robusto_wilcoxon"] = (pr["p_wilcoxon_naive_holm"] < alpha) == (pr["p_holm"] < alpha)
    pr["forma_difere"] = (pr[["ks_p_holm", "cvm_p_holm", "ad_p_holm"]] < alpha).sum(axis=1) >= 2
    pr["equiv_mas_forma_difere"] = (pr["desfecho"] == "equivalente") & pr["forma_difere"]
    pr["normal_ok"] = pr[["norm_ks_p", "norm_cvm_p", "norm_ad_p"]].min(axis=1) > alpha
    sm = summarize(pr, M, bases)
    sm["grupo_topo_indicativo"] = summarize(pr, M, bases, "desfecho_indicativo")["grupo_topo"].reindex(sm.index)
    return omni, pr, sm


def summarize(pr: pd.DataFrame, M: np.ndarray, bases: list[str], col: str = "desfecho") -> pd.DataFrame:
    """Por base: media, posto medio, #vitorias/#derrotas/#equivalencias significativas e pertinencia ao
    'grupo do topo' (nao significativamente pior que a melhor base em media; Holm sobre todos os pares)."""
    means = pd.Series(M.mean(axis=0), index=bases)
    ranks = pd.Series(P.avg_ranks(M), index=bases)
    wins = {b: 0 for b in bases}; losses = {b: 0 for b in bases}; eq = {b: 0 for b in bases}
    worse_than_best = set()
    best = means.idxmax()
    for r in pr.itertuples():
        o = getattr(r, col)
        if o == "a_melhor":
            wins[r.a] += 1; losses[r.b] += 1
        elif o == "b_melhor":
            wins[r.b] += 1; losses[r.a] += 1
        elif o == "equivalente":
            eq[r.a] += 1; eq[r.b] += 1
        if o in ("a_melhor", "b_melhor"):
            loser = r.b if o == "a_melhor" else r.a
            winner = r.a if o == "a_melhor" else r.b
            if winner == best:
                worse_than_best.add(loser)
    df = pd.DataFrame(dict(media=means, posto_medio=ranks, vitorias=pd.Series(wins), derrotas=pd.Series(losses),
                           equivalentes=pd.Series(eq)))
    df["melhor_em_media"] = df.index == best
    df["grupo_topo"] = [b not in worse_than_best for b in df.index]
    return df.sort_values("media", ascending=False)


def run_all_datasets(wide: pd.DataFrame, bases: list[str], meta: pd.DataFrame, **kw):
    omni_rows, pair_frames, sum_frames = [], [], []
    for ds, g in wide.groupby("dataset"):
        M = g[bases].to_numpy()
        rho = float(meta.loc[ds, "n_val"] / meta.loc[ds, "n_train"])
        omni, pr, sm = analyze_dataset(M, bases, rho, **kw)
        omni_rows.append(dict(dataset=ds, rho=rho, **omni))
        pair_frames.append(pr.assign(dataset=ds))
        sum_frames.append(sm.assign(dataset=ds).reset_index(names="basis"))
    omni = pd.DataFrame(omni_rows)
    pairs = pd.concat(pair_frames, ignore_index=True)
    summ = pd.concat(sum_frames, ignore_index=True)
    # significancia do omnibus entre datasets: BH (controle de FDR; as decisoes por dataset sao exploratorias)
    omni["p_friedman_deff_bh"] = P.benjamini_hochberg(omni["p_friedman_deff"].fillna(1.0).values)
    return omni, pairs, summ
