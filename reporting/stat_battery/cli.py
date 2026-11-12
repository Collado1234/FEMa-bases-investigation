"""Orquestrador.

  # camadas A + B a partir de results/ (escores por fold da config vencedora):
  python -m reporting.stat_battery.cli --results results --prefix v3 --metric balanced_accuracy \
         --groups reports/eda/dataset_characterization.csv --out reports/v4_stats --delta 0.01

  # camada C (opcional, apos scripts/dump_test_predictions.py):
  python -m reporting.stat_battery.cli ... --preds preds/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import by_group, delong, loading, per_dataset


def _md(df: pd.DataFrame, floatfmt=".4g") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt) if len(df) else "(vazio)"


def plot_group_heatmaps(res, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for nivel in ("base", "familia"):
        L = res[nivel]["ranks"]
        crits = list(L["criterio"].unique())
        fig, axes = plt.subplots(1, len(crits), figsize=(4.2 * len(crits), 0.32 * L["basis"].nunique() + 2.2), sharey=True)
        order = L.groupby("basis")["posto_medio"].mean().sort_values().index
        for ax, c in zip(np.atleast_1d(axes), crits):
            pv = L[L["criterio"] == c].pivot(index="basis", columns="nivel", values="posto_medio").loc[order]
            n = L[L["criterio"] == c].groupby("nivel")["n_datasets"].first()
            im = ax.imshow(pv.values, cmap="viridis_r", aspect="auto")
            ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels([f"{x}\n(n={n[x]})" for x in pv.columns], fontsize=8)
            ax.set_yticks(range(len(pv))); ax.set_yticklabels(pv.index, fontsize=8); ax.set_title(c, fontsize=10)
            for i in range(pv.shape[0]):
                for j in range(pv.shape[1]):
                    ax.text(j, i, f"{pv.values[i, j]:.1f}", ha="center", va="center", fontsize=6, color="w")
        fig.suptitle(f"Posto medio por grupo ({nivel}); 1 = melhor", y=1.0)
        fig.tight_layout()
        fig.savefig(out / f"heatmap_rank_por_grupo_{nivel}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


def plot_top_group(summ: pd.DataFrame, groups: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = summ.pivot(index="dataset", columns="basis", values="grupo_topo").astype(float)
    best = summ.pivot(index="dataset", columns="basis", values="melhor_em_media").astype(bool)
    order = groups.loc[T.index].sort_values(["IHDD", "n_amostras"]).index
    T = T.loc[order]; best = best.loc[order]
    cols = T.mean().sort_values(ascending=False).index
    T = T[cols]; best = best[cols]
    fig, ax = plt.subplots(figsize=(0.45 * T.shape[1] + 4, 0.3 * T.shape[0] + 2))
    ax.imshow(T.values, cmap="Greens", aspect="auto", vmin=0, vmax=1.4)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if best.values[i, j]:
                ax.text(j, i, "*", ha="center", va="center", color="k")
    ax.set_xticks(range(T.shape[1])); ax.set_xticklabels(cols, rotation=75, ha="right", fontsize=8)
    ax.set_yticks(range(T.shape[0]))
    ax.set_yticklabels([f"{d} [{groups.loc[d, 'cat_IHDD']}]" for d in T.index], fontsize=8)
    ax.set_title("Grupo do topo por dataset (verde = nao pior que a melhor, Holm; * = melhor em media)")
    fig.tight_layout(); fig.savefig(out / "matriz_grupo_topo.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--prefix", default="v3")
    ap.add_argument("--metric", default="balanced_accuracy")
    ap.add_argument("--groups", default="reports/eda/dataset_characterization.csv")
    ap.add_argument("--out", default="reports/v4_stats")
    ap.add_argument("--cache", default=None, help="prefixo de cache dos escores por fold (ex.: reports/v4_stats/cache)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.01, help="margem de equivalencia pratica (unidades da metrica)")
    ap.add_argument("--equiv-adjust", choices=["bh", "holm", "none"], default="bh")
    ap.add_argument("--edf-deff", action="store_true", help="encolhe KS/CvM/AD pelo tamanho efetivo N-B (conservador)")
    ap.add_argument("--n-perm", type=int, default=999)
    ap.add_argument("--n-perm-groups", type=int, default=9999)
    ap.add_argument("--preds", default=None, help="diretorio com <dataset>_s<seed>.npz (camada C)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cache = a.cache or str(out / "cache")

    folds, tuning = loading.load_fold_scores(a.results, a.prefix, a.metric, cache=cache)
    wide, problems, meta = loading.complete_blocks(folds)
    bases = sorted(folds["basis"].unique())
    groups = loading.load_groups(a.groups)
    tuning.to_csv(out / "tuning_info.csv", index=False)

    # ---------------- A
    omni, pairs, summ = per_dataset.run_all_datasets(wide, bases, meta, alpha=a.alpha, delta=a.delta, n_perm=a.n_perm,
                                                      equiv_adjust=a.equiv_adjust, edf_deff=a.edf_deff)
    omni.to_csv(out / "A_omnibus_por_dataset.csv", index=False)
    pairs.to_csv(out / "A_pares_por_dataset.csv", index=False)
    summ.to_csv(out / "A_resumo_por_dataset_base.csv", index=False)
    plot_top_group(summ, groups, out)

    # ---------------- B  (unidade = dataset; escore = media dos blocos)
    M = wide.groupby("dataset")[bases].mean()
    res = by_group.run_group_analysis(M, groups, a.alpha, a.n_perm_groups)
    for nivel, d in res.items():
        for nome, df in d.items():
            df.to_csv(out / f"B_{nivel}_{nome}.csv", index=nome == "rank_matrix")
    plot_group_heatmaps(res, out)

    # ---------------- C
    delong_df = None
    if a.preds:
        frames = []
        for f in sorted(Path(a.preds).glob("*.npz")):
            z = np.load(f)
            ds, seed = f.stem.rsplit("_s", 1)
            sc = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith("proba__")}
            if len(sc) < 2:
                continue
            frames.append(delong.delong_pairs(z["y_true"], sc).assign(dataset=ds, seed=int(seed)))
        if frames:
            delong_df = pd.concat(frames, ignore_index=True)
            delong_df.to_csv(out / "C_delong_pares.csv", index=False)

    write_report(out, a, wide, problems, omni, pairs, summ, res, groups, tuning, delong_df)
    print(f"ok -> {out}")


def write_report(out, a, wide, problems, omni, pairs, summ, res, groups, tuning, delong_df):
    L = [f"# Bateria estatistica (metrica: {a.metric}; alpha = {a.alpha}; margem de equivalencia delta = {a.delta})\n"]
    L.append(f"- Datasets: **{wide['dataset'].nunique()}**; bases: **{summ['basis'].nunique()}**; "
             f"blocos pareados (seed x repeat x fold) por dataset: {int(omni['n_blocos'].min())}-{int(omni['n_blocos'].max())}.")
    if problems:
        L.append("- Avisos de carga: " + "; ".join(f"{d}: {m}" for d, m in problems))
    L.append("\nMetodologia e justificativas: ver `METODOLOGIA.md`.\n")

    L.append("## A. Por dataset\n")
    L.append("### A.1 Omnibus (Friedman + Iman-Davenport, ingenuo vs. com tamanho efetivo de amostra)\n")
    o = omni[["dataset", "n_blocos", "deff", "n_eff", "p_friedman_naive", "p_friedman_deff", "p_friedman_deff_bh"]]
    L.append(_md(o))
    L.append(f"\nDatasets com ao menos uma diferenca detectavel (omnibus corrigido, BH<{a.alpha}): "
             f"**{int((omni['p_friedman_deff_bh'] < a.alpha).sum())}/{len(omni)}**.\n")

    L.append("### A.2 Desfecho dos pares (Holm dentro de cada dataset; 3 vias)\n")
    c = pairs.groupby("dataset")["desfecho"].value_counts().unstack(fill_value=0)
    for col in ("a_melhor", "b_melhor", "equivalente", "inconclusivo"):
        if col not in c:
            c[col] = 0
    c["diferenca_sig"] = c["a_melhor"] + c["b_melhor"]
    c["so_indicativo"] = (pairs["nivel_evidencia"] == "indicativo").groupby(pairs["dataset"]).sum()
    c["pares"] = c[["a_melhor", "b_melhor", "equivalente", "inconclusivo"]].sum(axis=1)
    c["equiv_mas_forma_difere"] = pairs.groupby("dataset")["equiv_mas_forma_difere"].sum()
    c["wilcoxon_discorda"] = (~pairs["robusto_wilcoxon"]).groupby(pairs["dataset"]).sum()
    L.append(_md(c[["pares", "diferenca_sig", "so_indicativo", "equivalente", "inconclusivo",
                    "equiv_mas_forma_difere", "wilcoxon_discorda"]].reset_index()))
    L.append("\n`diferenca_sig` = robusto (sobrevive a correcao N-B + Holm); `so_indicativo` = significativo apenas "
             "sem a correcao N-B (Holm). `wilcoxon_discorda` = Wilcoxon ingenuo (Holm) discorda do t corrigido.")

    L.append("\n### A.3 Grupo do topo por dataset (bases nao significativamente piores que a melhor em media)\n")
    rows = []
    for ds, g in summ.groupby("dataset"):
        top = g[g["grupo_topo"]].sort_values("media", ascending=False)
        rows.append(dict(dataset=ds, IHDD=groups.loc[ds, "cat_IHDD"], melhor=top.iloc[0]["basis"],
                         media_melhor=top.iloc[0]["media"], n_no_topo=len(top),
                         n_no_topo_indicativo=int(g["grupo_topo_indicativo"].sum()),
                         topo=", ".join(top["basis"].head(8)) + (" ..." if len(top) > 8 else "")))
    L.append(_md(pd.DataFrame(rows)))
    freq = summ.groupby("basis").agg(vezes_no_topo=("grupo_topo", "sum"), vezes_melhor=("melhor_em_media", "sum"),
                                     vezes_no_topo_indicativo=("grupo_topo_indicativo", "sum"),
                                     vitorias_sig=("vitorias", "sum"), derrotas_sig=("derrotas", "sum")).sort_values(
        "vezes_no_topo", ascending=False).reset_index()
    L.append("\n### A.4 Frequencia agregada entre datasets\n")
    L.append(_md(freq))
    L.append("\n![](matriz_grupo_topo.png)\n")

    L.append("## B. Por grupo de datasets\n")
    for nivel in ("familia", "base"):
        d = res[nivel]
        L.append(f"### B.{'1' if nivel == 'familia' else '2'} Nivel de {nivel}\n")
        L.append("**Dependencia do perfil de postos em relacao ao grupo (permutacao global):**\n")
        L.append(_md(d["dependencia"]))
        L.append("\n**Friedman dentro de cada grupo:**\n")
        L.append(_md(d["friedman"].drop(columns=["datasets"])))
        sig = d["kruskal"][d["kruskal"]["p_holm_global"] < a.alpha]
        L.append(f"\n**Kruskal-Wallis por base (Holm global): {len(sig)} significativos.**\n")
        L.append(_md(d["kruskal"].head(10).drop(columns=["p_holm_global"])))
        sp = d["spearman"][d["spearman"]["p_bh"] < a.alpha]
        L.append(f"\n**Spearman posto x covariavel (BH): {len(sp)} significativos.**\n")
        L.append(_md(d["spearman"].head(10)))
        L.append("")
    L.append("![](heatmap_rank_por_grupo_familia.png)\n![](heatmap_rank_por_grupo_base.png)\n")

    if delong_df is not None:
        L.append("## C. DeLong no holdout (AUC)\n")
        s = delong_df.assign(sig=delong_df["p_holm_pares"] < a.alpha).groupby("dataset").agg(
            seeds=("seed", "nunique"), pares=("sig", "size"), sig=("sig", "sum"))
        L.append(_md(s.reset_index()))

    # vies de selecao do tuning
    t = tuning.assign(gap=tuning["best_mean"] - tuning["median_mean"]).groupby("basis")["gap"].mean().sort_values()
    L.append("\n## Diagnostico: vies de selecao do random search (melhor - mediana das combinacoes, escala da metrica de tuning)\n")
    L.append(_md(t.reset_index()))
    (out / "relatorio.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
