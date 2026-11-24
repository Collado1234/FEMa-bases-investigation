"""Relatorios POR DATASET: ranking das bases em balanced accuracy, F1, precisao e revocacao, apontando
quais diferencas sao estatisticamente significativas e quais nao.

Para cada (dataset, metrica) usa os escores por fold da configuracao vencedora de cada base
(10 folds x 3 repeticoes por seed), pareados por bloco (seed, repeat, fold), e a MESMA maquinaria da
bateria (per_dataset.analyze_dataset):
  - t pareado com correcao de Nadeau & Bengio (rho = n_val/n_train) + Holm sobre todos os pares   -> 'robusto'
  - mesmo t SEM a correcao N-B + Holm                                                            -> 'indicativo'
  - TOST (margem delta) + BH                                                                     -> 'equivalente'
  - nada disso                                                                                   -> 'inconclusivo'

Saidas em --out:
  INDEX.md                               visao geral (dataset x metrica) + frequencias entre datasets
  por_dataset/<dataset>.md  (+ .png)     ranking completo das 4 metricas, com significancia vs melhor e vs controle
  ranking_por_dataset_metrica.csv        tabela longa (uma linha por dataset x metrica x base)
  pares_por_dataset_metrica.csv          todos os pares (diferenca, IC95%, p, p_Holm, desfecho)
  omnibus_por_dataset_metrica.csv        Friedman + Iman-Davenport por dataset e metrica

Uso:
  python -m reporting.stat_battery.dataset_reports --results results --prefix v4 --seeds 42 \
         --control shepard --out reports/v4_por_dataset
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import loading, per_dataset
from . import primitives as P

METRICS = ("balanced_accuracy", "f1", "precision", "recall")
LABEL = {"balanced_accuracy": "Balanced accuracy", "f1": "F1", "precision": "Precisao", "recall": "Revocacao"}
FAM = loading.BASIS_TO_FAMILY


# --------------------------------------------------------------------------- comparacao de pares
def _pair_lookup(pairs: pd.DataFrame) -> dict:
    return {(r.a, r.b): r for r in pairs.itertuples()}


def compare(lookup: dict, x: str, ref: str) -> dict:
    """x contra ref: diferenca (x - ref), IC95% e veredito em 5 niveis."""
    if (x, ref) in lookup:
        r, sign = lookup[(x, ref)], 1.0
    else:
        r, sign = lookup[(ref, x)], -1.0
    diff = sign * r.mean_diff
    lo, hi = (r.ci_low, r.ci_high) if sign > 0 else (-r.ci_high, -r.ci_low)
    if r.desfecho == "equivalente":
        v = "equivalente"
    elif r.desfecho in ("a_melhor", "b_melhor"):
        winner = r.a if r.desfecho == "a_melhor" else r.b
        v = "melhor (sig.)" if winner == x else "pior (sig.)"
    elif r.nivel_evidencia == "indicativo":
        v = "melhor (indicativo)" if diff > 0 else "pior (indicativo)"
    else:
        v = "inconclusivo"
    return dict(diff=diff, ci_low=lo, ci_high=hi, p_holm=r.p_holm, verdict=v)


# --------------------------------------------------------------------------- tabela de um dataset/metrica
def ranking_table(wide_ds: pd.DataFrame, pairs_ds: pd.DataFrame, summ_ds: pd.DataFrame, control: str) -> pd.DataFrame:
    bases = list(wide_ds.columns)
    means, sds = wide_ds.mean(), wide_ds.std(ddof=1)
    order = means.sort_values(ascending=False, kind="stable").index
    ranks = pd.Series(stats.rankdata(-means.loc[order].values, method="min"), index=order)
    best = order[0]
    lk = _pair_lookup(pairs_ds)
    s = summ_ds.set_index("basis")
    rows = []
    for b in order:
        vb = compare(lk, b, best) if b != best else dict(diff=0.0, ci_low=np.nan, ci_high=np.nan, p_holm=np.nan, verdict="(melhor)")
        vc = (compare(lk, b, control) if (b != control and control in bases)
              else dict(diff=np.nan, ci_low=np.nan, ci_high=np.nan, p_holm=np.nan, verdict="(controle)" if b == control else "-"))
        rows.append(dict(rank=int(ranks[b]), basis=b, familia=FAM.get(b, "?"), media=means[b], dp=sds[b],
                         posto_medio=s.loc[b, "posto_medio"],
                         delta_melhor=vb["diff"], p_holm_melhor=vb["p_holm"], vs_melhor=vb["verdict"],
                         grupo_topo=bool(s.loc[b, "grupo_topo"]),
                         vit=int(s.loc[b, "vitorias"]), der=int(s.loc[b, "derrotas"]), equiv=int(s.loc[b, "equivalentes"]),
                         delta_controle=vc["diff"], ic_low_controle=vc["ci_low"], ic_high_controle=vc["ci_high"],
                         p_holm_controle=vc["p_holm"], vs_controle=vc["verdict"]))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- markdown / figuras
def _fmt(df: pd.DataFrame) -> str:
    d = pd.DataFrame({
        "#": df["rank"], "base": df["basis"], "familia": df["familia"],
        "media": df["media"].map("{:.4f}".format), "dp": df["dp"].map("{:.4f}".format),
        "d vs melhor": df["delta_melhor"].map("{:+.4f}".format), "p_Holm": df["p_holm_melhor"].map(lambda v: "-" if pd.isna(v) else f"{v:.3g}"),
        "vs melhor": df["vs_melhor"], "topo": df["grupo_topo"].map({True: "sim", False: "nao"}),
        "V/D/E": [f"{v}/{d}/{e}" for v, d, e in zip(df["vit"], df["der"], df["equiv"])],
        "d vs controle": df["delta_controle"].map(lambda v: "-" if pd.isna(v) else f"{v:+.4f}"),
        "vs controle": df["vs_controle"]})
    return d.to_markdown(index=False)


def plot_dataset(ds: str, tables: dict, out: Path, control: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"(melhor)": "#1f4e9c", "melhor (sig.)": "#1f4e9c", "melhor (indicativo)": "#5b8dd6", "equivalente": "#2e9d52",
           "inconclusivo": "#2e9d52", "pior (indicativo)": "#e69f00", "pior (sig.)": "#c0392b"}
    ms = list(tables)
    n = max(len(t) for t in tables.values())
    fig, axes = plt.subplots(1, len(ms), figsize=(4.6 * len(ms), 0.27 * n + 1.8))
    for ax, m in zip(np.atleast_1d(axes), ms):
        t = tables[m].iloc[::-1]
        ax.barh([b + ("  [controle]" if b == control else "") for b in t["basis"]], t["media"],
                xerr=t["dp"], color=[col.get(v, "#888") for v in t["vs_melhor"]], alpha=0.9, ecolor="#777", capsize=1.5)
        lo = float((t["media"] - t["dp"]).min()); hi = float((t["media"] + t["dp"]).max())
        ax.set_xlim(max(0, lo - 0.02), min(1.0, hi + 0.01)); ax.set_title(LABEL.get(m, m), fontsize=10)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle(f"{ds} - verde: nao pior que a melhor | laranja: pior (indicativo) | vermelho: pior (sig., Holm+N-B) | barras: media +- dp dos folds (eixo x truncado)", fontsize=9)
    fig.tight_layout(); fig.savefig(out / f"{ds}.png", dpi=120); plt.close(fig)


def write_dataset_md(ds, tables, omni_rows, control, out, alpha, delta, n_blocks, with_fig):
    L = [f"# {ds}\n"]
    o = next(iter(omni_rows.values()))
    L.append(f"- Blocos pareados (seed x repeat x fold): **{int(o['n_blocos'])}** | tamanho efetivo N-B: **{o['n_eff']:.1f}** | "
             f"controle: **{control}** | alpha = {alpha} | margem de equivalencia delta = {delta}")
    for m, om in omni_rows.items():
        L.append(f"- Omnibus ({LABEL[m]}): Friedman + Iman-Davenport corrigido p = {om['p_friedman_deff']:.3g} "
                 f"(BH entre datasets: {om['p_friedman_deff_bh']:.3g}); ingenuo p = {om['p_friedman_naive']:.3g}")
    if with_fig:
        L.append(f"\n![]({ds}.png)\n")
    for m, t in tables.items():
        n_top = int(t["grupo_topo"].sum()); n_worse = int(t["vs_melhor"].eq("pior (sig.)").sum())
        L.append(f"\n## {LABEL[m]}\n")
        L.append(f"Melhor em media: **{t.iloc[0]['basis']}** ({t.iloc[0]['media']:.4f}). Grupo do topo (nao significativamente piores "
                 f"que a melhor): **{n_top}/{len(t)}** bases. Significativamente piores que a melhor: **{n_worse}**.\n")
        L.append(_fmt(t))
    # consistencia entre metricas
    ms = list(tables)
    if len(ms) > 1:
        means = pd.DataFrame({m: tables[m].set_index("basis")["media"] for m in ms})
        L.append("\n## Consistencia entre metricas\n")
        taus = {f"{a} x {b}": stats.kendalltau(means[a], means[b])[0] for i, a in enumerate(ms) for b in ms[i + 1:]}
        L.append("Kendall tau entre os rankings (medias): " + "; ".join(f"{k}: {v:.2f}" for k, v in taus.items()) + ".\n")
        tops = [set(tables[m].loc[tables[m]["grupo_topo"], "basis"]) for m in ms]
        inter = sorted(set.intersection(*tops))
        L.append(f"Bases no grupo do topo em **todas** as metricas ({len(inter)}): {', '.join(inter) if inter else '(nenhuma)'}.")
        worse = sorted(set.intersection(*[set(tables[m].loc[tables[m]['vs_melhor'].eq('pior (sig.)'), 'basis']) for m in ms]))
        L.append(f"\nBases significativamente piores que a melhor em **todas** as metricas: {', '.join(worse) if worse else '(nenhuma)'}.")
    L.append("\n---\n*Leitura:* `vs melhor` / `vs controle` = veredito do par (t pareado N-B + Holm sobre todos os pares; "
             "`indicativo` = significativo so' sem a correcao N-B; `equivalente` = TOST dentro de +-delta; `inconclusivo` = sem evidencia "
             "de diferenca NEM de equivalencia, geralmente falta de poder). V/D/E = vitorias/derrotas/equivalencias robustas contra "
             "as demais bases. Em problemas binarios, F1/precisao/revocacao usam a classe positiva = rotulo 1 do LabelEncoder.")
    (out / f"{ds}.md").write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- orquestracao
def build(results, prefix, out, metrics=METRICS, seeds=None, control="shepard", alpha=0.05, delta=0.01,
          n_perm=199, figs=True, refresh=False, context="classifier"):
    out = Path(out); (out / "por_dataset").mkdir(parents=True, exist_ok=True)
    per = {}
    for m in metrics:
        cache = out / f"cache_{m}"
        if refresh:
            for suf in ("_folds.csv", "_tuning.csv"):
                Path(f"{cache}{suf}").unlink(missing_ok=True)
        folds, _ = loading.load_fold_scores(results, prefix, m, context=context, cache=str(cache))
        if seeds:
            folds = folds[folds["seed"].isin(list(seeds))]
        if folds.empty:
            raise SystemExit(f"Sem dados para a metrica '{m}' (prefixo {prefix}, seeds {seeds}).")
        wide, problems, meta = loading.complete_blocks(folds)
        bases = sorted(folds["basis"].unique())
        omni, pairs, summ = per_dataset.run_all_datasets(wide, bases, meta, alpha=alpha, delta=delta, n_perm=n_perm,
                                                          equiv_adjust="bh", edf_deff=False)
        per[m] = dict(wide=wide, bases=bases, omni=omni, pairs=pairs, summ=summ, problems=problems)

    datasets = sorted(set.intersection(*[set(v["omni"]["dataset"]) for v in per.values()]))
    long_rows, pair_rows, omni_rows, idx_rows = [], [], [], []
    for ds in datasets:
        tables, om = {}, {}
        for m, v in per.items():
            w = v["wide"][v["wide"]["dataset"] == ds][v["bases"]]
            t = ranking_table(w, v["pairs"][v["pairs"]["dataset"] == ds], v["summ"][v["summ"]["dataset"] == ds], control)
            tables[m] = t
            o = v["omni"].set_index("dataset").loc[ds].to_dict()
            om[m] = o
            long_rows.append(t.assign(dataset=ds, metrica=m))
            pair_rows.append(v["pairs"][v["pairs"]["dataset"] == ds][[
                "dataset", "a", "b", "mean_diff", "ci_low", "ci_high", "p", "p_holm", "p_tost_adj", "desfecho", "nivel_evidencia"]]
                             .assign(metrica=m))
            omni_rows.append(dict(dataset=ds, metrica=m, **o))
            idx_rows.append(dict(dataset=ds, metrica=m, melhor=t.iloc[0]["basis"], media=t.iloc[0]["media"],
                                 n_topo=int(t["grupo_topo"].sum()), n_bases=len(t),
                                 n_piores_sig=int(t["vs_melhor"].eq("pior (sig.)").sum()),
                                 omnibus_p_bh=o["p_friedman_deff_bh"], n_blocos=int(o["n_blocos"])))
        write_dataset_md(ds, tables, om, control, out / "por_dataset", alpha, delta, int(next(iter(om.values()))["n_blocos"]), figs)
        if figs:
            plot_dataset(ds, tables, out / "por_dataset", control)

    long_df = pd.concat(long_rows, ignore_index=True)
    long_df.to_csv(out / "ranking_por_dataset_metrica.csv", index=False)
    pd.concat(pair_rows, ignore_index=True).to_csv(out / "pares_por_dataset_metrica.csv", index=False)
    pd.DataFrame(omni_rows).to_csv(out / "omnibus_por_dataset_metrica.csv", index=False)
    idx = pd.DataFrame(idx_rows)
    write_index(out, idx, long_df, metrics, datasets, control, alpha, delta, seeds, per)
    return idx, long_df


def write_index(out, idx, long_df, metrics, datasets, control, alpha, delta, seeds, per):
    L = ["# Resultados por dataset\n",
         f"- Datasets: **{len(datasets)}** | bases: **{long_df['basis'].nunique()}** | metricas: {', '.join(LABEL[m] for m in metrics)}",
         f"- Seeds usadas: **{sorted(seeds) if seeds else 'todas'}** | controle: **{control}** | alpha = {alpha} | delta (equivalencia) = {delta}",
         "- Cada pagina `por_dataset/<dataset>.md` traz o ranking completo, a significancia de cada base contra a MELHOR e contra o "
         "CONTROLE, o grupo do topo e a consistencia entre metricas.\n"]
    for m, v in per.items():
        if v["problems"]:
            L.append(f"- Avisos de carga ({LABEL[m]}): " + "; ".join(f"{d}: {msg}" for d, msg in v["problems"]))
    L.append("\n## Visao geral (melhor base por metrica; entre parenteses: bases no topo / piores significativas)\n")
    wide = idx.pivot(index="dataset", columns="metrica", values="melhor")
    top = idx.pivot(index="dataset", columns="metrica", values="n_topo")
    wor = idx.pivot(index="dataset", columns="metrica", values="n_piores_sig")
    tab = pd.DataFrame(index=wide.index)
    for m in metrics:
        tab[LABEL[m]] = [f"{wide.loc[d, m]} ({top.loc[d, m]}/{wor.loc[d, m]})" for d in wide.index]
    first = idx[idx["metrica"] == metrics[0]].set_index("dataset")
    tab["omnibus p (BH)"] = first["omnibus_p_bh"].map("{:.3g}".format)
    tab["blocos"] = first["n_blocos"]
    tab.insert(0, "dataset", [f"[{d}](por_dataset/{d}.md)" for d in tab.index])
    L.append(tab.reset_index(drop=True).to_markdown(index=False))
    L.append("\n## Frequencia entre datasets (por metrica)\n")
    for m in metrics:
        g = long_df[long_df["metrica"] == m].groupby("basis").agg(
            no_topo=("grupo_topo", "sum"), melhor_em_media=("rank", lambda r: int((r == 1).sum())),
            pior_sig=("vs_melhor", lambda s: int((s == "pior (sig.)").sum())),
            posto_medio=("rank", "mean")).sort_values(["no_topo", "posto_medio"], ascending=[False, True])
        g.insert(0, "familia", [FAM.get(b, "?") for b in g.index])
        g.to_csv(out / f"frequencia_{m}.csv")
        L.append(f"\n### {LABEL[m]}\n")
        L.append(g.reset_index().to_markdown(index=False, floatfmt=".2f"))
    L.append("\n> `no_topo` = em quantos datasets a base NAO e' significativamente pior que a melhor; `pior_sig` = em quantos e' "
             "significativamente pior. Inconclusivo nao e' equivalencia: veja o poder (n efetivo N-B) em cada pagina.")
    (out / "INDEX.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--prefix", default="v4")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42])
    ap.add_argument("--metrics", nargs="+", default=list(METRICS))
    ap.add_argument("--control", default="shepard")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--n-perm", type=int, default=199, help="permutacoes dos testes de forma (nao afetam a significancia)")
    ap.add_argument("--out", default="reports/v4_por_dataset")
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="ignora o cache e relê results/")
    a = ap.parse_args()
    idx, _ = build(a.results, a.prefix, a.out, a.metrics, a.seeds, a.control, a.alpha, a.delta, a.n_perm, not a.no_figs, a.refresh)
    print(f"ok -> {a.out}  ({idx['dataset'].nunique()} datasets x {len(a.metrics)} metricas)")


if __name__ == "__main__":
    main()
