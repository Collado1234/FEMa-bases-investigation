"""Relacao entre a base de interpolacao e o numero de vizinhos k escolhido pela busca.

Perguntas (uma seed, config vencedora de cada base x dataset):
  H1  As bases escolhem k DIFERENTES?            Friedman (blocos = datasets) sobre a posicao relativa de k na faixa de busca.
  H2  Bases melhores usam k maior (ou menor)?     Spearman(posto de desempenho, k) por dataset + Wilcoxon sobre os rhos.
  H3  O k escolhido depende do contexto?          Spearman(k mediano do dataset, log n / n_features / n_classes ...) [opcional].
  H4  A faixa de busca esta bem posta?            fracao de datasets em que o melhor valor cai na borda da faixa (k e parametros de escala).
  H5  Perfil k x desempenho por base              Spearman(k, escore) entre as combinacoes sorteadas (sinal: k maior ajuda ou atrapalha).

k e' comparado em POSICAO RELATIVA NA FAIXA (log), k_pos = ln(k/k_min)/ln(k_max/k_min), porque a faixa de k de cada dataset
ja' e' relativa a densidade amostral (k_max = ~2*sqrt(n_treino)); tambem reporta k/sqrt(n_treino).

Uso:
  python -m reporting.k_analysis --results results --prefix v4 --seeds 42 --groups reports/eda/dataset_characterization.csv \
         --out reports/v4_k
"""
from __future__ import annotations

import argparse
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from reporting.stat_battery import primitives as P
from reporting.stat_battery.loading import BASIS_TO_FAMILY, FAMILIES

SEED_RE = re.compile(r"_s(\d+)$")
EDGE = 0.10                                  # "borda" = 10% extremos da faixa (em log para k e escalas log)


def _pos(value: float, spec) -> float | None:
    """Posicao [0,1] de `value` na faixa descrita por `spec` (ex.: ["logint", 3, 148])."""
    if not isinstance(spec, (list, tuple)) or len(spec) != 3 or isinstance(spec[0], (int, float)):
        return None
    kind, lo, hi = spec
    if hi <= lo or value is None:
        return None
    if kind in ("loguniform", "logint"):
        return float((math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo)))
    return float((value - lo) / (hi - lo))


def _read_scope(d: Path):
    try:
        s = json.load(open(d / "summary.json", encoding="utf-8"))
    except FileNotFoundError:
        return None
    best = s.get("best_configuration") or {}
    if not best:
        return None
    combos = [dict(hp=c["hyperparameters"], score=(c["metrics"].get(s.get("primary_metric", "f1")) or {}).get("mean"))
              for c in s.get("ranking", [])]
    n_train = None
    for f in d.glob("run_*.json"):                          # basta um run para saber n_train_fold
        n_train = json.load(open(f, encoding="utf-8")).get("n_train_fold")
        break
    return dict(dir=d, summary=s, best=best, combos=combos, n_train_fold=n_train)


def load_scopes(results, prefix, seeds, context="classifier", workers=8):
    root = Path(results) / context
    scopes = [p for p in root.glob("*/*/*") if p.is_dir() and p.name.startswith(prefix) and SEED_RE.search(p.name)
              and (not seeds or int(SEED_RE.search(p.name).group(1)) in set(seeds))]
    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(_read_scope, scopes))
    rows, prof, par = [], [], []
    for d, r in zip(scopes, out):
        if r is None:
            continue
        basis, ds, exp = d.parts[-3], d.parts[-2], d.parts[-1]
        seed = int(SEED_RE.search(exp).group(1))
        s = r["summary"]
        space = (s.get("config") or {}).get("search_space") or {}
        ks = [c["hp"]["k"] for c in r["combos"] if "k" in c["hp"]]
        kspec = space.get("k")
        k_min, k_max = (kspec[1], kspec[2]) if kspec else (min(ks), max(ks))        # legado: faixa observada
        hp = r["best"]["hyperparameters"]
        k = int(hp["k"])
        metric = s.get("primary_metric", "f1")
        score = (r["best"]["metrics"].get(metric) or {}).get("mean")
        rows.append(dict(dataset=ds, basis=basis, familia=BASIS_TO_FAMILY.get(basis, "?"), seed=seed, k=k, k_min=k_min, k_max=k_max,
                         k_pos=_pos(k, ["logint", k_min, k_max]), n_train_fold=r["n_train_fold"],
                         k_over_sqrt_n=(k / math.sqrt(r["n_train_fold"]) if r["n_train_fold"] else np.nan),
                         score=score, metric=metric, **{f"hp_{p}": v for p, v in hp.items() if p != "k"}))
        for c in r["combos"]:
            if "k" in c["hp"] and c["score"] is not None:
                prof.append(dict(dataset=ds, basis=basis, seed=seed, k=c["hp"]["k"], score=c["score"]))
        for p, v in hp.items():                              # posicao do melhor valor de cada parametro de escala
            if p == "k":
                continue
            spec = space.get(p)
            if spec is None:                                 # legado: faixa observada
                vals = [c["hp"][p] for c in r["combos"] if p in c["hp"]]
                spec = ["uniform", min(vals), max(vals)] if vals else None
            pos = _pos(v, spec) if spec else None
            if pos is not None:
                par.append(dict(dataset=ds, basis=basis, seed=seed, param=p, value=v, pos=pos))
    best = pd.DataFrame(rows)
    best["rank_desempenho"] = best.groupby(["dataset", "seed"])["score"].rank(ascending=False, method="average")
    return best, pd.DataFrame(prof), pd.DataFrame(par)


# ------------------------------------------------------------------------------------------ analises
def h1_bases_differ(best: pd.DataFrame, col="k_pos"):
    M = best.groupby(["dataset", "basis"])[col].mean().unstack().dropna(axis=0, how="any")
    fr = P.friedman_id(M.to_numpy())
    ranks = pd.DataFrame(M.apply(lambda r: stats.rankdata(r.values), axis=1, result_type="expand").to_numpy(), columns=M.columns)
    tab = pd.DataFrame({"familia": [BASIS_TO_FAMILY.get(b, "?") for b in M.columns], f"{col}_medio": M.mean().values,
                        "posto_medio_do_k": ranks.mean().values}, index=M.columns).sort_values("posto_medio_do_k")
    # familias (media por dataset)
    fam = pd.DataFrame({f: M[[b for b in bs if b in M.columns]].mean(axis=1) for f, bs in FAMILIES.items()
                        if any(b in M.columns for b in bs)}).dropna()
    fam_fr = P.friedman_id(fam.to_numpy()) if fam.shape[1] > 1 else None
    W = fr["chi2"] / (M.shape[0] * (M.shape[1] - 1))
    return dict(N=M.shape[0], k=M.shape[1], p=fr["p_F"], F=fr["F"], chi2=fr["chi2"], kendall_W=W, table=tab,
                fam_p=(fam_fr["p_F"] if fam_fr else np.nan), fam_table=fam.mean().rename(f"{col}_medio").to_frame())


def h2_better_bases_k(best: pd.DataFrame, col="k_pos"):
    rows = []
    for (ds, seed), g in best.groupby(["dataset", "seed"]):
        g = g.dropna(subset=[col, "score"])
        if len(g) >= 5 and g[col].nunique() > 1 and g["score"].nunique() > 1:
            rho = stats.spearmanr(-g["score"], g[col])[0]          # + => piores bases escolhem k maior
            rows.append(dict(dataset=ds, seed=seed, rho_posto_k=-rho))   # + => MELHORES bases escolhem k maior
    d = pd.DataFrame(rows)
    if len(d) >= 5 and d["rho_posto_k"].abs().sum() > 0:
        w = stats.wilcoxon(d["rho_posto_k"], zero_method="pratt").pvalue
    else:
        w = np.nan
    return d, dict(n=len(d), rho_mediano=float(d["rho_posto_k"].median()) if len(d) else np.nan,
                   frac_positivo=float((d["rho_posto_k"] > 0).mean()) if len(d) else np.nan, p_wilcoxon=w)


def h3_context(best: pd.DataFrame, groups: pd.DataFrame | None):
    if groups is None:
        return None
    by = best.groupby("dataset").agg(k_pos=("k_pos", "median"), k_over_sqrt_n=("k_over_sqrt_n", "median"), k=("k", "median"))
    rows = []
    for cov in ("n_amostras", "n_features", "n_classes", "indice_desbalanceamento"):
        if cov not in groups.columns:
            continue
        x = groups.loc[by.index.intersection(groups.index), cov].astype(float)
        for tgt in ("k", "k_pos", "k_over_sqrt_n"):
            y = by.loc[x.index, tgt]
            if len(x) >= 5:
                rho, p = stats.spearmanr(np.log10(x), y)
                rows.append(dict(covariavel=f"log10({cov})", alvo=tgt, n=len(x), rho=rho, p=p))
    d = pd.DataFrame(rows)
    if len(d):
        d["p_bh"] = P.benjamini_hochberg(d["p"].fillna(1.0).values)
    return d


def h4_edges(best: pd.DataFrame, par: pd.DataFrame):
    rows = []
    for b, g in best.groupby("basis"):
        rows.append(dict(basis=b, param="k", n=len(g), borda_inf=float((g["k_pos"] <= EDGE).mean()),
                         borda_sup=float((g["k_pos"] >= 1 - EDGE).mean())))
    for (b, p), g in par.groupby(["basis", "param"]):
        rows.append(dict(basis=b, param=p, n=len(g), borda_inf=float((g["pos"] <= EDGE).mean()), borda_sup=float((g["pos"] >= 1 - EDGE).mean())))
    d = pd.DataFrame(rows)
    d["borda_total"] = d["borda_inf"] + d["borda_sup"]
    return d.sort_values("borda_total", ascending=False)


def h5_profiles(prof: pd.DataFrame):
    rows = []
    for (ds, b, s), g in prof.groupby(["dataset", "basis", "seed"]):
        if len(g) >= 5 and g["k"].nunique() > 2 and g["score"].nunique() > 1:
            rows.append(dict(dataset=ds, basis=b, rho=stats.spearmanr(g["k"], g["score"])[0]))
    d = pd.DataFrame(rows)
    if d.empty:
        return d, d
    s = d.groupby("basis")["rho"].agg(rho_mediano="median", frac_k_maior_ajuda=lambda r: float((r > 0).mean()), n="size")
    s["familia"] = [BASIS_TO_FAMILY.get(b, "?") for b in s.index]
    return d, s.sort_values("rho_mediano")


# ------------------------------------------------------------------------------------------ figuras
def figures(best, par, edges, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pal = dict(zip(FAMILIES, ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#7f7f7f"]))
    order = best.groupby("basis")["k_pos"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(order) + 1.5))
    for i, b in enumerate(order):
        v = best.loc[best["basis"] == b, "k_pos"].dropna()
        kw = dict(positions=[i], widths=0.6, showfliers=False, boxprops=dict(color=pal.get(BASIS_TO_FAMILY.get(b), "k")))
        try:
            ax.boxplot(v, orientation="horizontal", **kw)           # matplotlib >= 3.10
        except TypeError:
            ax.boxplot(v, vert=False, **kw)
        ax.scatter(v, np.full(len(v), i) + np.random.default_rng(0).uniform(-0.18, 0.18, len(v)), s=10,
                   color=pal.get(BASIS_TO_FAMILY.get(b), "k"), alpha=0.7)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8); ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("posicao relativa do k escolhido na faixa de busca (0 = k_min, 1 = k_max; escala log)")
    ax.set_title("k escolhido por base (cada ponto = um dataset)"); fig.tight_layout(); fig.savefig(out / "k_por_base.png", dpi=130); plt.close(fig)

    M = best.groupby(["dataset", "basis"])["k"].mean().unstack()[order]
    fig, ax = plt.subplots(figsize=(0.45 * M.shape[1] + 3, 0.3 * M.shape[0] + 2))
    im = ax.imshow(np.log10(M.values), cmap="viridis", aspect="auto")
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns, rotation=75, ha="right", fontsize=7)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(M.index, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M.values[i, j]:.0f}", ha="center", va="center", fontsize=5.5, color="w")
    ax.set_title("k escolhido (cor = log10 k)"); fig.colorbar(im, ax=ax, shrink=0.7); fig.tight_layout()
    fig.savefig(out / "k_heatmap_dataset_base.png", dpi=130); plt.close(fig)

    d = best.dropna(subset=["n_train_fold"])
    if len(d):
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for f, g in d.groupby("familia"):
            ax.scatter(g["n_train_fold"], g["k"], s=14, alpha=0.65, label=f, color=pal.get(f))
        xs = np.logspace(math.log10(d["n_train_fold"].min()), math.log10(d["n_train_fold"].max()), 50)
        ax.plot(xs, 2 * np.sqrt(xs), "k--", lw=1, label="2*sqrt(n) (limite da faixa)")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("n treino (fold)"); ax.set_ylabel("k escolhido")
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "k_vs_n.png", dpi=130); plt.close(fig)

    e = edges.copy(); e["rotulo"] = e["basis"] + ":" + e["param"]
    e = e.sort_values("borda_total").tail(25)
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(e) + 1.5))
    ax.barh(e["rotulo"], e["borda_inf"], color="#4c78a8", label="borda inferior")
    ax.barh(e["rotulo"], e["borda_sup"], left=e["borda_inf"], color="#e45756", label="borda superior")
    ax.set_xlabel("fracao dos datasets com o melhor valor na borda da faixa"); ax.legend(fontsize=8)
    ax.set_title("Faixas de busca saturadas (maiores fracoes)"); fig.tight_layout(); fig.savefig(out / "faixas_saturadas.png", dpi=130); plt.close(fig)


# ------------------------------------------------------------------------------------------ relatorio
def _md(df, **kw):
    return df.to_markdown(**kw) if len(df) else "(vazio)"


def run(results, prefix, out, seeds=(42,), groups_path=None, figs=True):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    best, prof, par = load_scopes(results, prefix, seeds)
    if best.empty:
        raise SystemExit("Sem summaries para o prefixo/seeds informados.")
    groups = None
    if groups_path and Path(groups_path).exists():
        groups = pd.read_csv(groups_path, encoding="utf-8-sig").set_index("dataset")
    h1 = h1_bases_differ(best); h2d, h2 = h2_better_bases_k(best); h3 = h3_context(best, groups)
    edges = h4_edges(best, par); h5d, h5 = h5_profiles(prof)
    best.to_csv(out / "k_best_por_dataset_base.csv", index=False)
    h1["table"].to_csv(out / "H1_k_por_base.csv"); h2d.to_csv(out / "H2_rho_por_dataset.csv", index=False)
    edges.to_csv(out / "H4_bordas_da_faixa.csv", index=False); h5.to_csv(out / "H5_perfil_k_desempenho.csv")
    if h3 is not None:
        h3.to_csv(out / "H3_contexto.csv", index=False)
    if figs:
        figures(best, par, edges, out)

    L = [f"# Base de interpolacao x numero de vizinhos k\n",
         f"- Seeds: {sorted(set(best['seed']))} | datasets: **{best['dataset'].nunique()}** | bases: **{best['basis'].nunique()}** | "
         f"metrica de selecao: `{best['metric'].iloc[0]}` (a config vencedora e' a de maior media dessa metrica)\n",
         "## H1. As bases escolhem k diferentes?\n",
         f"Friedman + Iman-Davenport sobre a posicao relativa de k (N={h1['N']} datasets, k={h1['k']} bases): "
         f"**p = {h1['p']:.3g}**, W de Kendall = {h1['kendall_W']:.3f}. Por FAMILIA: p = {h1['fam_p']:.3g}.\n",
         "(W perto de 0: o k escolhido nao depende da base; perto de 1: cada base tem seu k tipico.)\n",
         _md(h1["table"].reset_index(names="basis"), index=False, floatfmt=".3f"), "\n![](k_por_base.png)\n",
         "## H2. Bases com melhor desempenho usam k maior ou menor?\n",
         f"Spearman(posto de desempenho, k) por dataset e teste de Wilcoxon sobre os {h2['n']} rhos: mediana = {h2['rho_mediano']:.3f}, "
         f"{100 * h2['frac_positivo']:.0f}% positivos, **p = {h2['p_wilcoxon']:.3g}**. (rho > 0: as MELHORES bases escolhem k maior.)\n"]
    if h3 is not None:
        L += ["## H3. O k escolhido depende do contexto do dataset?\n", _md(h3, index=False, floatfmt=".3g"), "\n![](k_vs_n.png)\n"]
    L += ["## H4. A faixa de busca esta bem posta? (melhor valor na borda = faixa estreita ou mal centrada)\n",
          f"Borda = {int(EDGE * 100)}% extremos da faixa. Linhas com `borda_total` alto merecem revisao da faixa.\n",
          _md(edges.head(25), index=False, floatfmt=".2f"), "\n![](faixas_saturadas.png)\n",
          "## H5. Perfil k x desempenho por base (entre as combinacoes sorteadas)\n",
          "rho > 0 => k maior tende a melhorar; < 0 => piorar.\n", _md(h5.reset_index(names="basis"), index=False, floatfmt=".2f"),
          "\n![](k_heatmap_dataset_base.png)\n",
          "\n> Todas as inferencias usam o dataset como unidade; com 1 seed cada celula carrega o acaso de um unico sorteio de 20 combinacoes, "
          "entao leia padroes, nao valores isolados de k."]
    (out / "k_analysis.md").write_text("\n".join(L), encoding="utf-8")
    return dict(h1=h1, h2=h2, h3=h3, edges=edges, h5=h5, best=best)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--prefix", default="v4")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42])
    ap.add_argument("--groups", default="reports/eda/dataset_characterization.csv")
    ap.add_argument("--out", default="reports/v4_k")
    ap.add_argument("--no-figs", action="store_true")
    a = ap.parse_args()
    run(a.results, a.prefix, a.out, a.seeds, a.groups, not a.no_figs)
    print(f"ok -> {a.out}/k_analysis.md")


if __name__ == "__main__":
    main()
