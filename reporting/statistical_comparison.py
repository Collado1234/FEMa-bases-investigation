"""Comparacao estatistica entre bases, seguindo Demsar (2006) + pos-hoc moderno.

Unidade de analise = DATASET (nao o fold de CV): matriz N datasets x k bases, uma celula =
media, nas seeds disponiveis, da metrica de TESTE (holdout) em results/**/test_results.json.

  1. Friedman (tie-corrected) + correcao de Iman-Davenport (F_F)      [Demsar 2006]
  2. Distancia critica de Nemenyi (so' para o diagrama, descritivo)   [Demsar 2006]
  3. Pos-hoc: Wilcoxon signed-rank vs CONTROLE + correcao de Holm     [Garcia&Herrera 2008;
     Benavoli et al. 2016; Ismail Fawaz et al. 2019]
  4. Baselines externos (knn, logreg): Wilcoxon pareado entre datasets (+ Holm)

Uso:
    python -m reporting.statistical_comparison --metric balanced_accuracy --control shepard
Defina metrica primaria e controle ANTES de olhar os resultados (evita p-hacking).
"""
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SEED_RE = re.compile(r"_s(\d+)$")


# ----------------------------------------------------------------- carga
def load_cells(results: Path, prefix: str, metric: str, seeds=None, source: str = "test"):
    """-> dict[(grupo, modelo, dataset)][seed] = valor.  grupo: 'basis' | 'baseline'.

    source='test': metrica no HOLDOUT (test_results.json)  -> estimativa nao enviesada (primaria).
    source='cv'  : media de CV da config escolhida (summary.json) -> menos ruido, mas otimista
                   (selecao em cima do proprio CV); use como checagem de robustez.
    """
    import json
    fname = "test_results.json" if source == "test" else "summary.json"
    cells = defaultdict(dict)
    for f in results.rglob(fname):
        parts = f.relative_to(results).parts          # ctx/modelo/dataset/exp/<arquivo>
        if len(parts) != 5 or not parts[3].startswith(prefix):
            continue
        m = SEED_RE.search(parts[3])
        if not m or (seeds and int(m.group(1)) not in seeds):
            continue
        data = json.load(open(f, encoding="utf-8"))
        if source == "test":
            val = (data.get("metrics") or {}).get(metric)
        else:
            val = (((data.get("best_configuration") or {}).get("metrics") or {}).get(metric) or {}).get("mean")
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        group = "baseline" if parts[0] == "external_baselines" else "basis"
        cells[(group, parts[1], parts[2])][int(m.group(1))] = float(val)
    return cells


def build_matrix(cells, group: str):
    """DataFrame datasets x modelos, media nas seeds COMUNS a todos os modelos do dataset."""
    by_ds = defaultdict(dict)
    for (g, model, ds), seeds in cells.items():
        if g == group:
            by_ds[ds][model] = seeds
    rows, info, dropped = {}, {}, []
    models = sorted({m for d in by_ds.values() for m in d})
    for ds, d in sorted(by_ds.items()):
        if set(d) != set(models):
            dropped.append((ds, f"faltam modelos: {sorted(set(models) - set(d))[:4]}..."))
            continue
        common = set.intersection(*[set(s) for s in d.values()])
        if not common:
            dropped.append((ds, "sem seed em comum entre os modelos"))
            continue
        rows[ds] = {m: float(np.mean([d[m][s] for s in common])) for m in models}
        info[ds] = len(common)
    return pd.DataFrame(rows).T[models] if rows else pd.DataFrame(), info, dropped


# ------------------------------------------------------------- estatistica
def friedman_iman_davenport(M: pd.DataFrame):
    N, k = M.shape
    chi2, p_chi = stats.friedmanchisquare(*[M[c].values for c in M.columns])   # corrige empates
    denom = N * (k - 1) - chi2
    if not np.isfinite(chi2) or denom <= 0:
        return dict(N=N, k=k, chi2=chi2, p_chi2=p_chi, F=np.nan, p_F=np.nan)
    F = (N - 1) * chi2 / denom
    return dict(N=N, k=k, chi2=chi2, p_chi2=p_chi, F=F, p_F=stats.f.sf(F, k - 1, (k - 1) * (N - 1)))


def average_ranks(M: pd.DataFrame, higher_is_better=True) -> pd.Series:
    R = M.apply(lambda r: stats.rankdata(-r.values if higher_is_better else r.values), axis=1, result_type="expand")
    R.columns = M.columns
    return R.mean()


def nemenyi_cd(k: int, N: int, alpha=0.05) -> float:
    q = stats.studentized_range.ppf(1 - alpha, k, 10**6) / math.sqrt(2)
    return float(q * math.sqrt(k * (k + 1) / (6 * N)))


def holm(pvals: np.ndarray) -> np.ndarray:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def wilcoxon_pair(a: np.ndarray, b: np.ndarray):
    d = a - b
    nz = d[d != 0]
    if len(nz) == 0:
        return dict(p=1.0, r_rb=0.0, wins=0, ties=len(d), losses=0, mean_diff=0.0)
    p = stats.wilcoxon(d, zero_method="pratt", alternative="two-sided").pvalue
    ranks = stats.rankdata(np.abs(nz))
    wp, wm = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    return dict(p=float(p), r_rb=float((wp - wm) / (wp + wm)), wins=int((d > 0).sum()),
                ties=int((d == 0).sum()), losses=int((d < 0).sum()), mean_diff=float(d.mean()))


def vs_control(M: pd.DataFrame, control: str) -> pd.DataFrame:
    rows = [dict(basis=c, **wilcoxon_pair(M[c].values, M[control].values)) for c in M.columns if c != control]
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].values)
    out["significativo_holm_0.05"] = out["p_holm"] < 0.05
    return out.sort_values("p").reset_index(drop=True)


# ------------------------------------------------------------------ grafico
def plot_cd(avg: pd.Series, cd: float, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = avg.sort_values()
    k = len(s)
    fig, ax = plt.subplots(figsize=(10, 0.35 * k + 2))
    ax.hlines(range(k), s.min() - 0.3, s.values, color="#cccccc", lw=1)
    ax.plot(s.values, range(k), "o", color="#1f6feb")
    ax.set_yticks(range(k)); ax.set_yticklabels(s.index)
    ax.set_xlabel("posto medio (1 = melhor)")
    ax.set_ylim(k - 0.5, -2.4)                      # espaco no topo para a barra de CD
    ax.hlines(-1.4, s.min(), s.min() + cd, color="black", lw=2)
    ax.vlines([s.min(), s.min() + cd], -1.6, -1.2, color="black", lw=2)
    ax.text(s.min() + cd / 2, -1.9, f"CD = {cd:.2f}", ha="center", fontsize=9)
    ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# --------------------------------------------------------------------- main
def run(results: Path, prefix: str, metric: str, control: str, out: Path, alpha=0.05, higher_is_better=True,
        exclude=(), seeds=None, source="test"):
    out.mkdir(parents=True, exist_ok=True)
    cells = load_cells(results, prefix, metric, seeds, source)
    M, info, dropped = build_matrix(cells, "basis")
    M = M.drop(index=[d for d in exclude if d in M.index])
    if M.empty or control not in M.columns:
        raise SystemExit(f"Sem dados suficientes (matriz vazia ou controle '{control}' ausente).")

    tied = [d for d in M.index if M.loc[d].nunique() == 1]            # dataset sem informacao
    Mi = M.drop(index=tied)
    k = M.shape[1]
    fr = friedman_iman_davenport(Mi)
    avg = average_ranks(Mi, higher_is_better)
    cd = nemenyi_cd(k, len(Mi), alpha)
    post = vs_control(Mi, control)

    ranking = pd.DataFrame({"posto_medio": avg, "media_metrica": Mi.mean()}).sort_values("posto_medio")
    ranking.index.name = "basis"

    M.to_csv(out / "matrix_dataset_x_basis.csv")
    ranking.to_csv(out / "ranking_postos_medios.csv")
    post.to_csv(out / "posthoc_vs_controle_wilcoxon_holm.csv", index=False)
    plot_cd(avg, cd, out / "cd_diagram.png", f"Postos medios ({metric}), N={len(Mi)} datasets, k={k}")

    # baselines
    base_md = ""
    B, _, bdrop = build_matrix(cells, "baseline")
    if not B.empty:
        common = [d for d in Mi.index if d in B.index]
        if len(common) >= 5:
            best = ranking.index[0]
            rows = []
            for cand in dict.fromkeys([control, best]):
                for bl in B.columns:
                    rows.append(dict(base=cand, baseline=bl, n_datasets=len(common),
                                     **wilcoxon_pair(M.loc[common, cand].values, B.loc[common, bl].values)))
            bt = pd.DataFrame(rows); bt["p_holm"] = holm(bt["p"].values)
            bt.to_csv(out / "baselines_wilcoxon_holm.csv", index=False)
            base_md = "\n## Baselines (Wilcoxon pareado entre datasets, Holm)\n\n" + bt.round(4).to_markdown(index=False)
            base_md += f"\n\n(`{best}` e' a base de melhor posto medio: escolhida a posteriori, interprete com cautela.)\n"

    md = [f"# Comparacao estatistica das bases ({metric}; fonte: {'holdout de teste' if source == 'test' else 'media de CV da config escolhida'}; "
          f"seeds: {sorted(seeds) if seeds else 'todas'})", "",
          f"- Datasets na matriz: **{len(M)}** (usados no teste global: {len(Mi)}; "
          f"{len(tied)} sem informacao porque todas as bases empataram: {tied or 'nenhum'})",
          f"- Bases: **{k}** | controle: **{control}** | alpha = {alpha}",
          f"- Seeds por dataset (interseccao entre bases): min {min(info.values())}, max {max(info.values())}",
          f"- Datasets excluidos por dados incompletos: {dropped or 'nenhum'}", "",
          "## Teste global (Friedman + Iman-Davenport, Demsar 2006)", "",
          f"- chi2_F = {fr['chi2']:.3f} (p = {fr['p_chi2']:.3g})",
          f"- F_F = {fr['F']:.3f}, gl = ({k-1}, {(k-1)*(len(Mi)-1)}), **p = {fr['p_F']:.3g}**", "",
          f"Distancia critica de Nemenyi (so' para o diagrama): **CD = {cd:.2f}** postos.", "",
          "## Ranking por posto medio", "", ranking.round(4).to_markdown(), "",
          f"## Pos-hoc: cada base vs controle `{control}` (Wilcoxon signed-rank, zeros: Pratt; Holm)", "",
          post.round(4).to_markdown(index=False), base_md,
          "\n*Unidade de analise: dataset. Cada celula e' a media da metrica de teste nas seeds comuns.*"]
    (out / "statistical_report.md").write_text("\n".join(md), encoding="utf-8")
    return dict(friedman=fr, cd=cd, ranking=ranking, posthoc=post, matrix=M, tied=tied, dropped=dropped)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--prefix", default="v3")
    ap.add_argument("--metric", default="balanced_accuracy")
    ap.add_argument("--control", default="shepard")
    ap.add_argument("--out", default=str(ROOT / "reports" / "v3_stats"))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="usa so' estas seeds (ex.: 42)")
    ap.add_argument("--source", choices=["test", "cv"], default="test")
    ap.add_argument("--lower-is-better", action="store_true")
    a = ap.parse_args()
    r = run(Path(a.results), a.prefix, a.metric, a.control, Path(a.out), a.alpha, not a.lower_is_better, a.exclude,
            set(a.seeds) if a.seeds else None, a.source)
    print(f"Friedman/Iman-Davenport: F={r['friedman']['F']:.3f}, p={r['friedman']['p_F']:.3g} | CD={r['cd']:.2f} | "
          f"relatorio: {a.out}/statistical_report.md")