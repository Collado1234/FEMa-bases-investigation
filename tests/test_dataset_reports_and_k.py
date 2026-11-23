"""Testes dos relatorios por dataset e da analise k x base (usa results sintetico com verdade plantada).

Rodar: python -m pytest tests/test_dataset_reports_and_k.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reporting import k_analysis  # noqa: E402
from reporting.stat_battery import dataset_reports as DR  # noqa: E402
from tests import synth_results  # noqa: E402


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    synth_results.make(root, seeds=(42, 43))
    return root


@pytest.fixture(scope="module")
def reports(synth, tmp_path_factory):
    out = tmp_path_factory.mktemp("rep")
    idx, long_df = DR.build(synth, "v3", out, seeds=[42], control="shepard", n_perm=49, figs=True)
    return out, idx, long_df


def test_per_dataset_files_and_all_four_metrics(reports):
    out, idx, long_df = reports
    assert set(long_df["metrica"]) == {"balanced_accuracy", "f1", "precision", "recall"}
    assert (out / "INDEX.md").exists() and (out / "ranking_por_dataset_metrica.csv").exists()
    for ds in synth_results.DEFAULT_DATASETS:
        md = (out / "por_dataset" / f"{ds}.md").read_text(encoding="utf-8")
        for title in ("## Balanced accuracy", "## F1", "## Precisao", "## Revocacao", "## Consistencia entre metricas"):
            assert title in md
        assert (out / "por_dataset" / f"{ds}.png").exists()


def test_uses_only_requested_seed(reports):
    _, idx, _ = reports
    assert (idx["n_blocos"] == 30).all()                       # 10 folds x 3 repeats x 1 seed


def test_recovers_planted_ranking_and_significance(reports):
    _, _, long_df = reports
    t = long_df[(long_df.metrica == "balanced_accuracy") & (long_df.dataset == "isolet")].set_index("basis")
    assert t.loc["laplacian", "rank"] == 1
    assert t.loc["multiquadratic", "rank"] == t["rank"].max()
    assert t.loc["multiquadratic", "vs_melhor"] == "pior (sig.)" and t.loc["multiquadratic", "vs_controle"] == "pior (sig.)"
    assert bool(t.loc["wendland_c2", "grupo_topo"]) and not bool(t.loc["multiquadratic", "grupo_topo"])
    assert t.loc["radial", "vs_melhor"] in ("equivalente", "inconclusivo")      # radial ~ melhor
    assert t.loc["shepard", "vs_controle"] == "(controle)"


def test_gaussian_duplicates_are_equivalent_to_each_other(reports):
    out, _, _ = reports
    pr = pd.read_csv(out / "pares_por_dataset_metrica.csv")
    sub = pr[(pr.a == "radial") & (pr.b == "rbf_gaussian") & (pr.metrica == "balanced_accuracy")]
    # mesma funcao (efeito plantado igual): NUNCA pode sair diferenca robusta; a maioria e' equivalente
    # (onde cada base sorteia combinacoes diferentes o veredito correto e' 'inconclusivo', nao 'diferente')
    assert not sub["desfecho"].isin(["a_melhor", "b_melhor"]).any()
    assert (sub["desfecho"] == "equivalente").mean() >= 0.6


def test_metrics_are_ranked_consistently_in_planted_data(reports):
    _, _, long_df = reports
    for ds, g in long_df[long_df.basis.isin(["laplacian", "multiquadratic"])].groupby("dataset"):
        for m, h in g.groupby("metrica"):
            r = h.set_index("basis")["rank"]
            assert r["laplacian"] < r["multiquadratic"]


# ---------------------------------------------------------------------------- k x base
def test_k_analysis_detects_planted_base_dependence(synth, tmp_path):
    res = k_analysis.run(synth, "v3", tmp_path, seeds=[42], groups_path=None, figs=True)
    assert res["h1"]["p"] < 0.01 and res["h1"]["kendall_W"] > 0.3
    tab = res["h1"]["table"]
    assert tab.loc["wendland_c2", "k_pos_medio"] > tab.loc["multiquadratic", "k_pos_medio"] + 0.3   # compactas: k maior
    assert (tmp_path / "k_analysis.md").exists() and (tmp_path / "k_por_base.png").exists()
    assert res["h2"]["n"] >= 5


def test_k_analysis_edge_flags_saturated_range(tmp_path):
    root = tmp_path / "r"
    # todas as bases preferem k enorme => melhor k sempre colado na borda superior da faixa
    bases = synth_results.DEFAULT_BASES[:4]
    synth_results.make(root, bases=bases, datasets=synth_results.DEFAULT_DATASETS[:5], seeds=(42,),
                       k_pref={b: 500 for b in bases}, n_combos=20)
    res = k_analysis.run(root, "v3", tmp_path / "o", seeds=[42], figs=False)
    e = res["edges"].query("param == 'k'")
    assert e["borda_sup"].mean() > 0.6 and e["borda_inf"].mean() < 0.1


def test_k_analysis_context_with_groups(synth, tmp_path):
    ds = synth_results.DEFAULT_DATASETS
    g = pd.DataFrame({"dataset": ds, "n_amostras": np.linspace(100, 20000, len(ds)), "n_features": np.arange(4, 4 + len(ds)) * 7,
                      "n_classes": [3, 3, 3, 2, 7, 26, 26, 7]})
    gp = tmp_path / "groups.csv"; g.to_csv(gp, index=False)
    res = k_analysis.run(synth, "v3", tmp_path / "o", seeds=[42], groups_path=gp, figs=False)
    assert res["h3"] is not None and {"covariavel", "alvo", "rho", "p_bh"} <= set(res["h3"].columns)
