"""Testes da bateria estatistica (rodar: python -m pytest tests/test_stat_battery.py -q)."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reporting.stat_battery import loading, per_dataset, primitives as P  # noqa: E402
from tests import synth_results  # noqa: E402


def test_holm_bh_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(P.holm(p), [0.03, 0.06, 0.06, 0.02])
    assert np.all(P.benjamini_hochberg(p) <= P.holm(p) + 1e-12)


def test_edf_matches_scipy_no_ties():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0, 1, 40), rng.normal(0.3, 1.2, 35)
    r = P.edf_two_sample(a, b, deff=1.0, n_perm=1999)
    assert abs(r["ks_D"] - stats.ks_2samp(a, b).statistic) < 1e-12
    assert abs(r["cvm_w2"] - stats.cramervonmises_2samp(a, b).statistic) < 1e-9
    assert abs(r["ks_p"] - stats.ks_2samp(a, b).pvalue) < 0.05
    assert abs(r["cvm_p"] - stats.cramervonmises_2samp(a, b).pvalue) < 0.05


def test_edf_handles_ties_and_deff_shrinks_significance():
    rng = np.random.default_rng(4)
    a = np.round(rng.normal(0.80, 0.03, 30), 2); b = np.round(rng.normal(0.84, 0.03, 30), 2)
    p1 = P.edf_two_sample(a, b, deff=1.0)["ks_p"]; p2 = P.edf_two_sample(a, b, deff=4.0)["ks_p"]
    assert p2 >= p1


def test_paired_swap_matches_scipy_ks_and_deff_shrinks():
    rng = np.random.default_rng(8)
    blk = rng.normal(0, 0.03, 30)
    a, b = 0.8 + blk + rng.normal(0, 0.005, 30), 0.82 + blk + rng.normal(0, 0.005, 30)
    r1, r4 = P.edf_paired_swap(a, b, deff=1.0, n_perm=499), P.edf_paired_swap(a, b, deff=4.0, n_perm=499)
    assert abs(r1["ks_D"] - stats.ks_2samp(a, b).statistic) < 1e-12
    assert r4["ks_p"] >= r1["ks_p"] and r4["cvm_p"] >= r1["cvm_p"]
    # pareado detecta o deslocamento que o nao pareado perde (dificuldade do fold comum as duas bases)
    assert r1["cvm_p"] < P.edf_two_sample(a, b, n_perm=499)["cvm_p"]


def test_corrected_t_is_more_conservative_than_naive():
    d = np.random.default_rng(0).normal(0.01, 0.02, 30)
    assert P.corrected_t(d, 0.0)["p"] < P.corrected_t(d, 0.11)["p"]


def test_tost_detects_equivalence_and_not_difference():
    rng = np.random.default_rng(5)
    d0 = rng.normal(0.0, 0.004, 90)
    assert P.tost_corrected(d0, 0.11, 0.01)["p_tost"] < 0.05
    d1 = rng.normal(0.03, 0.004, 90)
    assert P.tost_corrected(d1, 0.11, 0.01)["p_tost"] > 0.5


def test_end_to_end_recovers_planted_truth():
    with tempfile.TemporaryDirectory() as tmp:
        synth_results.make(Path(tmp))
        folds, tuning = loading.load_fold_scores(tmp, "v3", "balanced_accuracy")
        assert folds["basis"].nunique() == 8 and len(folds) == 8 * 8 * 2 * 30
        wide, problems, meta = loading.complete_blocks(folds)
        bases = sorted(folds["basis"].unique())
        omni, pairs, summ = per_dataset.run_all_datasets(wide, bases, meta, n_perm=199, delta=0.02)
        assert (omni["p_friedman_deff"] < 0.05).all()                       # ha' diferencas plantadas em todos
        sub = pairs[(pairs.a == "multiquadratic") | (pairs.b == "multiquadratic")]
        worst = np.where(sub.a == "multiquadratic", sub.desfecho == "b_melhor", sub.desfecho == "a_melhor")
        assert worst.mean() > 0.9                                           # multiquadratic perde quase sempre
        pr = pairs[(pairs.a == "radial") & (pairs.b == "rbf_gaussian")]
        assert (pr.desfecho == "equivalente").mean() > 0.8                  # radial ~ rbf: equivalentes (TOST)
        top = summ[summ.dataset == "isolet"].set_index("basis")
        assert top.loc["wendland_c2", "grupo_topo"] and not top.loc["multiquadratic", "grupo_topo"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
