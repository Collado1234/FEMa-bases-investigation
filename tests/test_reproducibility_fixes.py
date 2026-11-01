"""Testes de regressao das correcoes de reprodutibilidade (pre-v3).

Rodar: python -m pytest tests/test_reproducibility_fixes.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from core import Basis, BasisParameters, BruteForceSearch, EuclideanDistance, FEMaClassifier
from persistence.summary_builder import _dedupe_runs
from pipeline.run_model import _check_config_consistency
from tuning.param_space import dedupe_combinations


def _search():
    return BruteForceSearch(metric=EuclideanDistance())


def _fit_predict(y_offset, basis="shepard", params=None, k=10):
    X, y = make_blobs(n_samples=400, centers=3, cluster_std=1.5, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    s = _search()
    m = FEMaClassifier(basis=Basis.get(basis, search=s), search=s)
    m.fit(Xtr, ytr + y_offset)
    labels, probs = m.predict(Xte, k=k, params=params or BasisParameters(z=2.0))
    return labels, probs, yte + y_offset


@pytest.mark.parametrize("offset", [1, 5])
def test_label_offset_invariance(offset):
    """Rotulos {1,2,3} devem dar as MESMAS predicoes de {0,1,2} (+offset)."""
    l0, p0, _ = _fit_predict(0)
    l1, p1, _ = _fit_predict(offset)
    assert np.array_equal(l0 + offset, l1)
    assert np.allclose(p0, p1)


def test_last_class_is_predictable():
    labels, _, y_true = _fit_predict(1)
    assert y_true.max() in set(labels), "a ultima classe nunca e' predita (bug dos rotulos)"


@pytest.mark.parametrize(
    "name,params,naive",
    [
        ("radial", dict(z=1.5), lambda d, p: np.exp(-0.5 * (d / p["z"]) ** 2)),
        ("rbf_gaussian", dict(epsilon=0.7), lambda d, p: np.exp(-((p["epsilon"] * d) ** 2))),
        ("laplacian", dict(epsilon=0.9), lambda d, p: np.exp(-p["epsilon"] * d)),
    ],
)
def test_log_path_matches_naive_formula(name, params, naive):
    """Onde nao ha underflow, o caminho log-sum-exp == phi/sum(phi)."""
    d = np.array([0.3, 0.8, 1.1, 1.9, 2.4])
    b = Basis.get(name, search=_search())
    phi = naive(d, params)
    assert np.allclose(b.compute_weights(d, BasisParameters(**params)), phi / phi.sum())


@pytest.mark.parametrize("name,key", [("radial", "z"), ("rbf_gaussian", "epsilon"), ("laplacian", "epsilon")])
def test_no_silent_uniform_fallback_on_underflow(name, key):
    """z muito pequeno (ou epsilon muito grande) NAO pode virar kNN uniforme."""
    d = np.array([1.0, 1.1, 1.4, 1.7, 2.0])
    b = Basis.get(name, search=_search())
    value = 0.0149 if key == "z" else 500.0
    w = b.compute_weights(d, BasisParameters(**{key: value}))
    assert np.isfinite(w).all() and np.isclose(w.sum(), 1.0)
    assert w[0] > 0.99, "peso deveria concentrar-se no vizinho mais proximo"
    assert b.fallback_fraction() == 0.0


def test_fallback_counter_for_non_log_bases():
    """Bases fora da familia log (ex.: wendland, suporte compacto) contam o fallback."""
    b = Basis.get("wendland_c2", search=_search())
    d = np.array([10.0, 11.0, 12.0])  # todos fora do raio => soma 0 => uniforme
    b.compute_weights(d, BasisParameters(h=1.0))
    assert b.n_weight_calls == 1 and b.n_uniform_fallback == 1


def test_tie_breaking_is_stable_and_lowest_index_first():
    s = _search()
    X = np.array([[0.0], [1.0], [1.0], [1.0], [1.0], [3.0]])
    s.build(X)
    idx, d = s.query(np.array([0.0]), k=3)
    assert list(idx) == [0, 1, 2]  # 0 (d=0) + empatados em d=1 por menor indice
    idx2, _ = s.query(np.array([0.0]), k=3)
    assert list(idx) == list(idx2)


def test_dedupe_combinations_keeps_order():
    combos = [{"k": 5}, {"k": 7}, {"k": 5}, {"k": 9}, {"k": 7}]
    assert dedupe_combinations(combos) == [{"k": 5}, {"k": 7}, {"k": 9}]


def test_summary_ignores_duplicate_runs():
    runs = [{"combo_id": "k=5", "repeat_idx": 0, "fold_idx": 0, "i": 1},
            {"combo_id": "k=5", "repeat_idx": 0, "fold_idx": 0, "i": 2},
            {"combo_id": "k=5", "repeat_idx": 0, "fold_idx": 1, "i": 3}]
    out = _dedupe_runs(runs)
    assert [r["i"] for r in out] == [1, 3]


def test_config_mismatch_is_rejected():
    runs = [{"config_hash": "aaa"}]
    with pytest.raises(RuntimeError):
        _check_config_consistency(runs, "bbb", ("classifier", "radial", "ds", "x"))
    _check_config_consistency(runs, "aaa", ("classifier", "radial", "ds", "x"))  # ok
    _check_config_consistency([{}], "bbb", ("a",))  # legado: so' avisa
