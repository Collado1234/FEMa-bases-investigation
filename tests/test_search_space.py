"""Testes do espaco de hiperparametros relativo a densidade e das correcoes numericas das bases.

Rodar: python -m pytest tests/test_search_space.py -v
"""
from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pytest

from core import Basis, BasisParameters, BruteForceSearch, EuclideanDistance
from models.fema import (PARAM_GRIDS, RANDOM_SEARCH_SPACES, FEMaPlugin, available_bases, k_upper_bound)
from tuning.random_search import sample_param_space
from utils.hashing import stable_hash

COMPACT = ["wendland_c2", "cubic_spline", "quartic_spline", "cosine"]


def _search():
    return BruteForceSearch(metric=EuclideanDistance())


# ------------------------------------------------------------------ suporte compacto
@pytest.mark.parametrize("name", COMPACT)
def test_h_rel_one_equals_automatic_radius(name):
    d = np.array([0.4, 0.9, 1.3, 2.2])
    b = Basis.get(name, search=_search())
    assert np.allclose(b.compute_weights(d, BasisParameters(h_rel=1.0)), b.compute_weights(d, BasisParameters()))


@pytest.mark.parametrize("name", COMPACT)
@pytest.mark.parametrize("scale", [0.01, 1.0, 37.0, 1e3])
def test_h_rel_is_scale_invariant(name, scale):
    """Propriedade central: multiplicar todas as distancias por uma constante (outra dimensao/escala de
    dados) NAO muda os pesos quando o raio e' relativo. Com h absoluto isso falha."""
    d = np.array([0.4, 0.9, 1.3, 2.2, 2.5])
    b = Basis.get(name, search=_search())
    base = b.compute_weights(d, BasisParameters(h_rel=1.5))
    assert np.allclose(b.compute_weights(scale * d, BasisParameters(h_rel=1.5)), base)


@pytest.mark.parametrize("name", COMPACT)
def test_absolute_h_is_not_scale_invariant_and_degenerates(name):
    """Documenta o defeito antigo: com h absoluto e distancias grandes, tudo cai fora do suporte."""
    d = np.array([20.0, 22.0, 25.0, 31.0])
    b = Basis.get(name, search=_search())
    w = b.compute_weights(d, BasisParameters(h=5.0))
    assert np.allclose(w, 0.25) and b.n_uniform_fallback == 1          # virou kNN uniforme (fallback)


@pytest.mark.parametrize("name", COMPACT)
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e4])
@pytest.mark.parametrize("h_rel", [1.05, 1.5, 3.0])
def test_h_rel_above_one_never_falls_back(name, scale, h_rel):
    rng = np.random.default_rng(0)
    d = np.sort(rng.uniform(1.0, 2.0, 12)) * scale
    b = Basis.get(name, search=_search())
    w = b.compute_weights(d, BasisParameters(h_rel=h_rel))
    assert b.n_uniform_fallback == 0 and np.isclose(w.sum(), 1.0) and (w > 0).all()
    assert w[0] > w[-1]                                                  # mais perto => mais peso


@pytest.mark.parametrize("name", COMPACT)
def test_all_neighbors_at_distance_zero_is_finite_uniform(name):
    b = Basis.get(name, search=_search())
    w = b.compute_weights(np.zeros(6), BasisParameters(h_rel=2.0))
    assert np.isfinite(w).all() and np.allclose(w, 1 / 6)


# ------------------------------------------------------------------ bases exponenciais / entropic
def test_entropic_has_no_nan_with_large_beta_and_distances():
    d = np.sort(np.random.default_rng(1).uniform(20, 40, 15))
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                   # qualquer 0/0 vira erro
        w = Basis.get("entropic", search=_search()).compute_weights(d, BasisParameters(beta=50.0))
    assert np.isfinite(w).all() and np.isclose(w.sum(), 1.0)


def test_gaussian_family_is_one_function_reparametrized():
    d = np.sort(np.random.default_rng(2).uniform(0.6, 2.5, 12))
    beta = 1.3
    s = _search()
    w_ent = Basis.get("entropic", search=s).compute_weights(d, BasisParameters(beta=beta))
    assert np.allclose(w_ent, Basis.get("rbf_gaussian", search=s).compute_weights(d, BasisParameters(epsilon=math.sqrt(beta))))
    assert np.allclose(w_ent, Basis.get("radial", search=s).compute_weights(d, BasisParameters(z=1 / math.sqrt(2 * beta))))


@pytest.mark.parametrize("eps", [0.3, 2.0, 50.0])
def test_laplacian_equals_softmax_radial_even_under_underflow(eps):
    d = np.sort(np.random.default_rng(3).uniform(15, 40, 15))
    s = _search()
    a = Basis.get("laplacian", search=s).compute_weights(d, BasisParameters(epsilon=eps))
    b = Basis.get("softmax_radial", search=s).compute_weights(d, BasisParameters(beta=eps))
    assert np.allclose(a, b)


# ------------------------------------------------------------------ espaco de k e random search
@pytest.mark.parametrize("n,expected", [(70, 14), (124, 23), (1257, 71), (5457, 148), (14000, 237), (20, 10)])
def test_k_upper_bound(n, expected):
    assert k_upper_bound(n) == expected


def test_k_space_depends_on_training_size():
    p = FEMaPlugin("classifier", "shepard")
    assert p.random_search_space()["k"] == ("randint", 3, 30)           # sem dados: legado
    p.bind_data(np.zeros((5457, 3)), None)
    assert p.random_search_space()["k"] == ("logint", 3, 148)
    grid = p.parameter_grid()["k"]
    assert grid == sorted(set(grid)) and grid[0] == 3 and grid[-1] == 148


def test_logint_sampling_stays_in_range_and_is_log_like():
    ks = [c["k"] for c in sample_param_space({"k": ("logint", 3, 148)}, 4000, seed=0)]
    assert min(ks) == 3 and max(ks) == 148 and all(isinstance(k, int) for k in ks)
    assert abs(np.median(ks) - math.sqrt(3 * 148)) < 4                    # mediana ~ media geometrica (21)


@pytest.mark.parametrize("basis", available_bases())
def test_every_basis_builds_a_valid_plugin_with_new_spaces(basis):
    p = FEMaPlugin("classifier", basis)                                   # _validate_basis_entry roda aqui
    p.bind_data(np.zeros((500, 4)), None)
    assert "k" in p.random_search_space() and "k" in p.parameter_grid()
    for name in COMPACT:
        assert PARAM_GRIDS[name] == {"h_rel": PARAM_GRIDS[name]["h_rel"]}
        assert RANDOM_SEARCH_SPACES[name]["h_rel"][1] > 1.0               # limite inferior > 1: sem fallback


def test_config_hash_changes_when_search_space_changes():
    a = FEMaPlugin("classifier", "cosine"); b = FEMaPlugin("classifier", "cosine"); b.bind_data(np.zeros((900, 3)), None)
    assert stable_hash({"s": a.random_search_space()}) != stable_hash({"s": b.random_search_space()})


# ------------------------------------------------------------------ ponta a ponta
def test_pipeline_uses_relative_spaces_and_never_falls_back(tmp_path, monkeypatch):
    import persistence.run_writer as rw
    monkeypatch.setattr(rw, "RESULTS_DIR", tmp_path)
    from pipeline.run_model import run_basis_experiment
    res = run_basis_experiment(context="classifier", basis="wendland_c2", dataset="breast_cancer", experiment_name="t_s42",
                               n_splits=3, n_repeats=1, tuning_n_iter=5, master_seed=42)
    s = res["summary"]
    lo, hi = 3, k_upper_bound(397)
    for combo in s["ranking"]:
        hp = combo["hyperparameters"]
        assert set(hp) == {"k", "h_rel"} and lo <= hp["k"] <= hi and 1.05 <= hp["h_rel"] <= 3.0
        assert combo["diagnostics"]["frac_uniform_fallback"]["mean"] == 0.0
    cfg = s["config"]["search_space"]
    assert cfg["k"][0] == "logint" and cfg["h_rel"][0] == "loguniform"
    # refazer com OUTRO espaco no mesmo diretorio deve ser recusado (config_hash)
    import models.fema as fm
    monkeypatch.setattr(fm, "_H_REL_SPACE", ("loguniform", 1.05, 4.0))
    monkeypatch.setitem(fm.RANDOM_SEARCH_SPACES, "wendland_c2", {"h_rel": ("loguniform", 1.05, 4.0)})
    with pytest.raises(RuntimeError):
        run_basis_experiment(context="classifier", basis="wendland_c2", dataset="breast_cancer", experiment_name="t_s42",
                             n_splits=3, n_repeats=1, tuning_n_iter=5, master_seed=42)
