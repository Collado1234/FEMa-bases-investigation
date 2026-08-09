"""
analysis/check_basis_convergence.py

Diagnostico para a pergunta: "quando duas ou mais bases do FEMa empatam
EXATAMENTE em todas as metricas de teste, isso significa que o FEMa
convergiu para a mesma solucao otima, ou e' um artefato?"

Existem DUAS explicacoes bem diferentes para um empate exato, e confundi-las
leva a uma conclusao cientifica errada na IC:

  (a) DUPLICATA ESTRUTURAL DE FORMULA
      As bases empatadas calculam, na pratica, a MESMA funcao matematica
      sobre a distancia - so' com hiperparametros de nomes/faixas
      diferentes (ex.: `laplacian(epsilon)` e `softmax_radial(beta)` sao
      ambas `exp(-p * d)`). Nesse caso elas empatam SEMPRE, em qualquer
      dataset, porque sao literalmente a mesma formula com etiqueta
      trocada - e' um problema de implementacao (bug de copia/cola ou
      redundancia nao documentada), nao uma conclusao sobre o FEMa.

  (b) EMPATE COINCIDENTE NESSE DATASET
      As bases empatadas tem formulas GENUINAMENTE diferentes, mas
      produziram a mesma predicao de classe para todas as amostras de
      teste desse dataset especifico - por exemplo porque o k escolhido
      e' grande o suficiente para que a decisao de classe (argmax da
      soma ponderada) fique insensivel a forma exata do kernel. Isso e'
      uma coincidencia legitima desse dataset/split, nao um bug.

Este script:
  1. Le' o comparison.json ja' gerado (reports/basis_comparison/classifier/<dataset>/comparison.json).
  2. Agrupa as bases cujas test_metrics sao identicas (ate' tolerancia numerica).
  3. Para cada grupo com 2+ bases, testa se as formulas sao estruturalmente
     identicas: instancia cada base com os hiperparametros que o tuning
     escolheu e compara os PESOS NORMALIZADOS (compute_weights - o que
     realmente entra na predicao) sobre varios vetores sinteticos de
     distancias de vizinhos (nao vem do dataset - e' um teste puramente
     matematico da formula, robusto contra "empatou so' por causa dos
     dados").
  4. Classifica cada grupo como (a) ou (b) e imprime/salva um relatorio.

Uso:
    python -m analysis.check_basis_convergence --dataset fetal_health
    python -m analysis.check_basis_convergence --dataset wine --experiment-name oficial_v1
    python -m analysis.check_basis_convergence --all
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from core import Basis
from core.math.basis.parameters import BasisParameters

REPORTS_ROOT = Path(__file__).resolve().parent.parent / "reports" / "basis_comparison" / "classifier"

METRIC_DECIMALS = 9   # arredondamento usado para considerar duas metricas "iguais"
FORMULA_TOL = 1e-9    # tolerancia absoluta para considerar dois pesos normalizados "iguais"

# Vetores sinteticos de distancias de vizinhos usados no teste de formula.
# Deliberadamente variados (spreads e tamanhos de k diferentes) para nao
# deixar passar uma duplicata que so' coincide num formato particular de
# vizinhanca.
_RNG = np.random.default_rng(42)
_SYNTHETIC_NEIGHBOR_SETS = [
    np.sort(_RNG.uniform(1e-3, 1.0, size=5)),
    np.sort(_RNG.uniform(1e-3, 3.0, size=10)),
    np.sort(_RNG.uniform(1e-3, 10.0, size=15)),
    np.sort(_RNG.uniform(1e-3, 50.0, size=30)),
    np.linspace(1e-3, 5.0, 20),
]


def _load_comparison(dataset: str) -> dict:
    path = REPORTS_ROOT / dataset / "comparison.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Nao encontrei {path}.\n"
            f"Rode antes: python main.py --compare --context classifier --dataset {dataset} "
            f"--experiment-name <nome_usado_na_rodada>"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _tied_groups(comparison: dict) -> Tuple[List[List[dict]], List[str]]:
    """Agrupa bases com status 'ok' cujas test_metrics sao identicas
    (ate' METRIC_DECIMALS casas). So' retorna grupos com 2+ bases."""
    metrics_compared = comparison["metrics_compared"]
    buckets: Dict[tuple, List[dict]] = {}
    for row in comparison["bases"]:
        if row.get("status") != "ok" or row.get("test_metrics") is None:
            continue
        tm = row["test_metrics"]
        key = tuple(
            None if tm.get(m) is None else round(tm[m], METRIC_DECIMALS)
            for m in metrics_compared
        )
        buckets.setdefault(key, []).append(row)

    groups = [rows for rows in buckets.values() if len(rows) > 1]
    groups.sort(key=len, reverse=True)
    return groups, metrics_compared


def _basis_params_from_hyperparams(hyperparams: dict) -> BasisParameters:
    """Converte o dict best_hyperparameters (que inclui 'k', que nao e'
    campo de BasisParameters) no objeto BasisParameters correspondente."""
    fields = {k: v for k, v in hyperparams.items() if k != "k"}
    return BasisParameters(**fields)


def _formula_identity_check(basis_a: str, hp_a: dict, basis_b: str, hp_b: dict) -> Tuple[bool, float]:
    """Compara os PESOS NORMALIZADOS (compute_weights) de duas bases, com
    os hiperparametros que cada uma teve escolhidos pelo tuning, sobre
    varios vetores sinteticos de distancias. Se os pesos batem em TODOS
    os vetores testados -> forte evidencia de duplicata estrutural de
    formula (nao depende do dataset). Retorna (identicas?, maior_diff)."""
    inst_a = Basis.get(basis_a)
    inst_b = Basis.get(basis_b)
    params_a = _basis_params_from_hyperparams(hp_a)
    params_b = _basis_params_from_hyperparams(hp_b)

    max_diff = 0.0
    for dists in _SYNTHETIC_NEIGHBOR_SETS:
        try:
            w_a = inst_a.compute_weights(dists, params_a)
            w_b = inst_b.compute_weights(dists, params_b)
        except Exception:
            # base nao suporta esse vetor sintetico (nao deveria acontecer,
            # mas nao queremos que o diagnostico quebre por causa disso)
            return False, float("nan")
        diff = float(np.max(np.abs(w_a - w_b)))
        max_diff = max(max_diff, diff)
        if max_diff > FORMULA_TOL:
            return False, max_diff
    return True, max_diff


def analyze_dataset(dataset: str, verbose: bool = True) -> dict:
    comparison = _load_comparison(dataset)
    groups, metrics_compared = _tied_groups(comparison)

    report = {"dataset": dataset, "total_bases": len(comparison["bases"]), "tied_groups": []}

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Dataset: {dataset}  ({len(comparison['bases'])} bases avaliadas)")
        print(f"{'=' * 70}")

    if not groups:
        if verbose:
            print("Nenhum empate exato entre bases - todas as metricas de teste sao distintas.")
        return report

    for group in groups:
        names = [r["basis"] for r in group]
        by_name = {r["basis"]: r for r in group}

        if verbose:
            print(f"\nGRUPO EMPATADO (n={len(names)}): {names}")
            print(f"  metricas: {group[0]['test_metrics']}")

        pair_results = []
        any_structural = False
        for a, b in combinations(names, 2):
            hp_a = by_name[a]["best_hyperparameters"]
            hp_b = by_name[b]["best_hyperparameters"]
            identical, max_diff = _formula_identity_check(a, hp_a, b, hp_b)
            any_structural = any_structural or identical
            pair_results.append(
                {"pair": [a, b], "structurally_identical": identical, "max_weight_diff": max_diff}
            )
            if verbose:
                verdict = "DUPLICATA ESTRUTURAL (mesma formula)" if identical else "formulas diferentes"
                print(f"    {a} vs {b}: {verdict}  (max diff pesos = {max_diff:.3e})")

        verdict = "DUPLICATA_ESTRUTURAL" if any_structural else "EMPATE_COINCIDENTE_NESTE_DATASET"
        if verbose:
            if any_structural:
                print(f"  -> VEREDITO: {verdict}. Pelo menos um par calcula a MESMA funcao matematica "
                      f"(so' com hiperparametros renomeados) - vai empatar em qualquer dataset. "
                      f"Vale reportar como redundancia de implementacao, nao como resultado do FEMa.")
            else:
                print(f"  -> VEREDITO: {verdict}. As formulas sao matematicamente diferentes; o empate "
                      f"e' uma coincidencia de predicao especifica desse dataset/split (ex.: k grande "
                      f"saturando a decisao de classe). Legitimo, mas nao indica que as bases sao "
                      f"'equivalentes' em geral.")

        report["tied_groups"].append({
            "bases": names,
            "test_metrics": group[0]["test_metrics"],
            "best_hyperparameters": {n: by_name[n]["best_hyperparameters"] for n in names},
            "verdict": verdict,
            "pairwise_formula_check": pair_results,
        })

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, default=None, help="Nome do dataset (ex: wine, breast_cancer, digits, fetal_health).")
    parser.add_argument("--all", action="store_true", help="Roda para todo dataset que tenha comparison.json gerado.")
    parser.add_argument("--output", type=str, default=None, help="Caminho do JSON de saida. Default: reports/basis_comparison/classifier/<dataset>/convergence_check.json")
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("Forneca --dataset <nome> ou --all.")

    targets = []
    if args.all:
        if not REPORTS_ROOT.exists():
            parser.error(f"{REPORTS_ROOT} nao existe - rode --compare em algum dataset primeiro.")
        targets = sorted(p.name for p in REPORTS_ROOT.iterdir() if (p / "comparison.json").exists())
        if not targets:
            parser.error(f"Nenhum comparison.json encontrado em {REPORTS_ROOT}.")
    else:
        targets = [args.dataset]

    for dataset in targets:
        report = analyze_dataset(dataset)
        out_path = Path(args.output) if (args.output and len(targets) == 1) else (
            REPORTS_ROOT / dataset / "convergence_check.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[salvo] {out_path}")


if __name__ == "__main__":
    main()
