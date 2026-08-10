"""
analysis/statistical_comparison.py

Comparacao estatistica pareada entre funcoes de base do FEMa (Friedman +
post-hoc Wilcoxon signed-rank com correcao de Holm), por dataset e por
metrica.

DE ONDE VEM O PAREAMENTO
-------------------------
summary.json (persistence/summary_builder.py) so' guarda media/desvio/n
AGREGADOS por combinacao de hiperparametros - os valores individuais das
30 avaliacoes (n_splits=10 x n_repeats=3) NAO estao la'. Eles existem,
individualmente, nos arquivos run_XXXX.json da mesma pasta
(results/<context>/<basis>/<dataset>/<experiment_name>/), cada um com os
campos `combo_id`, `repeat_idx`, `fold_idx` (ver pipeline/run_model.py).

Este script:
  1. Le summary.json de cada base para achar o combo_id VENCEDOR (a
     configuracao escolhida pelo tuning via F1 - "best_configuration").
  2. Le todos os run_*.json daquele diretorio, filtra pelos que pertencem
     ao combo vencedor, e indexa por (repeat_idx, fold_idx).
  3. Como todas as bases usam o mesmo master_seed (default 42, fixo no
     pipeline - ver pipeline/run_model.py::run_basis_experiment) para o
     mesmo dataset, (repeat_idx, fold_idx) identifica a MESMA particao de
     dados em todas as bases -> pareamento valido.
  4. Roda Friedman (teste global) e, se p<alpha, post-hoc Wilcoxon
     pareado com correcao de Holm + effect size (rank-biserial pareado).

IMPORTANTE - o que estas 30 execucoes SAO e o que NAO SAO:
  Sao as 30 avaliacoes de VALIDACAO CRUZADA (10 folds x 3 repeticoes) da
  configuracao de hiperparametros que venceu o tuning por F1. NAO sao 30
  execucoes independentes do modelo final treinado no conjunto de teste
  (o pipeline atual so' avalia o modelo final UMA vez no teste -
  test_results.json). Isso e' apropriado para comparar bases de forma
  pareada (segue a literatura de comparacao de classificadores via CV
  repetida, ex. Demsar 2006), mas a variabilidade capturada inclui
  variacao do conjunto de TREINO entre folds, nao so' aleatoriedade do
  modelo - deixe isso explicito na secao de limitacoes do relatorio.

Se um par base/dataset nao tiver run_*.json (por exemplo, se foram
apagados apos gerar o summary), o script AVISA explicitamente e nao
inventa nada - so' reporta que o teste pareado nao pode ser feito para
aquele caso.

Uso:
    python -m analysis.statistical_comparison \\
        --context classifier \\
        --datasets breast_cancer digits iris \\
        --bases shepard wendland_c2 laplacian inverse_multiquadratic radial \\
        --metrics accuracy balanced_accuracy f1 \\
        --experiment-name oficial_v1
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"
REPORTS_ROOT = Path(__file__).resolve().parent.parent / "reports"

ALPHA_DEFAULT = 0.05


class DataAvailabilityError(Exception):
    """Levantado quando os dados necessarios para o teste pareado nao
    estao disponiveis - a mensagem sempre explica exatamente o que falta
    e o que fazer para gerar."""


# --------------------------------------------------------------------------
# Carregamento dos dados pareados (run_*.json do combo vencedor)
# --------------------------------------------------------------------------

def _scope_dir(context: str, basis: str, dataset: str, experiment_name: str) -> Path:
    return RESULTS_ROOT / context / basis / dataset / experiment_name


def load_paired_fold_values(
    context: str, basis: str, dataset: str, experiment_name: str, metric_names: List[str]
) -> Tuple[Dict[Tuple[int, int], Dict[str, Optional[float]]], dict]:
    """Retorna ({(repeat_idx, fold_idx): {metric: valor}}, metadados)
    para o combo_id VENCEDOR (best_configuration do summary.json)."""
    scope_dir = _scope_dir(context, basis, dataset, experiment_name)
    summary_path = scope_dir / "summary.json"

    if not summary_path.exists():
        raise DataAvailabilityError(
            f"[{basis}/{dataset}] summary.json nao encontrado em {scope_dir}.\n"
            f"    Rode antes: python main.py --context {context} --basis {basis} --dataset {dataset} "
            f"--experiment-name {experiment_name} ..."
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best = summary.get("best_configuration")
    if not best:
        raise DataAvailabilityError(f"[{basis}/{dataset}] summary.json nao tem 'best_configuration'.")
    winning_combo_id = best["combo_id"]

    run_files = sorted(scope_dir.glob("run_*.json"))
    if not run_files:
        raise DataAvailabilityError(
            f"[{basis}/{dataset}] So' summary.json esta' disponivel em {scope_dir} (media/desvio "
            f"AGREGADOS) - nao ha' nenhum run_*.json. Os testes pareados (Friedman/Wilcoxon) exigem "
            f"os valores INDIVIDUAIS de cada fold/repeticao. Se os run_*.json foram apagados apos "
            f"gerar o summary, nao e' possivel fazer o teste pareado para essa base/dataset; so' e' "
            f"possivel reportar media e desvio-padrao descritivos (sem teste de significancia)."
        )

    paired: Dict[Tuple[int, int], Dict[str, Optional[float]]] = {}
    for rf in run_files:
        run = json.loads(rf.read_text(encoding="utf-8"))
        if run.get("combo_id") != winning_combo_id:
            continue
        if "repeat_idx" not in run or "fold_idx" not in run:
            raise DataAvailabilityError(
                f"[{basis}/{dataset}] {rf.name} nao tem os campos 'repeat_idx'/'fold_idx' - nao e' "
                f"possivel parear com as outras bases de forma confiavel (formato de run antigo?)."
            )
        key = (run["repeat_idx"], run["fold_idx"])
        paired[key] = {m: run["metrics"].get(m) for m in metric_names}

    if not paired:
        raise DataAvailabilityError(
            f"[{basis}/{dataset}] Nenhum run_*.json corresponde ao combo vencedor '{winning_combo_id}' - "
            f"os run_*.json existentes sao de outras combinacoes de hiperparametros."
        )

    return paired, {
        "winning_combo_id": winning_combo_id,
        "n_runs_found": len(paired),
        "hyperparameters": best["hyperparameters"],
    }


def build_paired_matrix(
    bases: List[str], dataset: str, metric: str, experiment_name: str, context: str = "classifier"
) -> Tuple[List[Tuple[int, int]], np.ndarray, Dict[str, dict]]:
    """Matriz pareada (n_folds x n_bases) para UMA metrica, usando a
    INTERSECAO das chaves (repeat_idx, fold_idx) disponiveis em todas as
    bases (com aviso se a intersecao for menor que o esperado - indica
    bases rodadas com seeds/CV diferentes entre si)."""
    per_basis_data, per_basis_meta, key_sets = {}, {}, []
    for basis in bases:
        paired, meta = load_paired_fold_values(context, basis, dataset, experiment_name, [metric])
        per_basis_data[basis] = paired
        per_basis_meta[basis] = meta
        key_sets.append(set(paired.keys()))

    common_keys = sorted(set.intersection(*key_sets)) if key_sets else []
    if not common_keys:
        raise DataAvailabilityError(
            f"[{dataset}/{metric}] Nenhuma chave (repeat_idx, fold_idx) em comum entre {bases}. "
            f"Provavelmente foram rodadas com master_seed ou n_splits/n_repeats diferentes entre si - "
            f"verifique se todas usaram os mesmos parametros de CV."
        )

    n_expected = max(len(s) for s in key_sets)
    if len(common_keys) < n_expected:
        print(
            f"  [aviso] {dataset}/{metric}: so' {len(common_keys)}/{n_expected} folds em comum entre "
            f"todas as bases - usando a intersecao (algumas bases tem runs extras/faltantes)."
        )

    matrix = np.array(
        [[per_basis_data[basis][key][metric] for basis in bases] for key in common_keys], dtype=float
    )
    return common_keys, matrix, per_basis_meta


# --------------------------------------------------------------------------
# Estatistica
# --------------------------------------------------------------------------

def holm_correction(pvalues: List[float]) -> List[float]:
    """Correcao de Holm (step-down, family-wise). Retorna p-values
    ajustados NA MESMA ORDEM da lista de entrada."""
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (n - rank) * pvalues[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def matched_pairs_rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Effect size para Wilcoxon signed-rank pareado:
    r = (soma_ranks_positivos - soma_ranks_negativos) / soma_total_ranks.
    r>0 => x tende a ser maior que y; |r| perto de 1 => diferenca
    consistente em quase todos os pares; |r| perto de 0 => diferenca
    inconsistente entre folds (mesmo que a media difira)."""
    d = x - y
    d_nonzero = d[d != 0]
    if len(d_nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(d_nonzero))
    w_plus = ranks[d_nonzero > 0].sum()
    w_minus = ranks[d_nonzero < 0].sum()
    total = ranks.sum()
    return float((w_plus - w_minus) / total) if total > 0 else 0.0


def kendalls_w(friedman_stat: float, n_subjects: int, n_treatments: int) -> float:
    """Coeficiente de concordancia de Kendall - effect size GLOBAL do
    Friedman (0 = nenhuma concordancia entre folds sobre o ranking das
    bases; 1 = concordancia perfeita)."""
    denom = n_subjects * (n_treatments - 1)
    return float(friedman_stat / denom) if denom > 0 else float("nan")


def analyze_dataset_metric(
    bases: List[str], dataset: str, metric: str, experiment_name: str,
    context: str = "classifier", alpha: float = ALPHA_DEFAULT,
) -> dict:
    fold_keys, matrix, meta = build_paired_matrix(bases, dataset, metric, experiment_name, context)
    n_folds, n_bases = matrix.shape

    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=1) if n_folds > 1 else np.zeros(n_bases)

    friedman_stat, friedman_p = friedmanchisquare(*[matrix[:, i] for i in range(n_bases)])
    w_kendall = kendalls_w(friedman_stat, n_folds, n_bases)

    result = {
        "dataset": dataset,
        "metric": metric,
        "n_folds_paired": n_folds,
        "bases": bases,
        "descriptive": {b: {"mean": float(means[i]), "std": float(stds[i])} for i, b in enumerate(bases)},
        "friedman": {
            "statistic": float(friedman_stat),
            "p_value": float(friedman_p),
            "significant": bool(friedman_p < alpha),
        },
        "kendalls_w": w_kendall,
        "posthoc": [],
    }

    if friedman_p < alpha:
        pairs = list(itertools.combinations(range(n_bases), 2))
        raw_pvalues, pair_stats = [], []
        for i, j in pairs:
            xi, xj = matrix[:, i], matrix[:, j]
            if np.allclose(xi, xj):
                stat, p = np.nan, 1.0
            else:
                try:
                    stat, p = wilcoxon(xi, xj, zero_method="wilcox")
                except ValueError:
                    # todas as diferencas sao zero, ou outro caso degenerado
                    stat, p = np.nan, 1.0
            raw_pvalues.append(p)
            pair_stats.append((i, j, stat))

        adjusted = holm_correction(raw_pvalues)

        for (i, j, stat), p_raw, p_adj in zip(pair_stats, raw_pvalues, adjusted):
            effect = matched_pairs_rank_biserial(matrix[:, i], matrix[:, j])
            result["posthoc"].append(
                {
                    "pair": [bases[i], bases[j]],
                    "wilcoxon_statistic": None if (isinstance(stat, float) and np.isnan(stat)) else float(stat),
                    "p_value": float(p_raw),
                    "p_value_holm": float(p_adj),
                    "significant_after_holm": bool(p_adj < alpha),
                    "effect_size_rank_biserial_r": effect,
                    "mean_diff": float(means[i] - means[j]),
                }
            )

    return result


# --------------------------------------------------------------------------
# Apresentacao
# --------------------------------------------------------------------------

def _print_result(res: dict) -> None:
    print(f"\n--- Metrica: {res['metric']}  (n_folds pareados = {res['n_folds_paired']}) ---")
    for b in res["bases"]:
        d = res["descriptive"][b]
        print(f"  {b:25s} media={d['mean']:.4f}  desvio={d['std']:.4f}")

    fr = res["friedman"]
    sig_txt = "SIM (p<0.05)" if fr["significant"] else "NAO"
    print(
        f"  Friedman: chi2={fr['statistic']:.4f}  p={fr['p_value']:.4g}  significativo={sig_txt}  "
        f"Kendall's W={res['kendalls_w']:.3f}"
    )

    if res["posthoc"]:
        print("  Post-hoc Wilcoxon pareado (p ajustado por Holm):")
        for ph in res["posthoc"]:
            flag = "*" if ph["significant_after_holm"] else " "
            print(
                f"    [{flag}] {ph['pair'][0]} vs {ph['pair'][1]}: "
                f"p_holm={ph['p_value_holm']:.4g}  r={ph['effect_size_rank_biserial_r']:+.3f}  "
                f"diff_medias={ph['mean_diff']:+.4f}"
            )
    elif fr["significant"]:
        print("  (post-hoc nao pode ser calculado)")
    else:
        print(
            "  Friedman NAO significativo -> post-hoc nao realizado. Isso significa AUSENCIA DE "
            "EVIDENCIA suficiente de diferenca entre as bases nesse dataset/metrica - nao e' prova "
            "de que elas sejam equivalentes."
        )


def _non_significant_pairs(res: dict) -> List[Tuple[str, str]]:
    """Pares SEM diferenca estatisticamente significativa (apos Holm).
    Se Friedman nao foi significativo, TODOS os pares contam como
    'sem evidencia de diferenca' (mas ver aviso em item 10 do pedido:
    isso nao e' prova de igualdade)."""
    bases = res["bases"]
    if not res["friedman"]["significant"]:
        return list(itertools.combinations(bases, 2))
    return [tuple(ph["pair"]) for ph in res["posthoc"] if not ph["significant_after_holm"]]


def build_summary_table(all_results: List[dict]) -> str:
    """Tabela compacta em Markdown, pronta para colar num trabalho
    cientifico (item 13 do pedido)."""
    lines = [
        "| Dataset | Métrica | " + " | ".join(f"{b} (média+/-dp)" for b in all_results[0]["bases"]) + " | Friedman chi2 | p-value | Kendall's W | Pares sig. (Holm) |",
        "|---" * (4 + len(all_results[0]["bases"])) + "|",
    ]
    for res in all_results:
        desc_cols = " | ".join(
            f"{res['descriptive'][b]['mean']:.4f}+/-{res['descriptive'][b]['std']:.4f}" for b in res["bases"]
        )
        fr = res["friedman"]
        sig_pairs = [ph["pair"] for ph in res["posthoc"] if ph["significant_after_holm"]]
        sig_txt = "; ".join(f"{a}!={b}" for a, b in sig_pairs) if sig_pairs else ("-" if fr["significant"] else "n/a (Friedman ns)")
        lines.append(
            f"| {res['dataset']} | {res['metric']} | {desc_cols} | {fr['statistic']:.3f} | "
            f"{fr['p_value']:.4g}{'*' if fr['significant'] else ''} | {res['kendalls_w']:.3f} | {sig_txt} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--context", type=str, default="classifier")
    parser.add_argument("--datasets", nargs="+", default=["breast_cancer", "digits", "iris"])
    parser.add_argument(
        "--bases", nargs="+",
        default=["shepard", "wendland_c2", "laplacian", "inverse_multiquadratic", "radial"],
    )
    parser.add_argument("--metrics", nargs="+", default=["accuracy", "balanced_accuracy", "f1"])
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    all_results: List[dict] = []
    for dataset in args.datasets:
        print(f"\n{'=' * 78}\nDATASET: {dataset}\n{'=' * 78}")
        for metric in args.metrics:
            try:
                res = analyze_dataset_metric(
                    args.bases, dataset, metric, args.experiment_name, args.context, args.alpha
                )
            except DataAvailabilityError as e:
                print(f"\n--- Metrica: {metric} ---")
                print(f"  [DADOS INSUFICIENTES] {e}")
                continue
            all_results.append(res)
            _print_result(res)

    if not all_results:
        print("\nNenhuma combinacao dataset/metrica pode ser analisada - ver mensagens acima.")
        return

    print(f"\n{'=' * 78}\nRESUMO - pares sem diferenca significativa (apos Holm)\n{'=' * 78}")
    for res in all_results:
        pairs = _non_significant_pairs(res)
        label = "TODOS os pares (Friedman ns - sem evidencia de diferenca)" if not res["friedman"]["significant"] else (
            "; ".join(f"{a}~={b}" for a, b in pairs) if pairs else "nenhum (todas as bases diferem entre si)"
        )
        print(f"  {res['dataset']:15s} {res['metric']:18s} -> {label}")

    print(f"\n{'=' * 78}\nTABELA FINAL (Markdown)\n{'=' * 78}\n")
    print(build_summary_table(all_results))

    out_path = Path(args.output) if args.output else (REPORTS_ROOT / "statistical_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[salvo] {out_path}")


if __name__ == "__main__":
    main()