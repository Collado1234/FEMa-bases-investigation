"""
Ponto de entrada da linha de comando para o pipeline de treino, tuning, avaliação e comparação de bases.

Uso:
    # Comparação de bases do FEMa (objeto de estudo do projeto)
    python main.py --experiment configs/experiments/fema_baseline.yaml
    python main.py --context classifier --all-bases --dataset fetal_health
    python main.py --context classifier --basis shepard --dataset fetal_health

    # Aciona a comparação entre bases já rodadas (CSV + JSON + plots)
    python main.py --compare --context classifier --dataset fetal_health

    # Roda todas as bases e já compara em seguida, num único comando
    python main.py --context classifier --all-bases --dataset fetal_health --compare

    # Baseline externo (referência metodológica, fora da comparação de bases)
    python main.py --baseline-model knn --dataset fetal_health
    python main.py --experiment configs/experiments/logreg_baseline.yaml

Este módulo lê as configurações base em `configs/base.yaml` e, em seguida,
aplica as definições do arquivo YAML de experimento para executar o
pipeline. Também suporta uma forma mais direta de execução via flags de
linha de comando, sem precisar de um arquivo YAML.
"""

import argparse
from pathlib import Path

from pipeline.run_model import run_all_bases, run_baseline_experiment, run_basis_experiment, run_from_experiment_file
from reporting.compare_bases import run_full_comparison


def main():
    """Parseia os argumentos da linha de comando e dispara a execução do pipeline."""
    parser = argparse.ArgumentParser(
        description="Pipeline de comparação de bases do FEMa e baselines externos (logreg, knn)."
    )

    parser.add_argument(
        "--experiment",
        type=str,
        nargs="+",
        help="Caminho(s) para arquivo(s) YAML de experimento.",
    )

    parser.add_argument(
        "--context",
        type=str,
        choices=["classifier", "regressor"],
        help="Contexto de execução do FEMa (usar com --basis ou --all-bases).",
    )
    parser.add_argument(
        "--basis",
        type=str,
        help="Nome da base de interpolação do FEMa a rodar (usar com --context e --dataset).",
    )
    parser.add_argument(
        "--all-bases",
        action="store_true",
        help="Roda TODAS as bases registradas em core.Basis.available() (usar com --context e --dataset).",
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        choices=["logreg", "knn"],
        help="Nome do baseline externo a rodar (alternativa a --context/--basis).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["fetal_health", "iris", "classification_data", "synthetic_demo"],
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="baseline",
        help="Nome do experimento (usado quando --context/--basis, --baseline-model ou --compare são fornecidos).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Aciona a comparacao entre bases (reporting/compare_bases.py::run_full_comparison) para "
            "--context/--dataset/--experiment-name. NAO roda experimentos por si so' - le' resultados "
            "ja' persistidos em results/<context>/<basis>/<dataset>/<experiment_name>/ e gera "
            "tabela (CSV), JSON consolidado e plots em --output-dir. Pode ser combinado com "
            "--all-bases (roda todas as bases e, em seguida, compara, num unico comando)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Diretorio de saida da comparacao (--compare). Default: reports/basis_comparison/<context>/<dataset>.",
    )

    args = parser.parse_args()

    if args.experiment:
        for exp_path in args.experiment:
            run_from_experiment_file(Path(exp_path))
    elif args.context and args.all_bases and args.dataset:
        run_all_bases(context=args.context, dataset=args.dataset, experiment_name=args.experiment_name)
    elif args.context and args.basis and args.dataset:
        run_basis_experiment(
            context=args.context, basis=args.basis, dataset=args.dataset, experiment_name=args.experiment_name
        )
    elif args.baseline_model and args.dataset:
        run_baseline_experiment(
            model_name=args.baseline_model, dataset=args.dataset, experiment_name=args.experiment_name
        )
    elif not args.compare:
        parser.error(
            "Forneca --experiment <arquivo.yaml>, ou --context + (--basis|--all-bases) + --dataset, "
            "ou --baseline-model + --dataset, ou --compare + --context + --dataset."
        )

    if args.compare:
        if not (args.context and args.dataset):
            parser.error("--compare precisa de --context e --dataset.")
        output_dir = args.output_dir or f"reports/basis_comparison/{args.context}/{args.dataset}"
        result = run_full_comparison(
            context=args.context, dataset=args.dataset, experiment_name=args.experiment_name, output_dir=output_dir
        )
        print(f"Comparacao gravada em: {result['csv_path']}, {result['json_path']}")
        print(
            f"{len(result['bar_plot_paths'])} graficos de barra e {len(result['curve_plot_paths'])} "
            f"curvas gerados em {output_dir}/plots/"
        )


if __name__ == "__main__":
    main()