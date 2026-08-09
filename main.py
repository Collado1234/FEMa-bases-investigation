"""
Ponto de entrada da linha de comando para o pipeline de treino, tuning, avaliação e comparação de bases.

Uso:
    # Comparação de bases do FEMa (objeto de estudo do projeto)
    python main.py --experiment configs/experiments/fema_baseline.yaml
    python main.py --context classifier --all-bases --dataset fetal_health
    python main.py --context classifier --basis shepard --dataset fetal_health

    # Protocolo oficial (F1 como metrica de ranking, random search, 10 folds x 3 repeticoes)
    python main.py --context classifier --all-bases --dataset digits \
        --ranking-metric f1 --tuning-strategy random_search --n-splits 10 --n-repeats 3

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
        choices=[
            "fetal_health", "iris", "classification_data",
            "digits_5class", "digits", "breast_cancer", "wine",
            "synthetic_demo",
        ],
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
    parser.add_argument(
        "--ranking-metric",
        type=str,
        choices=["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "mcc",
                 "mae", "mse", "rmse", "r2", "mape"],
        default=None,
        help=(
            "Metrica usada para escolher a MELHOR combinacao de hiperparametros apos o CV "
            "(persistence/summary_builder.py::build_summary). Default do pipeline: 'f1' para "
            "classificacao, 'rmse' para regressao - ou seja, F1 ja' e' o default e nao precisa "
            "ser passado explicitamente, mas fica disponivel para deixar isso explicito no comando "
            "ou trocar para outra metrica sem editar codigo."
        ),
    )
    parser.add_argument(
        "--tuning-strategy",
        type=str,
        choices=["grid_search", "random_search"],
        default=None,
        help="Estrategia de busca de hiperparametros (tuning/param_space.py). Default do pipeline: 'random_search'.",
    )
    parser.add_argument(
        "--tuning-n-iter",
        type=int,
        default=None,
        help="Numero de combinacoes avaliadas quando --tuning-strategy=random_search (ignorado em grid_search, que e' exaustivo). Default do pipeline: 20.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=None,
        help="Numero de folds do CV (tuning/cv_strategy.py). Default do pipeline: 5.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Numero de repeticoes do CV. Default do pipeline: 3.",
    )

    args = parser.parse_args()

    # Kwargs opcionais de tuning/CV/ranking, repassados so' quando informados
    # explicitamente no comando - se omitidos, prevalecem os defaults do
    # proprio pipeline (pipeline/run_model.py::run_basis_experiment), que
    # ja' usa 'f1' como ranking_metric padrao para classificacao.
    tuning_kwargs = {}
    if args.ranking_metric is not None:
        tuning_kwargs["ranking_metric"] = args.ranking_metric
    if args.tuning_strategy is not None:
        tuning_kwargs["tuning_strategy"] = args.tuning_strategy
    if args.tuning_n_iter is not None:
        tuning_kwargs["tuning_n_iter"] = args.tuning_n_iter
    if args.n_splits is not None:
        tuning_kwargs["n_splits"] = args.n_splits
    if args.n_repeats is not None:
        tuning_kwargs["n_repeats"] = args.n_repeats

    if args.experiment:
        for exp_path in args.experiment:
            run_from_experiment_file(Path(exp_path))
    elif args.context and args.all_bases and args.dataset:
        run_all_bases(
            context=args.context, dataset=args.dataset, experiment_name=args.experiment_name, **tuning_kwargs
        )
    elif args.context and args.basis and args.dataset:
        run_basis_experiment(
            context=args.context, basis=args.basis, dataset=args.dataset, experiment_name=args.experiment_name,
            **tuning_kwargs,
        )
    elif args.baseline_model and args.dataset:
        run_baseline_experiment(
            model_name=args.baseline_model, dataset=args.dataset, experiment_name=args.experiment_name,
            **tuning_kwargs,
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