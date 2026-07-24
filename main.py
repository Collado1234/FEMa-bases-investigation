"""
Ponto de entrada da linha de comando para o pipeline de treino, tuning e avaliação.

Uso:
    python main.py --experiment configs/experiments/fema_baseline.yaml
    python main.py --experiment configs/experiments/*.yaml   # roda vários arquivos (glob expandido pelo shell)
    python main.py --model fema --dataset fetal_health --experiment-name baseline

Este módulo lê as configurações base em `configs/base.yaml` e, em seguida,
aplica as definições do arquivo YAML de experimento para executar o pipeline.
Também suporta uma forma mais direta de execução, fornecendo `--model` e
`--dataset` para rodar um experimento com o nome informado.
"""

import argparse
from pathlib import Path

from pipeline.run_model import run_from_experiment_file, run_model


def main():
    """Parseia os argumentos da linha de comando e dispara a execução do pipeline."""
    parser = argparse.ArgumentParser(
        description="Pipeline de treino/tuning/avaliação do FEMa e baselines."
    )

    parser.add_argument(
        "--experiment",
        type=str,
        nargs="+",
        help="Caminho(s) para arquivo(s) YAML de experimento.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["fema", "logreg", "mlp"],
        help="Nome do modelo (alternativa a --experiment).",
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
        help="Nome do experimento usado quando --model e --dataset são fornecidos.",
    )

    args = parser.parse_args()

    if args.experiment:
        # Executa um ou mais experimentos a partir de arquivos YAML.
        for exp_path in args.experiment:
            run_from_experiment_file(Path(exp_path))
    elif args.model and args.dataset:
        # Executa um experimento rápido usando modelo + dataset.
        run_model(
            model_name=args.model,
            dataset=args.dataset,
            experiment_name=args.experiment_name,
        )
    else:
        parser.error("Forneca --experiment <arquivo.yaml> ou --model + --dataset.")


if __name__ == "__main__":
    main()