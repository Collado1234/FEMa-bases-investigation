"""Lancador do experimento v3 (rodar sem supervisao).

Exemplo:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    nohup python -u scripts/run_v3.py --tier small medium --seeds 42 43 44 45 46 > logs/v3.out 2>&1 &

- Seed no laco EXTERNO: se o processo cair, voce ja tem conjuntos completos de seeds.
- Idempotente/retomavel (checkpoint por combo/fold; config_hash impede misturar configs).
- Uma falha em um dataset NAO derruba o resto: e' registrada e o laco segue.
- Paralelismo: um processo por dataset (--datasets X). NUNCA dois processos no mesmo escopo.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.run_model import run_all_bases, run_baseline_experiment  # noqa: E402
from utils.logging_config import get_logger  # noqa: E402

logger = get_logger("scripts.run_v3")

TIERS = {
    "small": ["iris", "wine", "seeds", "vertebra_column", "zoo", "haberman", "blood_transfusion_service_center",
              "monks_problems_1", "monks_problems_2", "climate_model_simulation_crashes", "tic_tac_toe",
              "diabetes", "breast_cancer", "fetal_health"],
    "medium": ["segment", "page_blocks", "spambase", "digits", "mfeat_zernike", "mfeat_karhunen",
               "mfeat_fourier", "mfeat_factors", "mfeat_pixel"],
    "large": ["magic_telescope", "letter", "isolet", "musk", "gina_agnostic", "bng_glass"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", nargs="+", choices=list(TIERS), default=["small"])
    ap.add_argument("--datasets", nargs="+", help="sobrescreve --tier (ex.: para 1 processo por dataset)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--name", default="v3", help="prefixo do experiment_name (vira <name>_s<seed>)")
    ap.add_argument("--ranking-metric", default="f1")
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args()

    datasets = args.datasets or [d for t in args.tier for d in TIERS[t]]
    common = dict(
        cv_strategy="repeated_stratified_kfold", n_splits=args.n_splits, n_repeats=args.n_repeats,
        tuning_strategy="random_search", tuning_n_iter=args.n_iter, ranking_metric=args.ranking_metric,
    )
    failures = []
    for seed in args.seeds:
        exp = f"{args.name}_s{seed}"
        for ds in datasets:
            try:
                logger.info(f"##### {exp} | {ds} #####")
                run_all_bases("classifier", ds, experiment_name=exp, master_seed=seed, **common)
                if not args.no_baselines:
                    for model in ("knn", "logreg"):
                        run_baseline_experiment(model, ds, experiment_name=exp, master_seed=seed, **common)
            except Exception:
                failures.append((exp, ds))
                logger.error(f"FALHA em {exp}/{ds}:\n{traceback.format_exc()}")
    logger.info(f"FIM. Falhas: {failures or 'nenhuma'}")


if __name__ == "__main__":
    main()
