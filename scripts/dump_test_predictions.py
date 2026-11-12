"""Re-treina a configuracao VENCEDORA de cada (base, dataset, seed) e salva as probabilidades no HOLDOUT de
teste, necessarias para o teste de DeLong (results/ so' guarda metricas escalares).

  python scripts/dump_test_predictions.py --prefix v3 --out preds/

Reproducibilidade: o retreino usa a mesma seed de pipeline/run_model.py (derive_seed(master_seed,'final_fit',nome));
o script confere roc_auc recalculado contra o de test_results.json e AVISA se divergir.
Custo: 1 fit por (base, dataset, seed) - ordens de grandeza menor que o tuning."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.loader import load_dataset  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from models.fema import FEMaPlugin  # noqa: E402
from persistence.run_writer import RESULTS_DIR  # noqa: E402
from training.final_fit import fit_final_model  # noqa: E402
from utils.logging_config import get_logger  # noqa: E402

logger = get_logger("scripts.dump_test_predictions")
SEED_RE = re.compile(r"_s(\d+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="v3")
    ap.add_argument("--out", default="preds")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    scopes = {}
    for tr in (RESULTS_DIR / "classifier").glob("*/*/*/test_results.json"):
        basis, ds, exp = tr.parts[-4], tr.parts[-3], tr.parts[-2]
        m = SEED_RE.search(exp)
        if not (exp.startswith(args.prefix) and m) or (args.datasets and ds not in args.datasets):
            continue
        scopes.setdefault((ds, int(m.group(1))), []).append((basis, tr))

    for (ds, seed), items in sorted(scopes.items()):
        target = out / f"{ds}_s{seed}.npz"
        if target.exists():
            continue
        data = load_dataset(ds, seed=seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
        arrays = {"y_true": np.asarray(data.y_test)}
        for basis, tr in items:
            saved = json.load(open(tr, encoding="utf-8"))
            plugin = FEMaPlugin(context="classifier", basis=basis)
            est = fit_final_model(plugin, saved["best_hyperparameters"], data.X_train, data.y_train,
                                  master_seed=seed, X_val=data.X_val, y_val=data.y_val)
            proba = np.asarray(plugin.predict_proba(est, data.X_test))
            arrays[f"proba__{basis}"] = proba
            ref = (saved.get("metrics") or {}).get("roc_auc")
            if ref is not None:      # confere reproducibilidade do retreino
                yt = arrays["y_true"]
                auc = (roc_auc_score(yt, proba[:, 1]) if proba.shape[1] == 2
                       else roc_auc_score(yt, proba, multi_class="ovr", average="macro"))
                if abs(auc - ref) > 1e-9:
                    logger.warning(f"{ds} s{seed} {basis}: roc_auc recalculado {auc:.6f} != salvo {ref:.6f} "
                                   "(retreino NAO reproduziu o holdout; nao use estas predicoes)")
            logger.info(f"{ds} s{seed} {basis}: ok")
        np.savez_compressed(target, **arrays)
        logger.info(f"salvo {target}")


if __name__ == "__main__":
    main()
