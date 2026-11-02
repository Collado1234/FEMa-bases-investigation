"""Confere TODOS os datasets do registry (sem rodar experimentos) e grava um manifesto.

    python scripts/check_datasets.py                 # todos
    python scripts/check_datasets.py zoo haberman    # so' alguns

Para cada dataset: arquivo existe? carrega pelo mesmo loader do pipeline? Reporta formato,
n de classes e a MENOR classe no treino (com CV de 10 folds, classe com < 10 exemplos vira
aviso/roc_auc None em alguns folds). Grava data/dataset_manifest.csv (com sha256 de cada CSV),
que PODE ser versionado (data/raw/ e' ignorado pelo git) e prova quais dados foram usados.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from datasets.loader import load_dataset  # noqa: E402
from datasets.registry import _REGISTRY, available_datasets  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> None:
    names = sys.argv[1:] or available_datasets()
    rows, ok = [], 0
    print(f"{'dataset':36} {'status':9} {'treino':>8} {'feat':>5} {'cls':>4} {'menor classe':>12}  obs")
    for name in names:
        spec = _REGISTRY[name]
        f = spec.csv_path
        row = {"dataset": name, "file": str(f.name) if f else f"sklearn:{spec.sklearn_loader}",
               "sha256_16": sha256(f) if f and f.exists() else "", "status": ""}
        if f and not f.exists():
            row["status"] = "FALTANDO"
            print(f"{name:36} FALTANDO  esperado: data/raw/{f.name}")
            rows.append(row)
            continue
        try:
            d = load_dataset(name, seed=42, val_ratio=0.15, test_ratio=0.15)
            counts = np.bincount(d.y_train)
            obs = []
            if counts.min() < 10:
                obs.append("classe <10 no treino (aviso/roc_auc None em alguns folds)")
            if not np.array_equal(np.unique(d.y_train), np.arange(len(counts))):
                obs.append("ROTULOS NAO 0..K-1 (loader corrompido?)")
            row.update(status="ok", n_train=len(d.y_train), n_features=d.X_train.shape[1],
                       n_classes=len(counts), min_class_train=int(counts.min()))
            print(f"{name:36} ok        {len(d.y_train):8d} {d.X_train.shape[1]:5d} {len(counts):4d} {counts.min():12d}  {'; '.join(obs)}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            row["status"] = f"ERRO: {type(e).__name__}: {e}"[:200]
            print(f"{name:36} ERRO      {type(e).__name__}: {str(e)[:110]}")
        rows.append(row)

    out = ROOT / "data" / "dataset_manifest.csv"
    out.parent.mkdir(exist_ok=True)
    cols = ["dataset", "file", "sha256_16", "status", "n_train", "n_features", "n_classes", "min_class_train"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n{ok}/{len(names)} datasets ok. Manifesto: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()