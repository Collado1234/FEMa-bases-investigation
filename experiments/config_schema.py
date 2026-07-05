"""Builder + validação mínima da config de experimento carregada do YAML."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    model: str
    dataset: str
    experiment_name: str
    metrics: List[str]
    cv_strategy: str
    n_splits: int
    n_repeats: int
    tuning_strategy: str
    tuning_n_iter: int
    seed: int
    ranking_metric: str
    higher_is_better: bool
    include_val_in_final_train: bool
    fixed_hyperparameters: Optional[Dict[str, Any]]
    output_dir: str
    raw: Dict[str, Any] = field(repr=False)


_REQUIRED_FIELDS = ["model", "dataset"]


def load_experiment_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    missing = [f for f in _REQUIRED_FIELDS if f not in raw]
    if missing:
        raise ValueError(f"Config '{path}' não define campos obrigatórios: {missing}")

    return ExperimentConfig(
        model=raw["model"],
        dataset=raw["dataset"],
        experiment_name=raw.get("experiment_name", raw.get("config_name", "baseline")),
        metrics=raw.get("metrics", ["accuracy", "f1", "balanced_accuracy", "mcc"]),
        cv_strategy=raw.get("cv_strategy", "repeated_stratified_kfold"),
        n_splits=int(raw.get("n_splits", 5)),
        n_repeats=int(raw.get("n_repeats", 3)),
        tuning_strategy=raw.get("tuning_strategy", "random_search"),
        tuning_n_iter=int(raw.get("tuning_n_iter", 20)),
        seed=int(raw.get("seed", 42)),
        ranking_metric=raw.get("ranking_metric", "f1"),
        higher_is_better=bool(raw.get("higher_is_better", True)),
        include_val_in_final_train=bool(raw.get("include_val_in_final_train", False)),
        fixed_hyperparameters=raw.get("fixed_hyperparameters"),
        output_dir=raw.get("output_dir", "results"),
        raw=raw,
    )
