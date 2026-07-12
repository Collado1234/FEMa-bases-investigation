"""Config de experimento: um dataclass simples carregado de um YAML."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    model: str
    dataset: str
    experiment_name: str = "baseline"
    metrics: List[str] = None
    cv_strategy: str = "repeated_stratified_kfold"
    n_splits: int = 5
    n_repeats: int = 3
    tuning_strategy: str = "random_search"
    tuning_n_iter: int = 20
    seed: int = 42
    ranking_metric: str = "f1"
    higher_is_better: bool = True
    fixed_hyperparameters: Optional[Dict[str, Any]] = None
    output_dir: str = "results"

    def __post_init__(self):
        if self.metrics is None:
            object.__setattr__(self, "metrics", ["accuracy", "f1", "balanced_accuracy", "mcc"])


def load_experiment_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    for field in ("model", "dataset"):
        if field not in raw:
            raise ValueError(f"Config '{path}' não define o campo obrigatório '{field}'.")
    return ExperimentConfig(**raw)


def override(config: ExperimentConfig, **overrides) -> ExperimentConfig:
    return replace(config, **overrides)
