# FEMa-bases-investigation
Projeto de pesquisa para FAPESP que consiste na aplicação e avaliação de diferentes funções de base no FEMa (Finite Element Machine Classifier)
# FAPESP — Framework de experimentos (FEMa + baselines)

## Estrutura

```
core/            # NÃO TOCADO em redesenho — só 4 bugs de import corrigidos (ver abaixo)
src/
  utils.py       # logging, seed, hash determinístico (checkpoint)
  datasets.py    # DataSplit + carregamento de CSV/sintético + split train/val/test
  models.py      # FEMa (principal) + LogReg e MLP (baselines) — fábrica if/elif simples
  tuning.py      # grid_search / random_search (funções, sem strategy pattern)
  cross_validation.py  # K-Fold estratificado (com ou sem repetição)
  metrics.py     # métricas via sklearn.metrics (accuracy, f1, mcc, auc_roc, mae, r2, ...)
  persistence.py # grava cada run em JSON, monta checkpoint, agrega summary.json, escolhe o melhor
  curves.py      # ROC / precision-recall (cálculo manual)
  plots.py       # gráficos opcionais (confusion matrix, roc, actual vs predicted)
  config.py      # ExperimentConfig (dataclass) carregado de YAML
  pipeline.py     # run_experiment(config_path) — orquestra tudo
configs/          # exemplos de config YAML (fema.yaml, logreg.yaml, mlp.yaml, smoke_test.yaml)
tests/            # smoke test único do pipeline completo
run_experiment.py # CLI: python3 run_experiment.py configs/fema.yaml
```

Tudo fora de `core/` foi consolidado de ~40 arquivos/16 pastas (registries,
ABCs, plugin systems, um módulo por "fase" do pipeline) para 11 arquivos
sem abstrações que não tinham uso real. Trocar de modelo/dataset/estratégia
de tuning continua sendo só mudar um nome na config YAML.

1. `core/models/base_model.py`, `fema_classifier.py`, `fema_regressor.py`:
   importavam `..algebra...` — a pasta correta é `core/math`.
2. `fema_classifier.py`: `from models.base_model import FEMaBaseModel`
   (import absoluto errado) → `from .base_model import FEMaBaseModel`.
3. `core/__init__.py` e `core/math/basis/__init__.py`: importavam nomes
   inexistentes (`SheppardBasis` com dois "p", `BaseModel` em vez de
   `BaseBasis`, `.rbf_multiquadratic` em vez de `.multiquadratic`).
4. `euclidean_distance.py` / `manhattan_distance.py`: a condição de
   broadcasting só vetorizava quando `x1` era 1D e `x2` 2D — mas
   `BruteForceSearch.query()` chama `compute(X_train, sample)` (ordem
   invertida), fazendo o cálculo devolver um escalar único em vez do vetor
   de distâncias. Corrigido para vetorizar nos dois sentidos.

Nenhuma outra coisa em `core/` foi alterada — a lógica matemática (Shepard,
Radial, etc.) é exatamente a mesma de antes.

## Rodando

```bash
pip install -r requirements.txt  # pyyaml, scikit-learn, pandas, matplotlib, joblib
python3 run_experiment.py configs/fema.yaml
python3 -m pytest tests/ -v
```

Resultados vão para `results/<model>/<experiment_name>/`: um `run_XXXX.json`
por combinação×fold×repetição, `summary.json` com o ranking e a melhor
config, e `test_results.json` com a avaliação final no conjunto de teste.
