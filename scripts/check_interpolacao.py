"""Verifica a PROPRIEDADE DE INTERPOLACAO do FEMa nos pontos de treino.

Pergunta: para x_i do treino, o classificador devolve o proprio rotulo y_i?
Formalmente, com p(x) = vetor de probabilidades interpoladas e e_{y_i} o one-hot do rotulo,

    erro_i = || p(x_i) - e_{y_i} ||_inf = 1 - p(x_i)[y_i]   (as probabilidades somam 1)

    interpola em x_i  <=>  erro_i <= tol          (tol padrao: 1e-3, i.e. p(x_i)[y_i] >= 0.999)

Reporta tambem a versao FRACA (argmax p(x_i) == y_i, i.e. acuracia de ressubstituicao).

Configuracao: usa a configuracao VENCEDORA de cada (base, dataset, seed) em
results/<context>/<base>/<dataset>/<prefix>_s<seed>/summary.json
(best_configuration.hyperparameters) e reproduz o split/escala do pipeline
(config.master_seed/val_ratio/test_ratio gravados no summary). O modelo e' ajustado
no X_train inteiro, exatamente como o retreino final do pipeline.

Custo baixo: so' consulta uma AMOSTRA estratificada dos pontos de treino. O tamanho segue a
formula de proporcao com correcao de populacao finita (margem `--margin`, 95% de confianca,
p=0.5 pior caso), limitado por `--max-sample`. Com amostra de n pontos e ZERO falhas, o limite
superior (95%) da taxa de falha e' ~3/n (regra dos tres), reportado na coluna `falha_max_95`.

Pontos com features IDENTICAS e rotulos DIFERENTES no treino nao podem ser interpolados por
nenhuma funcao; sao excluidos da populacao e contados em `n_conflitos` (informativo).

Uso:
    python scripts/check_interpolacao.py --results results --prefix v3
    python scripts/check_interpolacao.py --datasets iris wine --bases shepard wendland_c2 --tol 1e-6   # exige interpolacao exata
Saida: tabela no terminal + CSV (--out, padrao reports/interpolacao/interpolacao.csv).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.loader import load_dataset  # noqa: E402
from models.fema import FEMaPlugin  # noqa: E402

SEED_RE = re.compile(r"_s(\d+)$")
Z95 = 1.959964


def sample_size(population: int, margin: float, max_sample: int) -> int:
    """n para estimar uma proporcao (p=0.5) com margem `margin` a 95%, com correcao finita."""
    n0 = (Z95 ** 2) * 0.25 / margin ** 2
    n = n0 / (1 + (n0 - 1) / population)
    return int(min(population, max_sample, math.ceil(n)))


def stratified_sample(y: np.ndarray, idx_pool: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Amostra estratificada por classe (proporcional, minimo 1 por classe) dentro de idx_pool."""
    if n >= len(idx_pool):
        return idx_pool
    classes, counts = np.unique(y[idx_pool], return_counts=True)
    quota = np.maximum(1, np.round(n * counts / counts.sum()).astype(int))
    chosen = []
    for c, q in zip(classes, quota):
        members = idx_pool[y[idx_pool] == c]
        chosen.append(rng.choice(members, size=min(q, len(members)), replace=False))
    return np.concatenate(chosen)


def conflicting_mask(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """True nas linhas cujo vetor de features aparece com mais de um rotulo distinto."""
    _, inv = np.unique(np.round(X, 12), axis=0, return_inverse=True)
    inv = inv.ravel()
    n_labels = pd.Series(y).groupby(inv).transform("nunique").to_numpy()
    return n_labels > 1


def wilson_lower(successes: int, n: int) -> float:
    if n == 0:
        return float("nan")
    p = successes / n
    d = 1 + Z95 ** 2 / n
    c = p + Z95 ** 2 / (2 * n)
    m = Z95 * math.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n * n))
    return (c - m) / d


def discover(results: Path, context: str, prefix: str, datasets, bases, seeds):
    """Gera (base, dataset, seed, summary_path) das configuracoes vencedoras existentes."""
    for sf in sorted((results / context).glob("*/*/*/summary.json")):
        exp = sf.parent.name
        m = SEED_RE.search(exp)
        if not m or not exp.startswith(prefix):
            continue
        base, ds, seed = sf.parent.parent.parent.name, sf.parent.parent.name, int(m.group(1))
        if (datasets and ds not in datasets) or (bases and base not in bases) or (seeds and seed not in seeds):
            continue
        yield base, ds, seed, sf


def check_one(base, ds, seed, summary_path: Path, context, data_cache, tol, margin, max_sample, rng_seed):
    s = json.load(open(summary_path, encoding="utf-8"))
    best = s.get("best_configuration")
    if not best:
        return None
    params = best["hyperparameters"]
    cfg = s.get("config", {})
    master_seed = cfg.get("master_seed", seed)
    key = (ds, master_seed)
    if key not in data_cache:
        data_cache.clear()           # mantem so' o dataset corrente (memoria)
        d = load_dataset(ds, seed=master_seed, val_ratio=cfg.get("val_ratio", 0.15), test_ratio=cfg.get("test_ratio", 0.15))
        X, y = np.asarray(d.X_train, float), np.asarray(d.y_train)
        conf = conflicting_mask(X, y)
        data_cache[key] = (X, y, conf)
    X, y, conf = data_cache[key]

    rng = np.random.default_rng([rng_seed, master_seed])
    pool = np.flatnonzero(~conf)
    n = sample_size(len(pool), margin, max_sample)
    idx = stratified_sample(y, pool, n, rng)

    plugin = FEMaPlugin(context, base)
    est = plugin.create_model(params, random_state=master_seed)
    est.fit(X, y)
    probs = est.predict_proba(X[idx])                       # (n, n_classes)
    classes = est._model.classes_
    col = np.searchsorted(classes, y[idx])
    p_true = probs[np.arange(len(idx)), col]
    err = 1.0 - p_true
    ok = err <= tol
    n_ok = int(ok.sum())
    n_s = len(idx)
    return dict(
        dataset=ds, base=base, seed=seed, k=params.get("k"),
        parametros={k: v for k, v in params.items() if k != "k"},
        n_treino=len(X), n_amostra=n_s, n_conflitos=int(conf.sum()),
        frac_interpola=n_ok / n_s, frac_interpola_ic95_inf=wilson_lower(n_ok, n_s),
        falha_max_95=(3.0 / n_s if n_ok == n_s else float("nan")),
        erro_medio=float(err.mean()), erro_max=float(err.max()), p_classe_media=float(p_true.mean()),
        acc_ressubstituicao=float((probs.argmax(1) == col).mean()),
        fallback_uniforme=est.diagnostics().get("frac_uniform_fallback", float("nan")),
        interpola=("SIM" if n_ok == n_s else ("PARCIAL" if n_ok > 0 else "NAO")),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--prefix", default="v3")
    ap.add_argument("--context", default="classifier")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--bases", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--tol", type=float, default=1e-3, help="tolerancia de |p(x_i)[y_i] - 1| (padrao 1e-3)")
    ap.add_argument("--margin", type=float, default=0.03, help="margem de erro da proporcao amostral (95%%)")
    ap.add_argument("--max-sample", type=int, default=1000, help="teto de pontos consultados por (base,dataset,seed)")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--out", default="reports/interpolacao/interpolacao.csv")
    a = ap.parse_args()

    rows, cache = [], {}
    items = list(discover(Path(a.results), a.context, a.prefix, a.datasets, a.bases, a.seeds))
    if not items:
        raise SystemExit(f"Nenhum summary.json com prefixo '{a.prefix}' em {a.results}/{a.context}.")
    # agrupa por (dataset, seed) para reaproveitar o dataset carregado
    items.sort(key=lambda t: (t[1], t[2], t[0]))
    for base, ds, seed, sf in items:
        try:
            r = check_one(base, ds, seed, sf, a.context, cache, a.tol, a.margin, a.max_sample, a.rng_seed)
        except Exception as e:                                   # uma falha nao derruba o resto
            print(f"[FALHA] {base}/{ds}/s{seed}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if r:
            rows.append(r)
            print(f"{ds:28s} s{seed:<3d} {base:22s} n={r['n_amostra']:5d}/{r['n_treino']:<6d} "
                  f"interpola={r['frac_interpola']:6.1%}  erro_max={r['erro_max']:.2e}  {r['interpola']}", flush=True)

    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.assign(parametros=df["parametros"].astype(str)).to_csv(out, index=False)

    print(f"\n=== Resumo por base (tol={a.tol:g}) ===")
    summ = df.groupby("base").agg(
        casos=("dataset", "size"), sim=("interpola", lambda s: int((s == "SIM").sum())),
        frac_interpola_media=("frac_interpola", "mean"), erro_max=("erro_max", "max"),
        acc_ressubst=("acc_ressubstituicao", "mean")).reset_index()
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()