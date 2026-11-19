"""Generate three paper-style figures for every registered FEMa basis.

Outputs (PNG, 300 dpi):
  figs/<basis>/probability_functions.png  -- two-class 1D probabilities
  figs/<basis>/working_mechanism.png      -- training set + k=1,3,5 regions
  figs/<basis>/probability_map.png        -- certainty grayscale + class map

All predictions use the project's actual Basis factory, BaseBasis.compute_weights
and FEMaClassifier. No basis formula is duplicated in this file.
Run from the project root:
    python -m reporting.generate_basis_figures
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from core.math.basis import Basis, BasisParameters
from core.math.distances.euclidean_distance import EuclideanDistance
from core.math.neighboor_search.brute_force import BruteForceSearch
from core.models.fema_classifier import FEMaClassifier
from reporting.plot_basis_analysis import BASIS_CONFIG, make_training_set

# Consistent palette close to the article figures.
CLASS_COLORS = ["red", "lime", "blue"]
CLASS_RGB = np.array([[1.0, 0.0, 0.0], [0.0, 0.82, 0.0], [0.0, 0.0, 1.0]])
DECISION_CMAP = ListedColormap(CLASS_COLORS)
DPI = 300


def representative_parameters(name: str) -> BasisParameters:
    """Use the middle curated value already defined by plot_basis_analysis."""
    cfg = BASIS_CONFIG.get(name)
    values: dict = dict((cfg or {}).get("fixed", {}))
    if cfg:
        p_name = cfg["param"]
        p_values = cfg.get("values", [])
        if p_values:
            values[p_name] = p_values[len(p_values) // 2]
    # The rational-quadratic sweep exposes l while alpha remains fixed;
    # compact-support bases expose h. These are already represented in cfg.
    allowed = set(BasisParameters.__dataclass_fields__)
    clean = {k: v for k, v in values.items() if k in allowed}
    return BasisParameters(**clean)


def make_classifier(name: str, X: np.ndarray, y: np.ndarray):
    search = BruteForceSearch(metric=EuclideanDistance())
    basis = Basis.get(name, search=search)
    model = FEMaClassifier(basis=basis, search=search)
    model.fit(np.asarray(X, dtype=float), np.asarray(y))
    return model


def _save(fig, path: Path, dpi: int = DPI):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path



def predict_grid_vectorized(model: FEMaClassifier, grid: np.ndarray, k: int,
                            params: BasisParameters, batch_size: int = 8192):
    """Batch equivalent of FEMaClassifier.predict for 2D plotting grids.

    It calls the SAME basis object's _log_phi/evaluate implementation and applies
    the row-wise normalization specified by BaseBasis.compute_weights. The batch
    path avoids millions of Python-level predict calls while preserving the model
    equations and per-query neighborhood semantics.
    """
    X_train = model.search.X_train
    y_train = np.asarray(model.classes_)
    n_train = len(X_train)
    k_eff = n_train if (k is None or k <= 0 or k >= n_train) else int(k)
    n_classes = len(model.classes_)
    # probability_classes has one row per class and one column per training sample
    onehot = model.probability_classes.T
    predictions = np.empty(len(grid), dtype=int)
    all_probs = np.empty((len(grid), n_classes), dtype=float)
    basis = model.basis

    for start in range(0, len(grid), batch_size):
        stop = min(start + batch_size, len(grid))
        query = grid[start:stop]
        distances = np.linalg.norm(query[:, None, :] - X_train[None, :, :], axis=2)
        # Stable sorting makes equal-distance behavior deterministic.
        indices = np.argsort(distances, axis=1, kind="stable")[:, :k_eff]
        dists = np.take_along_axis(distances, indices, axis=1)

        log_phi = basis._log_phi(dists, params)
        if log_phi is not None:
            log_phi = np.asarray(log_phi, dtype=float)
            shifted = np.exp(log_phi - np.max(log_phi, axis=1, keepdims=True))
            weights = shifted / shifted.sum(axis=1, keepdims=True)
        else:
            phi = np.asarray(basis.evaluate(dists, params), dtype=float)
            totals = phi.sum(axis=1, keepdims=True)
            weights = np.divide(phi, totals, out=np.zeros_like(phi),
                                where=~np.isclose(totals, 0.0))
            zero_rows = np.isclose(totals[:, 0], 0.0)
            if np.any(zero_rows):
                weights[zero_rows] = 1.0 / k_eff

        neighbor_labels = onehot[indices]
        probs = np.einsum("qk,qkc->qc", weights, neighbor_labels, optimize=True)
        all_probs[start:stop] = probs
        predictions[start:stop] = np.argmax(probs, axis=1)

    return predictions, all_probs


def plot_probability_functions(name: str, params: BasisParameters, out_dir: Path) -> Path:
    """Two-class 1D probability curves, in the spirit of article Fig. 3."""
    # Four fixed support samples, with alternating class labels.
    X = np.array([[-2.0], [-1.0], [0.0], [1.0]])
    y = np.array([0, 1, 0, 1])
    model = make_classifier(name, X, y)
    x = np.linspace(-3.0, 3.0, 601).reshape(-1, 1)
    _, probs = model.predict(x, k=0, params=params)

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    ax.plot(x[:, 0], probs[:, 0], color="red", lw=1.5, label="Classe 1")
    ax.plot(x[:, 0], probs[:, 1], color="black", lw=1.5, label="Classe 2")
    _, sample_probs = model.predict(X, k=0, params=params)
    ax.scatter(X[y == 0, 0], sample_probs[y == 0, 0], color="red", s=22, zorder=3)
    ax.scatter(X[y == 1, 0], sample_probs[y == 1, 1], color="black", s=22, zorder=3)
    ax.set(xlim=(-3, 3), ylim=(-0.03, 1.03), xlabel="x", ylabel="Probabilidade",
           title=f"Funções de probabilidade — {name}")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return _save(fig, out_dir / "probability_functions.png")


def _grid(resolution: int):
    xs = np.linspace(0.0, 1.0, resolution)
    ys = np.linspace(0.0, 1.0, resolution)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack((xx.ravel(), yy.ravel()))


def _scatter_training(ax, X, y, black=False):
    if black:
        ax.scatter(X[:, 0], X[:, 1], c="black", s=9, linewidths=0, zorder=3)
    else:
        for cls, color in enumerate(CLASS_COLORS):
            pts = X[y == cls]
            ax.scatter(pts[:, 0], pts[:, 1], c=color, s=20,
                       edgecolors="white", linewidths=0.35, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def plot_working_mechanism(name: str, params: BasisParameters, X, y,
                           out_dir: Path, resolution: int = 220) -> Path:
    """Training samples and three decision maps (k=1,3,5), article Fig. 4 style."""
    grid = _grid(resolution)
    model = make_classifier(name, X, y)
    fig, axes = plt.subplots(1, 4, figsize=(10.0, 2.8))

    _scatter_training(axes[0], X, y)
    axes[0].set_xlabel("(a)")
    for ax, k, panel in zip(axes[1:], (1, 3, 5), ("(b)", "(c)", "(d)")):
        pred, _ = predict_grid_vectorized(model, grid, k=k, params=params)
        image = pred.reshape(resolution, resolution)
        ax.imshow(image, extent=(0, 1, 0, 1), origin="lower", interpolation="nearest",
                  cmap=DECISION_CMAP, vmin=0, vmax=2)
        _scatter_training(ax, X, y, black=True)
        ax.set_xlabel(f"{panel} k = {k}")

    fig.suptitle(f"Mecanismo de classificação FEMa — {name}", y=1.02, fontsize=11)
    fig.tight_layout()
    return _save(fig, out_dir / "working_mechanism.png")


def plot_probability_map(name: str, params: BasisParameters, X, y,
                         out_dir: Path, resolution: int = 220) -> Path:
    """Grayscale certainty and class-coloured certainty map, article Fig. 5 style."""
    grid = _grid(resolution)
    model = make_classifier(name, X, y)
    _, probs = predict_grid_vectorized(model, grid, k=0, params=params)
    probs_img = probs.reshape(resolution, resolution, -1)
    pred = np.argmax(probs_img, axis=2)
    certainty = np.max(probs_img, axis=2)
    rgb = CLASS_RGB[pred] * certainty[..., None]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))
    axes[0].imshow(certainty, extent=(0, 1, 0, 1), origin="lower", cmap="gray",
                  vmin=0, vmax=1, interpolation="nearest")
    _scatter_training(axes[0], X, y)
    axes[0].set_xlabel("(a)")

    axes[1].imshow(np.clip(rgb, 0, 1), extent=(0, 1, 0, 1), origin="lower",
                   interpolation="nearest")
    _scatter_training(axes[1], X, y)
    axes[1].set_xlabel("(b)")

    fig.suptitle(f"Mapa de probabilidade e grau de certeza — {name}", y=1.02, fontsize=11)
    fig.tight_layout()
    return _save(fig, out_dir / "probability_map.png")


def generate_all(output_dir: str = "figs", resolution: int = 220,
                 names: Optional[list[str]] = None):
    output = Path(output_dir)
    X_train, y_train = make_training_set(n_per_class=5, seed=7)
    bases = names or Basis.available()
    rows, failures = [], []

    for name in bases:
        logging.info("Gerando figuras: %s", name)
        out_dir = output / name
        params = representative_parameters(name)
        try:
            paths = [
                plot_probability_functions(name, params, out_dir),
                plot_working_mechanism(name, params, X_train, y_train, out_dir, resolution),
                plot_probability_map(name, params, X_train, y_train, out_dir, resolution),
            ]
            rows.append((name, *[str(p) for p in paths]))
            logging.info("OK: %s", name)
        except Exception as exc:  # keep processing the remaining bases
            logging.exception("Falha ao gerar %s", name)
            failures.append((name, repr(exc)))

    output.mkdir(parents=True, exist_ok=True)
    with (output / "generation_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["basis", "probability_functions", "working_mechanism", "probability_map"])
        writer.writerows(rows)
    if failures:
        with (output / "generation_failures.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["basis", "error"])
            writer.writerows(failures)
    return rows, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="figs", help="Diretório de saída (padrão: figs)")
    parser.add_argument("--resolution", type=int, default=250,
                        help="Resolução dos mapas 2D (padrão: 220; Fig. 5 usa a mesma resolução)")
    parser.add_argument("--basis", nargs="*", help="Opcional: gerar apenas bases especificadas")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s: %(message)s")
    rows, failures = generate_all(args.output, args.resolution, args.basis)
    print(f"Concluído: {len(rows)} bases; {len(rows) * 3} figuras geradas.")
    if failures:
        print(f"Falhas: {len(failures)} — consulte {Path(args.output) / 'generation_failures.csv'}")
    print(f"Manifesto: {Path(args.output) / 'generation_manifest.csv'}")


if __name__ == "__main__":
    main()
