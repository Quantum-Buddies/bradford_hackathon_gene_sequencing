from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_roc_curves(
    y_true_by_split: Dict[str, np.ndarray],
    prob_by_split: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    _ensure_dir(out_path)
    plt.figure(figsize=(8, 6))
    any_curve = False
    for name in ["train", "val", "test"]:
        y = y_true_by_split.get(name)
        p = prob_by_split.get(name)
        if y is None or p is None:
            continue
        # Guard single-class splits
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
        any_curve = True

    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    if any_curve:
        plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()


def plot_confusion_matrices(
    y_true_by_split: Dict[str, np.ndarray],
    prob_by_split: Dict[str, np.ndarray],
    out_path: Path,
    threshold: float = 0.5,
) -> None:
    _ensure_dir(out_path)
    splits = ["train", "val", "test"]
    fig, axes = plt.subplots(1, len(splits), figsize=(14, 4))
    if len(splits) == 1:
        axes = [axes]
    for ax, name in zip(axes, splits):
        y = y_true_by_split.get(name)
        p = prob_by_split.get(name)
        ax.set_title(f"{name}")
        if y is None or p is None:
            ax.axis("off")
            continue
        y_pred = (p >= threshold).astype(int)
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", color="black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()


def plot_energy_trace(energies: np.ndarray, out_path: Path) -> None:
    _ensure_dir(out_path)
    plt.figure(figsize=(8, 4))
    plt.plot(energies, lw=1.2)
    plt.xlabel("Sample Index")
    plt.ylabel("Energy (neg log posterior)")
    plt.title("Energy Trace")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()


def plot_posterior_traces(samples: np.ndarray, out_path: Path) -> None:
    _ensure_dir(out_path)
    n_params = samples.shape[1]
    cols = min(4, n_params)
    rows = int(np.ceil(n_params / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.5 * rows), squeeze=False)
    for i in range(n_params):
        r, c = divmod(i, cols)
        axes[r][c].plot(samples[:, i], lw=1.0)
        axes[r][c].set_title(f"theta[{i}]")
    # Hide unused axes
    for j in range(n_params, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()


def plot_posterior_pairplot(samples: np.ndarray, out_path: Path, max_dims: int = 4) -> None:
    _ensure_dir(out_path)
    n_params = samples.shape[1]
    d = min(n_params, max_dims)
    S = samples[:, :d]
    fig, axes = plt.subplots(d, d, figsize=(2.8 * d, 2.8 * d), squeeze=False)
    for i in range(d):
        for j in range(d):
            ax = axes[i][j]
            if i == j:
                ax.hist(S[:, i], bins=30, color="C0", alpha=0.7)
            else:
                ax.scatter(S[:, j], S[:, i], s=4, alpha=0.4)
            if i == d - 1:
                ax.set_xlabel(f"θ{j}")
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(f"θ{i}")
            else:
                ax.set_yticklabels([])
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()


def plot_decision_boundary(
    Z: np.ndarray,
    y: np.ndarray,
    theta_mean: np.ndarray,
    out_path: Path,
    padding: float = 0.5,
    grid_steps: int = 200,
) -> None:
    """Plot decision boundary in 2D PCA space using mean posterior parameters.

    Only works when Z has exactly 2 columns (2D).
    """
    _ensure_dir(out_path)
    if Z.shape[1] != 2:
        # silently skip if not 2D
        return
    w = theta_mean[:-1]
    b = theta_mean[-1]
    x_min, x_max = Z[:, 0].min() - padding, Z[:, 0].max() + padding
    y_min, y_max = Z[:, 1].min() - padding, Z[:, 1].max() + padding
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_steps),
        np.linspace(y_min, y_max, grid_steps),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    logits = grid @ w + b
    probs = 1.0 / (1.0 + np.exp(-logits))
    Zmap = probs.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    # Fix colorbar scale to 0..1 (probabilities) to avoid confusing
    # scientific offset formatting on tight ranges
    levels = np.linspace(0.0, 1.0, 21)
    cs = plt.contourf(xx, yy, Zmap, levels=levels, vmin=0.0, vmax=1.0, cmap="RdBu", alpha=0.5)
    plt.colorbar(cs, label="P(y=1)")
    # Overlay data
    plt.scatter(Z[y == 0, 0], Z[y == 0, 1], s=10, c="C0", alpha=0.7, label="y=0")
    plt.scatter(Z[y == 1, 0], Z[y == 1, 1], s=10, c="C3", alpha=0.7, label="y=1")
    plt.legend()
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Decision Boundary (Posterior Mean)")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()


def plot_metric_bars(
    metrics_by_split: Dict[str, Dict],
    out_path: Path,
) -> None:
    """Bar charts for Accuracy (%), F1-weighted, and Loss across splits.

    metrics_by_split must contain keys: 'train', 'val', 'test', each with
    'accuracy', 'f1_weighted', and 'loss'.
    """
    _ensure_dir(out_path)
    splits = ["train", "val", "test"]
    acc = [100.0 * float(metrics_by_split[s]["accuracy"]) for s in splits]
    f1w = [float(metrics_by_split[s]["f1_weighted"]) for s in splits]
    loss = [float(metrics_by_split[s]["loss"]) for s in splits]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    # Accuracy
    axes[0].bar(splits, acc, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0].set_title("Accuracy (%)")
    for i, v in enumerate(acc):
        axes[0].text(i, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylim(0, max(100.0, max(acc) * 1.1))

    # F1 weighted
    axes[1].bar(splits, f1w, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[1].set_title("F1 (weighted)")
    axes[1].set_ylim(0, max(1.0, max(f1w) * 1.1))
    for i, v in enumerate(f1w):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    # Loss
    axes[2].bar(splits, loss, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[2].set_title("Loss (cross-entropy)")
    max_loss = max(loss) if loss else 0.0
    upper = max(1.0, max_loss * 1.2)
    axes[2].set_ylim(0, upper)
    for i, v in enumerate(loss):
        axes[2].text(i, v + 0.02 * (max(loss) if max(loss) > 0 else 1), f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    for ax in axes:
        ax.grid(True, axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()


def plot_classification_report_table(report: Dict, out_path: Path) -> None:
    """Render sklearn classification_report dict as a table figure (test split)."""
    _ensure_dir(out_path)
    # Keep only labels '0', '1', 'macro avg', 'weighted avg' if present
    keys = [k for k in ["0", "1", "macro avg", "weighted avg"] if k in report]
    cols = ["precision", "recall", "f1-score", "support"]
    cell_text = []
    for k in keys:
        row = report[k]
        cell_text.append([
            f"{row.get('precision', 0):.3f}",
            f"{row.get('recall', 0):.3f}",
            f"{row.get('f1-score', 0):.3f}",
            f"{int(row.get('support', 0))}",
        ])

    plt.figure(figsize=(6, 0.6 * (len(keys) + 2)))
    plt.axis('off')
    table = plt.table(
        cellText=cell_text,
        rowLabels=keys,
        colLabels=cols,
        loc='center',
        cellLoc='center',
    )
    table.scale(1, 1.5)
    plt.title("Classification Report (Test)")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
