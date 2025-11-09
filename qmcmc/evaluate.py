from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import json


def posterior_predictive(
    param_samples: np.ndarray,  # (S, d+1)
    X: np.ndarray,             # (n, d)
    predict_proba_fn,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute posterior predictive mean and std of P(y=1|x).

    Returns (mean_probs, std_probs) each of shape (n,).
    """
    S = param_samples.shape[0]
    n = X.shape[0]
    probs = np.zeros((S, n), dtype=np.float64)
    for s in range(S):
        probs[s] = predict_proba_fn(param_samples[s], X)
    return probs.mean(axis=0), probs.std(axis=0)


def compute_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict:
    y_pred = (prob >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    # ROC AUC defined only if both classes present; avoid calling to suppress warnings
    if np.unique(y_true).size < 2:
        metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true, prob))
    return metrics


def compute_detailed_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict:
    """Return metrics aligned with run_genomics_training.py expectations.

    Includes:
    - accuracy, f1 (binary), f1_weighted
    - roc_auc (None if undefined)
    - loss (mean cross-entropy)
    - confusion_matrix (2x2 list)
    - classification_report (dict)
    """
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=np.float64)
    y_pred = (prob >= 0.5).astype(int)

    # Base metrics
    acc = float(accuracy_score(y_true, y_pred))
    f1_bin = float(f1_score(y_true, y_pred))
    f1_w = float(f1_score(y_true, y_pred, average="weighted"))

    # ROC AUC if both classes are present
    if np.unique(y_true).size < 2:
        roc = None
    else:
        roc = float(roc_auc_score(y_true, prob))

    # Cross-entropy loss on probabilities
    eps = 1e-12
    ce = -np.mean(y_true * np.log(prob + eps) + (1 - y_true) * np.log(1 - prob + eps))

    # Confusion matrix and report
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    cr = classification_report(y_true, y_pred, output_dict=True)

    return {
        "accuracy": acc,
        "f1": f1_bin,
        "f1_weighted": f1_w,
        "roc_auc": roc,
        "loss": float(ce),
        "confusion_matrix": cm,
        "classification_report": cr,
    }


def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_numpy(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr)
