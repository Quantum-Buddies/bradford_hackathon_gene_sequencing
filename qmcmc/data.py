from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import numpy as np


@dataclass
class SplitData:
    X: np.ndarray
    y: np.ndarray


@dataclass
class EmbeddingSplits:
    train: SplitData
    val: SplitData
    test: SplitData
    embedding_dim: int
    n_classes: int


def _load_split(file_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    obj = torch.load(str(file_path), map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected format for {file_path}: expected dict with 'embeddings' and 'labels'")
    X = obj.get("embeddings")
    y = obj.get("labels")
    md = obj.get("metadata", {})

    if X is None or y is None:
        raise ValueError(f"Missing 'embeddings' or 'labels' in {file_path}")

    # Convert to numpy
    X_np = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
    y_np = y.detach().cpu().numpy().astype(np.int64) if hasattr(y, "detach") else np.asarray(y, dtype=np.int64)
    return X_np, y_np, md


def load_embeddings(embeddings_dir: str | Path) -> EmbeddingSplits:
    """Load train/val/test lambeq embeddings saved by lambeq_encoder.py.

    Expected files: train.pt, val.pt, test.pt with keys: 'embeddings', 'labels', 'metadata'.
    """
    base = Path(embeddings_dir)
    train_path = base / "train.pt"
    val_path = base / "val.pt"
    test_path = base / "test.pt"

    if not train_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {test_path}")

    X_tr, y_tr, md_tr = _load_split(train_path)
    X_va, y_va, md_va = _load_split(val_path)
    X_te, y_te, md_te = _load_split(test_path)

    # Derive dims/classes from metadata if present, else from arrays
    embedding_dim = int(md_tr.get("embedding_dim", X_tr.shape[1]))
    n_classes = int(md_tr.get("n_classes", 2))
    # sanity check
    _ = (md_va, md_te)  # not strictly used

    return EmbeddingSplits(
        train=SplitData(X_tr, y_tr),
        val=SplitData(X_va, y_va),
        test=SplitData(X_te, y_te),
        embedding_dim=embedding_dim,
        n_classes=n_classes,
    )

