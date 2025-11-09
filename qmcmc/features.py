from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json


@dataclass
class FeaturePipeline:
    scaler: StandardScaler
    pca: PCA

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.pca.transform(Xs)

    def export_metadata(self) -> Dict:
        return {
            "scaler": {
                "mean": self.scaler.mean_.tolist(),
                "scale": self.scaler.scale_.tolist(),
            },
            "pca": {
                "n_components": int(self.pca.n_components_),
                "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist(),
            },
        }


def build_feature_pipeline(X_train: np.ndarray, n_components: int) -> FeaturePipeline:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=42)
    pca.fit(Xs)
    return FeaturePipeline(scaler=scaler, pca=pca)


def save_preprocessing(pipeline: FeaturePipeline, out_path: Path) -> None:
    meta = pipeline.export_metadata()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
