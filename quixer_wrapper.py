"""Wrapper exposing the original TorchQuantum Quixer model under the expected interface."""
from __future__ import annotations

import sys

import torch
import torch.nn as nn

# Ensure the local Quixer package (TorchQuantum implementation) is importable
QUIXER_PATH = '/scratch/cbjp404/bradford_hackathon_gene_sequencing/Quixer'
if QUIXER_PATH not in sys.path:
    sys.path.insert(0, QUIXER_PATH)

from quixer.quixer_model import Quixer as _QuixerModel  # type: ignore  # noqa: E402


class QuixerClassifier(nn.Module):
    """Thin wrapper around the TorchQuantum Quixer model.

    Keeps the constructor signature expected by the hybrid genomics pipeline and
    exposes the underlying embedding via ``model.quixer.embedding`` so existing
    centroid initialisation code continues to work.
    """

    def __init__(
        self,
        *,
        n_qubits: int,
        n_tokens: int,
        qsvt_polynomial_degree: int,
        n_ansatz_layers: int,
        vocabulary_size: int,
        n_classes: int,
        embedding_dimension: int,
        dropout: float,
        batch_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.quixer = _QuixerModel(
            n_qubits=n_qubits,
            n_tokens=n_tokens,
            qsvt_polynomial_degree=qsvt_polynomial_degree,
            n_ansatz_layers=n_ansatz_layers,
            vocabulary_size=vocabulary_size,
            embedding_dimension=embedding_dimension,
            dropout=dropout,
            batch_size=batch_size,
            device=device,
        )
        self.embedding = self.quixer.embedding  # convenience alias

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        logits, _ = self.quixer(tokens)
        return logits


__all__ = ["QuixerClassifier"]
