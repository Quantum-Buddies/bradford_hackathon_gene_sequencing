"""QMCMC utilities for QNLP + QD-HMC demo.

Modules:
- data: load lambeq embeddings
- features: scaler + PCA
- posteriors: logistic posterior energy + gradient
- qdhmc_adapter: wrapper for TrueQDHMC from QDHMCMAINFILE.py
- evaluate: metrics and persistence helpers
"""

__all__ = [
    "data",
    "features",
    "posteriors",
    "qdhmc_adapter",
    "evaluate",
]

