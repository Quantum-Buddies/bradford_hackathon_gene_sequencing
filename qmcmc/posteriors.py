from __future__ import annotations

import numpy as np
from typing import Tuple


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    # For large negative values, exp(z) underflows; handle piecewise
    pos_mask = z >= 0
    neg_mask = ~pos_mask
    out = np.empty_like(z, dtype=np.float64)
    out[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))
    exp_z = np.exp(z[neg_mask])
    out[neg_mask] = exp_z / (1.0 + exp_z)
    return out


def logistic_energy_and_grad(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    prior_std: float = 1.0,
) -> Tuple[float, np.ndarray]:
    """Negative log posterior and gradient for Bayesian logistic regression.

    params: shape (d+1,) -> [w[0..d-1], b]
    X: shape (n, d)  (PCA-reduced features)
    y: shape (n,)    (0/1 labels)
    prior_std: Gaussian prior std for weights and bias
    """
    params = np.asarray(params, dtype=np.float64)
    w = params[:-1]
    b = params[-1]

    z = X @ w + b
    p = _sigmoid(z)

    # Negative log-likelihood (binary cross-entropy)
    # Add small epsilon for safety
    eps = 1e-12
    nll = -np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

    # Gaussian prior
    prior_var = prior_std ** 2
    reg = 0.5 * (np.sum(w ** 2) + b ** 2) / prior_var

    energy = nll + reg

    # Gradient
    error = p - y  # shape (n,)
    grad_w = X.T @ error + w / prior_var
    grad_b = np.sum(error) + b / prior_var
    grad = np.concatenate([grad_w, np.array([grad_b])])

    # Replace any non-finite values with safe defaults
    if not np.isfinite(energy):
        energy = 1e12
    if not np.all(np.isfinite(grad)):
        grad = np.zeros_like(params)

    return float(energy), grad


def predict_proba(params: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Compute P(y=1|X, params) for logistic model."""
    w = params[:-1]
    b = params[-1]
    z = X @ w + b
    return _sigmoid(z)

