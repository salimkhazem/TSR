"""Build symmetrized graph Laplacians from attention matrices."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def symmetrize_attention(A: ArrayLike) -> NDArray[np.float64]:
    """Symmetrize a row-stochastic attention matrix: A_tilde = (A + A^T)/2."""
    a = np.asarray(A, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"A must be square 2-D, got {a.shape}")
    return 0.5 * (a + a.T)


def combinatorial_laplacian(A_sym: ArrayLike) -> NDArray[np.float64]:
    """L = D - A on a symmetric weight matrix A_sym (already symmetrized)."""
    a = np.asarray(A_sym, dtype=np.float64)
    d = a.sum(axis=1)
    return np.diag(d) - a


def normalized_laplacian(A_sym: ArrayLike, eps: float = 1e-12) -> NDArray[np.float64]:
    """L_norm = I - D^{-1/2} A D^{-1/2} on a symmetric weight matrix.

    Stabilized: zero rows produce zero contribution rather than NaN.
    """
    a = np.asarray(A_sym, dtype=np.float64)
    d = a.sum(axis=1)
    d_inv_sqrt = np.where(d > eps, 1.0 / np.sqrt(np.maximum(d, eps)), 0.0)
    n = a.shape[0]
    L = np.eye(n) - (d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :])
    # Numerical symmetrization
    return 0.5 * (L + L.T)


def attention_to_laplacian(
    A: ArrayLike,
    kind: str = "combinatorial",
) -> NDArray[np.float64]:
    """Convenience: take a (possibly non-symmetric) attention matrix and return L.

    Args:
        A: [n, n] attention matrix (row-stochastic or not).
        kind: 'combinatorial' (default) or 'normalized'.
    """
    A_sym = symmetrize_attention(A)
    if kind == "combinatorial":
        return combinatorial_laplacian(A_sym)
    if kind == "normalized":
        return normalized_laplacian(A_sym)
    raise ValueError(f"unknown Laplacian kind: {kind!r}")
