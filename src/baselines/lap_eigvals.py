"""LapEigvals baseline (Binkowski et al., EMNLP 2025): top-K eigenvalues of the
attention-graph Laplacian, aggregated across layers."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import eigh

from ..fes.laplacian import attention_to_laplacian


def lap_eigvals_features(
    attentions_per_layer: Sequence[ArrayLike],
    top_k: int = 10,
    laplacian_kind: str = "combinatorial",
    head_aggregation: str = "mean",
) -> NDArray[np.float64]:
    """Compute the top-K Laplacian eigenvalues per layer and concatenate.

    Args:
        attentions_per_layer: sequence of attention tensors. Each may be
            [n, n] or [H, n, n].
        top_k: number of largest eigenvalues to keep per layer.
        laplacian_kind: 'combinatorial' or 'normalized'.
        head_aggregation: how to aggregate heads -> 'mean' | 'concat'.

    Returns:
        1-D feature vector of length L * top_k (or L * H * top_k for 'concat').
    """
    feats: List[np.ndarray] = []
    for layer in attentions_per_layer:
        a = np.asarray(layer, dtype=np.float64)
        if a.ndim == 2:
            heads = [a]
        elif a.ndim == 3:
            if head_aggregation == "mean":
                heads = [a.mean(axis=0)]
            else:
                heads = [a[h] for h in range(a.shape[0])]
        else:
            raise ValueError(f"unexpected attention shape: {a.shape}")
        for A in heads:
            L = attention_to_laplacian(A, kind=laplacian_kind)
            w = eigh(L, eigvals_only=True)
            # Top-k = largest
            w = np.sort(w)[::-1]
            if w.size < top_k:
                w = np.concatenate([w, np.zeros(top_k - w.size)])
            feats.append(w[:top_k])
    return np.concatenate(feats, axis=0)
