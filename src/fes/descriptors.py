"""Build the Free-Energy Signature (FES) descriptor Phi(x) from attention matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import eigh

from .laplacian import attention_to_laplacian
from .thermodynamics import (
    free_energy_F,
    spectral_entropy_S,
    heat_capacity_C,
    spectral_form_factor_g,
    unfold_spectrum,
)


@dataclass(frozen=True)
class FESConfig:
    """Configuration of the descriptor grids."""

    beta_grid: NDArray[np.float64]
    t_grid: NDArray[np.float64]
    laplacian_kind: str = "combinatorial"  # or "normalized"
    use_unfolded_sff: bool = True
    head_aggregation: str = "mean"  # 'mean' | 'concat' | 'first'

    @staticmethod
    def default(m_beta: int = 20, p_t: int = 30) -> "FESConfig":
        return FESConfig(
            beta_grid=np.logspace(-2, 2, m_beta),
            t_grid=np.logspace(-1, 3, p_t),
        )


def _layer_eigs(A: ArrayLike, kind: str) -> NDArray[np.float64]:
    """Eigenvalues of the symmetrized Laplacian built from one [n, n] attention."""
    L = attention_to_laplacian(A, kind=kind)
    # eigh returns ascending real eigenvalues for a symmetric matrix.
    w = eigh(L, eigvals_only=True)
    # Numerical floor: tiny negative eigenvalues from roundoff -> 0.
    w = np.where(w < 0, np.where(w > -1e-8, 0.0, w), w)
    return np.asarray(w, dtype=np.float64)


def build_layer_descriptor(
    A_heads: ArrayLike | Iterable[ArrayLike],
    cfg: FESConfig | None = None,
) -> NDArray[np.float64]:
    """Compute the per-layer descriptor [F | S | C | g] for one layer.

    Args:
        A_heads: either a single [n, n] attention matrix (one effective head),
            or an iterable of [n, n] matrices (one per head).
        cfg: FES configuration. Defaults to FESConfig.default().

    Returns:
        1-D array of length (3 * m_beta + p_t) containing
        [F(beta), S(beta), C(beta), g(t)] concatenated.
    """
    cfg = cfg or FESConfig.default()

    arr = np.asarray(A_heads)
    if arr.ndim == 2:
        heads = [arr]
    elif arr.ndim == 3:
        heads = [arr[h] for h in range(arr.shape[0])]
    else:
        heads = list(A_heads)  # type: ignore[arg-type]

    feats_per_head = []
    for A in heads:
        eigs = _layer_eigs(A, kind=cfg.laplacian_kind)
        F = free_energy_F(eigs, cfg.beta_grid)
        S = spectral_entropy_S(eigs, cfg.beta_grid)
        C = heat_capacity_C(eigs, cfg.beta_grid)
        eigs_for_sff = unfold_spectrum(eigs) if cfg.use_unfolded_sff else eigs
        g = spectral_form_factor_g(eigs_for_sff, cfg.t_grid)
        feats_per_head.append(np.concatenate([F, S, C, g]))

    feats = np.stack(feats_per_head, axis=0)  # [H, D]

    if cfg.head_aggregation == "mean":
        return feats.mean(axis=0)
    if cfg.head_aggregation == "first":
        return feats[0]
    if cfg.head_aggregation == "concat":
        return feats.reshape(-1)
    raise ValueError(f"unknown head_aggregation: {cfg.head_aggregation!r}")


def build_fes(
    attentions_per_layer: Iterable[ArrayLike],
    cfg: FESConfig | None = None,
) -> NDArray[np.float64]:
    """Compute the full FES descriptor Phi(x) by concatenating per-layer descriptors.

    Args:
        attentions_per_layer: iterable of per-layer attention tensors. Each
            element is either [n, n] (already head-aggregated) or [H, n, n].
        cfg: FES configuration.

    Returns:
        1-D Phi(x) of length L * (3 * m_beta + p_t) (or H times that, if
        head_aggregation == 'concat').
    """
    cfg = cfg or FESConfig.default()
    layer_feats = [build_layer_descriptor(A, cfg=cfg) for A in attentions_per_layer]
    return np.concatenate(layer_feats, axis=0)
