"""Thermodynamic + RMT functionals of a Hermitian operator's spectrum.

All functions take an array of real, non-negative eigenvalues `eigs` (shape [n])
and a grid (beta or t). Returns NumPy arrays. No autograd; everything is closed
form. Numerical stability: log-sum-exp for Z, central differences for S/C.

Reference forms:
    Z(beta)  = sum_k exp(-beta * lambda_k)
    F(beta)  = -1/beta * log Z(beta)
    S(beta)  = beta^2 * dF/dbeta = -d(beta F)/dbeta + F (computed via central diff)
    C(beta)  = -beta^2 * d^2(beta F)/dbeta^2     (heat capacity)
    g(t)     = |sum_k exp(-i t lambda_k)|^2 / n^2  (spectral form factor)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_eigs(eigs: ArrayLike) -> NDArray[np.float64]:
    e = np.asarray(eigs, dtype=np.float64).ravel()
    if e.size == 0:
        raise ValueError("eigs is empty")
    if np.any(~np.isfinite(e)):
        raise ValueError("eigs contains non-finite values")
    return e


def log_partition(eigs: ArrayLike, beta: ArrayLike) -> NDArray[np.float64]:
    """Numerically stable log Z(beta) = log sum_k exp(-beta * lambda_k).

    Uses log-sum-exp. Broadcasts over a beta grid.

    Args:
        eigs: [n] eigenvalues (real).
        beta: scalar or [m] inverse-temperature grid.

    Returns:
        [m] (or scalar) array of log Z(beta).
    """
    e = _as_eigs(eigs)
    b = np.atleast_1d(np.asarray(beta, dtype=np.float64))
    # shape: [m, n]
    x = -b[:, None] * e[None, :]
    m = np.max(x, axis=1, keepdims=True)
    logZ = (m + np.log(np.sum(np.exp(x - m), axis=1, keepdims=True))).ravel()
    return logZ if np.ndim(beta) > 0 else float(logZ[0])  # type: ignore[return-value]


def partition_Z(eigs: ArrayLike, beta: ArrayLike) -> NDArray[np.float64]:
    """Z(beta) = sum_k exp(-beta lambda_k)."""
    logZ = log_partition(eigs, beta)
    return np.exp(logZ)


def free_energy_F(eigs: ArrayLike, beta: ArrayLike) -> NDArray[np.float64]:
    """F(beta) = -1/beta * log Z(beta).

    Returns array shaped like `beta`. For beta -> 0 we apply the analytic limit
    F(0+) = -lim_{beta -> 0} log Z / beta = mean(eigs) - log(n)/0  -> diverges;
    by convention we return -log(n)/beta + mean(eigs) - i.e. the leading term.
    For all `beta > 0` we just compute -log Z / beta directly.
    """
    b = np.atleast_1d(np.asarray(beta, dtype=np.float64))
    if np.any(b <= 0):
        raise ValueError("beta must be strictly positive")
    logZ = log_partition(eigs, b)
    F = -logZ / b
    return F if np.ndim(beta) > 0 else float(F[0])  # type: ignore[return-value]


def _boltzmann_probs(eigs: NDArray[np.float64], beta: float) -> NDArray[np.float64]:
    """p_k = exp(-beta lambda_k) / Z(beta), stable."""
    x = -beta * eigs
    x = x - np.max(x)
    p = np.exp(x)
    p /= p.sum()
    return p


def spectral_entropy_S(eigs: ArrayLike, beta: ArrayLike) -> NDArray[np.float64]:
    """Thermodynamic entropy S(beta) of the Gibbs ensemble.

    S(beta) = - sum_k p_k log p_k  where p_k = exp(-beta lambda_k)/Z.
    This is the Shannon entropy of the Boltzmann distribution over eigenstates
    and equals the standard thermodynamic entropy of the spectrum.
    """
    e = _as_eigs(eigs)
    b = np.atleast_1d(np.asarray(beta, dtype=np.float64))
    S = np.empty_like(b)
    for i, bi in enumerate(b):
        p = _boltzmann_probs(e, float(bi))
        # 0 log 0 = 0 by convention.
        nz = p > 0
        S[i] = -np.sum(p[nz] * np.log(p[nz]))
    return S if np.ndim(beta) > 0 else float(S[0])  # type: ignore[return-value]


def heat_capacity_C(eigs: ArrayLike, beta: ArrayLike) -> NDArray[np.float64]:
    """Heat capacity C(beta) = beta^2 * Var_p[lambda] where p_k is Boltzmann.

    Identity: C = beta^2 (<H^2> - <H>^2). Stable and exact in one pass.
    """
    e = _as_eigs(eigs)
    b = np.atleast_1d(np.asarray(beta, dtype=np.float64))
    C = np.empty_like(b)
    for i, bi in enumerate(b):
        p = _boltzmann_probs(e, float(bi))
        mu = float(np.sum(p * e))
        var = float(np.sum(p * (e - mu) ** 2))
        C[i] = (float(bi) ** 2) * var
    return C if np.ndim(beta) > 0 else float(C[0])  # type: ignore[return-value]


def spectral_form_factor_g(
    eigs: ArrayLike,
    t: ArrayLike,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Spectral form factor g(t) = |sum_k exp(-i t lambda_k)|^2 / n^2.

    Args:
        eigs: [n] real eigenvalues.
        t: scalar or [p] time grid.
        normalize: if True, divides by n^2 (so g(0) = 1).

    Returns:
        [p] array of g(t) in [0, 1].
    """
    e = _as_eigs(eigs)
    tt = np.atleast_1d(np.asarray(t, dtype=np.float64))
    n = e.size
    # [p, n]
    phase = np.exp(-1j * tt[:, None] * e[None, :])
    s = np.sum(phase, axis=1)
    g = (s * np.conj(s)).real
    if normalize:
        g = g / (n * n)
    return g if np.ndim(t) > 0 else float(g[0])  # type: ignore[return-value]


def unfold_spectrum(eigs: ArrayLike) -> NDArray[np.float64]:
    """Rescale eigenvalues by mean level spacing so the unfolded density is ~1.

    Standard RMT unfolding: sort eigenvalues, replace lambda_k by k / N(lambda_k)
    where N is the staircase function. Here we use the simplest form: rescale by
    the mean nearest-neighbor spacing so that <s_i> = 1 globally. This is exact
    for uniform-density spectra and a good approximation otherwise.
    """
    e = np.sort(_as_eigs(eigs))
    if e.size < 2:
        return e.copy()
    spacings = np.diff(e)
    mean_spacing = float(spacings.mean())
    if mean_spacing <= 0:
        return e.copy()
    return (e - e[0]) / mean_spacing
