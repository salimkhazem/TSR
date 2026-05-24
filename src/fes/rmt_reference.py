"""Random Matrix Theory reference curves (GOE) for the spectral form factor.

We use the standard analytic GOE SFF (Mehta, Random Matrices, 3rd ed., 2004,
Chapter 6). After unfolding the spectrum to unit mean level spacing, the
connected SFF for the Gaussian Orthogonal Ensemble (beta_Dyson = 1) is

    K_GOE(tau) = 2*tau - tau * log(1 + 2*tau)              for 0 <= tau <= 1
                 2 - tau * log( (2*tau + 1) / (2*tau - 1) )  for tau > 1

where tau = t / (2 pi). For Poisson-like (integrable) spectra, K(tau) = 1.

For our purposes we report the *normalized* SFF g(t) = |sum_k e^{-i t lambda_k}|^2 / n^2.
After unfolding and removing the disconnected piece, this approaches K_GOE(tau)
in the large-n limit. We expose K_GOE as a reference curve on the same t-grid
used for empirical g(t).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..utils.compat import trapz


def goe_sff(tau: ArrayLike) -> NDArray[np.float64]:
    """Analytic GOE spectral form factor K(tau), tau = t/(2 pi) on unfolded eigs."""
    t = np.atleast_1d(np.asarray(tau, dtype=np.float64))
    out = np.empty_like(t)
    small = t <= 1.0
    large = ~small
    # Small-tau branch
    ts = t[small]
    out[small] = 2.0 * ts - ts * np.log1p(2.0 * ts)
    # Large-tau branch: stable form
    tl = t[large]
    # log((2t+1)/(2t-1)) = log1p(2/(2t-1))
    out[large] = 2.0 - tl * np.log1p(2.0 / (2.0 * tl - 1.0))
    return out if np.ndim(tau) > 0 else float(out[0])  # type: ignore[return-value]


def deviation_from_goe(
    g_empirical: ArrayLike,
    t_grid: ArrayLike,
    integrate: bool = True,
) -> float:
    """Integrated squared deviation D(x) = int_0^T (g(t) - K_GOE(t/(2 pi)))^2 dt.

    Args:
        g_empirical: [p] empirical g(t) values on `t_grid`.
        t_grid: [p] time grid.
        integrate: if True, returns the integral (trapezoidal); else returns
            the pointwise squared-residual vector.
    """
    g = np.asarray(g_empirical, dtype=np.float64).ravel()
    t = np.asarray(t_grid, dtype=np.float64).ravel()
    if g.shape != t.shape:
        raise ValueError(f"shape mismatch: g {g.shape}, t {t.shape}")
    tau = t / (2.0 * np.pi)
    ref = goe_sff(tau)
    resid2 = (g - ref) ** 2
    if not integrate:
        return resid2  # type: ignore[return-value]
    return float(trapz(resid2, t))
