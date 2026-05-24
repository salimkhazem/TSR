"""NumPy version compatibility shims."""

from __future__ import annotations

import numpy as np


def trapz(y, x=None, dx: float = 1.0, axis: int = -1):
    """Trapezoidal integration that works on both NumPy 1.x (`np.trapz`) and 2.x (`np.trapezoid`)."""
    fn = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
    if x is None:
        return fn(y, dx=dx, axis=axis)
    return fn(y, x=x, dx=dx, axis=axis)
