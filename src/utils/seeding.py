"""Deterministic seeding across NumPy, Python, and (optionally) PyTorch."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (if installed) for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def derived_seed(base: int, *keys: str | int) -> int:
    """Build a deterministic sub-seed from a base seed and a hierarchy of keys.

    Example: derived_seed(42, "llama-3-8b", "truthfulqa", split="val")
    """
    h = hash((base, *keys))
    return h & 0x7FFFFFFF
