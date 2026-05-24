"""Lightweight logging + run-metadata stamping."""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s :: %(message)s"


def get_logger(name: str = "fes", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "no-git"


def stamp(run_dir: str | Path, extra: dict[str, Any] | None = None) -> Path:
    """Write run metadata (env, commit, config) to <run_dir>/run.json."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
        "env": {k: v for k, v in os.environ.items() if k.startswith(("CUDA_", "HF_"))},
    }
    if extra:
        meta.update(extra)
    out = run_dir / "run.json"
    out.write_text(json.dumps(meta, indent=2, default=str))
    return out
