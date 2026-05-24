"""Evaluation metrics: AUROC, AUARC, ECE, Brier, bootstrap CIs."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_auc_score


def auroc(y_true: ArrayLike, scores: ArrayLike) -> float:
    """Area under the ROC curve. Higher score = positive class."""
    y = np.asarray(y_true, dtype=np.int64).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def auarc(
    y_correct: ArrayLike,
    confidence: ArrayLike,
) -> float:
    """Area Under the Accuracy-Rejection Curve.

    y_correct[i] is 1 if the model's response is correct, 0 otherwise.
    confidence[i] is the model's confidence (higher = more confident).
    AUARC sweeps rejection thresholds from 0 to 1 (fraction rejected) and
    computes the mean accuracy on the retained subset.
    """
    y = np.asarray(y_correct, dtype=np.int64).ravel()
    c = np.asarray(confidence, dtype=np.float64).ravel()
    n = y.size
    order = np.argsort(-c)  # most confident first
    y_sorted = y[order]
    # accuracy@k for k=1..n
    cum_correct = np.cumsum(y_sorted)
    ks = np.arange(1, n + 1)
    acc = cum_correct / ks
    return float(np.trapezoid(acc, ks / n) if hasattr(np, "trapezoid") else np.trapz(acc, ks / n))


def expected_calibration_error(
    y_true: ArrayLike,
    probs: ArrayLike,
    n_bins: int = 15,
) -> float:
    """Standard ECE with equal-width bins on confidence."""
    y = np.asarray(y_true, dtype=np.int64).ravel()
    p = np.asarray(probs, dtype=np.float64).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = y.size
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1])
        if i == n_bins - 1:
            m = (p >= bins[i]) & (p <= bins[i + 1])
        if not m.any():
            continue
        acc = y[m].mean()
        conf = p[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)


def brier_score(y_true: ArrayLike, probs: ArrayLike) -> float:
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(probs, dtype=np.float64).ravel()
    return float(np.mean((p - y) ** 2))


def bootstrap_ci(
    y_true: ArrayLike,
    scores: ArrayLike,
    metric_fn: Callable[[ArrayLike, ArrayLike], float] = auroc,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """(mean, lo, hi) percentile-bootstrap CI of metric_fn(y, scores)."""
    y = np.asarray(y_true).ravel()
    s = np.asarray(scores).ravel()
    rng = np.random.default_rng(seed)
    n = y.size
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.unique(y[idx]).size < 2:
            boot[b] = np.nan
            continue
        boot[b] = metric_fn(y[idx], s[idx])
    valid = boot[~np.isnan(boot)]
    if valid.size == 0:
        return float("nan"), float("nan"), float("nan")
    lo = float(np.percentile(valid, 100 * alpha / 2))
    hi = float(np.percentile(valid, 100 * (1 - alpha / 2)))
    return float(metric_fn(y, s)), lo, hi
