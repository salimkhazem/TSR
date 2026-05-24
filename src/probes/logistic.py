"""Logistic-regression probe with 5-fold CV.

Used for: turning a feature vector X[i] into a hallucination/correctness score.
The probe is a *closed*, simple model on purpose -- the FES descriptor is the
contribution; we don't want the probe to do heavy lifting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeResult:
    """Output of a probe-training run."""

    mean_auroc: float
    per_fold_auroc: List[float] = field(default_factory=list)
    scores: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    labels: NDArray[np.int64] = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    fold_assignment: NDArray[np.int64] = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    meta: Dict[str, float] = field(default_factory=dict)


class LogisticProbe:
    """Wrapper around scikit-learn's LogisticRegression with k-fold CV.

    The probe is regularized (L2) and standardizes inputs.
    """

    def __init__(self, n_splits: int = 5, C: float = 1.0, max_iter: int = 1000, seed: int = 0):
        self.n_splits = n_splits
        self.C = C
        self.max_iter = max_iter
        self.seed = seed

    def _make_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                solver="lbfgs",
                random_state=self.seed,
            )),
        ])

    def fit_eval(self, X: np.ndarray, y: np.ndarray) -> ProbeResult:
        """Run k-fold CV and return out-of-fold scores + AUROC stats."""
        from sklearn.metrics import roc_auc_score

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).ravel()
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        oof_scores = np.zeros(y.shape[0], dtype=np.float64)
        fold_assign = np.zeros(y.shape[0], dtype=np.int64)
        per_fold = []
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            pipe = self._make_pipeline()
            pipe.fit(X[tr], y[tr])
            scores = pipe.predict_proba(X[te])[:, 1]
            oof_scores[te] = scores
            fold_assign[te] = fold
            auc = float(roc_auc_score(y[te], scores))
            per_fold.append(auc)
        mean_auc = float(roc_auc_score(y, oof_scores))
        return ProbeResult(
            mean_auroc=mean_auc,
            per_fold_auroc=per_fold,
            scores=oof_scores,
            labels=y,
            fold_assignment=fold_assign,
            meta={"C": self.C, "n_splits": self.n_splits, "feat_dim": float(X.shape[1])},
        )
