"""Trivial LM-probability baselines: perplexity, mean log-prob, max softmax prob.

These act as the floor in our comparison table. The Day-2 pipeline records the
generated-token log-probs from the wrapper, so these features are essentially
free.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def mean_token_logprob(response_logprobs: ArrayLike) -> float:
    lp = np.asarray(response_logprobs, dtype=np.float64).ravel()
    if lp.size == 0:
        return 0.0
    return float(lp.mean())


def perplexity_score(response_logprobs: ArrayLike) -> float:
    """Perplexity = exp(-mean log p). Lower = model is more confident."""
    return float(np.exp(-mean_token_logprob(response_logprobs)))


def max_softmax_prob(response_logprobs: ArrayLike) -> float:
    """The maximum per-token probability used as a sample-level confidence score.

    Equivalent to exp(max log p). Higher = more confident.
    """
    lp = np.asarray(response_logprobs, dtype=np.float64).ravel()
    if lp.size == 0:
        return 0.0
    return float(np.exp(lp.max()))
