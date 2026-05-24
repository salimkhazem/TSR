"""Prompt templates per task. Kept simple and consistent across models so that
attention-spectrum comparisons are apples-to-apples."""

from __future__ import annotations

from typing import Callable, Dict


def _qa_prompt(question: str, **_: object) -> str:
    return f"Answer the following question concisely.\n\nQuestion: {question}\nAnswer:"


def _math_prompt(question: str, **_: object) -> str:
    return (
        "Solve the following problem step by step. End your response with 'Answer: <value>'.\n\n"
        f"Problem: {question}\nSolution:"
    )


def _claim_prompt(claim: str, **_: object) -> str:
    return (
        "Determine whether the following claim is SUPPORTED, REFUTED, or has NOT ENOUGH INFO. "
        "Reply with one of these three labels only.\n\n"
        f"Claim: {claim}\nLabel:"
    )


def _halueval_prompt(question: str, answer: str | None = None, **_: object) -> str:
    return (
        "Given a question and a candidate answer, decide whether the answer is hallucinated "
        "(YES) or not (NO).\n\n"
        f"Question: {question}\nCandidate answer: {answer}\nIs the answer hallucinated?"
    )


TASK_TEMPLATES: Dict[str, Callable[..., str]] = {
    "qa": _qa_prompt,
    "math": _math_prompt,
    "claim": _claim_prompt,
    "halueval": _halueval_prompt,
}


def format_prompt(task: str, **kwargs) -> str:
    """Format a prompt for the given task family."""
    if task not in TASK_TEMPLATES:
        raise KeyError(f"Unknown task family: {task!r}; known: {sorted(TASK_TEMPLATES)}")
    return TASK_TEMPLATES[task](**kwargs)
