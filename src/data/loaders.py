"""Deterministic dataset loaders for the 7 benchmarks.

Each loader returns a list of `Example(id, task, question, reference, meta)` dicts
ready for prompt formatting. Subsampling is deterministic (seeded).

The loaders use `datasets` lazily so that the rest of the code is importable
without HF data installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..utils.seeding import derived_seed


@dataclass
class Example:
    id: str
    task: str  # 'qa' | 'math' | 'claim' | 'halueval'
    question: str
    reference: Any  # gold answer or label
    meta: Dict[str, Any] = field(default_factory=dict)


def _shuffle_subsample(examples: List[Example], n: int | None, seed: int) -> List[Example]:
    import numpy as np

    if n is None or n >= len(examples):
        return examples
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(examples))[:n]
    return [examples[i] for i in sorted(idx)]


# ---------------------------------------------------------------------------
# Loaders -- one per dataset
# ---------------------------------------------------------------------------

def _load_truthfulqa(split: str, n: int | None, seed: int) -> List[Example]:
    from datasets import load_dataset as hf_load
    ds = hf_load("truthful_qa", "generation", split="validation")
    out: List[Example] = []
    for i, row in enumerate(ds):
        correct = row.get("correct_answers", []) or [row.get("best_answer", "")]
        out.append(
            Example(
                id=f"truthfulqa-{i}",
                task="qa",
                question=row["question"],
                reference={"correct": correct, "incorrect": row.get("incorrect_answers", [])},
                meta={"category": row.get("category", "")},
            )
        )
    return _shuffle_subsample(out, n, seed)


def _load_halueval(split: str, n: int | None, seed: int) -> List[Example]:
    from datasets import load_dataset as hf_load
    # We use the QA split for hallucination detection.
    ds = hf_load("pminervini/HaluEval", "qa", split="data")
    out: List[Example] = []
    for i, row in enumerate(ds):
        # Each row has a knowledge question, a right_answer and a hallucinated_answer.
        out.append(Example(
            id=f"halueval-{i}-right",
            task="halueval",
            question=row["question"],
            reference={"answer": row["right_answer"], "label": 0},  # 0 = not hallucinated
            meta={"knowledge": row.get("knowledge", "")},
        ))
        out.append(Example(
            id=f"halueval-{i}-hallucinated",
            task="halueval",
            question=row["question"],
            reference={"answer": row["hallucinated_answer"], "label": 1},
            meta={"knowledge": row.get("knowledge", "")},
        ))
    return _shuffle_subsample(out, n, seed)


def _load_triviaqa(split: str, n: int | None, seed: int) -> List[Example]:
    from datasets import load_dataset as hf_load
    ds = hf_load("trivia_qa", "rc.nocontext", split=split)
    out: List[Example] = []
    for i, row in enumerate(ds):
        gold = row["answer"]
        refs = (gold.get("aliases") or []) + [gold["value"]]
        out.append(Example(
            id=f"triviaqa-{i}",
            task="qa",
            question=row["question"],
            reference={"correct": refs},
        ))
    return _shuffle_subsample(out, n, seed)


def _load_nq_open(split: str, n: int | None, seed: int) -> List[Example]:
    from datasets import load_dataset as hf_load
    ds = hf_load("nq_open", split=split)
    out = [
        Example(
            id=f"nq-{i}",
            task="qa",
            question=row["question"],
            reference={"correct": row["answer"]},
        )
        for i, row in enumerate(ds)
    ]
    return _shuffle_subsample(out, n, seed)


def _load_gsm8k(split: str, n: int | None, seed: int) -> List[Example]:
    from datasets import load_dataset as hf_load
    ds = hf_load("gsm8k", "main", split=split)
    out = []
    for i, row in enumerate(ds):
        ans = row["answer"]
        gold = ans.split("####")[-1].strip() if "####" in ans else ans.strip()
        out.append(Example(
            id=f"gsm8k-{i}",
            task="math",
            question=row["question"],
            reference={"answer": gold, "full": ans},
        ))
    return _shuffle_subsample(out, n, seed)


def _load_math500(split: str, n: int | None, seed: int) -> List[Example]:
    from datasets import load_dataset as hf_load
    ds = hf_load("HuggingFaceH4/MATH-500", split=split)
    out = [
        Example(
            id=f"math500-{i}",
            task="math",
            question=row["problem"],
            reference={"answer": row["answer"], "solution": row.get("solution", "")},
            meta={"level": row.get("level", ""), "subject": row.get("subject", "")},
        )
        for i, row in enumerate(ds)
    ]
    return _shuffle_subsample(out, n, seed)


def _load_fever(split: str, n: int | None, seed: int) -> List[Example]:
    """Try several FEVER sources in order. The official `fever` is a loader
    script and is incompatible with `datasets>=4.0`."""
    from datasets import load_dataset as hf_load

    candidates = [
        # (path, config, split) — first that works wins.
        ("fever", "v1.0", split),                              # legacy script (needs datasets<4)
        ("copenlu/fever_gold_evidence", None, "dev"),          # parquet alternative
        ("pminervini/fever", None, split),                     # parquet mirror, if available
    ]
    last_err: Exception | None = None
    ds = None
    for path, cfg, sp in candidates:
        try:
            ds = hf_load(path, cfg, split=sp) if cfg else hf_load(path, split=sp)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    if ds is None:
        raise RuntimeError(
            "FEVER loader failed across all sources. "
            "Fix: `uv pip install \"datasets>=2.20,<4.0\"` or drop FEVER from the dataset list. "
            f"Last error: {last_err}"
        )
    out = []
    for i, row in enumerate(ds):
        claim = row.get("claim") or row.get("text") or row.get("hypothesis", "")
        label = row.get("label")
        if isinstance(label, str):
            # Normalize string labels -> {0=SUPPORTS, 1=REFUTES, 2=NEI}
            label = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}.get(label.upper(), label)
        out.append(Example(
            id=f"fever-{i}",
            task="claim",
            question=claim,
            reference={"label": label},
        ))
    return _shuffle_subsample(out, n, seed)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "truthfulqa": {"loader": _load_truthfulqa, "default_split": "validation", "default_n": None},
    "halueval":   {"loader": _load_halueval,   "default_split": "data",       "default_n": 4000},
    "triviaqa":   {"loader": _load_triviaqa,   "default_split": "validation", "default_n": 5000},
    "nq_open":    {"loader": _load_nq_open,    "default_split": "validation", "default_n": None},
    "gsm8k":      {"loader": _load_gsm8k,      "default_split": "test",       "default_n": None},
    "math500":    {"loader": _load_math500,    "default_split": "test",       "default_n": None},
    "fever":      {"loader": _load_fever,      "default_split": "labelled_dev","default_n": None},
}


def load_dataset(
    name: str,
    split: Optional[str] = None,
    n: Optional[int] = None,
    seed: int = 0,
) -> List[Example]:
    """Load a benchmark by name with deterministic subsampling.

    Returns:
        List of `Example`.
    """
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}; known: {sorted(DATASET_REGISTRY)}")
    spec = DATASET_REGISTRY[name]
    split = split or spec["default_split"]
    n = n if n is not None else spec["default_n"]
    return spec["loader"](split, n, derived_seed(seed, name, split or ""))
