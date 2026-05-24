"""End-to-end runner: (model, dataset) -> features -> probe -> metrics -> JSON.

Usage:
    python -m src.run \
        --model llama-3-8b --dataset truthfulqa --n 100 \
        --out results/llama3-8b/truthfulqa --device cuda:0

The pipeline:
    1. Load deterministic subsample of the dataset.
    2. For each example: prompt the model, extract per-layer (mean-pooled-head)
       attention over the prompt, decode the response, record token logprobs.
    3. Compute features:
        - FES descriptor  Phi(x)  (src/fes/descriptors.py)
        - LapEigvals (Binkowski 2025)
        - Noël 4-features (2026)
        - perplexity / mean-logprob / max-softmax-prob
    4. Build labels:
        - Hallucination-style datasets: 0/1 from gold label.
        - QA / math: exact-match (string-normalized) correctness.
    5. Train 5-fold logistic probe on each feature set; report AUROC with
       bootstrap CI, plus the unsupervised RMT-deviation score D(x).
    6. Dump to <out>/results.json plus a per-sample .parquet/.npz of features.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .baselines import lap_eigvals_features, noel_features, perplexity_score, mean_token_logprob, max_softmax_prob
from .data import format_prompt, load_dataset
from .data.loaders import Example
from .eval import auroc, bootstrap_ci
from .fes.descriptors import FESConfig, build_fes
from .fes.rmt_reference import deviation_from_goe
from .fes.thermodynamics import spectral_form_factor_g, unfold_spectrum
from .models import HFAttentionExtractor
from .probes import LogisticProbe
from .utils.logging import get_logger, stamp
from .utils.seeding import seed_everything

log = get_logger("run")


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).lower().strip()
    s = _PUNCT.sub(" ", s)
    return " ".join(s.split())


def _qa_correct(response: str, refs: List[str]) -> int:
    r = _normalize(response)
    for ref in refs:
        ref_n = _normalize(ref)
        if not ref_n:
            continue
        if ref_n == r or ref_n in r:
            return 1
    return 0


def _math_correct(response: str, gold_answer: str) -> int:
    gold = _normalize(gold_answer)
    r = _normalize(response)
    # Look for "answer: X" or any occurrence of the gold token.
    m = re.search(r"answer\s*[:=]\s*([\-\d\.\,/]+)", r)
    if m and _normalize(m.group(1)) == gold:
        return 1
    return int(gold in r)


def make_label(ex: Example, response: str) -> int:
    """Return 1 if the response is correct / not hallucinated, else 0.

    For "halueval" entries the label is precomputed in `ex.reference['label']`
    (1 = hallucinated). We invert it so 1 = "not hallucinated / correct".
    """
    if ex.task == "halueval":
        return int(1 - ex.reference["label"])
    if ex.task == "qa":
        refs = ex.reference.get("correct", [])
        if isinstance(refs, str):
            refs = [refs]
        return _qa_correct(response, refs)
    if ex.task == "math":
        return _math_correct(response, ex.reference["answer"])
    if ex.task == "claim":
        target = ex.reference["label"]
        # FEVER labels: 0=SUPPORTS, 1=REFUTES, 2=NEI
        labels = {0: "supported", 1: "refuted", 2: "not enough info"}
        return int(labels[target] in _normalize(response))
    raise ValueError(f"unknown task {ex.task!r}")


# ---------------------------------------------------------------------------
# Feature pipeline
# ---------------------------------------------------------------------------

def compute_features_for_example(
    attentions: List[np.ndarray],
    response_logprobs: np.ndarray,
    fes_cfg: FESConfig,
) -> Dict[str, np.ndarray | float]:
    """Compute all feature vectors and scalar scores for a single example."""
    phi = build_fes(attentions, cfg=fes_cfg)
    lap = lap_eigvals_features(attentions, top_k=10)
    noel = noel_features(attentions)

    # Unsupervised RMT-deviation score: average over layers.
    devs = []
    for A in attentions:
        from scipy.linalg import eigh
        from .fes.laplacian import attention_to_laplacian
        L = attention_to_laplacian(A, kind="combinatorial")
        w = eigh(L, eigvals_only=True)
        if (w > 1e-8).any():
            w = w[w > 1e-8]
        uf = unfold_spectrum(w)
        g = spectral_form_factor_g(uf, fes_cfg.t_grid)
        devs.append(deviation_from_goe(g, fes_cfg.t_grid))
    dev_mean = float(np.mean(devs)) if devs else float("nan")

    return {
        "phi": phi,
        "lap": lap,
        "noel": noel,
        "perplexity": perplexity_score(response_logprobs),
        "mean_logprob": mean_token_logprob(response_logprobs),
        "msp": max_softmax_prob(response_logprobs),
        "dev_from_goe": dev_mean,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--quantization", type=str, default=None, choices=[None, "nf4", "int8"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--m-beta", type=int, default=20)
    parser.add_argument("--p-t", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keep-heads", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip model loading; emit synthetic attention "
                             "(useful for pipeline-correctness checks).")
    args = parser.parse_args(argv)

    seed_everything(args.seed, deterministic_torch=False)
    args.out.mkdir(parents=True, exist_ok=True)
    stamp(args.out, extra={"args": vars(args)})

    log.info("loading dataset %s (n=%s)", args.dataset, args.n)
    examples = load_dataset(args.dataset, split=args.split, n=args.n, seed=args.seed) if not args.dry_run else _synthetic_examples(args.n or 8)
    log.info("got %d examples", len(examples))

    fes_cfg = FESConfig.default(m_beta=args.m_beta, p_t=args.p_t)

    if not args.dry_run:
        extractor = HFAttentionExtractor(
            model_key=args.model,
            device=args.device,
            quantization=args.quantization,
            max_new_tokens=args.max_new_tokens,
        )
        extractor.load()

    per_example: List[Dict] = []
    feats_phi, feats_lap, feats_noel = [], [], []
    msps, perps, devs = [], [], []
    labels = []

    t0 = time.time()
    for i, ex in enumerate(examples):
        prompt = format_prompt(ex.task, **{
            "question": ex.question if ex.task != "halueval" else ex.question,
            "answer": ex.reference.get("answer") if ex.task == "halueval" else None,
            "claim": ex.question if ex.task == "claim" else None,
        })

        if args.dry_run:
            attentions = _synthetic_attentions(seq=24, layers=4)
            response = "dummy"
            response_logprobs = -np.abs(np.random.default_rng(i).standard_normal(8))
        else:
            out = extractor.run(prompt, max_new_tokens=args.max_new_tokens, keep_heads=args.keep_heads)
            attentions = out.attentions_per_layer
            response = out.response
            response_logprobs = out.response_logprobs

        feats = compute_features_for_example(attentions, response_logprobs, fes_cfg)
        feats_phi.append(feats["phi"])
        feats_lap.append(feats["lap"])
        feats_noel.append(feats["noel"])
        msps.append(feats["msp"])
        perps.append(feats["perplexity"])
        devs.append(feats["dev_from_goe"])
        y = make_label(ex, response) if not args.dry_run else int(i % 2 == 0)
        labels.append(y)
        per_example.append({"id": ex.id, "label": y, "response": response[:200]})

        if (i + 1) % max(1, len(examples) // 10) == 0 or i + 1 == len(examples):
            log.info("  %d / %d  (elapsed %.1f s)", i + 1, len(examples), time.time() - t0)

    if not args.dry_run:
        extractor.unload()

    X_phi = np.stack(feats_phi, axis=0)
    X_lap = np.stack(feats_lap, axis=0)
    X_noel = np.stack(feats_noel, axis=0)
    y_arr = np.asarray(labels, dtype=np.int64)
    msps_arr = np.asarray(msps, dtype=np.float64)
    devs_arr = np.asarray(devs, dtype=np.float64)
    perps_arr = np.asarray(perps, dtype=np.float64)

    results: Dict[str, Dict] = {}
    for name, X in [("FES", X_phi), ("LapEigvals", X_lap), ("Noel-4", X_noel)]:
        probe = LogisticProbe(seed=args.seed)
        res = probe.fit_eval(X, y_arr)
        m, lo, hi = bootstrap_ci(res.labels, res.scores, n_boot=1000, seed=args.seed)
        results[name] = {
            "auroc_mean_oof": res.mean_auroc,
            "auroc_boot_ci_lo": lo,
            "auroc_boot_ci_hi": hi,
            "auroc_boot_mean": m,
            "feat_dim": int(X.shape[1]),
        }
        log.info("  %-12s AUROC = %.4f [%.4f, %.4f]", name, res.mean_auroc, lo, hi)

    # Unsupervised baselines
    for name, score in [("MSP", msps_arr), ("-perplexity", -perps_arr), ("Dev-from-GOE", devs_arr)]:
        if np.unique(y_arr).size < 2:
            results[name] = {"auroc": float("nan")}
            continue
        m, lo, hi = bootstrap_ci(y_arr, score, n_boot=1000, seed=args.seed)
        results[name] = {"auroc": m, "boot_ci": [lo, hi]}
        log.info("  %-12s AUROC = %.4f [%.4f, %.4f]", name, m, lo, hi)

    (args.out / "results.json").write_text(json.dumps({
        "model": args.model,
        "dataset": args.dataset,
        "n_examples": len(examples),
        "n_positive": int(y_arr.sum()),
        "n_negative": int((1 - y_arr).sum()),
        "results": results,
    }, indent=2))
    np.savez_compressed(
        args.out / "features.npz",
        phi=X_phi, lap=X_lap, noel=X_noel, labels=y_arr,
        msp=msps_arr, dev_from_goe=devs_arr, perplexity=perps_arr,
    )
    with (args.out / "examples.jsonl").open("w") as f:
        for rec in per_example:
            f.write(json.dumps(rec) + "\n")

    log.info("done in %.1f s, wrote -> %s", time.time() - t0, args.out)
    return 0


# ---------------------------------------------------------------------------
# Dry-run helpers
# ---------------------------------------------------------------------------

def _synthetic_examples(n: int) -> List[Example]:
    return [
        Example(
            id=f"dry-{i}",
            task="qa",
            question=f"What is {i} + {i}?",
            reference={"correct": [str(2 * i)]},
        )
        for i in range(n)
    ]


def _synthetic_attentions(seq: int, layers: int) -> List[np.ndarray]:
    rng = np.random.default_rng(0)
    out = []
    for _ in range(layers):
        A = rng.dirichlet(np.ones(seq), size=seq)  # row-stochastic
        out.append(A)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
