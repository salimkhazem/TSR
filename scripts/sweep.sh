#!/usr/bin/env bash
# Resume-friendly sweep with error skipping.
# Usage:
#   DEVICE=cuda:0 MODELS="llama-3-8b llama-3.1-8b" \
#   DATASETS="truthfulqa halueval triviaqa nq_open gsm8k math500 fever" \
#       bash scripts/sweep.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODELS="${MODELS:?set MODELS, e.g. \"llama-3-8b llama-3.1-8b\"}"
DATASETS="${DATASETS:-truthfulqa halueval triviaqa nq_open gsm8k math500}"
DEVICE="${DEVICE:-cuda:0}"
OUT_ROOT="${OUT_ROOT:-results/phase2}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
PYTHON="${PYTHON:-python}"

for M in $MODELS; do
  for D in $DATASETS; do
    OUT="$OUT_ROOT/$M/$D"
    if [ -f "$OUT/results.json" ]; then
      echo "[sweep] skip $M / $D (already done at $OUT)"
      continue
    fi
    mkdir -p "$OUT"
    LOG="$OUT/stderr.log"
    echo "[sweep] $(date -Iseconds) starting $M / $D -> $OUT"
    if ! "$PYTHON" -m src.run --model "$M" --dataset "$D" \
            --out "$OUT" --device "$DEVICE" $EXTRA_ARGS 2> >(tee "$LOG" >&2); then
      echo "[sweep] FAILED  $M / $D  (see $LOG); continuing..."
      mv "$OUT" "$OUT.failed.$(date +%s)" || true
    fi
  done
done
echo "[sweep] done."
