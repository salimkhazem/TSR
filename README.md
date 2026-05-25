# Thermodynamic Signatures of Reasoning

**Free-Energy + RMT Diagnostics for LLM Hallucination Detection**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-%E2%89%A52.3-ee4c2c)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)
<!-- [![Paper](https://img.shields.io/badge/paper-arXiv%3AXXXX.XXXXX-b31b1b)](https://arxiv.org/abs/XXXX.XXXXX) -->

Official implementation of *Free-Energy Signatures* (**FES**), a training-free spectral descriptor for hallucination detection in large language models. FES treats each attention layer's graph Laplacian as a Hamiltonian and extracts its thermodynamic potentials (partition function, free energy, spectral entropy, heat capacity) together with the random-matrix-theory (RMT) spectral form factor. Across **6 LLMs × 6 benchmarks**, FES achieves a new SOTA for training-free hallucination detection (**+6.5 AUROC** over the strongest spectral baseline) and reveals a falsifiable signature of generation quality: valid reasoning produces Wigner–Dyson level statistics, hallucinations produce Poisson-like statistics.

---

## Repository layout

```
.
├── src/
│   ├── fes/          # Core: Z, F, S, C, SFF, GOE reference
│   ├── baselines/    # LapEigvals, GoR-4, perplexity / MSP
│   ├── data/         # Dataset loaders + prompt templates
│   ├── models/       # HF wrapper with attention extraction
│   ├── probes/       # 5-fold logistic probe
│   ├── eval/         # AUROC / AUARC / ECE / Brier + bootstrap CI
│   ├── viz/          # Tables and figures for the paper
│   ├── theory/       # Perturbation tests for Theorem 1
│   ├── utils/        # Seeding, logging, compat shims
│   └── run.py        # End-to-end orchestrator
├── tests/            # 42 unit tests
├── scripts/          # toy_experiment.py, phase1_smoke.sh, sweep.sh
├── configs/          # YAMLs for models / datasets / experiment phases
├── paper/            # LaTeX sources, figures, tables, references
└── results/          # Per-(model, dataset) JSON + features.npz
```

---

## Installation

### Option A — uv (recommended)

```bash
uv venv 
source .venv/bin/activate
uv pip install --upgrade pip
uv pip install -e ".[llm,dev]"
```

### Option B — conda 

```bash
conda env create -f environment.yml
conda activate fes-llm
pip install -e ".[llm,dev]"
```

### Option C — venv + pip

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[llm,dev]"
```



### Hugging Face authentication

`meta-llama/Meta-Llama-3-8B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`, and
`google/gemma-2-9b-it` are **gated**. Accept their licenses on the Hugging Face
web UI under your account, then:

```bash
huggingface-cli login
huggingface-cli whoami
```

The other four models (Mistral, Qwen2.5, Phi-3) are open-weight.

### Optional — set a shared cache root

```bash
export HF_HOME=/path/to/hf_cache
export HF_DATASETS_CACHE=/path/to/hf_cache/datasets
```

---

## Quick start (≤ 5 min)

```bash
# 1. Unit tests (42 tests, ~1 s on CPU)
python -m pytest tests/ -v

# 2. Toy experiment — reproduces the GOE vs Poisson signature on synthetic graphs.
#    Writes paper/figures/toy.{pdf,png}.
python scripts/toy_experiment.py --n-graphs 500 --n-nodes 64
# Expect:
#   <r>(GOE)     ~= 0.5310   (Atas et al. prediction 0.5359)
#   <r>(Poisson) ~= 0.3869   (Atas et al. prediction 0.3863)
#   AUROC primary >= 0.99    -> "ACCEPTANCE GATE: PASS"
```

If either step fails, do **not** proceed to the full reproduction — the rest of
the paper depends on these numbers. See [Troubleshooting](#troubleshooting).

---

## Reproducing the paper

### Phase 1 — smoke test (~30 min on 1× A6000)

Llama-3-8B on 100 TruthfulQA items, full FES pipeline + probe.

```bash
DEVICE=cuda:0 N=100 bash scripts/phase1_smoke.sh
cat results/phase1_smoke/llama3-8b_truthfulqa/results.json | python -m json.tool
```

Acceptance gate: end-to-end runs without error, FES AUROC ≥ 0.60.

### Phase 2 — full empirical matrix (~6–30 h on 1× A6000)

`scripts/sweep.sh` is **resume-friendly**: it skips any cell where
`results/phase2/<M>/<D>/results.json` already exists, and failed cells are
renamed to `*.failed.<ts>` so the loop continues.

```bash
DEVICE=cuda:0 \
MODELS="llama-3-8b llama-3.1-8b mistral-7b qwen2.5-7b gemma-2-9b phi-3-medium" \
DATASETS="truthfulqa halueval triviaqa nq_open gsm8k math500" \
    bash scripts/sweep.sh
```

**FEVER is excluded** from the headline grid: its Hugging Face distribution is
script-based and `datasets>=4.0` removed loader-script support. The FEVER
sub-experiment uses a parquet mirror documented in
[paper/appendix/F_reproducibility.tex](paper/appendix/F_reproducibility.tex).

**Phi-3-medium (14B) may OOM** at fp16 on a single 48 GB GPU. Fall back to 4-bit:

```bash
DEVICE=cuda:0 MODELS="phi-3-medium" \
DATASETS="truthfulqa halueval triviaqa nq_open gsm8k math500" \
EXTRA_ARGS="--quantization nf4" \
    bash scripts/sweep.sh
```
---

## CLI reference

All experiments go through `python -m src.run`.

| Flag                  | Type       | Default     | Description                                                  |
| --------------------- | ---------- | ----------- | ------------------------------------------------------------ |
| `--model`             | str        | (required)  | Model key from `configs/models/`                             |
| `--dataset`           | str        | (required)  | Dataset key from `src/data/loaders.py`                       |
| `--out`               | Path       | (required)  | Output directory                                             |
| `--split`             | str        | `None`      | HF dataset split override                                    |
| `--n`                 | int        | full        | Number of examples to subsample (deterministic)              |
| `--seed`              | int        | `0`         | Global seed                                                  |
| `--device`            | str        | `cuda:0`    | Torch device                                                 |
| `--quantization`      | str        | `None`      | `None`, `nf4`, or `int8`                                     |
| `--max-new-tokens`    | int        | `256`       | Generation length cap                                        |
| `--m-beta`            | int        | `20`        | Number of inverse-temperatures in β-grid                     |
| `--p-t`               | int        | `30`        | Number of times in t-grid for the spectral form factor       |
| `--keep-heads`        | flag       | off         | Keep per-head attention instead of mean-pooling              |
| `--dry-run`           | flag       | off         | Validate config and exit without touching the GPU            |

---

## Configs

Global defaults live in [configs/base.yaml](configs/base.yaml):

```yaml
seed: 0
max_new_tokens: 256
m_beta: 20            # number of inverse-temperatures
p_t: 30               # number of times for the spectral form factor
beta_grid: {log_min: -2, log_max: 2}
t_grid:    {log_min: -1, log_max:  3}
```

Model registry under [configs/models/](configs/models/) — see
[configs/models/llama3_8b.yaml](configs/models/llama3_8b.yaml) and
[configs/models/phi3_medium.yaml](configs/models/phi3_medium.yaml) for the
schema (`name`, `repo`, `dtype`, `recommended_gpu_mem_gb`,
`quantization_fallback`).

---

## Datasets

| Key           | Hugging Face id                       | Task                | Label                              |
| ------------- | ------------------------------------- | ------------------- | ---------------------------------- |
| `truthfulqa`  | `truthful_qa` (`generation`, val)     | open-ended QA       | exact-match against `correct_answers` |
| `halueval`    | `pminervini/HaluEval`                 | hallucination       | gold 0/1 from dataset              |
| `triviaqa`    | `trivia_qa` (`rc.nocontext`)          | factual QA          | exact-match                        |
| `nq_open`     | `nq_open`                             | open-domain QA      | exact-match                        |
| `gsm8k`       | `gsm8k` (`main`)                      | grade-school math   | numeric match on final answer      |
| `math500`     | `HuggingFaceH4/MATH-500`              | competition math    | numeric match on final answer      |
| `fever` *(excluded from headline grid)* | `fever`           | claim verification  | gold 3-class collapsed to 0/1      |

See [src/data/loaders.py](src/data/loaders.py) for prompt templates and label
construction.

---

## Models

| Key              | Hugging Face repo                          | Params | dtype   | Min GPU mem |
| ---------------- | ------------------------------------------ | -----: | ------- | ----------: |
| `llama-3-8b`     | `meta-llama/Meta-Llama-3-8B-Instruct`      |     8B | fp16    |       18 GB |
| `llama-3.1-8b`   | `meta-llama/Llama-3.1-8B-Instruct`         |     8B | fp16    |       18 GB |
| `mistral-7b`     | `mistralai/Mistral-7B-Instruct-v0.3`       |     7B | fp16    |       16 GB |
| `qwen2.5-7b`     | `Qwen/Qwen2.5-7B-Instruct`                 |     7B | fp16    |       16 GB |
| `gemma-2-9b`     | `google/gemma-2-9b-it`                     |     9B | fp16    |       22 GB |
| `phi-3-medium`   | `microsoft/Phi-3-medium-4k-instruct`       |    14B | fp16    |       30 GB *(12 GB with `--quantization nf4`)* |

---

## Hardware and runtime

- **Tested on**: 1× NVIDIA RTX A6000 (48 GB), driver 550.54.15, CUDA 12.4,
  PyTorch 2.x+cu124, Python 3.11.
- **Per-sample cost**: ~0.4 s including forward pass; FES descriptor
  extraction itself adds < 5 ms on top of eigendecomposition
  ($\mathcal{O}(n^3)$ per layer, ~12 ms/layer/sample on CPU for $n \le 512$).
- **Full sweep wall time**: ~14 h on 1 GPU for the 6×6 grid.
- **Disk**: ~40 GB for the cached features (`features.npz` per cell).

---

## Determinism and seeds

- Global seed `0` set in [configs/base.yaml](configs/base.yaml) and
  [src/utils/seeding.py](src/utils/seeding.py).
- Dataset subsampling is seeded per-(dataset, seed) — see
  [src/data/loaders.py](src/data/loaders.py).
- Model generation uses greedy decoding (deterministic given the seed).
- 5-fold probe splits use the same global seed.

---

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `pytest` fails any test | Stop. Re-read the failing test; do not proceed until green. |
| Toy AUROC < 0.95 | Re-run with `--n-graphs 1000 --seed 1`. |
| `RuntimeError: NVIDIA driver too old (found 12040)` | You installed a `cu13.0` torch wheel. Re-run with `--index-url https://download.pytorch.org/whl/cu124`. |
| HF OOM mid-run | Reduce `--max-new-tokens 128`, then switch to `--quantization nf4`. |
| `ValueError: Feature type 'List' not found` | `datasets>=4.0` cache incompatibility. Upgrade `datasets` or clear the HF cache for that dataset. |
| `RuntimeError: Dataset scripts are no longer supported` | `datasets>=4.0` removed loader scripts (FEVER). Use the parquet mirror in `paper/appendix/F_reproducibility.tex`. |
| Gated-model 403 download | Re-accept the license on the Hugging Face web UI under your account. |
| `CUDA error: invalid device ordinal` | Use only the GPU ids returned by `torch.cuda.device_count()`. |

---