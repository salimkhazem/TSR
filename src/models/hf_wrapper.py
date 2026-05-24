"""Wrap a Hugging Face causal LM and extract per-layer post-softmax attention.

We deliberately use `attn_implementation="eager"` so that `output_attentions=True`
returns full [n_heads, seq, seq] attention tensors. SDPA / flash kernels do not
expose the post-softmax weights.

The wrapper supports:
  - Llama-3 / Llama-3.1
  - Mistral 7B
  - Qwen2.5
  - Phi-3 (medium-4k)
  - Gemma-2

For each generated response we return, optionally:
  - per-layer attention tensors aggregated over heads (mean pool)
  - per-layer per-head attention tensors (full, expensive)
  - the generated token ids and the model's per-token log-probs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Model registry (logical-name -> HF repo id, plus per-arch quirks)
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "llama-3-8b":     {"repo": "meta-llama/Meta-Llama-3-8B-Instruct",     "dtype": "float16"},
    "llama-3.1-8b":   {"repo": "meta-llama/Llama-3.1-8B-Instruct",        "dtype": "float16"},
    "mistral-7b":     {"repo": "mistralai/Mistral-7B-Instruct-v0.3",      "dtype": "float16"},
    "qwen2.5-7b":     {"repo": "Qwen/Qwen2.5-7B-Instruct",                "dtype": "float16"},
    "phi-3-medium":   {"repo": "microsoft/Phi-3-medium-4k-instruct",       "dtype": "float16"},
    "gemma-2-9b":     {"repo": "google/gemma-2-9b-it",                     "dtype": "bfloat16"},
}


@dataclass
class ModelOutput:
    """A single forward+generate result."""

    prompt: str
    response: str
    prompt_ids: List[int]
    response_ids: List[int]
    # attentions_per_layer[ell] has shape:
    #   [n_heads, seq, seq]  if keep_heads=True
    #   [seq, seq]           if keep_heads=False (mean pool over heads)
    attentions_per_layer: List[np.ndarray] = field(default_factory=list)
    # Per-token logprob of the chosen response token (for MSP / perplexity baselines).
    response_logprobs: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class HFAttentionExtractor:
    """Loads an HF causal LM and extracts attention during greedy generation.

    Designed to be a thin, stateful, single-GPU wrapper. Larger orchestration
    (multiple models, batching across GPUs) belongs in `src/run.py`.
    """

    def __init__(
        self,
        model_key: str,
        device: str = "cuda:0",
        dtype: Optional[str] = None,
        quantization: Optional[str] = None,
        max_new_tokens: int = 256,
        trust_remote_code: bool = True,
    ) -> None:
        if model_key not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model {model_key!r}; known: {sorted(MODEL_REGISTRY)}")
        self.model_key = model_key
        self.repo = MODEL_REGISTRY[model_key]["repo"]
        self.device = device
        self.dtype = dtype or MODEL_REGISTRY[model_key]["dtype"]
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.trust_remote_code = trust_remote_code
        self._tokenizer = None
        self._model = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        torch_dtype = dtype_map[self.dtype]

        # transformers >= 4.46 renamed torch_dtype -> dtype; fall back if old.
        import transformers
        _new_dtype_kw = tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 46)
        dtype_kw = "dtype" if _new_dtype_kw else "torch_dtype"

        kwargs: Dict[str, Any] = {
            dtype_kw: torch_dtype,
            "attn_implementation": "eager",
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
        }

        if self.quantization == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = {"": self.device}
        elif self.quantization == "int8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = {"": self.device}

        self._tokenizer = AutoTokenizer.from_pretrained(self.repo, trust_remote_code=self.trust_remote_code)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(self.repo, **kwargs)
        if self.quantization is None:
            self._model.to(self.device)
        self._model.eval()

    def unload(self) -> None:
        import gc
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _format_chat(self, prompt: str) -> str:
        """Apply the model's chat template if it has one; else return verbatim."""
        tok = self._tokenizer
        if hasattr(tok, "apply_chat_template") and tok.chat_template:
            return tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    @staticmethod
    def _attns_to_np_safe(attentions: Sequence[Any], keep_heads: bool) -> List[np.ndarray]:
        import torch
        out: List[np.ndarray] = []
        for layer in attentions:
            a = layer[0].detach().float().cpu().numpy()  # [H, S, S], float32
            if not keep_heads:
                a = a.mean(axis=0)
            out.append(a.astype(np.float64))
        return out

    def run(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        keep_heads: bool = False,
        return_attentions_on: str = "prompt",
    ) -> ModelOutput:
        """Generate a response and return (optionally) per-layer attention matrices.

        Args:
            prompt: user prompt (will be wrapped with the model's chat template).
            max_new_tokens: override the constructor's value.
            keep_heads: keep per-head attention (else mean over heads).
            return_attentions_on: 'prompt' -> attentions over the prompt only
                (single forward pass, much cheaper); 'full' -> attentions over the
                full generated sequence (one forward pass on the concatenated
                tokens; expensive for long generations).

        Returns:
            ModelOutput.
        """
        if self._model is None:
            self.load()

        import torch

        tok = self._tokenizer
        model = self._model

        text = self._format_chat(prompt)
        inputs = tok(text, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        max_new = max_new_tokens or self.max_new_tokens
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tok.pad_token_id,
            )
        seq = gen.sequences[0]  # [prompt + new]
        response_ids = seq[prompt_len:].tolist()
        response_text = tok.decode(response_ids, skip_special_tokens=True)

        # Per-token response logprobs
        scores = gen.scores  # tuple of [B, V] over generated steps
        logprobs = []
        for step, score in enumerate(scores):
            if step >= len(response_ids):
                break
            logp = torch.log_softmax(score[0], dim=-1)
            logprobs.append(float(logp[response_ids[step]].item()))
        response_logprobs = np.asarray(logprobs, dtype=np.float64)

        attentions: List[np.ndarray] = []
        if return_attentions_on == "prompt":
            with torch.no_grad():
                out = model(**inputs, output_attentions=True, use_cache=False)
            attentions = self._attns_to_np_safe(out.attentions, keep_heads)
        elif return_attentions_on == "full":
            full_ids = seq.unsqueeze(0)
            attn_mask = torch.ones_like(full_ids)
            with torch.no_grad():
                out = model(
                    input_ids=full_ids,
                    attention_mask=attn_mask,
                    output_attentions=True,
                    use_cache=False,
                )
            attentions = self._attns_to_np_safe(out.attentions, keep_heads)
        elif return_attentions_on == "none":
            attentions = []
        else:
            raise ValueError(f"unknown return_attentions_on: {return_attentions_on}")

        return ModelOutput(
            prompt=prompt,
            response=response_text,
            prompt_ids=inputs["input_ids"][0].tolist(),
            response_ids=response_ids,
            attentions_per_layer=attentions,
            response_logprobs=response_logprobs,
            meta={"model": self.model_key, "prompt_len": prompt_len},
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def iter_run(
        self,
        prompts: Iterable[str],
        **kwargs,
    ) -> Iterable[ModelOutput]:
        for p in prompts:
            yield self.run(p, **kwargs)
