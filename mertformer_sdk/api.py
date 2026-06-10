"""Public SDK API for MertFormer Titan."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import AutoTokenizer

from config.config import cfg
from model.transformers import MertFormer


def enable_lowbit_kernels(enabled: bool = True) -> None:
    """Enable or disable low-bit kernels globally (opt-in)."""
    try:
        from layers.bitlinear import set_lowbit_kernel_enabled
        set_lowbit_kernel_enabled(enabled)
    except Exception:
        # Keep silent to avoid breaking core training paths
        return


def _resolve_device(preferred: Optional[str] = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_ckpt_path(ckpt: str | Path) -> Path:
    if isinstance(ckpt, Path):
        return ckpt
    if ckpt == "latest":
        return Path(cfg.save_dir) / f"{cfg.model_name}_latest.pt"
    return Path(ckpt)


def load_model(
    ckpt: str | Path = "latest",
    device: Optional[str] = None,
    strict_checkpoint: bool = True,
) -> Tuple[MertFormer, AutoTokenizer, str]:
    """Load model + tokenizer with optional checkpoint.

    When ``strict_checkpoint`` is True (default), missing checkpoints raise an error.
    This prevents accidental random-weight usage in commercial/pilot flows.
    """
    device = _resolve_device(device)
    ckpt_path = _resolve_ckpt_path(ckpt)

    if strict_checkpoint and not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            "Use a valid checkpoint path or pass strict_checkpoint=False for random-weight mode."
        )

    # TR: [H4 fix] Tokenizer'i checkpoint kimliginden yukle (train/eval/demo tek
    #     kaynak), modeli o vocab'a resize et. Sessiz teacher/gpt2 fallback YOK;
    #     yanlis tokenizer'la decode = orijinal "111111" bug'i. eval/gsm8k.py ile
    #     birebir ayni desen.
    # EN: [H4 fix] Load the tokenizer from the checkpoint identity (single source
    #     for train/eval/demo) and resize the model to that vocab. No silent
    #     teacher/gpt2 fallback (wrong-tokenizer decode was the original bug).
    from utils.tokenizer_resolver import load_tokenizer_from_identity, resolve_tokenizer

    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        tokenizer = load_tokenizer_from_identity(
            checkpoint.get("tokenizer_id") if isinstance(checkpoint, dict) else None
        )
        cfg.vocab_size = len(tokenizer)
        model = MertFormer().to(device)
        model.resize_token_embeddings(len(tokenizer))
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state = checkpoint["model"]
            elif "model_state_dict" in checkpoint:
                state = checkpoint["model_state_dict"]
            else:
                state = checkpoint
        else:
            state = checkpoint
        model.load_state_dict(state)
    else:
        # Random-weights smoke only (strict_checkpoint=False): no checkpoint, so
        # no recorded identity -> use the single configured resolver.
        tokenizer = resolve_tokenizer(cfg)
        cfg.vocab_size = len(tokenizer)
        model = MertFormer().to(device)
        model.resize_token_embeddings(len(tokenizer))

    model.eval()
    return model, tokenizer, device


def generate(
    model: MertFormer,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate text with sampling (cache-aware)."""
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generated_ids = input_ids
    past_kv = None

    with torch.no_grad():
        for _ in range(max_new_tokens):
            curr_input = generated_ids[:, -1:] if past_kv is not None else generated_ids
            logits, _, past_kv = model(curr_input, past_key_values=past_kv, use_cache=True)
            next_token_logits = logits[..., -1, :] / max(temperature, 1e-6)

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float("-inf")

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

            # [tier-2 LOW] guard eos_token_id is not None before comparing.
            _eos = tokenizer.eos_token_id
            if _eos is not None and next_token_id.item() == _eos:
                break

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def benchmark(
    ckpt: str | Path = "latest",
    out_dir: str | Path = "reports/benchmarks",
    samples: int = 0,
    max_new_tokens: int = 256,
    strict_checkpoint: bool = True,
) -> dict:
    """Run HumanEval/MBPP generation and return counts."""
    from scripts.benchmarks_internal import load_dataset_safe, run_generation

    model, tokenizer, device = load_model(
        ckpt=ckpt,
        strict_checkpoint=strict_checkpoint,
    )
    out_dir = Path(out_dir)

    humaneval = load_dataset_safe("openai_humaneval", "openai_humaneval")
    mbpp = load_dataset_safe("mbpp", "sanitized")

    humaneval_count = run_generation(
        humaneval,
        tokenizer,
        model,
        device,
        out_dir / "humaneval_outputs.jsonl",
        max_new_tokens,
        samples,
    )
    mbpp_count = run_generation(
        mbpp,
        tokenizer,
        model,
        device,
        out_dir / "mbpp_outputs.jsonl",
        max_new_tokens,
        samples,
    )

    return {"humaneval": humaneval_count, "mbpp": mbpp_count}
