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


def load_model(
    ckpt: str | Path = "latest",
    device: Optional[str] = None,
) -> Tuple[MertFormer, AutoTokenizer, str]:
    """Load model + tokenizer with optional checkpoint."""
    device = _resolve_device(device)
    model = MertFormer().to(device)
    model.eval()

    ckpt_path: Optional[Path] = None
    if isinstance(ckpt, Path):
        ckpt_path = ckpt
    elif ckpt == "latest":
        ckpt_path = Path(cfg.save_dir) / f"{cfg.model_name}_latest.pt"
    else:
        ckpt_path = Path(ckpt)

    if ckpt_path and ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        state = checkpoint.get("model", checkpoint)
        model.load_state_dict(state)
    # else: keep random weights; caller can decide

    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

            if next_token_id.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def benchmark(
    ckpt: str | Path = "latest",
    out_dir: str | Path = "reports/benchmarks",
    samples: int = 0,
    max_new_tokens: int = 256,
) -> dict:
    """Run HumanEval/MBPP generation and return counts."""
    from scripts.benchmarks_internal import load_dataset_safe, run_generation

    model, tokenizer, device = load_model(ckpt=ckpt)
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
