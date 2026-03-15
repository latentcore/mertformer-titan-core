from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_mathfp_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("mathfp", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_checkpoint(user_path: str | None) -> Path:
    if user_path:
        return Path(user_path).expanduser().resolve()

    candidates = [
        Path("~/Downloads/content/mertformer_outputs/runs/run_20260315_050133/run_20260315_050133_model_final.pt").expanduser(),
        Path("~/Downloads/content/mertformer_outputs/checkpoints/kaggle_onefile_build30/mathfp_our_mertformer_step_002400.pt").expanduser(),
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    runs_dir = Path("~/Downloads/content/mertformer_outputs/runs").expanduser()
    if runs_dir.exists():
        run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for run in run_dirs:
            if not run.is_dir():
                continue
            model_files = list(run.glob("*_model_final.pt"))
            if model_files:
                return model_files[0].resolve()

    raise FileNotFoundError("No checkpoint found. Provide --ckpt path.")


def normalize_prompt(raw: str) -> str:
    text = raw.strip()
    if not text:
        return ""
    if "=" in text:
        return text.rstrip() + (" " if not text.rstrip().endswith(" ") else "")
    if any(op in text for op in ["+", "-", "*", "/"]):
        return text + " = "
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description="Math-fastproof interactive chat (until 'q').")
    ap.add_argument("--ckpt", help="Checkpoint .pt path (optional)")
    ap.add_argument("--device", choices=["cpu", "cuda", "mps"], help="Force device")
    args = ap.parse_args()

    ckpt_path = resolve_checkpoint(args.ckpt)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return 2

    device = args.device or detect_device()
    print(f"[mathfp] loading ckpt: {ckpt_path}")
    print(f"[mathfp] device: {device}")

    mathfp_script = Path(__file__).resolve().parent / "kaggle_onefile_demo_build30_colab_math_fastproof.py"
    mod = load_mathfp_module(mathfp_script)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = dict(ckpt.get("config", {}))
    state = ckpt.get("tokenizer_state", {})
    tokenizer = mod.SimpleTokenizer(vocab_size=int(state.get("target_vocab_size", cfg.get("vocab_size", 2048))))
    tokenizer.load_state_dict(state)
    vocab_size = max(int(getattr(tokenizer, "vocab_size_realized", 0)), 128)
    cfg.setdefault("vocab_size", int(vocab_size))
    cfg.setdefault("seq_len", 64)

    models = mod.mathfp_build_variant_models(cfg, vocab_size=vocab_size)
    model = models["our_mertformer"]
    model.load_state_dict(ckpt.get("model", {}), strict=False)
    model.to(device)
    model.eval()

    allowed_ids = mod._mathfp_allowed_answer_token_ids(tokenizer)

    print("Hazırım. Matematik sor ve Enter. Çıkmak için `q` yaz.")
    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nÇıkış.")
            break
        if not raw:
            continue
        if raw.lower() in {"q", "quit", "exit"}:
            print("Çıkış.")
            break
        prompt = normalize_prompt(raw)
        if not prompt:
            continue
        pred_text = mod.mathfp_generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=12,
            allowed_token_ids=allowed_ids,
        )
        pred = mod._mathfp_first_int(pred_text)
        if pred is None:
            print(pred_text)
        else:
            print(pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
