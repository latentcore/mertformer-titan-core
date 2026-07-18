"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Copyright (c) 2026 Mert Yünlü. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 - Pre-Training
Status : PRE-TRAINING (UNVERIFIED)

Module: Core training helpers (seeding, config validation, device
selection, RNG capture/restore, LR schedule, ONNX export, freeze policy).
Split out of train/train.py as pure code motion; train/train.py re-exports
every symbol so all historical imports keep working unchanged.
==============================================================================
"""

from config.build_label import BUILD_LABEL as __version__
__author__ = "Mert Yünlü"

import math
import random
from pathlib import Path
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.onnx
from torch.optim.lr_scheduler import LambdaLR

from config.config import cfg


# -----------------------------------------------------------------------------
# CORE TRAINING HELPERS (seeding, config checks, devices, RNG, schedule, export)
# -----------------------------------------------------------------------------


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # [PRO] Reproducibility Mode
    if getattr(cfg, "deterministic", False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            print("🔒 Deterministic Mode: ACTIVATED (Performance may slightly decrease)")
        except AttributeError:
            pass # Older pytorch versions


def validate_config(cfg: Any, stage: str = "pre") -> None:
    """Sanity check for critical config parameters (Staged)."""
    required = ["hidden_size", "num_layers", "max_seq_len", "model_name", "learning_rate"]

    # [PRO] Post-Teacher Validation (requires vocab_size)
    if stage == "post":
        required.append("vocab_size")

    for k in required:
        if not hasattr(cfg, k):
            raise ValueError(f"❌ CRITICAL CONFIG ERROR: Missing key '{k}'")

    if cfg.learning_rate <= 0: raise ValueError("Learning Rate must be > 0")
    if getattr(cfg, "num_experts", 0) < 1:
        # [PRO] Explicitly validate MoE config if use_moe flag exists, or just warn/error
        if getattr(cfg, "use_moe", False):
             raise ValueError("❌ CRITICAL: MoE enabled (use_moe=True) but num_experts < 1")
    if cfg.max_seq_len > 100000: print("⚠️ Warning: Unusual max_seq_len (>100k)")

    print(f"✅ Config Schema Verified ({stage}).")


def count_jsonl_records(path: Path) -> int:
    """
    Returns number of non-empty lines in a JSONL file.
    """
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def get_gpu_memory_usage(device: Optional[int] = None) -> tuple[float, float]:
    """
    Returns GPU memory usage (allocated/reserved) in GB.
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0
    # Allocated/Reserved memory report
    idx = device if device is not None else torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(idx) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(idx) / (1024 ** 3)
    return allocated, reserved


def get_curriculum_contract() -> tuple[List[str], List[float]]:
    """
    Portable training contract:
    - stage names and ratios come from config single source of truth.
    """
    names = list(getattr(cfg, "curriculum_stage_names", []))
    ratios = [float(x) for x in list(getattr(cfg, "curriculum_stage_ratios", []))]
    if len(names) != len(ratios):
        raise ValueError(
            f"Curriculum mismatch: stage names ({len(names)}) != stage ratios ({len(ratios)})."
        )
    if len(names) != 5:
        raise ValueError(f"Expected 5 curriculum stages, got {len(names)}.")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Curriculum ratios must sum to 1.0, got {sum(ratios):.8f}.")
    return names, ratios


def build_stage_boundaries(max_steps: int, stage_ratios: List[float]) -> List[int]:
    boundaries: List[int] = []
    cumulative = 0.0
    for ratio in stage_ratios[:-1]:
        cumulative += ratio
        boundaries.append(int(max_steps * cumulative))
    return boundaries


def get_student_device(accelerator: Any = None) -> torch.device:
    if accelerator:
        return accelerator.device
    return torch.device(cfg.device)


def get_teacher_device(accelerator: Any = None) -> torch.device:
    # Accelerate will handle device placement if initialized
    if accelerator:
        return accelerator.device
    if torch.cuda.is_available() and cfg.device == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def preflight_param_report(model: nn.Module) -> None:
    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Model Parametreleri: {n_params / 1e6:.2f} Milyon")
    print(f"📚 Trainable: {trainable / 1e6:.2f} Milyon")


def _capture_rng_state() -> dict:
    """[MED] Capture RNG states so resume is reproducible."""
    rng: dict = {}
    try:
        rng["torch"] = torch.get_rng_state()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
    except Exception:
        pass
    try:
        import numpy as _np
        st: Any = _np.random.get_state()
        # Convert numpy state to builtins so checkpoints load under weights_only=True.
        rng["numpy"] = [st[0], [int(v) for v in st[1]], int(st[2]), int(st[3]), float(st[4])]
    except Exception:
        pass
    try:
        # random.getstate() -> (int, tuple[int,...], float|None): builtin-safe.
        ver, state_tuple, gauss = random.getstate()
        rng["python"] = [int(ver), [int(v) for v in state_tuple], gauss]
    except Exception:
        pass
    return rng


def _restore_rng_state(rng: Any) -> None:
    """[MED] Restore RNG states from the checkpoint (best-effort)."""
    if not isinstance(rng, dict):
        return
    try:
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
    except Exception:
        pass
    try:
        if "cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    except Exception:
        pass
    try:
        if "numpy" in rng:
            import numpy as _np
            s = rng["numpy"]
            state = (s[0], _np.array(s[1], dtype=_np.uint32), int(s[2]), int(s[3]), float(s[4]))
            _np.random.set_state(state)
    except Exception:
        pass
    try:
        if "python" in rng:
            p = rng["python"]
            random.setstate((int(p[0]), tuple(int(v) for v in p[1]), p[2]))
    except Exception:
        pass


def _normalize_state_dict_keys_for_model(model_state: dict, model: nn.Module) -> dict:
    """
    Normalize checkpoint key prefixes when torch.compile wrappers differ between save/load.
    """
    if not model_state:
        return model_state
    model_keys = list(model.state_dict().keys())
    if not model_keys:
        return model_state

    ckpt_has_orig = any(k.startswith("_orig_mod.") for k in model_state.keys())
    model_has_orig = any(k.startswith("_orig_mod.") for k in model_keys)

    if ckpt_has_orig and not model_has_orig:
        return {
            (k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k): v
            for k, v in model_state.items()
        }
    if (not ckpt_has_orig) and model_has_orig:
        return {f"_orig_mod.{k}": v for k, v in model_state.items()}
    return model_state


def _infer_curriculum_stage_from_step(step: int, stage_boundaries: List[int]) -> int:
    if step >= stage_boundaries[3]:
        return 5
    if step >= stage_boundaries[2]:
        return 4
    if step >= stage_boundaries[1]:
        return 3
    if step >= stage_boundaries[0]:
        return 2
    return 1


# -----------------------------------------------------------------------------
# INFERENCE WRAPPER (ONNX CLEANER)
# -----------------------------------------------------------------------------


class MertFormerInferenceWrapper(nn.Module):
    """
    Lightweight model wrapper for inference only. Discards aux loss.
    """
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Swallow aux loss, return only logits
        logits, _, _ = self.model(input_ids)
        return logits


def export_to_onnx(model: nn.Module, save_dir: Path, model_name: str, device: torch.device) -> None:
    print(f"\n📦 ONNX DÖNÜŞÜMÜ BAŞLIYOR (CLEAN INFERENCE MODE)...")
    onnx_path = save_dir / f"{model_name}.onnx"

    # Wrap with inference wrapper (strips aux loss and complexity)
    inference_model = MertFormerInferenceWrapper(model)
    inference_model.eval()

    # [OPT] Use small dummy input (dynamic axes handle variable lengths)
    dummy_len = min(cfg.max_seq_len, 32)
    dummy_input = torch.randint(0, cfg.vocab_size, (1, dummy_len)).to(device)

    try:
        exported = torch.onnx.export(
            inference_model,
            dummy_input,  # type: ignore[arg-type]  # legacy export API accepts a bare Tensor
            str(onnx_path),
            export_params=True,
            opset_version=17, # opset 17: QDQ/BitNet export needs 13+ (12 was anachronistic)
            do_constant_folding=False, # [FIX] Disable folding to avoid graph capture errors
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        if hasattr(exported, "save"):
            exported.save(str(onnx_path))  # type: ignore[union-attr]  # guarded by hasattr above
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX export returned success but file was not materialized: {onnx_path}"
            )
        print(f"✅ ONNX BAŞARIYLA KAYDEDİLDİ: {onnx_path}")
    except Exception as e:
        print(f"❌ ONNX DÖNÜŞÜM HATASI: {e}")


# WSD scheduler (Warmup-Stable-Decay) for grokking.


def get_wsd_schedule(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
    stable_ratio: float = 0.8,
) -> LambdaLR:
    """Warm up, hold stable for X%, then decay sharply."""
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        stable_steps = int(num_training_steps * stable_ratio)
        if current_step < stable_steps:
            return 1.0 # Stable Phase

        # Decay Phase (Cosine from 1.0 to min_lr_ratio)
        decay_steps = num_training_steps - stable_steps
        progress = float(current_step - stable_steps) / float(max(1, decay_steps))
        # [2026-07-08] Clamp progress to [0, 1] before the cosine. In a normal single
        # run current_step never exceeds num_training_steps, but if TITAN_MAX_STEPS
        # changes between a checkpoint-saving run and a later resume, the restored
        # `last_epoch` is reinterpreted against a different num_training_steps closure
        # and progress can exceed 1 — at which point cos() turns back upward and the LR
        # RISES again instead of staying floored at min_lr_ratio. Behavior-preserving
        # in the normal case.
        progress = min(1.0, max(0.0, progress))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def get_rewarmup_schedule(
    optimizer: torch.optim.Optimizer,
    num_rewarmup_steps: int,
    num_training_steps: int,
    start_lr_ratio: float = 0.01,
    peak_lr_ratio: float = 1.0,
    min_lr_ratio: float = 0.1,
    stable_ratio: float = 0.8,
) -> LambdaLR:
    """Re-warmup schedule for POST-45K continuation training (SFT / DMSR ablation /
    additional pre-training) resuming from a checkpoint whose base-run LR had already
    decayed to its floor via get_wsd_schedule's min_lr_ratio. NOT used by the
    canonical/frozen 45K run itself -- see config.use_rewarmup_schedule (default off,
    BACKLOG "LR re-warmup" item).

    Resuming the base scheduler's own state at that point keeps the LR pinned at the
    floor for the entire continuation run, since current_step never re-enters the
    warmup or stable phase. This schedule re-anchors to step 0 for the continuation
    run: ramps linearly from start_lr_ratio (the base run's landing floor) up to
    peak_lr_ratio over num_rewarmup_steps, holds stable for stable_ratio of the
    remaining steps, then cosine-decays back down to min_lr_ratio -- the same WSD
    shape as get_wsd_schedule, just re-anchored and with a non-zero cold-start floor.
    """
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step):
        if current_step < num_rewarmup_steps:
            progress = float(current_step) / float(max(1, num_rewarmup_steps))
            return start_lr_ratio + (peak_lr_ratio - start_lr_ratio) * progress

        remaining_steps = max(1, num_training_steps - num_rewarmup_steps)
        stable_steps = num_rewarmup_steps + int(remaining_steps * stable_ratio)
        if current_step < stable_steps:
            return peak_lr_ratio

        # Decay phase (cosine from peak_lr_ratio down to min_lr_ratio). Same
        # progress-clamp discipline as get_wsd_schedule's [2026-07-08] fix, so a
        # changed num_training_steps between save and resume can't send the LR back up.
        decay_steps = num_training_steps - stable_steps
        progress = float(current_step - stable_steps) / float(max(1, decay_steps))
        progress = min(1.0, max(0.0, progress))
        cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, min_lr_ratio + (peak_lr_ratio - min_lr_ratio) * cosine_val)

    return LambdaLR(optimizer, lr_lambda)


# -----------------------------------------------------------------------------
# FREEZE SUPPORT
# -----------------------------------------------------------------------------


def apply_freeze_policy(model: nn.Module, freeze_core_layers: bool) -> None:
    """Freeze core layers, keep MoE Router and Liquid Layers trainable."""
    if not freeze_core_layers:
        return

    print("🔒 Applying freeze policy: Core layers frozen, MoE/Liquid trainable")

    for name, param in model.named_parameters():
        # Freeze everything by default
        param.requires_grad = False

        # Unfreeze MoE router and shared expert
        if 'router' in name or 'shared_expert' in name or 'shared_gate' in name:
            param.requires_grad = True

        # Unfreeze Liquid layers
        if 'liquid' in name.lower() or 'tau' in name:
            param.requires_grad = True

        # Unfreeze LM head and embeddings (always trainable)
        if 'lm_head' in name or 'tok_embeddings' in name:
            param.requires_grad = True

        # [TITAN FIX]: LayerNorms should always be trainable
        if 'norm' in name.lower():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"📊 Freeze Status: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M parameters trainable")
