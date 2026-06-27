"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Copyright (c) 2026 Mert Yünlü. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 - Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

__version__ = "1.0"
__author__ = "Mert Yünlü"

import glob
import json
import math
import os
import random
import sys
import time
import shutil
import psutil
from pathlib import Path
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


try:
    import wandb
except ImportError:
    wandb = None
try:
    import galore_torch
except ImportError:
    galore_torch = None
    
# Import DistillationManager
sys.path.append(str(Path(__file__).parent.parent))
from orchestrator.distillation_manager import DistillationManager
try:
    from train.continual_adapter import ContinualLearningAdapter
except Exception:
    ContinualLearningAdapter = None

# -----------------------------------------------------------------------------
# 0. PROJECT ROOT DETECTION
# -----------------------------------------------------------------------------
current_file = Path(__file__).resolve()
project_root = current_file.parent
for _ in range(4):
    if (project_root / "config").exists():
        break
    project_root = project_root.parent

sys.path.insert(0, str(project_root))
print(f"📍 PROJE ANA MERKEZİ: {project_root}")

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
try:
    from config.config import cfg
    from model.transformers import MertFormer
    from orchestrator.telemetry import system_snapshot
    from utils.logger import RunLogger
    from utils.liquid_safeguard import update_liquid_spike_state
    from utils.tokenizer_resolver import resolve_tokenizer, tokenizer_identity
except ImportError as e:
    print(f"❌ KRİTİK IMPORT HATASI: {e}")
    sys.exit(1)

try:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
except Exception:
    print("❌ Transformers kütüphanesi eksik! 'pip install transformers' yap.")
    sys.exit(1)

try:
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration, set_seed, DistributedDataParallelKwargs
except ImportError:
    print("❌ Accelerate kütüphanesi eksik! 'pip install accelerate' yap.")
    sys.exit(1)

# Identity of the tokenizer used for training; stamped into every checkpoint
# so eval/demo reload the exact same tokenizer (no train/eval mismatch).
RUNTIME_TOKENIZER_ID: Optional[dict] = None

# -----------------------------------------------------------------------------
# 0.5. SAVE_DIR FIX
# -----------------------------------------------------------------------------
if not hasattr(cfg, "save_dir"):
    cfg.save_dir = "checkpoints/mertformer_titan_prod"

# -----------------------------------------------------------------------------
# 0.6. SPLIT TRAINER MODULES (pure re-exports for backwards compatibility)
# -----------------------------------------------------------------------------
# train/train.py remains the canonical entry point and public import surface.
# Helper groups now live in train/trainer_core.py (core helpers),
# train/trainer_data.py (datasets/collate/teacher payloads) and
# train/trainer_eval.py (KD losses / metric reading). The explicit imports
# below keep every historical `from train.train import X` and
# `train.train.<name>` access working unchanged.
from train.trainer_core import (
    MertFormerInferenceWrapper,
    _capture_rng_state,
    _infer_curriculum_stage_from_step,
    _normalize_state_dict_keys_for_model,
    _restore_rng_state,
    apply_freeze_policy,
    build_stage_boundaries,
    count_jsonl_records,
    export_to_onnx,
    get_curriculum_contract,
    get_gpu_memory_usage,
    get_student_device,
    get_teacher_device,
    get_wsd_schedule,
    preflight_param_report,
    seed_all,
    validate_config,
)
from train.trainer_data import (
    CurriculumDataset,
    PrecomputedCurriculumDataset,
    ValidationJsonlDataset,
    _align_sparse_topk_payload,
    _encode_with_eos_labels,
    _is_sparse_topk_payload,
    _shift_teacher_payload,
    _stack_teacher_payloads,
    _teacher_payload_to_device,
    collate_fn,
)
from train.trainer_eval import (
    _kd_loss_sparse_topk,
    kd_loss_safe,
    read_metric_from_json,
)



# -----------------------------------------------------------------------------
# 1. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def check_disk_space(min_gb: float = 10.0, path: Optional[Path] = None) -> bool:
    """
    Checks if there is enough free disk space.
    """
    target = path or project_root
    try:
        # Disk free space measurement
        total, used, free = shutil.disk_usage(str(target))
        free_gb = free / (1024 ** 3)
        return free_gb >= float(min_gb)
    except Exception:
        return True  # Fail-open to avoid breaking training


def write_energy_telemetry_baseline(project_root: Path, stage: str = "bootstrap") -> None:
    """
    Records baseline system metrics before/during training.
    """
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    vm = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.1)
    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": stage,
        "cpu_percent": float(cpu),
        "ram_total_gb": float(vm.total / (1024 ** 3)),
        "ram_used_gb": float((vm.total - vm.available) / (1024 ** 3)),
        "device": str(cfg.device),
        "mixed_precision": "bf16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else ("fp16" if cfg.use_amp else "no"),
    }

    with (reports_dir / "system_stats.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    energy_baseline = {
        "generated_utc": payload["timestamp_utc"],
        "mode": "baseline",
        "cpu_percent": payload["cpu_percent"],
        "ram_used_gb": payload["ram_used_gb"],
        "device": payload["device"],
    }
    latency_baseline = {
        "generated_utc": payload["timestamp_utc"],
        "step": "pretrain_bootstrap",
        "note": "latency baseline placeholder; full device benchmarks in reports/bench_*.json",
    }
    thermal_baseline = {
        "generated_utc": payload["timestamp_utc"],
        "note": "host thermal API not available in pure Python cross-platform mode",
        "status": "measured_via_system_proxy",
    }

    (reports_dir / "energy_baseline.json").write_text(json.dumps(energy_baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports_dir / "latency_baseline.json").write_text(json.dumps(latency_baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports_dir / "thermal_baseline.json").write_text(json.dumps(thermal_baseline, ensure_ascii=False, indent=2), encoding="utf-8")


def write_training_runtime_manifest(payload: dict) -> None:
    out = project_root / "reports" / "training_runtime_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_checkpoint_smart(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    cfg: Any,
    keep_last_n: int = 3,
    val_loss: Optional[float] = None,
    best_val_loss: Optional[float] = None,
    write_final: bool = False,
) -> Optional[float]:
    """
    Smart checkpoint saver - keeps only last N checkpoints + best checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        step: Current training step
        cfg: Configuration object
        keep_last_n: Number of recent checkpoints to keep
        val_loss: Current validation loss (optional)
        best_val_loss: Best validation loss seen so far (optional)
        write_final: Also write the canonical final checkpoint alias
    
    Returns:
        float: Updated best_val_loss (if val_loss provided)
    """
    save_dir = project_root / cfg.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = f"{cfg.model_name}_step_{step}.pt"
    latest_name = f"{cfg.model_name}_latest.pt"
    best_name = f"{cfg.model_name}_best.pt"
    final_name = f"{cfg.model_name}_final.pt"

    save_path = save_dir / ckpt_name
    latest_path = save_dir / latest_name
    best_path = save_dir / best_name
    final_path = save_dir / final_name

    print(f"💾 Checkpoint Kaydediliyor: {ckpt_name}")

    checkpoint_best_val_loss = best_val_loss
    if val_loss is not None and (
        checkpoint_best_val_loss is None or val_loss < checkpoint_best_val_loss
    ):
        checkpoint_best_val_loss = val_loss

    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step,
        'config': str(cfg),
        # Record tokenizer identity so eval/demo reload the identical tokenizer.
        'tokenizer_id': RUNTIME_TOKENIZER_ID,
        # [H3] Record optimizer class so resume can warn on a class mismatch.
        'optimizer_class': type(optimizer).__name__,
        # [MED] RNG states for reproducible resume.
        'rng_state': _capture_rng_state(),
        'best_val_loss': (
            float(checkpoint_best_val_loss)
            if checkpoint_best_val_loss is not None
            and math.isfinite(float(checkpoint_best_val_loss))
            else None
        ),
    }
    
    # Add validation loss to state if provided
    if val_loss is not None:
        state['val_loss'] = val_loss
    if write_final:
        state['is_final'] = True

    # Save regular checkpoint atomically: write to a temp file then os.replace into
    # place. A mid-write process kill (exactly the provider failure we already hit)
    # can otherwise leave a half-written *.pt that crashes the next resume; os.replace
    # is atomic on the same filesystem, so a reader always sees a complete file.
    def _atomic_torch_save(_state, _dst):
        _tmp = f"{_dst}.tmp"
        torch.save(_state, _tmp)
        os.replace(_tmp, _dst)
    _atomic_torch_save(state, save_path)
    _atomic_torch_save(state, latest_path)
    if write_final:
        _atomic_torch_save(state, final_path)
        print(f"🏁 Final Checkpoint Kaydedildi: {final_name}")
    
    # Save best checkpoint if this is the best so far
    if val_loss is not None and best_val_loss is not None:
        if val_loss < best_val_loss:
            print(f"🏆 NEW BEST! Val Loss: {val_loss:.4f} (Previous: {best_val_loss:.4f})")
            _atomic_torch_save(state, best_path)
            best_val_loss = val_loss
        else:
            print(f"📊 Val Loss: {val_loss:.4f} (Best: {best_val_loss:.4f})")

    # Cleanup old checkpoints (but keep best.pt)
    search_pattern = str(save_dir / f"{cfg.model_name}_step_*.pt")
    all_ckpts = sorted(glob.glob(search_pattern), key=os.path.getmtime)

    if len(all_ckpts) > keep_last_n:
        files_to_delete = all_ckpts[:-keep_last_n]
        for f in files_to_delete:
            try:
                os.remove(f)
                print(f"🧹 Eski Checkpoint Silindi: {os.path.basename(f)}")
            except OSError as e:
                print(f"⚠️ Silme Hatası: {e}")
    
    return best_val_loss


def _discover_resume_checkpoint(cfg: Any) -> Optional[Path]:
    """
    Find resume checkpoint.
    Priority:
      1) TITAN_RESUME_FROM
      2) <save_dir>/<model_name>_latest.pt
      3) newest *_latest.pt under save_dir
    """
    explicit = os.getenv("TITAN_RESUME_FROM", "").strip()
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"TITAN_RESUME_FROM not found: {p}")
        return p

    save_dir = project_root / cfg.save_dir
    if not save_dir.exists():
        return None

    candidates: list[Path] = []
    model_name = str(getattr(cfg, "model_name", "")).strip()
    if model_name:
        candidates.append(save_dir / f"{model_name}_latest.pt")
    candidates.extend(
        sorted(save_dir.glob("*_latest.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    )

    seen: set[str] = set()
    for path in candidates:
        path_s = str(path.resolve())
        if path.exists() and path_s not in seen:
            seen.add(path_s)
            return path
    return None


def _partial_resume_allowed() -> bool:
    return os.getenv("TITAN_RESUME_ALLOW_PARTIAL", "0").strip() == "1"


def _enforce_resume_key_policy(
    missing: list[str],
    unexpected: list[str],
    ckpt_path: Path,
    is_main_process: bool,
) -> None:
    if not missing and not unexpected:
        return

    detail = (
        f"Resume checkpoint key mismatch for {ckpt_path}: "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )
    if _partial_resume_allowed():
        if is_main_process:
            print(
                "⚠️  Partial resume explicitly allowed "
                "(TITAN_RESUME_ALLOW_PARTIAL=1): "
                f"{detail}"
            )
        return

    missing_preview = ", ".join(missing[:8]) if missing else "none"
    unexpected_preview = ", ".join(unexpected[:8]) if unexpected else "none"
    raise RuntimeError(
        f"{detail}. Default closure policy requires exact model-state compatibility. "
        f"Missing preview: {missing_preview}. Unexpected preview: {unexpected_preview}. "
        "Set TITAN_RESUME_ALLOW_PARTIAL=1 only for an explicit exploratory migration."
    )


def _load_resume_payload(cfg: Any, model: nn.Module, is_main_process: bool = True) -> Optional[dict]:
    """
    Load model state for resume before optimizer/scheduler wiring.
    """
    auto_resume = os.getenv("TITAN_AUTO_RESUME", "1").strip() == "1"
    if not auto_resume:
        if is_main_process:
            print("ℹ️  Auto-resume disabled (TITAN_AUTO_RESUME=0).")
        return None

    ckpt_path = _discover_resume_checkpoint(cfg)
    if ckpt_path is None:
        if is_main_process:
            print("ℹ️  Auto-resume enabled, no checkpoint found. Starting fresh.")
        return None

    # weights_only=False: this is our OWN trusted checkpoint, and its optimizer
    # state (GaLoreAdamW8bit / bitsandbytes AdamW8bit) carries non-tensor objects
    # that the weights_only=True default (torch >= 2.6) refuses to unpickle, which
    # would make the entire resume crash — i.e. break the crash-recovery path we
    # depend on after a provider kill. Explicit, not relying on the torch default.
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "model" not in state:
        raise RuntimeError(f"Invalid checkpoint format: {ckpt_path}")

    model_state = _normalize_state_dict_keys_for_model(state["model"], model)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    missing = list(missing)
    unexpected = list(unexpected)
    _enforce_resume_key_policy(missing, unexpected, ckpt_path, is_main_process)

    step = int(state.get("step", 0))
    val_loss = state.get("val_loss")
    best_val_loss = state.get("best_val_loss")

    if is_main_process:
        print(f"♻️  Auto-resume checkpoint loaded: {ckpt_path}")
        print(f"   - resume_step: {step}")
        if val_loss is not None:
            print(f"   - resume_val_loss: {float(val_loss):.6f}")
        if best_val_loss is not None:
            print(f"   - resume_best_val_loss: {float(best_val_loss):.6f}")
        print(f"   - model missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")

    return {
        "checkpoint_path": str(ckpt_path),
        "state": state,
        "step": step,
        "val_loss": float(val_loss) if val_loss is not None else None,
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


# -----------------------------------------------------------------------------
# 5. TEACHER & MODEL SETUP
# -----------------------------------------------------------------------------
class TeacherBundle:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = get_teacher_device()
        self.require_gated_teacher = bool(getattr(cfg, "require_gated_teacher", False))
        hf_token = os.environ.get("HF_TOKEN")

        print(f"👨‍🏫 Teacher Hazırlanıyor: {cfg.teacher_model_id}")
        try:
            prefer_local_tokenizer = (
                bool(getattr(cfg, "use_tr_tokenizer", False))
                or not self.require_gated_teacher
                or os.environ.get("TITAN_OFFLINE", "1") == "1"
            )
            self.tokenizer = load_teacher_tokenizer(prefer_local=prefer_local_tokenizer)

            if cfg.distill_alpha > 0.0:
                print(f"🔄 Teacher ({cfg.teacher_model_id}) Loading...")
                # DDP FIX: Avoid device_map="auto" in DDP.
                # Accelerate handles student, but teacher is static.
                # In DDP, each process should load teacher to its own device (or CPU offload).
                # device_map="auto" tries to use all GPUs, causing conflict in DDP.
                teacher_device = (
                    f"cuda:{self.device.index if self.device.index is not None else torch.cuda.current_device()}"
                    if self.device.type == "cuda"
                    else "cpu"
                )
                teacher_load_kwargs = {
                    "token": hf_token,
                    "torch_dtype": torch.float16,
                    # [FIX] Use explicit string map for HF compatibility across versions/backends.
                    "device_map": {"": teacher_device},
                }
                try:
                    import bitsandbytes  # noqa: F401

                    teacher_load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                    )
                    print("✅ Teacher quantization: bitsandbytes 4-bit NF4.")
                except ImportError:
                    print("⚠️ bitsandbytes missing; loading teacher without 4-bit quantization.")

                self.model = AutoModelForCausalLM.from_pretrained(
                    cfg.teacher_model_id,
                    **teacher_load_kwargs,
                )
                self.model.eval()
                # Explicit move to device logic
                # For 4-bit, we don't manually .to(device), bitsandbytes handles it via Accelerator or requires manual 'device_map={"": device}' hack
                # Safest bet for 4bit + DDP: Let it sit, get_logits handles .to(device) or trust bnb
                
                print("✅ Teacher Model Yüklendi.")
            else:
                print("ℹ️ Distill Alpha 0, Teacher Model yüklenmedi (Sadece Tokenizer).")

        except OSError as e:
            if "gated repo" in str(e) or "token" in str(e) or "401" in str(e):
                print(f"⛔ TEACHER AUTH ERROR: {cfg.teacher_model_id} is gated/missing.")
                if self.require_gated_teacher:
                    raise RuntimeError(
                        "require_gated_teacher=true and teacher access failed. "
                        "Set HF_TOKEN and ensure gated model access is approved."
                    ) from e
                print("⚠️  Gated teacher unavailable; continuing without teacher distillation.")
                self.model = None
                cfg.distill_alpha = 0.0
                if self.tokenizer is None:
                    self.tokenizer = load_teacher_tokenizer(prefer_local=True)
            else:
                print(f"⚠️ Teacher Init Error: {e}")
                sys.exit(1)
        except Exception as e:
            print(f"⚠️ Teacher Init Error: {e}")
            sys.exit(1)

    @torch.no_grad()
    def get_logits(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        if self.model is None:
            return None
        # Ensure input is on correct device (Accelerate handles this but good to be safe)
        input_ids = input_ids.to(self.device)
        return self.model(input_ids).logits


def _tokenizer_candidates() -> list[Path]:
    candidates: list[Path] = []

    env_override = os.environ.get("TITAN_LOCAL_TOKENIZER_PATH", "").strip()
    if env_override:
        p = Path(env_override).expanduser()
        candidates.append(p if p.is_absolute() else project_root / p)

    configured = str(getattr(cfg, "tr_tokenizer_id", "") or "").strip()
    if configured:
        p = Path(configured).expanduser()
        candidates.append(p if p.is_absolute() else project_root / p)

    candidates.extend(
        [
            project_root / "data" / "tokenizer" / "tr",
            project_root / "tokenizer" / "tr",
        ]
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _ensure_pad_token(tokenizer: Any) -> Any:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_local_runtime_tokenizer() -> Any:
    last_error: Exception | None = None
    for path in _tokenizer_candidates():
        if not path.exists():
            continue
        if not (path / "tokenizer.json").exists():
            continue
        try:
            print(f"🔤 Using local runtime tokenizer: {path}")
            tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
            return _ensure_pad_token(tok)
        except Exception as exc:
            last_error = exc
            print(f"⚠️ Local tokenizer load failed for {path}: {exc}")
    if last_error is not None:
        raise RuntimeError(f"local tokenizer candidates failed: {last_error}") from last_error
    raise FileNotFoundError("no local tokenizer artifact found")


def load_teacher_tokenizer(prefer_local: bool = False) -> Any:
    """
    Load the training tokenizer from the single source of truth
    (resolve_tokenizer). The teacher-vs-TR choice is governed solely by
    cfg.use_tr_tokenizer so train/eval/demo agree. No gpt2/silent fallback.

    The legacy ``prefer_local`` argument is accepted for backward compatibility
    but no longer changes the tokenizer family -- policy lives in the config.
    """
    return resolve_tokenizer(cfg)


# -----------------------------------------------------------------------------
# 5.5 OPTIMIZER HELPER
# -----------------------------------------------------------------------------
def build_optimizer(body_params: List[Any], router_params: List[Any], cfg: Any) -> torch.optim.Optimizer:
    """[H3] Construct the optimizer honoring use_galore/use_8bit_adam (the class can
    only be chosen here). Logs the ACTIVE class so config==reality (kills the
    phantom-GaLore mismatch)."""
    body_group = {"params": body_params, "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay}
    router_group = {"params": router_params, "lr": cfg.learning_rate * 1.5, "weight_decay": 1e-4}
    opt = None
    if getattr(cfg, "use_galore", False) and galore_torch is not None:
        try:
            optim_cls = galore_torch.GaLoreAdamW8bit if cfg.use_8bit_adam else galore_torch.GaLoreAdamW
            opt = optim_cls([
                {**body_group, "rank": 128, "update_proj_gap": 200, "scale": 0.25},
                {**router_group, "rank": 64, "update_proj_gap": 200, "scale": 0.25},
            ])
        except Exception as exc:
            print(f"⚠️ GaLore optimizer unavailable ({exc}); falling back.")
            opt = None
    if opt is None and getattr(cfg, "use_8bit_adam", False):
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit([body_group, router_group])
        except Exception as exc:
            print(f"⚠️ 8-bit AdamW unavailable ({exc}); falling back to torch AdamW.")
            opt = None
    if opt is None:
        opt = torch.optim.AdamW([body_group, router_group])
    print(
        f"🚀 OPTIMIZER ACTIVE: {type(opt).__name__} "
        f"(use_galore={getattr(cfg, 'use_galore', False)}, "
        f"use_8bit_adam={getattr(cfg, 'use_8bit_adam', False)}, "
        f"galore_available={galore_torch is not None})"
    )
    return opt


# -----------------------------------------------------------------------------
# 7. MAIN TRAIN LOOP
# -----------------------------------------------------------------------------
def train() -> None:
    # Accelerate Init
    accelerator_project_config = ProjectConfiguration(project_dir=str(project_root), logging_dir=str(project_root / "logs"))
    # TF32 for Ampere (A100) speedup
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        print("⚡ TensorFloat-32 (TF32) activated for A100.")
    
    # [P1 FIX] The DDP reducer must tolerate params that produce no grad in a step:
    #   (a) the liquid/tau warmup+cooldown freeze flips requires_grad AFTER prepare(),
    #   (b) MoE top-2 routing leaves some experts unrouted per micro-batch.
    # find_unused_parameters=True prevents the multi-GPU reducer error; on single-GPU/CPU
    # Accelerate does not wrap in DDP, so this is a no-op there (test suite unaffected).
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.grad_accum_steps,
        mixed_precision="bf16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "fp16" if cfg.use_amp else "no",
        project_config=accelerator_project_config,
        log_with="all",
        kwargs_handlers=[ddp_kwargs],
    )

    # Seed everything
    set_seed(cfg.seed + accelerator.process_index)
    
    # Only main process validation
    if accelerator.is_main_process:
        validate_config(cfg, stage="pre")

    curriculum_stage_names, curriculum_stage_ratios = get_curriculum_contract()
    
    student_device = accelerator.device

    logs_dir = project_root / "logs"
    save_dir = project_root / cfg.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        write_energy_telemetry_baseline(project_root, stage="start")

    # Logger
    # Logger (Main Process Only)
    logger = None
    if accelerator.is_main_process:
        logger = RunLogger(
            cfg=cfg,
            log_dir=logs_dir,
            project_root=project_root,
            train_path=current_file,
            config_path=project_root / "config" / "config.py",
            also_csv=True
        )
        logger.log_meta()


    # Distillation manager: switch between online (TeacherBundle) and offline (precomputed logits)
    # Offline logits mode (distill without loading teacher model)
    use_offline_logits = getattr(cfg, "use_precomputed_logits", False)
    distill_manager = None
    teacher = None
    teacher_tokenizer = None

    if use_offline_logits:
        print("🚀 DISTILLATION: Usage of Pre-computed Logits ENABLED (Offline Mode)")
        print("   Teacher logits will be read from disk (no teacher VRAM load).")
        teacher_tokenizer = load_teacher_tokenizer()
        distill_manager = DistillationManager(cfg, teacher_tokenizer)
    else:
        teacher = TeacherBundle()
        teacher_tokenizer = teacher.tokenizer

    # CURRICULUM: Find stage datasets
    stage_info = [
        (curriculum_stage_names[0], project_root / "datasets" / "stage1" / "stage1_data.jsonl"),
        (curriculum_stage_names[1], project_root / "datasets" / "stage2" / "stage2_data.jsonl"),
        (curriculum_stage_names[2], project_root / "datasets" / "stage3" / "stage3_data.jsonl"),
        (curriculum_stage_names[3], project_root / "datasets" / "stage4_soul" / "stage4_data.jsonl"),
        (curriculum_stage_names[4], project_root / "datasets" / "stage5_tools" / "stage5_data.jsonl"),
    ]
    stage_paths = [p for _, p in stage_info]

    # Fallback/Auto-Download logic
    offline_mode = os.getenv("TITAN_OFFLINE", "1") == "1"
    if not all(p.exists() for p in stage_paths):
        fallback_path = project_root / "datasets" / "training_data.jsonl"
        if offline_mode:
            if fallback_path.exists():
                stage_paths = [fallback_path]
                stage_info = [("fallback", fallback_path)]
                print("ℹ️ Offline mode: using local fallback dataset (datasets/training_data.jsonl).")
            else:
                raise FileNotFoundError(
                    "Offline mode is enabled (TITAN_OFFLINE=1) but required stage datasets are missing. "
                    "Provide local stage*.jsonl files or run data pipeline in online mode."
                )
        else:
            print("⚠️  Datasets not found. Launching Data Alchemy Engine...")
            import subprocess
            try:
                # Automatic download trigger
                alchemy_script = project_root / "scripts" / "data_pipeline.py"
                subprocess.check_call([sys.executable, str(alchemy_script)])
                print("✅ Data Alchemy Complete. Re-checking datasets...")

                # Re-check the datasets
                if not all(p.exists() for p in stage_paths):
                    # Maybe only the fallback file was created; check it
                    if fallback_path.exists():
                        stage_paths = [fallback_path]
                        stage_info = [("fallback", fallback_path)]
                        print("ℹ️ Using fallback dataset after Alchemy.")
                    else:
                        raise FileNotFoundError("Data Alchemy ran but datasets are still missing!")
            except Exception as e:
                print(f"❌ Data Pipeline Failed: {e}")
                sys.exit(1)

    # Curriculum dataset
    if use_offline_logits:
        stage_names = [name for name, _ in stage_info]
        # [B2] Verify identity alignment (not just existence) so a misaligned
        #     shard set is treated as missing and train never silently corrupts KD.
        _verify_align = bool(getattr(cfg, "verify_logit_alignment", True))
        if not distill_manager.has_precomputed_logits(stage_names, verify_alignment=_verify_align):
            if offline_mode:
                raise RuntimeError(
                    "Precomputed logits are missing or incomplete while TITAN_OFFLINE=1. "
                    "Canonical offline_clean requires completed Phase-0 logits shards before training."
                )
            else:
                if bool(getattr(cfg, "require_gated_teacher", False)):
                    print(
                        "⚠️ Precomputed logits missing. Hard teacher policy active: "
                        "switching to ONLINE gated teacher generation."
                    )
                else:
                    print("⚠️ Precomputed logits not found for all stages. Falling back to ONLINE teacher.")
                teacher = TeacherBundle()
                teacher_tokenizer = teacher.tokenizer
                use_offline_logits = False
        else:
            # Logit-synced dataset (requires num_workers=0)
            curriculum_ds = PrecomputedCurriculumDataset(stage_info, cfg.max_seq_len, teacher_tokenizer, distill_manager)

    if not use_offline_logits:
        curriculum_ds = CurriculumDataset(stage_paths, cfg.max_seq_len, teacher_tokenizer, current_stage=1)

    num_workers = getattr(cfg, "dataloader_num_workers", 4)
    prefetch_factor = getattr(cfg, "dataloader_prefetch_factor", 2)
    if use_offline_logits:
        num_workers = 0  # deterministic alignment with precomputed logits
        prefetch_factor = None
    else:
        # Online curriculum stage changes live on the dataset object; worker copies do not see them.
        num_workers = 0
        prefetch_factor = None
    dl = DataLoader(
        curriculum_ds,
        batch_size=cfg.micro_batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=bool(getattr(cfg, "dataloader_pin_memory", torch.cuda.is_available())) and torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    # [PRO] Validation Setup
    val_path = project_root / "datasets" / "validation.jsonl"
    claim_mode = os.getenv("TITAN_CLAIM_MODE", "0") == "1"
    min_val_warn = int(getattr(cfg, "validation_min_samples_warn", 128))
    min_val_claim = int(getattr(cfg, "validation_min_samples_claim", 1000))
    if val_path.exists():
        val_count = count_jsonl_records(val_path)
        if accelerator.is_main_process:
            print(f"🧪 Validation records: {val_count}")
        if claim_mode and val_count < min_val_claim:
            raise RuntimeError(
                f"Claim mode requires validation >= {min_val_claim} samples, got {val_count}. "
                "Regenerate validation set before claim-grade training."
            )
        if val_count < min_val_warn and accelerator.is_main_process:
            print(
                f"⚠️ Validation set is small ({val_count}). "
                f"Claim-grade runs should use >= {min_val_claim} samples."
            )
        print(f"🔍 Validation Dataset Found: {val_path}")
        # [PRO] Use Deterministic Dataset
        val_ds = ValidationJsonlDataset(val_path, cfg.max_seq_len, teacher_tokenizer)
        # num_workers=0 ensures main process does sequential read
        val_dl = DataLoader(val_ds, batch_size=cfg.micro_batch_size, collate_fn=collate_fn, num_workers=0)
    else:
        print("⚠️ Validation set not found (datasets/validation.jsonl). Using training stream for sanity check.")
        # [R9] Build a SEPARATE DataLoader over the training dataset instead of aliasing `dl`:
        # passing the same object to accelerator.prepare twice wraps it twice and double-advances
        # the underlying IterableDataset. A distinct loader preserves the sanity-check intent safely.
        val_dl = DataLoader(dl.dataset, batch_size=cfg.micro_batch_size, collate_fn=collate_fn, num_workers=0)

    # CRITICAL FIX: Epoch Mode Calculation MOVED UP
    # Must be done BEFORE Scheduler initialization!
    # [PRO] Only override max_steps if explicitly requested via EPOCH_MODE or if undefined
    # This allows test scripts to set max_steps=2 without interference.
    if getattr(cfg, "epoch_mode", False) and (not hasattr(cfg, "max_steps") or cfg.max_steps is None or cfg.max_steps > 1000):
        # 12M samples (sweet spot for 3.67B model)
        NUM_EPOCHS = 3
        TOTAL_SAMPLES = 12_000_000  # Sweet spot: ~6B tokens
        cfg.max_steps = int((TOTAL_SAMPLES / (cfg.micro_batch_size * cfg.grad_accum_steps)) * NUM_EPOCHS)
        print(f"🔄 EPOCH MODE ACTIVATED: {NUM_EPOCHS} Epochs ({TOTAL_SAMPLES/1e6:.1f}M Samples) -> Max Steps: {cfg.max_steps}")
    else:
        print(f"ℹ️  Epoch Mode Skipped. Using provided max_steps: {cfg.max_steps}")

    # Derive vocab size from the real tokenizer via len() so added/special
    #     tokens are included (Llama-3: 128000 attr vs 128256 len). Closes the
    #     128256/128000 embedding/lm_head mismatch.
    cfg.vocab_size = len(teacher_tokenizer)
    global RUNTIME_TOKENIZER_ID
    RUNTIME_TOKENIZER_ID = tokenizer_identity(teacher_tokenizer, cfg)
    print(
        f"🔤 Runtime tokenizer locked: {RUNTIME_TOKENIZER_ID['name_or_path']} "
        f"(vocab={RUNTIME_TOKENIZER_ID['vocab_size']}, "
        f"use_tr_tokenizer={RUNTIME_TOKENIZER_ID['use_tr_tokenizer']})"
    )
    if accelerator.is_main_process:
        validate_config(cfg, stage="post")

    student = MertFormer()
    # Hard-align embedding/lm_head to the tokenizer (resize safety net).
    student.resize_token_embeddings(cfg.vocab_size)
    # Note: .to(device) is handled by accelerator.prepare, but explicit move is fine before prepare
    student.to(student_device)
    resume_payload = _load_resume_payload(cfg, student, is_main_process=accelerator.is_main_process)

    # -------------------------------------------------------------------------
    # Maximum performance: torch.compile with max-autotune
    # -------------------------------------------------------------------------
    use_compile = getattr(cfg, "use_torch_compile", True)
    compile_mode = getattr(cfg, "torch_compile_mode", "max-autotune")
    
    if torch.cuda.is_available() and hasattr(torch, 'compile') and use_compile:
        try:
            print(f"⚡ torch.compile ACTIVE - Mode: {compile_mode}")
            print("   First forward pass will take 10-15 minutes (kernel optimization)")
            print("   Subsequent steps will be 15-20% faster!")
            
            # max-autotune: Most aggressive optimization (best for training)
            # reduce-overhead: Balanced (good for inference)
            # default: Conservative (fallback)
            student = torch.compile(student, mode=compile_mode, fullgraph=False)
            print("✅ torch.compile successful!")
        except Exception as e:
            print(f"⚠️  torch.compile failed: {e}")
            print("   Falling back to eager mode (no performance loss, just no speedup)")
    else:
        if not torch.cuda.is_available():
            print("⚠️  torch.compile skipped (No CUDA)")
        elif not hasattr(torch, 'compile'):
            print("⚠️  torch.compile skipped (PyTorch < 2.0)")
        else:
            print("ℹ️  torch.compile disabled in config")


    # Apply freeze policy if enabled
    freeze_core = getattr(cfg, "freeze_core_layers", False)
    apply_freeze_policy(student, freeze_core)

    student.train()
    preflight_param_report(student)

    # GROKKING STRATEGY: Differential Learning Rates
    # Router and Tau parameters get a higher LR and lower weight decay so the
    # model resolves the "routing logic" quickly (Grokking).
    router_params = []
    body_params = []
    for n, p in student.named_parameters():
        if not p.requires_grad:
            continue
        if "router" in n or "tau" in n or "shared_gate" in n:
            router_params.append(p)
        else:
            body_params.append(p)

    print(f"🧠 Grokking Setup: {len(router_params)} Router Params | {len(body_params)} Body Params")

    # [H3] Build the real optimizer per cfg.use_galore/use_8bit_adam.
    opt = build_optimizer(body_params, router_params, cfg)

    # WSD Scheduler (Warmup-Stable-Decay) moved to global scope

    scheduler = get_wsd_schedule(
        opt,
        num_warmup_steps=int(cfg.max_steps * 0.1), # 10% Warmup
        num_training_steps=cfg.max_steps,
        min_lr_ratio=0.01
    )

    if resume_payload is not None:
        resume_state = resume_payload.get("state", {})
        # [H3] Warn if the optimizer class changed (state_dict incompatible).
        saved_opt_class = resume_state.get("optimizer_class")
        if saved_opt_class and saved_opt_class != type(opt).__name__ and accelerator.is_main_process:
            print(
                f"⚠️ Optimizer class changed since checkpoint "
                f"(saved={saved_opt_class}, now={type(opt).__name__}); "
                "optimizer state may be reset on load."
            )
        try:
            if "optimizer" in resume_state:
                opt.load_state_dict(resume_state["optimizer"])
            if "scheduler" in resume_state:
                scheduler.load_state_dict(resume_state["scheduler"])
            # [MED] Restore RNG states for reproducible resume.
            if "rng_state" in resume_state:
                _restore_rng_state(resume_state["rng_state"])
            if accelerator.is_main_process:
                print("✅ Resume optimizer/scheduler/RNG state restored.")
        except Exception as resume_exc:
            if accelerator.is_main_process:
                print(f"⚠️ Resume state restore warning (optimizer/scheduler): {resume_exc}")
                print("   Continuing with freshly initialized optimizer/scheduler.")

    # [ACCELERATE PREPARE]
    # DDP Magic happens here!
    student, opt, dl, val_dl, scheduler = accelerator.prepare(
        student, opt, dl, val_dl, scheduler
    )
    
    if accelerator.is_main_process:
        print("✅ Accelerate Preparation Complete")

    # Mixed Precision handled by Accelerate (no manual scaler needed)
    scaler = None

    # Early Stopping & Monitoring
    best_val_loss = float('inf')
    if resume_payload is not None:
        resume_best_val_loss = resume_payload.get("best_val_loss")
        resume_val_loss = resume_payload.get("val_loss")
        if resume_best_val_loss is not None:
            best_val_loss = float(resume_best_val_loss)
        elif resume_val_loss is not None:
            best_val_loss = float(resume_val_loss)
    patience_counter = 0
    early_stop_patience = getattr(cfg, "early_stop_patience", 5)
    val_check_interval = getattr(cfg, "val_check_interval", 1000)

    # Gradient Norm Monitoring
    max_grad_norm_seen = 0.0
    grad_norm_history = []
    loss_history = [] # Track Loss History for Signal-Based Curriculum
    grad_norm_collapse_threshold = 0.01  # If grad norm < 0.01, model stopped learning
    continual_adapter = None
    continual_state = None
    if bool(getattr(cfg, "use_continual_adapter", False)):
        if ContinualLearningAdapter is None:
            if accelerator.is_main_process:
                print("⚠️ Continual adapter requested but module import failed; disabling continual adapter.")
        else:
            continual_adapter = ContinualLearningAdapter(
                replay_capacity=int(getattr(cfg, "continual_replay_capacity", 2048)),
                loss_ema_decay=float(getattr(cfg, "continual_loss_ema_decay", 0.98)),
                drift_threshold=float(getattr(cfg, "continual_drift_threshold", 0.2)),
            )
            if accelerator.is_main_process:
                print(
                    "♻️ Continual adapter active "
                    f"(capacity={getattr(cfg, 'continual_replay_capacity', 2048)}, "
                    f"drift_threshold={getattr(cfg, 'continual_drift_threshold', 0.2):.3f})"
                )
    
    # SAFEGUARD: Liquid Auto-Freeze State
    liquid_frozen_until = 0 # Step count until Liquid is unfrozen
    liquid_spike_counter = 0 # SAFEGUARD: 3-Strike Rule
    liquid_spike_threshold = float(getattr(cfg, "liquid_spike_threshold", 5.0))
    liquid_spike_patience = int(getattr(cfg, "liquid_spike_patience", 3))
    liquid_spike_cooldown_steps = int(getattr(cfg, "liquid_spike_cooldown_steps", 200))

    # Curriculum Stage Tracking (single source from config ratios)
    stage_boundaries = build_stage_boundaries(cfg.max_steps, curriculum_stage_ratios)
    stage1_end, stage2_end, stage3_end, stage4_end = stage_boundaries
    resume_step = int(resume_payload["step"]) if resume_payload is not None else 0
    current_curriculum_stage = _infer_curriculum_stage_from_step(resume_step, stage_boundaries)
    curriculum_ds.set_stage(current_curriculum_stage)

    global_step = max(0, resume_step)
    early_stopped = False
    accum_loss = 0.0
    accum_count = 0  # [FIX] Counter for proper avg_loss calculation
    micro_step = 0
    start_time = time.time()
    tokens_processed = 0
    tokens_seen_total = 0

    token_budget_mode = str(getattr(cfg, "token_budget_mode", "fixed_steps")).lower()
    open_ended_mode = token_budget_mode == "open_ended"
    target_tokens_min = int(getattr(cfg, "target_tokens_min", 0))
    saturation_eval_interval_steps = int(getattr(cfg, "saturation_eval_interval_steps", 2000))
    saturation_patience_windows = int(getattr(cfg, "saturation_patience_windows", 3))
    val_improve_min_rel = float(getattr(cfg, "val_improve_min_rel", 0.002))
    golden_improve_min_abs = float(getattr(cfg, "golden_improve_min_abs", 0.01))
    gsm8k_improve_min_abs = float(getattr(cfg, "gsm8k_improve_min_abs", 0.002))
    max_consecutive_nan = int(getattr(cfg, "max_consecutive_nan", 3))
    max_consecutive_oom_backoff_fail = int(getattr(cfg, "max_consecutive_oom_backoff_fail", 5))

    consecutive_nan = 0
    consecutive_oom_backoff_fail = 0
    safety_brake_triggered = False
    safety_brake_reason = ""

    latest_val_loss = None
    best_val_saturation = None
    best_golden_score = None
    best_gsm8k_score = None
    saturation_plateau_windows = 0

    # Runtime manifest for portable handoff observability.
    if accelerator.is_main_process:
        write_training_runtime_manifest(
            {
                "status": "started",
                "token_budget_mode": token_budget_mode,
                "target_tokens_min": target_tokens_min,
                "curriculum_stage_names": curriculum_stage_names,
                "curriculum_stage_ratios": curriculum_stage_ratios,
                "max_steps_nominal": int(cfg.max_steps),
                "micro_batch_size": int(cfg.micro_batch_size),
                "grad_accum_steps": int(cfg.grad_accum_steps),
                "resume_enabled": os.getenv("TITAN_AUTO_RESUME", "1") == "1",
                "resume_checkpoint": resume_payload["checkpoint_path"] if resume_payload is not None else None,
                "resume_step": int(global_step),
            }
        )

    if accelerator.is_main_process:
        print(f"🚀 TITAN ONYX STORM TRAINING STARTING...")
        print(f"📊 Max Steps (nominal): {cfg.max_steps}")
        print(f"🧮 Token Budget: mode={token_budget_mode} | min_tokens={target_tokens_min}")
        if global_step > 0:
            print(f"♻️  Resuming from step {global_step} | current_stage={current_curriculum_stage}")
        print(f"📚 Curriculum Stages:")
        print(f"   - Stage 1 ({curriculum_stage_names[0]}): Steps 0-{stage1_end} ({curriculum_stage_ratios[0]*100:.1f}%)")
        print(f"   - Stage 2 ({curriculum_stage_names[1]}): Steps {stage1_end}-{stage2_end} ({curriculum_stage_ratios[1]*100:.1f}%)")
        print(f"   - Stage 3 ({curriculum_stage_names[2]}): Steps {stage2_end}-{stage3_end} ({curriculum_stage_ratios[2]*100:.1f}%)")
        print(f"   - Stage 4 ({curriculum_stage_names[3]}): Steps {stage3_end}-{stage4_end} ({curriculum_stage_ratios[3]*100:.1f}%)")
        print(f"   - Stage 5 ({curriculum_stage_names[4]}): Steps {stage4_end}-{cfg.max_steps} ({curriculum_stage_ratios[4]*100:.1f}%)")
        print(f"📊 Early Stopping: Patience={early_stop_patience} | Val Check: Every {val_check_interval} steps")

    try:
        dataloader_iter = iter(dl)
        while True:
            if not open_ended_mode and global_step >= cfg.max_steps:
                break
            # ---------------------------------------------------------------------
            # INTELLIGENT PILOT: Signal-Based Curriculum + Time Fallback
            # ---------------------------------------------------------------------
            
            # Use rolling average loss if available
            current_avg_loss = sum(loss_history[-100:]) / len(loss_history[-100:]) if loss_history else 999.0
            
            # Stage 1 -> 2
            if current_curriculum_stage == 1:
                # Loss < 1.5 signal OR Time Limit
                if (current_avg_loss < 1.5 and global_step > stage1_end * 0.5) or (global_step > stage1_end):
                    new_stage = 2
                else:
                    new_stage = 1
            # Stage 2 -> 3
            elif current_curriculum_stage == 2:
                # Loss < 1.2 signal OR Time Limit
                if (current_avg_loss < 1.2 and global_step > stage2_end * 0.5) or (global_step > stage2_end):
                    new_stage = 3
                else:
                    new_stage = 2
            # Stage 3 -> 4
            elif current_curriculum_stage == 3:
                # Loss < 1.0 signal OR Time Limit
                if (current_avg_loss < 1.0 and global_step > stage3_end * 0.5) or (global_step > stage3_end):
                    new_stage = 4
                else:
                    new_stage = 3
            # Stage 4 -> 5
            elif current_curriculum_stage == 4:
                # Loss < 0.9 signal OR Time Limit
                if (current_avg_loss < 0.9 and global_step > stage4_end * 0.5) or (global_step > stage4_end):
                    new_stage = 5
                else:
                    new_stage = 4
            else:
                new_stage = 5

            if new_stage != current_curriculum_stage:
                current_curriculum_stage = new_stage
                curriculum_ds.set_stage(new_stage)
                if accelerator.is_main_process:
                    print(f"📚 Curriculum Stage Updated: Stage {new_stage} (Signal: {current_avg_loss:.2f})")
            # ---------------------------------------------------------------------

            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dl)
                batch = next(dataloader_iter)

            t_logits = None
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                input_ids, labels, t_logits = batch
            else:
                input_ids, labels = batch

            # Accelerate handles device placement
            # input_ids = input_ids.to(student_device)
            # labels = labels.to(student_device)

            # ---------------------------------------------------------------------
            # SAFEGUARD: Liquid Warmup (Freeze Early Steps)
            # Liquid Re-Freeze Logic
            # ---------------------------------------------------------------------
            
            # 1. Warmup Phase
            if global_step < cfg.liquid_warmup_steps:
                 # Ensure Liquid is frozen
                 for n, p in student.named_parameters():
                     if "tau" in n or "liquid" in n:
                         p.requires_grad = False
                 
                 # SAFEGUARD: True freeze — grad=None so AdamW skips frozen params
                 # (zero_() left decoupled weight-decay still shrinking the 'frozen' params).
                 for p in student.parameters():
                     if not p.requires_grad and p.grad is not None:
                         p.grad = None
            elif global_step == cfg.liquid_warmup_steps:
                 # Unfreeze Logic - ONCE
                 if accelerator.is_main_process:
                    print(f"🔓 UNFREEZING LIQUID LAYERS at Step {global_step}!")
                 for n, p in student.named_parameters():
                     if "tau" in n or "liquid" in n:
                         p.requires_grad = True
                 
                 # NOTE: This branch ONLY toggles requires_grad=True above; it does NOT
                 # rebuild the optimizer or re-sync optimizer param groups.
                 # Optimizer rebuild is intentionally skipped for safety under Accelerate/DDP
                 # (a full rebuild would require re-wrapping with Accelerate).
                 # AdamW continues with its existing param groups; newly-unfrozen params
                 # are picked up because they are already registered in the optimizer.
                 pass
                         
            # 2. Emergency Cooldown Phase (Refreeze)
            elif liquid_frozen_until > 0:
                 if global_step < liquid_frozen_until:
                     # Keep frozen
                     for n, p in student.named_parameters():
                        if "tau" in n or "liquid" in n:
                            p.requires_grad = False
                     # SAFEGUARD: True freeze — grad=None so AdamW skips frozen params.
                     for p in student.parameters():
                         if not p.requires_grad and p.grad is not None:
                             p.grad = None
                             
                 elif global_step == liquid_frozen_until:
                     print(f"🧊 LIQUID COOLDOWN OVER. Unfreezing at {global_step}...")
                     for n, p in student.named_parameters():
                         if "tau" in n or "liquid" in n:
                             p.requires_grad = True
                      
                     # Accelerate wraps the optimizer after prepare; the optimizer
                     # already owns these params, so requires_grad is enough here.
                      
                     liquid_frozen_until = 0
            # ---------------------------------------------------------------------
            # ---------------------------------------------------------------------

            # ---------------------------------------------------------------------
            # [CRITICAL FIX] Correct DDP Accumulation using Context Manager
            # ---------------------------------------------------------------------
            try:
                with accelerator.accumulate(student):
                    # Mixed Precision context handled by accelerate.accumulate or autocast
                    # We use autocast for extra safety
                    with accelerator.autocast():
                        s_logits, aux_loss, _ = student(input_ids)

                        # Teacher logits: offline (precomputed) or online (teacher model)
                        if t_logits is None and teacher is not None and teacher.model is not None:
                            # Teacher is already on correct device via device_map fix
                            t_logits = teacher.get_logits(input_ids)

                        shift_logits = s_logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()

                        # [H2] Datasets mark padding as -100 (canonical ignore).
                        #     When pad_id != eos_id, also ignore raw pad_id (back-compat);
                        #     when pad_id == eos_id, do NOT mask by ==pad_id, else EOS
                        #     supervision is dropped (the bug: model never learns to stop).
                        pad_id = teacher_tokenizer.pad_token_id if teacher_tokenizer and teacher_tokenizer.pad_token_id is not None else 0
                        eos_id = teacher_tokenizer.eos_token_id if teacher_tokenizer is not None else None
                        IGNORE_INDEX = -100
                        ce_labels = shift_labels
                        if eos_id is None or pad_id != eos_id:
                            ce_labels = shift_labels.masked_fill(shift_labels == pad_id, IGNORE_INDEX)
                        loss_ce = F.cross_entropy(
                            shift_logits.view(-1, cfg.vocab_size),
                            ce_labels.view(-1),
                            ignore_index=IGNORE_INDEX,
                            label_smoothing=getattr(cfg, "label_smoothing", 0.0)
                        )

                        loss_distill = 0.0
                        if t_logits is not None:
                            t_logits = _teacher_payload_to_device(t_logits, shift_logits.device)
                            shift_t_logits = _shift_teacher_payload(t_logits)
                            kd_mask = ce_labels != IGNORE_INDEX
                            loss_distill = kd_loss_safe(shift_logits, shift_t_logits, cfg.teacher_temp, mask=kd_mask)

                        # Dynamic Distillation Alpha
                        progress_pct = min(global_step / max(1, cfg.max_steps), 1.0)
                        start_alpha = getattr(cfg, "distill_alpha", 0.8)
                        
                        if progress_pct < 0.3:
                            dyn_alpha = start_alpha
                        elif progress_pct > 0.8:
                            dyn_alpha = 0.15
                        else:
                            slope = (0.15 - start_alpha) / (0.8 - 0.3)
                            dyn_alpha = start_alpha + slope * (progress_pct - 0.3)
                        
                        alpha = dyn_alpha if t_logits is not None else 0.0
                        aux_coef = getattr(cfg, "router_aux_loss_coef", 0.01)
                        total_loss = (1.0 - alpha) * loss_ce + alpha * loss_distill + aux_coef * aux_loss

                    # NaN Check - [H1] COLLECTIVE skip decision.
                    # On multi-GPU a single-rank 'continue' would deadlock peers at the
                    #     collective backward/clip_grad_norm_ (NCCL collective). All-reduce
                    #     the skip flag so every rank takes the same branch
                    #     (sum>0 -> all skip together).
                    nan_flag = torch.tensor(
                        [0.0 if torch.isfinite(total_loss) else 1.0],
                        device=total_loss.device,
                        dtype=torch.float32,
                    )
                    if accelerator.num_processes > 1:
                        nan_flag = accelerator.reduce(nan_flag, reduction="sum")
                    if nan_flag.item() > 0:
                        consecutive_nan += 1
                        print(f"⚠️ NaN detected ({consecutive_nan}/{max_consecutive_nan}), skipping step (collective)")
                        opt.zero_grad()
                        if consecutive_nan >= max_consecutive_nan:
                            safety_brake_triggered = True
                            safety_brake_reason = "nan_divergence"
                            print("🛑 SAFETY BRAKE: consecutive NaN threshold reached.")
                            break
                        continue
                    consecutive_nan = 0

                    # Track micro-batch stats (for accurate averages)
                    accum_loss += total_loss.item()
                    accum_count += 1
                    # [B2/budget] Don't count pad tokens. Datasets mark pad as -100;
                    #     count only real (supervised) tokens so the token-budget
                    #     provenance is not pad-inflated.
                    real_tokens = int((labels != -100).sum().item())
                    tokens_processed += real_tokens
                    tokens_seen_total += real_tokens * max(1, accelerator.num_processes)

                    # Backward (Accelerate handles scaling and division by accum steps AUTOMATICALLY)
                    accelerator.backward(total_loss)

                    # Optimizer Step (Only runs when accumulation is complete)
                    if accelerator.sync_gradients:
                        consecutive_oom_backoff_fail = 0
                        # Clipping
                        grad_norm = accelerator.clip_grad_norm_(student.parameters(), cfg.grad_clip)

                        opt.step()
                        scheduler.step()
                        opt.zero_grad()

                        global_step += 1
                        if continual_adapter is not None:
                            replay_sample = None
                            if isinstance(input_ids, torch.Tensor) and input_ids.ndim >= 2 and input_ids.size(0) > 0:
                                replay_sample = input_ids[0]
                            continual_state = continual_adapter.update(
                                loss=float(total_loss.item()),
                                sample=replay_sample,
                            )

                        # SAFEGUARD: Liquid spike tracking (3-strike rule)
                        liquid_spike_counter, liquid_frozen_until, spike_triggered = update_liquid_spike_state(
                            loss_value=float(total_loss.item()),
                            threshold=liquid_spike_threshold,
                            counter=liquid_spike_counter,
                            patience=liquid_spike_patience,
                            frozen_until=liquid_frozen_until,
                            global_step=global_step,
                            cooldown_steps=liquid_spike_cooldown_steps,
                            enabled=bool(getattr(cfg, "use_liquid", False)) and global_step >= int(getattr(cfg, "liquid_warmup_steps", 0)),
                        )
                        if spike_triggered and accelerator.is_main_process:
                            print(
                                f"🧊 LIQUID SPIKE: loss>{liquid_spike_threshold:.2f} "
                                f"({liquid_spike_patience} strikes). Freezing until step {liquid_frozen_until}."
                            )
                    else:
                        continue

                    # Update Stats
                    grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

                    max_grad_norm_seen = max(max_grad_norm_seen, grad_norm_val)
                    grad_norm_history.append(grad_norm_val)
                    loss_history.append(total_loss.item())  # Use total_loss.item()
                    if len(grad_norm_history) > 100:
                        grad_norm_history.pop(0)
                    if len(loss_history) > 100:
                        loss_history.pop(0)

                    # GRADIENT NORM COLLAPSE DETECTION
                    if len(grad_norm_history) >= 10:
                        avg_recent = sum(grad_norm_history[-10:]) / 10
                        if avg_recent < grad_norm_collapse_threshold:
                            print(f"⚠️  WARNING: Gradient norm collapse detected! Avg: {avg_recent:.6f}")
                            print(f"   Model may have stopped learning. Consider adjusting learning rate.")

                    # Engine Overheating Protection — log a gradient spike, but do
                    # NOT permanently ratchet cfg.grad_clip down. The old behavior
                    # (cfg.grad_clip *= 0.7, floor 0.1, never recovers) could latch the
                    # clip at 0.1 after a handful of early BitNet-STE spikes and then
                    # silently over-clip the rest of the 45K run. clip_grad_norm_ already
                    # clamps THIS step at the configured value, and the NaN/Inf + 3-strike
                    # spike guards handle genuine instability, so the clip stays transient.
                    if grad_norm_val > 10.0:
                        print(f"⚠️  WARNING: Gradient norm {grad_norm_val:.2f} exceeds soft threshold (transient; clip held at {cfg.grad_clip}).")

                    # Log Metrics (Main Process)
                    if global_step % cfg.log_interval == 0 and logger and accelerator.is_main_process:
                        metrics = {
                            "loss": total_loss.item(),
                            "ce": loss_ce.item(),
                            "distill": loss_distill.item() if isinstance(loss_distill, torch.Tensor) else loss_distill,
                            "aux": aux_loss.item(),
                            "lr": scheduler.get_last_lr()[0],
                            "grad_norm": grad_norm_val,
                            "alpha": alpha,
                            "stage": current_curriculum_stage
                        }
                        if continual_state is not None:
                            metrics["continual_ema_loss"] = float(continual_state.running_loss_ema)
                            metrics["continual_replay_size"] = int(continual_state.replay_size)
                            metrics["continual_drift_alert"] = int(bool(continual_state.drift_alert))
                        metrics["global_step"] = global_step
                        logger.log_step(metrics)

                    # Logging only on Main Process
                    if global_step % cfg.log_interval == 0 and accelerator.is_main_process:
                        dt = time.time() - start_time
                        tok_s = tokens_processed / max(dt, 1e-6)
                        # [FIX] Use counter for proper averaging
                        avg_loss = accum_loss / max(1, accum_count)
                        lr_now = scheduler.get_last_lr()[0]

                        distill_val = loss_distill.item() if isinstance(loss_distill, torch.Tensor) else 0.0
                        aux_val = aux_loss.item() if isinstance(aux_loss, torch.Tensor) else 0.0
                        avg_grad_norm = sum(grad_norm_history[-10:]) / len(grad_norm_history[-10:]) if grad_norm_history else 0.0

                        print(f"Step {global_step} | Stage {current_curriculum_stage} | Loss: {avg_loss:.4f} | "
                              f"Tok/s: {tok_s:.0f} | LR: {lr_now:.2e} | GradNorm: {avg_grad_norm:.3f}")

                        write_training_runtime_manifest(
                            {
                                "status": "running",
                                "global_step": int(global_step),
                                "token_budget_mode": token_budget_mode,
                                "target_tokens_min": int(target_tokens_min),
                                "tokens_seen": int(tokens_seen_total),
                                "throughput_tok_s": float(tok_s),
                                "current_stage": int(current_curriculum_stage),
                                "latest_loss": float(avg_loss),
                            }
                        )

                        # Periodic safety checks
                        if global_step % 1000 == 0:
                            # Disk space check
                            if not check_disk_space(min_gb=50):
                                print("   ⚠️  Consider cleaning old checkpoints!")

                            # GPU memory report
                            if torch.cuda.is_available():
                                alloc, res = get_gpu_memory_usage()
                                print(f"   📊 GPU Memory: {alloc:.1f} GB / {res:.1f} GB")

                        log_data = {
                            "step": global_step,
                            "curriculum_stage": current_curriculum_stage,
                            "loss": avg_loss,
                            "ce": loss_ce.item(),
                            "kd": distill_val,
                            "aux": aux_val,
                            "tok_s": tok_s,
                            "interval_sec": float(dt),
                            "step_wall_sec": float(dt / max(accum_count, 1)),
                            "lr": lr_now,
                            "grad_norm": avg_grad_norm,
                            "max_grad_norm": max_grad_norm_seen
                        }
                        # Throttle the host system snapshot — it shells out for GPU/host
                        # stats, so at log_interval=1 it would fire a subprocess every step
                        # (~45K subprocess spawns across the run). Run it at most every
                        # TITAN_TELEMETRY_INTERVAL optimizer steps, independent of log_interval.
                        _telemetry_interval = int(os.environ.get("TITAN_TELEMETRY_INTERVAL", "100"))
                        if global_step % max(_telemetry_interval, 1) == 0:
                            telemetry_snapshot = system_snapshot(project_root)
                            for key, value in telemetry_snapshot.items():
                                if key == "timestamp_utc" or value is None:
                                    continue
                                log_data[key] = value
                        if continual_state is not None:
                            log_data["continual_ema_loss"] = float(continual_state.running_loss_ema)
                            log_data["continual_replay_size"] = int(continual_state.replay_size)
                            log_data["continual_drift_alert"] = int(bool(continual_state.drift_alert))
                            if continual_state.drift_alert:
                                print(
                                    f"   ⚠️ Continual Drift Alert: ema_loss={continual_state.running_loss_ema:.4f} "
                                    f"| replay={continual_state.replay_size}"
                                )

                        # [TELEMETRY] MoE Router Health (log-interval only)
                        moe_loads = []
                        moe_entropies = []
                        moe_overflows = []
                        # Handle DDP/Compile wrappers
                        model_ref = student.module if hasattr(student, "module") else student
                        model_ref = model_ref._orig_mod if hasattr(model_ref, "_orig_mod") else model_ref

                        for m in model_ref.modules():
                            if hasattr(m, "get_expert_load"):
                                moe_loads.append(m.get_expert_load())
                            if hasattr(m, "get_router_entropy"):
                                moe_entropies.append(m.get_router_entropy())
                            if hasattr(m, "last_capacity_overflow_ratio"):
                                moe_overflows.append(m.last_capacity_overflow_ratio)

                        if moe_loads:
                            # Stack: [Layers, Experts]
                            loads = torch.stack(moe_loads)  # (L, E)
                            # Metrics
                            max_load = loads.max().item()  # Worst case imbalance
                            avg_std = loads.std(dim=1).mean().item()  # Overall balance score (lower is better)
                            avg_entropy = (
                                torch.stack(moe_entropies).mean().item()
                                if moe_entropies
                                else float("nan")
                            )
                            avg_overflow = (
                                torch.stack(moe_overflows).mean().item()
                                if moe_overflows
                                else 0.0
                            )
                            collapse_alarm = float(getattr(cfg, "router_alarm_threshold", 0.40))

                            log_data["moe_max_load"] = max_load
                            log_data["moe_avg_std"] = avg_std
                            if not math.isnan(avg_entropy):
                                log_data["moe_load_entropy"] = avg_entropy
                            log_data["moe_capacity_overflow"] = avg_overflow

                            entropy_txt = f"{avg_entropy:.3f}" if not math.isnan(avg_entropy) else "n/a"
                            print(
                                f"   🧠 MoE Health: MaxLoad={max_load:.2f} | "
                                f"Balance(std)={avg_std:.3f} | Entropy={entropy_txt} | "
                                f"Overflow={avg_overflow:.3f}"
                            )

                            if max_load > collapse_alarm:
                                print(
                                    f"   ⚠️  EARLY IMBALANCE ALERT: Max Load {max_load:.2f} "
                                    f"(alarm>{collapse_alarm:.2f})"
                                )

                            # Router Collapse Warning + ACTION
                            if max_load > 0.85 and getattr(cfg, "active_experts", 1) < getattr(cfg, "num_experts", 4):
                                print(f"⚠️  ROUTER COLLAPSE DETECTED: Max Load {max_load:.2f} (Target: {1.0/cfg.num_experts:.2f})")
                                print(f"   🔧 ACTION: Jitter boost activated via MoE module (see moe.py collapse_detected)")
                                log_data["router_collapse"] = True

                        if logger:
                            logger.log_step(log_data)

                        accum_loss = 0.0
                        accum_count = 0  # [FIX] Reset counter
                        tokens_processed = 0
                        start_time = time.time()

                    # Validation & Early Stopping
                    if global_step % val_check_interval == 0 and global_step > 0:
                        student.eval()
                        val_loss_local = 0.0
                        val_samples_local = 0
                        val_steps = int(getattr(cfg, "val_steps", 10))  # [16] configurable via cfg.val_steps
                        try:
                            # [PRO] Use dedicated validation dataloader if available
                            val_iter = iter(val_dl)
                            for _ in range(val_steps):
                                try:
                                    _val_batch = next(val_iter)
                                except StopIteration:
                                    val_iter = iter(val_dl)
                                    _val_batch = next(val_iter)
                                # Robust to both 2-tuple (validation.jsonl) and
                                # 3-tuple (offline-logits fallback `val_dl = dl`)
                                # batches: validation CE only needs ids + labels;
                                # any teacher-logit payload (3rd element) is ignored.
                                val_input_ids, val_labels = _val_batch[0], _val_batch[1]

                                non_blocking = bool(getattr(cfg, "dataloader_non_blocking", True))
                                val_input_ids = val_input_ids.to(student_device, non_blocking=non_blocking)
                                val_labels = val_labels.to(student_device, non_blocking=non_blocking)

                                with torch.no_grad():
                                    val_logits, _, _ = student(val_input_ids, use_cache=False)
                                    val_shift_logits = val_logits[..., :-1, :].contiguous()
                                    val_shift_labels = val_labels[..., 1:].contiguous()
                                    # [H2] Same canonical -100 ignore as the train CE.
                                    _v_pad = teacher_tokenizer.pad_token_id if teacher_tokenizer and teacher_tokenizer.pad_token_id is not None else 0
                                    _v_eos = teacher_tokenizer.eos_token_id if teacher_tokenizer is not None else None
                                    if _v_eos is None or _v_pad != _v_eos:
                                        val_shift_labels = val_shift_labels.masked_fill(val_shift_labels == _v_pad, -100)
                                    val_batch_loss = F.cross_entropy(
                                        val_shift_logits.view(-1, cfg.vocab_size),
                                        val_shift_labels.view(-1),
                                        ignore_index=-100
                                    )
                                    val_loss_local += val_batch_loss.item()
                                    val_samples_local += 1
                        except Exception as e:
                            print(f"⚠️  Validation error: {e}, skipping early stopping check")
                            val_loss_local = 0.0
                            val_samples_local = 0

                        # DDP-safe global validation aggregation (all ranks agree on one loss value).
                        val_stats = torch.tensor(
                            [float(val_loss_local), float(val_samples_local)],
                            device=student_device,
                            dtype=torch.float32,
                        )
                        val_stats = accelerator.reduce(val_stats, reduction="sum")
                        val_loss_sum_global = float(val_stats[0].item())
                        val_samples_global = int(val_stats[1].item())

                        should_stop = False
                        if accelerator.is_main_process:
                            if val_samples_global > 0:
                                val_loss = val_loss_sum_global / max(1, val_samples_global)
                                latest_val_loss = float(val_loss)
                                print(f"📊 Validation Loss: {val_loss:.4f} (Best: {best_val_loss:.4f})")

                                # EVAL-DRIVEN EARLY STOPPING
                                if val_loss < best_val_loss:
                                    previous_best_val_loss = best_val_loss
                                    patience_counter = 0
                                    print("✅ New best validation loss! Saving checkpoint...")
                                    unwrapped_model = accelerator.unwrap_model(student)
                                    best_val_loss = save_checkpoint_smart(
                                        unwrapped_model,
                                        opt,
                                        scheduler,
                                        global_step,
                                        cfg,
                                        keep_last_n=3,
                                        val_loss=val_loss,
                                        best_val_loss=previous_best_val_loss,
                                    )
                                else:
                                    patience_counter += 1
                                    print(f"⏳ No improvement ({patience_counter}/{early_stop_patience})")
                                    if (not open_ended_mode) and patience_counter >= early_stop_patience:
                                        print(f"🛑 Early stopping triggered! Best val loss: {best_val_loss:.4f}")
                                        should_stop = True
                            else:
                                print("⚠️  Validation produced zero usable batches; skipping early stopping check.")

                        stop_tensor = torch.tensor(
                            1 if should_stop else 0,
                            device=student_device,
                            dtype=torch.int64,
                        )
                        if torch.distributed.is_available() and torch.distributed.is_initialized():
                            torch.distributed.broadcast(stop_tensor, src=0)
                        should_stop = bool(stop_tensor.item())

                        if should_stop:
                            early_stopped = True
                            student.train()
                            break

                        student.train()

                    if global_step % cfg.save_interval == 0 and accelerator.is_main_process:
                        unwrapped_model = accelerator.unwrap_model(student)
                        best_val_loss = save_checkpoint_smart(unwrapped_model, opt, scheduler, global_step, cfg, keep_last_n=3, val_loss=None, best_val_loss=best_val_loss)

                    # Open-ended saturation stop gate (active only after target token floor).
                    if open_ended_mode and global_step % saturation_eval_interval_steps == 0 and global_step > 0:
                        saturation_should_stop = False
                        if accelerator.is_main_process:
                            if tokens_seen_total < target_tokens_min:
                                print(
                                    f"🧮 Saturation gate inactive: tokens_seen={tokens_seen_total} "
                                    f"< target_tokens_min={target_tokens_min}"
                                )
                            else:
                                golden_summary_path = project_root / "reports" / "benchmarks" / "golden_summary.json"
                                gsm8k_summary_path = project_root / "reports" / "benchmarks" / "gsm8k_summary.json"
                                golden_score = read_metric_from_json(golden_summary_path, "assertion_score")
                                gsm8k_score = read_metric_from_json(gsm8k_summary_path, "accuracy")

                                signals_ready = (
                                    latest_val_loss is not None
                                    and golden_score is not None
                                    and gsm8k_score is not None
                                )

                                if not signals_ready:
                                    saturation_plateau_windows += 1
                                    print(
                                        "🟡 Saturation gate waiting: requires val + golden_summary + gsm8k_summary. "
                                        f"window={saturation_plateau_windows}/{saturation_patience_windows}"
                                    )
                                    if saturation_plateau_windows >= saturation_patience_windows:
                                        saturation_should_stop = True
                                else:
                                    val_improved = False
                                    if best_val_saturation is None:
                                        best_val_saturation = latest_val_loss
                                        val_improved = True
                                    else:
                                        rel_drop = (best_val_saturation - latest_val_loss) / max(
                                            abs(best_val_saturation), 1e-8
                                        )
                                        if rel_drop >= val_improve_min_rel:
                                            best_val_saturation = latest_val_loss
                                            val_improved = True

                                    golden_improved = False
                                    if best_golden_score is None:
                                        best_golden_score = golden_score
                                        golden_improved = True
                                    elif (golden_score - best_golden_score) >= golden_improve_min_abs:
                                        best_golden_score = golden_score
                                        golden_improved = True

                                    gsm8k_improved = False
                                    if best_gsm8k_score is None:
                                        best_gsm8k_score = gsm8k_score
                                        gsm8k_improved = True
                                    elif (gsm8k_score - best_gsm8k_score) >= gsm8k_improve_min_abs:
                                        best_gsm8k_score = gsm8k_score
                                        gsm8k_improved = True

                                    if val_improved or golden_improved or gsm8k_improved:
                                        saturation_plateau_windows = 0
                                    else:
                                        saturation_plateau_windows += 1

                                    print(
                                        "📉 Saturation window | "
                                        f"plateau={saturation_plateau_windows}/{saturation_patience_windows} | "
                                        f"val={latest_val_loss:.5f} | golden={golden_score:.4f} | gsm8k={gsm8k_score:.4f}"
                                    )

                                    if saturation_plateau_windows >= saturation_patience_windows:
                                        saturation_should_stop = True

                                write_training_runtime_manifest(
                                    {
                                        "status": "running",
                                        "global_step": int(global_step),
                                        "tokens_seen": int(tokens_seen_total),
                                        "token_budget_mode": token_budget_mode,
                                        "target_tokens_min": int(target_tokens_min),
                                        "latest_val_loss": latest_val_loss,
                                        "best_val_loss": float(best_val_loss) if best_val_loss != float("inf") else None,
                                        "golden_summary_path": str(golden_summary_path),
                                        "gsm8k_summary_path": str(gsm8k_summary_path),
                                        "plateau_windows": int(saturation_plateau_windows),
                                        "saturation_patience_windows": int(saturation_patience_windows),
                                    }
                                )

                        saturation_tensor = torch.tensor(
                            1 if saturation_should_stop else 0,
                            device=student_device,
                            dtype=torch.int64,
                        )
                        if torch.distributed.is_available() and torch.distributed.is_initialized():
                            torch.distributed.broadcast(saturation_tensor, src=0)
                        if bool(saturation_tensor.item()):
                            early_stopped = True
                            safety_brake_reason = "quality_saturation_plateau"
                            print("🛑 Saturation gate triggered stop (val+golden+gsm8k plateau).")
                            break

                    micro_step += 1
            except RuntimeError as runtime_exc:
                err_text = str(runtime_exc).lower()
                if "out of memory" in err_text:
                    consecutive_oom_backoff_fail += 1
                    print(
                        f"⚠️ OOM detected ({consecutive_oom_backoff_fail}/{max_consecutive_oom_backoff_fail}). "
                        "Clearing cache..."
                    )
                    opt.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # [H1] A single-rank OOM retry desyncs DDP (peers hang at the
                    #     collective backward/clip -> NCCL timeout). No safe multi-GPU
                    #     retry -> clean safety-brake stop; operator resumes with a
                    #     smaller batch. Single-process retry stays safe.
                    if accelerator.num_processes > 1:
                        safety_brake_triggered = True
                        safety_brake_reason = "oom_multigpu_no_safe_retry"
                        print("🛑 SAFETY BRAKE: multi-GPU OOM cannot be retried safely (DDP desync). Stopping.")
                        break
                    if consecutive_oom_backoff_fail >= max_consecutive_oom_backoff_fail:
                        safety_brake_triggered = True
                        safety_brake_reason = "oom_backoff_exhausted"
                        print("🛑 SAFETY BRAKE: OOM backoff retry budget exhausted.")
                        break
                    continue
                raise

        if accelerator.is_main_process:
            if safety_brake_triggered:
                print(f"🛑 Safety brake stop finalized. reason={safety_brake_reason}")
                unwrapped_model = accelerator.unwrap_model(student)
                best_val_loss = save_checkpoint_smart(
                    unwrapped_model,
                    opt,
                    scheduler,
                    global_step,
                    cfg,
                    keep_last_n=3,
                    val_loss=None,
                    best_val_loss=best_val_loss,
                    write_final=True,
                )
                if logger:
                    logger.finalize(
                        "safety_brake_stop",
                        extra={
                            "reason": safety_brake_reason,
                            "best_val_loss": best_val_loss,
                            "tokens_seen": int(tokens_seen_total),
                        },
                    )
            elif early_stopped:
                unwrapped_model = accelerator.unwrap_model(student)
                best_val_loss = save_checkpoint_smart(
                    unwrapped_model,
                    opt,
                    scheduler,
                    global_step,
                    cfg,
                    keep_last_n=3,
                    val_loss=None,
                    best_val_loss=best_val_loss,
                    write_final=True,
                )
                if logger:
                    logger.finalize(
                        "early_stopped",
                        extra={"best_val_loss": best_val_loss, "tokens_seen": int(tokens_seen_total)},
                    )
            else:
                unwrapped_model = accelerator.unwrap_model(student)
                best_val_loss = save_checkpoint_smart(
                    unwrapped_model,
                    opt,
                    scheduler,
                    global_step,
                    cfg,
                    keep_last_n=3,
                    val_loss=None,
                    best_val_loss=best_val_loss,
                    write_final=True,
                )
                if logger:
                    logger.finalize(
                        "completed",
                        extra={"best_val_loss": best_val_loss, "tokens_seen": int(tokens_seen_total)},
                    )
                export_to_onnx(unwrapped_model, save_dir, cfg.model_name, student_device)

    except KeyboardInterrupt:
        print("\n🛑 Durduruldu. Kaydediliyor...")
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(student)
            best_val_loss = save_checkpoint_smart(unwrapped_model, opt, scheduler, global_step, cfg, keep_last_n=3, val_loss=None, best_val_loss=best_val_loss)
            if 'logger' in locals() and logger: logger.finalize("aborted")

    except Exception as e:
        print(f"\n💥 HATA: {e}")
        if 'accelerator' in locals() and accelerator.is_main_process and 'logger' in locals() and logger:
            logger.finalize("failed", extra={"error": str(e)})
        raise


if __name__ == "__main__":
    train()
