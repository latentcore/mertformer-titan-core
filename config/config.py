"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) - Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

__version__ = "1.0-BUILD30-V2"
__author__ = "Mert"

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _cfg_verbose() -> bool:
    """Return True if config module should emit console output."""
    return os.environ.get("TITAN_CONFIG_VERBOSE", "0") == "1"


def _cfg_print(msg: str) -> None:
    if _cfg_verbose():
        print(msg)


def get_auto_dtype() -> Any:
    """
    Automatically selects the best and safest data type based on hardware.

    Returns:
        torch.dtype: Selected data type
    """
    # 1. NVIDIA Ampere and Above (A100, H100, 3090, 4090) -> FAST
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16

    # 2. Apple Silicon (M1, M2, M3, M4) -> SAFE
    # To avoid NaN errors during training on Mac,
    # using Float32 is the safest way.
    elif torch.backends.mps.is_available():
        return torch.float32

    # 3. Old GPU or CPU -> STANDARD
    else:
        return torch.float32


def auto_configure_batch_size(target_global_batch: int = 128, conf: Any = None) -> tuple[int, int]:
    """
    GRANDMASTER AUTO-PILOT - Physics-based VRAM calculation & optimization.
    
    Logic:
    1. Deep Hardware Inspection (Torch -> Nvidia-SMI -> Fallback)
    2. Static Memory Calculation (Params + Grads + Optimizer States)
    3. Dynamic Activation Modeling (Based on SeqLen, HiddenSize, Layers)
    4. Binary Search for Optimal Micro-Batch
    """
    import subprocess
    
    # -------------------------------------------------------------------------
    # 1. DEEP HARDWARE INSPECTION
    # -------------------------------------------------------------------------
    gpu_memory_gb = 0.0
    num_gpus = 0
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        # Method A: PyTorch Properties
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            pass
            
    # Method B: Nvidia-SMI (Cross-verification)
    if gpu_memory_gb == 0.0:
        try:
            cmd = "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits"
            result = subprocess.check_output(cmd.split(), encoding="utf-8")
            gpu_memory_gb = float(result.strip().split('\n')[0]) / 1024
            if num_gpus == 0: num_gpus = 1 # Assume 1 if SMI works
        except:
            pass

    # Critical Decision fallback
    if gpu_memory_gb < 1.0:
        _cfg_print("\n⛔ CRITICAL: NO GPU DETECTED OR VRAM UNREADABLE.")
        _cfg_print("   -> Switching to CPU/MPS Safe Mode (Very Slow)")
        return 1, target_global_batch  # Micro=1, Accum=Target
        
    _cfg_print(f"🔍 DEEP SCAN: {num_gpus}x GPU Detected | VRAM: {gpu_memory_gb:.2f} GB")

    # -------------------------------------------------------------------------
    # 2. PHYSICS-BASED MEMORY MODELING
    # -------------------------------------------------------------------------
    try:
        # Use provided config values if available (avoid recursive instantiation)
        use_8bit_adam = getattr(conf, "use_8bit_adam", True) if conf is not None else True
        max_seq_len = getattr(conf, "max_seq_len", 4096) if conf is not None else 4096

        # [P7] Measured dense total (~3.67B) for VRAM math; was the 2.64e9 design-target.
        # CUDA-only path (gated above); never runs on Mac/MPS. Does not change the published
        # design-target DEFAULT_PARAMS=2.64e9 in economics/flops_estimator.py.
        total_params = 3.673 * 10**9

        # A. Static Memory (Fixed Cost)
        # Weights (BF16=2 bytes) + Grads (BF16=2 bytes) = 4 bytes per param
        # Optimizer (8-bit Adam = 2 bytes state, 32-bit = 8 bytes state)
        bytes_per_param_static = 4 + (2 if use_8bit_adam else 8)

        static_mem_gb = (total_params * bytes_per_param_static) / (1024**3)
        # DDP Sharding Savings (Zero-1/2 equivalent effect estimate)
        if num_gpus > 1:
            static_mem_gb /= (num_gpus ** 0.5)  # Conservative sharding benefit

        static_mem_gb += 1.5  # CUDA Context + PyTorch Overhead buffer
    except Exception:
        # Fallback if config is not available
        static_mem_gb = 6.0
        max_seq_len = 4096

    # B. Dynamic Memory (Per Sample Cost)
    # Formula: Layers * SeqLen * Hidden * Batch * Buffer
    # Estimate: ~2MB per token for 2B model activations without checkpointing
    # With FlashAttn + Checkpointing: ~0.2MB per token
    token_mem_bytes = 0.35 * 1024 * 1024 # 0.35 MB per token (Empirical for 2.6B + GC)
    mem_per_sample_gb = (max_seq_len * token_mem_bytes) / (1024**3)

    # -------------------------------------------------------------------------
    # 3. SOLVER: FIND OPTIMAL MICRO-BATCH
    # -------------------------------------------------------------------------
    available_vram = gpu_memory_gb * 0.93 # Leave 7% Safety Margin (System UI etc)
    usable_vram = available_vram - static_mem_gb
    
    if usable_vram <= 0:
        _cfg_print("⚠️  WARNING: Model might be too large for this GPU even with Batch=1!")
        max_micro_batch = 1
    else:
        max_micro_batch = int(usable_vram / mem_per_sample_gb)
        max_micro_batch = max(1, max_micro_batch)

    # -------------------------------------------------------------------------
    # 4. BUILD 27 OPTIMIZATION
    # -------------------------------------------------------------------------
    ideal_micro_batch = target_global_batch // max(1, num_gpus)
    
    # Select best power of 2 that fits
    micro_batch = 1
    for p in [1, 2, 4, 8, 16, 32, 64, 128]:
        if p <= max_micro_batch and p <= ideal_micro_batch:
            micro_batch = p
            
    # Calculate Accumulation
    needed_total_micro = target_global_batch // max(1, num_gpus)
    grad_accum = max(1, needed_total_micro // micro_batch)
    
    final_global = micro_batch * grad_accum * max(1, num_gpus)
    
    # Report
    _cfg_print(f"📊 AUTO-PILOT SOLUTION:")
    _cfg_print(f"   ► Static Memory  : {static_mem_gb:.2f} GB (Weights+Opt+Ctx)")
    _cfg_print(f"   ► Sample Cost    : {mem_per_sample_gb:.2f} GB (seq_len={max_seq_len})")
    _cfg_print(f"   ► Max Fits VRAM  : {max_micro_batch} samples")
    _cfg_print(f"   ► Selected Micro : {micro_batch}")
    _cfg_print(f"   ► Grad Accum     : {grad_accum}")
    _cfg_print(f"   ► Final Global   : {final_global}")
    
    return micro_batch, grad_accum





@dataclass
class MertFormerConfig:
    # -------------------------------------------------------------------------
    # 1. SYSTEM IDENTITY
    # -------------------------------------------------------------------------
    model_name: str = "MertFormer_Titan_S25_Prod"
    version: str = "v1.0-TITAN-BUILD30-V2"

    # Auto-detect device (First NVIDIA, then Mac MPS, then CPU)
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    seed: int = 1453

    # -------------------------------------------------------------------------
    # 2. DIMENSIONS (DUAL NAMING - PREVENTS ERRORS)
    # -------------------------------------------------------------------------
    # Vocab & Seq
    vocab_size: int = 128256
    max_seq_len: int = 4096

    # Model Width
    hidden_size: int = 2048
    intermediate_size: int = 5632

    # [CRITICAL FIX] We define both names that the code looks for:
    num_hidden_layers: int = 18 # [USER OPT] S25/Mobile compatibility
    num_layers: int = 18  # <-- train code looks for this

    num_attention_heads: int = 16
    num_heads: int = 16  # <-- mla.py looks for this

    num_kv_heads: int = 8
    # [MATH FIX] 2048 / 16 = 128.
    head_dim: int = 128

    # Normalization and Dropout
    rms_norm_eps: float = 1e-6
    dropout: float = 0.1

    # [NEW] RoPE Theta: Frequency base for long context.
    # Standard is 10,000. Making it 100,000 eases "stretching" to 8K/16K later.
    rope_theta: float = 100000.0
    rope_base: float = 100000.0 # [FIX] Sync with rope_theta

    # -------------------------------------------------------------------------
    # 3. BITNET b1.58
    # -------------------------------------------------------------------------
    use_bitnet: bool = True  # BitNet active
    weight_quantization: str = "absmax_per_tensor"  # Weight quantize
    activation_bits: int = 8  # Activation bits

    # -------------------------------------------------------------------------
    # 4. SPARSE MoE
    # -------------------------------------------------------------------------
    use_moe: bool = True
    num_experts: int = 8
    num_experts_per_tok: int = 2
    active_experts: int = 2  # Alias for num_experts_per_tok (backward compatibility)
    # [P4] Explicit MoE expert FFN width. Previously a silent getattr default in
    # layers/moe.py (hidden_size*4 = 2048*4). Made explicit here at the SAME value so the
    # measured 3,672,982,022 (~3.67B) param count is unchanged — do NOT alter this number.
    moe_intermediate: int = 8192
    router_aux_loss_coef: float = 0.02
    aux_loss_coef: float = 0.02  # Alias for router_aux_loss_coef (backward compatibility)
    moe_every_n_layers: int = 3  # MoE on layers: 3, 6, 9, 12, 15, 18, 21 (0-indexed: 2, 5, 8, 11, 14, 17, 20)
    # Inference-only expert paging (keeps model math intact; optimizes residency)
    use_expert_paging: bool = False
    expert_paging_inference_only: bool = True
    expert_paging_lazy_init: bool = True
    expert_paging_cache_size: int = 2
    expert_paging_offload_device: str = "cpu"
    expert_paging_verbose: bool = False

    # V26.5: Switch Loss Option & Router Jitter for collapse prevention
    use_switch_loss: bool = True  # [AUDIT FIX] Enable Aggressive Load Balancing
    router_jitter: float = 0.02  # [AUDIT FIX] Increase exploration noise
    router_jitter_boost: float = 0.1  # Emergency jitter when collapse detected
    router_alarm_threshold: float = 0.40  # Early warning level for top-k load imbalance

    # -------------------------------------------------------------------------
    # 5. LIQUID LAYERS
    # -------------------------------------------------------------------------
    use_liquid: bool = True
    # [REPORT FIX] "cfc" instead of "euler" (the code already implements CfC; name corrected)
    liquid_solver: str = "cfc"
    liquid_layers_idx: list = field(default_factory=lambda: [4, 10, 16])  # [FIX] Adjusted for 18 layers (No OOB)
    liquid_every_n_layers: int = 0  # Disabled (using explicit indices instead)

    # Router Z-Loss for stability
    z_loss_coef: float = 1e-4
    # MoE capacity control (Switch-style overflow guard)
    moe_capacity_enforce: bool = True
    moe_capacity_factor: float = 1.25

    # V25.1 SAFEGUARD: Liquid Warmup Steps (Freeze for first N steps)
    liquid_warmup_steps: int = 10000
    # V26.11 SAFEGUARD: Liquid spike tracking (3-strike rule)
    liquid_spike_threshold: float = 5.0
    liquid_spike_patience: int = 3
    liquid_spike_cooldown_steps: int = 200

    # -------------------------------------------------------------------------
    # 6. SAFETY FUSE (QINN)
    # -------------------------------------------------------------------------
    use_qinn: bool = False # [AUDIT FIX] Disabled for NPU compatibility & Speed

    # -------------------------------------------------------------------------
    # 6.1 ADVANCED COGNITIVE EXTENSIONS (feature-flag, non-breaking defaults)
    # -------------------------------------------------------------------------
    # Hierarchical KV Cache (short/long split view for decode)
    use_hierarchical_kv_cache: bool = False
    hkv_short_window: int = 512
    hkv_long_stride: int = 8
    hkv_max_long_blocks: int = 128

    # Global workspace & neuromodulation
    use_global_workspace_broadcast: bool = False
    workspace_blend: float = 0.7
    use_neuromodulatory_gain: bool = False

    # Continuous-time latent state channel
    use_latent_ode_state_channel: bool = False
    latent_ode_dt: float = 1.0

    # MoE cross-expert sync and structural plasticity
    use_cross_expert_sync_bus: bool = False
    cross_expert_sync_gain: float = 0.05
    use_structural_plasticity: bool = False
    structural_ema_decay: float = 0.98
    structural_prune_threshold: float = 0.02
    structural_grow_threshold: float = 0.60
    structural_update_interval: int = 100

    # Hebbian and neuro-symbolic extensions
    use_hebbian_plasticity: bool = False
    hebbian_eta: float = 0.01
    hebbian_decay: float = 0.99
    use_neuro_symbolic_layer: bool = False
    neuro_symbolic_rules: int = 8
    use_world_model_head: bool = False
    world_model_horizon: int = 1
    use_lifelong_safety_layer: bool = False
    lifelong_ema_decay: float = 0.99
    lifelong_max_adaptation_gain: float = 0.05
    lifelong_drift_threshold: float = 0.35

    # Continual learning adapter (offline-safe scaffold)
    use_continual_adapter: bool = False
    continual_replay_capacity: int = 2048
    continual_drift_threshold: float = 0.2
    continual_loss_ema_decay: float = 0.98

    # Validation gate policy
    validation_min_samples_warn: int = 128
    validation_min_samples_claim: int = 1000

    # -------------------------------------------------------------------------
    # 6.2 TRAINING READINESS POLICY (PORTABLE TRAINING CONTRACT)
    # -------------------------------------------------------------------------
    # Single source of truth for curriculum stages and stage ratios
    curriculum_stage_names: list[str] = field(
        default_factory=lambda: [
            "stage1",
            "stage2",
            "stage3",
            "stage4",
            "stage5",
        ]
    )
    curriculum_stage_ratios: list[float] = field(
        # V2 curriculum rebalance: higher instruction + tool-use share.
        default_factory=lambda: [0.42, 0.30, 0.08, 0.08, 0.12]
    )

    # Token-first budget (env-overridable to support fixed-steps production profile)
    # Default now fixed_steps-aligned with 45k schedule (23.59296B tokens @ 128x4096).
    target_tokens_min: int = int(os.environ.get("TITAN_TARGET_TOKENS_MIN", 23_600_000_000))
    token_budget_mode: str = os.environ.get("TITAN_TOKEN_BUDGET_MODE", "fixed_steps")  # {"open_ended", "fixed_steps"}
    estimated_tokens_per_sample: int = 512

    # Dataset dedup policy (V2)
    dedup_enabled: bool = os.environ.get("TITAN_DEDUP_ENABLED", "1") == "1"
    dedup_scope: str = os.environ.get("TITAN_DEDUP_SCOPE", "global")  # {"global", "stage"}
    dedup_hash_bytes: int = int(os.environ.get("TITAN_DEDUP_HASH_BYTES", 8))
    dedup_max_entries: int = int(os.environ.get("TITAN_DEDUP_MAX", 2_000_000))
    dedup_normalize: bool = os.environ.get("TITAN_DEDUP_NORMALIZE", "1") == "1"
    allow_optional_sources: bool = os.environ.get("TITAN_ALLOW_OPTIONAL_SOURCES", "0") == "1"
    token_probe_samples: int = int(os.environ.get("TITAN_TOKEN_PROBE_SAMPLES", 64))

    # Distillation policy: canonical offline_clean lane is strict precomputed KD.
    require_gated_teacher: bool = os.environ.get("TITAN_REQUIRE_GATED_TEACHER", "1") == "1"

    # -------------------------------------------------------------------------
    # [USER OVERRIDE] Teacher Model configuration
    # User confirmed usage of Llama 3.3 70B (assumes A100/H100 or multi-gpu setup)
    teacher_model_id: str = os.environ.get("TITAN_TEACHER_MODEL_ID", "meta-llama/Llama-3.3-70B-Instruct")
    distill_alpha: float = 0.8 # [USER OPTIMIZATION] 80% trust in the teacher
    distill_intermediate_layers: bool = True

    # -------------------------------------------------------------------------
    # 7. TOKENIZER OPTIONS (OPT-IN)
    # -------------------------------------------------------------------------
    # Default is teacher tokenizer. Turkish tokenizer is opt-in only.
    use_tr_tokenizer: bool = os.environ.get("TITAN_USE_TR_TOKENIZER", "0") == "1"
    tr_tokenizer_id: str = os.environ.get("TITAN_TR_TOKENIZER_ID", "tokenizer/tr")

    # [B2] Sequence packing (EOS-separated, filled to max_seq_len) and precompute
    #     teacher-logit alignment verification. Both default ON. Packing must run on
    #     BOTH teacher and student via the single train/packing.py source.
    sequence_packing: bool = os.environ.get("TITAN_SEQUENCE_PACKING", "1") == "1"
    verify_logit_alignment: bool = os.environ.get("TITAN_VERIFY_LOGIT_ALIGNMENT", "1") == "1"

    # [CRITICAL FIX]
    # [REPORT FIX] 1.3 -> 1.0 (BitNet requires a sharp teacher)
    # [REPORT FIX] 1.3 -> 1.0 (BitNet requires a sharp teacher)
    teacher_temp: float = 1.0
    
    # [V27.0] Distillation Optimization
    use_precomputed_logits: bool = os.environ.get("TITAN_USE_PRECOMPUTED_LOGITS", "1") == "1"
    # [TITAN PREFLIGHT] Support override for testing
    precomputed_logits_path: str = os.environ.get("TITAN_LOGITS_PATH", "./datasets/logits/")
    


    # -------------------------------------------------------------------------
    # 8. HYPERPARAMETERS
    # -------------------------------------------------------------------------
    batch_size: int = int(os.environ.get("TITAN_BATCH_SIZE", "128"))
    
    # [V27.0 AUTO-CONFIG] Automatic GPU-based batch size optimization
    # micro_batch_size and grad_accum_steps are now computed automatically
    micro_batch_size: Optional[int] = field(default=None)  # Auto-configured
    grad_accum_steps: Optional[int] = field(default=None)  # Auto-configured
    
    def __post_init__(self):
        """Post-initialization: Auto-configure batch sizes if not set."""
        if self.micro_batch_size is None or self.grad_accum_steps is None:
            auto_micro, auto_accum = auto_configure_batch_size(target_global_batch=self.batch_size, conf=self)
            if self.micro_batch_size is None:
                self.micro_batch_size = auto_micro
            if self.grad_accum_steps is None:
                self.grad_accum_steps = auto_accum
        
        # [TITAN AUTO-TUNE] Adjust workers based on CPU cores
        try:
            import os
            cpu_count = os.cpu_count() or 4
            self.dataloader_num_workers = min(self.dataloader_num_workers, cpu_count)
        except:
            pass


    learning_rate: float = 1.5e-3
    weight_decay: float = 0.1
    warmup_steps: int = 3000  # [TITAN SCALE-UP] Adjusted for 45k total steps

    # Runtime fast paths (V2)
    liquid_fast_path: bool = os.environ.get("TITAN_LIQUID_FAST_PATH", "1") == "1"
    liquid_train_impl: str = os.environ.get("TITAN_LIQUID_TRAIN_IMPL", "baseline")
    moe_dispatch_mode: str = os.environ.get("TITAN_MOE_DISPATCH", "parallel")  # {"parallel", "sequential"}
    use_flash_attn_inference: bool = os.environ.get("TITAN_FLASH_ATTN_INFER", "0") == "1"
    # [TITAN PREFLIGHT] Support env var override for testing
    max_steps: int = int(os.environ.get("TITAN_MAX_STEPS", 45000))  

    # [V26.5 FIX] Explicitly disable Epoch Mode to respect max_steps=50k
    epoch_mode: bool = False

    min_lr_ratio: float = 0.1  # Minimum LR ratio for cosine decay
    # Reproducibility metadata strictness
    write_run_manifest: bool = True

    # Gradient Clipping
    # [REPORT FIX] 1.0 -> 2.0 (STE produces harsh gradients; some slack is needed)
    grad_clip: float = 2.0

    # UPGRADE: Early Stopping & Validation
    early_stop_patience: int = 5  # Stop if no improvement for N validation checks
    val_check_interval: int = int(os.environ.get("TITAN_VAL_CHECK_INTERVAL", "1000"))  # Run validation every N steps
    # [16] Number of validation micro-batches per check. Default 10 (behavior-preserving);
    # raise via TITAN_VAL_STEPS for a less noisy early-stopping signal on claim-grade runs.
    val_steps: int = int(os.environ.get("TITAN_VAL_STEPS", "10"))
    saturation_eval_interval_steps: int = 2000
    saturation_patience_windows: int = 3
    val_improve_min_rel: float = 0.002
    golden_improve_min_abs: float = 0.01
    gsm8k_improve_min_abs: float = 0.002
    max_consecutive_nan: int = 3
    max_consecutive_oom_backoff_fail: int = 5

    # -------------------------------------------------------------------------
    # 9. OUTPUT FORMAT
    # -------------------------------------------------------------------------
    output_dir: str = "./checkpoints/mertformer_titan_prod"
    save_dir: str = "./checkpoints/mertformer_titan_prod"

    # Logging Frequency
    log_interval: int = int(os.environ.get("TITAN_LOG_INTERVAL", "1"))
    save_interval: int = int(os.environ.get("TITAN_SAVE_INTERVAL", "1000"))

    export_format: str = "onnx_dynamic"

    # [UNIVERSAL AUTO-PILOT]
    # Selects Float32 on Mac, Bfloat16 on A100.
    param_dtype: Any = field(default_factory=get_auto_dtype)

    # FIX: Mixed Precision Training
    use_amp: bool = field(default_factory=lambda: torch.cuda.is_available())  # Auto-enable only on CUDA

    # [V26.5 DDP] Use Hugging Face Accelerate
    use_accelerate: bool = True

    # FIX: DataLoader Optimization
    dataloader_num_workers: int = 8 # [TITAN SPEED BOOST] Optimized for 8x A100
    dataloader_prefetch_factor: int = 2
    dataloader_pin_memory: bool = os.environ.get("TITAN_DATALOADER_PIN", "1") == "1"
    dataloader_non_blocking: bool = os.environ.get("TITAN_DATALOADER_NONBLOCKING", "1") == "1"

    # TITAN ONYX STORM: Advanced Training Features
    freeze_core_layers: bool = False  # If True, freeze everything except MoE Router and Liquid Layers

    # -------------------------------------------------------------------------
    # 10. V22.0 UPGRADES: Gradient Checkpointing, Label Smoothing, Attention Dropout
    # -------------------------------------------------------------------------
    # Gradient Checkpointing: 40% VRAM savings, for large batches
    use_gradient_checkpointing: bool = True

    # [SAFETY FIRST] Disable torch.compile for the first run to ensure dynamic routing works
    use_torch_compile: bool = False
    torch_compile_mode: str = "max-autotune"

    # [V27.0] Advanced Optimizers (VRAM Optimization)
    use_8bit_adam: bool = True  # Enable 8-bit Optimizer
    use_galore: bool = True     # Enable Gradient Low-Rank Projection

    # Label Smoothing: Overfitting prevention (0.0 = off, 0.1 = recommended)
    label_smoothing: float = 0.1

    # Attention Dropout: Regularization (used during training, off at inference)
    attention_dropout: float = 0.1

    # -------------------------------------------------------------------------
    # 11. FINAL CLOSURE FLAGS (Config-driven operations)
    # -------------------------------------------------------------------------
    enable_cpp_kernel: bool = True
    enable_metal_kernel: bool = True
    enable_vulkan_kernel: bool = True
    enable_npu_direct: bool = True
    enable_zero_copy: bool = True

    enable_formal_gate: bool = True
    enable_doc_quality_gate: bool = True
    enable_language_quality_gate: bool = True
    enable_release_lock: bool = True

    ram_hard_ceiling_gb: float = 12.0
    benchmark_profile: str = "safe"  # safe|full


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Best-effort YAML loader. Returns empty dict if missing or unavailable."""
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def _resolve_config_path(config_dir: Path, name: str) -> Path:
    """Resolve config path relative to config_dir unless absolute."""
    p = Path(name)
    return p if p.is_absolute() else (config_dir / name)


def _load_config_overlays() -> Dict[str, Any]:
    """Load base + optional overlays from config/*.yaml (env-controlled)."""
    config_dir = Path(__file__).resolve().parent
    merged: Dict[str, Any] = {}

    base_override = os.environ.get("MERTFORMER_CONFIG")
    if base_override:
        base_path = _resolve_config_path(config_dir, base_override)
        merged.update(_load_yaml(base_path))

    model_override = os.environ.get("MERTFORMER_MODEL_CONFIG")
    if model_override:
        merged.update(_load_yaml(_resolve_config_path(config_dir, f"model/{model_override}")))

    train_override = os.environ.get("MERTFORMER_TRAIN_CONFIG")
    if train_override:
        merged.update(_load_yaml(_resolve_config_path(config_dir, f"train/{train_override}")))

    export_override = os.environ.get("MERTFORMER_EXPORT_CONFIG")
    if export_override:
        merged.update(_load_yaml(_resolve_config_path(config_dir, f"export/{export_override}")))

    return merged


def _apply_overrides(cfg: MertFormerConfig, overrides: Dict[str, Any]) -> None:
    """Apply flat key overrides to config instance."""
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)


def _finalize_config(cfg: MertFormerConfig) -> None:
    """Finalize config after overrides (batch size auto-tune, worker cap)."""
    if cfg.micro_batch_size is None or cfg.grad_accum_steps is None:
        auto_micro, auto_accum = auto_configure_batch_size(target_global_batch=cfg.batch_size, conf=cfg)
        if cfg.micro_batch_size is None:
            cfg.micro_batch_size = auto_micro
        if cfg.grad_accum_steps is None:
            cfg.grad_accum_steps = auto_accum

    try:
        cpu_count = os.cpu_count() or 4
        cfg.dataloader_num_workers = min(cfg.dataloader_num_workers, cpu_count)
    except Exception:
        pass
    _validate_training_contract(cfg)


def _validate_training_contract(cfg: MertFormerConfig) -> None:
    """
    Validate portable training contract fields after env/config overlays.
    """
    names = list(getattr(cfg, "curriculum_stage_names", []))
    ratios = [float(x) for x in list(getattr(cfg, "curriculum_stage_ratios", []))]
    if not names or not ratios:
        raise ValueError("❌ curriculum_stage_names/curriculum_stage_ratios must be non-empty.")
    if len(names) != len(ratios):
        raise ValueError(
            "❌ curriculum_stage_names and curriculum_stage_ratios length mismatch "
            f"({len(names)} vs {len(ratios)})."
        )
    if len(names) != 5:
        raise ValueError(
            f"❌ Expected 5 curriculum stages for Build30 portability contract, got {len(names)}."
        )
    if any(r <= 0.0 for r in ratios):
        raise ValueError("❌ curriculum_stage_ratios must all be > 0.")
    ratio_sum = sum(ratios)
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"❌ curriculum_stage_ratios must sum to 1.0, got {ratio_sum:.8f}."
        )

    mode = str(getattr(cfg, "token_budget_mode", "open_ended")).strip().lower()
    if mode not in {"open_ended", "fixed_steps"}:
        raise ValueError(
            f"❌ token_budget_mode must be 'open_ended' or 'fixed_steps', got '{mode}'."
        )
    cfg.token_budget_mode = mode
    if int(getattr(cfg, "target_tokens_min", 0)) < 0:
        raise ValueError("❌ target_tokens_min must be >= 0.")
    if int(getattr(cfg, "estimated_tokens_per_sample", 0)) <= 0:
        raise ValueError("❌ estimated_tokens_per_sample must be > 0.")
    if int(getattr(cfg, "token_probe_samples", 0)) <= 0:
        raise ValueError("❌ token_probe_samples must be > 0.")

    # [P3] Non-fatal overshoot guard. In fixed_steps mode, warn (do NOT raise) when the
    # planned budget (max_steps × global batch_size × max_seq_len) significantly EXCEEDS
    # target_tokens_min — e.g. a batch-size fallback that silently inflates the run.
    # Asymmetric on purpose: undershoot (small smoke/test runs) stays silent, so the
    # max_steps=2 test path is unaffected.
    if mode == "fixed_steps":
        planned_tokens = (
            int(getattr(cfg, "max_steps", 0))
            * int(getattr(cfg, "batch_size", 0))
            * int(getattr(cfg, "max_seq_len", 0))
        )
        target_min = int(getattr(cfg, "target_tokens_min", 0))
        if target_min > 0 and planned_tokens > int(target_min * 1.05):
            print(
                f"⚠️  TOKEN BUDGET OVERSHOOT: planned {planned_tokens / 1e9:.2f}B tokens "
                f"(max_steps×batch_size×max_seq_len) exceeds target_tokens_min "
                f"{target_min / 1e9:.2f}B by >5%. Check for an inflated batch_size/max_steps "
                f"(e.g. an OOM batch fallback) before committing to a long run.",
                file=sys.stderr,
            )

    if bool(getattr(cfg, "require_gated_teacher", False)) and not str(
        getattr(cfg, "teacher_model_id", "")
    ).strip():
        raise ValueError("❌ require_gated_teacher=true but teacher_model_id is empty.")


# Config instance
cfg = MertFormerConfig()
_apply_overrides(cfg, _load_config_overlays())
_finalize_config(cfg)

def validate_layer_config(cfg: MertFormerConfig) -> None:
    """
    Validates that Liquid and MoE layers don't conflict.

    Args:
        cfg (MertFormerConfig): Configuration object
    Raises:
        ValueError: If layer overlap detected
    """
    num_layers = cfg.num_layers

    # Get MoE layers
    moe_layers = set()
    if cfg.use_moe and cfg.moe_every_n_layers > 0:
        for i in range(num_layers):
            if (i + 1) % cfg.moe_every_n_layers == 0:
                moe_layers.add(i)

    # Get Liquid layers
    liquid_layers = set()
    if cfg.use_liquid:
        if cfg.liquid_layers_idx:
            liquid_layers = set(cfg.liquid_layers_idx)
        elif cfg.liquid_every_n_layers > 0:
            for i in range(num_layers):
                if (i + 1) % cfg.liquid_every_n_layers == 0:
                    liquid_layers.add(i)

    # Check for overlap
    overlap = moe_layers & liquid_layers
    if overlap:
        raise ValueError(
            f"❌ CRITICAL: Liquid and MoE layers overlap on layers {sorted(overlap)}. "
            f"This will cause state synchronization issues. "
            f"MoE layers: {sorted(moe_layers)}, Liquid layers: {sorted(liquid_layers)}"
        )

    # TITAN: Strictly enforce BF16 for CUDA (S25 memory optimization)
    if cfg.device == "cuda" and torch.cuda.is_bf16_supported():
        if cfg.param_dtype != torch.bfloat16:
            _cfg_print("⚠️  Enforcing bfloat16 for S25 optimization (Overriding User Pref)")
            cfg.param_dtype = torch.bfloat16

# Validate on import
try:
    validate_layer_config(cfg)
    _cfg_print("✅ Layer configuration validated: No Liquid/MoE conflicts")
except ValueError as e:
    raise

if __name__ == "__main__":
    print(f"🔒 MERTFORMER SYSTEM SEALED (Pre-Training): {cfg.model_name}")
    print(f"✅ SAFETY CHECK: QINN={'ON' if cfg.use_qinn else 'OFF'} | MoE={cfg.num_experts}x{cfg.num_experts_per_tok}")
    print(f"⚙️  AUTO-DETECT: Device={cfg.device.upper()} | Dtype={cfg.param_dtype}")
    print(f"🚀 READY TO TRAIN")
