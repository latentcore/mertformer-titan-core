"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Copyright (c) 2026 Mert Yünlü. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) - Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

from config.build_label import BUILD_LABEL as __version__
__author__ = "Mert Yünlü"

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


def _estimate_total_params(conf: Any) -> float:
    """
    Analytical parameter-count estimate from a config object's own fields
    (embeddings + per-layer GQA attention + dense/MoE FFN + norms). Used only for the
    VRAM-budgeting heuristic below, not a claim-grade measured count (that lives in
    ``trainer_core.preflight_param_report()`` / ``FACTS.json``). Reads only ``conf``
    (never the global ``cfg``) to avoid recursive instantiation. Falls back to the
    canonical measured ~3.67B when a field is absent, so a bare ``conf=None`` (or a
    minimal stub) reproduces the previous constant's ballpark.
    """
    vocab_size = getattr(conf, "vocab_size", 128256)
    hidden_size = getattr(conf, "hidden_size", 2048)
    num_layers = getattr(conf, "num_layers", 18)
    num_heads = getattr(conf, "num_heads", 16)
    num_kv_heads = getattr(conf, "num_kv_heads", 8)
    head_dim = getattr(conf, "head_dim", hidden_size // max(num_heads, 1))
    intermediate_size = getattr(conf, "intermediate_size", hidden_size * 4)
    moe_intermediate = getattr(conf, "moe_intermediate", intermediate_size)
    use_moe = getattr(conf, "use_moe", True)
    num_experts = getattr(conf, "num_experts", 8)
    moe_every_n_layers = getattr(conf, "moe_every_n_layers", 3)

    embedding_params = vocab_size * hidden_size

    q_proj = hidden_size * (num_heads * head_dim)
    kv_proj = hidden_size * (num_kv_heads * head_dim)
    o_proj = (num_heads * head_dim) * hidden_size
    attn_per_layer = q_proj + 2 * kv_proj + o_proj

    # NOTE (fixed 2026-07-29): layers/mla.py's GQA also owns two _QKRMSNorm modules
    # (q_norm, k_norm), each a learnable weight of head_dim. Small (2 x 128 per layer)
    # but real, and it was missing.
    attn_per_layer += 2 * head_dim  # q_norm + k_norm

    dense_ffn_per_layer = 3 * hidden_size * intermediate_size
    moe_ffn_per_layer = 3 * hidden_size * moe_intermediate
    # NOTE (fixed 2026-07-29): the router is layers/moe.py's LiquidRouter, not a bare
    # nn.Linear. Besides main_proj (hidden x num_experts) it owns a depthwise Conv1d
    # `fluid_mixer` (groups=hidden, kernel=history_window=4 -> hidden * 4 weights, no
    # bias) and a second projection `fluid_gate` (hidden x num_experts). Counting only
    # main_proj undercounted every MoE layer by hidden * (4 + num_experts).
    history_window = 4  # LiquidRouter.history_window
    router_params = (
        hidden_size * num_experts          # main_proj
        + hidden_size * history_window     # fluid_mixer (depthwise Conv1d, bias=False)
        + hidden_size * num_experts        # fluid_gate
    )
    # NOTE (fixed 2026-07-27): layers/moe.py's MoE always instantiates one additional
    # "shared expert" (BitSwiGLU(hidden_size, moe_intermediate), unconditionally added
    # to every token via a learnable sigmoid gate) -- a 9th, always-present block that
    # is NOT part of num_experts (the routed/sparse pool). Previously omitted here,
    # undercounting every MoE layer by one expert-sized block (+ its scalar gate).
    shared_expert_params = moe_ffn_per_layer + 1  # +1 = the scalar shared_gate param
    moe_layer_params = moe_ffn_per_layer * num_experts + shared_expert_params + router_params

    if use_moe and moe_every_n_layers > 0:
        moe_count = num_layers // moe_every_n_layers
    else:
        moe_count = 0
    dense_count = num_layers - moe_count

    # NOTE (fixed 2026-07-29): the Liquid/CfC mixers were omitted ENTIRELY, and they are
    # the single largest gap -- ~50.35M params at the canonical config, i.e. 1.37% of the
    # model. Each LiquidMixer (layers/liquid.py) holds a LiquidCell with FOUR
    # hidden x hidden BitLinear projections (input_w, hidden_w, tau_input_w,
    # tau_hidden_w, all bias=False), a tau_bias of shape (1, hidden), and an
    # nn.LayerNorm(hidden) (weight + bias). Placement mirrors
    # layers/mertformer_block.py: explicit liquid_layers_idx wins, else the
    # liquid_every_n_layers cadence.
    #
    # Why this mattered: this omission (-50.5M) was silently cancelling the MHA-instead-
    # of-GQA attention overcount (+75.5M) that scripts/scaling_audit_math.py carried, so
    # the two errors together landed within 1% of the measured count and the drift test
    # passed. With both fixed, this function now reproduces the measured
    # 3,672,982,022 exactly at the canonical config.
    use_liquid = getattr(conf, "use_liquid", True)
    liquid_layers_idx = getattr(conf, "liquid_layers_idx", None) or []
    liquid_every_n_layers = int(getattr(conf, "liquid_every_n_layers", 0) or 0)
    if not use_liquid:
        liquid_count = 0
    elif liquid_layers_idx:
        liquid_count = len([i for i in liquid_layers_idx if 0 <= int(i) < num_layers])
    elif liquid_every_n_layers > 0:
        liquid_count = num_layers // liquid_every_n_layers
    else:
        liquid_count = 0
    liquid_per_layer = (
        4 * hidden_size * hidden_size  # input_w, hidden_w, tau_input_w, tau_hidden_w
        + hidden_size                  # tau_bias, shape (1, hidden)
        + 2 * hidden_size              # nn.LayerNorm weight + bias
    )

    total_params = embedding_params
    total_params += num_layers * attn_per_layer
    total_params += dense_count * dense_ffn_per_layer
    total_params += moe_count * moe_layer_params
    total_params += liquid_count * liquid_per_layer
    total_params += num_layers * (2 * hidden_size)  # norms
    total_params += hidden_size  # final norm
    return float(total_params)


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
        except Exception:
            pass
            
    # Method B: Nvidia-SMI (Cross-verification)
    if gpu_memory_gb == 0.0:
        try:
            cmd = "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits"
            result = subprocess.check_output(cmd.split(), encoding="utf-8")
            gpu_memory_gb = float(result.strip().split('\n')[0]) / 1024
            if num_gpus == 0: num_gpus = 1 # Assume 1 if SMI works
        except Exception:
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

        # [P7] Analytically derive param count from conf's own architecture fields
        # (see _estimate_total_params). CUDA-only path (gated above); never runs on
        # Mac/MPS. Does not change the published design-target DEFAULT_PARAMS=2.64e9
        # in economics/flops_estimator.py.
        total_params = _estimate_total_params(conf) if conf is not None else 3.673 * 10**9

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
    token_mem_bytes = 0.35 * 1024 * 1024 # 0.35 MB per token (Empirical for 3.67B + GC)
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
    # 4. Batch-size optimization
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
    num_heads: int = 16  # <-- read by layers/mla.py (which implements GQA, not latent-MLA)

    num_kv_heads: int = 8  # GQA: num_kv_heads(8) < num_heads(16). The "mla" naming is legacy; layers/mla.py is grouped-query attention (GQA), not latent-MLA.
    # [MATH FIX] 2048 / 16 = 128.
    head_dim: int = 128

    # Normalization and Dropout
    rms_norm_eps: float = 1e-6
    dropout: float = 0.1

    # Weight init std for the (tied) token embedding == output projection.
    # Keeps initial logits near-uniform (start loss ~= ln(vocab)) instead of the
    # ~19x-too-high start that nn.Embedding's default N(0,1) x sqrt(hidden) scaling
    # produced. BitLinear bodies keep PyTorch-default (fan-in-scaled) init.
    initializer_range: float = 0.02

    # [NEW] RoPE Theta: Frequency base for long context.
    # Standard is 10,000. Making it 100,000 eases "stretching" to 8K/16K later.
    # NOTE: `rope_theta` is informational only — GQA attention (`layers/mla.py`)
    # reads `rope_base`. Keep both equal; editing `rope_theta` alone has no effect.
    rope_theta: float = 100000.0
    rope_base: float = 100000.0  # canonical RoPE base read by GQA attention (kept == rope_theta)

    # -------------------------------------------------------------------------
    # 3. BITNET b1.58
    # -------------------------------------------------------------------------
    use_bitnet: bool = True  # BitNet active
    weight_quantization: str = "rms_per_channel"  # per-row RMS scale (layers/bitlinear.py weight_quant); ternary {-1,0,+1}
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
    moe_every_n_layers: int = 3  # MoE on layers: 3, 6, 9, 12, 15, 18 (0-indexed: 2, 5, 8, 11, 14, 17)
    # Inference-only expert paging (keeps model math intact; optimizes residency)
    use_expert_paging: bool = False
    expert_paging_inference_only: bool = True
    expert_paging_lazy_init: bool = True
    expert_paging_cache_size: int = 2
    expert_paging_offload_device: str = "cpu"
    expert_paging_verbose: bool = False

    # Switch Loss Option & Router Jitter for collapse prevention
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

    # Router Z-Loss for stability.
    # [2026-07-12 CANDIDATE, unverified by a dedicated GPU re-run — see BACKLOG]
    # z_loss gets folded into aux_loss inside layers/moe.py, then aux_loss is
    # scaled AGAIN by router_aux_loss_coef (0.02) in train/train.py — a
    # double-multiply that left the effective z-loss weight at
    # 1e-4 * 0.02 = 2e-6, ~50x below the ~1e-3 Switch-Transformer/ST-MoE
    # convention (a plain scalar applied once, not double-scaled). Two
    # independent RTX-5070 divergence runs (36M 2026-07-12, 171M 2026-07-12)
    # both showed GradNorm exploding into the trillions in lockstep with MoE
    # Overflow/Balance degrading, while loss stayed clip-protected enough that
    # the loss-based divergence guard never tripped — consistent with a
    # too-weak router-logit regularizer. Value below is chosen so that, AFTER
    # the existing double-multiply, the effective weight lands at 1e-3
    # (0.05 * 0.02 = 1e-3): a single-file, single-line compensating change
    # that does NOT alter the moe.py/train.py double-multiply structure
    # itself (a structural fix is a separate, larger change). This
    # deliberately reopens a 2026-07-08 pre-45K freeze decision on this exact
    # item; NOT verified on real hardware — the next RTX-5070/45K run is the
    # verification event, same discipline as F1/F3.
    z_loss_coef: float = 0.05
    # MoE capacity control (Switch-style overflow guard)
    moe_capacity_enforce: bool = True
    moe_capacity_factor: float = 1.25

    # SAFEGUARD: Liquid Warmup Steps (Freeze for first N steps)
    # env-overridable for parity with the other pre-45K-swept tunables
    # (TITAN_ROUTER_LR_MULT, TITAN_WARMUP_RATIO); default unchanged at 10000.
    liquid_warmup_steps: int = int(os.environ.get("TITAN_LIQUID_WARMUP_STEPS", "10000"))
    # SAFEGUARD: Liquid spike tracking (3-strike rule)
    # [2026-07-08] `liquid_spike_threshold` is now only the COLD-START fallback floor:
    # it is used while the loss EMA is still warming up (fewer than
    # `liquid_spike_ema_warmup_steps` observed optimizer steps). After that the guard
    # switches to the scale-relative test `loss > loss_ema * relative_multiplier`
    # (BACKLOG 2026-07-02 run-feedback: an absolute 5.0 is scale-blind — at a start
    # loss of ~10 it fires every single step, so the Liquid layers never train).
    liquid_spike_threshold: float = 5.0
    liquid_spike_relative_multiplier: float = 1.5
    liquid_spike_ema_warmup_steps: int = 100
    liquid_spike_ema_decay: float = 0.98
    liquid_spike_patience: int = 3
    liquid_spike_cooldown_steps: int = 200

    # -------------------------------------------------------------------------
    # 6. SAFETY FUSE (QINN)
    # -------------------------------------------------------------------------
    use_qinn: bool = False # [AUDIT FIX] Disabled for NPU compatibility & Speed
    # [2026-07-11] Was read only via getattr(cfg, "qinn_every_n_layers", 1) in
    # layers/mertformer_block.py with no backing field anywhere -- so a QINN
    # experiment could never actually configure the placement cadence, it was
    # permanently pinned to 1 (every layer) regardless of intent. Same default,
    # now a real, settable field.
    qinn_every_n_layers: int = 1

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
    # Teacher trust (KD weight). Env-overridable, default 0.8 (unchanged behavior).
    # Set TITAN_DISTILL_ALPHA=0 for a teacher-free smoke: the 70B teacher is never
    # loaded/downloaded (train.py only loads it when distill_alpha > 0) and the loss
    # becomes pure cross-entropy (alpha resolves to 0 when there are no teacher logits).
    distill_alpha: float = float(os.environ.get("TITAN_DISTILL_ALPHA", "0.8"))
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
    
    # Distillation optimization
    use_precomputed_logits: bool = os.environ.get("TITAN_USE_PRECOMPUTED_LOGITS", "1") == "1"
    # [TITAN PREFLIGHT] Support override for testing
    precomputed_logits_path: str = os.environ.get("TITAN_LOGITS_PATH", "./datasets/logits/")
    


    # -------------------------------------------------------------------------
    # 8. HYPERPARAMETERS
    # -------------------------------------------------------------------------
    # Canonical 45K target is batch_size=128 -> ~23.6B tokens (target_tokens_min).
    # TITAN_BATCH_SIZE=1024 (Ocean 1024-first profile) is an EXPLICIT opt-in that
    # yields ~188B tokens (8x); LR/curriculum are step-based and not rescaled. The
    # TITAN_STRICT_TOKEN_BUDGET guard below hard-fails a >5% overshoot. See
    # reports/ocean_2xh200_1024_first_launch_profile.md and DECISIONS.md.
    batch_size: int = int(os.environ.get("TITAN_BATCH_SIZE", "128"))

    # Auto-config: automatic GPU-based batch-size optimization
    # micro_batch_size and grad_accum_steps are now computed automatically
    micro_batch_size: Optional[int] = field(default=None)  # Auto-configured
    grad_accum_steps: Optional[int] = field(default=None)  # Auto-configured
    
    def __post_init__(self):
        """Post-initialization: CPU-worker cap only.

        [2026-07-29] Batch auto-configuration deliberately does NOT happen here any
        more -- it moved to `_finalize_config()`, which runs AFTER the YAML overlays are
        applied. Reason: `__post_init__` fires during `MertFormerConfig()` construction,
        i.e. BEFORE `_apply_overrides()`. It filled in `micro_batch_size` /
        `grad_accum_steps` from the pre-overlay `batch_size`, which then made
        `_finalize_config()`'s `is None` re-computation guard permanently false. So an
        overlay that set `batch_size: 1024` got a micro/accum pair still solved for
        `batch_size=128` -- the run silently trained at 1/8 of the intended global batch,
        and the TITAN_STRICT_TOKEN_BUDGET guard could not see it either because that
        guard reads `cfg.batch_size`, not the realized micro x accum product.

        Constructing MertFormerConfig() directly (tests, scripts) therefore leaves
        micro/accum as None until _finalize_config() is called. The module-level
        singleton below always calls it, so `cfg` is unchanged for every real consumer.
        """
        # [TITAN AUTO-TUNE] Adjust workers based on CPU cores
        try:
            import os
            cpu_count = os.cpu_count() or 4
            self.dataloader_num_workers = min(self.dataloader_num_workers, cpu_count)
        except Exception:
            pass


    # [2026-07-08 pre-45K stabilization] 1.5e-3 -> 3e-4.
    # Sweep STARTING POINT per BACKLOG.md 2026-07-02 run-feedback; NOT independently
    # verified safe. In that pre-flight the grad-norm explosion began at step ~80 of a
    # ~183-step warmup, i.e. at an effective LR of only ~6.6e-4 — the run diverged
    # DURING warmup, at a fraction of the 1.5e-3 peak. So 3e-4 is "next candidate to
    # test", not "known-good". Sweep without a commit via TITAN_LEARNING_RATE.
    learning_rate: float = float(os.environ.get("TITAN_LEARNING_RATE", "3e-4"))
    weight_decay: float = 0.1

    # Router/tau param group LR = learning_rate * router_lr_multiplier.
    # Was hardcoded x1.5 in train/train.py build_optimizer() ("Grokking Setup":
    # resolve the routing logic fast). BACKLOG 2026-07-02 explicitly asks to drop the
    # differential -> new default 1.0. TITAN_ROUTER_LR_MULT=1.5 re-tests the old path.
    router_lr_multiplier: float = float(os.environ.get("TITAN_ROUTER_LR_MULT", "1.0"))

    # Warmup. The scheduler used to hardcode int(max_steps * 0.1) and ignore
    # `warmup_steps` entirely. Both fields are now actually read:
    #   warmup_steps  > 0  -> explicit step count wins
    #   warmup_steps == 0  -> derived as int(max_steps * warmup_ratio)
    # warmup_ratio 0.10 -> 0.15 per BACKLOG 2026-07-02 ("lengthen warmup").
    warmup_ratio: float = float(os.environ.get("TITAN_WARMUP_RATIO", "0.15"))
    warmup_steps: int = int(os.environ.get("TITAN_WARMUP_STEPS", "0"))

    # Post-45K continuation-training (SFT / DMSR ablation / additional pre-training)
    # re-warmup (BACKLOG "LR re-warmup" item -- previously zero code). The canonical
    # 45K run's WSD schedule decays learning_rate down to min_lr_ratio by design;
    # resuming a NEW training run from that checkpoint with the same scheduler state
    # would keep the LR pinned at the floor, so continuation training effectively
    # never learns. Default OFF -- the canonical 45K path is byte-for-byte unaffected;
    # only set TITAN_USE_REWARMUP=1 for a deliberate post-45K continuation run. See
    # train/trainer_core.py::get_rewarmup_schedule().
    use_rewarmup_schedule: bool = os.environ.get("TITAN_USE_REWARMUP", "0") == "1"
    rewarmup_steps: int = int(os.environ.get("TITAN_REWARMUP_STEPS", "1000"))
    # Where the base run's LR floor was -- matches min_lr_ratio's default so the
    # continuation run's cold start lines up with wherever the base run actually
    # landed, not an arbitrary guess.
    rewarmup_start_lr_ratio: float = float(os.environ.get("TITAN_REWARMUP_START_RATIO", "0.01"))

    # Runtime fast paths (V2)
    liquid_fast_path: bool = os.environ.get("TITAN_LIQUID_FAST_PATH", "1") == "1"
    liquid_train_impl: str = os.environ.get("TITAN_LIQUID_TRAIN_IMPL", "baseline")
    moe_dispatch_mode: str = os.environ.get("TITAN_MOE_DISPATCH", "parallel")  # {"parallel", "sequential"}
    use_flash_attn_inference: bool = os.environ.get("TITAN_FLASH_ATTN_INFER", "0") == "1"
    # [TITAN PREFLIGHT] Support env var override for testing
    max_steps: int = int(os.environ.get("TITAN_MAX_STEPS", 45000))  

    # Explicitly disable Epoch Mode to respect max_steps
    epoch_mode: bool = False

    # Floor of the WSD cosine decay phase. Was a dead field (dataclass said 0.1 while
    # the scheduler was called with a hardcoded 0.01). Now actually read; the default
    # is set to 0.01 so RUNTIME BEHAVIOR IS UNCHANGED — only the mechanism moved from
    # hardcoded-literal to config-read.
    min_lr_ratio: float = 0.01
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

    # [2026-07-08] General loss-divergence circuit breaker.
    # NOT a BACKLOG item — added by the 2026-07-08 pre-45K pass and flagged as such.
    # Rationale: the NaN brake only catches non-finite loss, and the Liquid guard only
    # freezes Liquid params. Nothing caught "loss is finite but climbing steadily" —
    # exactly what the 2026-07-02 run did (10.4 -> 15.0, never NaN). This compares the
    # live loss EMA against the EMA snapshotted at the end of warmup and brakes after
    # `patience` consecutive breaches. Default ON (unattended 45K safety);
    # TITAN_DIVERGENCE_GUARD=0 disables it.
    use_divergence_guard: bool = os.environ.get("TITAN_DIVERGENCE_GUARD", "1") == "1"
    divergence_guard_multiplier: float = 1.5
    divergence_guard_patience: int = 50

    # [2026-07-11] Was read only via getattr(cfg, "deterministic", False) in
    # train/trainer_core.py:seed_all() with no backing field anywhere -- so
    # cudnn-deterministic / use_deterministic_algorithms mode could never actually be
    # turned on, it silently always fell through to False. Same default (off, matches
    # prior always-off behavior); TITAN_DETERMINISTIC=1 now genuinely enables it.
    deterministic: bool = os.environ.get("TITAN_DETERMINISTIC", "0") == "1"

    # -------------------------------------------------------------------------
    # 9. OUTPUT FORMAT
    # -------------------------------------------------------------------------
    output_dir: str = "./checkpoints/mertformer_titan_prod"
    save_dir: str = "./checkpoints/mertformer_titan_prod"

    # Logging Frequency
    # Default 10 (was 1): at 1 the per-step metric collection + host snapshot fire
    # every single optimizer step across a 45K run. 10 keeps a fine loss curve
    # (~4.5K points) while cutting logging overhead ~10x. Env-overridable. The
    # expensive host snapshot is throttled further via TITAN_TELEMETRY_INTERVAL.
    log_interval: int = int(os.environ.get("TITAN_LOG_INTERVAL", "10"))
    save_interval: int = int(os.environ.get("TITAN_SAVE_INTERVAL", "1000"))

    export_format: str = "onnx_dynamic"

    # [UNIVERSAL AUTO-PILOT]
    # Selects Float32 on Mac, Bfloat16 on A100.
    param_dtype: Any = field(default_factory=get_auto_dtype)

    # FIX: Mixed Precision Training
    use_amp: bool = field(default_factory=lambda: torch.cuda.is_available())  # Auto-enable only on CUDA

    # [DDP] Use Hugging Face Accelerate
    use_accelerate: bool = True

    # FIX: DataLoader Optimization
    # NOTE (honesty, same pattern as warmup_steps/min_lr_ratio before they were wired):
    # train/train.py FORCES num_workers=0 on BOTH real training paths — online
    # curriculum (stage changes live on the dataset object; worker copies never see
    # them) and offline precomputed logits (deterministic sample<->logit alignment).
    # So this value does not reach the canonical 45K run; it is only the pre-override
    # ceiling (further capped to os.cpu_count() in _finalize_config). Deliberate and
    # correct — do NOT "fix" train.py to honor it. See train/train.py ~L805-813.
    dataloader_num_workers: int = 8 # [TITAN SPEED BOOST] Optimized for 8x A100
    dataloader_prefetch_factor: int = 2
    dataloader_pin_memory: bool = os.environ.get("TITAN_DATALOADER_PIN", "1") == "1"
    dataloader_non_blocking: bool = os.environ.get("TITAN_DATALOADER_NONBLOCKING", "1") == "1"

    # TITAN ONYX STORM: Advanced Training Features
    freeze_core_layers: bool = False  # If True, freeze everything except MoE Router and Liquid Layers

    # -------------------------------------------------------------------------
    # 10. UPGRADES: Gradient Checkpointing, Label Smoothing, Attention Dropout
    # -------------------------------------------------------------------------
    # Gradient Checkpointing: 40% VRAM savings, for large batches
    use_gradient_checkpointing: bool = True

    # [SAFETY FIRST] Disable torch.compile for the first run to ensure dynamic routing works
    use_torch_compile: bool = False
    torch_compile_mode: str = "max-autotune"

    # Advanced optimizers (VRAM optimization)
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


def _load_yaml(path: Path, *, required: bool = False) -> Dict[str, Any]:
    """Load a YAML overlay into a flat dict.

    Returns {} when the file is absent and not ``required``. When an overlay was
    EXPLICITLY requested (an env var pointed at it), callers pass ``required=True``
    so a missing file, an unavailable YAML parser, malformed YAML, or a non-mapping
    top level RAISES instead of silently yielding {}. [2026-07-09] Hardened from the
    old fail-open behavior: a silently-ignored overlay must never let a run start on
    the wrong (canonical-default) config. Non-required calls keep the old tolerant
    behavior byte-for-byte. See BACKLOG "config overlay silent no-op".
    """
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"config overlay requested but not found: {path}. "
                "Fix the env var / filename or unset it."
            )
        return {}
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - yaml is a hard dep in practice
        if required:
            raise RuntimeError(
                f"config overlay {path} requested but PyYAML is unavailable: {exc}"
            ) from exc
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        if required:
            raise ValueError(f"config overlay {path} is not valid YAML: {exc}") from exc
        return {}
    if data is None:
        return {}
    if isinstance(data, dict):
        return data
    if required:
        raise ValueError(
            f"config overlay {path} must be a top-level mapping, got {type(data).__name__}."
        )
    return {}


def _resolve_config_path(config_dir: Path, name: str) -> Path:
    """Resolve config path relative to config_dir unless absolute."""
    p = Path(name)
    return p if p.is_absolute() else (config_dir / name)


def _load_config_overlays() -> Dict[str, Any]:
    """Load base + optional overlays from config/*.yaml (env-controlled)."""
    config_dir = Path(__file__).resolve().parent
    merged: Dict[str, Any] = {}

    # An env var being SET is an explicit request for that overlay, so pass
    # required=True: a typo'd path / filename now RAISES instead of silently
    # falling back to canonical defaults. [2026-07-09] See BACKLOG "config overlay
    # silent no-op".
    base_override = os.environ.get("MERTFORMER_CONFIG")
    if base_override:
        base_path = _resolve_config_path(config_dir, base_override)
        merged.update(_load_yaml(base_path, required=True))

    model_override = os.environ.get("MERTFORMER_MODEL_CONFIG")
    if model_override:
        merged.update(_load_yaml(_resolve_config_path(config_dir, f"model/{model_override}"), required=True))

    train_override = os.environ.get("MERTFORMER_TRAIN_CONFIG")
    if train_override:
        merged.update(_load_yaml(_resolve_config_path(config_dir, f"train/{train_override}"), required=True))

    export_override = os.environ.get("MERTFORMER_EXPORT_CONFIG")
    if export_override:
        merged.update(_load_yaml(_resolve_config_path(config_dir, f"export/{export_override}"), required=True))

    return merged


def _apply_overrides(cfg: MertFormerConfig, overrides: Dict[str, Any]) -> None:
    """Apply flat key overrides to the config instance.

    [2026-07-09] Hardened: an unknown key now RAISES instead of being silently
    dropped. Every key in every shipped overlay was verified to be a real
    MertFormerConfig attribute (see tests/test_config_overlay_strict.py), so this
    only fires on a genuine typo/misconfig — a mistyped key must fail loudly, not
    vanish and let the run use the default. See BACKLOG "config overlay silent no-op".
    """
    unknown = sorted(k for k in overrides if not hasattr(cfg, k))
    if unknown:
        raise AttributeError(
            f"unknown config overlay key(s): {unknown} — not attributes of "
            "MertFormerConfig. Fix the overlay YAML (typo?) or remove the key(s)."
        )
    for key, value in overrides.items():
        setattr(cfg, key, value)


def _finalize_config(cfg: MertFormerConfig) -> None:
    """Finalize config AFTER overrides (batch size auto-tune, worker cap, contract).

    This is the only place batch auto-configuration happens, precisely because it runs
    after `_apply_overrides()` -- see the `__post_init__` docstring for the overlay bug
    that motivated moving it here.
    """
    if cfg.micro_batch_size is None or cfg.grad_accum_steps is None:
        auto_micro, auto_accum = auto_configure_batch_size(target_global_batch=cfg.batch_size, conf=cfg)
        if cfg.micro_batch_size is None:
            cfg.micro_batch_size = auto_micro
        if cfg.grad_accum_steps is None:
            cfg.grad_accum_steps = auto_accum

    # Honesty guard: surface a micro x accum product that does not reconstruct the
    # requested global batch_size. Warn rather than raise -- an operator may pin
    # micro/accum deliberately (small-GPU smoke runs do exactly this), and a hard
    # failure here would break every reduced-size script in scripts/. But it must never
    # be silent: this mismatch is what the pre-2026-07-29 overlay ordering produced
    # invisibly, and it changes the effective token budget of the whole run.
    try:
        micro = int(cfg.micro_batch_size or 0)
        accum = int(cfg.grad_accum_steps or 0)
        requested = int(cfg.batch_size or 0)
        realized = micro * accum
        if micro > 0 and accum > 0 and requested > 0 and realized != requested:
            print(
                f"⚠️  BATCH SHAPE MISMATCH: batch_size={requested} but "
                f"micro_batch_size({micro}) x grad_accum_steps({accum}) = {realized}. "
                f"Per-process global batch is {realized}, not {requested}; the token "
                f"budget derived from batch_size will not match what actually trains. "
                f"(Expected when micro/accum are pinned on purpose; unexpected if an "
                f"overlay changed batch_size alone.)",
                file=sys.stderr,
            )
    except (TypeError, ValueError):
        pass

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
            if os.environ.get("TITAN_STRICT_TOKEN_BUDGET", "0").strip().lower() in {"1", "true", "yes"}:
                # [13] Opt-in launch-checklist guard: escalate the overshoot to a hard error
                # (default off, so smoke/test runs and the normal path are unaffected).
                raise ValueError(
                    f"❌ TITAN_STRICT_TOKEN_BUDGET=1: planned {planned_tokens / 1e9:.2f}B tokens exceeds "
                    f"target_tokens_min {target_min / 1e9:.2f}B by >5%. Fix batch_size/max_steps or unset the flag."
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
