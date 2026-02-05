"""
==============================================================================
MERTFORMER TITAN (ONYX STORM)
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 27) — Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

__version__ = "1.0-BUILD27"
__author__ = "Mert"

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import torch


def _cfg_verbose() -> bool:
    """Return True if config module should emit console output."""
    return os.environ.get("TITAN_CONFIG_VERBOSE", "0") == "1"


def _cfg_print(msg: str) -> None:
    if _cfg_verbose():
        print(msg)


def get_auto_dtype() -> Any:
    """
    TR: Donanıma göre en iyi ve en güvenli veri tipini otomatik seçer.
    EN: Automatically selects the best and safest data type based on hardware.

    Returns:
        torch.dtype: Seçilen veri tipi / Selected data type
    """
    # TR: 1. NVIDIA Ampere ve Üstü (A100, H100, 3090, 4090) -> HIZLI
    # EN: 1. NVIDIA Ampere and Above (A100, H100, 3090, 4090) -> FAST
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16

    # TR: 2. Apple Silicon (M1, M2, M3, M4) -> GÜVENLİ
    # EN: 2. Apple Silicon (M1, M2, M3, M4) -> SAFE
    # TR: Mac'te eğitim sırasında NaN (Tanımsız Sayı) hatası almamak için
    # EN: To avoid NaN errors during training on Mac
    # TR: Float32 kullanmak en sağlam yoldur.
    # EN: Using Float32 is the safest way.
    elif torch.backends.mps.is_available():
        return torch.float32

    # TR: 3. Eski GPU veya CPU -> STANDART / EN: 3. Old GPU or CPU -> STANDARD
    else:
        return torch.float32


def auto_configure_batch_size(target_global_batch: int = 128, conf: Any = None):
    """
    TR: GRANDMASTER AUTO-PILOT - Fizik tabanlı VRAM hesaplama ve optimizasyon.
    EN: GRANDMASTER AUTO-PILOT - Physics-based VRAM calculation & optimization.
    
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

        total_params = 2.64 * 10**9

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
    # TR: 1. SİSTEM KİMLİĞİ / EN: 1. SYSTEM IDENTITY
    # -------------------------------------------------------------------------
    model_name: str = "MertFormer_Titan_S25_Prod"
    version: str = "v1.0-TITAN-BUILD27"

    # TR: Cihazı Otomatik Bul (Once NVIDIA, Yoksa Mac MPS, Yoksa CPU)
    # EN: Auto-detect device (First NVIDIA, then Mac MPS, then CPU)
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    seed: int = 1453

    # -------------------------------------------------------------------------
    # TR: 2. BOYUTLAR (ÇİFT İSİMLENDİRME - HATAYI ÖNLER)
    # EN: 2. DIMENSIONS (DUAL NAMING - PREVENTS ERRORS)
    # -------------------------------------------------------------------------
    # TR: Vocab & Seq / EN: Vocab & Seq
    vocab_size: int = 128256
    max_seq_len: int = 4096

    # TR: Model Genişliği / EN: Model Width
    hidden_size: int = 2048
    intermediate_size: int = 5632

    # TR: [KRİTİK DÜZELTME] Kodun aradığı her iki ismi de tanımlıyoruz:
    # EN: [CRITICAL FIX] We define both names that the code looks for:
    num_hidden_layers: int = 18 # TR: [USER OPT] S25/Mobil uyumu / EN: [USER OPT] S25/Mobile compatibility
    num_layers: int = 18  # <-- train code bunu arıyor

    num_attention_heads: int = 16
    num_heads: int = 16  # <-- mla.py bunu arıyor

    num_kv_heads: int = 8
    # TR: [MATEMATİK DÜZELTME] 2048 / 16 = 128.
    # EN: [MATH FIX] 2048 / 16 = 128.
    head_dim: int = 128

    # TR: Normalizasyon ve Dropout / EN: Normalization and Dropout
    rms_norm_eps: float = 1e-6
    dropout: float = 0.1

    # TR: [YENİ] RoPE Theta: Uzun bağlam (Long Context) için frekans tabanı.
    # EN: [NEW] RoPE Theta: Frequency base for long context.
    # TR: Standart 10.000'dir. Bunu 100.000 yapmak, ileride 8K/16K'ya "esnemeyi" kolaylaştırır.
    # EN: Standard is 10,000. Making it 100,000 eases "stretching" to 8K/16K later.
    rope_theta: float = 100000.0
    rope_base: float = 100000.0 # TR: [FIX] rope_theta ile senkron / EN: [FIX] Sync with rope_theta

    # -------------------------------------------------------------------------
    # TR: 3. BITNET b1.58 / EN: 3. BITNET b1.58
    # -------------------------------------------------------------------------
    use_bitnet: bool = True  # TR: BitNet aktif / EN: BitNet active
    weight_quantization: str = "absmax_per_tensor"  # TR: Ağırlık quantize / EN: Weight quantize
    activation_bits: int = 8  # TR: Aktivasyon bit / EN: Activation bits

    # -------------------------------------------------------------------------
    # TR: 4. SPARSE MoE / EN: 4. SPARSE MoE
    # -------------------------------------------------------------------------
    use_moe: bool = True
    num_experts: int = 8
    num_experts_per_tok: int = 2
    active_experts: int = 2  # Alias for num_experts_per_tok (backward compatibility)
    router_aux_loss_coef: float = 0.02
    aux_loss_coef: float = 0.02  # Alias for router_aux_loss_coef (backward compatibility)
    moe_every_n_layers: int = 3  # MoE on layers: 3, 6, 9, 12, 15, 18, 21 (0-indexed: 2, 5, 8, 11, 14, 17, 20)

    # V26.5: Switch Loss Option & Router Jitter for collapse prevention
    # V26.5: Switch Loss Option & Router Jitter for collapse prevention
    use_switch_loss: bool = True  # [AUDIT FIX] Enable Aggressive Load Balancing
    router_jitter: float = 0.02  # [AUDIT FIX] Increase exploration noise
    router_jitter_boost: float = 0.1  # Emergency jitter when collapse detected

    # -------------------------------------------------------------------------
    # 5. LIQUID LAYERS
    # -------------------------------------------------------------------------
    use_liquid: bool = True
    # [RAPOR DÜZELTME] "euler" yerine "cfc" (Kod zaten CfC uyguluyor, isim düzeltildi)
    liquid_solver: str = "cfc"
    liquid_layers_idx: list = field(default_factory=lambda: [4, 10, 16])  # [FIX] Adjusted for 18 layers (No OOB)
    liquid_every_n_layers: int = 0  # Disabled (using explicit indices instead)

    # Router Z-Loss for stability
    z_loss_coef: float = 1e-4

    # V25.1 SAFEGUARD: Liquid Warmup Steps (Freeze for first N steps)
    liquid_warmup_steps: int = 10000

    # -------------------------------------------------------------------------
    # 6. GÜVENLİK SİGORTASI (QINN)
    # -------------------------------------------------------------------------
    use_qinn: bool = False # [AUDIT FIX] Disabled for NPU compatibility & Speed

    # -------------------------------------------------------------------------
    # [USER OVERRIDE] Teacher Model configuration
    # User confirmed usage of Llama 3.3 70B (assumes A100/H100 or multi-gpu setup)
    teacher_model_id: str = "meta-llama/Llama-3.3-70B-Instruct"
    distill_alpha: float = 0.8 # [USER OPTIMIZATION] Teacher'a %80 güven
    distill_intermediate_layers: bool = True

    # -------------------------------------------------------------------------
    # 7. TOKENIZER OPTIONS (OPT-IN)
    # -------------------------------------------------------------------------
    # TR: Varsayilan ogretmen tokenizer kullanilir. Turkce tokenizer sadece opt-in.
    # EN: Default is teacher tokenizer. Turkish tokenizer is opt-in only.
    use_tr_tokenizer: bool = False
    tr_tokenizer_id: str = "tokenizer/tr"

    # [KRİTİK DÜZELTME]
    # [RAPOR DÜZELTME] 1.3 -> 1.0 (BitNet için keskin öğretmen gerekir)
    # [RAPOR DÜZELTME] 1.3 -> 1.0 (BitNet için keskin öğretmen gerekir)
    teacher_temp: float = 1.0
    
    # [V27.0] Distillation Optimization
    use_precomputed_logits: bool = True  # Utilize offline logits to save VRAM
    # [TITAN PREFLIGHT] Support override for testing
    precomputed_logits_path: str = os.environ.get("TITAN_LOGITS_PATH", "./datasets/logits/")
    


    # -------------------------------------------------------------------------
    # 8. HİPERPARAMETRELER
    # -------------------------------------------------------------------------
    batch_size: int = 128
    
    # [V27.0 AUTO-CONFIG] Otomatik GPU-based batch size optimization
    # micro_batch_size ve grad_accum_steps artık otomatik hesaplanıyor
    micro_batch_size: int = field(default=None)  # Auto-configured
    grad_accum_steps: int = field(default=None)  # Auto-configured
    
    def __post_init__(self):
        """Post-initialization: Auto-configure batch sizes if not set."""
        if self.micro_batch_size is None or self.grad_accum_steps is None:
            auto_micro, auto_accum = auto_configure_batch_size(target_global_batch=self.batch_size, conf=self)
            if self.micro_batch_size is None:
                self.micro_batch_size = auto_micro
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
    # [TITAN PREFLIGHT] Support env var override for testing
    max_steps: int = int(os.environ.get("TITAN_MAX_STEPS", 45000))  

    # [V26.5 FIX] Explicitly disable Epoch Mode to respect max_steps=50k
    epoch_mode: bool = False

    min_lr_ratio: float = 0.1  # Minimum LR ratio for cosine decay

    # Gradyan Kırpma
    # [RAPOR DÜZELTME] 1.0 -> 2.0 (STE sert gradyanlar üretir, biraz esneklik lazım)
    grad_clip: float = 2.0

    # UPGRADE: Early Stopping & Validation
    early_stop_patience: int = 5  # Stop if no improvement for N validation checks
    val_check_interval: int = 1000  # Run validation every N steps

    # -------------------------------------------------------------------------
    # 9. ÇIKTI FORMATI
    # -------------------------------------------------------------------------
    output_dir: str = "./checkpoints/mertformer_titan_prod"
    save_dir: str = "./checkpoints/mertformer_titan_prod"

    # Loglama Sıklığı
    log_interval: int = 1
    save_interval: int = 1000

    export_format: str = "onnx_dynamic"

    # [EVRENSEL OTOMATİK PİLOT]
    # Mac ise Float32, A100 ise Bfloat16 seçer.
    param_dtype: Any = field(default_factory=get_auto_dtype)

    # FIX: Mixed Precision Training
    use_amp: bool = field(default_factory=lambda: torch.cuda.is_available())  # Auto-enable only on CUDA

    # [V26.5 DDP] Use Hugging Face Accelerate
    use_accelerate: bool = True

    # FIX: DataLoader Optimization
    dataloader_num_workers: int = 8 # [TITAN SPEED BOOST] Optimized for 8x A100
    dataloader_prefetch_factor: int = 2

    # TITAN ONYX STORM: Advanced Training Features
    freeze_core_layers: bool = False  # If True, freeze everything except MoE Router and Liquid Layers

    # -------------------------------------------------------------------------
    # 10. V22.0 UPGRADES: Gradient Checkpointing, Label Smoothing, Attention Dropout
    # -------------------------------------------------------------------------
    # Gradient Checkpointing: VRAM %40 tasarrufu, büyük batch için
    use_gradient_checkpointing: bool = True

    # [SAFETY FIRST] Disable torch.compile for the first run to ensure dynamic routing works
    use_torch_compile: bool = False
    torch_compile_mode: str = "max-autotune"

    # [V27.0] Advanced Optimizers (VRAM Optimization)
    use_8bit_adam: bool = True  # Enable 8-bit Optimizer
    use_galore: bool = True     # Enable Gradient Low-Rank Projection

    # Label Smoothing: Overfit önleme (0.0 = kapalı, 0.1 = önerilen)
    label_smoothing: float = 0.1

    # Attention Dropout: Regularization (eğitimde kullanılır, inference'ta kapalı)
    attention_dropout: float = 0.1


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


# Config instance
cfg = MertFormerConfig()
_apply_overrides(cfg, _load_config_overlays())
_finalize_config(cfg)

def validate_layer_config(cfg: MertFormerConfig) -> None:
    """
    TR: Liquid ve MoE katmanlarının çakışmadığını doğrular.
    EN: Validates that Liquid and MoE layers don't conflict.

    Args:
        cfg (MertFormerConfig): Yapılandırma nesnesi / Configuration object
    Raises:
        ValueError: Katman çakışması tespit edilirse / If layer overlap detected
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
