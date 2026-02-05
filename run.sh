#!/bin/bash
# ==============================================================================
# 🚀 MERTFORMER TITAN (ONYX STORM) - ULTIMATE LAUNCHPAD
# ------------------------------------------------------------------------------
# Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
# Version: v27.0-FINAL (Locked & Sealed)
# Status : PRODUCTION READY (LOCKED)
# ==============================================================================

# ------------------------------------------------------------------------------
# 🔑 1. KİMLİK BİLGİLERİ (GÖMÜLÜ & HAZIR)
# ------------------------------------------------------------------------------

# Argument Parsing
RUN_TEST=false
if [[ "$1" == "--test" ]] || [[ "$1" == "--verify" ]]; then
    RUN_TEST=true
    echo "🔍 ULTIMATE PREFLIGHT MODE ACTIVE"
fi

# ------------------------------------------------------------------------------
# 🔑 1. KİMLİK BİLGİLERİ (SECURE ENV LOADER)
# ------------------------------------------------------------------------------
if [ -f .env ]; then
    echo "🔐 Loading secrets from .env file..."
    export $(grep -v '^#' .env | xargs)
else
    echo "⚠️  .env file not found! Checking environment variables..."
fi

# Validation
if [[ -z "$HF_TOKEN" ]]; then
    echo "❌ ERROR: HF_TOKEN is missing. Please set it in .env or environment."
    exit 1
fi
if [[ -z "$WANDB_API_KEY" ]]; then
    echo "⚠️  WARNING: WANDB_API_KEY is missing. Logging will be disabled/offline."
fi

# Proje Adı
export WANDB_PROJECT="mertformer-titan"

# ------------------------------------------------------------------------------
# 🖥️ 2. SİSTEM HAZIRLIĞI
# ------------------------------------------------------------------------------
OS_TYPE=$(uname -s)
echo "🖥️  Detected OS: $OS_TYPE"
echo "ℹ️  Defaults: use_tr_tokenizer=false | low-bit kernel opt-in (MERTFORMER_LOWBIT_KERNEL=1) | tensorcore opt-in (MERTFORMER_TENSORCORE=1) | BENCHMARK_SAMPLES=0"

# Update local hardware report (best-effort)
python3 scripts/update_system_hardware.py || echo "⚠️  system_hardware report update failed (continuing)"

# Bellek Yönetimi (OOM Riskini Azaltır)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Tokenizer Deadlock Önleyici
export TOKENIZERS_PARALLELISM=false

# WandB Giriş (Otomatik)
echo "🔐 Logging into WandB..."
pip install wandb --quiet
wandb login "$WANDB_API_KEY" --relogin

# ------------------------------------------------------------------------------
# 📦 3. KURULUMLAR (AKILLI MOD)
# ------------------------------------------------------------------------------
echo "📦 Installing Dependencies..."

# PyTorch Kontrolü
if ! python3 -c "import torch" &> /dev/null; then
    echo "📦 PyTorch kuruluyor..."
    pip install "torch>=2.0" --quiet
fi

# Temel Paketler (requirements.txt)
pip install -r requirements.txt --quiet

# Accelerate & BitsAndBytes (Linux vs Mac Ayrımı)
if ! python3 -c "import accelerate" &> /dev/null; then
    if [[ "$OS_TYPE" == "Darwin" ]]; then
         pip install accelerate
         echo "⚠️  Mac detected: Bitsandbytes skipped."
    else
         pip install accelerate bitsandbytes
    fi
fi

# FLASH ATTENTION 2 (Sadece Linux/A100 için - Otomatik Kurulum)
if [[ "$OS_TYPE" == "Linux" ]]; then
    if ! python3 -c "import flash_attn" &> /dev/null; then
        echo "⚡ Installing Flash Attention 2 (Bu işlem 5-10 dk sürebilir, LÜTFEN BEKLE!)..."
        pip install flash-attn --no-build-isolation || echo "⚠️ Flash Attention installation failed. Continuing without it..."
    else
        echo "⚡ Flash Attention zaten kurulu. Devam ediliyor..."
    fi
else
    echo "🍎 Mac Detected: Skipping Flash Attention."
fi

# ------------------------------------------------------------------------------
# ⚡ 4.5. NCCL TUNING (MULTI-GPU OPTIMIZATION) - V27.0
# ------------------------------------------------------------------------------
# NCCL: NVIDIA Collective Communications Library (Multi-GPU iletişim optimizasyonu)
# Sadece Linux + Multi-GPU sistemlerde aktif olur
# ------------------------------------------------------------------------------

if [[ "$OS_TYPE" == "Linux" ]] && command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    
    if [ "$NUM_GPUS" -gt 1 ]; then
        echo "🔥 NCCL Tuning: $NUM_GPUS GPU detected, optimizing communication..."
        
        # P2P (Peer-to-Peer) GPU Transfer
        # NVLink varsa 0, yoksa 1 (otomatik tespit)
        if nvidia-smi nvlink --status &> /dev/null; then
            export NCCL_P2P_DISABLE=0
            echo "   ✅ NVLink detected: P2P enabled"
        else
            export NCCL_P2P_DISABLE=1
            echo "   ⚠️  No NVLink: P2P disabled"
        fi
        
        # InfiniBand (Yüksek hızlı network)
        # Çoğu sistemde yok, disable ediyoruz (hata önleme)
        export NCCL_IB_DISABLE=1
        
        # Network Interface (Otomatik tespit)
        # eth0, ens3, enp0s3 gibi interface'leri dene
        for iface in eth0 ens3 enp0s3 eno1; do
            if ip link show "$iface" &> /dev/null; then
                export NCCL_SOCKET_IFNAME="$iface"
                echo "   ✅ Network Interface: $iface"
                break
            fi
        done
        
        # NCCL Optimizations
        export NCCL_ASYNC_ERROR_HANDLING=1  # Hata toleransı
        export NCCL_BLOCKING_WAIT=0         # Non-blocking (daha hızlı)
        
        # Debug (sadece sorun varsa aktif et)
        # export NCCL_DEBUG=INFO
        
        echo "   🚀 NCCL Optimization: ACTIVE (+5-10% speedup on multi-GPU)"
    else
        echo "ℹ️  Single GPU detected: NCCL tuning skipped (not needed)"
    fi
else
    echo "ℹ️  Mac/CPU detected: NCCL tuning skipped"
fi

# ------------------------------------------------------------------------------
# ⚙️ 4. ACCELERATE YAPILANDIRMASI (AUTO-CONFIG) - GÜNCELLENDİ
# ------------------------------------------------------------------------------
if [ ! -f ~/.cache/huggingface/accelerate/default_config.yaml ]; then
    echo "⚙️ Auto-Configuring Accelerate..."
    mkdir -p ~/.cache/huggingface/accelerate

    if command -v nvidia-smi &> /dev/null; then
        NUM_PROCS=$(nvidia-smi -L | wc -l)
    else
        NUM_PROCS=1
    fi

    # MANTIK DÜZELTMESİ: Tek GPU ise MULTI_GPU olmamalı
    DIST_TYPE="MULTI_GPU"
    USE_CPU="false"
    if [ "$NUM_PROCS" -eq 1 ]; then
        DIST_TYPE="NO"
    fi
    if [[ "$OS_TYPE" == "Darwin" ]]; then
        DIST_TYPE="NO" # Mac için şimdilik en güvenli yol
    fi

    cat <<EOT > ~/.cache/huggingface/accelerate/default_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: $DIST_TYPE
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: $NUM_PROCS
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOT
    echo "✅ Config created for $NUM_PROCS Processes ($DIST_TYPE mode)."
fi

# ------------------------------------------------------------------------------
# 🔍 5. ULTIMATE PRE-FLIGHT GATE
# ------------------------------------------------------------------------------
echo "🔍 Performing Ultimate Pre-Flight Check..."

# Her durumda (test modu olsa da olmasa da) preflight çalışır. 
# Sadece --test modunda ise test bitince durur, normal modda ise başarılıysa eğitime geçer.
python3 scripts/titan_preflight.py
if [ $? -ne 0 ]; then
    echo "❌ ULTIMATE PREFLIGHT FAILED! Check logs at logs/preflight/titan_preflight.log"
    exit 1
fi
echo "✅ ULTIMATE PREFLIGHT PASSED. SYSTEM IS VERIFIED."

if [ "$RUN_TEST" = true ]; then
    exit 0 # Sadece test istenmişse burada dur.
fi

# ------------------------------------------------------------------------------
# 🛡️ 5.5 OPERATOR MODE GATE (FULL)
# ------------------------------------------------------------------------------
if [ -z "$OPERATOR_GATE_SKIP" ]; then
    echo "🛡️  OPERATOR MODE GATE (FULL) STARTING..."
    python3 scripts/operator_mode_gate.py --full --no-pytest
    if [ $? -ne 0 ]; then
        echo "❌ OPERATOR MODE GATE FAILED! Aborting launch."
        exit 1
    fi
    echo "✅ OPERATOR MODE GATE PASSED."
else
    echo "⚠️  OPERATOR MODE GATE SKIPPED (OPERATOR_GATE_SKIP set)."
fi

# ------------------------------------------------------------------------------
# 🚀 6. ATEŞLEME
# ------------------------------------------------------------------------------
echo "🚀 TITAN LAUNCHING..."
mkdir -p logs

# Port 29501 çakışmayı önler, logları hem ekrana hem dosyaya yazar.
# V27.0 SMART RUNNER: Starts Parallel Data Pipeline -> Distillation -> Training
python3 scripts/smart_runner.py 2>&1 | tee logs/production_run.log

# ------------------------------------------------------------------------------
# 📊 6.5. INTERNAL BENCHMARKS (HUMANEVAL/MBPP)
# ------------------------------------------------------------------------------
if [ -z "$BENCHMARK_SKIP" ]; then
    echo "📊 Running internal benchmarks (HumanEval/MBPP)..."
    BENCHMARK_SAMPLES=${BENCHMARK_SAMPLES:-0}
    BENCHMARK_CKPT_PATH=$(python3 - <<'PY'
from pathlib import Path
import os

def pick_checkpoint():
    candidates = []
    env_ckpt = os.environ.get("BENCHMARK_CKPT")
    if env_ckpt:
        candidates.append(Path(env_ckpt))

    try:
        from config.config import cfg
        save_dir = Path(cfg.save_dir)
        model_name = cfg.model_name
    except Exception:
        save_dir = Path("checkpoints")
        model_name = None

    if model_name:
        candidates.append(save_dir / f"{model_name}_latest.pt")
        candidates.append(save_dir / f"{model_name}_best.pt")

    if save_dir.exists():
        candidates += sorted(save_dir.glob("*_latest.pt"))
        candidates += sorted(save_dir.glob("*_best.pt"))
        candidates += sorted(save_dir.glob("*.pt"))

    root = Path("checkpoints")
    if root.exists():
        candidates += sorted(root.rglob("*_latest.pt"))
        candidates += sorted(root.rglob("*_best.pt"))
        candidates += sorted(root.rglob("*.pt"))

    for c in candidates:
        if c and c.exists():
            return str(c)
    return ""

print(pick_checkpoint())
PY
    )
    if [ -z "$BENCHMARK_CKPT_PATH" ]; then
        echo "⚠️ Benchmarks skipped (checkpoint not found)."
    else
        python3 scripts/benchmarks_internal.py --run --samples "$BENCHMARK_SAMPLES" --ckpt "$BENCHMARK_CKPT_PATH" || echo "⚠️ Benchmarks failed or unavailable. Continuing..."
    fi
else
    echo "⚠️ Benchmarks skipped (BENCHMARK_SKIP set)."
fi

# ------------------------------------------------------------------------------
# 🧪 6.7. GOLDEN SAMPLE EVAL (BEST + LATEST)
# ------------------------------------------------------------------------------
if [ -z "$GOLDEN_EVAL_SKIP" ]; then
    echo "🧪 Running golden sample eval (best + latest)..."
    readarray -t GOLDEN_CKPTS < <(python3 - <<'PY'
from pathlib import Path
import os

def pick_checkpoint(suffix: str) -> str:
    candidates = []
    env_key = f"GOLDEN_EVAL_CKPT_{suffix.upper()}"
    env_ckpt = os.environ.get(env_key)
    if env_ckpt:
        candidates.append(Path(env_ckpt))

    try:
        from config.config import cfg
        save_dir = Path(cfg.save_dir)
        model_name = cfg.model_name
    except Exception:
        save_dir = Path("checkpoints")
        model_name = None

    if model_name:
        candidates.append(save_dir / f"{model_name}_{suffix}.pt")

    if save_dir.exists():
        candidates += sorted(save_dir.glob(f"*_{suffix}.pt"))

    root = Path("checkpoints")
    if root.exists():
        candidates += sorted(root.rglob(f"*_{suffix}.pt"))

    for c in candidates:
        if c and c.exists():
            return str(c)
    return ""

best = pick_checkpoint("best")
latest = pick_checkpoint("latest")

seen = set()
ordered = []
for ckpt in (best, latest):
    if ckpt and ckpt not in seen:
        ordered.append(ckpt)
        seen.add(ckpt)

print("\n".join(ordered))
PY
    )
    if [ "${#GOLDEN_CKPTS[@]}" -eq 0 ]; then
        echo "⚠️ Golden eval skipped (checkpoint not found)."
    else
        for ckpt in "${GOLDEN_CKPTS[@]}"; do
            echo "🧪 Golden eval on: $ckpt"
            python3 scripts/golden_eval.py --run-model --ckpt "$ckpt" 2>&1 | tee -a logs/golden_eval.log
        done
    fi
else
    echo "⚠️ Golden eval skipped (GOLDEN_EVAL_SKIP set)."
fi

# Eğitim bitti, şimdi paketle ve temizle
echo "🚀 EĞİTİM TAMAMLANDI. MOBİL EXPORT BAŞLATILIYOR..."
python3 scripts/mobile_export.py
echo "✅ TÜM İŞLEMLER BİTTİ. TELEFONA HAZIR!"
