#!/bin/bash
# ==============================================================================
# 🚀 MERTFORMER TITAN (ONYX STORM) - ULTIMATE LAUNCHPAD
# ------------------------------------------------------------------------------
# Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
# Version: v1.0 (Build 27) — Pre-Training
# Status : PRE-TRAINING (UNVERIFIED)
# ==============================================================================

# ------------------------------------------------------------------------------
# 🔑 1. KİMLİK BİLGİLERİ (GÖMÜLÜ & HAZIR)
# ------------------------------------------------------------------------------

# Argument Parsing
RUN_TEST=false
RUN_SITL=false
RUN_CLEANROOM=false

case "${1:-}" in
    --test|--verify)
        RUN_TEST=true
        echo "🔍 ULTIMATE PREFLIGHT MODE ACTIVE"
        ;;
    --sitl-demo)
        RUN_SITL=true
        echo "🚁 SITL DEMO MODE ACTIVE"
        ;;
    --cleanroom-verify)
        RUN_CLEANROOM=true
        echo "🧪 CLEAN-ROOM VERIFY MODE ACTIVE"
        ;;
esac

# ------------------------------------------------------------------------------
# 🔧 1.1 OFFLINE-FIRST MODE + PYTHON SELECTION
# ------------------------------------------------------------------------------
# Default: offline-first (no external logins/downloads unless explicitly enabled).
if [[ -z "${TITAN_OFFLINE+x}" ]]; then
    export TITAN_OFFLINE=1
fi

# Default: WandB is off in offline mode, on in online mode (unless overridden).
if [[ -z "${TITAN_WANDB+x}" ]]; then
    if [[ "${TITAN_OFFLINE}" == "0" ]]; then
        export TITAN_WANDB=1
    else
        export TITAN_WANDB=0
    fi
fi

# Prefer repo venv python if present (avoids broken shebang CLIs).
PYTHON_BIN="${TITAN_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
    if [[ -x ".titan-venv/bin/python" ]]; then
        PYTHON_BIN=".titan-venv/bin/python"
    else
        # Best-effort bootstrap for a single-command experience.
        if [[ "${TITAN_BOOTSTRAP:-1}" == "1" ]] && [[ -f "scripts/bootstrap_venv.sh" ]]; then
            echo "🧪 Bootstrapping local venv (.titan-venv) ..."
            bash scripts/bootstrap_venv.sh || { echo "❌ Venv bootstrap failed."; exit 1; }
            PYTHON_BIN=".titan-venv/bin/python"
        else
            PYTHON_BIN="python3"
        fi
    fi
fi

# Fast path: deterministic SITL proof flow (offline, no training start).
if [ "$RUN_SITL" = true ]; then
    PILOT_ID="${SITL_PILOT_ID:-pilot_001}"
    SITL_RUNS="${SITL_RUNS:-3}"
    SITL_STEPS="${SITL_STEPS:-120}"
    "$PYTHON_BIN" scripts/drone_sitl_demo.py --pilot-id "$PILOT_ID" --runs "$SITL_RUNS" --steps "$SITL_STEPS"
    exit $?
fi

# Fast path: clean-room reproducibility gate on a fresh local clone.
if [ "$RUN_CLEANROOM" = true ]; then
    CLEANROOM_WORKDIR="${CLEANROOM_WORKDIR:-/tmp/nihai_cleanroom_b27}"
    TITAN_CLEANROOM_PYTHON="${TITAN_CLEANROOM_PYTHON:-python3.11}" \
        bash scripts/cleanroom_verify.sh "$CLEANROOM_WORKDIR"
    exit $?
fi

# ------------------------------------------------------------------------------
# 🔑 1.2 ENV LOADER (SAFE)
# ------------------------------------------------------------------------------
if [ -f .env ]; then
    echo "🔐 Loading env from .env file..."
    while IFS= read -r line; do
        # Skip comments/empty
        [[ -z "${line}" ]] && continue
        [[ "${line}" =~ ^[[:space:]]*# ]] && continue
        if [[ "${line}" == *"="* ]]; then
            key="${line%%=*}"
            val="${line#*=}"
            # strip surrounding quotes
            val="${val%\"}"; val="${val#\"}"
            val="${val%\'}"; val="${val#\'}"
            export "${key}=${val}"
        fi
    done < .env
else
    echo "ℹ️  .env file not found. Using existing environment variables."
fi

# Secrets validation: required only in online mode (or if preflight explicitly requires).
if [[ "${TITAN_OFFLINE}" == "0" ]]; then
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "❌ ERROR: HF_TOKEN is missing (online mode). Set it in .env or environment."
        exit 1
    fi
fi
if [[ "${TITAN_WANDB}" == "1" ]] && [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "⚠️  WARNING: WANDB_API_KEY missing; disabling WandB (set TITAN_WANDB=0 to silence)."
    export TITAN_WANDB=0
fi

# Proje Adı
export WANDB_PROJECT="mertformer-titan"

# ------------------------------------------------------------------------------
# 🖥️ 2. SİSTEM HAZIRLIĞI
# ------------------------------------------------------------------------------
OS_TYPE=$(uname -s)
echo "🖥️  Detected OS: $OS_TYPE"
echo "ℹ️  Defaults: use_tr_tokenizer=false | low-bit kernel opt-in (MERTFORMER_LOWBIT_KERNEL=1) | tensorcore opt-in (MERTFORMER_TENSORCORE=1) | BENCHMARK_SAMPLES=0"

# Version consistency check (best-effort but fail on mismatch)
"$PYTHON_BIN" scripts/version_checker.py || { echo "❌ Version check failed."; exit 1; }

# Update local hardware report (best-effort).
# NOTE: This writes tracked docs under `reports/`. Avoid dirtying the repo during
# preflight-only runs (review/CI friendly).
if [ "$RUN_TEST" = true ]; then
    echo "ℹ️  Skipping system_hardware report update in --test/--verify mode."
else
    "$PYTHON_BIN" scripts/update_system_hardware.py || echo "⚠️  system_hardware report update failed (continuing)"
fi

# Bellek Yönetimi (OOM Riskini Azaltır)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Tokenizer Deadlock Önleyici
export TOKENIZERS_PARALLELISM=false

# WandB Login (explicit + online-only)
if [[ "${TITAN_OFFLINE}" == "0" ]] && [[ "${TITAN_WANDB}" == "1" ]]; then
    echo "🔐 Logging into WandB..."
    "$PYTHON_BIN" -m pip install wandb --quiet
    "$PYTHON_BIN" -m wandb login "$WANDB_API_KEY" --relogin
else
    echo "ℹ️  WandB login skipped (offline/default)."
fi

# ------------------------------------------------------------------------------
# 📦 3. KURULUMLAR (AKILLI MOD)
# ------------------------------------------------------------------------------
if [[ "${TITAN_INSTALL:-0}" == "1" ]]; then
    echo "📦 Installing Dependencies (TITAN_INSTALL=1)..."
else
    echo "ℹ️  Skipping dependency installation (set TITAN_INSTALL=1 to install)."
fi

# PyTorch Kontrolü
if [[ "${TITAN_INSTALL:-0}" == "1" ]]; then
    if ! "$PYTHON_BIN" -c "import torch" &> /dev/null; then
        echo "📦 PyTorch kuruluyor..."
        "$PYTHON_BIN" -m pip install "torch>=2.0" --quiet
    fi
fi

# Temel Paketler (requirements.txt)
if [[ "${TITAN_INSTALL:-0}" == "1" ]]; then
    "$PYTHON_BIN" -m pip install -r requirements.txt --quiet
fi

# Accelerate & BitsAndBytes (Linux vs Mac Ayrımı)
if [[ "${TITAN_INSTALL:-0}" == "1" ]]; then
    if ! "$PYTHON_BIN" -c "import accelerate" &> /dev/null; then
        if [[ "$OS_TYPE" == "Darwin" ]]; then
            "$PYTHON_BIN" -m pip install accelerate
            echo "⚠️  Mac detected: Bitsandbytes skipped."
        else
            "$PYTHON_BIN" -m pip install accelerate bitsandbytes
        fi
    fi
fi

# FLASH ATTENTION 2 (Sadece Linux/A100 için - Otomatik Kurulum)
if [[ "${TITAN_INSTALL:-0}" == "1" ]]; then
    if [[ "$OS_TYPE" == "Linux" ]]; then
        if ! "$PYTHON_BIN" -c "import flash_attn" &> /dev/null; then
            echo "⚡ Installing Flash Attention 2 (Bu işlem 5-10 dk sürebilir, LÜTFEN BEKLE!)..."
            "$PYTHON_BIN" -m pip install flash-attn --no-build-isolation || echo "⚠️ Flash Attention installation failed. Continuing without it..."
        else
            echo "⚡ Flash Attention zaten kurulu. Devam ediliyor..."
        fi
    else
        echo "🍎 Mac Detected: Skipping Flash Attention."
    fi
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
"$PYTHON_BIN" scripts/titan_preflight.py
if [ $? -ne 0 ]; then
    echo "❌ ULTIMATE PREFLIGHT FAILED! Check logs at logs/preflight/titan_preflight.log"
    exit 1
fi
echo "✅ ULTIMATE PREFLIGHT PASSED. SYSTEM IS VERIFIED."

if [ "$RUN_TEST" = true ]; then
    exit 0 # Sadece test istenmişse burada dur.
fi

# Offline-first safety: do not start network-heavy training pipeline by accident.
if [[ "${TITAN_OFFLINE}" != "0" ]]; then
    echo "❌ Offline-first mode active (TITAN_OFFLINE=1). Training pipeline is disabled by default."
    echo "   To start training, run with: TITAN_OFFLINE=0 (and optionally TITAN_WANDB=1)."
    exit 2
fi

# ------------------------------------------------------------------------------
# 🛡️ 5.5 OPERATOR MODE GATE (FULL)
# ------------------------------------------------------------------------------
if [ -z "$OPERATOR_GATE_SKIP" ]; then
    echo "🛡️  OPERATOR MODE GATE (FULL) STARTING..."
    "$PYTHON_BIN" scripts/operator_mode_gate.py --full --no-pytest
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
"$PYTHON_BIN" scripts/smart_runner.py 2>&1 | tee logs/production_run.log

# ------------------------------------------------------------------------------
# 📊 6.5. INTERNAL BENCHMARKS (HUMANEVAL/MBPP)
# ------------------------------------------------------------------------------
if [ -z "$BENCHMARK_SKIP" ]; then
    echo "📊 Running internal benchmarks (HumanEval/MBPP)..."
    BENCHMARK_SAMPLES=${BENCHMARK_SAMPLES:-0}
    BENCHMARK_CKPT_PATH=$("$PYTHON_BIN" - <<'PY'
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
        "$PYTHON_BIN" scripts/benchmarks_internal.py --run --samples "$BENCHMARK_SAMPLES" --ckpt "$BENCHMARK_CKPT_PATH" || echo "⚠️ Benchmarks failed or unavailable. Continuing..."
    fi
else
    echo "⚠️ Benchmarks skipped (BENCHMARK_SKIP set)."
fi

# ------------------------------------------------------------------------------
# 🧪 6.7. GOLDEN SAMPLE EVAL (BEST + LATEST)
# ------------------------------------------------------------------------------
if [ -z "$GOLDEN_EVAL_SKIP" ]; then
    echo "🧪 Running golden sample eval (best + latest)..."
    readarray -t GOLDEN_CKPTS < <("$PYTHON_BIN" - <<'PY'
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
            "$PYTHON_BIN" scripts/golden_eval.py --run-model --ckpt "$ckpt" 2>&1 | tee -a logs/golden_eval.log
        done
    fi
else
    echo "⚠️ Golden eval skipped (GOLDEN_EVAL_SKIP set)."
fi

# Eğitim bitti, şimdi paketle ve temizle
echo "🚀 EĞİTİM TAMAMLANDI. MOBİL EXPORT BAŞLATILIYOR..."
"$PYTHON_BIN" scripts/mobile_export.py
echo "🧾 Updating unified logbook..."
"$PYTHON_BIN" scripts/logbook_build.py --append || echo "⚠️ Logbook update failed (continuing)"
echo "✅ TÜM İŞLEMLER BİTTİ. TELEFONA HAZIR!"
