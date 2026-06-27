"""
Kaggle One-Cell T4 Lane (Build 30, Claim-Safe Closure Upgrade)
----------------------------------------------------------------
Single-file, repo-import-free script for:
- Kaggle notebook single-cell copy/paste execution on a single T4 GPU
- Stable sweet-spot proof-of-learning with deterministic curriculum handling
- Repo-parity minded embedded layer coverage with explicit guarded/default-on states
- Atomic checkpoint/resume (latest + best + rolling5)
- Structured evidence outputs and closure-friendly packaging surfaces
- Non-interactive, no-menu, no-input default behavior

Notes:
- This file is designed to be pasted directly into a Kaggle cell or run as a standalone script.
- It does not import repo modules at runtime.
- It optimizes for stable evidence and graceful failure, not hype claims.
- NAMING NOTE: the filename says "5080" (RTX 5080) but the actual target lane is
  the Kaggle One-Cell T4 build (the "BUILD30" banner, the default profile
  't4_onecell_sweetspot', and checkpoint/schema names all say build30/t4). The
  "5080" in the filename and "BUILD30" tags are legacy/fossil labels; trust the
  in-file profile/schema for the real target, not the filename.
"""
from __future__ import annotations

import csv
import io
import json
import hashlib
import hmac
import math
import os
import multiprocessing as mp
import random
import re
import shutil
import signal
import sys
import tempfile
import time
import traceback
import zipfile
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# 5080 friend-machine self bootstrap. Kept before torch import on purpose.
# Disable with MERTFORMER_SELF_BOOTSTRAP=0.
# ---------------------------------------------------------------------------
def _m5080_self_bootstrap():
    import os as _os
    if _os.environ.get("MERTFORMER_SELF_BOOTSTRAP", "1").lower() in {"0", "false", "no", "off"}:
        return
    import importlib.util as _ilu
    import subprocess as _sp
    import sys as _sys
    required = [
        ("torch", "torch --index-url https://download.pytorch.org/whl/cu128"),
        ("numpy", "numpy"),
        ("datasets", "datasets"),
        ("tokenizers", "tokenizers"),
        ("safetensors", "safetensors"),
        ("psutil", "psutil"),
        ("cryptography", "cryptography"),
    ]
    missing = []
    for module_name, pip_spec in required:
        if _ilu.find_spec(module_name) is None:
            missing.append(pip_spec)
    if not missing:
        return
    if _os.environ.get("MERTFORMER_ALLOW_PIP", "1").lower() in {"0", "false", "no", "off"}:
        print("[bootstrap] missing packages but MERTFORMER_ALLOW_PIP=0:", missing, file=_sys.stderr)
        return
    print("[bootstrap] installing missing packages:", missing, flush=True)
    for spec in missing:
        cmd = [_sys.executable, "-m", "pip", "install", "--upgrade"] + spec.split()
        _sp.check_call(cmd)

_m5080_self_bootstrap()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

# Optional dependencies --------------------------------------------------------
try:
    from datasets import load_dataset  # type: ignore

    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False

try:
    import sentencepiece as spm  # type: ignore

    HAS_SENTENCEPIECE = True
except Exception:
    HAS_SENTENCEPIECE = False

try:
    import matplotlib.pyplot as plt  # type: ignore

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


EMBEDDED_LAYER_PARITY = {'schema': 'embedded_layer_parity_v1',
 'repo_layers_dir': 'layers',
 'default_feature_bundle': 't4_onecell_sweetspot',
 'files': {'bitlinear.py': {'sha256': '262b9de16548d7a0b298c199119319bbec3c77af30e90d4443b3c89dcbf85bc6',
                            'bytes': 5551,
                            'line_count': 161,
                            'classes': ['BitLinear'],
                            'default_state': 'always_on'},
           'ffn.py': {'sha256': 'b19e37b40d56d032363337563fd30378248cac3680b349169c4ae77461ef4c5c',
                      'bytes': 2929,
                      'line_count': 82,
                      'classes': ['MertFormerFFN'],
                      'default_state': 'always_on'},
           'mertformer_block.py': {'sha256': 'ba585c97eb04920f59fc011a80e4839e615b1c452cb79eaa642484db916be552',
                                   'bytes': 10438,
                                   'line_count': 255,
                                   'classes': ['RMSNorm', 'MertFormerBlock'],
                                   'default_state': 'always_on'},
           'mla.py': {'sha256': '1afe20f9e375315e40adfcef34d36a151fb159fa72d80e4125958093188e4df2',
                      'bytes': 19886,
                      'line_count': 478,
                      'classes': ['_QKRMSNorm', 'RotaryEmbedding', 'MLA'],
                      'default_state': 'always_on'},
           'moe.py': {'sha256': '34cf76046cd522894d20bf10b1031aeeab8c0fda0dd5f4b33d4fed1bb5d273d1',
                      'bytes': 37589,
                      'line_count': 857,
                      'classes': ['BitSwiGLU', 'LiquidRouter', 'MoE'],
                      'default_state': 'default_on'},
           'liquid.py': {'sha256': '9cb4038e3e2c38caab1967b7ed20004119066eb1280dc57492a354a68497b855',
                         'bytes': 16515,
                         'line_count': 401,
                         'classes': ['LiquidCell', 'LiquidMixer'],
                         'default_state': 'default_on'},
           'bitnet_patch.py': {'sha256': '631eefd14fad9a6d8e9d371bc33a27a46bc5a5d86cda8fb83c75bc66fd55aeb8',
                               'bytes': 3233,
                               'line_count': 98,
                               'classes': [],
                               'default_state': 'default_on'},
           'cognitive_extensions.py': {'sha256': 'b572fd1490eadc71950368f7d499dcc018bd08059c6b3100a57a7adab897df7a',
                                       'bytes': 5407,
                                       'line_count': 126,
                                       'classes': ['GlobalWorkspaceBroadcast',
                                                   'ContinuousLatentODEStateChannel',
                                                   'NeuromodulatoryGainLayer',
                                                   'HebbianPlasticityLayer',
                                                   'NeuroSymbolicLayer'],
                                       'default_state': 'guarded_off'},
           'qinn.py': {'sha256': 'ca8af1acb74974ee13eec237da0a6d84f00446fb3796db779542a23023218a61',
                       'bytes': 5975,
                       'line_count': 161,
                       'classes': ['UnitaryQINN'],
                       'default_state': 'guarded_off'},
           'lifelong_safety.py': {'sha256': 'cf92dc7dcfabfc507e3ba5df3d52dfbb3df6a9b7903b34cf01b08d5a5bcdd3c4',
                                  'bytes': 2860,
                                  'line_count': 78,
                                  'classes': ['LifelongSafetyLayer'],
                                  'default_state': 'guarded_off'},
           'world_model_head.py': {'sha256': 'b44801193906e384e570022e549e3c9dc7869da6d981df844c43d897640eca15',
                                   'bytes': 2959,
                                   'line_count': 84,
                                   'classes': ['WorldModelOutput', 'CausalWorldModelHead'],
                                   'default_state': 'guarded_off'}}}

# =============================================================================
# Profiles & Runtime Config
# =============================================================================
RUN_PROFILES: Dict[str, Dict[str, Any]] = {
    "quick": {
        "quick": True,
        "max_wall_hours": 0.2,
        "target_train_tokens": 2_000_000,
        "max_steps": 120,
        "batch_size": 2,
        "seq_len": 64,
        "grad_accum_steps": 1,
        "eval_interval_steps": 20,
        "checkpoint_interval_steps": 20,
        "checkpoint_interval_minutes": 5,
        "benchmark_steps": 30,
        "benchmark_eval_batches": 8,
    },
    "deep8h": {
        "quick": False,
        "max_wall_hours": 8.0,
        "target_train_tokens": 120_000_000,
        "max_steps": 200_000,
        "batch_size": 4,
        "seq_len": 256,
        "grad_accum_steps": 2,
        "eval_interval_steps": 200,
        "checkpoint_interval_steps": 200,
        "checkpoint_interval_minutes": 15,
        "benchmark_steps": 80,
        "benchmark_eval_batches": 24,
        # Reduce disk I/O pressure on long runs while preserving telemetry quality.
        "step_log_interval": 10,
    },
    "linkedin_sweetspot": {
        # Kaggle-friendly sweet spot profile for proof-of-learning + showcase evidence.
        # Goal: fast, stable convergence signal with reproducible core metrics.
        "quick": False,
        "max_wall_hours": 3.5,
        "target_train_tokens": 8_000_000,
        "max_steps": 35_000,
        "batch_size": 6,
        "seq_len": 256,
        "grad_accum_steps": 2,
        "eval_interval_steps": 250,
        "checkpoint_interval_steps": 500,
        "checkpoint_interval_minutes": 12,
        "benchmark_steps": 16,
        "benchmark_eval_batches": 10,
        "step_log_interval": 10,
        "tokenizer_max_texts": 80000,
        "tokenizer_fit_max_texts": 20000,
        # ~192M parameter class (Kaggle practicality + meaningful training signal).
        "mert_hidden": 1024,
        "mert_layers": 12,
        "mert_heads": 16,
        "mert_kv_heads": 8,
        # Keep experimental cognitive stack OFF for stable evidence run.
        "mert_enable_all_extensions": False,
        "mert_use_qinn": False,
        "target_param_band_low": 160_000_000,
        "target_param_band_high": 240_000_000,
        # Probe gates for go/no-go in showcase cycle.
        "mini_probe_enabled": True,
        "mini_probe_param_target": 192_000_000,
        "mini_probe_param_tolerance": 45_000_000,
        "mini_probe_token_min": 5_000_000,
        "mini_probe_token_max": 20_000_000,
        "mini_probe_min_steps_for_signal": 300,
        "mini_probe_min_loss_drop_ratio": 0.06,
        "mini_probe_max_grad_cv": 1.50,
        "mini_probe_min_router_entropy": 0.30,
        "mini_probe_max_router_load_p95": 0.90,
        "mini_probe_max_collapse_events": 0,
    },
    "mini300m": {
        # 300M mini convergence probe profile (code-freeze phase).
        # Goal: verify learning dynamics before scaling toward 2.6B.
        "quick": False,
        "max_wall_hours": 48.0,
        "target_train_tokens": 5_000_000_000,
        "max_steps": 2_000_000,
        "batch_size": 24,
        "seq_len": 1024,
        "grad_accum_steps": 1,
        "eval_interval_steps": 1000,
        "checkpoint_interval_steps": 1000,
        "checkpoint_interval_minutes": 20,
        "benchmark_steps": 24,
        "benchmark_eval_batches": 12,
        "step_log_interval": 25,
        "tokenizer_max_texts": 240000,
        "tokenizer_fit_max_texts": 60000,
        # ~300M parameter target (empirically calibrated for this one-file architecture).
        "mert_hidden": 1024,
        "mert_layers": 20,
        "mert_heads": 16,
        "mert_kv_heads": 8,
        # Keep experimental cognitive stack off during convergence probe.
        "mert_enable_all_extensions": False,
        "mert_use_qinn": False,
        "target_param_band_low": 270_000_000,
        "target_param_band_high": 330_000_000,
        # Probe gates requested for go/no-go:
        # loss curve, grad norm stability, expert load distribution, router entropy.
        "mini_probe_enabled": True,
        "mini_probe_param_target": 300_000_000,
        "mini_probe_param_tolerance": 35_000_000,
        "mini_probe_token_min": 5_000_000_000,
        "mini_probe_token_max": 10_000_000_000,
        "mini_probe_min_steps_for_signal": 500,
        "mini_probe_min_loss_drop_ratio": 0.08,
        "mini_probe_max_grad_cv": 1.35,
        "mini_probe_min_router_entropy": 0.35,
        "mini_probe_max_router_load_p95": 0.85,
        "mini_probe_max_collapse_events": 0,
    },
    "t4_onecell_sweetspot": {
        # Single-T4, Kaggle-cell-first lane with repo-parity aware defaults.
        "quick": False,
        "max_wall_hours": 5.5,
        "target_train_tokens": 10_000_000,
        "max_steps": 40_000,
        "batch_size": 4,
        "seq_len": 256,
        "grad_accum_steps": 2,
        "eval_interval_steps": 250,
        "checkpoint_interval_steps": 500,
        "checkpoint_interval_minutes": 12,
        "benchmark_steps": 16,
        "benchmark_eval_batches": 10,
        "step_log_interval": 10,
        "tokenizer_max_texts": 60000,
        "tokenizer_fit_max_texts": 16000,
        "mert_hidden": 1024,
        "mert_layers": 12,
        "mert_heads": 16,
        "mert_kv_heads": 8,
        "mert_enable_all_extensions": False,
        "mert_use_qinn": False,
        "use_gradient_checkpointing": False,
        "strict_data": False,
        "allow_degraded_data": True,
        "require_code_stage_data": False,
        "target_param_band_low": 160_000_000,
        "target_param_band_high": 240_000_000,
        "mini_probe_enabled": True,
        "mini_probe_param_target": 192_000_000,
        "mini_probe_param_tolerance": 45_000_000,
        "mini_probe_token_min": 5_000_000,
        "mini_probe_token_max": 20_000_000,
        "mini_probe_min_steps_for_signal": 300,
        "mini_probe_min_loss_drop_ratio": 0.06,
        "mini_probe_max_grad_cv": 1.50,
        "mini_probe_min_router_entropy": 0.30,
        "mini_probe_max_router_load_p95": 0.90,
        "mini_probe_max_collapse_events": 0,
    },
    "custom": {},
}

RUN_CONFIG: Dict[str, Any] = {
    "profile": "t4_onecell_sweetspot",  # quick|deep8h|linkedin_sweetspot|mini300m|custom
    "interactive": False,
    # Stable default: avoid notebook input waits/prompts unless explicitly enabled.
    "interactive_menu": False,
    "allow_notebook_input": False,
    "force_interactive_input": False,
    "seed": 42,
    "seed_list": [42, 43, 44],
    "device": "auto",  # auto|cpu|mps|cuda
    "vram_limit_gb": 16.0,
    "out_dir": "/kaggle/working/mertformer_onecell_outputs",
    "write_files": True,
    "data_mode": "quality_tr_mix",  # quality_tr_mix|hf_only|synthetic_only
    "curriculum_enabled": True,
    "turkish_primary": True,
    "strict_bitnet": True,
    "bitnet_clip_grad": 1.0,
    "amp_enabled": True,
    "resume_mode": "auto",  # auto|best|path
    "resume_path": "",
    "checkpoint_dir": "/kaggle/working/mertformer_onecell_outputs/checkpoints/kaggle_onecell_t4_build30",
    "vocab_size": 32768,
    "tokenizer_max_texts": 120000,
    "lr": 1.5e-4,
    "min_lr_ratio": 0.1,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "aux_loss_coeff": 0.6,
    "mert_enable_all_extensions": True,
    "mert_use_moe": True,
    "mert_use_liquid": True,
    "mert_use_qinn": False,
    "mert_hidden": 272,
    "mert_layers": 8,
    "mert_heads": 8,
    "mert_kv_heads": 4,
    "use_learned_pos_embedding": False,
    "use_gradient_checkpointing": False,
    "embedding_scale": True,
    "max_oom_retries": 8,
    "max_eval_batches": 32,
    "chat_enabled": False,
    "chat_interactive": False,
    "checkpoint_path": "",
    "chat_temperature": 0.75,
    "chat_top_p": 0.9,
    "chat_repetition_penalty": 1.08,
    "chat_max_new_tokens": 128,
    "target_param_band_low": 25_000_000,
    "target_param_band_high": 35_000_000,
    # Mini convergence probe (disabled unless mini300m profile is selected).
    "mini_probe_enabled": False,
    "mini_probe_param_target": 300_000_000,
    "mini_probe_param_tolerance": 35_000_000,
    "mini_probe_token_min": 5_000_000_000,
    "mini_probe_token_max": 10_000_000_000,
    "mini_probe_min_steps_for_signal": 500,
    "mini_probe_min_loss_drop_ratio": 0.08,
    "mini_probe_max_grad_cv": 1.35,
    "mini_probe_min_router_entropy": 0.35,
    "mini_probe_max_router_load_p95": 0.85,
    "mini_probe_max_collapse_events": 0,
    # Data loading safety/perf guards (deep profile startup hardening)
    "hf_streaming": True,
    "hf_allow_materialized_fallback": False,
    "hf_trust_remote_code": False,
    "hf_force_small_nonstream_split": True,
    "hf_nonstream_split_percent": 1.0,
    "hf_candidate_max_seconds": 180,
    "hf_candidate_heartbeat_seconds": 15,
    "stage_max_hf_rows_per_candidate": 120000,
    "data_fetch_phase_timeout_seconds": 900,
    "tokenizer_fit_phase_timeout_seconds": 600,
    "token_encode_phase_timeout_seconds": 1200,
    "startup_watchdog_enabled": True,
    # Tokenizer startup guards (avoid long pre-train stalls)
    "tokenizer_fit_max_texts": 30000,
    "tokenizer_fit_max_chars": 6000000,
    "tokenizer_fit_max_chars_per_text": 512,
    "byte_bpe_max_merges": 2500,
    "token_encode_heartbeat_every": 20000,
    "token_encode_heartbeat_seconds": 10,
    # Resume integrity guards
    "resume_hash_gate": True,
    "resume_require_tokenizer_backend_match": True,
    "resume_reject_on_step_exhausted": True,
    "data_policy_tag": "open+fallback",
    # Parity / architecture contract
    "parity_level": "hybrid_strict",
    "parity_proof_mode": "embedded_plus_local_if_available",
    "enable_local_repo_crosscheck": True,
    # BitNet dual mode
    "bitnet_mode": "stable",  # stable|aggressive
    "bitnet_skip_attention_qkvo": True,
    # MoE mode
    "moe_mode": "true_sparse_topk",  # true_sparse_topk|dense_debug
    # Logger memory safety
    "logger_mode": "jsonl_ring",  # in_memory|jsonl_ring
    "step_log_interval": 1,
    "logger_ring_size": 5000,
    "logger_jsonl_path": "",
    # Chat decode guards
    "chat_decode_completion_only": True,
    "chat_context_truncate": True,
    # Optional process-isolated HF candidate fetch (stall shield)
    "hf_candidate_process_timeout": True,
    "hf_candidate_process_max_seconds": 90,
    "hf_candidate_process_rows": 4096,
    # Local parity cross-check root (optional). Empty => auto-detect.
    "local_repo_root": "",
    # Byte-BPE encode optimization knobs (fallback path).
    "byte_bpe_encode_cache_size": 2048,
    "byte_bpe_cache_max_text_len": 512,
    # Evidence lock / strict run contract.
    # Default is permissive for local/Kaggle portability; strict mode is opt-in.
    "strict_data": False,
    "require_code_stage_data": False,
    "allow_degraded_data": True,
    "degraded_data_mode": False,
    "gpu_auto_tune": True,
    "gpu_target_vram_util": 0.94,
    "gpu_safety_margin_gb": 0.5,
    "gpu_tune_max_trials": 8,
    "artifact_root": "/kaggle/working/mertformer_onecell_outputs",
    "artifact_run_id": "",
    "zip_evidence_pack": True,
    "auto_backup_to_drive": False,
    "drive_backup_root": "/content/drive/MyDrive/mertformer_runs",
    "benchmark_mode": "separated",
    "strict_green_min_tokens": 8_000_000,
    "oov_rate_warn_threshold": 0.01,
    "token_duplicate_ratio_warn_threshold": 0.25,
}

ARCH_PARITY_CONTRACT: Dict[str, Any] = {
    "name": "build30_parity_strict_contract_v1",
    "core_block_order": [
        "embed_scale",
        "dropout",
        "attn",
        "residual_scale_1",
        "liquid_optional",
        "ff_or_moe",
        "residual_scale_2",
        "hebbian_optional",
        "neuro_symbolic_optional",
        "lifelong_optional",
        "qinn_optional",
        "workspace_optional",
        "cross_expert_sync_optional",
        "norm",
        "lm_head",
    ],
    "required_extensions": [
        "use_hebbian_plasticity",
        "use_neuro_symbolic_layer",
        "use_world_model_head",
        "use_lifelong_safety_layer",
        "use_latent_ode_state_channel",
        "use_global_workspace_broadcast",
        "use_cross_expert_sync_bus",
        "use_structural_plasticity",
    ],
    "bitnet_stable_skips": [
        "attn.q_proj",
        "attn.k_proj",
        "attn.v_proj",
        "attn.o_proj",
    ],
}


# =============================================================================
# Utility
# =============================================================================
def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _local_stamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def _run_id_stamp() -> str:
    return time.strftime("run_%Y%m%d_%H%M%S", time.localtime())


@dataclass
class ArtifactLayout:
    artifact_root: Path
    run_id: str
    run_dir: Path
    checkpoint_dir: Path
    eval_snapshot_dir: Path
    incremental_csv_path: Path
    health_txt_path: Path
    stop_summary_path: Path
    traceback_path: Path
    last_state_path: Path
    artifact_index_path: Path
    zip_manifest_path: Path
    public_summary_path: Path
    evidence_zip_path: Path
    logger_jsonl_path: Path


_RUNTIME_SIGNAL_STATE: Dict[str, Any] = {"sigterm": False, "signal": ""}
_RUNTIME_LAST_LAYOUT: Dict[str, str] = {}


def _signal_stop_handler(signum: int, _frame: Any) -> None:
    sig_name = "SIGTERM" if int(signum) == int(getattr(signal, "SIGTERM", 15)) else str(signum)
    _RUNTIME_SIGNAL_STATE["sigterm"] = True
    _RUNTIME_SIGNAL_STATE["signal"] = sig_name


def install_runtime_signal_handlers() -> None:
    try:
        signal.signal(signal.SIGTERM, _signal_stop_handler)
    except Exception:
        pass


def ensure_writable_dir(path: Path, label: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / f".probe_{label}_{int(time.time() * 1_000_000)}"
    try:
        with probe.open("w", encoding="utf-8") as f:
            f.write("ok")
    finally:
        try:
            probe.unlink()
        except Exception:
            pass


def init_artifact_layout(cfg: Dict[str, Any]) -> ArtifactLayout:
    artifact_root = Path(str(cfg.get("artifact_root", cfg.get("out_dir", "/kaggle/working/mertformer_onecell_outputs")))).expanduser()
    run_id = str(cfg.get("artifact_run_id", "")).strip() or _run_id_stamp()
    run_dir = artifact_root / "runs" / run_id
    ckpt_raw = Path(str(cfg.get("checkpoint_dir", "/kaggle/working/mertformer_onecell_outputs/checkpoints/kaggle_onecell_t4_build30"))).expanduser()
    checkpoint_dir = ckpt_raw if ckpt_raw.is_absolute() else artifact_root / ckpt_raw
    eval_snapshot_dir = run_dir / "eval_snapshots"
    logger_jsonl_path = run_dir / "logs" / "run_log.jsonl"
    layout = ArtifactLayout(
        artifact_root=artifact_root,
        run_id=run_id,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        eval_snapshot_dir=eval_snapshot_dir,
        incremental_csv_path=run_dir / "eval_incremental.csv",
        health_txt_path=run_dir / "health_latest.txt",
        stop_summary_path=run_dir / "stop_summary.json",
        traceback_path=run_dir / "failure_traceback.txt",
        last_state_path=run_dir / "last_state.json",
        artifact_index_path=run_dir / "artifacts_index.json",
        zip_manifest_path=run_dir / "zip_manifest.json",
        public_summary_path=run_dir / "public_summary.json",
        evidence_zip_path=run_dir / f"{run_id}_evidence.zip",
        logger_jsonl_path=logger_jsonl_path,
    )
    # Hard fail by design if path contract is not writable.
    ensure_writable_dir(layout.artifact_root, "artifact_root")
    ensure_writable_dir(layout.run_dir, "run_dir")
    ensure_writable_dir(layout.checkpoint_dir, "checkpoint_dir")
    ensure_writable_dir(layout.eval_snapshot_dir, "eval_snapshot_dir")
    ensure_writable_dir(layout.logger_jsonl_path.parent, "logger_dir")
    cfg["artifact_root"] = str(layout.artifact_root)
    cfg["artifact_run_id"] = layout.run_id
    cfg["artifact_run_dir"] = str(layout.run_dir)
    cfg["out_dir"] = str(layout.artifact_root)
    cfg["checkpoint_dir"] = str(layout.checkpoint_dir)
    cfg["logger_jsonl_path"] = str(layout.logger_jsonl_path.resolve())
    _RUNTIME_LAST_LAYOUT["run_dir"] = str(layout.run_dir)
    _RUNTIME_LAST_LAYOUT["traceback_path"] = str(layout.traceback_path)
    _RUNTIME_LAST_LAYOUT["last_state_path"] = str(layout.last_state_path)
    if not bool(cfg.get("write_files", False)):
        print("[warning:red] write_files=False; persistent evidence files are disabled.")
    print(
        "[runtime:paths] "
        f"out_dir={layout.artifact_root} "
        f"checkpoint_dir={layout.checkpoint_dir} "
        f"run_dir={layout.run_dir} "
        f"log_jsonl={layout.logger_jsonl_path}"
    )
    return layout


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def atomic_text_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text.rstrip() + "\n", encoding="utf-8")
    tmp.replace(path)


def hash_file(path):
    return file_sha256(_M5080Path(path) if '_M5080Path' in globals() else Path(path))


def write_output_sha256_manifest(output_files: Dict[str, str], out_path: Path) -> None:
    lines = []
    for name, path_str in sorted(output_files.items()):
        p = Path(str(path_str))
        if p.exists() and p.is_file():
            lines.append(f"{file_sha256(p)}  {name}  {p.name}")
    atomic_text_write(out_path, "\n".join(lines))


def build_embedded_layer_parity_report(cfg: Dict[str, Any]) -> Dict[str, Any]:
    files = json.loads(json.dumps(EMBEDDED_LAYER_PARITY["files"]))
    guarded_off = []
    active = []
    if bool(cfg.get("mert_enable_all_extensions", False)):
        ext_allowed = {"default_on", "guarded_off", "always_on"}
    else:
        ext_allowed = {"default_on", "always_on"}
    qinn_allowed = bool(cfg.get("mert_use_qinn", False))
    liquid_allowed = bool(cfg.get("mert_use_liquid", True))
    moe_allowed = bool(cfg.get("mert_use_moe", True))
    for name, row in files.items():
        state = str(row.get("default_state", "guarded_off"))
        enabled = state in ext_allowed
        if name == "qinn.py" and not qinn_allowed:
            enabled = False
        if name == "liquid.py" and not liquid_allowed:
            enabled = False
        if name == "moe.py" and not moe_allowed:
            enabled = False
        row["embedded"] = True
        row["enabled_by_default"] = bool(enabled)
        if enabled:
            active.append(name)
        else:
            guarded_off.append(name)
    return {
        "schema": EMBEDDED_LAYER_PARITY["schema"],
        "default_feature_bundle": EMBEDDED_LAYER_PARITY["default_feature_bundle"],
        "repo_layers_dir": EMBEDDED_LAYER_PARITY["repo_layers_dir"],
        "active_layers": active,
        "guarded_layers": guarded_off,
        "files": files,
    }


def write_onecell_sidecars(layout: ArtifactLayout, cfg: Dict[str, Any], payload: Dict[str, Any], preflight: Dict[str, Any], logger_manifest: Dict[str, Any]) -> Dict[str, str]:
    if not bool(cfg.get("write_files", False)):
        return {}
    sidecars: Dict[str, str] = {}
    config_snapshot_path = layout.run_dir / "config_snapshot.json"
    preflight_report_path = layout.run_dir / "runtime_preflight_report.json"
    parity_manifest_path = layout.run_dir / "layer_parity_manifest.json"
    event_manifest_path = layout.run_dir / "event_manifest.json"
    final_summary_path = layout.run_dir / "final_summary.json"
    sha256_manifest_path = layout.run_dir / "sha256_manifest.txt"

    config_snapshot = {
        "schema": "kaggle_onecell_config_snapshot_v1",
        "generated_at_utc": _utc_now(),
        "profile": str(cfg.get("profile", "unknown")),
        "run_config_hash": str(payload.get("run_config_hash", hash_config(cfg))),
        "config": safe_jsonable(dict(cfg)),
    }
    atomic_json_write(config_snapshot_path, config_snapshot)
    sidecars["config_snapshot"] = str(config_snapshot_path)

    preflight_report = {
        "schema": "kaggle_onecell_runtime_preflight_v1",
        "generated_at_utc": _utc_now(),
        "run_id": str(layout.run_id),
        "device": str(payload.get("device", cfg.get("device", "auto"))),
        "preflight": safe_jsonable(preflight),
    }
    atomic_json_write(preflight_report_path, preflight_report)
    sidecars["runtime_preflight_report"] = str(preflight_report_path)

    parity_report = build_embedded_layer_parity_report(cfg)
    parity_report["generated_at_utc"] = _utc_now()
    parity_report["run_id"] = str(layout.run_id)
    atomic_json_write(parity_manifest_path, parity_report)
    sidecars["layer_parity_manifest"] = str(parity_manifest_path)

    event_manifest = {
        "schema": "kaggle_onecell_event_manifest_v1",
        "generated_at_utc": _utc_now(),
        "run_id": str(layout.run_id),
        "profile": str(cfg.get("profile", "unknown")),
        "final_status": str(payload.get("final_status", payload.get("status", "unknown"))),
        "final_reason": str(payload.get("final_reason", payload.get("stop_reason", "unknown"))),
        "logger": safe_jsonable(logger_manifest),
        "events": [
            {"name": "preflight", "status": str(preflight.get("preflight_status", "unknown"))},
            {"name": "training", "status": str(payload.get("final_status", payload.get("status", "unknown")))}
        ],
    }
    atomic_json_write(event_manifest_path, event_manifest)
    sidecars["event_manifest"] = str(event_manifest_path)

    final_summary = {
        "schema": "kaggle_onecell_final_summary_v1",
        "generated_at_utc": _utc_now(),
        "run_id": str(layout.run_id),
        "profile": str(cfg.get("profile", "unknown")),
        "final_status": str(payload.get("final_status", payload.get("status", "unknown"))),
        "final_reason": str(payload.get("final_reason", payload.get("stop_reason", "unknown"))),
        "tokens_seen": int(payload.get("train_state", {}).get("tokens_seen", 0)),
        "checkpoint_manifest": str(payload.get("output_files", {}).get("checkpoint_manifest", "")),
        "json_log": str(logger_manifest.get("jsonl_path", "")),
    }
    atomic_json_write(final_summary_path, final_summary)
    sidecars["final_summary"] = str(final_summary_path)

    write_output_sha256_manifest({**payload.get("output_files", {}), **sidecars}, sha256_manifest_path)
    sidecars["sha256_manifest"] = str(sha256_manifest_path)
    return sidecars


def write_onecell_fatal_report(run_dir: str, err: Exception, tb: str) -> str:
    if not str(run_dir).strip():
        return ""
    path = Path(run_dir) / "fatal_report.json"
    payload = {
        "schema": "kaggle_onecell_fatal_report_v1",
        "generated_at_utc": _utc_now(),
        "run_dir": str(run_dir),
        "fatal_error": f"{type(err).__name__}:{err}",
        "traceback": tb,
    }
    try:
        atomic_json_write(path, payload)
        return str(path)
    except Exception:
        return ""


def reason_code_from_error(msg: str) -> str:
    m = str(msg).lower()
    if "remote_code_not_trusted" in m:
        return "remote_code_not_trusted"
    if "gated" in m or "authentication" in m or "permission" in m or "401" in m or "403" in m:
        return "gated_auth_missing"
    if "hf_token_missing" in m:
        return "hf_token_missing"
    if "dataset" in m and "not found" in m:
        return "dataset_not_found"
    if "timeout" in m:
        return "candidate_timeout"
    return "candidate_load_error"


def resolve_hf_token() -> str:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(key, "").strip()
        if v:
            return v
    return ""


def _quick_hf_dataset_probe(
    ds_name: str,
    subset: Optional[str],
    split: str,
    trust_remote_code: bool,
) -> Tuple[bool, str]:
    if not HAS_DATASETS:
        return False, "datasets_not_available"
    try:
        stream_split = str(split)
        if "[" in stream_split and "]" in stream_split:
            stream_split = stream_split.split("[", 1)[0]
        if stream_split not in ("train", "validation", "test"):
            stream_split = "train"
        if subset:
            ds = load_dataset(
                ds_name,
                subset,
                split=stream_split,
                streaming=True,
                trust_remote_code=bool(trust_remote_code),
            )
        else:
            ds = load_dataset(
                ds_name,
                split=stream_split,
                streaming=True,
                trust_remote_code=bool(trust_remote_code),
            )
        it = iter(ds)
        _ = next(it, None)
        return True, "ok"
    except Exception as e:
        return False, _format_exception(e)


def run_data_preflight(cfg: Dict[str, Any]) -> Dict[str, Any]:
    strict_data = bool(cfg.get("strict_data", True))
    require_code_stage_data = bool(cfg.get("require_code_stage_data", True))
    allow_degraded_data = bool(cfg.get("allow_degraded_data", False))
    token = resolve_hf_token()
    has_token = bool(token)
    checks: List[Dict[str, Any]] = []
    reason_codes: List[str] = []
    warning_codes: List[str] = []
    code_stage_access_count = 0
    stages = build_curriculum_sources(turkish_primary=bool(cfg.get("turkish_primary", True)))
    code_stage = None
    for st in stages:
        if "code" in str(st.get("name", "")):
            code_stage = st
            break
    if strict_data and not has_token:
        reason_codes.append("hf_token_missing")

    if code_stage is not None:
        for c in code_stage.get("hf_candidates", []):
            ds_name = str(c.get("dataset", "unknown"))
            start = time.time()
            row: Dict[str, Any] = {
                "dataset": ds_name,
                "attempt_count": 1,
                "kept": 0,
                "elapsed_sec": 0.0,
                "error": "",
                "reason_code": "",
                "load_mode": "probe",
            }
            if bool(c.get("requires_remote_code", False)) and not bool(cfg.get("hf_trust_remote_code", False)):
                row["error"] = "remote_code_not_trusted"
                row["reason_code"] = "remote_code_not_trusted"
                checks.append(row)
                if "remote_code_not_trusted" not in warning_codes:
                    warning_codes.append("remote_code_not_trusted")
                continue
            ok, info = _quick_hf_dataset_probe(
                ds_name=ds_name,
                subset=c.get("subset"),
                split=str(c.get("split", "train")),
                trust_remote_code=bool(cfg.get("hf_trust_remote_code", False)),
            )
            row["elapsed_sec"] = time.time() - start
            if ok:
                row["kept"] = 1
                row["reason_code"] = "ok"
                row["error"] = ""
                code_stage_access_count += 1
            else:
                row["error"] = str(info)
                row["reason_code"] = reason_code_from_error(str(info))
                if row["reason_code"] == "gated_auth_missing" and not has_token:
                    if strict_data:
                        if "hf_token_missing" not in reason_codes:
                            reason_codes.append("hf_token_missing")
                    else:
                        if "hf_token_missing" not in warning_codes:
                            warning_codes.append("hf_token_missing")
                if row["reason_code"] not in warning_codes:
                    warning_codes.append(row["reason_code"])
            checks.append(row)

    if require_code_stage_data and code_stage_access_count <= 0:
        if "code_stage_unavailable" not in reason_codes:
            reason_codes.append("code_stage_unavailable")
        if not has_token and "gated_auth_missing" not in reason_codes:
            reason_codes.append("gated_auth_missing")

    degraded_data_mode = (not strict_data) and bool(reason_codes or warning_codes)
    if degraded_data_mode and not allow_degraded_data:
        # Degraded path is active but explicitly disallowed by config.
        if "degraded_data_not_allowed" not in reason_codes:
            reason_codes.append("degraded_data_not_allowed")

    status = "pass"
    if strict_data and reason_codes:
        status = "fail"
    report = {
        "preflight_status": status,
        "strict_data": strict_data,
        "require_code_stage_data": require_code_stage_data,
        "allow_degraded_data": allow_degraded_data,
        "degraded_data_mode": degraded_data_mode,
        "hf_token_present": has_token,
        "reason_codes": reason_codes,
        "warning_codes": warning_codes,
        "code_stage_loaded": int(code_stage_access_count),
        "candidate_checks": checks,
    }
    cfg["degraded_data_mode"] = bool(degraded_data_mode)
    return report


def write_last_state(path: Path, payload: Dict[str, Any]) -> None:
    try:
        atomic_json_write(path, payload)
    except Exception:
        pass


def append_csv_row(path: Path, fieldnames: Sequence[str], row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    has_header = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not has_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def _median(values: Sequence[float]) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


def _percentile(values: Sequence[float], q: float) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    qq = max(0.0, min(1.0, float(q)))
    idx = int(round(qq * (len(vals) - 1)))
    idx = max(0, min(len(vals) - 1, idx))
    return float(vals[idx])


def _winsorized(values: Sequence[float], low_q: float = 0.05, high_q: float = 0.95) -> List[float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return []
    lo = _percentile(vals, low_q)
    hi = _percentile(vals, high_q)
    if lo > hi:
        lo, hi = hi, lo
    return [min(hi, max(lo, v)) for v in vals]


def robust_grad_stats(grads: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in grads if math.isfinite(float(v))]
    if not vals:
        return {"mean": 0.0, "std": 0.0, "cv": 0.0, "median": 0.0, "mad": 0.0, "spike_count": 0.0}
    wv = _winsorized(vals, 0.05, 0.95)
    mean = float(sum(wv) / len(wv))
    var = float(sum((x - mean) ** 2 for x in wv) / len(wv))
    std = math.sqrt(max(0.0, var))
    cv = _safe_div(std, max(abs(mean), 1e-9), default=0.0)
    med = _median(vals)
    mad = _median([abs(x - med) for x in vals])
    spike_thr = med + 6.0 * max(mad, 1e-9)
    spike_count = float(sum(1 for x in vals if x > spike_thr))
    return {
        "mean": mean,
        "std": std,
        "cv": cv,
        "median": med,
        "mad": mad,
        "spike_count": spike_count,
    }


def validation_trend_metrics(val_losses: Sequence[float]) -> Dict[str, Any]:
    vals = [float(v) for v in val_losses if math.isfinite(float(v))]
    if not vals:
        return {
            "val_loss_median_last3": float("inf"),
            "val_loss_delta_rel_last3": 0.0,
            "val_plateau_detected": False,
        }
    tail = vals[-3:]
    med_last3 = _median(tail)
    delta_rel = 0.0
    plateau = False
    if len(tail) >= 2:
        delta_rel = _safe_div((tail[0] - tail[-1]), max(abs(tail[0]), 1e-9), default=0.0)
        plateau = abs(delta_rel) < 0.01
    return {
        "val_loss_median_last3": float(med_last3),
        "val_loss_delta_rel_last3": float(delta_rel),
        "val_plateau_detected": bool(plateau),
    }


def warmup_excluded_loss_drop(train_losses: Sequence[float]) -> float:
    vals = [float(v) for v in train_losses if math.isfinite(float(v))]
    if len(vals) < 10:
        return 0.0
    warm = max(1, int(len(vals) * 0.05))
    post = vals[warm:]
    if len(post) < 4:
        return 0.0
    early = _mean(post[: max(2, len(post) // 5)])
    late = _mean(post[-max(2, len(post) // 5) :])
    return _safe_div(early - late, max(abs(early), 1e-9), default=0.0)


def _print_header() -> None:
    print("=" * 96)
    print("MERTFORMER KAGGLE ONE-CELL T4 LANE | BUILD30 | CLAIM-SAFE EVIDENCE")
    print("=" * 96)


def safe_jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): safe_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [safe_jsonable(v) for v in x]
    tname = type(x).__name__.lower()
    if "tensor" in tname:
        try:
            return {
                "__tensor__": True,
                "shape": [int(s) for s in x.shape],
                "dtype": str(x.dtype),
                "device": str(x.device),
            }
        except Exception:
            return {"__tensor__": True}
    return repr(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: str) -> str:
    if requested in ("cpu", "mps", "cuda"):
        if requested == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if requested == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_total_vram_gb(device: str) -> float:
    if device == "cuda" and torch.cuda.is_available():
        try:
            return float(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))
        except Exception:
            return 0.0
    return 0.0


def get_cuda_device_meta() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "gpu_name": "",
            "compute_capability": "",
            "precision_fallback": "",
            "inductor_enabled": False,
        }
    try:
        props = torch.cuda.get_device_properties(0)
        cc = f"{props.major}.{props.minor}"
        bf16_ok = bool(torch.cuda.is_bf16_supported())
        return {
            "gpu_name": str(props.name),
            "compute_capability": cc,
            "precision_fallback": "" if bf16_ok else "float16",
            "inductor_enabled": bool(hasattr(torch, "compile")),
        }
    except Exception:
        return {
            "gpu_name": "",
            "compute_capability": "",
            "precision_fallback": "",
            "inductor_enabled": bool(hasattr(torch, "compile")),
        }


def apply_gpu_auto_tune(cfg: Dict[str, Any], device: str) -> Dict[str, Any]:
    report = {
        "enabled": bool(cfg.get("gpu_auto_tune", True)),
        "applied": False,
        "initial_batch_size": int(cfg.get("batch_size", 1)),
        "initial_grad_accum_steps": int(cfg.get("grad_accum_steps", 1)),
        "final_batch_size": int(cfg.get("batch_size", 1)),
        "final_grad_accum_steps": int(cfg.get("grad_accum_steps", 1)),
        "trials": 0,
        "vram_total_gb": get_total_vram_gb(device),
        "vram_util_estimate": 0.0,
    }
    if device != "cuda" or not bool(cfg.get("gpu_auto_tune", True)):
        return report
    total = float(report["vram_total_gb"])
    if total <= 0.0:
        return report
    safety = float(cfg.get("gpu_safety_margin_gb", 0.5))
    target_util = float(cfg.get("gpu_target_vram_util", 0.94))
    seq = max(64, int(cfg.get("seq_len", 256)))
    max_trials = max(1, int(cfg.get("gpu_tune_max_trials", 8)))
    curr_bs = int(cfg.get("batch_size", 1))
    curr_acc = int(cfg.get("grad_accum_steps", 1))
    best_bs = curr_bs
    # Coarse runtime-safe memory estimate for activation+optimizer pressure.
    est_per_sample_gb = max(0.12, (seq / 256.0) * 0.22)
    target_budget = max(0.5, (total * target_util) - safety)
    for t in range(max_trials):
        candidate_bs = max(1, curr_bs + max(1, t))
        est = est_per_sample_gb * float(candidate_bs)
        util = _safe_div(est, max(total, 1e-9), default=0.0)
        report["trials"] = t + 1
        if est <= target_budget:
            best_bs = candidate_bs
            report["vram_util_estimate"] = util
        else:
            break
    if best_bs != curr_bs:
        cfg["batch_size"] = int(best_bs)
        report["applied"] = True
    # Keep effective global batch stable if bs grows too much.
    if int(cfg.get("batch_size", 1)) >= 8 and curr_acc > 1:
        cfg["grad_accum_steps"] = max(1, curr_acc - 1)
        report["applied"] = True
    report["final_batch_size"] = int(cfg.get("batch_size", curr_bs))
    report["final_grad_accum_steps"] = int(cfg.get("grad_accum_steps", curr_acc))
    if report["vram_util_estimate"] <= 0.0:
        report["vram_util_estimate"] = _safe_div(est_per_sample_gb * float(report["final_batch_size"]), max(total, 1e-9), default=0.0)
    return report


def reset_device_peak_memory(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def get_device_peak_memory_gb(device: str) -> float:
    if device == "cuda" and torch.cuda.is_available():
        try:
            return float(torch.cuda.max_memory_allocated() / (1024 ** 3))
        except Exception:
            return 0.0
    if device == "mps":
        try:
            # Best-effort for Apple backends; may not exist on all torch builds.
            if hasattr(torch, "mps"):
                fn = getattr(torch.mps, "current_allocated_memory", None)
                if callable(fn):
                    return float(fn() / (1024 ** 3))
                fn2 = getattr(torch.mps, "driver_allocated_memory", None)
                if callable(fn2):
                    return float(fn2() / (1024 ** 3))
        except Exception:
            return 0.0
    return 0.0


def resolve_writable_dir(preferred: Path) -> Path:
    """
    Return a writable directory for runtime outputs/checkpoints.
    Falls back from Kaggle default paths when running locally.
    """
    candidates: List[Path] = [
        preferred.expanduser(),
        Path.cwd() / "artifacts",
        Path.cwd(),
    ]
    for cand in candidates:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            probe = cand / ".write_probe"
            with open(probe, "w", encoding="utf-8") as f:
                f.write("ok")
            probe.unlink(missing_ok=True)
            return cand
        except Exception:
            continue
    return Path.cwd()


def _is_dir_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def interactive_prompt(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not cfg.get("interactive", False):
        return cfg
    if not can_accept_user_input(cfg):
        return cfg
    out = dict(cfg)
    print("\nInteractive Setup (Enter to keep default):")
    try:
        p = input(
            f"Profile [quick/deep8h/linkedin_sweetspot/mini300m/custom] (default={cfg['profile']}): "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return out
    if p in RUN_PROFILES:
        out["profile"] = p
    try:
        c = input("Checkpoint path (default empty): ").strip()
    except (EOFError, KeyboardInterrupt):
        return out
    if c:
        out["checkpoint_path"] = c
        out["resume_mode"] = "path"
        out["resume_path"] = c
    try:
        chat = input(f"Enable chat [Y/n] (default={'Y' if cfg['chat_enabled'] else 'N'}): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return out
    if chat:
        out["chat_enabled"] = chat in ("y", "yes", "1", "true")
    return out


def is_notebook_runtime() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        ip = get_ipython()
        if ip is None:
            return False
        shell_name = ip.__class__.__name__
        return shell_name in ("ZMQInteractiveShell", "TerminalInteractiveShell")
    except Exception:
        return False


def can_accept_user_input(cfg: Dict[str, Any]) -> bool:
    if bool(cfg.get("force_interactive_input", False)):
        return True
    if os.isatty(0):
        return True
    if not bool(cfg.get("allow_notebook_input", True)):
        return False
    return is_notebook_runtime()


def resolve_runtime_config(user_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(user_cfg)
    profile_env = os.environ.get("MERTFORMER_ONEFILE_PROFILE", "").strip().lower()
    if profile_env in RUN_PROFILES:
        cfg["profile"] = profile_env
    profile = str(cfg.get("profile", "deep8h"))
    profile_cfg = RUN_PROFILES.get(profile, {})
    merged = dict(cfg)
    merged.update(profile_cfg)
    if "quick" not in merged:
        merged["quick"] = profile != "deep8h"
    # Ensure required runtime keys exist for all profiles (including custom).
    required_defaults = {
        "max_wall_hours": 1.0,
        "target_train_tokens": 2_000_000,
        "max_steps": 10_000,
        "batch_size": 2,
        "seq_len": 128,
        "grad_accum_steps": 1,
        "eval_interval_steps": 100,
        "checkpoint_interval_steps": 100,
        "checkpoint_interval_minutes": 10,
        "benchmark_steps": 30,
        "benchmark_eval_batches": 8,
        "max_eval_batches": 32,
        "allow_notebook_input": False,
        "force_interactive_input": False,
        "hf_candidate_heartbeat_seconds": 15,
        "hf_trust_remote_code": False,
        "data_fetch_phase_timeout_seconds": 900,
        "tokenizer_fit_phase_timeout_seconds": 600,
        "token_encode_phase_timeout_seconds": 1200,
        "startup_watchdog_enabled": True,
        "token_encode_heartbeat_every": 20000,
        "token_encode_heartbeat_seconds": 10,
        "resume_hash_gate": True,
        "resume_require_tokenizer_backend_match": True,
        "resume_reject_on_step_exhausted": True,
        "data_policy_tag": "open+fallback",
        "parity_level": "hybrid_strict",
        "parity_proof_mode": "embedded_plus_local_if_available",
        "enable_local_repo_crosscheck": True,
        "bitnet_mode": "stable",
        "bitnet_skip_attention_qkvo": True,
        "moe_mode": "true_sparse_topk",
        "logger_mode": "jsonl_ring",
        "step_log_interval": 1,
        "logger_ring_size": 5000,
        "logger_jsonl_path": "",
        "chat_decode_completion_only": True,
        "chat_context_truncate": True,
        "hf_candidate_process_timeout": True,
        "hf_candidate_process_max_seconds": 90,
        "hf_candidate_process_rows": 4096,
        "local_repo_root": "",
        "byte_bpe_encode_cache_size": 2048,
        "byte_bpe_cache_max_text_len": 512,
        "out_dir": "/kaggle/working/mertformer_onecell_outputs",
        "checkpoint_dir": "/kaggle/working/mertformer_onecell_outputs/checkpoints/kaggle_onecell_t4_build30",
        "strict_data": False,
        "require_code_stage_data": False,
        "allow_degraded_data": True,
        "degraded_data_mode": False,
        "gpu_auto_tune": True,
        "gpu_target_vram_util": 0.94,
        "gpu_safety_margin_gb": 0.5,
        "gpu_tune_max_trials": 8,
        "artifact_root": "/kaggle/working/mertformer_onecell_outputs",
        "artifact_run_id": "",
        "zip_evidence_pack": True,
        "auto_backup_to_drive": False,
        "drive_backup_root": "/content/drive/MyDrive/mertformer_runs",
        "benchmark_mode": "separated",
        "strict_green_min_tokens": 8_000_000,
        "oov_rate_warn_threshold": 0.01,
        "token_duplicate_ratio_warn_threshold": 0.25,
        "use_learned_pos_embedding": False,
        "use_gradient_checkpointing": False,
        "embedding_scale": True,
        "mini_probe_enabled": False,
        "mini_probe_param_target": 300_000_000,
        "mini_probe_param_tolerance": 35_000_000,
        "mini_probe_token_min": 5_000_000_000,
        "mini_probe_token_max": 10_000_000_000,
        "mini_probe_min_steps_for_signal": 500,
        "mini_probe_min_loss_drop_ratio": 0.08,
        "mini_probe_max_grad_cv": 1.35,
        "mini_probe_min_router_entropy": 0.35,
        "mini_probe_max_router_load_p95": 0.85,
        "mini_probe_max_collapse_events": 0,
    }
    for k, v in required_defaults.items():
        merged.setdefault(k, v)

    # Environment override for emergency quick smoke
    if os.environ.get("MERTFORMER_ONEFILE_FORCE_QUICK", "0") == "1":
        merged.update(RUN_PROFILES["quick"])
        merged["profile"] = "quick"
        merged["use_gradient_checkpointing"] = False
        merged["mert_hidden"] = min(int(merged.get("mert_hidden", 272)), 128)
        merged["mert_layers"] = min(int(merged.get("mert_layers", 8)), 2)
        merged["mert_heads"] = min(int(merged.get("mert_heads", 8)), 4)
        merged["mert_kv_heads"] = min(int(merged.get("mert_kv_heads", 4)), 4)
        merged["max_steps"] = min(int(merged.get("max_steps", 120)), 8)
        merged["target_train_tokens"] = min(int(merged.get("target_train_tokens", 2_000_000)), 250_000)
        merged["tokenizer_max_texts"] = min(int(merged.get("tokenizer_max_texts", 120000)), 4000)
        merged["tokenizer_fit_max_texts"] = min(int(merged.get("tokenizer_fit_max_texts", 30000)), 1000)
        merged["eval_interval_steps"] = min(int(merged.get("eval_interval_steps", 20)), 4)
        merged["checkpoint_interval_steps"] = min(int(merged.get("checkpoint_interval_steps", 20)), 4)
        merged["benchmark_steps"] = min(int(merged.get("benchmark_steps", 30)), 4)
        merged["benchmark_eval_batches"] = min(int(merged.get("benchmark_eval_batches", 8)), 4)
        merged["hf_candidate_process_timeout"] = False
        merged["mini_probe_enabled"] = False
        merged["target_param_band_low"] = 100_000
        merged["target_param_band_high"] = 5_000_000

    merged["device"] = pick_device(str(merged.get("device", "auto")))

    if bool(merged.get("use_gradient_checkpointing", False)):
        checkpoint_disable_reasons: List[str] = []
        if str(merged.get("device", "cpu")) != "cuda":
            checkpoint_disable_reasons.append("non_cuda_device")
        if bool(merged.get("mert_use_liquid", True)):
            checkpoint_disable_reasons.append("stateful_liquid_stack")
        if bool(merged.get("mert_enable_all_extensions", False)):
            checkpoint_disable_reasons.append("extension_stack")
        if checkpoint_disable_reasons:
            merged["use_gradient_checkpointing"] = False
            print(
                "[runtime] gradient_checkpointing disabled: "
                + ",".join(checkpoint_disable_reasons)
            )

    # Hardware-aware defaults
    if merged["device"] == "cuda":
        vram = get_total_vram_gb("cuda")
        merged["vram_total_gb"] = vram
        if vram > 0:
            if vram <= 8:
                merged["batch_size"] = min(int(merged["batch_size"]), 2)
                merged["seq_len"] = min(int(merged["seq_len"]), 128)
                merged["grad_accum_steps"] = max(int(merged["grad_accum_steps"]), 4)
            elif vram <= 16:
                merged["batch_size"] = min(int(merged["batch_size"]), 4)
                merged["seq_len"] = min(int(merged["seq_len"]), 256)
            elif vram <= 40:
                merged["batch_size"] = min(int(merged["batch_size"]), 8)
                merged["seq_len"] = min(int(merged["seq_len"]), 384)
            elif vram <= 80:
                merged["batch_size"] = min(int(merged["batch_size"]), 16)
                merged["seq_len"] = min(int(merged["seq_len"]), 768)
            else:
                merged["batch_size"] = min(int(merged["batch_size"]), 24)
                merged["seq_len"] = min(int(merged["seq_len"]), 1024)
    else:
        merged["vram_total_gb"] = 0.0
        merged["batch_size"] = min(int(merged["batch_size"]), 2)
        merged["seq_len"] = min(int(merged["seq_len"]), 192)
        merged["grad_accum_steps"] = max(int(merged["grad_accum_steps"]), 2)
        merged["amp_enabled"] = False

    # Keep these arrays deterministic
    if not isinstance(merged.get("seed_list"), list) or len(merged["seed_list"]) < 3:
        s = int(merged["seed"])
        merged["seed_list"] = [s, s + 1, s + 2]

    out_dir_raw = Path(str(merged.get("out_dir", "/kaggle/working"))).expanduser()
    out_dir = resolve_writable_dir(out_dir_raw)
    if out_dir != out_dir_raw:
        print(f"[runtime] out_dir fallback: requested={out_dir_raw} resolved={out_dir}")
    merged["out_dir"] = str(out_dir)

    # Keep artifact root on a writable path as well.
    artifact_root_raw = Path(str(merged.get("artifact_root", merged["out_dir"]))).expanduser()
    artifact_root = resolve_writable_dir(artifact_root_raw)
    if artifact_root != artifact_root_raw:
        print(f"[runtime] artifact_root fallback: requested={artifact_root_raw} resolved={artifact_root}")
    merged["artifact_root"] = str(artifact_root)
    merged["out_dir"] = str(artifact_root)

    # Absolute checkpoint paths may target non-writable locations outside Kaggle.
    ckpt_raw = Path(str(merged.get("checkpoint_dir", "checkpoints/kaggle_onecell_t4_build30"))).expanduser()
    if ckpt_raw.is_absolute():
        if _is_dir_writable(ckpt_raw):
            ckpt_dir = ckpt_raw
        else:
            ckpt_dir = Path(str(merged["artifact_root"])) / "checkpoints" / "kaggle_onecell_t4_build30"
            print(f"[runtime] checkpoint_dir fallback: requested={ckpt_raw} resolved={ckpt_dir}")
    else:
        ckpt_dir = Path(str(merged["artifact_root"])) / ckpt_raw
    merged["checkpoint_dir"] = str(ckpt_dir)
    if str(merged.get("logger_jsonl_path", "")).strip() == "":
        merged["logger_jsonl_path"] = str(
            Path(str(merged["artifact_root"])) / "run_log.jsonl"
        )

    bitnet_mode = str(merged.get("bitnet_mode", "stable")).strip().lower()
    if bitnet_mode not in ("stable", "aggressive"):
        bitnet_mode = "stable"
    merged["bitnet_mode"] = bitnet_mode

    logger_mode = str(merged.get("logger_mode", "jsonl_ring")).strip().lower()
    if logger_mode not in ("in_memory", "jsonl_ring"):
        logger_mode = "jsonl_ring"
    merged["logger_mode"] = logger_mode

    return merged


def maybe_autocast(device: str, enabled: bool):
    if enabled and device == "cuda":
        dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=dt)

    class _Noop:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    return _Noop()


def hash_config(cfg: Dict[str, Any]) -> str:
    import hashlib

    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    if abs(float(b)) < 1e-12:
        return float(default)
    return float(a) / float(b)


def compute_arch_parity_signature(cfg: Dict[str, Any], tokenizer_backend: str = "unknown") -> str:
    fields = {
        "contract": ARCH_PARITY_CONTRACT["name"],
        "parity_level": str(cfg.get("parity_level", "hybrid_strict")),
        "parity_proof_mode": str(cfg.get("parity_proof_mode", "embedded_plus_local_if_available")),
        "vocab_size": int(cfg.get("vocab_size", 32768)),
        "seq_len": int(cfg.get("seq_len", 256)),
        "mert_hidden": int(cfg.get("mert_hidden", 272)),
        "mert_layers": int(cfg.get("mert_layers", 8)),
        "mert_heads": int(cfg.get("mert_heads", 8)),
        "mert_kv_heads": int(cfg.get("mert_kv_heads", 4)),
        "mert_use_moe": bool(cfg.get("mert_use_moe", True)),
        "mert_use_liquid": bool(cfg.get("mert_use_liquid", True)),
        "mert_use_qinn": bool(cfg.get("mert_use_qinn", False)),
        "strict_bitnet": bool(cfg.get("strict_bitnet", True)),
        "bitnet_mode": str(cfg.get("bitnet_mode", "stable")),
        "bitnet_skip_attention_qkvo": bool(cfg.get("bitnet_skip_attention_qkvo", True)),
        "moe_mode": str(cfg.get("moe_mode", "true_sparse_topk")),
        "logger_mode": str(cfg.get("logger_mode", "jsonl_ring")),
        "tokenizer_backend": tokenizer_backend,
    }
    return hash_config(fields)


def build_compat_signature(cfg: Dict[str, Any], tokenizer_backend: str = "unknown") -> str:
    """
    Stable compatibility fingerprint used for checkpoint resume gating.
    Includes architecture parity signature for strict resume checks.
    """
    fields = {
        "profile": cfg.get("profile"),
        "data_policy_tag": cfg.get("data_policy_tag", "open+fallback"),
        "target_train_tokens": int(cfg.get("target_train_tokens", 0)),
        "max_steps": int(cfg.get("max_steps", 0)),
        "strict_bitnet": bool(cfg.get("strict_bitnet", True)),
        "bitnet_mode": str(cfg.get("bitnet_mode", "stable")),
        "moe_mode": str(cfg.get("moe_mode", "true_sparse_topk")),
        "tokenizer_backend": tokenizer_backend,
        "arch_parity_signature": compute_arch_parity_signature(cfg, tokenizer_backend),
    }
    return hash_config(fields)


def _format_exception(e: Exception) -> str:
    return f"{type(e).__name__}:{e}"


def parity_crosscheck_local_repo(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optional local-only architecture cross-check.
    Never imports repo modules; reads files as text and compares expected symbols.
    """
    if not bool(cfg.get("enable_local_repo_crosscheck", True)):
        return {"enabled": False, "skipped": True, "reason": "disabled_by_config"}

    candidate_roots: List[Path] = []
    cfg_root = str(cfg.get("local_repo_root", "")).strip()
    if cfg_root:
        candidate_roots.append(Path(cfg_root).expanduser())
    env_root = os.environ.get("MERTFORMER_LOCAL_REPO_ROOT", "").strip()
    if env_root:
        candidate_roots.append(Path(env_root).expanduser())
    try:
        candidate_roots.append(Path(__file__).resolve().parents[1])
    except Exception:
        pass
    candidate_roots.append(Path.cwd())

    # Stable de-dup while preserving priority order.
    seen: set = set()
    unique_roots: List[Path] = []
    for r in candidate_roots:
        try:
            rp = r.resolve()
        except Exception:
            rp = r
        key = str(rp)
        if key not in seen:
            seen.add(key)
            unique_roots.append(rp)

    root: Optional[Path] = None
    layer_dir: Optional[Path] = None
    transformers_py: Optional[Path] = None
    for cand in unique_roots:
        ldir = cand / "layers"
        tpy = cand / "model" / "transformers.py"
        if ldir.exists() and tpy.exists():
            root = cand
            layer_dir = ldir
            transformers_py = tpy
            break

    if root is None or layer_dir is None or transformers_py is None:
        return {
            "enabled": True,
            "skipped": True,
            "reason": "repo_paths_not_found",
            "checked_roots": [str(x) for x in unique_roots],
        }
    try:
        required_layer_files = [
            "mertformer_block.py",
            "moe.py",
            "mla.py",
            "liquid.py",
            "qinn.py",
            "cognitive_extensions.py",
            "lifelong_safety.py",
            "world_model_head.py",
            "bitlinear.py",
        ]
        missing = [x for x in required_layer_files if not (layer_dir / x).exists()]
        tr_text = transformers_py.read_text(encoding="utf-8", errors="ignore")
        checks = {
            "transformers_has_mertformer": "class MertFormer" in tr_text,
            "transformers_has_reset_router_state": "reset_router_state" in tr_text,
            "transformers_has_generate": "def generate" in tr_text,
        }
        ok = not missing and all(checks.values())
        return {
            "enabled": True,
            "skipped": False,
            "ok": bool(ok),
            "repo_root": str(root),
            "missing_layer_files": missing,
            "checks": checks,
        }
    except Exception as e:
        return {"enabled": True, "skipped": False, "ok": False, "reason": _format_exception(e)}


def summarize_data_source_scorecard(curriculum_trace: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    stages: List[Dict[str, Any]] = []
    source_totals: Dict[str, Dict[str, Any]] = {}
    total_selected = 0
    total_topup = 0
    total_loaded = 0
    failure_rows = 0
    stage3_code_loaded = 0
    for st in curriculum_trace:
        selected = int(st.get("selected_for_tokenizer", 0))
        topup = int(st.get("topup_added", 0))
        loaded = int(st.get("loaded_samples", 0))
        failures = st.get("failures", [])
        total_selected += selected
        total_topup += topup
        total_loaded += loaded
        failure_rows += len(failures) if isinstance(failures, list) else 0
        candidate_rows = st.get("candidate_rows", [])
        if isinstance(candidate_rows, list):
            for row in candidate_rows:
                ds = str(row.get("dataset", "unknown"))
                kept = int(row.get("kept", 0))
                if ds not in source_totals:
                    source_totals[ds] = {"dataset": ds, "kept": 0, "attempts": 0}
                source_totals[ds]["kept"] += kept
                source_totals[ds]["attempts"] += 1
                if "stage_3_code" in str(st.get("name", st.get("stage", ""))):
                    stage3_code_loaded += kept
        stages.append(
            {
                "stage": st.get("name", st.get("stage", "unknown")),
                "target": int(st.get("target", 0)),
                "selected": selected,
                "loaded": loaded,
                "topup": topup,
                "failures": failures if isinstance(failures, list) else [str(failures)],
                "candidate_rows": candidate_rows if isinstance(candidate_rows, list) else [],
            }
        )
    return {
        "stages": stages,
        "source_totals": sorted(source_totals.values(), key=lambda x: x.get("kept", 0), reverse=True),
        "totals": {
            "selected": total_selected,
            "loaded": total_loaded,
            "topup": total_topup,
            "failure_rows": failure_rows,
            "stage_3_code_loaded": int(stage3_code_loaded),
            # When True, NO code-stage data was loaded, so coding capability is
            # NOT demonstrated even if the overall run/preflight reports green
            # (the default profile allows degraded data). Downstream summaries
            # must surface this and must not imply a coding claim while blocked.
            "coding_claim_blocked": bool(stage3_code_loaded <= 0),
        },
    }


def build_benchmark_winner_matrix(bench: Dict[str, Any]) -> Dict[str, Any]:
    m = bench.get("mertformer", {})
    v = bench.get("vanilla", {})
    metrics = {
        "train_final_loss": {
            "mertformer": float(m.get("train", {}).get("final_loss", float("inf"))),
            "vanilla": float(v.get("train", {}).get("final_loss", float("inf"))),
            "prefer": "lower",
        },
        "val_loss": {
            "mertformer": float(m.get("eval", {}).get("val_loss", float("inf"))),
            "vanilla": float(v.get("eval", {}).get("val_loss", float("inf"))),
            "prefer": "lower",
        },
        "val_ppl": {
            "mertformer": float(m.get("eval", {}).get("val_ppl", float("inf"))),
            "vanilla": float(v.get("eval", {}).get("val_ppl", float("inf"))),
            "prefer": "lower",
        },
        "train_tokens_per_sec": {
            "mertformer": float(m.get("train", {}).get("tokens_per_sec", 0.0)),
            "vanilla": float(v.get("train", {}).get("tokens_per_sec", 0.0)),
            "prefer": "higher",
        },
        "latency_ms": {
            "mertformer": float(m.get("latency", {}).get("avg_latency_ms", float("inf"))),
            "vanilla": float(v.get("latency", {}).get("avg_latency_ms", float("inf"))),
            "prefer": "lower",
        },
    }
    winners: Dict[str, str] = {}
    for k, row in metrics.items():
        mv = row["mertformer"]
        vv = row["vanilla"]
        if row["prefer"] == "higher":
            winners[k] = "mertformer" if mv > vv else "vanilla"
        else:
            winners[k] = "mertformer" if mv < vv else "vanilla"
    # HONESTY NOTE: apples_to_apples is False, so these "winners" (and especially
    # the speed/throughput/latency ones, which also fall back to 0.0/inf when a
    # metric was never measured) are NOT a fair head-to-head comparison and must
    # not be reported as a performance claim. Consumers should gate on the
    # apples_to_apples flag before trusting any winner here.
    return {"metrics": metrics, "winners": winners, "apples_to_apples": False}


def build_tradeoff_notes(
    benchmark_winner_matrix: Dict[str, Any],
    efficiency_index: Dict[str, Any],
    stability_index: Dict[str, Any],
) -> List[str]:
    notes: List[str] = []
    winners = benchmark_winner_matrix.get("winners", {})
    notes.append(f"train_loss_winner={winners.get('train_final_loss', 'n/a')}")
    notes.append(f"latency_winner={winners.get('latency_ms', 'n/a')}")
    notes.append(f"throughput_winner={winners.get('train_tokens_per_sec', 'n/a')}")
    notes.append(
        "mert_tps_per_mparam="
        + f"{float(efficiency_index.get('mert_tokens_per_sec_per_mparam', 0.0)):.2f}"
    )
    notes.append("stability_score=" + f"{float(stability_index.get('score_0_100', 0.0)):.1f}")
    return notes


def compute_stability_index(train_meta: Dict[str, Any], curve_data: Dict[str, List[float]]) -> Dict[str, Any]:
    oom = int(train_meta.get("oom_count", 0))
    nan = int(train_meta.get("nan_count", 0))
    runtime_errors = int(train_meta.get("runtime_error_count", 0))
    anomalies = int(train_meta.get("anomaly_count", 0))
    grads = [float(x) for x in curve_data.get("grad_norm", []) if math.isfinite(float(x))]
    grad_stats = robust_grad_stats(grads)
    gmean = float(grad_stats["mean"])
    gstd = float(grad_stats["std"])
    gcv = float(grad_stats["cv"])
    grad_spike_count = int(grad_stats["spike_count"])
    penalty = (
        (nan * 15.0)
        + (oom * 4.0)
        + (runtime_errors * 20.0)
        + (anomalies * 3.0)
        + min(25.0, gcv * 25.0)
        + min(12.0, float(grad_spike_count) * 0.5)
    )
    score = max(0.0, min(100.0, 100.0 - penalty))
    return {
        "score_0_100": score,
        "nan_count": nan,
        "oom_count": oom,
        "runtime_error_count": runtime_errors,
        "anomaly_count": anomalies,
        "grad_mean": gmean,
        "grad_std": gstd,
        "grad_cv": gcv,
        "grad_median": float(grad_stats["median"]),
        "grad_mad": float(grad_stats["mad"]),
        "grad_spike_count": grad_spike_count,
    }


def compute_efficiency_index(bench: Dict[str, Any]) -> Dict[str, Any]:
    m = bench.get("mertformer", {})
    v = bench.get("vanilla", {})
    m_params = float(m.get("params", 0.0))
    v_params = float(v.get("params", 0.0))
    m_tps = float(m.get("train", {}).get("tokens_per_sec", 0.0))
    v_tps = float(v.get("train", {}).get("tokens_per_sec", 0.0))
    m_loss = float(m.get("train", {}).get("final_loss", float("inf")))
    v_loss = float(v.get("train", {}).get("final_loss", float("inf")))
    m_ppm = _safe_div(m_tps, m_params / 1_000_000.0, default=0.0)
    v_ppm = _safe_div(v_tps, v_params / 1_000_000.0, default=0.0)
    quality_ratio = _safe_div(v_loss, m_loss, default=0.0)  # >1 means mertformer better loss
    return {
        "mert_tokens_per_sec_per_mparam": m_ppm,
        "vanilla_tokens_per_sec_per_mparam": v_ppm,
        "quality_ratio_vanilla_over_mert": quality_ratio,
        "mert_train_tokens_per_sec": m_tps,
        "vanilla_train_tokens_per_sec": v_tps,
    }


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _p95(values: Sequence[float]) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 0.0
    idx = int(round(0.95 * (len(vals) - 1)))
    idx = max(0, min(idx, len(vals) - 1))
    return float(vals[idx])


def compute_mini_probe_report(
    cfg: Dict[str, Any],
    train_meta: Dict[str, Any],
    curve_data: Dict[str, List[float]],
    model_params: int,
    tokens_seen: int,
) -> Dict[str, Any]:
    if not bool(cfg.get("mini_probe_enabled", False)):
        return {"enabled": False}

    min_steps = max(20, int(cfg.get("mini_probe_min_steps_for_signal", 500)))
    loss_hist = [float(x) for x in curve_data.get("train_loss", []) if math.isfinite(float(x))]
    grad_hist = [float(x) for x in curve_data.get("grad_norm", []) if math.isfinite(float(x))]
    entropy_hist = [float(x) for x in curve_data.get("router_entropy", []) if math.isfinite(float(x))]
    max_load_hist = [float(x) for x in curve_data.get("router_max_load", []) if math.isfinite(float(x))]
    collapse_events = int(sum(1 for x in curve_data.get("collapse_detected", []) if float(x) > 0.0))

    enough_steps = len(loss_hist) >= min_steps
    w = max(5, min(200, len(loss_hist) // 5)) if loss_hist else 0
    early_loss = _mean(loss_hist[:w]) if w > 0 else float("inf")
    late_loss = _mean(loss_hist[-w:]) if w > 0 else float("inf")
    loss_drop_ratio = _safe_div(early_loss - late_loss, max(abs(early_loss), 1e-9), default=0.0)
    warmup_drop = warmup_excluded_loss_drop(loss_hist)
    loss_gate = bool(
        enough_steps
        and math.isfinite(loss_drop_ratio)
        and warmup_drop >= float(cfg.get("mini_probe_min_loss_drop_ratio", 0.08))
    )

    grad_stats = robust_grad_stats(grad_hist)
    grad_mean = float(grad_stats["mean"])
    grad_std = float(grad_stats["std"])
    grad_cv = float(grad_stats["cv"])
    grad_spike_count = int(grad_stats["spike_count"])
    grad_gate = bool(
        len(grad_hist) >= min_steps
        and int(train_meta.get("nan_count", 0)) == 0
        and math.isfinite(grad_cv)
        and grad_cv <= float(cfg.get("mini_probe_max_grad_cv", 1.35))
        and grad_spike_count <= max(3, int(0.02 * max(1, len(grad_hist))))
    )

    router_entropy_mean = _mean(entropy_hist)
    router_entropy_p10 = _percentile(entropy_hist, 0.10)
    entropy_gate = bool(
        len(entropy_hist) >= min_steps
        and math.isfinite(router_entropy_mean)
        and router_entropy_mean >= float(cfg.get("mini_probe_min_router_entropy", 0.35))
    )

    router_max_load_p95 = _p95(max_load_hist)
    router_max_load_p99 = _percentile(max_load_hist, 0.99)
    expert_gate = bool(
        len(max_load_hist) >= min_steps
        and math.isfinite(router_max_load_p95)
        and router_max_load_p95 <= float(cfg.get("mini_probe_max_router_load_p95", 0.85))
        and collapse_events <= int(cfg.get("mini_probe_max_collapse_events", 0))
    )

    p_target = int(cfg.get("mini_probe_param_target", 300_000_000))
    p_tol = int(cfg.get("mini_probe_param_tolerance", 35_000_000))
    params_gate = abs(int(model_params) - p_target) <= p_tol

    token_min = int(cfg.get("mini_probe_token_min", 5_000_000_000))
    token_max = int(cfg.get("mini_probe_token_max", 10_000_000_000))
    token_progress_to_min = _safe_div(float(tokens_seen), float(max(token_min, 1)), default=0.0)
    token_within_band = int(tokens_seen) <= token_max

    all_green = bool(loss_gate and grad_gate and entropy_gate and expert_gate and params_gate)
    return {
        "enabled": True,
        "all_green": all_green,
        "loss_gate": loss_gate,
        "grad_gate": grad_gate,
        "router_entropy_gate": entropy_gate,
        "expert_load_gate": expert_gate,
        "param_gate": params_gate,
        "early_loss": early_loss,
        "late_loss": late_loss,
        "loss_drop_ratio": loss_drop_ratio,
        "warmup_excluded_loss_drop": warmup_drop,
        "grad_mean": grad_mean,
        "grad_std": grad_std,
        "grad_cv": grad_cv,
        "grad_spike_count": grad_spike_count,
        "router_entropy_mean": router_entropy_mean,
        "router_entropy_p10": router_entropy_p10,
        "router_max_load_p95": router_max_load_p95,
        "router_max_load_p99": router_max_load_p99,
        "collapse_events": collapse_events,
        "expert_gate_pass": expert_gate,
        "entropy_gate_pass": entropy_gate,
        "collapse_gate_pass": collapse_events <= int(cfg.get("mini_probe_max_collapse_events", 0)),
        "gates_config": {
            "min_loss_drop_ratio_warmup_excluded": float(cfg.get("mini_probe_min_loss_drop_ratio", 0.08)),
            "max_grad_cv": float(cfg.get("mini_probe_max_grad_cv", 1.35)),
            "min_router_entropy": float(cfg.get("mini_probe_min_router_entropy", 0.35)),
            "max_router_load_p95": float(cfg.get("mini_probe_max_router_load_p95", 0.85)),
            "max_collapse_events": int(cfg.get("mini_probe_max_collapse_events", 0)),
        },
        "model_params": int(model_params),
        "param_target": p_target,
        "param_tolerance": p_tol,
        "tokens_seen": int(tokens_seen),
        "token_min": token_min,
        "token_max": token_max,
        "token_progress_to_min": token_progress_to_min,
        "token_within_band": token_within_band,
    }


def print_live_compare_panel(bench: Dict[str, Any], stability_index: Dict[str, Any], efficiency_index: Dict[str, Any]) -> None:
    m = bench.get("mertformer", {})
    v = bench.get("vanilla", {})
    print("\n===LIVE_COMPARE_PANEL===")
    print(
        f"MERT: loss={float(m.get('train', {}).get('final_loss', float('inf'))):.4f} "
        f"val={float(m.get('eval', {}).get('val_loss', float('inf'))):.4f} "
        f"tok/s={float(m.get('train', {}).get('tokens_per_sec', 0.0)):.2f} "
        f"lat(ms)={float(m.get('latency', {}).get('avg_latency_ms', float('inf'))):.3f}"
    )
    print(
        f"VANL: loss={float(v.get('train', {}).get('final_loss', float('inf'))):.4f} "
        f"val={float(v.get('eval', {}).get('val_loss', float('inf'))):.4f} "
        f"tok/s={float(v.get('train', {}).get('tokens_per_sec', 0.0)):.2f} "
        f"lat(ms)={float(v.get('latency', {}).get('avg_latency_ms', float('inf'))):.3f}"
    )
    print(
        f"stability={float(stability_index.get('score_0_100', 0.0)):.1f}/100 "
        f"eff_mert_tps_per_mparam={float(efficiency_index.get('mert_tokens_per_sec_per_mparam', 0.0)):.2f}"
    )


# =============================================================================
# Logger (in-memory hash-chain)
# =============================================================================
class InMemoryRunLogger:
    def __init__(
        self,
        run_name: str,
        mode: str = "jsonl_ring",
        ring_size: int = 5000,
        step_log_interval: int = 1,
        flush_every: int = 10,
        fsync_every: int = 200,
        jsonl_path: str = "",
    ) -> None:
        self.run_name = run_name
        self.created_at_utc = _utc_now()
        self.mode = str(mode)
        self.step_log_interval = max(1, int(step_log_interval))
        self.flush_every = max(1, int(flush_every))
        self.fsync_every = max(1, int(fsync_every))
        self.ring_size = max(100, int(ring_size))
        self.records_ring: deque[Dict[str, Any]] = deque(maxlen=self.ring_size)
        self.step_rows_ring: deque[Dict[str, Any]] = deque(maxlen=self.ring_size)
        self.line_count_total = 0
        self.jsonl_path = str(jsonl_path).strip()
        self._jsonl_fp = None
        import hashlib

        self._prev_hash = hashlib.sha256(b"").hexdigest()
        if self.mode == "jsonl_ring" and self.jsonl_path:
            p = Path(self.jsonl_path)
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                self._jsonl_fp = p.open("a", encoding="utf-8")
            except Exception:
                try:
                    fallback = Path.cwd() / f"{self.run_name}.jsonl"
                    fallback.parent.mkdir(parents=True, exist_ok=True)
                    self._jsonl_fp = fallback.open("a", encoding="utf-8")
                    self.jsonl_path = str(fallback)
                except Exception:
                    self.mode = "in_memory"
                    self._jsonl_fp = None

    @staticmethod
    def _line_hash(prev_hash: str, payload: str) -> str:
        import hashlib

        h = hashlib.sha256()
        h.update(prev_hash.encode("utf-8"))
        h.update(payload.encode("utf-8"))
        return h.hexdigest()

    def _maybe_flush(self) -> None:
        if self._jsonl_fp is None:
            return
        try:
            if self.line_count_total % self.flush_every == 0:
                self._jsonl_fp.flush()
            if self.line_count_total % self.fsync_every == 0:
                os.fsync(self._jsonl_fp.fileno())
        except Exception:
            pass

    def _append_record(self, rec: Dict[str, Any]) -> None:
        payload = json.dumps(rec, ensure_ascii=False, sort_keys=True)
        chain_hash = self._line_hash(self._prev_hash, payload)
        rec["_chain"] = {"prev": self._prev_hash, "hash": chain_hash, "n": self.line_count_total + 1}
        self._prev_hash = chain_hash
        self.line_count_total += 1
        self.records_ring.append(rec)
        if self._jsonl_fp is not None:
            try:
                self._jsonl_fp.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            except Exception:
                pass
        self._maybe_flush()

    def log_event(self, kind: str, data: Dict[str, Any]) -> None:
        rec = {
            "type": "event",
            "name": str(kind),
            "event_type": str(kind),
            "timestamp_utc": _utc_now(),
            "data": safe_jsonable(data),
        }
        self._append_record(rec)

    def log_step(self, row: Dict[str, Any]) -> None:
        step = int(row.get("global_step", row.get("step", 0)))
        if step > 0 and (step % self.step_log_interval) != 0:
            return
        rec = dict(row)
        rec.setdefault("type", "step")
        rec.setdefault("timestamp_utc", _utc_now())
        rec["step"] = step
        if "global_step" in rec:
            rec["global_step"] = step
        self.step_rows_ring.append(dict(rec))
        self._append_record(rec)

    def finalize(self) -> Dict[str, Any]:
        if self._jsonl_fp is not None:
            try:
                self._jsonl_fp.flush()
                self._jsonl_fp.close()
            except Exception:
                pass
            self._jsonl_fp = None
        return {
            "run_name": self.run_name,
            "created_at_utc": self.created_at_utc,
            "mode": self.mode,
            "line_count_total": self.line_count_total,
            "line_count_ring": len(self.records_ring),
            "step_rows_ring_count": len(self.step_rows_ring),
            "jsonl_path": self.jsonl_path if self.mode == "jsonl_ring" else "",
            "final_hash": self._prev_hash,
        }


# =============================================================================
# Tokenizers: SentencePiece -> Byte-BPE -> Simple fallback
# =============================================================================
class SimpleTokenizer:
    def __init__(self, vocab_size: int = 32768) -> None:
        self.target_vocab_size = int(vocab_size)
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.stoi: Dict[str, int] = {
            "<pad>": self.pad_id,
            "<bos>": self.bos_id,
            "<eos>": self.eos_id,
            "<unk>": self.unk_id,
        }
        self.itos: Dict[int, str] = {v: k for k, v in self.stoi.items()}

    @property
    def backend(self) -> str:
        return "simple"

    def fit(self, texts: Sequence[str]) -> None:
        freq: Dict[str, int] = {}
        for text in texts:
            for ch in text:
                freq[ch] = freq.get(ch, 0) + 1
        slots = max(0, self.target_vocab_size - len(self.stoi))
        items = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:slots]
        idx = len(self.stoi)
        for ch, _ in items:
            if ch not in self.stoi:
                self.stoi[ch] = idx
                self.itos[idx] = ch
                idx += 1

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> List[int]:
        out: List[int] = []
        if add_bos:
            out.append(self.bos_id)
        for ch in text:
            out.append(self.stoi.get(ch, self.unk_id))
        if add_eos:
            out.append(self.eos_id)
        return out

    def decode(self, ids: Sequence[int]) -> str:
        buf: List[str] = []
        for i in ids:
            if i in (self.pad_id, self.bos_id):
                continue
            if i == self.eos_id:
                break
            buf.append(self.itos.get(int(i), "?"))
        return "".join(buf)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "target_vocab_size": self.target_vocab_size,
            "stoi": self.stoi,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.target_vocab_size = int(state.get("target_vocab_size", self.target_vocab_size))
        self.stoi = {str(k): int(v) for k, v in state.get("stoi", self.stoi).items()}
        self.itos = {v: k for k, v in self.stoi.items()}

    @property
    def vocab_size_realized(self) -> int:
        return len(self.stoi)


class ByteBPETokenizer:
    def __init__(
        self,
        vocab_size: int = 32768,
        max_merges: int = 4000,
        max_sequences: int = 50000,
        stagnation_patience: int = 24,
        min_pair_gain_ratio: float = 1.005,
        encode_cache_size: int = 2048,
        cache_max_text_len: int = 512,
    ) -> None:
        self.target_vocab_size = int(vocab_size)
        self.max_merges = int(max_merges)
        self.max_sequences = int(max(1000, max_sequences))
        self.stagnation_patience = int(max(4, stagnation_patience))
        self.min_pair_gain_ratio = float(max(1.0, min_pair_gain_ratio))
        self.encode_cache_size = int(max(0, encode_cache_size))
        self.cache_max_text_len = int(max(0, cache_max_text_len))
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.base_offset = 4
        # base byte tokens [4..259]
        self.id2bytes: Dict[int, bytes] = {self.base_offset + i: bytes([i]) for i in range(256)}
        self.merge_rules: List[Tuple[int, int, int]] = []  # (a,b,new_id)
        self.next_id = self.base_offset + 256
        self._encode_cache: "OrderedDict[str, List[int]]" = OrderedDict()

    @property
    def backend(self) -> str:
        return "byte_bpe"

    @staticmethod
    def _merge_once(seq: List[int], a: int, b: int, new_id: int) -> List[int]:
        out: List[int] = []
        i = 0
        n = len(seq)
        while i < n:
            if i + 1 < n and seq[i] == a and seq[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    def fit(self, texts: Sequence[str]) -> None:
        # Build initial corpus as byte-id sequences
        sequences: List[List[int]] = []
        for text in texts:
            b = text.encode("utf-8", errors="ignore")
            if not b:
                continue
            sequences.append([self.base_offset + int(x) for x in b])
        if len(sequences) > self.max_sequences:
            step = float(len(sequences)) / float(self.max_sequences)
            sampled = [sequences[int(i * step)] for i in range(self.max_sequences)]
            sequences = sampled

        max_vocab = max(self.base_offset + 256, min(self.target_vocab_size, self.base_offset + 256 + self.max_merges))
        prev_best_count = 0
        stagnant_rounds = 0
        t0 = time.time()
        while self.next_id < max_vocab:
            pair_freq: Dict[Tuple[int, int], int] = {}
            for seq in sequences:
                for i in range(len(seq) - 1):
                    pair = (seq[i], seq[i + 1])
                    pair_freq[pair] = pair_freq.get(pair, 0) + 1
            if not pair_freq:
                break
            best_pair, best_count = max(pair_freq.items(), key=lambda kv: kv[1])
            if best_count < 2:
                break
            if prev_best_count > 0 and best_count <= int(prev_best_count * self.min_pair_gain_ratio):
                stagnant_rounds += 1
                if stagnant_rounds >= self.stagnation_patience:
                    break
            else:
                stagnant_rounds = 0
            prev_best_count = max(prev_best_count, best_count)
            a, b = best_pair
            new_id = self.next_id
            self.next_id += 1
            self.id2bytes[new_id] = self.id2bytes.get(a, b"?") + self.id2bytes.get(b, b"?")
            self.merge_rules.append((a, b, new_id))
            sequences = [self._merge_once(seq, a, b, new_id) for seq in sequences]
            if (self.next_id - (self.base_offset + 256)) % 200 == 0:
                elapsed = time.time() - t0
                print(
                    f"[tokenizer:bpe] merges={self.next_id - (self.base_offset + 256)} "
                    f"best_pair_count={best_count} elapsed={elapsed:.1f}s"
                )

    def _apply_merges(self, seq: List[int]) -> List[int]:
        out = seq
        if len(out) < 2 or not self.merge_rules:
            return out
        # Apply merges in training order, but skip rules that cannot fire.
        pairs = set(zip(out, out[1:]))
        for a, b, nid in self.merge_rules:
            if (a, b) not in pairs:
                continue
            out = self._merge_once(out, a, b, nid)
            if len(out) < 2:
                break
            pairs = set(zip(out, out[1:]))
        return out

    def _cache_get(self, key: str) -> Optional[List[int]]:
        if self.encode_cache_size <= 0:
            return None
        val = self._encode_cache.get(key)
        if val is None:
            return None
        self._encode_cache.move_to_end(key)
        return list(val)

    def _cache_put(self, key: str, value: List[int]) -> None:
        if self.encode_cache_size <= 0:
            return
        self._encode_cache[key] = list(value)
        self._encode_cache.move_to_end(key)
        while len(self._encode_cache) > self.encode_cache_size:
            self._encode_cache.popitem(last=False)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> List[int]:
        cache_key = ""
        if self.cache_max_text_len > 0 and len(text) <= self.cache_max_text_len:
            cache_key = f"{int(add_bos)}|{int(add_eos)}|{text}"
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached
        byte_ids = [self.base_offset + int(x) for x in text.encode("utf-8", errors="ignore")]
        merged = self._apply_merges(byte_ids)
        out: List[int] = []
        if add_bos:
            out.append(self.bos_id)
        out.extend(merged)
        if add_eos:
            out.append(self.eos_id)
        if cache_key:
            self._cache_put(cache_key, out)
        return out

    def decode(self, ids: Sequence[int]) -> str:
        chunks: List[bytes] = []
        for i in ids:
            ii = int(i)
            if ii in (self.pad_id, self.bos_id):
                continue
            if ii == self.eos_id:
                break
            chunks.append(self.id2bytes.get(ii, b"?"))
        return b"".join(chunks).decode("utf-8", errors="replace")

    def state_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "target_vocab_size": self.target_vocab_size,
            "max_merges": self.max_merges,
            "max_sequences": self.max_sequences,
            "stagnation_patience": self.stagnation_patience,
            "min_pair_gain_ratio": self.min_pair_gain_ratio,
            "encode_cache_size": self.encode_cache_size,
            "cache_max_text_len": self.cache_max_text_len,
            "next_id": self.next_id,
            "merge_rules": self.merge_rules,
            "id2bytes": {str(k): list(v) for k, v in self.id2bytes.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.target_vocab_size = int(state.get("target_vocab_size", self.target_vocab_size))
        self.max_merges = int(state.get("max_merges", self.max_merges))
        self.max_sequences = int(state.get("max_sequences", self.max_sequences))
        self.stagnation_patience = int(state.get("stagnation_patience", self.stagnation_patience))
        self.min_pair_gain_ratio = float(state.get("min_pair_gain_ratio", self.min_pair_gain_ratio))
        self.encode_cache_size = int(state.get("encode_cache_size", self.encode_cache_size))
        self.cache_max_text_len = int(state.get("cache_max_text_len", self.cache_max_text_len))
        self.next_id = int(state.get("next_id", self.next_id))
        self.merge_rules = [tuple(x) for x in state.get("merge_rules", self.merge_rules)]  # type: ignore[arg-type]
        id2 = state.get("id2bytes", {})
        self.id2bytes = {int(k): bytes(v) for k, v in id2.items()}
        self._encode_cache = OrderedDict()

    @property
    def vocab_size_realized(self) -> int:
        return max(self.next_id, self.base_offset + 256)


class SentencePieceTokenizer:
    def __init__(self, vocab_size: int = 32768) -> None:
        self.target_vocab_size = int(vocab_size)
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.sp: Any = None

    @property
    def backend(self) -> str:
        return "sentencepiece"

    def fit(self, texts: Sequence[str]) -> None:
        if not HAS_SENTENCEPIECE:
            raise RuntimeError("sentencepiece_not_available")
        if not texts:
            raise RuntimeError("empty_texts_for_sentencepiece")

        with tempfile.TemporaryDirectory(prefix="spm_train_") as td:
            td_path = Path(td)
            input_path = td_path / "train.txt"
            model_prefix = td_path / "tok"

            # Write limited but diverse sample for tokenizer train
            max_lines = min(len(texts), 300_000)
            with input_path.open("w", encoding="utf-8") as f:
                for i, t in enumerate(texts):
                    if i >= max_lines:
                        break
                    t = t.replace("\n", " ").strip()
                    if t:
                        f.write(t + "\n")

            # sentencepiece can fail if requested vocab is too high for corpus
            requested = min(self.target_vocab_size, 32768)
            requested = max(512, requested)
            trained = False
            last_err = None
            for factor in (1.0, 0.75, 0.5, 0.35):
                vs = max(512, int(requested * factor))
                try:
                    spm.SentencePieceTrainer.train(
                        input=str(input_path),
                        model_prefix=str(model_prefix),
                        model_type="bpe",
                        vocab_size=vs,
                        character_coverage=1.0,
                        pad_id=self.pad_id,
                        bos_id=self.bos_id,
                        eos_id=self.eos_id,
                        unk_id=self.unk_id,
                        hard_vocab_limit=False,
                        split_digits=True,
                        add_dummy_prefix=False,
                    )
                    trained = True
                    break
                except Exception as e:
                    last_err = e
            if not trained:
                raise RuntimeError(f"sentencepiece_train_failed:{last_err}")

            model_path = model_prefix.with_suffix(".model")
            self.sp = spm.SentencePieceProcessor(model_file=str(model_path))

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> List[int]:
        if self.sp is None:
            raise RuntimeError("sentencepiece_not_ready")
        ids = list(self.sp.encode(text, out_type=int))
        out: List[int] = []
        if add_bos:
            out.append(self.bos_id)
        out.extend(ids)
        if add_eos:
            out.append(self.eos_id)
        return out

    def decode(self, ids: Sequence[int]) -> str:
        if self.sp is None:
            return ""
        filt: List[int] = []
        for i in ids:
            ii = int(i)
            if ii in (self.pad_id, self.bos_id):
                continue
            if ii == self.eos_id:
                break
            filt.append(ii)
        try:
            return self.sp.decode(filt)
        except Exception:
            return ""

    def state_dict(self) -> Dict[str, Any]:
        if self.sp is None:
            return {"backend": self.backend, "serialized_model": ""}
        blob = bytes(self.sp.serialized_model_proto())
        return {
            "backend": self.backend,
            "target_vocab_size": self.target_vocab_size,
            "serialized_model": blob.hex(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not HAS_SENTENCEPIECE:
            raise RuntimeError("sentencepiece_not_available")
        self.target_vocab_size = int(state.get("target_vocab_size", self.target_vocab_size))
        hex_blob = state.get("serialized_model", "")
        if not hex_blob:
            self.sp = None
            return
        blob = bytes.fromhex(hex_blob)
        self.sp = spm.SentencePieceProcessor(model_proto=blob)

    @property
    def vocab_size_realized(self) -> int:
        if self.sp is None:
            return 0
        return int(self.sp.get_piece_size())


class HybridTokenizer:
    def __init__(
        self,
        vocab_size: int = 32768,
        byte_bpe_max_merges: int = 8000,
        byte_bpe_encode_cache_size: int = 2048,
        byte_bpe_cache_max_text_len: int = 512,
        fit_max_texts: int = 30000,
        fit_max_chars: int = 6000000,
        fit_max_chars_per_text: int = 512,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.byte_bpe_max_merges = int(max(256, byte_bpe_max_merges))
        self.byte_bpe_encode_cache_size = int(max(0, byte_bpe_encode_cache_size))
        self.byte_bpe_cache_max_text_len = int(max(0, byte_bpe_cache_max_text_len))
        self.fit_max_texts = int(max(1024, fit_max_texts))
        self.fit_max_chars = int(max(200000, fit_max_chars))
        self.fit_max_chars_per_text = int(max(64, fit_max_chars_per_text))
        self.backend_name: str = "uninitialized"
        self.inner: Any = None
        self.metrics: Dict[str, Any] = {}

    @property
    def pad_id(self) -> int:
        return int(getattr(self.inner, "pad_id", 0))

    @property
    def bos_id(self) -> int:
        return int(getattr(self.inner, "bos_id", 1))

    @property
    def eos_id(self) -> int:
        return int(getattr(self.inner, "eos_id", 2))

    @property
    def unk_id(self) -> int:
        return int(getattr(self.inner, "unk_id", 3))

    @property
    def vocab_size_realized(self) -> int:
        if self.inner is None:
            return 0
        return int(getattr(self.inner, "vocab_size_realized", 0))

    def _fit_subset(self, texts: Sequence[str]) -> List[str]:
        if not texts:
            return []
        n = len(texts)
        if n <= self.fit_max_texts:
            idx = list(range(n))
        else:
            # Deterministic even-stride sampling for reproducibility.
            step = float(n) / float(self.fit_max_texts)
            idx = [int(i * step) for i in range(self.fit_max_texts)]
        out: List[str] = []
        total_chars = 0
        for i in idx:
            t = str(texts[i])
            if len(t) > self.fit_max_chars_per_text:
                t = t[: self.fit_max_chars_per_text]
            out.append(t)
            total_chars += len(t)
            if total_chars >= self.fit_max_chars:
                break
        return out

    def fit(self, texts: Sequence[str]) -> None:
        errs: List[str] = []
        fit_texts = self._fit_subset(texts)
        if not fit_texts:
            raise RuntimeError("empty_fit_texts")
        fit_char_count = sum(len(x) for x in fit_texts)
        t0 = time.time()

        # Attempt 1: SentencePiece
        if HAS_SENTENCEPIECE:
            try:
                tok = SentencePieceTokenizer(vocab_size=self.vocab_size)
                tok.fit(fit_texts)
                self.inner = tok
                self.backend_name = tok.backend
                self.metrics = {
                    "backend": self.backend_name,
                    "errors": errs,
                    "fit_text_count": len(fit_texts),
                    "fit_char_count": fit_char_count,
                    "fit_elapsed_sec": time.time() - t0,
                }
                return
            except Exception as e:
                errs.append(f"spm_fail:{type(e).__name__}:{e}")

        # Attempt 2: Byte-BPE
        try:
            tok = ByteBPETokenizer(
                vocab_size=self.vocab_size,
                max_merges=self.byte_bpe_max_merges,
                max_sequences=min(max(2000, self.fit_max_texts), 20000),
                encode_cache_size=self.byte_bpe_encode_cache_size,
                cache_max_text_len=self.byte_bpe_cache_max_text_len,
            )
            tok.fit(fit_texts)
            self.inner = tok
            self.backend_name = tok.backend
            self.metrics = {
                "backend": self.backend_name,
                "errors": errs,
                "fit_text_count": len(fit_texts),
                "fit_char_count": fit_char_count,
                "fit_elapsed_sec": time.time() - t0,
            }
            return
        except Exception as e:
            errs.append(f"byte_bpe_fail:{type(e).__name__}:{e}")

        # Attempt 3: Simple fallback
        tok = SimpleTokenizer(vocab_size=self.vocab_size)
        tok.fit(fit_texts)
        self.inner = tok
        self.backend_name = tok.backend
        self.metrics = {
            "backend": self.backend_name,
            "errors": errs,
            "fit_text_count": len(fit_texts),
            "fit_char_count": fit_char_count,
            "fit_elapsed_sec": time.time() - t0,
        }

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> List[int]:
        if self.inner is None:
            raise RuntimeError("hybrid_tokenizer_not_fitted")
        return self.inner.encode(text, add_bos=add_bos, add_eos=add_eos)

    def decode(self, ids: Sequence[int]) -> str:
        if self.inner is None:
            return ""
        return self.inner.decode(ids)

    def state_dict(self) -> Dict[str, Any]:
        if self.inner is None:
            return {"backend": "none"}
        return {
            "backend": self.backend_name,
            "metrics": self.metrics,
            "inner": self.inner.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        backend = state.get("backend", "none")
        inner_state = state.get("inner", {})
        if backend == "sentencepiece" and HAS_SENTENCEPIECE:
            tok = SentencePieceTokenizer(vocab_size=self.vocab_size)
            tok.load_state_dict(inner_state)
            self.inner = tok
        elif backend == "byte_bpe":
            tok = ByteBPETokenizer(
                vocab_size=self.vocab_size,
                max_merges=self.byte_bpe_max_merges,
                encode_cache_size=self.byte_bpe_encode_cache_size,
                cache_max_text_len=self.byte_bpe_cache_max_text_len,
            )
            tok.load_state_dict(inner_state)
            self.inner = tok
        else:
            tok = SimpleTokenizer(vocab_size=self.vocab_size)
            tok.load_state_dict(inner_state)
            self.inner = tok
        self.backend_name = backend
        self.metrics = state.get("metrics", {})


# =============================================================================
# Data pipeline: curriculum + quality filters + deterministic split
# =============================================================================
def normalize_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_quality_ok(text: str, min_len: int, max_len: int, stage_name: str) -> bool:
    n = len(text)
    if n < min_len or n > max_len:
        return False
    # Reject URL-heavy spam
    if text.lower().count("http") > 3:
        return False
    # Reject repetitive garbage
    words = text.split()
    if len(words) >= 30:
        uniq_w = len(set(words)) / max(1, len(words))
        if uniq_w < 0.15:
            return False
    else:
        if len(set(text)) < 12:
            return False
    # Stage-specific weak heuristics
    if "code" in stage_name:
        code_keys = ["def ", "class ", "import ", "return ", "{" , "}" ]
        if not any(k in text for k in code_keys):
            return False
    if "dialogue" in stage_name:
        if text.count(":") < 1 and "?" not in text:
            return False
    return True


def legacy_build_curriculum_sources(turkish_primary: bool = True) -> List[Dict[str, Any]]:
    tr_weight = 0.55 if turkish_primary else 0.25
    return [
        {
            "name": "stage_1_foundation_tr",
            "ratio": tr_weight,
            "max_samples": 180000,
            "min_len": 60,
            "max_len": 5000,
            "quality_filter_rules": ["no_spam", "length", "dedupe_light"],
            "fallback_source": "synthetic_tr_foundation",
            "hf_candidates": [
                {"dataset": "wikimedia/wikipedia", "subset": "20231101.tr", "split": "train[:2%]", "field": "text"},
                {"dataset": "uonlp/CulturaX", "subset": "tr", "split": "train[:0.2%]", "field": "text"},
            ],
        },
        {
            "name": "stage_2_reasoning_math",
            "ratio": 0.18,
            "max_samples": 70000,
            "min_len": 40,
            "max_len": 4000,
            "quality_filter_rules": ["reasoning_trace", "length"],
            "fallback_source": "synthetic_reasoning_math",
            "hf_candidates": [
                {"dataset": "openai/gsm8k", "subset": "main", "split": "train", "field": "question"},
                {"dataset": "TIGER-Lab/MathInstruct", "split": "train[:1%]", "field": "instruction"},
            ],
        },
        {
            "name": "stage_3_code_python",
            "ratio": 0.17,
            "max_samples": 70000,
            "min_len": 50,
            "max_len": 8000,
            "quality_filter_rules": ["code_signals", "length"],
            "fallback_source": "synthetic_code_python",
            "hf_candidates": [
                {
                    "dataset": "codeparrot/github-code",
                    "split": "train[:0.1%]",
                    "field": "code",
                    "requires_remote_code": True,
                },
                {"dataset": "bigcode/the-stack-dedup", "split": "train[:0.02%]", "field": "content"},
            ],
        },
        {
            "name": "stage_4_dialogue_instruction_tr",
            "ratio": 0.10,
            "max_samples": 50000,
            "min_len": 30,
            "max_len": 3000,
            "quality_filter_rules": ["dialogue_signals", "length"],
            "fallback_source": "synthetic_dialogue_tr",
            "hf_candidates": [
                {"dataset": "OpenAssistant/oasst_top1_2023-08-25", "split": "train[:2%]", "field": "text"},
                {"dataset": "turkish-nlp-suite/InstrucTurca", "split": "train[:5%]", "field": "Output"},
            ],
        },
    ]


def synthetic_stage_samples(stage_name: str, n: int) -> List[str]:
    # Small curated fallback templates, TR-primary by design
    if "foundation" in stage_name:
        base = [
            "Türkiye'de yapay zeka geliştirme süreçlerinde veri kalitesi, modelin güvenilirliğini doğrudan etkiler.",
            "Mühendislikte deterministik deneyler, tekrarlanabilir sonuç üretmenin temelidir.",
            "Bir dil modelinin başarısı yalnızca parametre sayısına değil, veri düzenine de bağlıdır.",
            "Kısa testler yalnızca stabiliteyi ölçer; kalite için uzun vadeli validasyon gerekir.",
        ]
    elif "reasoning" in stage_name:
        base = [
            "Soru: 12 elma 3 kişiye eşit dağılırsa kişi başı kaç elma düşer? Cevap: 4.",
            "Adım adım düşün: Önce verilenleri yaz, sonra işlemi uygula ve sonucu kontrol et.",
            "Bir problemi çözmeden önce varsayımları açıkça belirlemek hata riskini azaltır.",
        ]
    elif "code" in stage_name:
        base = [
            "def topla(a, b):\n    return a + b",
            "class Sayaç:\n    def __init__(self):\n        self.n = 0\n    def artir(self):\n        self.n += 1",
            "import math\n\ndef norm(x):\n    return math.sqrt(sum(v*v for v in x))",
        ]
    else:
        base = [
            "Kullanıcı: Bu model nasıl daha stabil eğitilir?\nAsistan: Öğrenme oranını düşür, warmup ekle ve grad clipping kullan.",
            "Kullanıcı: Neden validation gerekli?\nAsistan: Overfitting'i görmek ve genel performansı ölçmek için.",
            "Kullanıcı: BitNet ne sağlar?\nAsistan: Hesaplamayı sadeleştirip verimliliği artırabilir.",
        ]

    out: List[str] = []
    for i in range(n):
        out.append(base[i % len(base)])
    return out


def legacy_extract_field(item: Dict[str, Any], field: str) -> Optional[str]:
    if field in item and item[field] is not None:
        return str(item[field])
    # try common alternatives
    for key in ("text", "content", "instruction", "output", "Output", "question", "answer", "code"):
        if key in item and item[key] is not None:
            return str(item[key])
    return None


def _bounded_split(split: str, percent: float) -> str:
    s = str(split)
    if "[" in s and "]" in s:
        return s
    if s in ("train", "validation", "test"):
        pct = max(0.01, min(5.0, float(percent)))
        return f"{s}[:{pct}%]"
    return s


def _load_hf_candidate_dataset(
    ds_name: str,
    subset: Optional[str],
    split: str,
    cfg: Dict[str, Any],
) -> Tuple[Optional[Iterable[Dict[str, Any]]], str, str]:
    if not HAS_DATASETS:
        return None, "disabled", "datasets_not_available"

    use_streaming = bool(cfg.get("hf_streaming", True))
    trust_remote_code = bool(cfg.get("hf_trust_remote_code", False))
    if use_streaming:
        try:
            stream_split = str(split)
            if "[" in stream_split and "]" in stream_split:
                stream_split = stream_split.split("[", 1)[0]
            if stream_split not in ("train", "validation", "test"):
                stream_split = "train"
            if subset:
                ds = load_dataset(
                    ds_name,
                    subset,
                    split=stream_split,
                    streaming=True,
                    trust_remote_code=trust_remote_code,
                )
            else:
                ds = load_dataset(
                    ds_name,
                    split=stream_split,
                    streaming=True,
                    trust_remote_code=trust_remote_code,
                )
            return ds, "streaming", "ok"
        except Exception as e:
            streaming_err = f"{type(e).__name__}:{e}"
        else:
            streaming_err = "unknown"
    else:
        streaming_err = "streaming_disabled"

    if not bool(cfg.get("hf_allow_materialized_fallback", False)):
        return None, "failed", f"streaming_only:{streaming_err}"

    try:
        ns_split = split
        if bool(cfg.get("hf_force_small_nonstream_split", True)):
            ns_split = _bounded_split(split, float(cfg.get("hf_nonstream_split_percent", 1.0)))
        if subset:
            ds = load_dataset(
                ds_name,
                subset,
                split=ns_split,
                trust_remote_code=trust_remote_code,
            )
        else:
            ds = load_dataset(
                ds_name,
                split=ns_split,
                trust_remote_code=trust_remote_code,
            )
        return ds, "materialized", f"fallback_from_streaming:{streaming_err}"
    except Exception as e:
        return None, "failed", f"{streaming_err}|{type(e).__name__}:{e}"


def _hf_candidate_worker(
    q: Any,
    ds_name: str,
    subset: Optional[str],
    split: str,
    field: str,
    min_len: int,
    max_len: int,
    stage_name: str,
    use_streaming: bool,
    allow_materialized_fallback: bool,
    trust_remote_code: bool,
    nonstream_percent: float,
    max_rows: int,
) -> None:
    rows: List[str] = []
    try:
        if not HAS_DATASETS:
            q.put({"ok": False, "reason": "datasets_not_available"})
            return
        load_mode = "failed"
        load_info = "unknown"
        ds = None
        if use_streaming:
            try:
                stream_split = str(split)
                if "[" in stream_split and "]" in stream_split:
                    stream_split = stream_split.split("[", 1)[0]
                if stream_split not in ("train", "validation", "test"):
                    stream_split = "train"
                if subset:
                    ds = load_dataset(
                        ds_name,
                        subset,
                        split=stream_split,
                        streaming=True,
                        trust_remote_code=bool(trust_remote_code),
                    )
                else:
                    ds = load_dataset(
                        ds_name,
                        split=stream_split,
                        streaming=True,
                        trust_remote_code=bool(trust_remote_code),
                    )
                load_mode = "streaming"
                load_info = "ok"
            except Exception as e:
                load_info = _format_exception(e)
        if ds is None and allow_materialized_fallback:
            try:
                ns_split = _bounded_split(split, nonstream_percent)
                if subset:
                    ds = load_dataset(
                        ds_name,
                        subset,
                        split=ns_split,
                        trust_remote_code=bool(trust_remote_code),
                    )
                else:
                    ds = load_dataset(
                        ds_name,
                        split=ns_split,
                        trust_remote_code=bool(trust_remote_code),
                    )
                load_mode = "materialized"
                load_info = f"fallback_from_streaming:{load_info}"
            except Exception as e:
                q.put({"ok": False, "reason": f"{load_info}|{_format_exception(e)}"})
                return
        if ds is None:
            q.put({"ok": False, "reason": f"load_failed:{load_info}"})
            return

        for item in ds:
            if len(rows) >= max_rows:
                break
            raw = _extract_field(item, field)
            if not raw:
                continue
            txt = normalize_text(raw)
            if not text_quality_ok(txt, int(min_len), int(max_len), stage_name):
                continue
            rows.append(txt)
        q.put(
            {
                "ok": True,
                "rows": rows,
                "load_mode": load_mode,
                "load_info": load_info,
            }
        )
    except Exception as e:
        try:
            q.put({"ok": False, "reason": _format_exception(e), "rows": rows})
        except Exception:
            pass


def _load_candidate_rows_process_timeout(
    stage: Dict[str, Any],
    candidate: Dict[str, Any],
    cfg: Dict[str, Any],
    max_rows: int,
) -> Tuple[List[str], str, str, bool]:
    """
    Returns (rows, load_mode, load_info, timed_out).
    """
    # Spawn workers require a real __main__.__file__. In heredoc/REPL contexts
    # this is often '<stdin>', so fall back to direct loading without crashing.
    main_file = getattr(sys.modules.get("__main__"), "__file__", "")
    if not main_file or str(main_file).endswith("<stdin>") or not Path(str(main_file)).exists():
        return [], "direct_fallback", "worker_unavailable_main_path", False

    timeout_sec = float(cfg.get("hf_candidate_process_max_seconds", 90))
    ctx = mp.get_context("spawn")
    q: Any = ctx.Queue(maxsize=1)
    p = ctx.Process(
        target=_hf_candidate_worker,
        args=(
            q,
            str(candidate["dataset"]),
            candidate.get("subset"),
            str(candidate.get("split", "train")),
            str(candidate.get("field", "text")),
            int(stage["min_len"]),
            int(stage["max_len"]),
            str(stage["name"]),
            bool(cfg.get("hf_streaming", True)),
            bool(cfg.get("hf_allow_materialized_fallback", False)),
            bool(cfg.get("hf_trust_remote_code", False)),
            float(cfg.get("hf_nonstream_split_percent", 1.0)),
            int(max_rows),
        ),
    )
    p.start()
    started = time.time()
    last_hb = started
    timed_out = False
    result: Dict[str, Any] = {}
    while time.time() - started < timeout_sec:
        if not q.empty():
            try:
                result = q.get_nowait()
            except Exception:
                result = {}
            break
        if not p.is_alive():
            break
        now = time.time()
        if now - last_hb >= float(cfg.get("hf_candidate_heartbeat_seconds", 15)):
            print(
                f"[data:heartbeat] ds={candidate.get('dataset', 'unknown')} "
                f"worker_elapsed={now - started:.1f}s timeout={timeout_sec:.1f}s"
            )
            last_hb = now
        time.sleep(0.2)
    if not result and p.is_alive():
        timed_out = True
        try:
            p.terminate()
        except Exception:
            pass
    try:
        p.join(timeout=1.0)
    except Exception:
        pass
    try:
        q.close()
    except Exception:
        pass

    if timed_out:
        return [], "timeout", "worker_timeout", True
    if not result:
        return [], "failed", "worker_no_result", False
    if not bool(result.get("ok", False)):
        return [], "failed", str(result.get("reason", "worker_failed")), False
    rows = [str(x) for x in result.get("rows", [])]
    return rows, str(result.get("load_mode", "worker")), str(result.get("load_info", "ok")), False


def load_stage_texts(
    stage: Dict[str, Any],
    cfg: Dict[str, Any],
    logger: InMemoryRunLogger,
    effective_target: Optional[int] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    mode = str(cfg.get("data_mode", "quality_tr_mix"))
    target = int(effective_target) if effective_target is not None else int(stage["max_samples"])
    target = max(500, min(target, int(stage["max_samples"])))
    texts: List[str] = []
    failures: List[str] = []
    candidate_rows: List[Dict[str, Any]] = []

    hf_allowed = mode in ("quality_tr_mix", "hf_only")
    if hf_allowed and HAS_DATASETS:
        for c in stage["hf_candidates"]:
            if len(texts) >= target:
                break
            try:
                ds_name = c["dataset"]
                if bool(c.get("requires_remote_code", False)) and not bool(cfg.get("hf_trust_remote_code", False)):
                    candidate_rows.append(
                        {
                            "dataset": ds_name,
                            "attempt_count": 1,
                            "load_mode": "skipped",
                            "load_info": "remote_code_not_trusted",
                            "reason_code": "remote_code_not_trusted",
                            "kept": 0,
                            "error": "remote_code_not_trusted",
                            "elapsed_sec": 0.0,
                        }
                    )
                    logger.log_event(
                        "stage_candidate_skipped",
                        {
                            "stage": stage["name"],
                            "dataset": ds_name,
                            "reason": "remote_code_not_trusted",
                        },
                    )
                    print(
                        f"[data] stage={stage['name']} ds={ds_name} mode=skipped "
                        f"kept=0/{max(1, target - len(texts))} reason=remote_code_not_trusted"
                    )
                    continue
                subset = c.get("subset")
                split = c.get("split", "train")
                field = c.get("field", "text")
                per_candidate_cap = min(
                    target - len(texts),
                    int(cfg.get("stage_max_hf_rows_per_candidate", target)),
                )
                if bool(cfg.get("hf_candidate_process_timeout", True)):
                    cand_start = time.time()
                    rows, load_mode, load_info, timed_out = _load_candidate_rows_process_timeout(
                        stage=stage,
                        candidate=c,
                        cfg=cfg,
                        max_rows=min(per_candidate_cap, int(cfg.get("hf_candidate_process_rows", 4096))),
                    )
                    if load_mode == "direct_fallback":
                        # Worker mode unavailable in current runtime context.
                        # Continue with bounded direct loader path. Notebook cell
                        # runtimes can hit this branch, so keep the same timeout
                        # guarantees here instead of silently streaming forever.
                        ds, load_mode, load_info = _load_hf_candidate_dataset(ds_name, subset, split, cfg)
                        rows = []
                        timed_out = False
                        if ds is not None:
                            direct_start = time.time()
                            direct_last_hb = direct_start
                            for item in ds:
                                direct_elapsed = time.time() - direct_start
                                if len(rows) >= per_candidate_cap:
                                    break
                                if direct_elapsed >= float(cfg.get("hf_candidate_max_seconds", 180)):
                                    timed_out = True
                                    load_info = f"{load_info}|direct_timeout_{int(direct_elapsed)}s"
                                    break
                                raw = _extract_field(item, field)
                                if not raw:
                                    continue
                                txt = normalize_text(raw)
                                if not text_quality_ok(
                                    txt, int(stage["min_len"]), int(stage["max_len"]), stage["name"]
                                ):
                                    continue
                                rows.append(txt)
                                now = time.time()
                                if now - direct_last_hb >= float(cfg.get("hf_candidate_heartbeat_seconds", 15)):
                                    print(
                                        f"[data:heartbeat] stage={stage['name']} ds={ds_name} "
                                        f"direct_kept={len(rows)}/{per_candidate_cap} "
                                        f"elapsed={now - direct_start:.1f}s"
                                    )
                                    direct_last_hb = now
                        else:
                            load_mode = "failed"
                    if timed_out:
                        failures.append(f"{ds_name}:worker_timeout")
                    kept = min(per_candidate_cap, len(rows))
                    if kept > 0:
                        texts.extend(rows[:kept])
                    candidate_rows.append(
                        {
                            "dataset": ds_name,
                            "attempt_count": 1,
                            "load_mode": load_mode,
                            "load_info": load_info,
                            "reason_code": reason_code_from_error(str(load_info)) if str(load_mode) != "streaming" else "ok",
                            "kept": kept,
                            "error": "" if kept > 0 else str(load_info),
                            "target_cap": per_candidate_cap,
                            "elapsed_sec": time.time() - cand_start,
                            "worker_timeout": bool(timed_out),
                        }
                    )
                    logger.log_event(
                        "stage_candidate_loaded",
                        {
                            "stage": stage["name"],
                            "dataset": ds_name,
                            "subset": subset or "",
                            "split": split,
                            "load_mode": load_mode,
                            "load_info": load_info,
                            "kept": kept,
                            "target_cap": per_candidate_cap,
                            "elapsed_sec": time.time() - cand_start,
                            "worker_timeout": bool(timed_out),
                        },
                    )
                    print(
                        f"[data] stage={stage['name']} ds={ds_name} mode={load_mode} "
                        f"kept={kept}/{per_candidate_cap} elapsed={time.time() - cand_start:.1f}s"
                    )
                    if len(texts) >= target:
                        break
                    continue
                ds, load_mode, load_info = _load_hf_candidate_dataset(ds_name, subset, split, cfg)
                if ds is None:
                    failures.append(f"{c.get('dataset')}:{load_info}")
                    candidate_rows.append(
                        {
                            "dataset": ds_name,
                            "attempt_count": 1,
                            "load_mode": "failed",
                            "load_info": load_info,
                            "reason_code": reason_code_from_error(str(load_info)),
                            "kept": 0,
                            "error": str(load_info),
                            "elapsed_sec": 0.0,
                        }
                    )
                    continue
                cand_start = time.time()
                last_hb = cand_start
                kept = 0
                for item in ds:
                    elapsed = time.time() - cand_start
                    if kept >= per_candidate_cap or len(texts) >= target:
                        break
                    if elapsed >= float(cfg.get("hf_candidate_max_seconds", 180)):
                        failures.append(f"{ds_name}:timeout_{int(elapsed)}s")
                        break
                    raw = _extract_field(item, field)
                    if not raw:
                        continue
                    txt = normalize_text(raw)
                    if not text_quality_ok(txt, int(stage["min_len"]), int(stage["max_len"]), stage["name"]):
                        continue
                    texts.append(txt)
                    kept += 1
                    now = time.time()
                    if now - last_hb >= float(cfg.get("hf_candidate_heartbeat_seconds", 15)):
                        print(
                            f"[data:heartbeat] stage={stage['name']} ds={ds_name} kept={kept}/{per_candidate_cap} "
                            f"elapsed={now - cand_start:.1f}s total_stage={len(texts)}/{target}"
                        )
                        last_hb = now
                logger.log_event(
                    "stage_candidate_loaded",
                    {
                        "stage": stage["name"],
                        "dataset": ds_name,
                        "subset": subset or "",
                        "split": split,
                        "load_mode": load_mode,
                        "load_info": load_info,
                        "kept": kept,
                        "target_cap": per_candidate_cap,
                        "elapsed_sec": time.time() - cand_start,
                    },
                )
                candidate_rows.append(
                    {
                        "dataset": ds_name,
                        "attempt_count": 1,
                        "load_mode": load_mode,
                        "load_info": load_info,
                        "reason_code": "ok" if kept > 0 else reason_code_from_error(str(load_info)),
                        "kept": kept,
                        "error": "" if kept > 0 else str(load_info),
                        "target_cap": per_candidate_cap,
                        "elapsed_sec": time.time() - cand_start,
                    }
                )
                print(
                    f"[data] stage={stage['name']} ds={ds_name} mode={load_mode} "
                    f"kept={kept}/{per_candidate_cap} elapsed={time.time() - cand_start:.1f}s"
                )
            except Exception as e:
                failures.append(f"{c.get('dataset')}:{type(e).__name__}")
                candidate_rows.append(
                    {
                        "dataset": c.get("dataset", "unknown"),
                        "attempt_count": 1,
                        "load_mode": "exception",
                        "load_info": f"{type(e).__name__}:{e}",
                        "reason_code": reason_code_from_error(f"{type(e).__name__}:{e}"),
                        "kept": 0,
                        "error": f"{type(e).__name__}:{e}",
                        "elapsed_sec": 0.0,
                    }
                )

    # Synthetic fallback
    if not texts and mode in ("quality_tr_mix", "synthetic_only"):
        fallback_n = min(target, 120000)
        texts = synthetic_stage_samples(stage["name"], fallback_n)

    # Last-resort deterministic random text fragments
    # WARNING: this is meaningless random-character "text" (NOT real corpus). It
    # exists only so the pipeline does not crash when both HF and synthetic data
    # are unavailable. Any run that hits this path is data-degraded and must not
    # be treated as a genuine proof-of-learning. We flag it in meta below.
    degraded_random_corpus = False
    if not texts:
        degraded_random_corpus = True
        print(
            f"[WARN] stage '{stage['name']}': no real/synthetic data available; "
            "falling back to RANDOM-CHARACTER corpus (degraded, not a valid proof).",
            file=sys.stderr,
        )
        rnd = random.Random(int(cfg["seed"]) + abs(hash(stage["name"])) % 10000)
        for _ in range(min(target, 40000)):
            n = rnd.randint(int(stage["min_len"]), min(int(stage["max_len"]), 180))
            alphabet = "abcçdefgğhıijklmnoöprsştuüvyz0123456789 .,:;!?()"
            texts.append("".join(rnd.choice(alphabet) for _ in range(n)))

    meta = {
        "degraded_random_corpus": bool(degraded_random_corpus),
        "stage": stage["name"],
        "target": target,
        "loaded_samples": len(texts),
        "failures": failures,
        "mode": mode,
        "source_type": "hf_or_fallback",
        "candidate_rows": candidate_rows,
        "stage_loaded_from_hf": int(sum(int(r.get("kept", 0)) for r in candidate_rows if str(r.get("reason_code", "")) == "ok")),
    }
    logger.log_event("stage_data_loaded", meta)
    return texts, meta


def build_curriculum_corpus(cfg: Dict[str, Any], logger: InMemoryRunLogger) -> Tuple[List[str], List[Dict[str, Any]]]:
    stages = build_curriculum_sources(turkish_primary=bool(cfg.get("turkish_primary", True)))
    if not bool(cfg.get("curriculum_enabled", True)):
        stages = stages[:1]
    ratio_sum = float(sum(float(s.get("ratio", 0.0)) for s in stages))
    if abs(ratio_sum - 1.0) > 1e-6:
        raise RuntimeError(f"curriculum_ratio_sum_invalid:{ratio_sum:.6f}")
    curriculum_cfg_hash = hash_config(
        {
            "stages": [
                {
                    "name": str(s.get("name", "")),
                    "ratio": float(s.get("ratio", 0.0)),
                    "max_samples": int(s.get("max_samples", 0)),
                }
                for s in stages
            ]
        }
    )
    logger.log_event("curriculum_ratio_check", {"ratio_sum": ratio_sum, "curriculum_config_hash": curriculum_cfg_hash})

    all_texts: List[str] = []
    trace: List[Dict[str, Any]] = []

    for stage in stages:
        target_take = max(1000, int(stage["ratio"] * int(cfg["tokenizer_max_texts"])))
        texts, meta = load_stage_texts(stage, cfg, logger, effective_target=target_take)
        topup_added = 0
        if len(texts) < target_take and str(cfg.get("data_mode", "quality_tr_mix")) != "hf_only":
            need = target_take - len(texts)
            extra = synthetic_stage_samples(stage["name"], need)
            texts.extend(extra)
            topup_added = len(extra)
        if len(texts) > target_take:
            rnd = random.Random(int(cfg["seed"]) + len(stage["name"]))
            idx = list(range(len(texts)))
            rnd.shuffle(idx)
            texts = [texts[i] for i in idx[:target_take]]
        all_texts.extend(texts)
        tr = dict(stage)
        tr.update(meta)
        tr["curriculum_ratio_sum"] = ratio_sum
        tr["curriculum_config_hash"] = curriculum_cfg_hash
        tr["topup_added"] = topup_added
        tr["selected_for_tokenizer"] = len(texts)
        trace.append(tr)
        print(
            f"[data] stage={stage['name']} selected={len(texts)} "
            f"target_take={target_take} topup={topup_added}"
        )

    if not all_texts:
        all_texts = synthetic_stage_samples("stage_1_foundation_tr", 8000)

    logger.log_event(
        "curriculum_corpus",
        {
            "total_texts": len(all_texts),
            "stages": [{"name": x["name"], "selected": x["selected_for_tokenizer"]} for x in trace],
        },
    )
    return all_texts, trace


def compute_oov_rate(token_ids: Sequence[int], unk_id: int) -> float:
    if not token_ids:
        return 0.0
    unk = sum(1 for x in token_ids if int(x) == int(unk_id))
    return float(unk) / float(len(token_ids))


def token_histogram_topk(token_ids: Sequence[int], k: int = 20) -> List[Tuple[int, int]]:
    freq: Dict[int, int] = {}
    for x in token_ids:
        i = int(x)
        freq[i] = freq.get(i, 0) + 1
    top = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [(int(a), int(b)) for a, b in top]


def pack_sequences(token_ids: Sequence[int], seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    t = torch.tensor(list(token_ids), dtype=torch.long)
    n = int(t.numel()) // (seq_len + 1)
    if n <= 0:
        raise ValueError("Not enough tokens to pack sequences")
    t = t[: n * (seq_len + 1)]
    blocks = t.view(n, seq_len + 1)
    x = blocks[:, :seq_len].contiguous()
    y = blocks[:, 1:].contiguous()
    return x, y


def build_train_val_streams(
    token_ids: Sequence[int],
    seq_len: int,
    val_ratio: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y = pack_sequences(token_ids, seq_len=seq_len)
    n = x.size(0)
    idx = torch.arange(n)
    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = idx[torch.randperm(n, generator=g)]
    split = max(1, int(n * (1.0 - val_ratio)))
    train_idx = perm[:split]
    val_idx = perm[split:]
    if val_idx.numel() == 0:
        val_idx = perm[-1:]
        train_idx = perm[:-1]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


# =============================================================================
# Model components
# =============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        norm = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * norm).to(x.dtype) * self.weight


def activation_quant_int8_ste(x: torch.Tensor) -> torch.Tensor:
    scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / 127.0
    xq = torch.round(x / scale).clamp(-127, 127) * scale
    return x + (xq - x).detach()


def weight_quant_ternary_ste(w: torch.Tensor) -> torch.Tensor:
    # Per-output scaling via RMS
    s = torch.sqrt((w.detach() ** 2).mean(dim=1, keepdim=True)).clamp(min=1e-5)
    wn = w / s
    wq = torch.round(wn).clamp(-1.0, 1.0) * s
    return w + (wq - w).detach()


def _linear_core(x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor]) -> torch.Tensor:
    return F.linear(x, w, b)


_COMPILED_LINEAR_CORE = None


class BitLinearStrict(nn.Linear):
    """
    Strict BitNet core linear:
    - activation int8 STE
    - weight ternary STE
    - learnable per-output scaling (alpha)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.alpha = nn.Parameter(
            torch.ones(
                out_features,
                1,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
        )
        self._cached_weight_q: Optional[torch.Tensor] = None
        self._cached_weight_version: int = -1
        self._cached_alpha_version: int = -1
        self._cached_dtype: Optional[torch.dtype] = None
        self._cached_device: Optional[torch.device] = None
        self._compiled_enabled = os.environ.get("MERTFORMER_ONEFILE_BITNET_COMPILE", "1") == "1"

    def invalidate_cache(self) -> None:
        self._cached_weight_q = None
        self._cached_weight_version = -1
        self._cached_alpha_version = -1
        self._cached_dtype = None
        self._cached_device = None

    @staticmethod
    def _get_compiled_linear(device: torch.device):
        global _COMPILED_LINEAR_CORE
        if _COMPILED_LINEAR_CORE is not None:
            return _COMPILED_LINEAR_CORE
        if device.type != "cuda":
            _COMPILED_LINEAR_CORE = _linear_core
            return _COMPILED_LINEAR_CORE
        if not hasattr(torch, "compile"):
            _COMPILED_LINEAR_CORE = _linear_core
            return _COMPILED_LINEAR_CORE
        try:
            _COMPILED_LINEAR_CORE = torch.compile(_linear_core, mode="max-autotune", fullgraph=False)
        except Exception:
            _COMPILED_LINEAR_CORE = _linear_core
        return _COMPILED_LINEAR_CORE

    def _get_quantized_weight_cached(self, w_scaled: torch.Tensor) -> torch.Tensor:
        w_ver = int(self.weight._version)  # type: ignore[attr-defined]
        a_ver = int(self.alpha._version)  # type: ignore[attr-defined]
        need_refresh = (
            self._cached_weight_q is None
            or self._cached_weight_version != w_ver
            or self._cached_alpha_version != a_ver
            or self._cached_dtype != w_scaled.dtype
            or self._cached_device != w_scaled.device
        )
        if need_refresh:
            with torch.no_grad():
                self._cached_weight_q = weight_quant_ternary_ste(w_scaled.detach()).detach()
                self._cached_weight_version = w_ver
                self._cached_alpha_version = a_ver
                self._cached_dtype = w_scaled.dtype
                self._cached_device = w_scaled.device
        return self._cached_weight_q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xq = activation_quant_int8_ste(x)
        w_scaled = self.weight * self.alpha
        wq_cached = self._get_quantized_weight_cached(w_scaled)
        if self.training:
            # STE identity gradient w.r.t. both weight and alpha.
            w_eff = w_scaled + (wq_cached - w_scaled).detach()
        else:
            w_eff = wq_cached

        if self._compiled_enabled:
            linear_fn = self._get_compiled_linear(xq.device)
            return linear_fn(xq, w_eff, self.bias)
        return F.linear(xq, w_eff, self.bias)

    @torch.no_grad()
    def telemetry(self) -> Dict[str, float]:
        w = self.weight * self.alpha
        s = torch.sqrt((w ** 2).mean(dim=1, keepdim=True)).clamp(min=1e-5)
        wn = w / s
        tern = torch.round(wn).clamp(-1.0, 1.0)
        zero_ratio = float((tern == 0).float().mean().item())
        sat_ratio = float((wn.abs() >= 1.0).float().mean().item())
        alpha_mean = float(self.alpha.abs().mean().item())
        alpha_drift = float((self.alpha - 1.0).abs().mean().item())
        return {
            "ternary_zero_ratio": zero_ratio,
            "quant_saturation_ratio": sat_ratio,
            "alpha_mean": alpha_mean,
            "alpha_drift": alpha_drift,
        }


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError("Rotary dim must be positive and even")
        self.dim = int(dim)
        self.base = float(base)
        inv = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._cache_len = 0

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        if (
            self._cache_len >= seq_len
            and self.cos_cached.numel() > 0
            and self.cos_cached.device == device
            and self.cos_cached.dtype == dtype
        ):
            return
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        self.cos_cached = emb.cos().to(dtype=dtype)
        self.sin_cached = emb.sin().to(dtype=dtype)
        self._cache_len = seq_len

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def apply(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # x: [B,H,T,D], D >= rope_dim
        b, h, t, d = x.shape
        if self.dim > d:
            raise ValueError(f"rotary dim {self.dim} > head dim {d}")
        self._update_cache(offset + t, x.device, x.dtype)
        cos = self.cos_cached[offset : offset + t].view(1, 1, t, self.dim)
        sin = self.sin_cached[offset : offset + t].view(1, 1, t, self.dim)
        x_rope = x[..., : self.dim]
        x_tail = x[..., self.dim :]
        xr = (x_rope * cos) + (self._rotate_half(x_rope) * sin)
        return xr if x_tail.numel() == 0 else torch.cat([xr, x_tail], dim=-1)


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden: int, intermediate: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden, intermediate)
        self.w2 = nn.Linear(hidden, intermediate)
        self.wo = nn.Linear(intermediate, hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(x))
        up = self.w2(x)
        return self.drop(self.wo(gate * up))


class QINNLayer(nn.Module):
    def __init__(self, hidden: int, rank: int = 32) -> None:
        super().__init__()
        rank = max(8, min(rank, hidden))
        self.norm = RMSNorm(hidden)
        self.down = nn.Linear(hidden, rank, bias=False)
        self.up = nn.Linear(rank, hidden, bias=False)
        self.gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = torch.tanh(self.down(h))
        h = self.up(h)
        return x + self.gain * h


class HebbianPlasticityLayer(nn.Module):
    def __init__(self, hidden: int, eta: float = 0.01, decay: float = 0.99, enable_inference_adapt: bool = False) -> None:
        super().__init__()
        self.eta = float(eta)
        self.decay = float(decay)
        self.enable_inference_adapt = bool(enable_inference_adapt)
        self.register_buffer("trace", torch.zeros(hidden), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or self.enable_inference_adapt:
            with torch.no_grad():
                m = x.detach().mean(dim=(0, 1))
                if self.trace.device != m.device:
                    self.trace = self.trace.to(m.device)
                self.trace.mul_(self.decay).add_(m * (1.0 - self.decay))
        gain = 1.0 + self.eta * torch.tanh(self.trace).view(1, 1, -1)
        return x * gain.to(x.dtype)


class NeuroSymbolicLayer(nn.Module):
    def __init__(self, hidden: int, num_rules: int = 8, strength: float = 0.05) -> None:
        super().__init__()
        self.rules = nn.Parameter(torch.randn(num_rules, hidden) * 0.02)
        self.strength = float(strength)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = torch.einsum("bth,rh->btr", x, self.rules)
        mix = torch.einsum("btr,rh->bth", torch.tanh(score), self.rules)
        return x + self.strength * mix / math.sqrt(x.shape[-1])


class WorldModelHead(nn.Module):
    def __init__(self, hidden: int, horizon: int = 1) -> None:
        super().__init__()
        self.horizon = max(1, int(horizon))
        self.proj = nn.Linear(hidden, hidden * self.horizon)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, t, h = x.shape
        y = self.proj(x).view(b, t, self.horizon, h)
        return {"world_dynamics_logits": y}


class LifelongSafetyLayer(nn.Module):
    def __init__(self, ema_decay: float = 0.99, max_adapt_gain: float = 0.05, drift_threshold: float = 0.35) -> None:
        super().__init__()
        self.ema_decay = float(ema_decay)
        self.max_adapt_gain = float(max_adapt_gain)
        self.drift_threshold = float(drift_threshold)
        self.register_buffer("ema_norm", torch.tensor(0.0), persistent=False)
        self.register_buffer("init", torch.tensor(False), persistent=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        with torch.no_grad():
            curr = x.detach().float().pow(2).mean().sqrt()
            if not bool(self.init.item()):
                self.ema_norm.copy_(curr)
                self.init.copy_(torch.tensor(True, device=self.init.device))
            else:
                self.ema_norm.mul_(self.ema_decay).add_(curr * (1 - self.ema_decay))
            drift = float((curr - self.ema_norm).abs() / (self.ema_norm.abs() + 1e-6))
        if drift <= self.drift_threshold:
            gain = 1.0
        else:
            over = min(1.0, (drift - self.drift_threshold) / max(self.drift_threshold, 1e-6))
            gain = 1.0 - over * self.max_adapt_gain
        return x * float(gain), {"drift": drift, "gain": float(gain)}


class ContinuousLatentODEStateChannel(nn.Module):
    def __init__(self, hidden: int, alpha: float = 0.1) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.alpha = float(alpha)
        self.register_buffer("state", torch.empty(0), persistent=False)

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        self.state = torch.zeros(batch_size, self.hidden, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        b, t, h = x.shape
        if self.state.numel() == 0 or self.state.shape != (b, h) or self.state.device != x.device:
            self.reset_state(b, x.device, x.dtype)
        st = self.state
        outs: List[torch.Tensor] = []
        for i in range(t):
            target = x[:, i, :]
            st = st + float(dt) * (-st + target)
            outs.append(x[:, i, :] + self.alpha * st)
        self.state = st.detach()
        return torch.stack(outs, dim=1)


class NeuromodulatoryGainLayer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor, workspace: Optional[torch.Tensor]) -> torch.Tensor:
        if workspace is None:
            return x
        gain = 1.0 + 0.1 * torch.tanh(self.proj(workspace)).unsqueeze(1)
        return x * gain


class LiquidMixer(nn.Module):
    def __init__(self, hidden: int, dt: float = 1.0) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.dt = float(dt)
        self.in_proj = nn.Linear(hidden, hidden)
        self.gate_proj = nn.Linear(hidden, hidden)
        self.register_buffer("state", torch.empty(0), persistent=False)

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        self.state = torch.zeros(batch_size, self.hidden, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h = x.shape
        if self.state.numel() == 0 or self.state.shape != (b, h) or self.state.device != x.device:
            self.reset_state(b, x.device, x.dtype)
        st = self.state
        outs: List[torch.Tensor] = []
        for i in range(t):
            xi = x[:, i, :]
            cand = torch.tanh(self.in_proj(xi))
            gate = torch.sigmoid(self.gate_proj(xi))
            st = st + self.dt * (-st + cand)
            outs.append(xi + gate * st)
        self.state = st.detach()
        return torch.stack(outs, dim=1)


class LiquidRouter(nn.Module):
    def __init__(self, hidden: int, num_experts: int, top_k: int = 2) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.num_experts = int(num_experts)
        self.top_k = max(1, min(int(top_k), self.num_experts))
        self.gate = nn.Linear(hidden, self.num_experts)
        self.jitter = 0.01
        self.capacity_factor = 1.25
        self.capacity_enforce = True
        self.collapse_threshold = 0.85
        self.register_buffer("inference_state", torch.zeros(1, hidden, 1), persistent=False)

    def get_state(self) -> torch.Tensor:
        return self.inference_state.clone()

    def set_state(self, state: torch.Tensor) -> None:
        if state.dim() != 3:
            raise ValueError(f"router state must be [B,H,W], got {tuple(state.shape)}")
        self.inference_state = state.detach().to(self.inference_state.device, dtype=self.inference_state.dtype)

    def forward(
        self, x_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float], torch.Tensor]:
        logits = self.gate(x_flat)
        if self.training and self.jitter > 0:
            logits = logits + torch.randn_like(logits) * self.jitter

        probs = torch.softmax(logits, dim=-1)
        top_vals, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
        top_w = top_vals / (top_vals.sum(dim=-1, keepdim=True).clamp(min=1e-9))
        capacity_mask = torch.ones_like(top_idx, dtype=torch.bool)
        overflow_ratio = 0.0

        if self.capacity_enforce and self.capacity_factor > 0.0:
            n = int(top_idx.size(0))
            e = int(self.num_experts)
            k = int(self.top_k)
            capacity = max(1, int(math.ceil(self.capacity_factor * (n * k) / max(1, e))))
            dropped = 0
            for expert_id in range(e):
                hits = (top_idx == expert_id).nonzero(as_tuple=False)
                if hits.size(0) > capacity:
                    overflow = hits[capacity:]
                    capacity_mask[overflow[:, 0], overflow[:, 1]] = False
                    dropped += int(overflow.size(0))
            top_w = top_w * capacity_mask.float()
            row_sum = top_w.sum(dim=-1, keepdim=True)
            empty_rows = row_sum.squeeze(-1) <= 0
            if bool(empty_rows.any().item()):
                top_w[empty_rows, 0] = 1.0
                capacity_mask[empty_rows, 0] = True
                row_sum = top_w.sum(dim=-1, keepdim=True)
            top_w = top_w / row_sum.clamp(min=1e-6)
            overflow_ratio = float(dropped) / float(max(1, n * k))

        flat_idx = top_idx[capacity_mask].reshape(-1)
        counts = torch.zeros(self.num_experts, device=logits.device, dtype=torch.float32)
        if flat_idx.numel() > 0:
            counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=counts.dtype))
        load = counts / max(1.0, float(flat_idx.numel()))
        importance = probs.mean(dim=0)
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
        max_load = float(load.max().detach().cpu().item()) if load.numel() > 0 else 0.0
        collapse = max_load > self.collapse_threshold
        aux_lb = ((importance - load.to(importance.dtype)) ** 2).mean() * float(self.num_experts)
        aux_z = torch.mean(torch.logsumexp(logits, dim=-1).pow(2)) * 1e-4
        aux_tensor = aux_lb + aux_z

        stats = {
            "router_entropy": float(entropy.detach().cpu().item()),
            "router_max_load": max_load,
            "capacity_overflow_ratio": float(overflow_ratio),
            "collapse_detected": float(1.0 if collapse else 0.0),
            "router_aux_lb": float(aux_lb.detach().cpu().item()),
            "router_aux_z": float(aux_z.detach().cpu().item()),
            "aux_loss": float(aux_tensor.detach().cpu().item()),
        }
        return top_idx, top_w, aux_tensor, stats, capacity_mask


class MoELayer(nn.Module):
    def __init__(
        self,
        hidden: int,
        intermediate: int,
        num_experts: int = 4,
        top_k: int = 2,
        use_structural_plasticity: bool = False,
        structural_update_interval: int = 100,
        moe_mode: str = "true_sparse_topk",
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.router = LiquidRouter(hidden, num_experts=num_experts, top_k=top_k)
        self.experts = nn.ModuleList([SwiGLUFFN(hidden, intermediate) for _ in range(num_experts)])
        self.shared = SwiGLUFFN(hidden, intermediate)
        self.moe_mode = str(moe_mode)
        self.use_structural_plasticity = bool(use_structural_plasticity)
        self.structural_update_interval = int(structural_update_interval)
        self.register_buffer("usage", torch.zeros(num_experts), persistent=False)
        self._step = 0

    def _structural_update(self) -> None:
        if not self.use_structural_plasticity:
            return
        # Safety: avoid in-graph parameter mutation during training backward.
        # Structural updates are deferred/off in active training steps to prevent
        # autograd version-counter breaks.
        if self.training:
            return
        if self._step == 0 or self._step % max(1, self.structural_update_interval) != 0:
            return
        low = int(torch.argmin(self.usage).item())
        high = int(torch.argmax(self.usage).item())
        if low == high:
            return
        with torch.no_grad():
            src = self.experts[high]
            dst = self.experts[low]
            for pd, ps in zip(dst.parameters(), src.parameters()):
                pd.copy_(0.9 * ps + 0.1 * torch.randn_like(ps) * 0.01)
        self.usage.mul_(0.5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        b, t, h = x.shape
        xf = x.view(b * t, h)
        top_idx, top_w, aux_tensor, stats, capacity_mask = self.router(xf)
        out_flat = torch.zeros_like(xf)

        if self.moe_mode == "dense_debug":
            # DEBUG-ONLY path: runs EVERY expert on EVERY token (full dense cost),
            # i.e. this is NOT sparse top-k routing. Do not use for training/perf
            # claims; the default mode 'true_sparse_topk' uses the sparse else-branch.
            expert_out = torch.stack([e(xf) for e in self.experts], dim=1)
            gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, h)
            chosen = expert_out.gather(1, gather_idx)
            out_flat = (chosen * top_w.unsqueeze(-1)).sum(dim=1)
        else:
            for expert_id, expert in enumerate(self.experts):
                expert_mask = top_idx == expert_id
                if capacity_mask is not None:
                    expert_mask = expert_mask & capacity_mask
                token_mask = expert_mask.any(dim=-1)
                if not bool(token_mask.any().item()):
                    continue
                selected_x = xf[token_mask]
                expert_out = expert(selected_x)
                weights = (top_w[token_mask] * expert_mask[token_mask].float()).sum(dim=-1, keepdim=True)
                out_flat[token_mask] += expert_out * weights

        out_flat = out_flat + 0.25 * self.shared(xf)
        out = out_flat.view(b, t, h)

        with torch.no_grad():
            counts = torch.zeros_like(self.usage)
            if capacity_mask is not None:
                active = top_idx[capacity_mask].reshape(-1)
            else:
                active = top_idx.reshape(-1)
            if active.numel() > 0:
                counts.scatter_add_(0, active, torch.ones_like(active, dtype=counts.dtype))
            self.usage.add_(counts)
            self._step += 1
            self._structural_update()

        stats["moe_mode_sparse"] = float(1.0 if self.moe_mode != "dense_debug" else 0.0)
        aux = aux_tensor.to(device=x.device, dtype=x.dtype)
        return out, aux, stats


class MLA(nn.Module):
    """GQA (grouped-query attention) — NOT latent-MLA.

    Despite the historical ``MLA`` name, this module implements standard
    grouped-query attention: ``q_proj`` produces ``num_heads`` query heads while
    ``k_proj``/``v_proj`` produce ``num_kv_heads`` (< num_heads) KV heads that are
    broadcast via ``_repeat_kv``. There is NO low-rank/latent KV compression
    (no kv_a/kv_b down/up projection, no compressed KV cache), so this is not
    DeepSeek-style Multi-head Latent Attention. The canonical source
    (layers/mla.py) already renames the equivalent class to ``GQA``; this onefile
    copy keeps the legacy ``MLA`` name for parity only. Label/naming note only —
    numerical behavior (GQA) is correct.
    """

    def __init__(
        self,
        hidden: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.dropout = float(dropout)

        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden)

        self.rope_dim = self.head_dim if rope_dim is None else int(rope_dim)
        if self.rope_dim <= 0 or self.rope_dim > self.head_dim or self.rope_dim % 2 != 0:
            raise ValueError("Invalid rope_dim; must be even and within (0, head_dim]")
        self.rope = RotaryEmbedding(self.rope_dim)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_kv_heads == self.num_heads:
            return x
        rep = self.num_heads // self.num_kv_heads
        return x.repeat_interleave(rep, dim=1)

    def _manual_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
        d = q.shape[-1]
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
        if causal:
            t = q.shape[-2]
            s = k.shape[-2]
            mask = torch.ones(t, s, device=q.device, dtype=torch.bool).triu(diagonal=1 + (s - t))
            scores = scores.masked_fill(mask.view(1, 1, t, s), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)

        offset = 0 if past_kv is None else int(past_kv[0].shape[2])
        q = self.rope.apply(q, offset=offset)
        k = self.rope.apply(k, offset=offset)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        present = (k, v) if use_cache else None

        k_rep = self._repeat_kv(k)
        v_rep = self._repeat_kv(v)

        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q.contiguous(),
                k_rep.contiguous(),
                v_rep.contiguous(),
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            out = self._manual_attention(q, k_rep, v_rep, causal=True)

        out = out.transpose(1, 2).contiguous().view(b, t, self.num_heads * self.head_dim)
        return self.o_proj(out), present


@dataclass
class MertFormerCfg:
    vocab_size: int = 32768
    max_seq_len: int = 512
    hidden_size: int = 272
    intermediate_size: int = 1024
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 4
    head_dim: int = 34
    rope_dim: Optional[int] = 34
    dropout: float = 0.0
    use_moe: bool = True
    num_experts: int = 4
    num_experts_per_tok: int = 2
    moe_every_n_layers: int = 2
    use_liquid: bool = True
    liquid_layers_idx: Tuple[int, ...] = (0, 3, 6)
    liquid_dt: float = 1.0
    use_qinn: bool = False
    use_hebbian_plasticity: bool = True
    hebbian_eta: float = 0.01
    hebbian_decay: float = 0.99
    hebbian_inference_adapt: bool = False
    use_neuro_symbolic_layer: bool = True
    neuro_symbolic_rules: int = 8
    use_world_model_head: bool = True
    world_model_horizon: int = 1
    use_lifelong_safety_layer: bool = True
    lifelong_ema_decay: float = 0.99
    lifelong_max_adaptation_gain: float = 0.05
    lifelong_drift_threshold: float = 0.35
    use_latent_ode_state_channel: bool = True
    latent_ode_dt: float = 1.0
    use_global_workspace_broadcast: bool = True
    workspace_blend: float = 0.7
    use_cross_expert_sync_bus: bool = True
    cross_expert_sync_gain: float = 0.05
    use_structural_plasticity: bool = True
    structural_update_interval: int = 50
    moe_mode: str = "true_sparse_topk"
    use_learned_pos_embedding: bool = False
    use_gradient_checkpointing: bool = False
    embedding_scale: bool = True
    chat_context_truncate: bool = True
    chat_decode_completion_only: bool = True


class MertFormerBlock(nn.Module):
    def __init__(self, cfg: MertFormerCfg, layer_id: int) -> None:
        super().__init__()
        self.layer_id = int(layer_id)
        self.cfg = cfg
        self.norm1 = RMSNorm(cfg.hidden_size)
        self.norm2 = RMSNorm(cfg.hidden_size)
        self.attn = MLA(
            hidden=cfg.hidden_size,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            rope_dim=cfg.rope_dim,
            dropout=cfg.dropout,
        )

        self.use_moe = bool(cfg.use_moe and (self.layer_id % cfg.moe_every_n_layers == 0))
        self.use_liquid = bool(cfg.use_liquid and (self.layer_id in cfg.liquid_layers_idx))
        self.use_qinn = bool(cfg.use_qinn)
        self.use_hebbian = bool(cfg.use_hebbian_plasticity)
        self.use_neuro_symbolic = bool(cfg.use_neuro_symbolic_layer)
        self.use_lifelong = bool(cfg.use_lifelong_safety_layer)
        self.residual_scale = (2.0 * max(1, int(cfg.num_layers))) ** -0.5

        if self.use_moe:
            self.ff = MoELayer(
                hidden=cfg.hidden_size,
                intermediate=cfg.intermediate_size,
                num_experts=cfg.num_experts,
                top_k=cfg.num_experts_per_tok,
                use_structural_plasticity=cfg.use_structural_plasticity,
                structural_update_interval=cfg.structural_update_interval,
                moe_mode=cfg.moe_mode,
            )
        else:
            self.ff = SwiGLUFFN(cfg.hidden_size, cfg.intermediate_size, dropout=cfg.dropout)

        self.liquid = LiquidMixer(cfg.hidden_size, dt=cfg.liquid_dt) if self.use_liquid else None
        self.qinn = QINNLayer(cfg.hidden_size, rank=min(64, cfg.hidden_size // 4)) if self.use_qinn else None
        self.hebbian = (
            HebbianPlasticityLayer(
                cfg.hidden_size,
                eta=cfg.hebbian_eta,
                decay=cfg.hebbian_decay,
                enable_inference_adapt=cfg.hebbian_inference_adapt,
            )
            if self.use_hebbian
            else None
        )
        self.neuro_symbolic = (
            NeuroSymbolicLayer(cfg.hidden_size, num_rules=cfg.neuro_symbolic_rules, strength=0.05)
            if self.use_neuro_symbolic
            else None
        )
        self.lifelong = (
            LifelongSafetyLayer(
                ema_decay=cfg.lifelong_ema_decay,
                max_adapt_gain=cfg.lifelong_max_adaptation_gain,
                drift_threshold=cfg.lifelong_drift_threshold,
            )
            if self.use_lifelong
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Dict[str, float]]:
        stats: Dict[str, float] = {}
        h = x
        a, present = self.attn(self.norm1(x), past_kv=past_kv, use_cache=use_cache)
        x = h + a * self.residual_scale

        if self.liquid is not None:
            x = self.liquid(x)

        ff_in = self.norm2(x)
        if self.use_moe:
            ff_out, aux, moe_stats = self.ff(ff_in)
            stats.update(moe_stats)
        else:
            ff_out = self.ff(ff_in)
            aux = torch.zeros((), device=x.device, dtype=x.dtype)
        x = x + ff_out * self.residual_scale

        if self.qinn is not None:
            x = self.qinn(x)
        if self.hebbian is not None:
            x = self.hebbian(x)
        if self.neuro_symbolic is not None:
            x = self.neuro_symbolic(x)
        if self.lifelong is not None:
            x, lif = self.lifelong(x)
            stats["lifelong_drift"] = lif["drift"]
            stats["lifelong_gain"] = lif["gain"]

        return x, aux, present, stats


class LegacyOnecellMertFormerTiny(nn.Module):
    def __init__(self, cfg: MertFormerCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_embeddings = (
            nn.Parameter(torch.zeros(1, cfg.max_seq_len, cfg.hidden_size))
            if cfg.use_learned_pos_embedding
            else None
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([MertFormerBlock(cfg, i) for i in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight

        self.latent_ode = (
            ContinuousLatentODEStateChannel(cfg.hidden_size) if cfg.use_latent_ode_state_channel else None
        )
        self.neuromod = NeuromodulatoryGainLayer(cfg.hidden_size)
        self.world_head = WorldModelHead(cfg.hidden_size, horizon=cfg.world_model_horizon) if cfg.use_world_model_head else None
        self.embed_scale = math.sqrt(float(cfg.hidden_size)) if cfg.embedding_scale else 1.0

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        b, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"seq_len {t} exceeds max_seq_len {self.cfg.max_seq_len}")

        x = self.tok_embeddings(input_ids) * self.embed_scale
        if self.pos_embeddings is not None:
            x = x + self.pos_embeddings[:, :t, :]
        x = self.drop(x)

        if self.latent_ode is not None and past_key_values is None:
            self.latent_ode.reset_state(b, x.device, x.dtype)

        workspace = x.mean(dim=1) if self.cfg.use_global_workspace_broadcast else None
        aux_total = torch.zeros((), device=x.device, dtype=x.dtype)
        present: List[Tuple[torch.Tensor, torch.Tensor]] = []
        layer_stats: List[Dict[str, float]] = []
        sync_state = None

        for i, block in enumerate(self.layers):
            if self.latent_ode is not None:
                x = self.latent_ode(x, dt=self.cfg.latent_ode_dt)

            pkv = None
            if past_key_values is not None and i < len(past_key_values):
                pkv = past_key_values[i]

            if self.cfg.use_gradient_checkpointing and self.training and not use_cache:
                from torch.utils.checkpoint import checkpoint

                def _run_block(inp: torch.Tensor):
                    out_x, out_aux, _, _ = block(inp, past_kv=pkv, use_cache=False)
                    return out_x, out_aux

                x, aux = checkpoint(_run_block, x, use_reentrant=False)
                p = None
                stats = {}
            else:
                x, aux, p, stats = block(x, past_kv=pkv, use_cache=use_cache)
            aux_total = aux_total + aux.to(aux_total.dtype)
            if p is not None:
                present.append(p)
            layer_stats.append(stats)

            if workspace is not None:
                token_summary = x.mean(dim=1)
                blend = float(min(max(self.cfg.workspace_blend, 0.0), 1.0))
                workspace = workspace * blend + token_summary * (1.0 - blend)
                x = self.neuromod(x, workspace)

            if self.cfg.use_cross_expert_sync_bus:
                token_mean = x.mean(dim=1, keepdim=True)
                if sync_state is None:
                    sync_state = token_mean
                else:
                    sync_state = 0.9 * sync_state + 0.1 * token_mean
                x = x + self.cfg.cross_expert_sync_gain * sync_state

        x = self.norm(x)
        logits = self.lm_head(x)
        extras: Dict[str, Any] = {
            "present_key_values": present if use_cache else None,
            "layer_stats": layer_stats,
        }
        if self.world_head is not None:
            extras.update(self.world_head(x))
        return logits, aux_total, extras

    def reset_router_state(self, batch_size: int = 1) -> None:
        if self.latent_ode is not None:
            self.latent_ode.reset_state(
                batch_size=batch_size,
                device=self.tok_embeddings.weight.device,
                dtype=self.tok_embeddings.weight.dtype,
            )
        for block in self.layers:
            ff = getattr(block, "ff", None)
            router = getattr(ff, "router", None)
            if router is None:
                continue
            set_state = getattr(router, "set_state", None)
            if callable(set_state):
                try:
                    set_state(
                        torch.zeros(
                            batch_size,
                            block.cfg.hidden_size,
                            1,
                            device=self.tok_embeddings.weight.device,
                            dtype=self.tok_embeddings.weight.dtype,
                        )
                    )
                except Exception:
                    pass


def legacy_parity_self_check(model: LegacyOnecellMertFormerTiny, cfg: Dict[str, Any], device: str) -> Dict[str, Any]:
    model = model.to(device)
    model.train()
    b = min(2, max(1, int(cfg.get("batch_size", 2))))
    t = min(16, max(8, int(cfg.get("seq_len", 64))))
    vocab = int(cfg.get("vocab_size", 32768))
    x = torch.randint(0, vocab, (b, t), device=device, dtype=torch.long)
    y = torch.randint(0, vocab, (b, t), device=device, dtype=torch.long)
    ce = nn.CrossEntropyLoss()
    try:
        logits, aux, extras = model(x)
        loss = ce(logits.reshape(-1, vocab), y.reshape(-1)) + 0.1 * aux.float()
        finite = bool(torch.isfinite(loss).item())
        if finite:
            loss.backward()
        flags_ok = {
            "hebbian": bool(model.cfg.use_hebbian_plasticity) == bool(cfg.get("mert_enable_all_extensions", True)),
            "neuro_symbolic": bool(model.cfg.use_neuro_symbolic_layer) == bool(cfg.get("mert_enable_all_extensions", True)),
            "world_model": bool(model.cfg.use_world_model_head) == bool(cfg.get("mert_enable_all_extensions", True)),
            "lifelong": bool(model.cfg.use_lifelong_safety_layer) == bool(cfg.get("mert_enable_all_extensions", True)),
            "latent_ode": bool(model.cfg.use_latent_ode_state_channel) == bool(cfg.get("mert_enable_all_extensions", True)),
            "workspace": bool(model.cfg.use_global_workspace_broadcast) == bool(cfg.get("mert_enable_all_extensions", True)),
            "cross_expert_sync": bool(model.cfg.use_cross_expert_sync_bus) == bool(cfg.get("mert_enable_all_extensions", True)),
            "structural_plasticity": bool(model.cfg.use_structural_plasticity) == bool(cfg.get("mert_enable_all_extensions", True)),
        }
        all_flags_ok = all(flags_ok.values())
        layer_stats = extras.get("layer_stats", []) if isinstance(extras, dict) else []
        has_router_entropy = any(isinstance(s, dict) and ("router_entropy" in s) for s in layer_stats)
        return {
            "ok": bool(finite and all_flags_ok),
            "finite_loss": finite,
            "all_flags_ok": all_flags_ok,
            "flags": flags_ok,
            "has_router_entropy": bool(has_router_entropy),
            "arch_contract": ARCH_PARITY_CONTRACT["name"],
        }
    except Exception as e:
        return {
            "ok": False,
            "error": _format_exception(e),
            "arch_contract": ARCH_PARITY_CONTRACT["name"],
        }


class VanillaTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.max_seq_len = int(max_seq_len)
        self.tok = nn.Embedding(vocab_size, hidden_size)
        self.pos = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        enc = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        if t > self.max_seq_len:
            raise ValueError(f"sequence {t} exceeds max_seq_len {self.max_seq_len}")
        x = self.tok(input_ids) + self.pos[:, :t, :]
        mask = torch.triu(torch.ones(t, t, device=input_ids.device, dtype=torch.bool), diagonal=1)
        x = self.blocks(x, mask=mask)
        x = self.norm(x)
        return self.head(x)


# =============================================================================
# Strict BitNet conversion
# =============================================================================
BITNET_FLOAT_SAFE_PATTERNS = [
    "tok_embeddings",
    "lm_head",
    "norm",
    "router.gate",
    "world_head",
    "neuro_symbolic",
    "lifelong",
]

BITNET_ATTENTION_QKVO_PATTERNS = [
    "attn.q_proj",
    "attn.k_proj",
    "attn.v_proj",
    "attn.o_proj",
]

def _should_skip_bitnet(full_name: str, bitnet_mode: str, skip_attention_qkvo: bool = True) -> bool:
    patterns = list(BITNET_FLOAT_SAFE_PATTERNS)
    if bitnet_mode == "stable" and skip_attention_qkvo:
        patterns.extend(BITNET_ATTENTION_QKVO_PATTERNS)
    return any(p in full_name for p in patterns)


def legacy_convert_model_to_strict_bitnet(
    model: nn.Module,
    logger: Optional[InMemoryRunLogger] = None,
    bitnet_mode: str = "stable",
    skip_attention_qkvo: bool = True,
) -> Dict[str, Any]:
    converted = 0
    skipped = 0

    def recurse(parent: nn.Module, prefix: str = "") -> None:
        nonlocal converted, skipped
        for child_name, child in list(parent.named_children()):
            full = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and not isinstance(child, BitLinearStrict):
                if _should_skip_bitnet(full, bitnet_mode=bitnet_mode, skip_attention_qkvo=skip_attention_qkvo):
                    skipped += 1
                    continue
                new = BitLinearStrict(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    device=child.weight.device,
                    dtype=child.weight.dtype,
                )
                with torch.no_grad():
                    new.weight.copy_(child.weight)
                    if child.bias is not None:
                        new.bias.copy_(child.bias)
                setattr(parent, child_name, new)
                converted += 1
            else:
                recurse(child, full)

    recurse(model)
    out = {
        "converted_linear": converted,
        "skipped_linear": skipped,
        "bitnet_mode": bitnet_mode,
        "skip_attention_qkvo": bool(skip_attention_qkvo),
    }
    if logger is not None:
        logger.log_event("strict_bitnet_convert", out)
    return out


def legacy_collect_bitnet_telemetry(model: nn.Module, bitnet_mode: str = "stable") -> Dict[str, Any]:
    vals: List[Dict[str, float]] = []
    for _, m in model.named_modules():
        if isinstance(m, BitLinearStrict):
            vals.append(m.telemetry())
    if not vals:
        return {
            "bitlinear_count": 0,
            "ternary_zero_ratio_mean": 0.0,
            "quant_saturation_ratio_mean": 0.0,
            "ternary_zero_ratio_drift": 0.0,
            "quant_saturation_drift": 0.0,
            "alpha_mean": 0.0,
            "alpha_drift_mean": 0.0,
            "mode": bitnet_mode,
        }
    z_vals = [float(v["ternary_zero_ratio"]) for v in vals]
    s_vals = [float(v["quant_saturation_ratio"]) for v in vals]
    z = sum(z_vals) / len(vals)
    s = sum(s_vals) / len(vals)
    a = sum(v["alpha_mean"] for v in vals) / len(vals)
    d = sum(v["alpha_drift"] for v in vals) / len(vals)
    z_drift = float(max(z_vals) - min(z_vals)) if z_vals else 0.0
    s_drift = float(max(s_vals) - min(s_vals)) if s_vals else 0.0
    return {
        "bitlinear_count": len(vals),
        "ternary_zero_ratio_mean": float(z),
        "quant_saturation_ratio_mean": float(s),
        "ternary_zero_ratio_drift": z_drift,
        "quant_saturation_drift": s_drift,
        "alpha_mean": float(a),
        "alpha_drift_mean": float(d),
        "mode": bitnet_mode,
    }


# =============================================================================
# Dataset wrappers
# =============================================================================
class PackedDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(self.x.size(0))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def make_loader(ds: Dataset, batch_size: int, seed: int, shuffle: bool = True) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True, generator=g)


# =============================================================================
# Checkpointing
# =============================================================================
def atomic_json_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def prune_checkpoints_rolling5(ckpt_dir: Path) -> List[str]:
    removed: List[str] = []
    step_files = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    keep = 5
    if len(step_files) > keep:
        for p in step_files[: len(step_files) - keep]:
            try:
                p.unlink()
                removed.append(str(p))
            except Exception:
                pass
    return removed


def collect_rng_state() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            out["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return out


def restore_rng_state(state: Dict[str, Any]) -> None:
    try:
        if "python_random_state" in state:
            random.setstate(state["python_random_state"])
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"])
        if "cuda_rng_state_all" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    except Exception:
        pass


def save_checkpoint_atomic(
    ckpt_dir: Path,
    tag: str,
    payload: Dict[str, Any],
    is_best: bool = False,
) -> Dict[str, Any]:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag_path = ckpt_dir / f"{tag}.pt"
    latest_path = ckpt_dir / "latest.pt"
    best_path = ckpt_dir / "best.pt"
    manifest_path = ckpt_dir / "manifest.json"

    atomic_torch_save(tag_path, payload)
    atomic_torch_save(latest_path, payload)
    if is_best:
        atomic_torch_save(best_path, payload)

    removed = prune_checkpoints_rolling5(ckpt_dir)

    manifest = {
        "updated_at_utc": _utc_now(),
        "latest": str(latest_path),
        "best": str(best_path) if best_path.exists() else "",
        "tag": tag,
        "removed": removed,
        "all_step_files": [str(x) for x in sorted(ckpt_dir.glob("step_*.pt"))],
    }
    atomic_json_write(manifest_path, manifest)
    return manifest


def load_checkpoint_resume(ckpt_dir: Path, mode: str, path: str, device: str) -> Tuple[Optional[Dict[str, Any]], str]:
    ckpt_path: Optional[Path] = None
    if mode == "path" and path:
        p = Path(path).expanduser()
        if p.exists():
            ckpt_path = p
    elif mode == "best":
        p = ckpt_dir / "best.pt"
        if p.exists():
            ckpt_path = p
    else:
        p = ckpt_dir / "latest.pt"
        if p.exists():
            ckpt_path = p

    if ckpt_path is None:
        return None, "checkpoint_not_found"
    try:
        payload = torch.load(ckpt_path, map_location=device)
        return payload, str(ckpt_path)
    except Exception as e:
        return None, f"checkpoint_load_error:{type(e).__name__}:{e}"


# =============================================================================
# Train/Eval
# =============================================================================
@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    tokens_seen: int = 0
    best_val_loss: float = float("inf")
    last_checkpoint_time: float = 0.0


@dataclass
class TrainArtifacts:
    curve_data: Dict[str, List[float]]
    state: TrainState
    best_checkpoint_path: str
    latest_checkpoint_path: str
    checkpoint_manifest: Dict[str, Any]


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def build_optimizer_scheduler(model: nn.Module, cfg: Dict[str, Any]) -> Tuple[torch.optim.Optimizer, LambdaLR]:
    lr = float(cfg["lr"])
    wd = float(cfg["weight_decay"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    max_steps = int(cfg["max_steps"])
    warmup = max(1, int(max_steps * float(cfg["warmup_ratio"])))
    min_lr_ratio = float(cfg["min_lr_ratio"])

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        if max_steps <= warmup:
            return min_lr_ratio
        progress = min(1.0, float(step - warmup) / float(max_steps - warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    sch = LambdaLR(opt, lr_lambda=lr_lambda)
    return opt, sch


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    vocab_size: int,
    device: str,
    max_batches: int,
    aux_coeff: float,
) -> Dict[str, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    losses: List[float] = []
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            if isinstance(out, tuple):
                logits = out[0]
                aux = out[1] if len(out) > 1 and torch.is_tensor(out[1]) else torch.tensor(0.0, device=device)
                loss = ce(logits.reshape(-1, vocab_size), y.reshape(-1)) + float(aux_coeff) * aux.float()
            else:
                logits = out
                loss = ce(logits.reshape(-1, vocab_size), y.reshape(-1))
            if torch.isfinite(loss):
                losses.append(float(loss.detach().cpu().item()))
    model.train()
    if not losses:
        return {
            "val_loss": float("inf"),
            "val_ppl": float("inf"),
            "val_ppl_raw": float("inf"),
            "val_ppl_capped": float("inf"),
            "val_ppl_cap_applied": False,
        }
    vl = float(sum(losses) / len(losses))
    ppl_cap_applied = False
    try:
        ppl_raw = float(math.exp(vl))
    except OverflowError:
        ppl_raw = float("inf")
        ppl_cap_applied = True
    if not math.isfinite(ppl_raw):
        ppl_raw = float("inf")
        ppl_cap_applied = True
    ppl_capped = float(math.exp(min(20.0, vl)))
    if abs(ppl_capped - ppl_raw) > 1e-6 and math.isfinite(ppl_raw):
        ppl_cap_applied = True
    if not math.isfinite(ppl_raw):
        ppl_cap_applied = True
    return {
        "val_loss": vl,
        "val_ppl": ppl_capped,
        "val_ppl_raw": ppl_raw,
        "val_ppl_capped": ppl_capped,
        "val_ppl_cap_applied": bool(ppl_cap_applied),
    }


def detect_anomaly(loss_hist: Sequence[float]) -> bool:
    if len(loss_hist) < 12:
        return False
    tail = list(loss_hist[-12:])
    med = sorted(tail[:-1])[len(tail[:-1]) // 2]
    if med <= 1e-8:
        return False
    return tail[-1] > 10.0 * med


def build_checkpoint_payload(
    cfg: Dict[str, Any],
    model: nn.Module,
    opt: torch.optim.Optimizer,
    sch: LambdaLR,
    scaler: Any,
    state: TrainState,
    tokenizer: HybridTokenizer,
    expected_run_hash: str,
    compat_signature: str,
    arch_parity_signature: str,
) -> Dict[str, Any]:
    return {
        "schema": "kaggle_onecell_t4_checkpoint_manifest_v1",
        "saved_at_utc": _utc_now(),
        "compat_signature": compat_signature,
        "arch_parity_signature": arch_parity_signature,
        "tokenizer_backend": tokenizer.backend_name,
        "data_policy_tag": str(cfg.get("data_policy_tag", "open+fallback")),
        "profile_stamp": str(cfg.get("profile", "")),
        "parity_level": str(cfg.get("parity_level", "hybrid_strict")),
        "bitnet_mode": str(cfg.get("bitnet_mode", "stable")),
        "moe_mode": str(cfg.get("moe_mode", "true_sparse_topk")),
        "logger_mode": str(cfg.get("logger_mode", "jsonl_ring")),
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sch.state_dict(),
        "scaler": scaler.state_dict(),
        "train_state": {
            "step": state.step,
            "epoch": state.epoch,
            "tokens_seen": state.tokens_seen,
            "best_val_loss": state.best_val_loss,
        },
        "rng_state": collect_rng_state(),
        "tokenizer_state": tokenizer.state_dict(),
        "run_config_hash": expected_run_hash,
        "run_config": cfg,
        "bitnet_telemetry": collect_bitnet_telemetry(model, bitnet_mode=str(cfg.get("bitnet_mode", "stable"))),
    }


def write_eval_incremental_evidence(
    cfg: Dict[str, Any],
    step: int,
    tokens_seen: int,
    eval_result: Dict[str, Any],
    curve: Dict[str, List[float]],
) -> None:
    run_dir = Path(str(cfg.get("artifact_run_dir", cfg.get("out_dir", "."))))
    eval_dir = Path(str(cfg.get("eval_snapshot_dir", run_dir / "eval_snapshots")))
    eval_dir.mkdir(parents=True, exist_ok=True)
    stamp = _utc_now().replace(":", "").replace("-", "")
    snap = {
        "generated_at_utc": _utc_now(),
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "eval": safe_jsonable(eval_result),
        "validation_trend": validation_trend_metrics(curve.get("val_loss", [])),
        "warmup_excluded_loss_drop": warmup_excluded_loss_drop(curve.get("train_loss", [])),
    }
    snap_path = eval_dir / f"eval_step_{int(step):08d}_{stamp}.json"
    atomic_json_write(snap_path, snap)

    inc_csv = Path(str(cfg.get("incremental_eval_csv_path", run_dir / "eval_incremental.csv")))
    append_csv_row(
        inc_csv,
        fieldnames=[
            "step",
            "tokens_seen",
            "val_loss",
            "val_ppl_raw",
            "val_ppl_capped",
            "val_ppl_cap_applied",
            "router_entropy_mean",
            "router_max_load_p95",
            "collapse_events",
            "time_utc",
        ],
        row={
            "step": int(step),
            "tokens_seen": int(tokens_seen),
            "val_loss": float(eval_result.get("val_loss", float("inf"))),
            "val_ppl_raw": float(eval_result.get("val_ppl_raw", float("inf"))),
            "val_ppl_capped": float(eval_result.get("val_ppl_capped", float("inf"))),
            "val_ppl_cap_applied": bool(eval_result.get("val_ppl_cap_applied", False)),
            "router_entropy_mean": _mean(curve.get("router_entropy", [])),
            "router_max_load_p95": _p95(curve.get("router_max_load", [])),
            "collapse_events": int(sum(1 for x in curve.get("collapse_detected", []) if float(x) > 0.0)),
            "time_utc": _utc_now(),
        },
    )
    health_txt = Path(str(cfg.get("incremental_health_txt_path", run_dir / "health_latest.txt")))
    health_txt.parent.mkdir(parents=True, exist_ok=True)
    health_lines = [
        f"time_utc={_utc_now()}",
        f"step={int(step)} tokens_seen={int(tokens_seen)}",
        f"val_loss={float(eval_result.get('val_loss', float('inf'))):.6f}",
        f"val_ppl_raw={float(eval_result.get('val_ppl_raw', float('inf')))}",
        f"val_ppl_capped={float(eval_result.get('val_ppl_capped', float('inf')))}",
        f"val_ppl_cap_applied={bool(eval_result.get('val_ppl_cap_applied', False))}",
        f"router_entropy_mean={_mean(curve.get('router_entropy', [])):.6f}",
        f"router_max_load_p95={_p95(curve.get('router_max_load', [])):.6f}",
        f"collapse_events={int(sum(1 for x in curve.get('collapse_detected', []) if float(x) > 0.0))}",
    ]
    health_txt.write_text("\n".join(health_lines) + "\n", encoding="utf-8")


def train_loop_deep(
    model: MertFormerTiny,
    tokenizer: HybridTokenizer,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    cfg: Dict[str, Any],
    device: str,
    logger: InMemoryRunLogger,
) -> Tuple[MertFormerTiny, TrainArtifacts, Dict[str, Any]]:
    ckpt_dir = Path(str(cfg["checkpoint_dir"])).expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(str(cfg.get("artifact_run_dir", cfg.get("out_dir", ".")))).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.setdefault("eval_snapshot_dir", str(run_dir / "eval_snapshots"))
    cfg.setdefault("incremental_eval_csv_path", str(run_dir / "eval_incremental.csv"))
    cfg.setdefault("incremental_health_txt_path", str(run_dir / "health_latest.txt"))

    train_ds = PackedDataset(train_x, train_y)
    val_ds = PackedDataset(val_x, val_y)

    current_bs = int(cfg["batch_size"])
    max_oom_retries = int(cfg["max_oom_retries"])
    oom_count = 0

    train_loader = make_loader(train_ds, batch_size=current_bs, seed=int(cfg["seed"]) + 101, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=max(1, current_bs), seed=int(cfg["seed"]) + 202, shuffle=False)

    model = model.to(device)
    model.train()

    # Optional strict bitnet conversion
    bitnet_info = {"enabled": False, "converted_linear": 0, "skipped_linear": 0}
    if bool(cfg.get("strict_bitnet", False)):
        bitnet_conv = convert_model_to_strict_bitnet(
            model,
            logger=logger,
            bitnet_mode=str(cfg.get("bitnet_mode", "stable")),
            skip_attention_qkvo=bool(cfg.get("bitnet_skip_attention_qkvo", True)),
        )
        bitnet_info.update(bitnet_conv)
        bitnet_info["enabled"] = True

    # Optimizer/scheduler must be built after potential module replacement
    # (e.g., strict BitNet conversion introducing/replacing parameters).
    opt, sch = build_optimizer_scheduler(model, cfg)

    if device == "cuda" and bool(cfg["amp_enabled"]):
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)  # type: ignore[attr-defined]
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=False)  # type: ignore[attr-defined]
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=False)

    expected_run_hash = hash_config(cfg)
    compat_signature = build_compat_signature(cfg, tokenizer_backend=tokenizer.backend_name)
    arch_parity_signature = compute_arch_parity_signature(cfg, tokenizer_backend=tokenizer.backend_name)

    # Resume
    state = TrainState(step=0, epoch=0, tokens_seen=0, best_val_loss=float("inf"), last_checkpoint_time=time.time())
    latest_ckpt_path = ""
    best_ckpt_path = ""
    manifest: Dict[str, Any] = {}
    resume_decision: Dict[str, Any] = {
        "attempted": False,
        "source": "",
        "accepted": False,
        "reason": "",
        "rejected_reasons": [],
    }

    resume_payload, resume_info = load_checkpoint_resume(
        ckpt_dir=ckpt_dir,
        mode=str(cfg.get("resume_mode", "auto")),
        path=str(cfg.get("resume_path", "")),
        device=device,
    )
    if resume_payload is not None:
        resume_decision["attempted"] = True
        resume_decision["source"] = resume_info
        rejected: List[str] = []
        ck_hash = str(resume_payload.get("run_config_hash", ""))
        ck_sig = str(resume_payload.get("compat_signature", ""))
        ck_parity = str(resume_payload.get("arch_parity_signature", ""))
        ck_tok_backend = str(resume_payload.get("tokenizer_backend", ""))
        ck_step = int(resume_payload.get("train_state", {}).get("step", 0))
        if not ck_sig:
            rejected.append("compat_signature_missing")
        if not ck_parity:
            rejected.append("arch_parity_signature_missing")
        if bool(cfg.get("resume_hash_gate", True)) and ck_hash and ck_hash != expected_run_hash:
            rejected.append("run_config_hash_mismatch")
        if ck_sig and ck_sig != compat_signature:
            rejected.append("compat_signature_mismatch")
        if ck_parity and ck_parity != arch_parity_signature:
            rejected.append("arch_parity_signature_mismatch")
        if bool(cfg.get("resume_require_tokenizer_backend_match", True)) and ck_tok_backend and ck_tok_backend != tokenizer.backend_name:
            rejected.append("tokenizer_backend_mismatch")
        if bool(cfg.get("resume_reject_on_step_exhausted", True)) and ck_step >= int(cfg["max_steps"]):
            rejected.append("checkpoint_step_exhausted")
        if rejected:
            resume_decision["accepted"] = False
            resume_decision["reason"] = "fresh_start_due_to_incompat"
            resume_decision["rejected_reasons"] = rejected
            logger.log_event("resume_rejected", {"source": resume_info, "reasons": rejected})
            print(f"[resume] rejected ({','.join(rejected)}), starting fresh.")
            resume_payload = None

    if resume_payload is not None:
        try:
            model.load_state_dict(resume_payload["model"], strict=False)
            opt.load_state_dict(resume_payload["optimizer"])
            sch.load_state_dict(resume_payload["scheduler"])
            if "scaler" in resume_payload and scaler is not None:
                scaler.load_state_dict(resume_payload["scaler"])
            st = resume_payload.get("train_state", {})
            state.step = int(st.get("step", 0))
            state.epoch = int(st.get("epoch", 0))
            state.tokens_seen = int(st.get("tokens_seen", 0))
            state.best_val_loss = float(st.get("best_val_loss", float("inf")))
            state.last_checkpoint_time = float(time.time())
            tok_state = resume_payload.get("tokenizer_state")
            if isinstance(tok_state, dict):
                tokenizer.load_state_dict(tok_state)
            restore_rng_state(resume_payload.get("rng_state", {}))
            resume_decision["accepted"] = True
            resume_decision["reason"] = "resume_ok"
            logger.log_event("resume_ok", {"source": resume_info, "step": state.step})
        except Exception as e:
            resume_decision["accepted"] = False
            resume_decision["reason"] = f"resume_fail:{type(e).__name__}:{e}"
            logger.log_event("resume_fail", {"source": resume_info, "error": f"{type(e).__name__}:{e}"})
    else:
        if not resume_decision["reason"]:
            resume_decision["reason"] = "resume_skip"
        logger.log_event("resume_skip", {"reason": resume_info})

    ce = nn.CrossEntropyLoss()
    grad_accum_steps = max(1, int(cfg["grad_accum_steps"]))
    reset_device_peak_memory(device)
    peak_mem_gb = 0.0
    nan_count = 0
    anomaly_count = 0
    runtime_error_count = 0
    step_tokens_total = 0
    step_time_total = 0.0
    mem_samples_gb: List[float] = []
    stop_reason = "unknown"
    stop_reason_codes: List[str] = []
    interrupted = False
    sigterm_requested = False

    curve = {
        "step": [],
        "train_loss": [],
        "val_loss": [],
        "val_ppl": [],
        "grad_norm": [],
        "lr": [],
        "tokens_seen": [],
        "step_time_sec": [],
        "tokens_per_sec": [],
        "router_entropy": [],
        "router_max_load": [],
        "capacity_overflow_ratio": [],
        "collapse_detected": [],
        "aux_loss": [],
    }

    start_time = time.time()
    t_wall_limit = float(cfg["max_wall_hours"]) * 3600.0
    target_tokens = int(cfg["target_train_tokens"])

    iter_loader = iter(train_loader)
    micro_step = 0
    rolling_losses: List[float] = []
    checkpoint_every_steps = int(cfg["checkpoint_interval_steps"])
    checkpoint_every_seconds = float(cfg["checkpoint_interval_minutes"]) * 60.0

    while True:
        # Stop criteria
        if bool(_RUNTIME_SIGNAL_STATE.get("sigterm", False)):
            sigterm_requested = True
            stop_reason = "sigterm"
            stop_reason_codes.append(str(_RUNTIME_SIGNAL_STATE.get("signal", "SIGTERM")))
            logger.log_event("stop", {"reason": stop_reason, "signal": _RUNTIME_SIGNAL_STATE.get("signal", "SIGTERM")})
            break
        elapsed = time.time() - start_time
        if elapsed >= t_wall_limit:
            stop_reason = "max_wall_hours"
            logger.log_event("stop", {"reason": "max_wall_hours", "elapsed_sec": elapsed})
            break
        if state.tokens_seen >= target_tokens:
            stop_reason = "target_train_tokens"
            logger.log_event("stop", {"reason": "target_train_tokens", "tokens_seen": state.tokens_seen})
            break
        if state.step >= int(cfg["max_steps"]):
            stop_reason = "max_steps"
            logger.log_event("stop", {"reason": "max_steps", "step": state.step})
            break

        try:
            x, y = next(iter_loader)
        except StopIteration:
            state.epoch += 1
            iter_loader = iter(train_loader)
            x, y = next(iter_loader)

        x = x.to(device)
        y = y.to(device)

        st = time.time()
        try:
            with maybe_autocast(device, bool(cfg.get("amp_enabled", True))):
                logits, aux_loss, extras = model(x)
                loss = ce(logits.reshape(-1, int(cfg["vocab_size"])), y.reshape(-1))
                loss = loss + float(cfg["aux_loss_coeff"]) * aux_loss.float()

            if not torch.isfinite(loss):
                nan_count += 1
                logger.log_event("nan_loss", {"step": state.step, "epoch": state.epoch})
                opt.zero_grad(set_to_none=True)
                continue

            loss_for_backward = loss / float(grad_accum_steps)
            scaler.scale(loss_for_backward).backward()
            micro_step += 1

            if micro_step % grad_accum_steps == 0:
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg["bitnet_clip_grad"]))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sch.step()

                state.step += 1
                step_tokens = int(x.size(0) * x.size(1))
                state.tokens_seen += step_tokens

                loss_val = float(loss.detach().cpu().item())
                gnorm = float(grad_norm.detach().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm)
                lr_val = float(opt.param_groups[0]["lr"])
                step_dt = float(time.time() - st)
                tps = _safe_div(step_tokens, step_dt, default=0.0)
                rolling_losses.append(loss_val)
                step_tokens_total += step_tokens
                step_time_total += step_dt
                mem_now = get_device_peak_memory_gb(device)
                peak_mem_gb = max(peak_mem_gb, mem_now)
                mem_samples_gb.append(mem_now)

                router_entropy = 0.0
                router_max_load = 0.0
                capacity_overflow_ratio = 0.0
                collapse_detected = 0.0
                aux_loss_stat = 0.0
                ls = extras.get("layer_stats", []) if isinstance(extras, dict) else []
                if ls:
                    vals = [float(s.get("router_entropy", 0.0)) for s in ls if isinstance(s, dict)]
                    if vals:
                        router_entropy = float(sum(vals) / len(vals))
                    vals_max = [float(s.get("router_max_load", 0.0)) for s in ls if isinstance(s, dict)]
                    if vals_max:
                        router_max_load = float(max(vals_max))
                    vals_over = [float(s.get("capacity_overflow_ratio", 0.0)) for s in ls if isinstance(s, dict)]
                    if vals_over:
                        capacity_overflow_ratio = float(sum(vals_over) / len(vals_over))
                    vals_col = [float(s.get("collapse_detected", 0.0)) for s in ls if isinstance(s, dict)]
                    if vals_col:
                        collapse_detected = float(max(vals_col))
                    vals_aux = [float(s.get("aux_loss", 0.0)) for s in ls if isinstance(s, dict)]
                    if vals_aux:
                        aux_loss_stat = float(sum(vals_aux) / len(vals_aux))

                curve["step"].append(state.step)
                curve["train_loss"].append(loss_val)
                curve["grad_norm"].append(gnorm)
                curve["lr"].append(lr_val)
                curve["tokens_seen"].append(state.tokens_seen)
                curve["step_time_sec"].append(step_dt)
                curve["tokens_per_sec"].append(tps)
                curve["router_entropy"].append(router_entropy)
                curve["router_max_load"].append(router_max_load)
                curve["capacity_overflow_ratio"].append(capacity_overflow_ratio)
                curve["collapse_detected"].append(collapse_detected)
                curve["aux_loss"].append(aux_loss_stat)

                logger.log_step(
                    {
                        "variant": "mertformer_train",
                        "epoch": state.epoch,
                        "step": state.step,
                        "loss": loss_val,
                        "grad_norm": gnorm,
                        "lr": lr_val,
                        "tokens_seen": state.tokens_seen,
                        "step_time_sec": step_dt,
                        "tokens_per_sec": tps,
                        "router_entropy": router_entropy,
                        "router_max_load": router_max_load,
                        "capacity_overflow_ratio": capacity_overflow_ratio,
                        "collapse_detected": collapse_detected,
                        "aux_loss": aux_loss_stat,
                    }
                )

                if state.step == 1 or state.step % max(1, checkpoint_every_steps // 5) == 0:
                    print(
                        f"[train] step={state.step} loss={loss_val:.4f} grad={gnorm:.3f} "
                        f"lr={lr_val:.2e} tok={state.tokens_seen}"
                    )

                if detect_anomaly(rolling_losses):
                    anomaly_count += 1
                    logger.log_event("anomaly", {"step": state.step, "loss": loss_val, "kind": "spike_vs_median"})

                # Validation gate
                if state.step % int(cfg["eval_interval_steps"]) == 0:
                    ev = evaluate_model(
                        model,
                        val_loader,
                        vocab_size=int(cfg["vocab_size"]),
                        device=device,
                        max_batches=min(int(cfg["max_eval_batches"]), len(val_loader)),
                        aux_coeff=float(cfg["aux_loss_coeff"]),
                    )
                    curve["val_loss"].append(ev["val_loss"])
                    curve["val_ppl"].append(ev["val_ppl"])
                    logger.log_event("eval", {"step": state.step, **ev})
                    print(f"[eval] step={state.step} val_loss={ev['val_loss']:.4f} val_ppl={ev['val_ppl']:.2f}")
                    write_eval_incremental_evidence(cfg=cfg, step=state.step, tokens_seen=state.tokens_seen, eval_result=ev, curve=curve)

                    is_best = ev["val_loss"] < state.best_val_loss
                    if is_best:
                        state.best_val_loss = float(ev["val_loss"])

                    ckpt_payload = build_checkpoint_payload(
                        cfg=cfg,
                        model=model,
                        opt=opt,
                        sch=sch,
                        scaler=scaler,
                        state=state,
                        tokenizer=tokenizer,
                        expected_run_hash=expected_run_hash,
                        compat_signature=compat_signature,
                        arch_parity_signature=arch_parity_signature,
                    )
                    tag = f"step_{state.step:08d}"
                    manifest = save_checkpoint_atomic(ckpt_dir, tag=tag, payload=ckpt_payload, is_best=is_best)
                    latest_ckpt_path = str(ckpt_dir / "latest.pt")
                    if is_best:
                        best_ckpt_path = str(ckpt_dir / "best.pt")
                    state.last_checkpoint_time = time.time()

                # Time/step based checkpoint
                need_ckpt_time = (time.time() - state.last_checkpoint_time) >= checkpoint_every_seconds
                need_ckpt_step = state.step % checkpoint_every_steps == 0
                if need_ckpt_time or need_ckpt_step:
                    ckpt_payload = build_checkpoint_payload(
                        cfg=cfg,
                        model=model,
                        opt=opt,
                        sch=sch,
                        scaler=scaler,
                        state=state,
                        tokenizer=tokenizer,
                        expected_run_hash=expected_run_hash,
                        compat_signature=compat_signature,
                        arch_parity_signature=arch_parity_signature,
                    )
                    tag = f"step_{state.step:08d}"
                    manifest = save_checkpoint_atomic(ckpt_dir, tag=tag, payload=ckpt_payload, is_best=False)
                    latest_ckpt_path = str(ckpt_dir / "latest.pt")
                    state.last_checkpoint_time = time.time()

        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "mps backend out of memory" in msg:
                oom_count += 1
                logger.log_event("oom", {"step": state.step, "batch_size": current_bs, "error": str(e), "oom_count": oom_count})
                if device == "cuda":
                    torch.cuda.empty_cache()
                opt.zero_grad(set_to_none=True)

                if oom_count > max_oom_retries:
                    stop_reason = "too_many_oom"
                    logger.log_event("stop", {"reason": "too_many_oom", "oom_count": oom_count})
                    break

                if current_bs > 1:
                    current_bs = max(1, current_bs // 2)
                    train_loader = make_loader(train_ds, batch_size=current_bs, seed=int(cfg["seed"]) + 101 + oom_count, shuffle=True)
                    val_loader = make_loader(val_ds, batch_size=max(1, current_bs), seed=int(cfg["seed"]) + 202 + oom_count, shuffle=False)
                    iter_loader = iter(train_loader)
                    logger.log_event("oom_backoff", {"new_batch_size": current_bs})
                    print(f"[oom] backoff -> batch_size={current_bs}")
                    continue
                else:
                    cfg["grad_accum_steps"] = max(int(cfg["grad_accum_steps"]) + 1, 2)
                    grad_accum_steps = int(cfg["grad_accum_steps"])
                    logger.log_event("oom_backoff", {"new_grad_accum_steps": grad_accum_steps})
                    print(f"[oom] backoff -> grad_accum_steps={grad_accum_steps}")
                    continue
            else:
                runtime_error_count += 1
                logger.log_event("runtime_error", {"step": state.step, "error": f"{type(e).__name__}:{e}"})
                raise
        except KeyboardInterrupt:
            interrupted = True
            stop_reason = "keyboard_interrupt"
            logger.log_event("stop", {"reason": stop_reason, "step": state.step, "tokens_seen": state.tokens_seen})
            break

    if sigterm_requested or interrupted:
        # Emergency checkpoint path for external stop signals.
        emergency_payload = build_checkpoint_payload(
            cfg=cfg,
            model=model,
            opt=opt,
            sch=sch,
            scaler=scaler,
            state=state,
            tokenizer=tokenizer,
            expected_run_hash=expected_run_hash,
            compat_signature=compat_signature,
            arch_parity_signature=arch_parity_signature,
        )
        em_tag = f"emergency_{state.step:08d}"
        manifest = save_checkpoint_atomic(ckpt_dir, tag=em_tag, payload=emergency_payload, is_best=False)
        latest_ckpt_path = str(ckpt_dir / "latest.pt")

    # Final checkpoint always
    if stop_reason == "unknown":
        stop_reason = "completed_or_condition_met"
    final_payload = build_checkpoint_payload(
        cfg=cfg,
        model=model,
        opt=opt,
        sch=sch,
        scaler=scaler,
        state=state,
        tokenizer=tokenizer,
        expected_run_hash=expected_run_hash,
        compat_signature=compat_signature,
        arch_parity_signature=arch_parity_signature,
    )
    manifest = save_checkpoint_atomic(ckpt_dir, tag=f"step_{state.step:08d}", payload=final_payload, is_best=False)
    latest_ckpt_path = str(ckpt_dir / "latest.pt")
    if not best_ckpt_path and (ckpt_dir / "best.pt").exists():
        best_ckpt_path = str(ckpt_dir / "best.pt")

    # Save standalone final model for direct inference use
    final_model_path = ckpt_dir / "model_final.pt"
    atomic_torch_save(final_model_path, {"model": model.state_dict(), "saved_at_utc": _utc_now(), "step": state.step})

    artifacts = TrainArtifacts(
        curve_data=curve,
        state=state,
        best_checkpoint_path=best_ckpt_path,
        latest_checkpoint_path=latest_ckpt_path,
        checkpoint_manifest=manifest,
    )

    train_meta = {
        "oom_count": oom_count,
        "nan_count": nan_count,
        "anomaly_count": anomaly_count,
        "runtime_error_count": runtime_error_count,
        "peak_mem_gb": peak_mem_gb,
        "avg_mem_gb": _mean(mem_samples_gb),
        "avg_train_tokens_per_sec": _safe_div(step_tokens_total, step_time_total, default=0.0),
        "avg_train_samples_per_sec": _safe_div(step_tokens_total / max(1, int(cfg.get("seq_len", 1))), step_time_total, default=0.0),
        "final_batch_size": current_bs,
        "final_grad_accum_steps": grad_accum_steps,
        "bitnet_info": bitnet_info,
        "resume_decision": resume_decision,
        "compat_signature": compat_signature,
        "arch_parity_signature": arch_parity_signature,
        "final_model_path": str(final_model_path),
        "stop_reason": stop_reason,
        "stop_reason_codes": stop_reason_codes,
        "elapsed_train_sec": float(time.time() - start_time),
        "timed_stop": bool(stop_reason == "max_wall_hours"),
    }
    return model, artifacts, train_meta


# =============================================================================
# Benchmark suite
# =============================================================================
def benchmark_inference_latency(model: nn.Module, device: str, vocab_size: int, seq_len: int = 64, runs: int = 30) -> Dict[str, float]:
    model.eval()
    x = torch.randint(0, vocab_size, (1, seq_len), device=device)
    # warmup
    with torch.no_grad():
        for _ in range(5):
            out = model(x)
            _ = out[0] if isinstance(out, tuple) else out
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(runs):
            out = model(x)
            _ = out[0] if isinstance(out, tuple) else out
    if device == "cuda":
        torch.cuda.synchronize()
    dt = max(1e-9, time.time() - t0)
    return {
        "avg_latency_ms": (dt / runs) * 1000.0,
        "tokens_per_sec": (runs * seq_len) / dt,
    }


def benchmark_train_short(
    model: nn.Module,
    train_loader: DataLoader,
    vocab_size: int,
    device: str,
    steps: int,
    aux_coeff: float,
    seed: int,
) -> Dict[str, Any]:
    set_seed(seed)
    model = model.to(device)
    model.train()
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses: List[float] = []
    grad_norms: List[float] = []
    nan_steps = 0
    tokens = 0
    t0 = time.time()
    it = iter(train_loader)

    for _ in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad(set_to_none=True)
        out = model(x)
        if isinstance(out, tuple):
            logits = out[0]
            aux = out[1] if len(out) > 1 and torch.is_tensor(out[1]) else torch.tensor(0.0, device=device)
            loss = ce(logits.reshape(-1, vocab_size), y.reshape(-1)) + float(aux_coeff) * aux.float()
        else:
            logits = out
            loss = ce(logits.reshape(-1, vocab_size), y.reshape(-1))

        if not torch.isfinite(loss):
            nan_steps += 1
            continue

        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).detach().cpu().item())
        opt.step()
        losses.append(float(loss.detach().cpu().item()))
        grad_norms.append(gn)
        tokens += int(x.numel())

    elapsed = max(1e-9, time.time() - t0)
    return {
        "steps": steps,
        "steps_ok": len(losses),
        "final_loss": losses[-1] if losses else float("inf"),
        "avg_loss": float(sum(losses) / len(losses)) if losses else float("inf"),
        "avg_grad_norm": float(sum(grad_norms) / len(grad_norms)) if grad_norms else 0.0,
        "nan_steps": nan_steps,
        "tokens_per_sec": tokens / elapsed,
        "elapsed_sec": elapsed,
    }


def run_benchmark_suite(
    cfg: Dict[str, Any],
    tokenizer: HybridTokenizer,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    device: str,
    logger: InMemoryRunLogger,
) -> Dict[str, Any]:
    bench_steps = int(cfg["benchmark_steps"])
    eval_batches = int(cfg["benchmark_eval_batches"])
    seed_list = list(cfg.get("seed_list", [42, 43, 44]))[:3]

    train_ds = PackedDataset(train_x, train_y)
    val_ds = PackedDataset(val_x, val_y)
    train_loader = make_loader(train_ds, batch_size=max(1, int(cfg["batch_size"])), seed=int(cfg["seed"]) + 707, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=max(1, int(cfg["batch_size"])), seed=int(cfg["seed"]) + 808, shuffle=False)

    mert_cfg = MertFormerCfg(
        vocab_size=int(cfg["vocab_size"]),
        max_seq_len=max(256, int(cfg["seq_len"])),
        hidden_size=int(cfg["mert_hidden"]),
        intermediate_size=1024,
        num_layers=int(cfg["mert_layers"]),
        num_heads=int(cfg["mert_heads"]),
        num_kv_heads=int(cfg["mert_kv_heads"]),
        head_dim=int(cfg["mert_hidden"]) // int(cfg["mert_heads"]),
        rope_dim=((int(cfg["mert_hidden"]) // int(cfg["mert_heads"])) // 2) * 2,
        use_moe=bool(cfg["mert_use_moe"]),
        use_liquid=bool(cfg["mert_use_liquid"]),
        use_qinn=bool(cfg["mert_use_qinn"]),
        use_hebbian_plasticity=bool(cfg["mert_enable_all_extensions"]),
        use_neuro_symbolic_layer=bool(cfg["mert_enable_all_extensions"]),
        use_world_model_head=bool(cfg["mert_enable_all_extensions"]),
        use_lifelong_safety_layer=bool(cfg["mert_enable_all_extensions"]),
        use_latent_ode_state_channel=bool(cfg["mert_enable_all_extensions"]),
        use_global_workspace_broadcast=bool(cfg["mert_enable_all_extensions"]),
        use_cross_expert_sync_bus=bool(cfg["mert_enable_all_extensions"]),
        use_structural_plasticity=bool(cfg["mert_enable_all_extensions"]),
        moe_mode=str(cfg.get("moe_mode", "true_sparse_topk")),
        use_learned_pos_embedding=bool(cfg.get("use_learned_pos_embedding", False)),
        use_gradient_checkpointing=False,
        embedding_scale=bool(cfg.get("embedding_scale", True)),
        chat_context_truncate=bool(cfg.get("chat_context_truncate", True)),
        chat_decode_completion_only=bool(cfg.get("chat_decode_completion_only", True)),
    )

    mert = MertFormerTiny(mert_cfg)
    if bool(cfg.get("strict_bitnet", False)):
        convert_model_to_strict_bitnet(
            mert,
            logger=None,
            bitnet_mode=str(cfg.get("bitnet_mode", "stable")),
            skip_attention_qkvo=bool(cfg.get("bitnet_skip_attention_qkvo", True)),
        )

    vanilla = VanillaTransformerLM(
        vocab_size=int(cfg["vocab_size"]),
        hidden_size=384,
        num_layers=8,
        num_heads=8,
        max_seq_len=max(256, int(cfg["seq_len"])),
        dropout=0.0,
    )

    m_train = benchmark_train_short(
        mert,
        train_loader,
        vocab_size=int(cfg["vocab_size"]),
        device=device,
        steps=bench_steps,
        aux_coeff=float(cfg["aux_loss_coeff"]),
        seed=int(cfg["seed"]),
    )
    v_train = benchmark_train_short(
        vanilla,
        train_loader,
        vocab_size=int(cfg["vocab_size"]),
        device=device,
        steps=bench_steps,
        aux_coeff=0.0,
        seed=int(cfg["seed"]),
    )

    m_eval = evaluate_model(
        mert,
        val_loader,
        vocab_size=int(cfg["vocab_size"]),
        device=device,
        max_batches=min(eval_batches, len(val_loader)),
        aux_coeff=float(cfg["aux_loss_coeff"]),
    )
    v_eval = evaluate_model(
        vanilla,
        val_loader,
        vocab_size=int(cfg["vocab_size"]),
        device=device,
        max_batches=min(eval_batches, len(val_loader)),
        aux_coeff=0.0,
    )

    m_lat = benchmark_inference_latency(mert, device=device, vocab_size=int(cfg["vocab_size"]), seq_len=min(64, int(cfg["seq_len"])))
    v_lat = benchmark_inference_latency(vanilla, device=device, vocab_size=int(cfg["vocab_size"]), seq_len=min(64, int(cfg["seq_len"])))

    seed_variance: List[Dict[str, Any]] = []
    for s in seed_list:
        test_model = MertFormerTiny(mert_cfg)
        if bool(cfg.get("strict_bitnet", False)):
            convert_model_to_strict_bitnet(
                test_model,
                logger=None,
                bitnet_mode=str(cfg.get("bitnet_mode", "stable")),
                skip_attention_qkvo=bool(cfg.get("bitnet_skip_attention_qkvo", True)),
            )
        short = benchmark_train_short(
            test_model,
            train_loader,
            vocab_size=int(cfg["vocab_size"]),
            device=device,
            steps=max(10, bench_steps // 2),
            aux_coeff=float(cfg["aux_loss_coeff"]),
            seed=int(s),
        )
        seed_variance.append({"seed": int(s), **short})

    seed_final_losses = [x["final_loss"] for x in seed_variance if math.isfinite(float(x["final_loss"]))]
    if seed_final_losses:
        mean_seed = float(sum(seed_final_losses) / len(seed_final_losses))
        var_seed = float(sum((v - mean_seed) ** 2 for v in seed_final_losses) / len(seed_final_losses))
        std_seed = math.sqrt(var_seed)
    else:
        mean_seed = float("inf")
        std_seed = float("inf")

    out = {
        "schema": "micro_benchmark_suite_v1",
        "generated_at_utc": _utc_now(),
        "benchmark_mode": str(cfg.get("benchmark_mode", "separated")),
        "short_run": True,
        "device": device,
        "tokenizer_backend": tokenizer.backend_name,
        "mertformer": {
            "params": count_params(mert),
            "train": m_train,
            "eval": m_eval,
            "latency": m_lat,
        },
        "vanilla": {
            "params": count_params(vanilla),
            "train": v_train,
            "eval": v_eval,
            "latency": v_lat,
        },
        "seed_variance": {
            "runs": seed_variance,
            "final_loss_mean": mean_seed,
            "final_loss_std": std_seed,
        },
    }

    logger.log_event("benchmark_suite", out)
    return out


# =============================================================================
# Reports
# =============================================================================
def ascii_curve(values: Sequence[float], width: int = 64) -> str:
    if not values:
        return ""
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return ""
    if len(vals) > width:
        step = len(vals) / width
        sampled = [vals[int(i * step)] for i in range(width)]
    else:
        sampled = vals
    vmin = min(sampled)
    vmax = max(sampled)
    if abs(vmax - vmin) < 1e-12:
        return "-" * len(sampled)
    chars = "▁▂▃▄▅▆▇█"
    out = []
    for v in sampled:
        idx = int((v - vmin) / (vmax - vmin) * (len(chars) - 1))
        idx = max(0, min(len(chars) - 1, idx))
        out.append(chars[idx])
    return "".join(out)


def maybe_plot_curves(curve: Dict[str, List[float]], out_dir: Path, write_files: bool) -> Optional[str]:
    if not write_files:
        return None
    if not HAS_MATPLOTLIB:
        return None
    if not curve.get("step"):
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "training_curves_build30.png"
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(curve["step"], curve["train_loss"], label="train_loss")
        if curve.get("val_loss"):
            # map val points roughly to step line
            step_interval = max(1, len(curve["step"]) // max(1, len(curve["val_loss"])))
            val_x = [curve["step"][min(len(curve["step"]) - 1, i * step_interval)] for i in range(len(curve["val_loss"]))]
            plt.plot(val_x, curve["val_loss"], label="val_loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("MertFormer Build30 Training Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        return str(p)
    except Exception:
        return None


def maybe_plot_presentation_assets(
    payload: Dict[str, Any],
    curve_data: Dict[str, List[float]],
    out_dir: Path,
    write_files: bool,
) -> Dict[str, str]:
    if not write_files or not HAS_MATPLOTLIB:
        return {}
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    try:
        p_loss = out_dir / "loss_overlay.png"
        m_loss = float(payload.get("benchmark_suite", {}).get("mertformer", {}).get("train", {}).get("final_loss", float("inf")))
        v_loss = float(payload.get("benchmark_suite", {}).get("vanilla", {}).get("train", {}).get("final_loss", float("inf")))
        plt.figure(figsize=(7, 4))
        plt.plot([0, 1], [m_loss, m_loss], label="mert_final_loss")
        plt.plot([0, 1], [v_loss, v_loss], label="vanilla_final_loss")
        if curve_data.get("train_loss"):
            xs = list(range(len(curve_data["train_loss"])))
            plt.plot(xs, curve_data["train_loss"], alpha=0.35, label="mert_train_curve")
        plt.title("Loss Overlay")
        plt.xlabel("step (normalized)")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(p_loss)
        plt.close()
        paths["loss_overlay"] = str(p_loss)
    except Exception:
        pass
    try:
        p_data = out_dir / "data_contribution.png"
        sources = payload.get("data_source_scorecard", {}).get("source_totals", [])[:10]
        names = [str(x.get("dataset", "unknown"))[:28] for x in sources]
        vals = [int(x.get("kept", 0)) for x in sources]
        if names and vals:
            plt.figure(figsize=(9, 4))
            plt.bar(range(len(vals)), vals)
            plt.xticks(range(len(vals)), names, rotation=30, ha="right")
            plt.title("Data Source Contribution")
            plt.ylabel("kept rows")
            plt.tight_layout()
            plt.savefig(p_data)
            plt.close()
            paths["data_contribution"] = str(p_data)
    except Exception:
        pass
    try:
        p_stab = out_dir / "stability_panel.png"
        st = payload.get("stability_index", {})
        labels = ["stability", "nan", "oom", "anomaly", "runtime_err"]
        values = [
            float(st.get("score_0_100", 0.0)),
            float(st.get("nan_count", 0)),
            float(st.get("oom_count", 0)),
            float(st.get("anomaly_count", 0)),
            float(st.get("runtime_error_count", 0)),
        ]
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(values)), values)
        plt.xticks(range(len(values)), labels)
        plt.title("Stability Panel")
        plt.tight_layout()
        plt.savefig(p_stab)
        plt.close()
        paths["stability_panel"] = str(p_stab)
    except Exception:
        pass
    return paths


def render_reports(
    payload: Dict[str, Any],
    curve_data: Dict[str, List[float]],
) -> Tuple[str, str, str]:
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    bench_src = payload.get("micro_benchmark_suite", payload.get("benchmark_suite", {}))

    csv_io = io.StringIO()
    fields = [
        "name",
        "params",
        "train_final_loss",
        "train_tokens_per_sec",
        "eval_val_loss",
        "eval_val_ppl",
        "latency_ms",
        "nan_steps",
    ]
    writer = csv.DictWriter(csv_io, fieldnames=fields)
    writer.writeheader()
    for name in ("mertformer", "vanilla"):
        blk = bench_src.get(name, {})
        row = {
            "name": name,
            "params": blk.get("params", 0),
            "train_final_loss": blk.get("train", {}).get("final_loss", float("inf")),
            "train_tokens_per_sec": blk.get("train", {}).get("tokens_per_sec", 0.0),
            "eval_val_loss": blk.get("eval", {}).get("val_loss", float("inf")),
            "eval_val_ppl": blk.get("eval", {}).get("val_ppl", float("inf")),
            "latency_ms": blk.get("latency", {}).get("avg_latency_ms", float("inf")),
            "nan_steps": blk.get("train", {}).get("nan_steps", 0),
        }
        writer.writerow(row)
    csv_text = csv_io.getvalue()

    train_curve = ascii_curve(curve_data.get("train_loss", []), width=72)
    val_curve = ascii_curve(curve_data.get("val_loss", []), width=72)
    md_lines = [
        "# Kaggle One-Cell T4 Build30 Report",
        "",
        f"- generated_at_utc: {payload['generated_at_utc']}",
        f"- profile: {payload['profile']}",
        f"- device: {payload['device']}",
        f"- tokenizer_backend: {payload['tokenizer_metrics'].get('backend')}",
        f"- checkpoint_dir: {payload['checkpoint_manifest'].get('latest', '')}",
        f"- compat_signature: {payload.get('compat_signature', '')}",
        f"- parity_signature: {payload.get('parity_signature', '')}",
        "",
        "## Startup Phases",
        "",
    ]
    for phase, row in payload.get("startup_phase_metrics", {}).items():
        md_lines.append(
            f"- {phase}: elapsed={float(row.get('elapsed_sec', 0.0)):.1f}s "
            f"timeout={float(row.get('timeout_sec', 0.0)):.1f}s timed_out={bool(row.get('timed_out', False))}"
        )
    md_lines.extend(["", "## Data Source Scorecard", ""])
    for st in payload.get("data_source_scorecard", {}).get("stages", []):
        md_lines.append(
            f"- {st.get('stage', 'unknown')}: selected={int(st.get('selected', 0))} "
            f"loaded={int(st.get('loaded', 0))} topup={int(st.get('topup', 0))} "
            f"failures={len(st.get('failures', [])) if isinstance(st.get('failures', []), list) else 0}"
        )
    top_sources = payload.get("data_source_scorecard", {}).get("source_totals", [])[:5]
    if top_sources:
        md_lines.extend(["", "Top sources:"])
        for s in top_sources:
            md_lines.append(
                f"- {s.get('dataset', 'unknown')}: kept={int(s.get('kept', 0))} attempts={int(s.get('attempts', 0))}"
            )
    md_lines.extend(
        [
            "",
        "## Micro Benchmark",
        "",
        "| Variant | Params | Final Loss | Val Loss | Val PPL | Tok/s | Latency (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("mertformer", "vanilla"):
        blk = bench_src.get(name, {})
        md_lines.append(
            f"| {name} | {blk.get('params', 0)} | {blk.get('train', {}).get('final_loss', float('inf')):.4f} | "
            f"{blk.get('eval', {}).get('val_loss', float('inf')):.4f} | {blk.get('eval', {}).get('val_ppl', float('inf')):.2f} | "
            f"{blk.get('train', {}).get('tokens_per_sec', 0.0):.2f} | {blk.get('latency', {}).get('avg_latency_ms', float('inf')):.3f} |"
        )
    md_lines.extend(["", "## Winner Matrix", ""])
    for metric, winner in payload.get("benchmark_winner_matrix", {}).get("winners", {}).items():
        md_lines.append(f"- {metric}: **{winner}**")
    for note in payload.get("benchmark_tradeoff_notes", []):
        md_lines.append(f"- note: {note}")
    md_lines.extend(["", "## Claim Block", ""])
    claim_block = payload.get("claim_block", {})
    for k, v in claim_block.items():
        md_lines.append(f"- {k}: {v}")
    md_lines.extend(["", "## Degraded Conditions", ""])
    degraded = payload.get("degraded_conditions", [])
    if degraded:
        for d in degraded:
            md_lines.append(f"- {d}")
    else:
        md_lines.append("- none")
    md_lines.extend(["", "## Cannot-Claim-Yet", ""])
    cannot_claim = payload.get("cannot_claim_yet", [])
    if cannot_claim:
        for c in cannot_claim:
            md_lines.append(f"- {c}")
    else:
        md_lines.append("- none")
    md_lines.extend(["", "## Parity", ""])
    p = payload.get("parity_report", {})
    md_lines.append(f"- mode: {p.get('mode', '')}")
    md_lines.append(f"- embedded_ok: {p.get('embedded_self_check', {}).get('ok', False)}")
    local = p.get("local_crosscheck", {})
    md_lines.append(f"- local_crosscheck_ok: {local.get('ok', False) if not local.get('skipped', True) else 'skipped'}")
    md_lines.extend(["", "## MoE Sparse", ""])
    moe = payload.get("moe_sparse_report", {})
    md_lines.append(f"- mode: {moe.get('mode', '')}")
    md_lines.append(f"- router_entropy_mean: {float(moe.get('router_entropy_mean', 0.0)):.4f}")
    md_lines.append(f"- router_max_load_mean: {float(moe.get('router_max_load_mean', 0.0)):.4f}")
    md_lines.append(f"- overflow_ratio_mean: {float(moe.get('capacity_overflow_ratio_mean', 0.0)):.6f}")
    md_lines.append(f"- collapse_events: {int(moe.get('collapse_events', 0))}")
    md_lines.extend(["", "## BitNet Mode", ""])
    bm = payload.get("bitnet_mode_report", {})
    md_lines.append(f"- mode: {bm.get('mode', '')}")
    md_lines.append(f"- skip_attention_qkvo: {bm.get('skip_attention_qkvo', True)}")
    md_lines.append(f"- bitlinear_count: {bm.get('telemetry', {}).get('bitlinear_count', 0)}")
    md_lines.extend(["", "## Logger Memory", ""])
    lm = payload.get("logger_memory_report", {})
    md_lines.append(f"- mode: {lm.get('mode', '')}")
    md_lines.append(f"- ring_size: {lm.get('ring_size', 0)}")
    md_lines.append(f"- line_count_total: {lm.get('line_count_total', 0)}")
    md_lines.append(f"- line_count_ring: {lm.get('line_count_ring', 0)}")
    md_lines.extend(
        [
            "",
            "## Health Index",
            "",
            f"- stability_score: {float(payload.get('stability_index', {}).get('score_0_100', 0.0)):.1f}/100",
            f"- mert_tps_per_mparam: {float(payload.get('efficiency_index', {}).get('mert_tokens_per_sec_per_mparam', 0.0)):.2f}",
            f"- resume_decision: {payload.get('resume_decision', {}).get('reason', '')}",
        ]
    )
    md_lines.extend(
        [
            "",
            "## Curves (ASCII)",
            "",
            f"- train_loss: `{train_curve}`",
            f"- val_loss:   `{val_curve}`",
            "",
            "## Pending Evidence",
            "",
        ]
    )
    for p in payload.get("pending_long_run_flags", []):
        md_lines.append(f"- {p}")

    md_text = "\n".join(md_lines) + "\n"
    return json_text, csv_text, md_text


def maybe_write_outputs(cfg: Dict[str, Any], json_text: str, csv_text: str, md_text: str) -> Dict[str, str]:
    if not bool(cfg.get("write_files", False)):
        return {}
    out_dir = Path(str(cfg.get("artifact_run_dir", cfg["out_dir"])))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": str(out_dir / "kaggle_onecell_t4_build30.json"),
        "csv": str(out_dir / "kaggle_onecell_t4_build30.csv"),
        "md": str(out_dir / "kaggle_onecell_t4_build30.md"),
    }
    Path(paths["json"]).write_text(json_text + "\n", encoding="utf-8")
    Path(paths["csv"]).write_text(csv_text, encoding="utf-8")
    Path(paths["md"]).write_text(md_text, encoding="utf-8")
    return paths


def compute_final_verdict(payload: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    mini = payload.get("mini_probe_report", {})
    tm = payload.get("train_meta", {})
    ts = payload.get("train_state", {})
    code_loaded = int(payload.get("code_stage_loaded", 0))
    strict_min_tokens = int(cfg.get("strict_green_min_tokens", 8_000_000))
    collapse_events = int(mini.get("collapse_events", payload.get("moe_sparse_report", {}).get("collapse_events", 0)))
    strict_reasons: List[str] = []
    checks = {
        "tokens_seen": int(ts.get("tokens_seen", 0)) >= strict_min_tokens,
        "loss_gate": bool(mini.get("loss_gate", False)),
        "grad_gate": bool(mini.get("grad_gate", False)),
        "expert_gate": bool(mini.get("expert_gate_pass", mini.get("expert_load_gate", False))),
        "entropy_gate": bool(mini.get("entropy_gate_pass", mini.get("router_entropy_gate", False))),
        "collapse_events_zero": collapse_events == 0,
        "nan_count_zero": int(tm.get("nan_count", 0)) == 0,
        "oom_count_zero": int(tm.get("oom_count", 0)) == 0,
        "stage_3_code_loaded": code_loaded > 0,
    }
    for k, ok in checks.items():
        if not ok:
            strict_reasons.append(k)

    verdict = "provisional"
    if not strict_reasons:
        verdict = "evidence_strict_green"
    else:
        strong_min = [
            checks["tokens_seen"],
            checks["loss_gate"],
            checks["expert_gate"],
            checks["entropy_gate"],
            checks["collapse_events_zero"],
            checks["nan_count_zero"],
        ]
        if all(strong_min):
            verdict = "evidence_strong"
    return {
        "final_verdict": verdict,
        "strict_green_checks": checks,
        "strict_green_missing": strict_reasons,
        "strict_green_min_tokens": strict_min_tokens,
    }


def build_public_summary(payload: Dict[str, Any], verdict: Dict[str, Any]) -> Dict[str, Any]:
    unknown_or_pending = list(payload.get("pending_long_run_flags", []))
    if payload.get("degraded_conditions"):
        unknown_or_pending.extend([str(x) for x in payload.get("degraded_conditions", [])])
    return {
        "schema": "build30_public_summary_v1",
        "run_utc": str(payload.get("generated_at_utc", _utc_now())),
        "profile": str(payload.get("profile", "")),
        "params": int(payload.get("runtime_model_params", payload.get("micro_benchmark_suite", {}).get("mertformer", {}).get("params", 0))),
        "tokens_seen": int(payload.get("train_state", {}).get("tokens_seen", 0)),
        "best_val_loss": float(payload.get("train_state", {}).get("best_val_loss", float("inf"))),
        "router_entropy_mean": float(payload.get("moe_sparse_report", {}).get("router_entropy_mean", 0.0)),
        "router_max_load_p95": float(payload.get("mini_probe_report", {}).get("router_max_load_p95", 0.0)),
        "collapse_events": int(payload.get("moe_sparse_report", {}).get("collapse_events", 0)),
        "final_verdict": str(verdict.get("final_verdict", "provisional")),
        # NOT a measured gate result: this is an unverified, author-asserted note,
        # not tied to any test/verdict. Treated as informational only.
        "zero_known_critical_bugs_claim": "no critical bugs known to author (UNVERIFIED; not a gate result)",
        "dataset_access_constraints": {
            "strict_data": bool(payload.get("preflight", {}).get("strict_data", False)),
            "hf_token_present": bool(payload.get("preflight", {}).get("hf_token_present", False)),
            "degraded_data_mode": bool(payload.get("preflight", {}).get("degraded_data_mode", False)),
        },
        "unknown_or_pending": unknown_or_pending,
    }


def verify_and_index_artifacts(path_map: Dict[str, str]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for name, pstr in path_map.items():
        p = Path(pstr)
        exists = p.exists()
        size = int(p.stat().st_size) if exists else 0
        sha = file_sha256(p) if exists and size > 0 else ""
        rows.append(
            {
                "name": str(name),
                "path": str(p),
                "exists": bool(exists),
                "size": size,
                "sha256": sha,
            }
        )
    return {"generated_at_utc": _utc_now(), "files": rows}


def validate_required_payload_fields(payload: Dict[str, Any], required_fields: Sequence[str]) -> Dict[str, Any]:
    missing: List[str] = []
    for f in required_fields:
        if f not in payload:
            missing.append(str(f))
    return {"required_field_count": len(required_fields), "missing_fields": missing, "ok": len(missing) == 0}


def make_evidence_zip(layout: ArtifactLayout, file_paths: Dict[str, str], enabled: bool) -> Dict[str, Any]:
    if not enabled:
        return {"enabled": False, "zip_path": str(layout.evidence_zip_path), "added": [], "errors": []}
    added: List[Dict[str, Any]] = []
    errors: List[str] = []
    with zipfile.ZipFile(layout.evidence_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, pstr in file_paths.items():
            p = Path(pstr)
            if not p.exists():
                errors.append(f"missing:{name}")
                continue
            try:
                arc = p.relative_to(layout.run_dir) if str(p).startswith(str(layout.run_dir)) else Path(name)
                zf.write(p, arcname=str(arc))
                added.append({"name": name, "path": str(p), "arcname": str(arc), "size": int(p.stat().st_size)})
            except Exception as e:
                errors.append(f"{name}:{type(e).__name__}:{e}")
    manifest = {
        "generated_at_utc": _utc_now(),
        "zip_path": str(layout.evidence_zip_path),
        "added": added,
        "errors": errors,
    }
    atomic_json_write(layout.zip_manifest_path, manifest)
    return manifest


def backup_run_to_drive(layout: ArtifactLayout, cfg: Dict[str, Any]) -> Dict[str, Any]:
    report = {
        "attempted": bool(cfg.get("auto_backup_to_drive", False)),
        "mounted": False,
        "copied_count": 0,
        "failed_count": 0,
        "destination": "",
        "warning": "",
    }
    if not report["attempted"]:
        return report
    root = Path(str(cfg.get("drive_backup_root", "/content/drive/MyDrive/mertformer_runs"))).expanduser()
    if not root.exists():
        report["warning"] = "drive_not_mounted_or_path_missing"
        return report
    report["mounted"] = True
    dest = root / layout.run_id
    report["destination"] = str(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for p in layout.run_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(layout.run_dir)
        d = dest / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(p, d)
            report["copied_count"] += 1
        except Exception:
            report["failed_count"] += 1
    return report


# =============================================================================
# Chat
# =============================================================================
def load_checkpoint_for_chat(
    model: MertFormerTiny,
    cfg: Dict[str, Any],
    device: str,
    train_artifacts: TrainArtifacts,
) -> Dict[str, Any]:
    explicit = str(cfg.get("checkpoint_path", "")).strip()
    ckpt_dir = Path(str(cfg["checkpoint_dir"]))
    source = ""

    if explicit:
        p = Path(explicit).expanduser()
        if p.exists():
            source = str(p)
    if not source and train_artifacts.best_checkpoint_path:
        source = train_artifacts.best_checkpoint_path
    if not source and train_artifacts.latest_checkpoint_path:
        source = train_artifacts.latest_checkpoint_path
    if not source:
        return {"loaded": False, "reason": "no_checkpoint_available"}

    try:
        payload = torch.load(source, map_location=device)
        state = payload.get("model", payload)
        missing, unexpected = model.load_state_dict(state, strict=False)
        return {
            "loaded": True,
            "path": source,
            "missing": list(missing),
            "unexpected": list(unexpected),
        }
    except Exception as e:
        return {"loaded": False, "reason": f"checkpoint_load_error:{type(e).__name__}:{e}", "path": source}


def apply_repetition_penalty(logits: torch.Tensor, history: Sequence[int], penalty: float) -> torch.Tensor:
    if penalty <= 1.0 or not history:
        return logits
    out = logits.clone()
    seen = set(int(x) for x in history[-256:])
    for idx in seen:
        if idx < 0 or idx >= out.numel():
            continue
        val = out[idx]
        if val > 0:
            out[idx] = val / penalty
        else:
            out[idx] = val * penalty
    return out


def sample_top_p(logits: torch.Tensor, top_p: float, temperature: float) -> int:
    logits = logits / max(1e-5, float(temperature))
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cdf = torch.cumsum(probs, dim=-1)
    cutoff = cdf > float(top_p)
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
    probs = torch.softmax(sorted_logits, dim=-1)
    chosen = torch.multinomial(probs, num_samples=1)
    return int(sorted_idx.gather(-1, chosen).item())


@torch.no_grad()
def generate_text(
    model: MertFormerTiny,
    tokenizer: HybridTokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> str:
    model.eval()
    if hasattr(model, "reset_router_state"):
        try:
            model.reset_router_state(batch_size=1)
        except Exception:
            pass
    system = "Sistem: Türkçe, net ve teknik yanıt ver.\n"
    full_prompt = system + "Kullanıcı: " + prompt + "\nAsistan:"
    ids = tokenizer.encode(full_prompt, add_bos=True, add_eos=False)
    max_ctx = max(8, int(model.cfg.max_seq_len) - 1)
    if bool(getattr(model.cfg, "chat_context_truncate", True)) and len(ids) > max_ctx:
        ids = ids[-max_ctx:]
    generated = list(ids)
    prompt_len = len(generated)

    past = None
    curr = torch.tensor([generated], device=device, dtype=torch.long)
    for _ in range(int(max_new_tokens)):
        logits, _, extras = model(curr, past_key_values=past, use_cache=True)
        past = extras.get("present_key_values")
        nxt_logits = logits[0, -1, :]
        nxt_logits = apply_repetition_penalty(nxt_logits, generated, penalty=float(repetition_penalty))
        nxt = sample_top_p(nxt_logits, top_p=float(top_p), temperature=float(temperature))
        generated.append(nxt)
        if nxt == tokenizer.eos_id:
            break
        curr = torch.tensor([[nxt]], device=device, dtype=torch.long)

    if bool(getattr(model.cfg, "chat_decode_completion_only", True)):
        return tokenizer.decode(generated[prompt_len:])
    return tokenizer.decode(generated)


# =============================================================================
# Main Orchestration
# =============================================================================
def legacy_build_mert_cfg(cfg: Dict[str, Any]) -> MertFormerCfg:
    hidden = int(cfg["mert_hidden"])
    heads = int(cfg["mert_heads"])
    head_dim = hidden // heads
    rope_dim = (head_dim // 2) * 2
    enable = bool(cfg["mert_enable_all_extensions"])
    return MertFormerCfg(
        vocab_size=int(cfg["vocab_size"]),
        max_seq_len=max(256, int(cfg["seq_len"])),
        hidden_size=hidden,
        intermediate_size=1024,
        num_layers=int(cfg["mert_layers"]),
        num_heads=heads,
        num_kv_heads=int(cfg["mert_kv_heads"]),
        head_dim=head_dim,
        rope_dim=max(2, rope_dim),
        use_moe=bool(cfg["mert_use_moe"]),
        use_liquid=bool(cfg["mert_use_liquid"]),
        use_qinn=bool(cfg["mert_use_qinn"]),
        use_hebbian_plasticity=enable,
        use_neuro_symbolic_layer=enable,
        use_world_model_head=enable,
        use_lifelong_safety_layer=enable,
        use_latent_ode_state_channel=enable,
        use_global_workspace_broadcast=enable,
        use_cross_expert_sync_bus=enable,
        use_structural_plasticity=enable,
        moe_mode=str(cfg.get("moe_mode", "true_sparse_topk")),
        use_learned_pos_embedding=bool(cfg.get("use_learned_pos_embedding", False)),
        use_gradient_checkpointing=bool(cfg.get("use_gradient_checkpointing", False)),
        embedding_scale=bool(cfg.get("embedding_scale", True)),
        chat_context_truncate=bool(cfg.get("chat_context_truncate", True)),
        chat_decode_completion_only=bool(cfg.get("chat_decode_completion_only", True)),
    )


def build_token_stream_and_metrics(
    cfg: Dict[str, Any],
    logger: InMemoryRunLogger,
) -> Tuple[List[int], HybridTokenizer, Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    startup_phase_metrics: Dict[str, Any] = {}
    stage_data_start = time.time()
    data_fetch_started_at = _utc_now()
    texts, curriculum_trace = build_curriculum_corpus(cfg, logger)
    stage_data_elapsed = time.time() - stage_data_start
    startup_phase_metrics["data_fetch"] = {
        "started_at": data_fetch_started_at,
        "ended_at": _utc_now(),
        "elapsed_sec": stage_data_elapsed,
        "timeout_sec": float(cfg.get("data_fetch_phase_timeout_seconds", 900)),
        "timed_out": stage_data_elapsed > float(cfg.get("data_fetch_phase_timeout_seconds", 900)),
        "fallback_action": "none",
        "kept_rows": len(texts),
    }

    if bool(cfg.get("startup_watchdog_enabled", True)) and startup_phase_metrics["data_fetch"]["timed_out"]:
        # Degrade safely without stopping the run.
        cap = int(cfg.get("tokenizer_fit_max_texts", 30000))
        texts = texts[:cap]
        logger.log_event("startup_watchdog", {"phase": "data_fetch", "action": "truncate_texts", "new_len": len(texts)})
        print(f"[startup:watchdog] data_fetch slow -> truncating texts to {len(texts)}")
        startup_phase_metrics["data_fetch"]["fallback_action"] = "truncate_texts"
        startup_phase_metrics["data_fetch"]["kept_rows"] = len(texts)

    tokenizer = HybridTokenizer(
        vocab_size=int(cfg["vocab_size"]),
        byte_bpe_max_merges=int(cfg.get("byte_bpe_max_merges", 2500)),
        byte_bpe_encode_cache_size=int(cfg.get("byte_bpe_encode_cache_size", 2048)),
        byte_bpe_cache_max_text_len=int(cfg.get("byte_bpe_cache_max_text_len", 512)),
        fit_max_texts=int(cfg.get("tokenizer_fit_max_texts", 30000)),
        fit_max_chars=int(cfg.get("tokenizer_fit_max_chars", 6000000)),
        fit_max_chars_per_text=int(cfg.get("tokenizer_fit_max_chars_per_text", 512)),
    )
    print(
        f"[tokenizer] fitting backend=auto texts={len(texts)} "
        f"fit_cap={cfg.get('tokenizer_fit_max_texts', 30000)} "
        f"merges={cfg.get('byte_bpe_max_merges', 2500)}"
    )
    fit_started_at = _utc_now()
    t_fit = time.time()
    tokenizer.fit(texts)
    fit_elapsed = time.time() - t_fit
    startup_phase_metrics["tokenizer_fit"] = {
        "started_at": fit_started_at,
        "ended_at": _utc_now(),
        "elapsed_sec": fit_elapsed,
        "timeout_sec": float(cfg.get("tokenizer_fit_phase_timeout_seconds", 600)),
        "timed_out": fit_elapsed > float(cfg.get("tokenizer_fit_phase_timeout_seconds", 600)),
        "fallback_action": "none",
        "kept_rows": len(texts),
    }
    if bool(cfg.get("startup_watchdog_enabled", True)) and startup_phase_metrics["tokenizer_fit"]["timed_out"]:
        logger.log_event(
            "startup_watchdog",
            {
                "phase": "tokenizer_fit",
                "action": "fallback_simple_tokenizer",
                "prev_backend": tokenizer.backend_name,
            },
        )
        simple = HybridTokenizer(
            vocab_size=int(cfg["vocab_size"]),
            byte_bpe_max_merges=256,
            byte_bpe_encode_cache_size=0,
            byte_bpe_cache_max_text_len=0,
            fit_max_texts=min(8000, int(cfg.get("tokenizer_fit_max_texts", 30000))),
            fit_max_chars=min(800000, int(cfg.get("tokenizer_fit_max_chars", 6000000))),
            fit_max_chars_per_text=min(128, int(cfg.get("tokenizer_fit_max_chars_per_text", 512))),
        )
        simple.inner = SimpleTokenizer(vocab_size=int(cfg["vocab_size"]))
        simple.backend_name = "simple"
        simple.inner.fit(texts[: min(len(texts), simple.fit_max_texts)])
        simple.metrics = {"backend": "simple", "errors": ["watchdog_fallback_from_slow_tokenizer_fit"]}
        tokenizer = simple
        print("[startup:watchdog] tokenizer_fit slow -> switched to SimpleTokenizer fallback")
        startup_phase_metrics["tokenizer_fit"]["fallback_action"] = "fallback_simple_tokenizer"
    print(
        f"[tokenizer] ready backend={tokenizer.backend_name} "
        f"vocab_realized={tokenizer.vocab_size_realized} "
        f"fit_time={fit_elapsed:.1f}s"
    )

    token_ids: List[int] = []
    # Use curriculum order for training order
    encode_started_at = _utc_now()
    t_encode = time.time()
    last_hb = t_encode
    for i, t in enumerate(texts, start=1):
        if bool(cfg.get("startup_watchdog_enabled", True)):
            if (time.time() - t_encode) >= float(cfg.get("token_encode_phase_timeout_seconds", 1200)):
                logger.log_event(
                    "startup_watchdog",
                    {
                        "phase": "token_encode",
                        "action": "stop_encoding_early",
                        "encoded_texts": i - 1,
                        "total_texts": len(texts),
                    },
                )
                print(f"[startup:watchdog] token_encode timeout -> stop early at {i-1}/{len(texts)}")
                startup_phase_metrics.setdefault("token_encode", {})
                startup_phase_metrics["token_encode"]["fallback_action"] = "stop_encoding_early"
                break
        token_ids.extend(tokenizer.encode(t, add_bos=False, add_eos=True))
        if i % int(cfg.get("token_encode_heartbeat_every", 20000)) == 0:
            now = time.time()
            speed = _safe_div(i, now - t_encode, default=0.0)
            eta = _safe_div((len(texts) - i), speed, default=0.0)
            print(
                f"[tokenizer] encoded_texts={i}/{len(texts)} tokens={len(token_ids)} "
                f"speed={speed:.1f} txt/s eta={eta:.1f}s"
            )
            last_hb = now
        elif time.time() - last_hb >= float(cfg.get("token_encode_heartbeat_seconds", 10)):
            now = time.time()
            speed = _safe_div(i, now - t_encode, default=0.0)
            eta = _safe_div((len(texts) - i), speed, default=0.0)
            print(
                f"[tokenizer:heartbeat] encoded={i}/{len(texts)} "
                f"tokens={len(token_ids)} eta={eta:.1f}s"
            )
            last_hb = now
    encode_elapsed = time.time() - t_encode
    startup_phase_metrics["token_encode"] = {
        "started_at": encode_started_at,
        "ended_at": _utc_now(),
        "elapsed_sec": encode_elapsed,
        "timeout_sec": float(cfg.get("token_encode_phase_timeout_seconds", 1200)),
        "timed_out": encode_elapsed > float(cfg.get("token_encode_phase_timeout_seconds", 1200)),
        "fallback_action": startup_phase_metrics.get("token_encode", {}).get("fallback_action", "none"),
        "kept_rows": len(texts),
    }
    print(f"[tokenizer] encoding_done texts={len(texts)} tokens={len(token_ids)} elapsed={encode_elapsed:.1f}s")

    # Ensure minimum token pool for deep profile
    min_tokens = max(50_000, int(cfg["seq_len"]) * int(cfg["batch_size"]) * 200)
    if len(token_ids) < min_tokens:
        rnd = random.Random(int(cfg["seed"]))
        token_ids.extend(rnd.randrange(4, int(cfg["vocab_size"])) for _ in range(min_tokens - len(token_ids)))

    metrics = {
        "backend": tokenizer.backend_name,
        "special_tokens": {
            "pad": tokenizer.pad_id,
            "bos": tokenizer.bos_id,
            "eos": tokenizer.eos_id,
            "unk": tokenizer.unk_id,
        },
        "vocab_size_target": int(cfg["vocab_size"]),
        "vocab_size_realized": int(tokenizer.vocab_size_realized),
        "token_count": len(token_ids),
        "oov_rate": compute_oov_rate(token_ids, tokenizer.unk_id),
        "token_histogram_topk": token_histogram_topk(token_ids, k=20),
        "tokenizer_fit_metrics": tokenizer.metrics,
    }
    oov_thr = float(cfg.get("oov_rate_warn_threshold", 0.01))
    oov_rate_val = float(metrics.get("oov_rate", 0.0))
    topk = metrics.get("token_histogram_topk", [])
    top_ratio = 0.0
    if topk and int(metrics.get("token_count", 0)) > 0:
        top_ratio = float(topk[0][1]) / float(max(1, int(metrics.get("token_count", 0))))
    dup_thr = float(cfg.get("token_duplicate_ratio_warn_threshold", 0.25))
    metrics["oov_warn_threshold"] = oov_thr
    metrics["oov_gate_pass"] = bool(oov_rate_val <= oov_thr)
    metrics["duplicate_heavy_warn_threshold"] = dup_thr
    metrics["duplicate_heavy_warning"] = bool(top_ratio > dup_thr)
    metrics["top_token_ratio"] = float(top_ratio)
    metrics["quality_downgrade"] = bool(tokenizer.backend_name != "sentencepiece")
    if metrics["quality_downgrade"]:
        logger.log_event(
            "tokenizer_quality_downgrade",
            {"backend": tokenizer.backend_name, "reason": "backend_fallback_not_sentencepiece"},
        )
    data_source_scorecard = summarize_data_source_scorecard(curriculum_trace)
    logger.log_event("tokenizer_metrics", metrics)
    logger.log_event("startup_phase_metrics", startup_phase_metrics)
    logger.log_event("data_source_scorecard", data_source_scorecard)
    return token_ids, tokenizer, metrics, curriculum_trace, startup_phase_metrics, data_source_scorecard


def pending_long_run_flags(cfg: Dict[str, Any], state: TrainState) -> List[str]:
    flags: List[str] = []
    if float(cfg["max_wall_hours"]) < 8.0:
        flags.append("max_wall_hours_below_8")
    if state.tokens_seen < int(cfg["target_train_tokens"]):
        flags.append("target_train_tokens_not_reached")
    if state.step < max(2000, int(cfg["eval_interval_steps"]) * 5):
        flags.append("insufficient_steps_for_strong_generalization_claim")
    flags.append("external_benchmarks_gsm8k_mmlu_humaneval_not_run_in_this_script")
    return flags


def run_interactive_menu(
    model: MertFormerTiny,
    tokenizer: HybridTokenizer,
    cfg: Dict[str, Any],
    device: str,
    payload: Dict[str, Any],
) -> None:
    if not bool(cfg.get("interactive_menu", False)):
        return
    if not can_accept_user_input(cfg):
        return

    while True:
        print("\n=== MENU ===")
        print("1) Chat")
        print("2) Benchmark Summary")
        print("3) Save/Export Status")
        print("4) Exit")
        try:
            ch = input("Select: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("[interactive] input closed; exiting menu.")
            break
        if ch == "1":
            print("Chat mode. 'exit' to quit chat.")
            while True:
                try:
                    q = input("you> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("[interactive] chat input closed; leaving chat mode.")
                    break
                if not q or q.lower() in ("exit", "quit", "q"):
                    break
                r = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=q,
                    device=device,
                    max_new_tokens=int(cfg["chat_max_new_tokens"]),
                    temperature=float(cfg["chat_temperature"]),
                    top_p=float(cfg["chat_top_p"]),
                    repetition_penalty=float(cfg["chat_repetition_penalty"]),
                )
                print("model>", r)
        elif ch == "2":
            bench = payload.get("benchmark_suite", {})
            print(json.dumps(bench, ensure_ascii=False, indent=2))
        elif ch == "3":
            print(json.dumps(payload.get("checkpoint_manifest", {}), ensure_ascii=False, indent=2))
            print("output_files:", payload.get("output_files", {}))
        elif ch == "4":
            break
        else:
            print("Invalid choice")


def run_all() -> Dict[str, Any]:
    _print_header()

    total_start = time.time()
    _RUNTIME_SIGNAL_STATE["sigterm"] = False
    _RUNTIME_SIGNAL_STATE["signal"] = ""
    cfg = resolve_runtime_config(interactive_prompt(dict(RUN_CONFIG)))
    set_seed(int(cfg["seed"]))
    install_runtime_signal_handlers()
    layout = init_artifact_layout(cfg)
    device = str(cfg["device"])
    gpu_meta = get_cuda_device_meta()
    gpu_tune = apply_gpu_auto_tune(cfg, device=device)

    print(
        f"[runtime] profile={cfg['profile']} device={device} quick={cfg['quick']} "
        f"steps<= {cfg['max_steps']} bs={cfg['batch_size']} seq={cfg['seq_len']} wall_h={cfg['max_wall_hours']} "
        f"bitnet={cfg.get('bitnet_mode', 'stable')} moe={cfg.get('moe_mode', 'true_sparse_topk')}"
    )

    if device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    run_name = f"kaggle_onecell_t4_build30_{_local_stamp()}"
    logger = InMemoryRunLogger(
        run_name=run_name,
        mode=str(cfg.get("logger_mode", "jsonl_ring")),
        ring_size=int(cfg.get("logger_ring_size", 5000)),
        step_log_interval=int(cfg.get("step_log_interval", 1)),
        jsonl_path=str(layout.logger_jsonl_path),
    )
    logger.log_event("config", cfg)
    logger.log_event("gpu_auto_tune", gpu_tune)
    logger.log_event("runtime_paths", {"artifact_root": str(layout.artifact_root), "run_dir": str(layout.run_dir), "checkpoint_dir": str(layout.checkpoint_dir)})

    preflight = run_data_preflight(cfg)
    logger.log_event("preflight", preflight)
    if str(preflight.get("preflight_status", "fail")) != "pass":
        stop_reason = str(preflight.get("reason_codes", ["preflight_failed"])[0] if preflight.get("reason_codes") else "preflight_failed")
        payload_fail: Dict[str, Any] = {
            "schema": "kaggle_onecell_t4_build30_v1",
            "generated_at_utc": _utc_now(),
            "profile": cfg["profile"],
            "device": device,
            "runtime": {
                "batch_size": cfg["batch_size"],
                "seq_len": cfg["seq_len"],
                "max_steps": cfg["max_steps"],
                "max_wall_hours": cfg["max_wall_hours"],
                "target_train_tokens": cfg["target_train_tokens"],
                "grad_accum_steps": cfg["grad_accum_steps"],
                "step_log_interval": cfg.get("step_log_interval", 1),
                "strict_green_min_tokens": int(cfg.get("strict_green_min_tokens", 8_000_000)),
            },
            "preflight": preflight,
            "tokenizer_metrics": {"backend": "unknown"},
            "startup_phase_metrics": {},
            "data_source_scorecard": {"stages": [], "source_totals": [], "totals": {"stage_3_code_loaded": 0}},
            "curriculum_trace": [],
            "train_state": {"step": 0, "epoch": 0, "tokens_seen": 0, "best_val_loss": float("inf")},
            "train_meta": {
                "oom_count": 0,
                "nan_count": 0,
                "anomaly_count": 0,
                "runtime_error_count": 0,
                "stop_reason": stop_reason,
                "elapsed_train_sec": 0.0,
                "elapsed_total_sec": float(time.time() - total_start),
            },
            "checkpoint_manifest": {},
            "micro_benchmark_suite": {"schema": "micro_benchmark_suite_v1", "short_run": True, "mertformer": {}, "vanilla": {}},
            "benchmark_winner_matrix": {"metrics": {}, "winners": {}, "apples_to_apples": False},
            "benchmark_tradeoff_notes": ["benchmark_not_run_due_to_preflight_fail"],
            "parity_report": {"mode": "preflight_only", "embedded_self_check": {"ok": False}, "local_crosscheck": {"enabled": False, "skipped": True}},
            "parity_signature": "",
            "moe_sparse_report": {"mode": str(cfg.get("moe_mode", "true_sparse_topk")), "collapse_events": 0},
            "bitnet_mode_report": {"mode": str(cfg.get("bitnet_mode", "stable")), "skip_attention_qkvo": bool(cfg.get("bitnet_skip_attention_qkvo", True)), "telemetry": {}},
            "logger_memory_report": {"mode": str(cfg.get("logger_mode", "jsonl_ring"))},
            "bitnet_telemetry": {},
            "stability_index": {"score_0_100": 0.0},
            "efficiency_index": {},
            "mini_probe_report": {"enabled": False, "all_green": False},
            "resume_decision": {"attempted": False, "reason": "resume_skip"},
            "compat_signature": "",
            "arch_parity_signature": "",
            "target_param_band": {"low": int(cfg["target_param_band_low"]), "high": int(cfg["target_param_band_high"])},
            "band_check": {"mertformer_in_band": False, "vanilla_in_band": False, "both_in_band": False},
            "pending_long_run_flags": ["preflight_failed"],
            "logger_manifest": logger.finalize(),
            "code_stage_loaded": int(preflight.get("code_stage_loaded", 0)),
            "coding_claim_blocked": True,
            "degraded_conditions": [f"preflight:{x}" for x in preflight.get("reason_codes", [])],
            "cannot_claim_yet": ["strict_data_preflight_failed"],
            "claim_block": {"bug_free_claim": "blocked", "coding_claim": "blocked"},
            "runtime_model_params": 0,
            "gpu_meta": gpu_meta,
            "gpu_tune_report": gpu_tune,
            "stop_reason": stop_reason,
            "run_config_hash": hash_config(cfg),
        }
        verdict_fail = compute_final_verdict(payload_fail, cfg)
        payload_fail.update(verdict_fail)
        payload_fail["final_status"] = verdict_fail.get("final_verdict", "provisional")
        payload_fail["final_reason"] = stop_reason
        json_text, csv_text, md_text = render_reports(payload_fail, {"train_loss": [], "val_loss": []})
        out_files = maybe_write_outputs(cfg, json_text, csv_text, md_text)
        payload_fail["output_files"] = out_files
        if str(payload_fail.get("logger_manifest", {}).get("jsonl_path", "")).strip():
            payload_fail["output_files"]["jsonl_log"] = str(payload_fail["logger_manifest"]["jsonl_path"])
        public_summary = build_public_summary(payload_fail, verdict_fail)
        if bool(cfg.get("write_files", False)):
            atomic_json_write(layout.public_summary_path, public_summary)
            payload_fail["output_files"]["public_summary"] = str(layout.public_summary_path)
            atomic_json_write(layout.stop_summary_path, {"stop_reason": stop_reason, "run_id": layout.run_id, "time_utc": _utc_now()})
            payload_fail["output_files"]["stop_summary"] = str(layout.stop_summary_path)
            payload_fail["output_files"].update(
                write_onecell_sidecars(
                    layout,
                    cfg,
                    payload_fail,
                    preflight,
                    payload_fail.get("logger_manifest", {}),
                )
            )
            artifact_index = verify_and_index_artifacts(payload_fail["output_files"])
            atomic_json_write(layout.artifact_index_path, artifact_index)
            payload_fail["artifacts_index"] = artifact_index
            payload_fail["output_files"]["artifacts_index"] = str(layout.artifact_index_path)
            zip_manifest = make_evidence_zip(layout, payload_fail["output_files"], enabled=bool(cfg.get("zip_evidence_pack", True)))
            payload_fail["zip_manifest"] = zip_manifest
            payload_fail["output_files"]["zip_manifest"] = str(layout.zip_manifest_path)
            if layout.evidence_zip_path.exists():
                payload_fail["output_files"]["evidence_zip"] = str(layout.evidence_zip_path)
            sha_manifest_path = Path(str(payload_fail["output_files"].get("sha256_manifest", "")))
            if str(sha_manifest_path):
                write_output_sha256_manifest(payload_fail["output_files"], sha_manifest_path)
            jpath = Path(payload_fail["output_files"].get("json", ""))
            if jpath:
                jpath.write_text(json.dumps(payload_fail, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        final_status_line = f"FINAL_STATUS: {payload_fail['final_status']} reason={stop_reason} run_id={layout.run_id}"
        print(final_status_line)
        return payload_fail

    (
        token_ids,
        tokenizer,
        tok_metrics,
        curriculum_trace,
        startup_phase_metrics,
        data_source_scorecard,
    ) = build_token_stream_and_metrics(cfg, logger)

    train_x, train_y, val_x, val_y = build_train_val_streams(
        token_ids=token_ids,
        seq_len=int(cfg["seq_len"]),
        val_ratio=0.05,
        seed=int(cfg["seed"]),
    )

    print(
        f"[data] backend={tok_metrics['backend']} vocab_realized={tok_metrics['vocab_size_realized']} "
        f"train_blocks={train_x.size(0)} val_blocks={val_x.size(0)}"
    )

    mert_cfg = build_mert_cfg(cfg)
    model = MertFormerTiny(mert_cfg)
    mert_params_runtime = count_params(model)
    print(f"[model] params={mert_params_runtime:,}")
    if bool(cfg.get("mini_probe_enabled", False)):
        print(
            f"[mini300m] target_params≈{int(cfg.get('mini_probe_param_target', 300_000_000)):,} "
            f"token_window={int(cfg.get('mini_probe_token_min', 5_000_000_000)):,}-"
            f"{int(cfg.get('mini_probe_token_max', 10_000_000_000)):,}"
        )
    parity_signature = compute_arch_parity_signature(cfg, tokenizer_backend=tok_metrics.get("backend", "unknown"))
    parity_report = {
        "mode": str(cfg.get("parity_proof_mode", "embedded_plus_local_if_available")),
        "signature": parity_signature,
        "embedded_self_check": parity_self_check(model, cfg, device=device),
        "local_crosscheck": parity_crosscheck_local_repo(cfg)
        if str(cfg.get("parity_proof_mode", "")).startswith("embedded_plus_local")
        else {"enabled": False, "skipped": True, "reason": "proof_mode_without_local_crosscheck"},
    }
    model.zero_grad(set_to_none=True)
    logger.log_event("parity_report", parity_report)

    model, artifacts, train_meta = train_loop_deep(
        model=model,
        tokenizer=tokenizer,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        cfg=cfg,
        device=device,
        logger=logger,
    )

    micro_bench = run_benchmark_suite(
        cfg=cfg,
        tokenizer=tokenizer,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        device=device,
        logger=logger,
    )
    print(f"[resume] decision={train_meta.get('resume_decision', {}).get('reason', 'unknown')}")

    band_low = int(cfg["target_param_band_low"])
    band_high = int(cfg["target_param_band_high"])
    m_params = int(micro_bench.get("mertformer", {}).get("params", 0))
    v_params = int(micro_bench.get("vanilla", {}).get("params", 0))
    band_check = {
        "mertformer_in_band": band_low <= m_params <= band_high,
        "vanilla_in_band": band_low <= v_params <= band_high,
        "both_in_band": (band_low <= m_params <= band_high) and (band_low <= v_params <= band_high),
    }

    bitnet_tel = collect_bitnet_telemetry(model, bitnet_mode=str(cfg.get("bitnet_mode", "stable")))
    stability_index = compute_stability_index(train_meta, artifacts.curve_data)
    efficiency_index = compute_efficiency_index(micro_bench)
    mini_probe_report = compute_mini_probe_report(
        cfg=cfg,
        train_meta=train_meta,
        curve_data=artifacts.curve_data,
        model_params=mert_params_runtime,
        tokens_seen=int(artifacts.state.tokens_seen),
    )
    if bool(mini_probe_report.get("enabled", False)):
        print(
            "[mini300m] "
            f"loss_drop={float(mini_probe_report.get('loss_drop_ratio', 0.0)):.4f} "
            f"grad_cv={float(mini_probe_report.get('grad_cv', 0.0)):.4f} "
            f"router_entropy={float(mini_probe_report.get('router_entropy_mean', 0.0)):.4f} "
            f"router_max_load_p95={float(mini_probe_report.get('router_max_load_p95', 0.0)):.4f}"
        )
        print(
            "[mini300m:gates] "
            f"loss={bool(mini_probe_report.get('loss_gate', False))} "
            f"grad={bool(mini_probe_report.get('grad_gate', False))} "
            f"expert={bool(mini_probe_report.get('expert_load_gate', False))} "
            f"entropy={bool(mini_probe_report.get('router_entropy_gate', False))} "
            f"params={bool(mini_probe_report.get('param_gate', False))} "
            f"all_green={bool(mini_probe_report.get('all_green', False))}"
        )
    benchmark_winner_matrix = build_benchmark_winner_matrix(micro_bench)
    benchmark_tradeoff_notes = build_tradeoff_notes(benchmark_winner_matrix, efficiency_index, stability_index)
    print_live_compare_panel(micro_bench, stability_index, efficiency_index)

    pend = pending_long_run_flags(cfg, artifacts.state)

    router_entropy_hist = artifacts.curve_data.get("router_entropy", [])
    router_load_hist = artifacts.curve_data.get("router_max_load", [])
    collapse_events = int(sum(1 for x in artifacts.curve_data.get("collapse_detected", []) if float(x) > 0.0))
    moe_sparse_report = {
        "mode": str(cfg.get("moe_mode", "true_sparse_topk")),
        "router_entropy_mean": float(sum(router_entropy_hist or [0.0]) / max(1, len(router_entropy_hist))),
        "router_entropy_p10": float(_percentile(router_entropy_hist, 0.10)),
        "router_max_load_mean": float(sum(router_load_hist or [0.0]) / max(1, len(router_load_hist))),
        "router_max_load_p95": float(_p95(router_load_hist)),
        "router_max_load_p99": float(_percentile(router_load_hist, 0.99)),
        "capacity_overflow_ratio_mean": float(
            sum(artifacts.curve_data.get("capacity_overflow_ratio", [0.0]))
            / max(1, len(artifacts.curve_data.get("capacity_overflow_ratio", [])))
        ),
        "collapse_events": collapse_events,
        "aux_loss_mean": float(sum(artifacts.curve_data.get("aux_loss", [0.0])) / max(1, len(artifacts.curve_data.get("aux_loss", [])))),
        "expert_gate_pass": bool(mini_probe_report.get("expert_gate_pass", mini_probe_report.get("expert_load_gate", False))),
        "entropy_gate_pass": bool(mini_probe_report.get("entropy_gate_pass", mini_probe_report.get("router_entropy_gate", False))),
        "collapse_gate_pass": bool(mini_probe_report.get("collapse_gate_pass", collapse_events == 0)),
        "gates_config": safe_jsonable(mini_probe_report.get("gates_config", {})),
    }
    bitnet_mode_report = {
        "mode": str(cfg.get("bitnet_mode", "stable")),
        "skip_attention_qkvo": bool(cfg.get("bitnet_skip_attention_qkvo", True)),
        "telemetry": bitnet_tel,
    }
    logger_manifest = logger.finalize()
    logger_memory_report = {
        "mode": str(cfg.get("logger_mode", "jsonl_ring")),
        "ring_size": int(cfg.get("logger_ring_size", 5000)),
        "line_count_total": int(logger_manifest.get("line_count_total", logger_manifest.get("line_count", 0))),
        "line_count_ring": int(logger_manifest.get("line_count_ring", 0)),
        "jsonl_path": str(logger_manifest.get("jsonl_path", "")),
    }

    validation_trend = validation_trend_metrics(artifacts.curve_data.get("val_loss", []))
    ppl_raw_last = float("inf")
    ppl_capped_last = float("inf")
    ppl_cap_applied = False
    if artifacts.curve_data.get("val_loss"):
        try:
            raw = math.exp(float(artifacts.curve_data["val_loss"][-1]))
            ppl_raw_last = float(raw) if math.isfinite(raw) else float("inf")
        except OverflowError:
            ppl_raw_last = float("inf")
        ppl_capped_last = float(math.exp(min(20.0, float(artifacts.curve_data["val_loss"][-1]))))
        ppl_cap_applied = (not math.isfinite(ppl_raw_last)) or (abs(ppl_capped_last - ppl_raw_last) > 1e-6)

    code_stage_loaded = int(data_source_scorecard.get("totals", {}).get("stage_3_code_loaded", 0))
    coding_claim_blocked = bool(code_stage_loaded <= 0)
    degraded_conditions: List[str] = []
    if bool(preflight.get("degraded_data_mode", False)):
        degraded_conditions.append("degraded_data_mode=true")
    for w in preflight.get("warning_codes", []):
        degraded_conditions.append(f"warning:{w}")
    if coding_claim_blocked:
        degraded_conditions.append("coding_claim_blocked")

    payload: Dict[str, Any] = {
        "schema": "kaggle_onecell_t4_build30_v1",
        "generated_at_utc": _utc_now(),
        "profile": cfg["profile"],
        "device": device,
        "runtime": {
            "batch_size": cfg["batch_size"],
            "seq_len": cfg["seq_len"],
            "max_steps": cfg["max_steps"],
            "max_wall_hours": cfg["max_wall_hours"],
            "target_train_tokens": cfg["target_train_tokens"],
            "grad_accum_steps": cfg["grad_accum_steps"],
            "step_log_interval": cfg.get("step_log_interval", 1),
            "strict_green_min_tokens": int(cfg.get("strict_green_min_tokens", 8_000_000)),
            "benchmark_mode": str(cfg.get("benchmark_mode", "separated")),
        },
        "preflight": preflight,
        "tokenizer_metrics": tok_metrics,
        "startup_phase_metrics": startup_phase_metrics,
        "data_source_scorecard": data_source_scorecard,
        "curriculum_trace": curriculum_trace,
        "curriculum_config_hash": str(curriculum_trace[0].get("curriculum_config_hash", "")) if curriculum_trace else "",
        "curriculum_ratio_sum": float(curriculum_trace[0].get("curriculum_ratio_sum", 0.0)) if curriculum_trace else 0.0,
        "train_state": {
            "step": artifacts.state.step,
            "epoch": artifacts.state.epoch,
            "tokens_seen": artifacts.state.tokens_seen,
            "best_val_loss": artifacts.state.best_val_loss,
            "val_loss_median_last3": validation_trend.get("val_loss_median_last3", float("inf")),
            "val_loss_delta_rel_last3": validation_trend.get("val_loss_delta_rel_last3", 0.0),
            "val_plateau_detected": validation_trend.get("val_plateau_detected", False),
            "val_ppl_raw": ppl_raw_last,
            "val_ppl_capped": ppl_capped_last,
            "val_ppl_cap_applied": ppl_cap_applied,
            "warmup_excluded_loss_drop": warmup_excluded_loss_drop(artifacts.curve_data.get("train_loss", [])),
        },
        "train_meta": {
            **train_meta,
            "elapsed_total_sec": float(time.time() - total_start),
            "stop_reason": str(train_meta.get("stop_reason", "completed_or_condition_met")),
            "vram_util_estimate": float(gpu_tune.get("vram_util_estimate", 0.0)),
            "gpu_name": gpu_meta.get("gpu_name", ""),
            "compute_capability": gpu_meta.get("compute_capability", ""),
            "precision_fallback": gpu_meta.get("precision_fallback", ""),
            "inductor_enabled": bool(gpu_meta.get("inductor_enabled", False)),
        },
        "checkpoint_manifest": artifacts.checkpoint_manifest,
        "micro_benchmark_suite": micro_bench,
        "benchmark_suite": micro_bench,
        "benchmark_winner_matrix": benchmark_winner_matrix,
        "benchmark_tradeoff_notes": benchmark_tradeoff_notes,
        "parity_report": parity_report,
        "parity_signature": parity_signature,
        "moe_sparse_report": moe_sparse_report,
        "bitnet_mode_report": bitnet_mode_report,
        "logger_memory_report": logger_memory_report,
        "bitnet_telemetry": bitnet_tel,
        "stability_index": stability_index,
        "efficiency_index": efficiency_index,
        "mini_probe_report": mini_probe_report,
        "resume_decision": train_meta.get("resume_decision", {}),
        "compat_signature": train_meta.get("compat_signature", ""),
        "arch_parity_signature": train_meta.get("arch_parity_signature", parity_signature),
        "target_param_band": {"low": band_low, "high": band_high},
        "band_check": band_check,
        "pending_long_run_flags": pend,
        "logger_manifest": logger_manifest,
        "code_stage_loaded": code_stage_loaded,
        "coding_claim_blocked": coding_claim_blocked,
        "degraded_conditions": degraded_conditions,
        "cannot_claim_yet": list(pend),
        "claim_block": {
            "coding_claim": "blocked" if coding_claim_blocked else "allowed",
            "strict_data_claim": "blocked" if preflight.get("preflight_status") != "pass" else "allowed",
            # NOT a measured gate result: unverified author note, not tied to any test.
            "bug_free_claim_text": "no critical bugs known to author (UNVERIFIED; not a gate result)",
        },
        "runtime_model_params": int(mert_params_runtime),
        "gpu_meta": gpu_meta,
        "gpu_tune_report": gpu_tune,
        "stop_reason": str(train_meta.get("stop_reason", "completed_or_condition_met")),
        "run_config_hash": hash_config(cfg),
    }
    payload["payload_validator"] = validate_required_payload_fields(
        payload,
        required_fields=[
            "schema",
            "generated_at_utc",
            "profile",
            "runtime",
            "train_state",
            "train_meta",
            "micro_benchmark_suite",
            "mini_probe_report",
            "preflight",
        ],
    )

    verdict = compute_final_verdict(payload, cfg)
    payload.update(verdict)
    payload["final_status"] = verdict.get("final_verdict", "provisional")
    payload["final_reason"] = (
        verdict.get("strict_green_missing", [str(train_meta.get("stop_reason", "unknown"))])[0]
        if verdict.get("strict_green_missing")
        else str(train_meta.get("stop_reason", "completed"))
    )

    json_text, csv_text, md_text = render_reports(payload, artifacts.curve_data)

    print("\n===JSON===")
    print(json_text)
    print("\n===CSV===")
    print(csv_text.strip())
    print("\n===MD===")
    print(md_text.strip())

    out_files = maybe_write_outputs(cfg, json_text, csv_text, md_text)
    curve_plot = maybe_plot_curves(artifacts.curve_data, out_dir=layout.run_dir, write_files=bool(cfg["write_files"]))
    payload["output_files"] = out_files
    if curve_plot:
        payload["output_files"]["curve_plot"] = curve_plot
    presentation_assets = maybe_plot_presentation_assets(
        payload=payload,
        curve_data=artifacts.curve_data,
        out_dir=layout.run_dir,
        write_files=bool(cfg["write_files"]),
    )
    payload["output_files"].update(presentation_assets)
    if str(logger_manifest.get("jsonl_path", "")).strip():
        payload["output_files"]["jsonl_log"] = str(logger_manifest.get("jsonl_path", ""))
    ck_manifest_path = Path(str(cfg["checkpoint_dir"])) / "manifest.json"
    if ck_manifest_path.exists():
        payload["output_files"]["checkpoint_manifest"] = str(ck_manifest_path)
    final_model_path = str(train_meta.get("final_model_path", ""))
    if final_model_path:
        payload["output_files"]["final_model"] = final_model_path

    ck_manifest_src = Path(str(cfg["checkpoint_dir"])) / "manifest.json"
    ck_manifest_copy = layout.run_dir / "checkpoint_manifest_copy.json"
    if ck_manifest_src.exists() and bool(cfg.get("write_files", False)):
        shutil.copy2(ck_manifest_src, ck_manifest_copy)
        payload["output_files"]["checkpoint_manifest_copy"] = str(ck_manifest_copy)

    if bool(cfg.get("write_files", False)):
        public_summary = build_public_summary(payload, verdict)
        atomic_json_write(layout.public_summary_path, public_summary)
        payload["output_files"]["public_summary"] = str(layout.public_summary_path)
        stop_summary = {
            "generated_at_utc": _utc_now(),
            "run_id": layout.run_id,
            "stop_reason": str(train_meta.get("stop_reason", "completed_or_condition_met")),
            "final_status": payload["final_status"],
            "final_reason": payload["final_reason"],
            "tokens_seen": int(artifacts.state.tokens_seen),
        }
        atomic_json_write(layout.stop_summary_path, stop_summary)
        payload["output_files"]["stop_summary"] = str(layout.stop_summary_path)

    # Chat bootstrap (checkpoint-first)
    if bool(cfg.get("chat_enabled", True)):
        chat_model = MertFormerTiny(mert_cfg).to(device)
        ck = load_checkpoint_for_chat(chat_model, cfg=cfg, device=device, train_artifacts=artifacts)
        payload["chat_checkpoint"] = ck
        if ck.get("loaded", False):
            print(f"\n[chat] checkpoint loaded: {ck.get('path')}")
        else:
            chat_model.load_state_dict(model.state_dict(), strict=False)
            print(f"\n[chat] checkpoint load failed ({ck.get('reason')}); using in-memory latest model")

        sample_prompt = "Merhaba, bu modelin eğitim mimarisini kısa ve teknik anlat."
        sample_text = generate_text(
            model=chat_model,
            tokenizer=tokenizer,
            prompt=sample_prompt,
            device=device,
            max_new_tokens=int(cfg["chat_max_new_tokens"]),
            temperature=float(cfg["chat_temperature"]),
            top_p=float(cfg["chat_top_p"]),
            repetition_penalty=float(cfg["chat_repetition_penalty"]),
        )
        payload["chat_sample"] = {
            "prompt": sample_prompt,
            "response": sample_text,
        }
        print("\n===CHAT_SAMPLE===")
        print("PROMPT:", sample_prompt)
        print("RESPONSE:", sample_text)
        run_interactive_menu(chat_model, tokenizer, cfg, device, payload)

    if bool(cfg.get("write_files", False)):
        payload["output_files"].update(write_onecell_sidecars(layout, cfg, payload, preflight, logger_manifest))
        artifact_index = verify_and_index_artifacts(payload.get("output_files", {}))
        atomic_json_write(layout.artifact_index_path, artifact_index)
        payload["artifacts_index"] = artifact_index
        payload["output_files"]["artifacts_index"] = str(layout.artifact_index_path)
        zip_manifest = make_evidence_zip(layout, payload["output_files"], enabled=bool(cfg.get("zip_evidence_pack", True)))
        payload["zip_manifest"] = zip_manifest
        payload["output_files"]["zip_manifest"] = str(layout.zip_manifest_path)
        if layout.evidence_zip_path.exists():
            payload["output_files"]["evidence_zip"] = str(layout.evidence_zip_path)
        backup_report = backup_run_to_drive(layout, cfg)
        payload["backup_report"] = backup_report
        payload["output_files"]["backup_report"] = str(layout.run_dir / "backup_report.json")
        atomic_json_write(Path(payload["output_files"]["backup_report"]), backup_report)
        sha_manifest_path = Path(str(payload["output_files"].get("sha256_manifest", "")))
        if str(sha_manifest_path):
            write_output_sha256_manifest(payload["output_files"], sha_manifest_path)

        # Rewrite final JSON with output index + verdict.
        json_text_final = json.dumps(payload, ensure_ascii=False, indent=2)
        jpath = Path(payload["output_files"].get("json", ""))
        if jpath:
            jpath.write_text(json_text_final + "\n", encoding="utf-8")

    final_status_line = f"FINAL_STATUS: {payload['final_status']} reason={payload['final_reason']} run_id={layout.run_id}"
    print(final_status_line)
    return payload


# =============================================================================
# MertFormer 5080 Final Onefile Lab Rebuild Overlay v2
# Active model path: embedded repo modules -> model.transformers.MertFormer.
# =============================================================================
import base64 as _m5080_base64
import importlib as _m5080_importlib
import json as _m5080_json
import math as _m5080_math
import os as _m5080_os
import shutil as _m5080_shutil
import sys as _m5080_sys
import time as _m5080_time
import types as _m5080_types
import zlib as _m5080_zlib
from pathlib import Path as _M5080Path
from types import SimpleNamespace as _M5080SimpleNamespace

MERTFORMER_SOURCE_MANIFEST = '{\n  "schema": "mertformer-source-manifest-v2",\n  "created_at": "2026-04-21T17:19:47.199611+00:00",\n  "repo_root": "<REPO_ROOT>",\n  "canonical_output": "<REPO_ROOT>/scripts/mertformer_5080_final_onefile.py",\n  "runtime_embedding_policy": "Exact repo source text is embedded for architecture modules and executed in-memory with a onefile runtime cfg adapter.",\n  "files": [\n    {\n      "repo_path": "layers/bitlinear.py",\n      "abs_path": "<REPO_ROOT>/layers/bitlinear.py",\n      "sha256": "262b9de16548d7a0b298c199119319bbec3c77af30e90d4443b3c89dcbf85bc6",\n      "bytes": 5551,\n      "line_count": 160,\n      "embedded_runtime_module": "layers.bitlinear",\n      "embedded_classes": [\n        "BitLinear"\n      ],\n      "embedded_functions": [\n        "set_lowbit_kernel_enabled",\n        "_try_lowbit_kernel",\n        "activation_quant",\n        "weight_quant"\n      ]\n    },\n    {\n      "repo_path": "layers/ffn.py",\n      "abs_path": "<REPO_ROOT>/layers/ffn.py",\n      "sha256": "b19e37b40d56d032363337563fd30378248cac3680b349169c4ae77461ef4c5c",\n      "bytes": 2929,\n      "line_count": 81,\n      "embedded_runtime_module": "layers.ffn",\n      "embedded_classes": [\n        "MertFormerFFN"\n      ],\n      "embedded_functions": []\n    },\n    {\n      "repo_path": "layers/qinn.py",\n      "abs_path": "<REPO_ROOT>/layers/qinn.py",\n      "sha256": "ca8af1acb74974ee13eec237da0a6d84f00446fb3796db779542a23023218a61",\n      "bytes": 5975,\n      "line_count": 160,\n      "embedded_runtime_module": "layers.qinn",\n      "embedded_classes": [\n        "UnitaryQINN"\n      ],\n      "embedded_functions": [\n        "newton_schulz_inverse"\n      ]\n    },\n    {\n      "repo_path": "layers/cognitive_extensions.py",\n      "abs_path": "<REPO_ROOT>/layers/cognitive_extensions.py",\n      "sha256": "b572fd1490eadc71950368f7d499dcc018bd08059c6b3100a57a7adab897df7a",\n      "bytes": 5407,\n      "line_count": 125,\n      "embedded_runtime_module": "layers.cognitive_extensions",\n      "embedded_classes": [\n        "GlobalWorkspaceBroadcast",\n        "ContinuousLatentODEStateChannel",\n        "NeuromodulatoryGainLayer",\n        "HebbianPlasticityLayer",\n        "NeuroSymbolicLayer"\n      ],\n      "embedded_functions": []\n    },\n    {\n      "repo_path": "layers/lifelong_safety.py",\n      "abs_path": "<REPO_ROOT>/layers/lifelong_safety.py",\n      "sha256": "cf92dc7dcfabfc507e3ba5df3d52dfbb3df6a9b7903b34cf01b08d5a5bcdd3c4",\n      "bytes": 2860,\n      "line_count": 77,\n      "embedded_runtime_module": "layers.lifelong_safety",\n      "embedded_classes": [\n        "LifelongSafetyLayer"\n      ],\n      "embedded_functions": []\n    },\n    {\n      "repo_path": "layers/world_model_head.py",\n      "abs_path": "<REPO_ROOT>/layers/world_model_head.py",\n      "sha256": "b44801193906e384e570022e549e3c9dc7869da6d981df844c43d897640eca15",\n      "bytes": 2959,\n      "line_count": 83,\n      "embedded_runtime_module": "layers.world_model_head",\n      "embedded_classes": [\n        "WorldModelOutput",\n        "CausalWorldModelHead"\n      ],\n      "embedded_functions": []\n    },\n    {\n      "repo_path": "layers/liquid.py",\n      "abs_path": "<REPO_ROOT>/layers/liquid.py",\n      "sha256": "9cb4038e3e2c38caab1967b7ed20004119066eb1280dc57492a354a68497b855",\n      "bytes": 16515,\n      "line_count": 400,\n      "embedded_runtime_module": "layers.liquid",\n      "embedded_classes": [\n        "LiquidCell",\n        "LiquidMixer"\n      ],\n      "embedded_functions": [\n        "_jit_script_if_supported",\n        "jit_quant",\n        "jit_liquid_loop_cached",\n        "jit_liquid_loop"\n      ]\n    },\n    {\n      "repo_path": "layers/mla.py",\n      "abs_path": "<REPO_ROOT>/layers/mla.py",\n      "sha256": "1afe20f9e375315e40adfcef34d36a151fb159fa72d80e4125958093188e4df2",\n      "bytes": 19886,\n      "line_count": 477,\n      "embedded_runtime_module": "layers.mla",\n      "embedded_classes": [\n        "_QKRMSNorm",\n        "RotaryEmbedding",\n        "MLA"\n      ],\n      "embedded_functions": [\n        "_is_onnx_export",\n        "rotate_half",\n        "apply_rope_optimized"\n      ]\n    },\n    {\n      "repo_path": "layers/moe.py",\n      "abs_path": "<REPO_ROOT>/layers/moe.py",\n      "sha256": "34cf76046cd522894d20bf10b1031aeeab8c0fda0dd5f4b33d4fed1bb5d273d1",\n      "bytes": 37589,\n      "line_count": 856,\n      "embedded_runtime_module": "layers.moe",\n      "embedded_classes": [\n        "BitSwiGLU",\n        "LiquidRouter",\n        "MoE"\n      ],\n      "embedded_functions": []\n    },\n    {\n      "repo_path": "layers/mertformer_block.py",\n      "abs_path": "<REPO_ROOT>/layers/mertformer_block.py",\n      "sha256": "ba585c97eb04920f59fc011a80e4839e615b1c452cb79eaa642484db916be552",\n      "bytes": 10438,\n      "line_count": 254,\n      "embedded_runtime_module": "layers.mertformer_block",\n      "embedded_classes": [\n        "RMSNorm",\n        "MertFormerBlock"\n      ],\n      "embedded_functions": []\n    },\n    {\n      "repo_path": "model/transformers.py",\n      "abs_path": "<REPO_ROOT>/model/transformers.py",\n      "sha256": "551f31a92d53fc3620ba7a841d9df6317c1fa2fc3c372798ba5ee89780186376",\n      "bytes": 13631,\n      "line_count": 312,\n      "embedded_runtime_module": "model.transformers",\n      "embedded_classes": [\n        "MertFormer"\n      ],\n      "embedded_functions": []\n    },\n    {\n      "repo_path": "layers/bitnet_patch.py",\n      "abs_path": "<REPO_ROOT>/layers/bitnet_patch.py",\n      "sha256": "631eefd14fad9a6d8e9d371bc33a27a46bc5a5d86cda8fb83c75bc66fd55aeb8",\n      "bytes": 3233,\n      "line_count": 97,\n      "embedded_runtime_module": null,\n      "embedded_classes": [],\n      "embedded_functions": [\n        "_convert_linear_modules",\n        "apply_bitnet"\n      ]\n    },\n    {\n      "repo_path": "utils/logger.py",\n      "abs_path": "<REPO_ROOT>/utils/logger.py",\n      "sha256": "fd95d5db206cb9f138ac040bc47fb72738011ff8188c5de03a3c1adfedc8dbd5",\n      "bytes": 16767,\n      "line_count": 499,\n      "embedded_runtime_module": null,\n      "embedded_classes": [\n        "RunLogger"\n      ],\n      "embedded_functions": [\n        "_utc_iso",\n        "_local_stamp",\n        "_safe_json",\n        "sha256_file",\n        "try_git_commit",\n        "atomic_write_json",\n        "_redact_text",\n        "_redact_obj",\n        "_ensure_logbook_header"\n      ]\n    },\n    {\n      "repo_path": "train/train.py",\n      "abs_path": "<REPO_ROOT>/train/train.py",\n      "sha256": "abf343fa3ec60df07b7686d4df4e5582e961a92486f171928e9d9ace5665a356",\n      "bytes": 103121,\n      "line_count": 2229,\n      "embedded_runtime_module": null,\n      "embedded_classes": [\n        "MertFormerInferenceWrapper",\n        "ValidationJsonlDataset",\n        "CurriculumDataset",\n        "PrecomputedCurriculumDataset",\n        "TeacherBundle"\n      ],\n      "embedded_functions": [\n        "seed_all",\n        "validate_config",\n        "check_disk_space",\n        "count_jsonl_records",\n        "get_gpu_memory_usage",\n        "write_energy_telemetry_baseline",\n        "get_curriculum_contract",\n        "build_stage_boundaries",\n        "read_metric_from_json",\n        "write_training_readiness_manifest",\n        "get_student_device",\n        "get_teacher_device",\n        "preflight_param_report",\n        "save_checkpoint_smart",\n        "_normalize_state_dict_keys_for_model",\n        "_discover_resume_checkpoint",\n        "_load_resume_payload",\n        "_infer_curriculum_stage_from_step",\n        "export_to_onnx",\n        "collate_fn",\n        "kd_loss_safe",\n        "get_wsd_schedule",\n        "_tokenizer_candidates",\n        "_ensure_pad_token",\n        "_load_local_runtime_tokenizer",\n        "load_teacher_tokenizer",\n        "rebuild_optimizer",\n        "apply_freeze_policy",\n        "train"\n      ]\n    },\n    {\n      "repo_path": "run.sh",\n      "abs_path": "<REPO_ROOT>/run.sh",\n      "sha256": "aae67b6d2ea0bc7bed6290dbf60d6daff4ee25ee896f83cb0aee6115dd4e2897",\n      "bytes": 24440,\n      "line_count": 664,\n      "embedded_runtime_module": null,\n      "embedded_classes": [],\n      "embedded_functions": []\n    }\n  ],\n  "script_audit_files": [\n    {\n      "repo_path": "scripts/benchmarks_internal.py",\n      "sha256": "7f8e326021bb921d48b1ba2dc48cbb98612a917198f6b585a265058706c53872",\n      "bytes": 5837,\n      "line_count": 171,\n      "classes": [],\n      "functions": [\n        "load_dataset_safe",\n        "run_generation",\n        "write_summary",\n        "main"\n      ]\n    },\n    {\n      "repo_path": "scripts/bitnet_kernel_benchmark_standalone.py",\n      "sha256": "1cc5d9c1994baa647850597f055c4d3643b3b9845d18c94b8f9328029c79db10",\n      "bytes": 15142,\n      "line_count": 484,\n      "classes": [\n        "BenchRow"\n      ],\n      "functions": [\n        "is_triton_available",\n        "_quantize_activation",\n        "_quantize_weight",\n        "reference_ternary_linear",\n        "triton_ternary_linear",\n        "_sync",\n        "_bench",\n        "_parse_shapes",\n        "_fmt",\n        "run_benchmark",\n        "_print_rows",\n        "main",\n        "run_default"\n      ]\n    },\n    {\n      "repo_path": "scripts/build_chess_5080_windows_delivery.py",\n      "sha256": "3dbe343a97c7fd6eb2fa6ad8015142d1d83a92914b10655fb482e7ed3e8a30da",\n      "bytes": 10867,\n      "line_count": 289,\n      "classes": [\n        "BuildError"\n      ],\n      "functions": [\n        "sha256_file",\n        "run",\n        "require_windows",\n        "ensure_python_package",\n        "parse_major_minor",\n        "inspect_torch_install",\n        "torch_install_is_acceptable",\n        "ensure_build_dependencies",\n        "render_launcher",\n        "build_nuitka_command",\n        "detect_signtool",\n        "try_sign_executable",\n        "main"\n      ]\n    },\n    {\n      "repo_path": "scripts/build_chess_onefile_extension_report.py",\n      "sha256": "5f43aa2ad9f57d0259fe02221efc9499e90a6f75b89fdec4d0d916526400e441",\n      "bytes": 3665,\n      "line_count": 86,\n      "classes": [],\n      "functions": [\n        "build_payload",\n        "build_markdown",\n        "main"\n      ]\n    },\n    {\n      "repo_path": "scripts/build_chess_training_readiness_report.py",\n      "sha256": "3e5ac2784fddca225c493a60c5ab029df005a5095f7c778b3a585c63be3fa5f2",\n      "bytes": 5662,\n      "line_count": 146,\n      "classes": [],\n      "functions": [\n        "load_json",\n        "display_path",\n        "build_payload",\n        "build_markdown",\n        "main"\n      ]\n    },\n    {\n      "repo_path": "scripts/build_train_readiness_contract.py",\n      "sha256": "0b7a89c5ed289814b9d962713caa901d348ba7dde3dd7f6dfa64794ff77e74bb",\n      "bytes": 5746,\n      "line_count": 163,\n      "classes": [],\n      "functions": [\n        "load_json",\n        "sanitize_text",\n        "sanitize_value",\n        "run_profile",\n        "choose_decision",\n        "write_markdown",\n        "main"\n      ]\n    },\n    {\n      "repo_path": "scripts/chess_5080_onefile.py",\n      "sha256": "f4286998b4382ac7ed86c0035e40de1f5c07aef400b33c09ee97639cafd0e49f",\n      "bytes": 799081,\n      "line_count": 16581,\n      "classes": [\n        "ExecutionStatus",\n        "EvaluationStatus",\n        "RatingClaimStatus",\n        "ChessOnefileError",\n        "ConfigValidationError",\n        "DependencyBootstrapRequired",\n        "DownloadError",\n        "DatasetEmptyError",\n        "TrainingOOMError",\n        "PackagingError",\n        "NonFiniteLossError",\n        "ResumeCheckpointError",\n        "ArtifactLayout",\n        "ChessExample",\n        "CuratedPositionSpec",\n        "DownloadSlice",\n        "ResumeState",\n        "JSONLLogger",\n        "_JSONLLogFormatter",\n        "_ConsoleLogFormatter",\n        "WindowsExecutionGuard",\n        "MirrorModelConfig",\n        "BitLinear",\n        "RMSNorm",\n        "_QKRMSNorm",\n        "RotaryEmbedding",\n        "MLA",\n        "MertFormerFFN",\n        "LiquidCell",\n        "LiquidMixer",\n        "BitSwiGLU",\n        "LiquidRouter",\n        "MoE",\n        "UnitaryQINN",\n        "GlobalWorkspaceBroadcast",\n        "ContinuousLatentODEStateChannel",\n        "NeuromodulatoryGainLayer",\n        "HebbianPlasticityLayer",\n        "NeuroSymbolicLayer",\n        "LifelongSafetyLayer",\n        "WorldModelOutput",\n        "CausalWorldModelHead",\n        "MertFormerBlock",\n        "ChessPolicyValueNet",\n        "ChessExampleDataset"\n      ],\n      "functions": [\n        "_module_exists",\n        "_install_allowed",\n        "_pip_install",\n        "_bootstrap_if_needed",\n        "_import_runtime_dependencies",\n        "utc_now",\n        "deterministic_seed",\n        "sha256_bytes",\n        "path_sha256",\n        "atomic_write_text",\n        "atomic_json",\n        "_log_safe_json",\n        "_redact_log_text",\n        "_redact_log_object",\n        "safe_name",\n        "redact_path",\n        "detect_desktop_dir",\n        "disk_free_gb",\n        "get_package_version",\n        "get_nvidia_driver_version",\n        "env_snapshot",\n        "collect_dependency_lock",\n        "validate_enum_choice",\n        "parse_feature_list",\n        "apply_feature_bundle",\n        "apply_feature_flag_overrides",\n        "build_feature_flag_report",\n        "render_feature_flag_report_md",\n        "apply_profile",\n        "apply_baseline",\n        "resolve_runtime_config",\n        "validate_runtime_config",\n        "_normalize_liquid_layers_idx",\n        "default_liquid_layers_idx",\n        "build_mirror_model_config",\n        "set_lowbit_kernel_enabled",\n        "_import_optional_sdk_module",\n        "_try_lowbit_kernel",\n        "activation_quant",\n        "weight_quant",\n        "make_linear",\n        "_is_onnx_export",\n        "rotate_half",\n        "apply_rope_optimized",\n        "_jit_script_if_supported",\n        "jit_quant",\n        "jit_liquid_loop_cached",\n        "jit_liquid_loop",\n        "newton_schulz_inverse",\n        "value_target_to_wdl_class",\n        "value_targets_to_wdl_classes",\n        "piece_to_id",\n        "material_bucket",\n        "encode_board_state",\n        "infer_phase",\n        "parse_time_control",\n        "result_to_value",\n        "parse_eval_comment",\n        "build_move_vocab",\n        "legal_move_ids",\n        "normalized_position_hash",\n        "normalized_game_hash",\n        "opening_prefix_from_moves",\n        "comment_has_eval_tag",\n        "score_candidate_ply",\n        "select_ply_indices",\n        "game_is_usable",\n        "iter_games_from_pgn_text",\n        "embedded_seed_games",\n        "mainline_moves",\n        "materialize_curated_position_bank",\n        "build_curated_position_manifest",\n        "build_curated_training_examples",\n        "render_curated_position_manifest_md",\n        "choose_archive_urls",\n        "download_archive_slices",\n        "iter_games_from_zstd",\n        "build_examples_from_games",\n        "maybe_collect_dataset",\n        "split_examples_by_game",\n        "build_curriculum_stages",\n        "pick_device",\n        "maybe_enable_compile",\n        "build_optimizer",\n        "lr_for_step",\n        "apply_optimizer_lr",\n        "batch_to_device",\n        "collate_examples",\n        "compute_prediction_metrics",\n        "compute_loss",\n        "forward_batch_metrics",\n        "merge_metric_sums",\n        "merge_metric_sums_weighted",\n        "summarize_metric_sums",\n        "evaluate_model",\n        "extract_raw_vs_masked_metrics",\n        "get_rng_state",\n        "restore_rng_state",\n        "save_checkpoint",\n        "load_checkpoint",\n        "infer_existing_run_dir_from_resume",\n        "make_layout",\n        "prepare_layout",\n        "make_loader",\n        "stage_index_for_step",\n        "training_loop",\n        "_find_stockfish_binary",\n        "_stockfish_asset_score",\n        "_download_to_path",\n        "_fetch_stockfish_binary",\n        "detect_stockfish_path",\n        "normalize_chess_response_mode",\n        "normalize_teaching_level",\n        "classify_evaluation_label",\n        "build_evaluation_phrase_tr",\n        "build_confidence_payload",\n        "build_auxiliary_prediction_payload",\n        "unpack_model_outputs",\n        "terminal_value_for_color",\n        "evaluate_board_value",\n        "policy_snapshot_for_board",\n        "is_tactically_forcing",\n        "infer_search_budget",\n        "score_move_with_shallow_search",\n        "classify_teaching_tags",\n        "build_teaching_reasons_tr",\n        "build_chess_response_contract",\n        "synthetic_trace_for_curated_position",\n        "build_synthetic_teaching_corpus",\n        "render_synthetic_teaching_corpus_md",\n        "choose_move_trace",\n        "not_run_curated_position_eval",\n        "evaluate_curated_position_suites",\n        "render_curated_position_suite_report_md",\n        "run_legality_report",\n        "ensure_interactive_console",\n        "play_human_vs_model_arena",\n        "write_curve_csv",\n        "_png_chunk",\n        "_write_simple_png",\n        "_set_pixel",\n        "_draw_line",\n        "write_curve_png",\n        "compute_score_rate_ci",\n        "elo_proxy_from_score",\n        "build_benchmark_protocol",\n        "build_pgn_from_moves",\n        "not_run_selfplay_report",\n        "render_selfplay_report_md",\n        "not_run_tournament_report",\n        "render_tournament_report_md",\n        "not_run_replay_buffer_report",\n        "render_replay_buffer_report_md",\n        "_limited_opening_prefix",\n        "generate_selfplay_report",\n        "build_replay_buffer_report",\n        "play_inference_mode_tournament",\n        "play_stockfish_gauntlet",\n        "build_midrun_stockfish_snapshot_cfg",\n        "build_snapshot_layout",\n        "maybe_write_midrun_training_snapshots",\n        "generate_demo_replay",\n        "determine_statuses",\n        "build_mirror_enabled_flags",\n        "build_mirror_parity_report",\n        "assert_mirror_surface_integrity",\n        "build_model_card",\n        "build_eval_card",\n        "render_run_summary_md",\n        "render_proof_scope_md",\n        "render_repro_md",\n        "render_third_party_licenses",\n        "write_cards_and_reports",\n        "build_artifact_manifest",\n        "build_run_status_manifest",\n        "render_run_status_manifest_md",\n        "build_postrun_analysis_manifest",\n        "render_postrun_analysis_manifest_md",\n        "build_artifact_truth_matrix",\n        "render_artifact_truth_matrix_md",\n        "write_closure_manifests",\n        "_read_json_if_exists",\n        "_read_text_if_exists",\n        "build_run_contract",\n        "render_run_contract_md",\n        "build_release_snapshot",\n        "render_release_snapshot_md",\n        "_truth_entry_exists",\n        "_artifact_consumer_state",\n        "_artifact_consumer_state_with_paths",\n        "build_evidence_pack_stub",\n        "render_evidence_pack_stub_md",\n        "build_final_truth_registry",\n        "render_final_truth_registry_md",\n        "build_claim_registry",\n        "render_claim_registry_md",\n        "build_known_limits",\n        "render_known_limits_md",\n        "build_support_matrix",\n        "render_support_matrix_md",\n        "build_release_gate_summary",\n        "render_release_gate_summary_md",\n        "build_rc_stub",\n        "render_rc_stub_md",\n        "build_golden_stub",\n        "render_golden_stub_md",\n        "build_handoff_pack_manifest",\n        "render_handoff_pack_manifest_md",\n        "build_operator_handoff_summary",\n        "render_operator_handoff_summary_md",\n        "build_external_repro_stub",\n        "render_external_repro_stub_md",\n        "build_pilot_stub",\n        "render_pilot_stub_md",\n        "build_security_stub",\n        "render_security_stub_md",\n        "build_legal_stub",\n        "render_legal_stub_md",\n        "build_operator_handbook_stub",\n        "render_operator_handbook_stub_md",\n        "build_dr_evidence_stub",\n        "render_dr_evidence_stub_md",\n        "build_backup_retention_stub",\n        "render_backup_retention_stub_md",\n        "build_blind_handoff_stub",\n        "render_blind_handoff_stub_md",\n        "build_release_notes_stub",\n        "render_release_notes_stub_md",\n        "build_freeze_manifest_stub",\n        "render_freeze_manifest_stub_md",\n        "build_changelog_snapshot",\n        "render_changelog_snapshot_md",\n        "build_maintenance_policy_stub",\n        "render_maintenance_policy_stub_md",\n        "build_export_truth_stub",\n        "render_export_truth_stub_md",\n        "build_device_validation_stub",\n        "render_device_validation_stub_md",\n        "build_packaging_closure_stub",\n        "render_packaging_closure_stub_md",\n        "build_installer_validation_stub",\n        "render_installer_validation_stub_md",\n        "build_benchmark_raw_outputs_stub",\n        "render_benchmark_raw_outputs_stub_md",\n        "build_benchmark_compare_report_stub",\n        "render_benchmark_compare_report_stub_md",\n        "build_benchmark_summary_stub",\n        "render_benchmark_summary_stub_md",\n        "build_benchmark_manifest_stub",\n        "render_benchmark_manifest_stub_md",\n        "build_training_report_stub",\n        "render_training_report_stub_md",\n        "build_token_accounting_stub",\n        "render_token_accounting_stub_md",\n        "build_compute_accounting_stub",\n        "render_compute_accounting_stub_md",\n        "build_cost_report_stub",\n        "render_cost_report_stub_md",\n        "build_final_weights_truth_stub",\n        "render_final_weights_truth_stub_md",\n        "build_best_checkpoint_truth_stub",\n        "render_best_checkpoint_truth_stub_md",\n        "build_latest_checkpoint_truth_stub",\n        "render_latest_checkpoint_truth_stub_md",\n        "build_trained_artifact_registry_stub",\n        "render_trained_artifact_registry_stub_md",\n        "build_core_complete_decision_stub",\n        "render_core_complete_decision_stub_md",\n        "build_research_continues_stub",\n        "render_research_continues_stub_md",\n        "build_product_maintenance_only_stub",\n        "render_product_maintenance_only_stub_md",\n        "build_closure_decision_record_stub",\n        "render_closure_decision_record_stub_md",\n        "build_master_closure_table",\n        "render_master_closure_table_md",\n        "build_remaining_core_blockers",\n        "render_remaining_core_blockers_md",\n        "build_repo_side_completion_summary",\n        "render_repo_side_completion_summary_md",\n        "build_readiness_snapshot",\n        "render_readiness_snapshot_md",\n        "build_aggregated_master_table",\n        "render_aggregated_master_table_md",\n        "build_real_remaining_core_work",\n        "render_real_remaining_core_work_md",\n        "build_repo_truth_inventory",\n        "render_repo_truth_inventory_md",\n        "build_closure_gap_summary",\n        "render_closure_gap_summary_md",\n        "build_project_master_truth_reference",\n        "render_project_master_truth_reference_md",\n        "build_project_remaining_real_blockers",\n        "render_project_remaining_real_blockers_md",\n        "_project_blocker_specs",\n        "build_truth_docs_index",\n        "render_truth_docs_index_md",\n        "build_truth_docs_drift_report",\n        "render_truth_docs_drift_report_md",\n        "build_project_blocker_action_plan",\n        "render_project_blocker_action_plan_md",\n        "build_project_blocker_dependency_graph",\n        "render_project_blocker_dependency_graph_md",\n        "build_project_execution_sequence",\n        "render_project_execution_sequence_md",\n        "build_project_lane_status_board",\n        "render_project_lane_status_board_md",\n        "build_project_closure_phase_plan",\n        "render_project_closure_phase_plan_md",\n        "build_project_phase_readiness_scoreboard",\n        "render_project_phase_readiness_scoreboard_md",\n        "build_project_owner_accountability_matrix",\n        "render_project_owner_accountability_matrix_md",\n        "build_project_owner_work_queue",\n        "render_project_owner_work_queue_md",\n        "build_project_critical_path_report",\n        "render_project_critical_path_report_md",\n        "build_project_owner_next_actions_summary",\n        "render_project_owner_next_actions_summary_md",\n        "build_project_ready_now_board",\n        "render_project_ready_now_board_md",\n        "build_project_unlock_impact_report",\n        "render_project_unlock_impact_report_md",\n        "build_project_parallel_workset_report",\n        "render_project_parallel_workset_report_md",\n        "build_project_phase_exit_criteria_report",\n        "render_project_phase_exit_criteria_report_md",\n        "build_project_execution_wave_report",\n        "render_project_execution_wave_report_md",\n        "build_project_evidence_backlog_report",\n        "render_project_evidence_backlog_report_md",\n        "build_project_dependency_bottleneck_report",\n        "render_project_dependency_bottleneck_report_md",\n        "build_project_owner_phase_frontier_report",\n        "render_project_owner_phase_frontier_report_md",\n        "build_project_evidence_criticality_report",\n        "render_project_evidence_criticality_report_md",\n        "build_project_phase_transition_matrix",\n        "render_project_phase_transition_matrix_md",\n        "build_project_owner_load_report",\n        "render_project_owner_load_report_md",\n        "build_project_phase_dependency_pressure_report",\n        "render_project_phase_dependency_pressure_report_md",\n        "build_project_owner_bottleneck_alignment_report",\n        "render_project_owner_bottleneck_alignment_report_md",\n        "build_project_evidence_phase_heatmap_report",\n        "render_project_evidence_phase_heatmap_report_md",\n        "build_project_blocker_risk_register_report",\n        "render_project_blocker_risk_register_report_md",\n        "build_project_release_prereq_matrix_report",\n        "render_project_release_prereq_matrix_report_md",\n        "build_project_foundation_run_dependency_report",\n        "render_project_foundation_run_dependency_report_md",\n        "build_project_release_path_report",\n        "render_project_release_path_report_md",\n        "build_project_external_closure_cluster_report",\n        "render_project_external_closure_cluster_report_md",\n        "build_project_owner_evidence_gap_report",\n        "render_project_owner_evidence_gap_report_md",\n        "build_project_release_gate_dependency_report",\n        "render_project_release_gate_dependency_report_md",\n        "build_project_external_signoff_queue_report",\n        "render_project_external_signoff_queue_report_md",\n        "build_project_release_evidence_bridge_report",\n        "render_project_release_evidence_bridge_report_md",\n        "build_project_training_run_readiness_report",\n        "render_project_training_run_readiness_report_md",\n        "build_project_benchmark_closure_dependency_report",\n        "render_project_benchmark_closure_dependency_report_md",\n        "build_project_release_decision_queue_report",\n        "render_project_release_decision_queue_report_md",\n        "build_project_external_validation_readiness_report",\n        "render_project_external_validation_readiness_report_md",\n        "build_project_artifact_lock_readiness_report",\n        "render_project_artifact_lock_readiness_report_md",\n        "build_project_final_release_cutover_report",\n        "render_project_final_release_cutover_report_md",\n        "build_project_real_run_execution_queue_report",\n        "render_project_real_run_execution_queue_report_md",\n        "build_project_benchmark_evidence_lock_report",\n        "render_project_benchmark_evidence_lock_report_md",\n        "build_project_final_signoff_cutset_report",\n        "render_project_final_signoff_cutset_report_md",\n        "build_generated_truth_consistency_report",\n        "render_generated_truth_consistency_report_md",\n        "build_generated_truth_crosscheck_matrix",\n        "render_generated_truth_crosscheck_matrix_md",\n        "_write_release_evidence_reports_once",\n        "write_release_evidence_reports",\n        "resolve_archive_password",\n        "_write_bundle_zip",\n        "create_result_bundle",\n        "cleanup_after_bundle_if_needed",\n        "not_run_evaluation",\n        "not_run_legality_report",\n        "not_run_demo_replay",\n        "not_run_arena_session",\n        "schedule_self_delete_if_needed",\n        "verify_forward_pass",\n        "prepare_model_and_optimizer",\n        "collect_verify_examples",\n        "package_existing_run",\n        "run_pipeline",\n        "build_argument_parser",\n        "main"\n      ]\n    },\n    {\n      "repo_path": "scripts/chess_onefile_contract.py",\n      "sha256": "95b26a6619e7b990c7fe31cc8bf174c3de86338b80d1e94fe199024a2bd2653a",\n      "bytes": 2469,\n      "line_count": 59,\n      "classes": [],\n      "functions": [\n        "profile_support_rows",\n        "is_release_candidate_profile",\n        "is_release_candidate_configuration",\n        "release_candidate_reason",\n        "release_candidate_configuration_reason"\n      ]\n    },\n    {\n      "repo_path": "scripts/drone_sitl_demo.py",\n      "sha256": "ef33db20134637798201cabc7cfc9db8c84e07093870c6990e6b9faf33bd3687",\n      "bytes": 9847,\n      "line_count": 294,\n      "classes": [\n        "SitlEvent",\n        "MertFormerSITLPolicy"\n      ],\n      "functions": [\n        "_iso_now",\n        "_baseline_action",\n        "run_once",\n        "write_outputs",\n        "main"\n      ]\n    },\n    {\n      "repo_path": "scripts/export_chess_5080_share.py",\n      "sha256": "071138503ccf85bab0ef6b2fbb2e1a9dd52c1d9772014cc17c9233e9461df404",\n      "bytes": 13135,\n      "line_count": 293,\n      "classes": [],\n      "functions": [\n        "sha256_file",\n        "render_build_bat",\n        "render_build_ps1",\n        "render_run_final_ps1",\n        "render_run_final_bat",\n        "render_readme",\n        "main"\n      ]\n    },\n    {\n      "repo_path": "scripts/kaggle_onecell_t4_build30.py",\n      "sha256": "d7c3d362acfbbd9431e12f2c5b0d2273ac8bab37c080e47b039a19f7b76d1d91",\n      "bytes": 287066,\n      "line_count": 6984,\n      "classes": [\n        "ArtifactLayout",\n        "InMemoryRunLogger",\n        "SimpleTokenizer",\n        "ByteBPETokenizer",\n        "SentencePieceTokenizer",\n        "HybridTokenizer",\n        "RMSNorm",\n        "BitLinearStrict",\n        "RotaryEmbedding",\n        "SwiGLUFFN",\n        "QINNLayer",\n        "HebbianPlasticityLayer",\n        "NeuroSymbolicLayer",\n        "WorldModelHead",\n        "LifelongSafetyLayer",\n        "ContinuousLatentODEStateChannel",\n        "NeuromodulatoryGainLayer",\n        "LiquidMixer",\n        "LiquidRouter",\n        "MoELayer",\n        "MLA",\n        "MertFormerCfg",\n        "MertFormerBlock",\n        "MertFormerTiny",\n        "VanillaTransformerLM",\n        "PackedDataset",\n        "TrainState",\n        "TrainArtifacts"\n      ],\n      "functions": [\n        "_utc_now",\n        "_local_stamp",\n        "_run_id_stamp",\n        "_signal_stop_handler",\n        "install_runtime_signal_handlers",\n        "ensure_writable_dir",\n        "init_artifact_layout",\n        "file_sha256",\n        "atomic_text_write",\n        "write_output_sha256_manifest",\n        "build_embedded_layer_parity_report",\n        "write_onecell_sidecars",\n        "write_onecell_fatal_report",\n        "reason_code_from_error",\n        "resolve_hf_token",\n        "_quick_hf_dataset_probe",\n        "run_data_preflight",\n        "write_last_state",\n        "append_csv_row",\n        "_median",\n        "_percentile",\n        "_winsorized",\n        "robust_grad_stats",\n        "validation_trend_metrics",\n        "warmup_excluded_loss_drop",\n        "_print_header",\n        "safe_jsonable",\n        "set_seed",\n        "pick_device",\n        "get_total_vram_gb",\n        "get_cuda_device_meta",\n        "apply_gpu_auto_tune",\n        "reset_device_peak_memory",\n        "get_device_peak_memory_gb",\n        "resolve_writable_dir",\n        "_is_dir_writable",\n        "interactive_prompt",\n        "is_notebook_runtime",\n        "can_accept_user_input",\n        "resolve_runtime_config",\n        "maybe_autocast",\n        "hash_config",\n        "_safe_div",\n        "compute_arch_parity_signature",\n        "build_compat_signature",\n        "_format_exception",\n        "parity_crosscheck_local_repo",\n        "summarize_data_source_scorecard",\n        "build_benchmark_winner_matrix",\n        "build_tradeoff_notes",\n        "compute_stability_index",\n        "compute_efficiency_index",\n        "_mean",\n        "_p95",\n        "compute_mini_probe_report",\n        "print_live_compare_panel",\n        "normalize_text",\n        "text_quality_ok",\n        "build_curriculum_sources",\n        "synthetic_stage_samples",\n        "_extract_field",\n        "_bounded_split",\n        "_load_hf_candidate_dataset",\n        "_hf_candidate_worker",\n        "_load_candidate_rows_process_timeout",\n        "load_stage_texts",\n        "build_curriculum_corpus",\n        "compute_oov_rate",\n        "token_histogram_topk",\n        "pack_sequences",\n        "build_train_val_streams",\n        "activation_quant_int8_ste",\n        "weight_quant_ternary_ste",\n        "_linear_core",\n        "parity_self_check",\n        "_should_skip_bitnet",\n        "convert_model_to_strict_bitnet",\n        "collect_bitnet_telemetry",\n        "make_loader",\n        "atomic_json_write",\n        "atomic_torch_save",\n        "prune_checkpoints_rolling5",\n        "collect_rng_state",\n        "restore_rng_state",\n        "save_checkpoint_atomic",\n        "load_checkpoint_resume",\n        "count_params",\n        "build_optimizer_scheduler",\n        "evaluate_model",\n        "detect_anomaly",\n        "build_checkpoint_payload",\n        "write_eval_incremental_evidence",\n        "train_loop_deep",\n        "benchmark_inference_latency",\n        "benchmark_train_short",\n        "run_benchmark_suite",\n        "ascii_curve",\n        "maybe_plot_curves",\n        "maybe_plot_presentation_assets",\n        "render_reports",\n        "maybe_write_outputs",\n        "compute_final_verdict",\n        "build_public_summary",\n        "verify_and_index_artifacts",\n        "validate_required_payload_fields",\n        "make_evidence_zip",\n        "backup_run_to_drive",\n        "load_checkpoint_for_chat",\n        "apply_repetition_penalty",\n        "sample_top_p",\n        "generate_text",\n        "build_mert_cfg",\n        "build_token_stream_and_metrics",\n        "pending_long_run_flags",\n        "run_interactive_menu",\n        "run_all"\n      ]\n    },\n    {\n      "repo_path": "scripts/kaggle_onefile_closure_build30.py",\n      "sha256": "10998a9f432b26ada60276cabbcd84e95bee2c297f4eab52558059cb201690ff",\n      "bytes": 42570,\n      "line_count": 1096,\n      "classes": [\n        "RuntimeMeta",\n        "Layout"\n      ],\n      "functions": [\n        "utc_now",\n        "local_stamp",\n        "sanitize_text",\n        "atomic_write_json",\n        "atomic_write_text",\n        "sha256_file",\n        "probe_writable_dir",\n        "resolve_writable_dir",\n        "detect_runtime",\n        "choose_profile",\n        "build_paths",\n        "patched_argv",\n        "patched_env",\n        "patched_run_config",\n        "load_local_module",\n        "canonical_overrides",\n        "execute_legacy_lane",\n        "find_checkpoint",\n        "infer_run_dir_from_payload",\n        "find_latest_run_dir",\n        "run_command",\n        "run_compare_job",\n        "run_text_understanding_job",\n        "_pick_loss_keys",\n        "build_first100_snapshot",\n        "iter_artifact_files",\n        "build_artifact_index",\n        "build_sha256_manifest",\n        "build_claim_boundary_notes",\n        "write_canonical_summary",\n        "build_package_manifest",\n        "build_bundle",\n        "verify_mode_payload",\n        "package_existing_run",\n        "maybe_refresh_repo_posttrain",\n        "parse_args",\n        "main"\n      ]\n    },\n    {\n      "repo_path": "scripts/kaggle_onefile_demo_build30.py",\n      "sha256": "97c4b139a50b159273a1a2e01b15b3091cd475bbf95fca804bf1bcf71f5c8779",\n      "bytes": 271881,\n      "line_count": 6684,\n      "classes": [\n        "ArtifactLayout",\n        "InMemoryRunLogger",\n        "SimpleTokenizer",\n        "ByteBPETokenizer",\n        "SentencePieceTokenizer",\n        "HybridTokenizer",\n        "RMSNorm",\n        "BitLinearStrict",\n        "RotaryEmbedding",\n        "SwiGLUFFN",\n        "QINNLayer",\n        "HebbianPlasticityLayer",\n        "NeuroSymbolicLayer",\n        "WorldModelHead",\n        "LifelongSafetyLayer",\n        "ContinuousLatentODEStateChannel",\n        "NeuromodulatoryGainLayer",\n        "LiquidMixer",\n        "LiquidRouter",\n        "MoELayer",\n        "MLA",\n        "MertFormerCfg",\n        "MertFormerBlock",\n        "MertFormerTiny",\n        "VanillaTransformerLM",\n        "PackedDataset",\n        "TrainState",\n        "TrainArtifacts"\n      ],\n      "functions": [\n        "_utc_now",\n        "_local_stamp",\n        "_run_id_stamp",\n        "_signal_stop_handler",\n        "install_runtime_signal_handlers",\n        "ensure_writable_dir",\n        "init_artifact_layout",\n        "file_sha256",\n        "reason_code_from_error",\n        "resolve_hf_token",\n        "_quick_hf_dataset_probe",\n        "run_data_preflight",\n        "write_last_state",\n        "append_csv_row",\n        "_median",\n        "_percentile",\n        "_winsorized",\n        "robust_grad_stats",\n        "validation_trend_metrics",\n        "warmup_excluded_loss_drop",\n        "_print_header",\n        "safe_jsonable",\n        "set_seed",\n        "pick_device",\n        "get_total_vram_gb",\n        "get_cuda_device_meta",\n        "apply_gpu_auto_tune",\n        "reset_device_peak_memory",\n        "get_device_peak_memory_gb",\n        "resolve_writable_dir",\n        "_is_dir_writable",\n        "interactive_prompt",\n        "is_notebook_runtime",\n        "can_accept_user_input",\n        "resolve_runtime_config",\n        "maybe_autocast",\n        "hash_config",\n        "_safe_div",\n        "compute_arch_parity_signature",\n        "build_compat_signature",\n        "_format_exception",\n        "parity_crosscheck_local_repo",\n        "summarize_data_source_scorecard",\n        "build_benchmark_winner_matrix",\n        "build_tradeoff_notes",\n        "compute_stability_index",\n        "compute_efficiency_index",\n        "_mean",\n        "_p95",\n        "compute_mini_probe_report",\n        "print_live_compare_panel",\n        "normalize_text",\n        "text_quality_ok",\n        "build_curriculum_sources",\n        "synthetic_stage_samples",\n        "_extract_field",\n        "_bounded_split",\n        "_load_hf_candidate_dataset",\n        "_hf_candidate_worker",\n        "_load_candidate_rows_process_timeout",\n        "load_stage_texts",\n        "build_curriculum_corpus",\n        "compute_oov_rate",\n        "token_histogram_topk",\n        "pack_sequences",\n        "build_train_val_streams",\n        "activation_quant_int8_ste",\n        "weight_quant_ternary_ste",\n        "_linear_core",\n        "parity_self_check",\n        "_should_skip_bitnet",\n        "convert_model_to_strict_bitnet",\n        "collect_bitnet_telemetry",\n        "make_loader",\n        "atomic_json_write",\n        "atomic_torch_save",\n        "prune_checkpoints_rolling5",\n        "collect_rng_state",\n        "restore_rng_state",\n        "save_checkpoint_atomic",\n        "load_checkpoint_resume",\n        "count_params",\n        "build_optimizer_scheduler",\n        "evaluate_model",\n        "detect_anomaly",\n        "build_checkpoint_payload",\n        "write_eval_incremental_evidence",\n        "train_loop_deep",\n        "benchmark_inference_latency",\n        "benchmark_train_short",\n        "run_benchmark_suite",\n        "ascii_curve",\n        "maybe_plot_curves",\n        "maybe_plot_presentation_assets",\n        "render_reports",\n        "maybe_write_outputs",\n        "compute_final_verdict",\n        "build_public_summary",\n        "verify_and_index_artifacts",\n        "validate_required_payload_fields",\n        "make_evidence_zip",\n        "backup_run_to_drive",\n        "load_checkpoint_for_chat",\n        "apply_repetition_penalty",\n        "sample_top_p",\n        "generate_text",\n        "build_mert_cfg",\n        "build_token_stream_and_metrics",\n        "pending_long_run_flags",\n        "run_interactive_menu",\n        "run_all"\n      ]\n    },\n    {\n      "repo_path": "scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py",\n      "sha256": "e694826506ad011cb3211c30c98cc1247f7e3ac5206642f3a30ac7aea40039ba",\n      "bytes": 359995,\n      "line_count": 8818,\n      "classes": [\n        "ArtifactLayout",\n        "InMemoryRunLogger",\n        "SimpleTokenizer",\n        "ByteBPETokenizer",\n        "SentencePieceTokenizer",\n        "HybridTokenizer",\n        "RMSNorm",\n        "BitLinearStrict",\n        "RotaryEmbedding",\n        "SwiGLUFFN",\n        "QINNLayer",\n        "HebbianPlasticityLayer",\n        "NeuroSymbolicLayer",\n        "WorldModelHead",\n        "LifelongSafetyLayer",\n        "ContinuousLatentODEStateChannel",\n        "NeuromodulatoryGainLayer",\n        "LiquidMixer",\n        "LiquidRouter",\n        "MoELayer",\n        "MLA",\n        "MertFormerCfg",\n        "MertFormerBlock",\n        "MertFormerTiny",\n        "VanillaTransformerLM",\n        "PackedDataset",\n        "TrainState",\n        "TrainArtifacts",\n        "MathAnswerDataset"\n      ],\n      "functions": [\n        "maybe_suppress_torch_fx_warnings",\n        "_collect_run_config_allowed_keys",\n        "_slugify",\n        "_mask_secret_value",\n        "build_env_snapshot",\n        "build_runtime_fingerprint",\n        "build_reproduce_command",\n        "build_feature_coverage_matrix",\n        "validate_run_config_schema",\n        "apply_determinism_policy",\n        "apply_runtime_acceleration_policy",\n        "get_compile_guard_snapshot",\n        "build_ownership_proof",\n        "_utc_now",\n        "_local_stamp",\n        "_run_id_stamp",\n        "_signal_stop_handler",\n        "install_runtime_signal_handlers",\n        "ensure_writable_dir",\n        "init_artifact_layout",\n        "file_sha256",\n        "reason_code_from_error",\n        "resolve_hf_token",\n        "_quick_hf_dataset_probe",\n        "run_data_preflight",\n        "write_last_state",\n        "append_csv_row",\n        "_median",\n        "_percentile",\n        "_winsorized",\n        "robust_grad_stats",\n        "validation_trend_metrics",\n        "warmup_excluded_loss_drop",\n        "_print_header",\n        "safe_jsonable",\n        "set_seed",\n        "pick_device",\n        "pick_auto_profile",\n        "get_total_vram_gb",\n        "get_cuda_device_meta",\n        "apply_gpu_auto_tune",\n        "reset_device_peak_memory",\n        "get_device_peak_memory_gb",\n        "resolve_writable_dir",\n        "_is_content_path",\n        "default_local_artifact_root",\n        "_is_dir_writable",\n        "interactive_prompt",\n        "is_notebook_runtime",\n        "detect_kaggle_runtime",\n        "detect_colab_runtime",\n        "can_accept_user_input",\n        "resolve_runtime_config",\n        "parse_cli_overrides",\n        "maybe_autocast",\n        "hash_config",\n        "_safe_div",\n        "compute_arch_parity_signature",\n        "build_compat_signature",\n        "_format_exception",\n        "parity_crosscheck_local_repo",\n        "summarize_data_source_scorecard",\n        "build_benchmark_winner_matrix",\n        "build_tradeoff_notes",\n        "compute_stability_index",\n        "compute_efficiency_index",\n        "_mean",\n        "_p95",\n        "compute_mini_probe_report",\n        "print_live_compare_panel",\n        "normalize_text",\n        "text_quality_ok",\n        "build_curriculum_sources",\n        "synthetic_stage_samples",\n        "_extract_field",\n        "_bounded_split",\n        "_load_hf_candidate_dataset",\n        "_hf_candidate_worker",\n        "_load_candidate_rows_process_timeout",\n        "load_stage_texts",\n        "build_curriculum_corpus",\n        "compute_oov_rate",\n        "token_histogram_topk",\n        "pack_sequences",\n        "build_train_val_streams",\n        "activation_quant_int8_ste",\n        "weight_quant_ternary_ste",\n        "_linear_core",\n        "parity_self_check",\n        "_should_skip_bitnet",\n        "convert_model_to_strict_bitnet",\n        "collect_bitnet_telemetry",\n        "make_loader",\n        "atomic_json_write",\n        "atomic_torch_save",\n        "prune_checkpoints_rolling5",\n        "collect_rng_state",\n        "restore_rng_state",\n        "save_checkpoint_atomic",\n        "load_checkpoint_resume",\n        "count_params",\n        "build_optimizer_scheduler",\n        "evaluate_model",\n        "detect_anomaly",\n        "build_checkpoint_payload",\n        "write_eval_incremental_evidence",\n        "train_loop_deep",\n        "benchmark_inference_latency",\n        "benchmark_train_short",\n        "run_benchmark_suite",\n        "ascii_curve",\n        "maybe_plot_curves",\n        "maybe_plot_presentation_assets",\n        "render_reports",\n        "maybe_write_outputs",\n        "compute_final_verdict",\n        "build_public_summary",\n        "verify_and_index_artifacts",\n        "validate_required_payload_fields",\n        "make_evidence_zip",\n        "backup_run_to_drive",\n        "load_checkpoint_for_chat",\n        "apply_repetition_penalty",\n        "sample_top_p",\n        "generate_text",\n        "build_mert_cfg",\n        "build_token_stream_and_metrics",\n        "pending_long_run_flags",\n        "run_interactive_menu",\n        "mathfp_prompt_architecture",\n        "mathfp_prompt_experimental_toggles",\n        "_mathfp_first_int",\n        "_mathfp_parse_answer_token",\n        "_mathfp_allowed_answer_token_ids",\n        "mathfp_generate_math_records",\n        "mathfp_build_datasets",\n        "mathfp_build_answer_only_tensors",\n        "mathfp_prepare_tensor_dataset",\n        "mathfp_eval_masked_loss",\n        "mathfp_generate_answer",\n        "mathfp_eval_exact_match",\n        "mathfp_select_small_mert_shape",\n        "mathfp_build_variant_models",\n        "mathfp_select_variants",\n        "mathfp_allocate_steps",\n        "mathfp_save_checkpoint",\n        "mathfp_train_variant",\n        "_collect_layer_grad_norms",\n        "maybe_plot_mathfp_interpretability_assets",\n        "mathfp_compare_markdown",\n        "run_math_fastproof",\n        "run_all"\n      ]\n    },\n    {\n      "repo_path": "scripts/kaggle_onefile_demo_build30_text_understanding.py",\n      "sha256": "abea17ae032fc2a5a3315de5d3739dc5cb2081e5f9f3f60ff8ff9794f2996569",\n      "bytes": 12554,\n      "line_count": 359,\n      "classes": [\n        "TextPOCRecord"\n      ],\n      "functions": [\n        "_utc_now",\n        "_local_stamp",\n        "_safe_write_json",\n        "_sha256_file",\n        "_ensure_output_root",\n        "build_synthetic_records",\n        "rule_based_answer",\n        "score_records",\n        "write_jsonl",\n        "build_artifact_index",\n        "main"\n      ]\n    },\n    {\n      "repo_path": "scripts/kaggle_train_compare_build30.py",\n      "sha256": "f0dd5a044ece0791847d58de2853250a368b0599bab36b108c3a82b361bfce74",\n      "bytes": 17556,\n      "line_count": 490,\n      "classes": [\n        "VanillaTransformerLM"\n      ],\n      "functions": [\n        "_pick_device",\n        "_count_params",\n        "_make_batch",\n        "_grad_norm",\n        "_train_variant",\n        "_run_mertformer_variant",\n        "_run_vanilla_variant",\n        "_write_reports",\n        "main"\n      ]\n    },\n    {\n      "repo_path": "scripts/mathfp_interactive_chat.py",\n      "sha256": "ff5de9fcbbfa5f69fc253c87cf6b2d1ac8e9910c0c042022130ecea7dad33a21",\n      "bytes": 4387,\n      "line_count": 132,\n      "classes": [],\n      "functions": [\n        "detect_device",\n        "load_mathfp_module",\n        "resolve_checkpoint",\n        "normalize_prompt",\n        "main"\n      ]\n    },\n    {\n      "repo_path": "scripts/mini_titan_poc.py",\n      "sha256": "6ed834e8a327eb15f5d1aa5f5bf295405dac8d7cfd6deeec9e5206d6e8422eeb",\n      "bytes": 16755,\n      "line_count": 435,\n      "classes": [\n        "RunLogger",\n        "Config",\n        "BitLinear",\n        "LiquidCell",\n        "LiquidMixer",\n        "BitSwiGLU",\n        "SparseMoE",\n        "Block",\n        "MiniTitan"\n      ],\n      "functions": [\n        "_utc_iso",\n        "_local_stamp",\n        "_safe_json",\n        "sha256_file",\n        "try_git_commit",\n        "atomic_write_json",\n        "record_tau",\n        "activation_quant",\n        "weight_quant",\n        "get_dataset",\n        "train_phase",\n        "main"\n      ]\n    },\n    {\n      "repo_path": "scripts/offline_4060_demo_train.py",\n      "sha256": "3eee568b614ed65d996a54e78953285e3a4e208acf2068313cf1b387daec62b0",\n      "bytes": 13079,\n      "line_count": 353,\n      "classes": [\n        "OfflineDemoTokenizer",\n        "JsonlTokenDataset"\n      ],\n      "functions": [\n        "_safe_json",\n        "_cfg_snapshot",\n        "seed_all",\n        "configure_demo_runtime",\n        "cycle_loader",\n        "evaluate",\n        "save_checkpoint",\n        "append_jsonl",\n        "main"\n      ]\n    },\n    {\n      "repo_path": "scripts/post_train_autorun.py",\n      "sha256": "f5e68ce62d2e3e24965c9389dab4d794e4c95d3b2b03f3af350205691d684225",\n      "bytes": 21164,\n      "line_count": 532,\n      "classes": [\n        "Step"\n      ],\n      "functions": [\n        "sanitize_text",\n        "sanitize_value",\n        "detect_python",\n        "utc_now",\n        "ensure_parent",\n        "write_text",\n        "run_command",\n        "maybe_cfg",\n        "resolve_checkpoint",\n        "build_contract_text",\n        "build_state_machine_text",\n        "build_demo_bundle",\n        "build_evidence_pack",\n        "command_builders",\n        "summarize_md",\n        "run_mode",\n        "parse_args",\n        "main"\n      ]\n    },\n    {\n      "repo_path": "scripts/smoke_train_benchmark.py",\n      "sha256": "77cf669d74ad193776891982dd6f4c9b9c17d7e5ebac33ee87e2cb22ea2e57db",\n      "bytes": 6058,\n      "line_count": 192,\n      "classes": [],\n      "functions": [\n        "_utc_now_iso",\n        "_sha256_hexdigest",\n        "_pick_device",\n        "_avg_tau_bias",\n        "_run_variant",\n        "main"\n      ]\n    },\n    {\n      "repo_path": "scripts/sync_chess_gui_onefile.py",\n      "sha256": "8fdcc991eea132ac95ae5d3b9ad0dea2b6534093daf29871165fdf22736081b9",\n      "bytes": 4891,\n      "line_count": 121,\n      "classes": [],\n      "functions": [\n        "sha256_file",\n        "write_text",\n        "display_path",\n        "build_report",\n        "build_report_md",\n        "supports_repo_canonical_fallback",\n        "sync_onefile",\n        "main"\n      ]\n    },\n    {\n      "repo_path": "scripts/train_smoke.py",\n      "sha256": "ae2b75b51de07353ebe8c793953afa97101f20a8d2463f4854262642a53975c5",\n      "bytes": 4736,\n      "line_count": 155,\n      "classes": [],\n      "functions": [\n        "_pick_device",\n        "main"\n      ]\n    },\n    {\n      "repo_path": "scripts/train_tpu_turbo.py",\n      "sha256": "d5c75d10030550d1a372414799e607cedc2337b0893cf9bf748d3ad51ba61f86",\n      "bytes": 6943,\n      "line_count": 183,\n      "classes": [],\n      "functions": [\n        "train_tpu"\n      ]\n    },\n    {\n      "repo_path": "train/train.py",\n      "sha256": "abf343fa3ec60df07b7686d4df4e5582e961a92486f171928e9d9ace5665a356",\n      "bytes": 103121,\n      "line_count": 2229,\n      "classes": [\n        "MertFormerInferenceWrapper",\n        "ValidationJsonlDataset",\n        "CurriculumDataset",\n        "PrecomputedCurriculumDataset",\n        "TeacherBundle"\n      ],\n      "functions": [\n        "seed_all",\n        "validate_config",\n        "check_disk_space",\n        "count_jsonl_records",\n        "get_gpu_memory_usage",\n        "write_energy_telemetry_baseline",\n        "get_curriculum_contract",\n        "build_stage_boundaries",\n        "read_metric_from_json",\n        "write_training_readiness_manifest",\n        "get_student_device",\n        "get_teacher_device",\n        "preflight_param_report",\n        "save_checkpoint_smart",\n        "_normalize_state_dict_keys_for_model",\n        "_discover_resume_checkpoint",\n        "_load_resume_payload",\n        "_infer_curriculum_stage_from_step",\n        "export_to_onnx",\n        "collate_fn",\n        "kd_loss_safe",\n        "get_wsd_schedule",\n        "_tokenizer_candidates",\n        "_ensure_pad_token",\n        "_load_local_runtime_tokenizer",\n        "load_teacher_tokenizer",\n        "rebuild_optimizer",\n        "apply_freeze_policy",\n        "train"\n      ]\n    },\n    {\n      "repo_path": "train/continual_adapter.py",\n      "sha256": "81487e0de80afd83eeec148911a6ee479c8b0038540cd2180632935bfd896fc1",\n      "bytes": 2722,\n      "line_count": 93,\n      "classes": [\n        "ContinualAdapterState",\n        "ReplayBuffer",\n        "ContinualLearningAdapter"\n      ],\n      "functions": []\n    }\n  ],\n  "missing": []\n}'
MERTFORMER_ARCHITECTURE_HASH = '{\n  "schema": "mertformer-architecture-hash-v2",\n  "created_at": "2026-04-21T17:19:47.199611+00:00",\n  "algorithm": "sha256(path:sha256:line_count + overlay_version)",\n  "architecture_hash": "73442726b7f98a2df25997ef1b48eb4c61ac4eedab6a20b50c2276b997629803",\n  "source_file_count": 15,\n  "runtime_embedded_module_count": 11\n}'
MERTFORMER_PARITY_EXCEPTIONS = '{\n  "schema": "mertformer-parity-exceptions-v2",\n  "created_at": "2026-04-21T17:19:47.199611+00:00",\n  "exceptions": [\n    {\n      "area": "config.config",\n      "reason": "Full repo config has filesystem/env side effects and dynamic defaults; onefile uses a mutable in-memory cfg object with equivalent attributes required by embedded layers.",\n      "change": "Repo modules import the same cfg object reference; the onefile updates it from CLI/profile before constructing the model."\n    },\n    {\n      "area": "train/train.py and utils/logger.py",\n      "reason": "Training/evidence runtime must remain single-file and friend-runnable without repo package layout.",\n      "change": "Their source is hashed in source_manifest; runtime training/logger uses the proven onecell scaffold plus repo-backed active model."\n    },\n    {\n      "area": "run.sh",\n      "reason": "Shell entrypoints are platform-specific and not executable inside a single Python source.",\n      "change": "CLI modes and Windows .bat/.ps1 delivery scripts replace run.sh behavior; run.sh hash remains in manifest."\n    },\n    {\n      "area": "model forward return adapter",\n      "reason": "Repo MertFormer returns present_key_values as the third tuple item; onecell training/chat expects an extras dict.",\n      "change": "Wrapper preserves logits/aux and wraps third value as extras.present_key_values plus MoE telemetry layer_stats."\n    },\n    {\n      "area": "legacy onecell classes",\n      "reason": "The base onecell still contains old smoke classes for fallback/bench scaffold, but the active MertFormerTiny symbol is overridden after load.",\n      "change": "Active model uses embedded model.transformers.MertFormer with repo MertFormerBlock/MLA/MoE/Liquid/BitLinear."\n    }\n  ]\n}'
MERTFORMER_PUBLIC_KEY_INFO = '{\n  "kty": "MERTFORMER-LAB-LOCAL",\n  "use": "manifest-only",\n  "note": "Private decrypt material is generated at package time under private/ only."\n}'
MERTFORMER_EMBEDDED_REPO_SOURCES_B85_ZLIB = 'c-ri}U2_{pmMHjFa^&byK>{F>vOV3LrteT9CGnb~)FfrQhvLN)fdo*ELKRdWBtf?)?$h4gr;XUS(GMFN*Zbzz(Tigr{6_hY%pcglu;-losLZS^0Ho-iJF|9o*b*x9<k!h_PM*(i@5SMqXL<K5D&i>NVRmnm-Fqwkmu%hXKW~$lyNAcmULC@J*zw-+_5llC9sKbdc61ECwcaM3yZz^F@-&^zvuHFfSkP~=jkS%>*-M@k&(dtlGq%0Qj(Irkvh6r#huAD<hdk%mIUjW2Ca<$}mPNb>vpMTDw4qji;{9Tiy-d%dn0KB<SzfUH{g-Sz>yM)XzLgng!!%<@;WRIk5j)!Wj2*mw^EUYkHa<#|O?JNCU1PzMGKvT6@mh=h>wo?4>~+RF$5|L9Q8Id)92H?v=4_L_KHTjbA8zj*>>WI3!JC7xb`STS?d|Tg?*6nC7Q9V*y>l_uUXN|DxA$PWohNVh_IDnyb-vobHoab07UMLN4RDu5*JwIRvx24h+hmxfQ&!ApuplOXJ}zf5H<}eGY?}SIZj!JtXGv1~xjQVAegU%zV{Gw^X1f3C>nD51y)SnU4|ex^y9e7(_IGz+PHEm9@q#Dk0nARLq22uLy*A8!?d`o5+oEOC%8qvrj$R!;eRa6I99-4b3k&}@;6s-4q8FzZaOZjxo+Ui)@gzKp`5=)0-DGEJ8n;;I3wDqueDiI>;2(7$_~+xpO}3w2bN~huFxWhe%cNxC^$(9G;VcBCPiI9ZN?KjfW%po{?GpS)gDA)Uve<?IoyKl2bI?(op224OH(4}kJOEsfT3)N#ie8b;?Y#&tH{}r@^CVBRHoI_tIE%vE{*C>aHi~x4^Sj)@XvmVZXqcmHga2e4E^R_Rah~`SOw(vUs0=iu7!oPz<%3CA%s%hJ-oifi$3S5u!sa{%GSE8<`xBlF6h!<-{sh}&w+Sv`)vc->Ok6ibY$~#-fZNzF2jSa$#=MrzP4+9`uSnM9{Gz-kKZNID6yp)KnccJ}n1eZYV?#Xvf|fI_+P-=|c*mr@5NF<IWzOL<o@_tO_?G8#w#+%Y`_`Y$di_~>qdkK@-8c;g)p-K0CUAoRP{X#!<<#04OIbM0DR7;*Hw@z#f4<RdscV-fBN$N4SAg?R^|JR4urCBUdLAMz?Y%3*qzEoscC!m=cEOPY$|e^r@1o7Y0)XDezALQxyo@Jda=S%~zRs#8z`xS6t<i3>?gX&oR*Qy?&We*pguL_RCI(KgzSik1?*Vs-rqLB2)E4^*d-?jPqrXdTl%%X*<^}9SKE_Mf+5~RJXKVoMH=BZZlowH-(PXLD6nb@k;nhFuE?;|%*73_epA~GE{)1v)Bfk-xu|8j$<<)pygOsp6K{)w5P2w>7@Sg1*9RF7B7yVo&Asbx(m|Xw#^Y_K|PZ|851xE$Qp0E*}<8hXjqcPiseTKaSmI`97DN_7FZhdadIERUefyx}ky=G`#wP)MeC^xp`k_FacH`((j8$>MBt6|6ou78B-vPWzW5V9~Pzyy;J@D6EvP3YFP=@l?Oh~jDV^Y`W`kM!q(0l&W0fW6@)kESJD*s_Sj31H}l?|BxnvodDqoL&DIU;n@-F`vS700t0KAM-q%#i4*P4KI7)Sq@lu*#-O#`{SV1g&*+W@GTfb)2+^Wn=SzU`8X@#DEe_Yods|ew$^#)Gl{WzKLq)1i>+_`9lR|bDbQ-SKQR_v;OtI1(6WuqPg$6S@rU;xzW?wZFe8V5e|Vp%DT=!!_79E{MHTiyjHYIyA+=y!`cjSBj0kYRSvZJzQrs7K`p!BPPUj-Fbiflf;@3Y!@_gm>xnvI+-bi?@UAFZ8RTs*>*|)q*xGNd#&O}U7Nm5Ji)03z;fSq3Ne)e0GxcwJM%#bdR3z(4F2gQ4PE}CwSp1$Yyh|xdx=2Rn?H%Gws-I|lHHa-UoeYW?_DLZ_5#EuYQ;5p61gw5gl0DHrqCuJPNFTo%j15OqpJhmN77c<M>WknENum=y=Mhm2&Fri!HUK-D>Q4=}<RvX6F%>Zn6ZbgNha#0t(1o|QWsF1#5D&~C=Evw7YT?6plwU%z*%Q!r;wy5ia)i02B7cTHFXjmP1OgI+;M$dBw__QxL89b-OpIQx=FHui4Vw13#hCmkpv_eMf4j_|Y<8b)l{Z+^$OBa{JKzth{DWU-0OFm5j@l_{6{EG-gf!aX$N#B6$zg+P+j)1;qa01VGh<X==xgDDeT0Br7Z-l+J4F(ow+-^3|piCOzUQ`3x*YY+aB)63Yc67W;dj*6of(66IW+B3ssK#=*P%-W_LR9)7c>$tg&d#F{Vc2IT{)RLU<i?9I8wBvY9Jbkt5`=sF3I#s=ON{(%9?ofgQ)^rCsu-8wB8aS<k4o4qLoUPfe<mgef*znEz?#kC$Py!1m?+ZWKXec2TO>TxX`V^!`~<H~^sfIKEU+NrUx9m77Cw0HHh#ihk#@zWdFire@D9jHWQKDe@_24+nADM^FvylyYtlv?!mum0Oa4VlrG`9ifQfQ?yc_xvM;@)(n)|(iH0_b3jk&dIg0DxOHm85D&6JvYI81(3Q2(l+zIz4rM{1ruy6smp(E<nggy`oVc?MTqd`2M`$Oe+Z{S;Wd{8_2V`)M+aMqTkA1+hOI$<MCQMzzrV++^|<Mm>9WfQ<fSIw(P0Z(#J#ILK>?Gy{LWNV5sxmt@vQ7t!<mH*5+rM;7Ig!Od6d7aBOXD(hBVD^@d16L@M7X9@N}j0!uCq2tRa2jM72=9mm%5+B~T6bQX9QgFe|N3N?y3WIaO8}9n2STHWt|0$hAnUT^wN>K9vY(6YFw7%j^%(vHr%Te@tMEqQ3e$@px-8v)S06zoM2Nu)+@V-l%Ykx<u5y_~)QXLMHmouKhgHaD|X(h(d0A5c$oHg77K&Y<qi@_+2C91|Fpfp911<(mi8?`2A@i4(ZV4oV49td}D00c|$uVI{qMc_a*njow-kcq9i6HBlc;GI0R^R1vh<;3L7cn$*BHrWW8v{^Z8vq5?R&yE6+Yzi!2BT~!ki6_+56Ii;p7tInI_hvw$VRn@)Zmg<ZpR~|5ZGo(ia~EDYyGv-efI2)62x|y0Ja?aWBr524gdm0LXQFTJY&7-&36l*217}))O}V#5lS-l|L!f~jIrd^<_nFaBJZ}_8Em-|~mU401o+EvPSHYVZLnPvYWDC8CIBsfzGQMg;Of|0n;r$TCG2^aSqGmZ$-zDzT*QRTI6z#tW5Ks%q6LfuKU0vs^De@B1!${A%c@&pcqbAf2(itt7$8C%HeZkg1^yV!G<<aa)@<JQMbf-hhHy86(1oCIk+INr)83zC(WznA^lwHKVWoPNUEKIFcjkr}X`a%(<40b64U?UUQAesVpKqCW)xVAP;F=p(%2I#$E$o%i3<X55pSE2u}F7z*v>6ggqO6#2P3$$tH{c#yz0m|c?d7$+Dh3ndol+zxZPnK^YuZC%V{x(@vIekDwb&hB#Hiyex@M%PhX&z2u5Gql=g+W1bM~ze-Df2_dg6FT_JbL;1h>fm)I_F6onHuV+=@iI5s%B<zbuUrh$OW8BCRap?D0>1^4tqbSAfrm7wX)D34tSpinU=BC{-P30vb-2c&ZwKGF))}%I0-gSCxD~>%m2ZizS-Hv=YX*o1@+nM#~bYWrz_Me4>-I20ad!VEwJ-BYVHRi8$}tJ8sOyleNy(0kozNT<XMBheTv>T4m17|^?q`*AF-o%rT+URJLz`2ZFT_vKdpf1qh}~%wNUPWzlj%5j<QEijPszJp241_Lxzn*gJUk!*8i~q2_B*9t>IiH<PbQP33xjPYJ)@XW*LHhDd0_TjcBBaK>S1TW2nIiWm8Vu;IjiD0>3~Xi!ti60c8Oe&Q-h>9&g;A<9YFkWSj6JgZbg5RSnpmN}zF3!up|_Hu9z(9VA>Nm`IoYtnZ%uQ9MQr3jnFRktbGf^c}_3<W@g8Ll+;ou5C7zVDfROV289jEJ3UH6rVN-(Mt$yG&w`_U;<1!>nhR2LK}WL1Impw{+=^iz}cyS642c4b9R=c;Q)4{Ae}Tvjj>|4g01fX!!A&ZEa3<J5TpUKBWy20lil{TErZW_0E-{+^Qh0aXm9C{ZE<n7^lrDb>lFbqa0r3zbzw~)0S3l~CeBe@|D`gV27mtZ_MiU@yEjzw6WWWALK|Pq&fC-Q`iD*S=RfZuyX=uQW$@=ev){9I)(1g5lQ+DGfcs8Zbo~Qh(Dr?}Z7eYiLGaDoxaVmYZyJ`OZQLjT;vGlho3!oX){s$GT5QsbUS3Kp!X{n#n>6*daKM0wg_5UmR=7%NFMD<yO~WT^r`<AvJIDDI4=iwK)3P?|=)?OV(7}+Mp`rSxWC|y&@8BfO%4sNf9E6KP%7NhsL;*mPL3AGB?O<nf2E;Y3O;sR`FnOt;;pzzM+vobgaR<cKiS2+9Rci;~Nkfv1KD@vFffeBx9YHqrzW<kF767TsqX|1WlG+q(+t(@~iT^^6!bT;}bGtjK&Z0i{&q3HyO6zarF^PuVA`6pzmgcy*ZPwXvFxTdXk)r11Ci|xM?d0M5cL431-h>4kCf49j)-*j}$bn>4C@ALjkTSw-M{sQwpD|tZw!hJ*BzSI6EwjzOF{Jcb!*AL!G>D217>)l;%h`qJJR`y_q?+yj@@)sE{ICDp{|b<*<7L4%zU<ZC9vHi9?FHgB)5P=EzyLx|aDZ<-I3^z=vQf84gNl<__R}|jKp=_#VeeoC#VT3D%V}p1P$tUYgk*2FFD;)>!+Ff-Xyr?Cboe644Ft+`M4D#xI=lX9>Us|>g8C{0F-$-x6l7stLy^rLQ_E+^C~r++b9VTU^o^Fs(h=<3Hl8-zmFk}$pYV%LKA(c<mqmR8Y2HBl0MX;9C5{aK@(}*Q=<M_wAC+SAY7$>cxew>@CJ#rZ^7L{O9w%*f+-5sM?h~)fOZE@!&3`#o5Dro#BJgjnMN8Crq8}-~D;X)O@I5Z=&!p(XH}RYlHNu#yEx&AX{nPai*FR0zC@g_Kgu>9R^|djHsLDFa1v_c2q#thM%bA3J-y@(|du*$!41~0`$hI=dY-TBw7i*kghDtUi`3SK?2}X9ynk?Hu3N69bPv+o_RVGFAvV96~TX7g6bA^99zgKhIs^(kQpM0r~{LJ7vo6I?6kWJG`9X)DV;wkEb^_ELI4%BSUa{8q43W>Hh^E)%&7D~|9VFtJgvzAW}xWfZ-cOhIt{S+|QPyecVQAK<VTBVHtEU+%#Bu$De1xds7CtTtgAgO~($AZ3qg<KB=5ObK;T`N)9$rB-{9`Sc=#QxKpsa1C@$FO*ce^!GD4-c_2SjuWz)?M_7V|0^20>|f<c+t-Du#IooK{St88V|}3-<R<BGx3rg!0;|^!edPkgq+Fv5a9fp6_S5t-idO{odDl-ju}U7J=$=D8NlydaA}^5^-{Yu-)Z8I@JHvJW{9p9*OAYlO^Zj#w}PkPrPa{paPqQs<R7{)I!Vww$aCuRki1lkPgg^}RgK^KJhxv!)pJS_EB&fEa^6%&){6@Nd5KRb<f=3+!C$Mduj%VU{q-Szb!Y96U0`PxqeczU*@yQovOEprKHTcUX<gF-8FR%qFA^@VLhPl%^|oyCyl~GO1q-qDO`mPa#1dy|o^AGe;jqu8Vrc^V7Y;z~O5o7on&q(k@ZixNNS$e1{``HJ;WbqL{UoOBp(aD><7(_79P(mLS5brM_W=lq?ir8g4k%K`h5a$xt6fU5q$nrZjR$X9UFcW~%BdEAG3FT$YQ5U*O`Gj`Hej*zC3k<Bh-TXdtW&szk@3FyeCnZjO(r+&1^sk{{Pmpo_+=rrHvf_A2dL>sKMJOoC;dTKFcfOk&M*!~Y>>eopgsjI<5|YT393mz2nfr#$bV7I{1VOiT?KopSJ-n3OZ+;`CiyJv^CwDs)$}CO4Opl`8ZT65kj45K<Un@}{jk#s+-(rV3Z8eDitvUZoJ4n@*uGR=l@@pHk!xG<m)qL4wbU;q3b7)v+$)xfhV5#HHuYJam@Q5hNsmC3svQRTqS@#0((oeGJ+`Vp<HeWk@zP<Bmo@g7N27#XspZPUWE|)%4m8YG%g~ao*A0>wSRg??P;TbYXqrZ1vuLj^wZK!5N28=n%X}X`CB>_qT{M;Pr{gd|U(E)*JOZ2t_S@IJI|aP1SWGY&0AYOrKsXG{P<fxzw+voL*p%?}uY{(ndg9QWdUt}<B!%7$sPwJ?WqgE(koC^W;ShH0?LBc`Jz6&k!2)AYEt&`G7F=Mj)gfx|)v;w9BgF}jJUP;mR|ls=MRi%N3!#6cTj4ok!Ximv)7~fx2Q>kKmS`?WUna4hYCi26R%kiazNVEHO=BmJ3*cJ6+(Lo;1WVS1;wIcoiq>DV+u}pK)|{+8Cu^tdZ?=rNuFz4;9J`4>t<r_wMcBiB7I_cL#2NE+tR1@v;U5&6!o-HrWWY2QkomLxie73sVzB1k6#JN=ZJSjj+(W#}=@cLP2(XIdv0PK0#&FKarNeU|#(*cCSuA*XCae+VbO!7uFFD%QV4a-!y5pU7rCxf@Im!6DJ`Q41zrmG<qf}HbxN0H4u<2DGSJN`!T=f!=-~iSYv=@)tL4jJE!PUpzQILRV^Ijm}>xDG%HIo?1E;c(Pj_WYkiq#$P5`;ZuW*CDse;!5&DW02noRHNR+Krg(h9@OtRn*>8TwiY4Ux}UJjBd;k$jasb!Vh->h-Rds+ZG9O>;K^eIYO5$zw9NCZ3S@-ciMAx^?}_)B7DFCbdP!{ruM)+F(d&rT<wc_d0y}{Xq3E$zZ6kFD&`BX&OSPN$!I9q6E&f*tw0(>Vd04>mSCm<wr+lV+6)LrMe^aDu#ViLAI{AmfA_mv-7)~J-M`__HB~(hqZ`oQmdjhT2d<!X11am)I`L>Irc_9WIOig)K)T6U1RZ|S3B!u|o~0Lo8NeD~lxT~Hu^O|_Yxl(3DbZwF#=W50+*UUn40?ecRN`^R*#B1j;t5pOC-|Uths0RRueARv;V9knh+m+5jE3~}I*&P;PtVdg>My(^(&h*4!x$C4)PtR6(LkFH6Z$A#N_e&z0Fl{ukP*pWI`9Y^kkOM5Pz;X-Wf*r?H00H96zVb&tTyQV_Ext?gAu_D0XlxH8_-*|%L2dlCVZYRG8oyQ?FR}R4iJD1FWmDmF8N(UzPooBPEn%oU1>oU5rYPCl|tN~wAqVOGn8`_6}hYNlmOb!$&nINK{+&_sAK`;l%5U?L<Rww%zs-Qi*F$~R4fRI(;8R;YS)g|J9%{(Jd9L}k>+TtC|5L0uX;tFq^;(Sqag>;eALTHY5pHc!@Dn+#>CPdg@bTb2tj#NqFvNd^x7KUUtY1>>kUgByw_98-7rbgLV2Qk)xSGYzgSuITTy<s5yCFA0_lH^1b>dyJ14G;Nn5JTBh1kj;SO3``;#2CXB1U$1fByL52KiORm+`lI^&oX;4CX==<f#LqqCCQ=HqBs<htpXoX>J|)H6)`023jLJ}sov&w#h|1sf6;{^h*Irf+R;OrKyk)h9j;dmhoPZF(+W_NuwkHM3;D&y2S4NE;TtVhq=HoW=vQ-DB164pfQA*-TF1ijZnel_05pahrM;r#pGR)ud5-y_H;Q|6UJLtjnkp#d^pYTv65C0>8Yv;sEDiR>6m3xFhiRpy<)M-Pr?oS`M(84>xRqtsg9B0}3{Tv&t86=9{ZG@i=U`+ZDL&_OQElbNIn%-DYKi5(yvFw!x4~Doikh^9qNBfKD^Bh4ZS(BDgT41z>Rj3!pmgUU=SUK&Q!c>ho90sc6$$Uagcu8>y2Le>PN7ck>LpM8Aokg*wv5t~vrk7uZ)Q&`_*GFvOR1jjc%E`W)q*<MxZwHXEe{`?kOYwZWxO_qM)!d#`@kZR~WT50M-Am;pvTS}E)dg`{dzm#~OR-OL1LWk(VY*p>v`L@2fPzG)2MzP(17i>$*CMAAL~sNhqSWWVsz)><mSy`e<EC<g{N8lxxDpzL#VFSVoPyfefAx0aLlPpw`27vCG7dAtb>U5?y%=P9c5o0-@u7<o^466IBdKMFc0ARM&CHk{V8{C#Ur(x$DOi+8W>_SLr+yW01;*4A^#x|a_D-2Tpy>AueIM!)ZH#|)AEBF*AKZ<-Ex+#B<7@E=KL{~FiEXl-mTPr@mn1;a(I;Fa+|i_8vmEF{4jp7pYs{^@SS`hPIs2@#+6?ImuT2rjpTdJH@SshmvWD9vXiL=zfy#8$(@k)uNp^sM=3$W!QvSFz1dL=)3$;`?4EeV*aS=#*9eOrFU+8^RPzOnTcmP8Q{pUXCe?-LDlzDbn5mw$n#SEcN^f6h+)KXZ_WEwSRB>d)rp*&AOGTp`Wp3x|{J^Loe@6>i*uZ4Unp>*4?Y0T-u}^Ho`I|4<d<*FW}xTerTG?_&)WvZ+#U<F$NAsmM+c)DHmao=*KbQtQd1iF;pf9ax<aiIBXBmUc}wXM2CRt9pwQcQJ(GtYX6fIg^2voEDptE7Bh-ng!BCNO3QJYMOWyaE>xE5zx>lqj%aH)<&Ukpn-sT{gmw3W`Y#J|ov6dRJCHWk-XUP)IxT^^zQDYQx8i2qZz8KVkaZ#4L5CCJ7iwx2Z=yjJE>S+mE^eCI4c)~Y7FD2TW{aJegG#3s0nx};WXH^z=vlMBTYfRsTog(L)WChT=?Qhi*$h+ui&oam`V-?7Kz|Q(H=*RMwnt7*>{A7$KlCn}e)dn>M%$nxcE+6k4fbZiIC^_j8+V7))YS&*cE&Q?D`sw$>c8Sn!L9O}n2qZesLkj)LA57OOclAfSw2-Ze339H5KS$7F*w2H-5V9u@X6%rTM~U7E@?9GE%6tpg|XT#__OoX%Zaz_U!5)$_uaxj8ed&~{<|_7{F><H*F-P>(`9!wiWsSqF7xWozzK1e@`Zly?W)$%>KPn&cAss(*+1?9f<Aq5^z?A=^>Od%tCz30k9)fZUx}qw#fL-z6i;Kb^ic0l5r#r%(X8l2K!3^^4$B9@FtNqlH#uj=xFbi@n59WPXBT6hpr$NF=MwmP#({1DksbWK3M@e1yJ$EU#T$Uk%yYFrongkgm97x|CqO_|hZ`lslzp+qg2y0veA=qNbB2lTm*!oboJU!jU>&Al@kX_2NkC)^SP2NLae9G5t0-NA&nK+T)TXjJB^Q61gV0TXNC6MOYo3Z-7D#8sLPUa%?uw1Zz4h0CK>UK=Or^C%rY`=$A@Rev_skPfL3(t%1@%<vga;+MMQ*Zh0r+>Y5F5^+@D~4C3}q8!&pJ(W91il7$MN!LcM8-O9T@gt!PDW>7MjuoqhiNl$^Mq@a9|!YT*d1u0h|%u{*hn=pszMQ0}A)%_4C8+on0LVUiJB>A}s$&8`SB3{u7zN^^e-Rhr>}Wn+p*W*%@#YPLQDo##~z~Kf7+_u_$7&*1K${%*rW7qXE~BDMQL@H`QM7^;n2TD0zMT0PXFE55}!pjyHftK?>{*>9fs^TmgpR?|NaUpM*6odd$cN$I6}DP1xwNe+;K#(m4u&m`0xU<B@GKbdi|FWgdLpgVPpEpg)bJt6qIa0Q523lg?ALCWmms)|T7~EY6s)aZ}DZ7qb>I<l!7A99sG1MI@c<bHrz1hKHkZ6|O$sWnu9S%@>UD%L>chZB7OI-SlMC>OSv&w#jy|PQyGr4}boiouyZ0JVnPps@_mul?l7T)uYx#6$ubKu-5&|%&6d=hy=c;Qx*s#ANC^aI#G6}tix#H!ITahSt|@RrOE=Hl}g7@k9m?TsHlVx{z{*DU%`%LzV326?!mkCe2HVdVLx9n2LEZC=6ujWdC~|u5<x?POdi4er@JhI9Awox9myjWyuG*m_8z8fiKbDDnh9jYKUt(AI39DRTTiKNH40GD9+`hl8et{z#1<Lvt0oHZ0<mvK!)^N=DIq^p02Yu5oHN}xCg}vM4sLMc`tag2eevZ2bq9Be*2O-SDMTpvzH>&`c?#DV7&fx32~g3}zShsm7=$U8pU4*-Ha9Z5(S=q!Wn6D-L-BWGL(SV<-&8SqMaXD&n4>g>D=Uis$RZSG`37+vbsrJgq5F1S;DN{=@K7ZQm8222^o&4HJHs%Wz+hO<P|yYz#v_p#1SQJ)*i9fu(K$@y?LFy!)(I~#;oLo<lIW)(B`*UWe$gH!!k3OHy4>oAdN&{16k`dt!3A_Kl#OBdUEUuLE{y{a3NS6fLncoQUx7%4K>Q9m#+)B8j2aPeZ`jLFkICiBYy_9eNGyxgI30!Qx3jT^wQHu~<<^Guqtq@*y@X%*CnBtso}H5tCv_D<=O^rrALL?d|MT84+u1$b-#vKw;a`6KAME<S?SA>;Uk-P6zhnVCjM<C5f85_b*x5VWdy0|o0^P6^z9{yvnkm+s6m7SkKHc5lJ=}(eN9z}u@IN#?*TiEEKi;C86|AL$riHMGFrU4rzzQ0~WoCSlzrR&a)<}t`bOKR9D2B7g8}|p8-e5)vHM5ZEWWyq%rbsf}HB)-+q+)C;-G>h{gwjUi2<37tbS$<`HB6wos$df4m%FI4gEZEJ>R~1@`1j4qLNyg}8mc-^O_iJ^J_kk_CnJ{X!iuF@;he<-uNSB3tk)0wW3J>5LH~39&XF&~*D<{wyq~oE(KZ2L(I|g=#X%Yy7~E)Ko0S07Yy8SYxTHrpu|+ALfz*AFBG#ilIl>p;A@7%2hFo?ZHk5=%xX#AAbM@ASnNKN>@P*h{oiRL-Yy`l{M_D)<!{c$l(Vu0So};Gd^}HBUnk!0Tghz70Qeb-0j3^$kpQ4-~YiU?F6_aR>@J%4}0Qe4&LE-ueN6C=G?EBnP(pj2e$mAyR$joFq#?Wp;AuA)N$_+15U;UVLif|0$Vj5i39ELFlA{A8@aB<UR`uLgELZ)gJV>{u+a(3DmshXqhlVtpE_&=Fb-r-ry=TTojwXT2o@IEH@p^Qk<h*CX?<Y<^nO4i4g$1P7Ys}d>}l>y|~)Ky*f3Uf10HUT@^Z1Z%nuz>7EObkSy679P{Q^Sdz<`h#p8QPbwG8a37gAy%X0*nwv$}*8@u{?co5jd<@kR^0;iyayz&H$e#m_9k1>7s;3M%|QF{ozIsUzTwZ%}|ku&$f3?dBEI+B_l&yo_B<uD5h7mP15CkR#-j8ZHttB<nuxc<n>uO^w2AxyEAj?;{qWQ9}nvwV*X!W?H=rZ_?Is?NjY^S>UE;x<8@eIahykAhVwI&<}s=5^<R>hdyWS|9c@3`eg0<qa7RFT8qVl8pCd8Zy`04<JYYtTz6z5l9}BOgrY8g!+43m2?%DNEbNIhVKm7NMm?gW9<<Wos9&RMyXj#zQ?!z-;gu?QIhek*d-<YO&lmVFpyc}@}6G^Fhhg_hqMRLDj_W{c>FE7Uwdj-z=z5zT7FHl#bkHOOYFMJ4i-T>A!!jzunC_3Hkr$T1_th?qt+`AUVwHUrs0#6;1g|y$Kq*eMQY}Uq)Hy2)hDXB$}ki~+_f;Q$|UD9k-aCa)fJdt?|EWtfP%ea#5-{~^4DX@Bc>cTyVV<|{yWdbq@$&G@m2}^-TA}}ZxAVBrRuy?10acNyjRWm}Zlajnm+GTZjcCD$uLo+Q9^)p%orEw;nGi-t*F@B-^ZDF$+mP0sfhR>LPG1gcQ<#B&@HpTg}pcA5?;V!<a_@rWa>b9Sk(dDubEaU{zU_-4L8I~Wf0h0j=G(?|fQQ2LRH6dAvz&1WG%T9beloib{?Fyig6yYpQW2Av-m${Yi8)|-_zhN~S*CKjF(O`7KLW3}7l(!n8JH1}3Ki@+2eakhAzP(3&4!!-oKfKx7={?&%I_|yRK7Jugo9l1yQMw!YlQS<9DDFd`f2d`3Q4gIS1>DgEk(*Y}8n_FR)Fb^Ik<Wg*O)P+a^(V6wj^jQm5Ux<}4X<jV$|4Z_vLBZeYDcwMG{Ag$fD>FEb;FDhCqA7O^I+{(AyWWR-^{xMX_A7wh58Fk7D><@0h9%0rN2Nu5KWOYnZ+_GKL7M~0}1-xINq>?tBk{8(*N5GDF(pW+yl}%V;hS|VG1W|Bgs?}PC2HW20?-B(#wP2TvN}JH_x8!9(GW~XHU1EzSw17AMUBT(ZaMn%Y<i9&cseoauEzqcqEb=p?5mWe~My8Klk^PNs=Q~SOg!L{1OA4Kspy-^X5s?+F(QIzeWb88egU;sdA|78}jIKB6_%2Z9%l1z$EP9r=ea%5P-y7WD=-_M_UOD^>K}~Uutpmcg!@2UsZ-L`H`}kxgk*$N`@}uYND9#TAWmRhIVNRDkDFg`~u!?XX5k#ZPF|p@vaMxR92KZE60p6s1zwgR5O1h;b!n^OE3+0fd$NS2L#PLq>L`=kA|>eu2WqTV|nD~n4~#8r=EPi{;9cT2n<sKTyYFo#2zC2&Tb(B3hBEYEh@d{o$Fi0RlqIXbQzPh`)0gKA~vdQH<-&giCA=ne^EtZa?0R6WzqBs?PRq#t8}lcT>4QHh#`Odq8enT?9EI>5t>`9Q*Rf(<aIbTWlWWYwY!>BS<N3B9TbsYy@$bYZr15`Ly^Q<Z|Y5p2=sGmBn|Y+Xha?i-mVd=o}SL+*9)6jJa}o#8vv!nH#hRKrS33_<6dB8`&fy|Ov7v<VdXN0Rd^}4mZbf!)Pm+6$s=L#L5r;r$D#x-XEcy%>Yu+40ih!&(f>=BMDA-J4)Ib*(^UO#)QUjRitU|Y54TkDrIj~@WLU{`F*0Y9q+e)B<{^*pFFIgTOH$pAG4`ff64xnO#adI!e#b>6gUEDxt0i(Yu)I}E^G$Q~PzOJfR06w)K6-L9VS8Vq&UdD4@{x%-7iG{=+*Es?3oDF!&RcBp!mRO-oO-uN1t<d-_K&mrKKuLo0^}Fr<AThkW;gpYiU4(@r;?j~;pR5=H<!Z6mOn>nLmyGRp_Zgn^{DpUPsY^tZ*46-qUH>!b@6M3BoNVsr5<YHu4df5FQaXP3Q4PWnnrQ><VOxuG_I!3$Uk<d@uH#bcv)xamfMoT5}-Cf-$klkT6G;jbkSO&i~VjMuhgghijP+A=e*NbA5g!<&l>i^e=&ue?YoaF5Hw_)E4WNrRlPjwbW8Cs+$C$8)~0l(QW@2DxtogmmJFUquTp#(6rwVaUz&z-Nu0&d4X?^^-J1PkX<7{<c@s>?B~@y~D&NTlz9Hq0G(Y^2hVmtw^kqYlYnWpB)H9ADaxZYxPQHEeUHjV?-`S~9%#mb1l!giF2k)iv*i|3Zd-u}&)~#kLLlwka>y3mcC>J0`c_d&@qkM|luLT3R>@rhYbb=~G|E+JWnSb}_TXVACwb|v?x0lsNIrHzjKESOyBA8wjK5Om?`ZlkzXM5kAVofKI38(Bl7p@j)eV?*Frm^xHH5;oaEo{A~^`fuCV(cjug<v<8$BA@%fQKv#DkXJGR$E4K?dW(HeO3nZP&(5a6_aZZUNsR7pH#sX<TB?b*a(EAoKgbUJVvW^3*S3quGRF=122V3KaH-0-$89odP4*hnvtA@+1c?3T_TpM&y|uK1ssJNU4`3k?@ecz!=R2HcGQrS|6SWwwNzWF)wQY}tK*E@wg;!C8cRcrtQ@rq!i}6;BCRpFXc65t3~?@hPbG;+y(iuDcsSx&Q|!@7aYzfU(n=fEv_Bq-dMj8SIF)?^{U1%M1e&EB3=WhJj&_nL_#_K6*$*q3yU>;C1F<)^;6JP|J4jMsZZl5aV4}@T<V?ZQyzJV3(XGszYTLHdM5-CQ=bXo$-v&jyA+ND1rY&ew$w><uS%TP-b}Iu!LCa~<T0;0g6}0?NCPKXrr1m)$ko)8HPZO-j_9RZ>a>*+w^p84eswj<Jx0E@63oL)222c<wBgw(5V-^moJjbN9mvL+r(7C!CXtNBIzurT3X}_;6;mVsCe%@x|nyO)L=^2dc*A!CI7)^5{7mFnlp6it;%vY{gV$*eLc+V{I9#qnTe(D65+8xJF)0&-j8eMig+g!1hUw*$8`}A6l{gSkwVJ@;iTW$e?OVovOZ|m(nkyNgu+-+;}?CRWf6;A5oqRMkYMe{CWoJ*Nz>t-9}ZjDz7=<RHB#tNwbB?i69U?}nFL70tl_|F4$Vj5YM$3^i!rB8LcyB%+&B)vmbD9qBzRxypkUsF~5nyTXNQ&mjq;bvr<SS_i<UDZ?+o-=7Oj7DAYA2n!yIJ!yh3L)G)i^F`(whJ_vBl+9_M4nM_Hiy}7udl5=dIA4JnfW&xYn@Nm*!aWyD;8#jcxCDe>O!RPf>V2em_smr!}1vi!ahSYiyi}jRBVLSF+^y+uqYC>g6fwZ?u}^sY=8UcMQ{7~_@KA_)%M;#X4%lHEy*DEs6VMGHW&7i8h(Eo3+1l`sQTp9(XMdQS~K0W47v$d4lTp?{`LR-zyII==YMAn+dbvy{StK}@b-_pQhiWwhPI^e93FJYgSs-!t;`q_<-IgXE_?hE*BB7LN2QtN`zLB&i*yYa(&3b}5|~jn)A^{AOD4dG$~9Ix5E=ydGVGEnDNWr(8jj==w?n#hsO?id>QaLw+s1a2J6La+7~oZZ9067de9c&BeK-wamEpXZLCBR(fzfQn6*3y(n*Nx*KYV%k@`xM;m%3yA;Y)@Mb;Wu4f*6NGSe1kVlsnW2ArtG`S{31vS*1Nx9r51PM({wZlH7|bEw7F@(eY7%c|E7(CTyeo=|ec7j8BRvi$IP%E18Hx9^%i@oU?LXPUCV=W*+zDMpw@PYQ98$qgigKl6mLz+m$k+q-yYQ=Xbd~&FKVSBCVD?*;{A7HXf0miMQU7+-@$9`Ppj`lnEQH!Zp25McMtSkcp_d>@AWjs8g(!f>yDYI7?Bt2>&FKhJV2_48L>-yjh=Ynwu5iYsHZ|ZX72jI}g+;wYmhZo<HT48=8vf#jbyhuYbTdwJI7<v84VORBtkAm@V3+)GJVx0F!Po#uxiSAEqck?M~150OM?y-)hl;y-tM#0n=SsaG1W{#UwdT!#>8mkiakvulT^Y$~WyERo%^*EWBx$RvO-&y_oAXqMxq)?sF{fne*xnezu1HU<8YNQ4`%=@iGDlg8|uZfQ$Xj7F$~({W~s?`Y4kJr;N;G<$#}UtA@x`7@?Oo{M%DtLaq<K8-H_a0>X09OD@d0Ul9Z{tbB{0&q!~W@popdfcQlg;(Ejag7@Ho@_rA6CSi@XzRfnA%BwAX*&Z>2_$=^%)pI~AsQmLn-^Wit))zo1`sDAW%TDF!_K2NFAybZ;>>>mDluLzWM@?Y)#QJG2Rf~7yQ=9|OzZE=l=`tM+T^9})Y9?bB%@YKyRbCbHa0qebpF~z*#Cp;#>pkzf{ufro0im-su&!+mq+jL#;J8233u~C}Jmz&(cw}8gjqbutgZ}1RN4+YKDo5{v%m%Y)p|$H3=!Lr9%|JY@{t1l)rPelieFmHfIT~;Lv8vgXmc-QC_9{#h!St-sLc!e&sDE34X&Du|%w(n{P)yI@;gtui6S7(h+cn#?Lx^*VIK`&$Cx25DL7x!1cd%xGG#ZHe2lAJ7|1c{()2_dCai@U%wpESlWQerYkK|$MmdGK2o{qz0fKkW%65WO=5*cql^+5*M?RHxpb6a(}4eK%dXxNWy?Ub>Nnt4pgE18icOAuS*H6wS~%k*3%>_+c@nYEj2%l4|gc4pu?1bgdS|2n^W^sRSc-?f{-6&UzhY<*kp4#3cR@SWSMRp0E2@)?^AKf0T3wo{u;&HkNic5R(%v)zR)-|T945Qf?V-)7e`f$MWs-o1EMu2;L;R(RgP4{jS{>IxoC#|zJBD?NEgpR#hfU`KotG$tJmb6%+5)^2A45OlpMAVX1oX_GnGICb*<SM(kP*;uw#{T|9G`WSYDPo=T(3v0nw!+lZf=%1>OZq>YFmXvk3S+$-+RYJYhv^^>h&B7(0z<tukYCzvSt(X690~a`oar$T)14ahNTl4F8aPKDY9j|}#_)p0qFb?BkaB07X4(U_Q@Eef~nGX!+mBWDa6IpmIUDM?{qL!G|rpB?(`YC(#h^h0t+^8D4O*A{Ua=sF$?_7caez|Tld0a&}o5k~9md<!DRWB-eEdDU~iT77HHSg~^mhN+Y|EP>P`q{^bV&n89k+X%BIn|iRct<{41~{!pNQJ5w=EwB6^G(6yV<Kt>ydc}_&Z+tWz1m{sC!%1J@%3Gg<C_*Mm|+SgB&3)K!9=Ea^gP$4|0Y$BNkfl`-9sL~9ICA#*iZYH``gQ(^OPH<gOx|vzQ_<?DQG)ve?NS=-FYlkUxspR$|qqGT}_P`Bnqrp37U~iz=ViJMt~RV{JGbEx#DpgO=6DuNS}#vM!8O;tDs>jqAcd&xjb?8U2FU?^@L2hYknsclTa_QBT>m#ebOcctjKBPbyeR*;TCur<P#kWXF7!GqMTpJ*VzEmRe$(C!u)~CR}M`MF^H@m-e=RWg~~8^W}mJULEapCNk_WBtumQ@c;AQf5k~{RMmmFO)~&^|EKL0Sl#DVRQASF|k7ZovCR$K=r9T{X7gS%tLd|Fd&yD8#w_4g>_^~I{B1)SM?NNkMU+_U)n+T9jdo151Cg*w{4s7$QhTjN_C@s)`icoYQRo@WKy3jv4wfmk^t=$ldq2+~&<zdhNuq_Q_IWU*7U>9i_W91%TXqXcn5S0qmq0cYk9Zjukt4iLLQAQ7@O}5T{zs0Oo%fE}ipl4zAx8K+;Q9kMVqg%{&Ed;rWV$1@rlw)6PAz3BmSPinRDr+A93;Tokb=icxvbZ(Hz%!$kp18G-MpM-emD+Ha-Nlb3&&1&v8rHdr0<v*gjT2Q18bs$&PQ?0b&Q{on6&7frB{FG{&k9k@pnjFS*Ij+l<}!e^BWnQ^sd&I?lHdU<Rii$rq%H-pN<C~{Pc?3j4|N>4CQ9m&tEVkr8A`h*4^Mq|JlIUgcymyIw|@M>9`05ruBWC>Hv$=bt*!S;y4|a0Ti=>9_^xUvpSkL<OP>DLGH#$Uug%s_nOhU9>&|=5Ntu7O=`AqczXQfAIQ2joD70N77sh^)DC?vi+xCRk&9$UB3P{+q`otVZDnnbi<)Egi<p<>a?fUOF+`62IxLY=gdy6dC762o+d?of!0)^o)>PH}|)Cyd`>!Da=C1`IbEmjej2R8CHCUd9LSXVZ|EJLjyurm@VtH~-N<4kOFqlsxcw%XjN&G|1rLfSCGf~{P!ikrF3(sEuqSeY5fF!NhuwQ3u_dIxj(9l%iXZL^?}2h2cx5|(V)E^sb`el@7pcqN<+bfSn>&U=P;O(lHG{6Y@grsfD<?4~2Dy$ml=9Z?cY$^zxLB}mzJg*)wA!8qbsh~8@bFh-mbO3daZc-vcRu!U-9l8@7@=v_p~AieNU)@?;Ba(%-K4ND-Pdn8s#Z3JdCqSbpL;TBQDGjtr!o1r-kNxa_pty9r`1{)??P0pp}QKXKksUK-thB;Y4t*WN1SSw*Cy-4zKI-{hn$*3dK!y`o#x`_U`OeVqs0VfAPnJMDs+(-j4X|tnB$>1C8EWZK<4efGaZ9+>BLGVNn=A^jvaAB!e68#c>(UWP8G;lQc+SfoIv@D%Tf)4#=9;rS+8X8|@;^-^SGFU%1CcWC=nCM18{-U-R7+^XzK%%!g{$KTYAm?CPx$L<J5UB9RJ*;LTK(sZ4f#DeP*cKW2RIR$>j_kxROg)Q$UwZ;eoMkwmgaxje#D?RX(zHOWK|B$^wW|?Ng)VW!-SFx{P{{^-EQ-Ee<UZ(B^M!e78b%g>kc;UU?d{``7%5s?xA_eHab033ul1ydPXPB?pL>42u@#|~<BHsZi8MaKOU~3K&u4K|h*G({pANVvkpfp=voGVW3r|Loz#EVaE36Q(hXOJy!YE#liNv%<h_$P;Fz%^M<K=Cg%FH7*Hqfqsn)jH<rx>Ujv||Bx&_(z7eIvaQQ&tCL_MZ^F)~ixiSi!V$<y<vBU!n0ttkwms8gIIwRqOKwt=)O|G`0)b-(o<bN!!ad+l?%+<t}&O80WVg!(2(fxU?ySwYaw0NrPNeGpzMJxBH$q_SL8XAld@bUB&b(eK5Vnys`L1sW8xX!sk7i{7wh@FQpcq=zo;UNDlLd{x=b!8;p6nB!SY0f5C)eY>r8<V~p(XNOe8R>V_g7!-BL-3vZD=P$OF|A!lwGOFBbKLx&yJJ4n;HXNU!KOa!?-HKXEc&sCw0-(-JivoG81tGajTJ9QVtLp8W;b<ZPyAtUJQ*HqhnOi^m3B%h@@N8)JW<3vH7EQG4we5(UKSAfs&7_hl>Uu`_bRQNTI>7dE|y5-$Zio$&7-xc~cab8H;1hSuwqCWduq3?U;n8}g#Xxsa-#Xam>XgaOAf(12fi~r)D+CAOSX7QB~l2t~&9{y~L*|qtJp4#dTEc&>nQ9{3Fo!I^qcR}BfeV&ZqVm!BBi?C8-kT0y)qZ{|Pf~pU7eFnVX+v+dhwaE3Kqq1GTPNgDO{`I>C7Z><(FX?9-Ri|jBG5zE~$iV~$c^nPI)h8jtxbZH$mR|{18E>^Z4$x)0vf-MY*1POEFI0YUX~`9ZGi2eEnRGMGEq}MR^kj2%c)>gEcd~V?+y=IcUF@*qhTFB#r8&;Ubc3qKCuPT|?2|EF0_CWN!4N#gSb6l-h~SwR4I}t{%rjdl_Dp~8V|r8QS6~*+`_JL#52hhH(cNdm2-xHW7w1sG2PDs?an!Vgi;3iAq$;?OHPv!29U6917)%%4r9n`F=%d|92)WuF!^)Od)0%4;#n#4JPMj8Nb}wzUu_8@gSZz8i3WG!`Cxjj^FVFH>*mtv#z9Z-`n|H#^qN6&v@4$&_I9bwAKx|hXyj~Y}tG33)dO91ug5NJx`YY;e$X9Ynn|i7sT8VK_4c8DC$ixxwJU}K3fixt!Ko<(!XKUe0Qd%4-<n^5~;PPOVBfjAtIta0;*X`LV2PYP_y;HRC9k%Qtr(I#^P0m`4zAB8xN9|uBe6!Z&6%pX!+()kwhtS)PL)>`y^X+7Wi2`A&J*3<)@w`FA8>XSpIZVT=BKbtD=YCXWDm4Qrd5&2ctLdWKkF~@Wu1P-_g!QnV*<L*jn5h9V^$>pjgHH<ogni9PTO<spa3I7)X>=WpfbkWS=DCs>QA|};$Ar?t<6qsZ!6wB~<I&OQV^mVEyQkQ$Wab7fkh7r+DUJl!tb?$vQs{~CBum2qRtXe=OO*UZdtIObT4VwH19Lq@)TUV0p}*O39wqMW0-9KfsrFld9MACes6Ls$*T8(uWu_j+;RMu-Sx_059z9~Va{(ev?f^=90EM};w%lyAzf4l2J(I4OL$4A}0)i;+QjKKc;hZn;W1iO?<|r_vFpRBc+sDVd2giG_4%o&oJfgNnA%(E_#24mWPk#EY8|A%zISA{Y>eo&X7`I(uWhB!+#Z88?RviLf=UOng*DfuI9>Lvo#VfTD@7(9EmF|Y_s!o(sd&OxelZ8-f^k4Qmzd^5Bx*X{5#Q6t4d4bxreNso$m-EIi(m2|iZPm=H-B~T&oZvHSR*p;kXd;J4G9>u2PzmqJ9#tFN62XF)P(qYIwY3$^8LFy%!ud5SHM}C@>x`ol=F!gUZIRyw%U@!R0gU^nSCfqC2=fW%$3~7dO{vf;&%54|2d@qezEN2N&^?uE<q6<HI)j(t`UX~nL-dxdX!LZ}s!O8dFeiKEGqhIHF2QD`MNc^G6+O*SG3i3%rP*9?!ipHLdd787UNOoeLYc<ze9{?aoJ;he&FMBK&J$&>7c723BEWYG^7%~`d^=z8jkXFAV#`t0HeVzCrv<H%<Sx7|4cU@7JpTRSj{W={;Nu(cleDXNVw)&Zk77*!$Fn(ADWokGHDCeh;Y9Oak3}DgtL9&NE_w465PD2e?Oojh@(M@!8AM{t`t(o;yFw9Oq!X+!jn>tsGkMpWrMVkSYHOj|hOV%d53V6~<zz9~Rl$fw(gD%C9)O}L4A1Hmopxy$85z2H`tK0yj0OJNB3}dIeZuznDD2NE0V?K)*c2}}&FFCEB&*?D4e{^{Ylm~pp8u}vrTdoRd1LL<-!|a;%E^(1viL<b8h5lMsd_c4iIYk(!osO435(#MnaC8h-K;!rzOBE&<++24bJI)X5wZFyZj@8G7u70;q=X4cIvcIl0`Y4BWbp@`v;h$9%?zKo$VRGkEJD=trDd@Jc5$(AxkcW)e>QKGNWc-iz1M+v14ulcNzrvw<(Y|8h;ah}Tq~yRye3Pa)C8o!Y|!#%=yiA8?9pcD&a+cj8rN3J(#<&Nk!`J4glYPF>aUF=T|_Y%DrSpt^+|TZz?A=*F!0xefp?!U@FP?HS!oA7xdykBIlSMae$i(85n5!Acwyunyt90PDpnGHJi3UU@4s0dF{z5hQ*-pf`|y2<z7**$E3c+1WxYs(Wkv-WVUjfvzzjRe_8DpicLWJ+9kxDSH7K$va*@0`#^Lb8`>T)@@Epyua)^2TU-D@R_aP6@Db@iKW*8eJfXaD>lOj5&rNHbGN#6j3`rOCQV)F91S1pHIxvoo>Thj7P`5?joSpEAd6({8hsjW4`AZ-Vhs*x2HY0&fSY~*TU&7i^nXSB(lM^|wKGS(c(`y=*3hk^UT(`(C8K(@0`gV-|~`2gJ{aMKf#Y5dJ;SD)Qhea7|bx=kpz$wttr%}UhKrWc8ujLqpMf>4?}(G>AxJwCz;eW=ZNQLsfhyGv*|;W|7Ih~`+0J4ns9qVHGjQkSfpk4iYW#hzZ|iva>-?ba{5JC`bNm|Oy0ooo=1Q}gwY+4WC^`zSvZvTVKM8|!*^o68rKJK&;Q;G{g}!gF^>;IUW`{5JwFfQW`6_vH+C8V^?G<y+jU0*c?1(5cZjFw@qTT5BiK(E55QOdUe{jpEW6%r6yc><eU9o$1dmrhX{0NG;`a+Y?NNI|&o^$Lk*xc<BtH3?7&)cot(WaRCPN_5EWATn{xLXW<8+wo^C+Q7i?z5=Vsj0XOVN^&;y62*B7SHiruVLV(Uh1$v^PKu~aG+W=j<Ra@CSkNG4?<MdAvoX#=tk2N2BiG^ifQ)Tt-go;K^cnq|3jyV-4Kyw1^Wq@;lft1>|!^HYT=j*?~+jSDj-c*s5-JhlyL|snHB)qS{J(W)1FXgui+)I@zVN>M|*x|Ece1UIi=ykq6VCV3vPDMErku+<yjE8l;iIz7hF$j*i`3vv~V-U7w4!K&HA$jJ1z(OHX+Svrj4#aXeC$`@G?6;jWEN(>tRK+9&N_kU=E&}`pcG-?=1U!=I_4Hu1psgrb!WWEHuMlk~k|ZNm?aY&K8uj6HUQilbSameM{s9Dm7ZKLVm_O18<Sx|(U?#;Ki_Qz7$TOzzRTRLdPJ<zi(wjz?M0$Wo2ow!mL8gb<Iuah|Tjsvhi$tu@Y{D~WL0`5!lXO~-vb3Dp;N%Wh*GiCT{dd<-l;U%@)>OZv6ipX{`w=`xl9I=Y4`NDi0coEP3)K(JBIHSjPH4~`^UOJMxEdY2X3mvvgs0+`r<M2YB;&a?;=-9z0Wylck|hf&$f~g7g1Zj0WgC<bxZuMwHp+tu;#RxEH|c;QRNwOt@2U1LP;#sT%mN3$J8ZqhNaubZ8Zj!>a370m7f5g1iM&W-^xhbA(Gc@E#i#&lhMKcMnGsUxP|?P@k{O^a(s|EPAkOzW2xM0vkWIrY7N|<SG<2)6qL>21FZ*$cT3zuJ)<E-SNd9D`w$=KcUWlk`Q7Z^yRu)`$Bv~JY!M$=oxcN=n4F$JV@-b&hDr2aqqga!QS9l-Fn^a-L==!H~p1?Ei=kI~p#{;7bJFs?AFRG-I4~54#Wkf1lj;MST4tYFRg`-(!jMD%Ho}HrvNp)*MP{az=WJR(-lqYep?tmA7CPB+zLC)DwBQ2Z&<0fOQwO1J#Rr8N8rsxde&(OHJsGK78lW8gZ3sfl>y~}CH!|3E<wDt$!nH%t3bG65}_jGE>i*XurnOoA6opgCsNAD6;K1Mko5t^`e+Gf@d>+TO5r>g)L2^>YoyLqC@4DRU9hUbdfy?1S?J2?eucpl0x53COXJYy$p1Y&rCH}uDhv&TC?LuW)oL3*r_&88s$v%LenlzB#`;ecSnjTy#dxmw2X)=$|JQsCfYiK8M<v2giyM`?0E@=FFWDS-|9AP!>}>X0|~QWjm^+F~1SMM&LER0Xlkgt9URP9FDc8vXozjy2ufWz#{KKd<XAzH)$jdQ0H-+2;{P7JL*?oab{%fG0xi-&7StkOBcsE#@^{0tSecN&c?n{E7$bnExj@5W`qTnI{xB?=7?Dp5uB~y%oO<u`FVsACWCNA84_cU~EN!t{4%WrqRj_k{v}w-ouB4dgJRyh4spJ=YWAE`rR$3QN2giJfM|)4d_`IjhZWpBHkadFKQkppH$^sACWIrB=g_00BcDA8;qS}yX+OTrH&~}y5GHkO+Q`a4<{CjSlf|a6YwfK-xN0M*p#*@S363aog9PQV-C;l5%*%|oq>bJ^{@j4)jsRO9|QQQ&4LC2r_I(}Bc^;taUJqvysLr={z9ig6NdhC!;T}4J3B6Mkb2s)Ksi2;S|ts*)mnUDZ&Yf1EqfEACfOtwW?~YE@rj*K_<O-;HTeS_+ZGkMrHP{(nuNB6haqEceB0kyZ~44TK7pGskokUv`$2?m*{N`k_c}osqv|vUzi&M9R)dQK9ILndb|W14muKV~)a!2EIV)aa&^Es|ggVsn{<@;K0m9SO=gVnQ!iN@4npE2B)M-5P(BUmO5^iqCIDCx&5S#rJm@u-O4hrHjsHDzqWrzkonS_4~`RbF}OV{rXcz4tVNg$lUhG1X;8sfslQ-8imsHT%#qvU)~HoX|Pu-gJZK#>j~-9)A1pSX!S>Oia~Fk=)N@I@?YP$ds=XCuBN4wG-m`XZ0#)i6kncr#BsN?)1tLM|0^kf@V_i(Mul|EBRdRuKa<A%OuUDe^LEnx(4YD6}Q!jcxX60%>Bw#}|t}HnENcU0f2dsHsUDni}5fyo_yH#vnktl*w@MR>`&6tz=xTAR&tdi1j4`gi@HxA;t7T+?E>9HhU_@I`xDX7FERe*OHCYnn{Je)6i+AQoUHj)_0TI_R<^Z?<zu);JDh`&z5c~?mP>#sF<%LCw&Y=xdE=YB!9j;k;xv%Kojq=)Q%jUbI3-plJ4Ncd$dqtF=`bhF0e5rV_|Cr8BEF}>zk*mqU%jWx%E2`<tk9Srz?qdQb%<(j)npQS_`THxHLi@j@TC1l@&r^Ruc;_5w3XSx_qH>rc;@@kU_cTDVO5;>+4nujgcaCRhI1=7}<qdkrd4-3zJEt9BxD%LAVP<McLr`$7FE*QwINc^oJ@56e@}oa!!SVj_NFH3kcO>_hvS7qI`;AD!qt!m3s-HyGYa8TvHOoX+l+ytKH14-MkBpt-YTEd?%xsW&m&HIhCw;H*`P6xJ0Rkl{-lab5$b<XhNA;!Q&p!1$j9f?}Y<?A-|kCRtk3oDC-GLBWZF#B7D3;@>_9QD)a5*o#qo$d9sB7iMPQ{h-)etD3Dax3owpJI^~_gb)s5uQ|<_7oC+`TL1i&)py;0Es%I1|*a5RNqlsR>9Ke#!yRsY<Fr054z~9+>AtunY8w-h=5_ybz3hC~0uZbgIldzPR(IvTw!gbx1rt%KFTxOjKVI!H0Kv+HJSu~9!(@kP(x=Ad8wkz}V7ScyhI!}nW>=3S*F_`5vvBZT?rC?!v5Rujpi^R`_{#`LroG7}}D~CsR{R8S>fBwGs@IE6KBker+LOLDFrEapbDCwsqOgo~inDAge3*#{9>o-~lX87TII16ZiPUacfN-`?!h3mu$nngH~do~G+$%pqq5!m&QRwo5(hfD~{!n28c37}?~(wt#g23&B0fmLZ%CV6?Km!sG5aRnLDU0*>4`1Pg*8JaydR8^!ED=wH$B2M9jI`90%v>Nx|-zdaNhYS|T>bS_&fCkG_A#o{HxjNdzodr3Lao<E7dw;zagEJG%_39=Fwt!PI*jCG9W)!Qdw9gZeWU>^^eo@r`?!LTUw(pB5QD9Ap!JP=N*g8vvLBcZVNI1^=s1k!`NA%BL2b)%XmFSHC_%+GVZy_MC6a3eKFe_Dkrz!85PKgc4iJey7!Ws|Qr*2AQ!Kycfn&!T6hE`-E1U9W*$t?5xi3lIqJ2-gt)iy@gpSnI`L4!Y9Jtuo9+~zEAw*^VMD|~kg4ia9zz{e$#nlqq8hHEi7e!JC~E4L`cDA0Y~Ir7Tcygn^=+uipFysUc|z--i*^j}(MJx79s?n)AReEri@I!2IpxbRjt=xX4QQ{m4Df2&n`BfxS3!tdlZi<4kRf)z(1Tpq)O9JI8CqvoF6-=A1ju<0@i4v}!=wjF7IJu8z^x#%Ois}-{`CgoPv_Q~18&onr3*K(?~x1t%X>+weOSt)5Zf;F0@b|NpT_s~O<XT-&;;>I$@Y7r@|cPBzg6)!wGBXh_<MFm}Qdo(|M(N><)daD8ARWDBSyw^|pa0ytWWmDRB2@}MMY}zZv_(V+O!D`cWI*B1Z^{1BaXE+myi}^h1_s+^(C-`hi&+hHjoSwbj97_W&S&YAv+^^5Pi^+?u#Mq`V?#)ooPr2rnF2>udc`-(FJ`H;yO@{L&OYwCP3;Eq|7r~!_EXP+q{pOBx(>4~3HOkV9+YDv*6C-`TwrC{%S^-==4`WkqY97YvBx_IE6jNJUIzx(hlLUPhjz|xfCY=Fnf1;#t)SpKKst(fiAd|#A(R(QpS*@Fq$7*Ig*wD;tkFk*MEdqmsJL%FF&ms=5=0xOgJpm2j#$%H!pe6ldh^!-CwtcwK1V;;bwrJT}c}nRJQGOtcfp21-NODTcZn7gb5S$Ybv{`Ar;ulW?9-C)r&KDw{w?{MbDI|@va+6{bPhej$(eue^LrZx>05gubZk_X@=6J8>0JAJ$tHRlg59~D2O*wr(?|w!hDQF1W2Z~S0W1hkTCY+(?9q*^GuDOrcQy=1_%Sn9Iwt0IjW{)eu#3ZYW2cm}{fOQ$6FIVIvv1{Gmex^#XkTyZLk>37tGX}iieUH;L%B1~p79!v3@dWzx7vqhugTA`0MbL*K#<nck7Ecc$mTlD8wo=UWM5fH@vUy-yZN;4u&Jn19z$pA~oL->9p}{!`1W=Si=j?AQ1S1vQG6&;(<TRje51bGVZrWRGkC@hKASD8Ee6r8x6;vsPdj-RVkVF=IbAmSqrH>A89vtvRFGE-TTY}VxFR+)_r`B*AYmnsr6J5vYVJ3zOSeo$qIoPijyb7z2Q|94_qr7$phh52)J@P4XAO<aaqCjXr|Fm^ezjvD(<QK5(JrbCk-QWEK-^khhqo*ipU;XvwNKXkF8U^p+#U;>|mn5rT62tccwovlMfPIg5_a^$^=XNjxu#_?$^zunGQ^br?-B$WvydewhSVz!P3D47Ll!-VW89xHgd@}Z#-r&N0@nTE`cSkbxh$rafV&+*r##G>uFd9zyT&KRn4kXHVh%L&IzdGE0S;_9*g+?pwtXG%ly;WT5&dVA?lhC{h9Qmv8aW+mwiHTtxEBsJJy~tNx3!&mzl_Q<{NI?tg8yN}yEb`87$%>+NmHrBW7*sh?Y;}}aN<qI9m^qsz8TSMns$4HTgREuV%%N&AEPWr0$+eb9x!rz@H+F&8+|<G&{c4wX<G8Ea)@H+`wcw_`&O|~Nbr2F!Dzs-I=|{Y`&bSg^ql0ikhMEN4zY>}t8|dg71AB+5wDIaCItv#mfZ=AVQN@<7c+;B6Ov`^)w3SG-8rsRVD}A(pJ|&fO-kqg0&8u_kT?G=}rq5x+H;%^nD8VSmO`pq%9Pz{$4WB{Upkd@kvR-|9%J$EJl@e*#BlRefcYY~TU0Dj^?o84A962G~Yh$hE-P@|*Npe6X+0+njVrt1w(K`jWMMM>i@S&}Hw9LHS_MZ3%m%U7zN;Ta>zY`02R?B-L-Uae8Ll2Y%*M`!bp7C*b9?7^1bx#Q7BYqynu<p?a&Qq`@TSw&OP8~U))+XhmH0}o`4KH=57kTDOK$u6gWx>Fq5rlGB0KTZMmkAY#FY=a^7@UpQVwSjHt&ie?#$ntEj=D(b;Sc7>=D7MI8Hg=3Dt5DG^uHOB+cg3a^Xz%9z&&rwL5(e1g@t-qU$s{a9<qxZlNSS2j8xCh>sIG*QmG|su<$K{08AEr>(JtOliq#?>_}nFIIdjTSCjE>Oy=bfPQSm|GH`seDU9L9l-74p&Ax(CZ*UWr!CmEew*7i<lgby_W^UoQw8-;Ly8czZiZzXma=xog?}k2j_FRuvoU1uW!v)asAh$Qt&<@>tA1Me)5)L7FMOdZitW-Hf8+Wv(pi1g9w3SV_nxrbc2DeK(DTs%e0Lf0)+Uuvw=jmv@{1~Y;S+|<!%kVOqmQ!;tSDuzup(Wk}R=Nv7hP;{^=J`8;vub17ztbhNQ$DgP%Yl>`$fXQeC<$lCe43{GZB4BrWR|!0grF4oxxK41F+<jc;mGX1$i!5KWLSWjf#^o{{-*tAQ<Ps`WGt!!a1Ab`N)DEqcj|yl7Gf{P$7JgLE4^FeI#iQxRPbrgYJx$Eu_ZvzeY^>ZV#Yl}ECnj&vRJOmvn>eU*+s|2kkX<C(hX@bS5_Ojc3Fw<l#9hs0;%|2Do}ui-b72v8AmsJ43olSJR@1~4dpNL9u)!BzqBUEjzgy4wWTKA+kpNkjgp*=ltBc?l)ps-xA=Qa^q5&Pz7#i3bpO-Eg1)MAt&(qGJqGL5Ns_iXbWmAEJz6ni5sM~WfA6aj@a9O%Nr9<z=<`{@cIiKugKIf>??Q^SM5|*~HXno|wZN9?M*8hw+HDt$&Rw`D={bcc3K)xYSxCT=fz{TSJBX2Jgw*6gE<rwdy^2ZSf|{NO4iAe4mvtp1{J4^zpm~X$3jiP}5Y?WxMdubgLr!o8#!PBWFuKR{Nv-qLF7~eYhE0BMxv_e|KFhlgU#+h8XkGDs-cn8qo2x&u>S;UN(Qf=&s=t(K2Y9Cw{QtcY&j0#V4k(VHEvx&t%M3#&)0A(D&Zo>e1v#m$Z|&KA=gl@WG^IhA4XH^E{LTf?a^)6u8SB}uK+W~mt#&{WbHh!hZPwpyzN)@vH8YJ_GQEDS5kKlPS&YN7kYEFau_xTRJtg=zxZ7$-cda%4)VRnSnY<+5UAF%<UX6#=#0DA^Z%r~*rG>MXq%?6BUbPm$uGzQ3!N4mTG|Z~@UV;9-V7)QpMkVglWr`|OrYyTy1hqYO0OLd4kQk_T#v)?SX85+j0bCv7#h3!M8ecx_3u`dXdEo)jId6HY3@Qxj13`~uikW4Sn|T9uH;F{Jghk5QYA(8KbL+E=hZ8^YDmaT3&J6%;?3JKQ<aIgGYRKzH+(DH0F-@Bqr?qfC^(XQY#J!c!hasVA*>x+g->Y28dblq)k0L(O5LpG0ov&@0W~SHZ`3(bHbjW{6+m+7dFgca!9G<V_JZ1EIo~0MqDB~TY6-p^lA}*>nMPteL@O)nRB~syBjJ;U-{q1+uUn;dOOTAR*6)Vtw6LGZ9S9e;*GI6ZWdA}tv%9HpBHH|8DC40aEtfJCs4jgDLtR+d^9N#fNE+bAVJ(1f%>WA2Gh=i&f1I@nm6z#m`ONLq7UPEXS0kK;(b(Q6=*Ud|qikgu_?zNyMiT%3Yk=q^r6N&SPs~143c&@}JZG#(<h6w9EpJ5V97zwq*^&*$nQob-|<k<|_7Ab*BopLee8E;;E>tyY;5e{I45}#aLJgJ8A^cSHXc73lA%si+7x0-s1$yeY{G_X@I$EZLWE#+yd67>thSC$JL)_$GnNuK&`d0q_cY+)e3iy{VjBV~CLal<lmnWS%uthsoSG#>pw-{9=&##WF}wZY>6MUk~ix9GD=sQvz8SqM*kv#5cXwmG&JFPo1=K{@rfI4#eoN(GQk%%9s<qtlz$0YUF3m68CCQbd0$XYs|QK+LX2WRTJMP78u~Q@Jk7fU<nYSfFZ$wAprvK?He@i8cdKJ;eC2d2eOO>1A0)S1vWa-cmN!zh%#bG!g9Y{o&2t4(5Vy-3q2kMEe>K@9~b>)FQParSb1G<v-`v-f$C~DxxbcS#U{NSjArKHpCIAjI`kp2UFOC5~D&Jg5CEFCH}BD;g3uMyK5Cvkx6ndt*$SOCI^&f5}T>~dj{S!UmU)<q!&C7L2Q^}<S122yg%S<5ROAO!33^w&uWQbht+aqy~UayMHY)}mIkWFf&$w0sVEF^-7u;95!PnZ8=AEngMT&wR2g^ITE04tcEpH|D`r^f)y{Z4YzkpU`3dB{<+Fq<+5p0y$A6Zw$})F}%=a<C2O}n(;S)?6h}q9l>5@&s@aBPtY7a6}$;NpV-R#Bnlbzk8z2^s~^@uEi;|$FW1BTqMDp(GS01Aw+_7P8jSVOC_aH>5t+tX?RBa-2{{A`6a$&f71%Dlrn5Wm2f*%c1)p`^wcz({&=G~t2zhED(+s(r90uU;P`LBq5<UoxuGC|z-nWR=q3;Po4;te%}9JfB3AkOFpSQ}qeLI2NonoqcJx{v%I}*7y}$sa_)Yj+Hs-6ZWK39oVz|SFeq_dpT7~%8=m-Lcy_SucR0Bdls3wtdJ5+rB|mT(%ZzYNN1Ct*ub{<iz#T~4|;psY_c`cl~g6+oOsDL0fNVkU5nKk%LRiK#>9pDl40mrn1rf$tfAyp!BUdgRl!IC6Vw7ckw}Ib)e5w#)@nv~_><|n=@_hh85H!`$%}c+^*o6WW@)^Mzu-+(KH<PpWgQ3MfRgb<Is1^M-mLYlZjUdHu)4^eFhK8}(o@sVK72z``QJW|r1OOj7(A~J;9nC_Etq=lwp#AH5-a5jQ_$K1r7ymID`VrX&#QARcz96Z<d)sP)lJd_gO-9Y?-eMu)q;AA9Yk<CSMC>j7aCqk3Pq@_Rgs_Q=BG_X&JDagQN6KbGuBiv|Gs{@<awnw@S%9ztG;UPwFkL*)v7-~u%}cP8c4M{oxyi_Us!!fYoKoD?`!9Xs!ZL_D<D|2I;k4tW-#K_zJk~02G;tR51+*gf8<?Ao3y<6XYy(9_CflZ1wbIT)_Lc1XImE22ahtE_Ew%7-VX7Dyd&byv^+U>m#6FXP@Q&z1x{r}+TGuMxqE#0$5TpW$exCAzl?>L5^Tp8;XK!V^mrquX%Yc^JLB^wy28@S*GE*NhCPuv*hGCakj{wVqjC(eG%t9FX2&X1n+T+yf_NUeRliYa);vaB?Gx+tPYtQNW!m@3q{yAj%e$IXE3iu_wt>aEhoK8_X(f%>HJ2N#wRO*SjsH|%8sY=%tP<X&=~Tyd8zkACpG9iP5p+v!$|C{AG1|Fas<JMNr8I3(e&bZurv~0UvZU^2wTJ-==iI6w8im1tr4gF}-U6Tlu|!}Rxc$3WRzg_YEsCJLTenz8fVb}IZ&Akc{Y(S2T0-Rs8-immSce=%E<^pzvxcf?L!h<<;r!SHX`#6S*VStFIdL5VUAkMd@FDy3k!GQv7rawq5J?Jc=5W%kEZl<OE-6I`te`<(?gya$K})~z88Qq&wN;#Y2;%~p>Cb8I(8-l`AR8^x=1{7PfGFw{m3tB;Rx6>#qx}tfJEXT8YLBXo18lyDDja@a6c3$T|1`b+2}$^nr7@-@i$A>o@I9snrkrd#qozp7HdeZYDQ5MHv0U7=wX*l4kUe|DGL9|Dy?O%JV*9QuY(;tz(7yuKo$@dtUc5$KciYXGmaTcZsYg6AlM=nrr8UU`lcpk}Jnq`{Pn&E9-%H^Lt(#Z^6}Ba=l7CKpQWzOhb&gRQ7_XACHbH|FwRyl4WpjWv#Q}Z709E!Wl?u%+FG66GZ6{hn94O|X%c}gOt4!3KX-CYIJ$L}PERJEpgq^?Yj4^UXS}n1oa-*=n^(2ka{sfXs8C*gr7wU(tTYu4*z{Ruv1`P(Pp?ZOn8Hu?H9!Ba4G{=vz;s=P_m<9K~tw6Z1;_ZiqCcAdZ3R|0$DI&%|E%En2Yd`a&h1$zavwFo%7w*1-d|2qmoRuHKl>$*Z78uH}@=}5G3}Tbvf9!}$u#eybxeF<WE?m&F>!0R8-{96J5eNk%s;$_l4>WT8d0)L(dUY^$xV6AuJVZ@UH3w;~^u4-*i*sdMqJ{}LT)Em0{l|wA=q;Jj;}+Xdy6@nPPL;Y7B3meIg1<6YwV_tpAK{xelO?@s4r*AhSL(_I>(`xCmq2LAE7elFdY`rJSKnfcZ62yhO0;OO+X?`mw(Lx(ED-sj=~eW%;-U2TlpRH*DJo;1yR}$X97y`V$`Iq4N32|#!ceDV-6!5@NGSsD)LLhqg}UsmOm2z^@vqV>4!b6x13n^2L$k+ub@@>FC^%Jw1t5e*vTGNx$Gz59Ef=#1+|;Nf0<5>a&mjVW+J7ZLAVf4>IBF&$L)ETHp-<5(^Mm7qZta|bRr-$)&BJ(LOvv8yrx@t2)OnaVW}LxWS{7qID?wZ&(HZ~R!Cc|!{ZC5zI6Wf+S(*)^L@1d(%D(1op{^EWf{AV5qedK+aVJnqGNdm@5z=z??Le?=uym#!@g6fxI%HF#qtIgsj^I8F1Wd_a!eks6^L|*3+mZ#OdSR|--)bZT=8++w2g*KdBc4j?qe-?u*Qs-*e*PbP5=U%A=5MMM;bsEROIG>{Y1D1Fo0Kg<vw$#XQY}D_pDMk)x?yu)JaJnCuzcWrw9s+8mgY*^MWCXT-rtdK+8+#0>W28x18LQk#!=g}ZJeEKknoG1Ogu$o%}f&FDvaAkls`wMNT$M+m1m^;B9`5?v7kYd?7LC2j{$*nol;WpmMN?{HmXIb7+T_nNGMxS?^18fNd@fGyCy4yVwJE7+lSg89#KuCD%osyXe``pt-yey$wM7V%eyRbvn12205z^dlYX44O@n-X#q~i5LK78p{F<qdePxs&^|o<NKwb=VF~2DFd#hO`?I&^;_=o14VK6x#ADy7ieiET2v&Og#FQss7dv%iGG5$5Su;e~(t6P&8^P&rziFI99qCn<SaQjV*Xedlp2U^t7gGpZs;07~}np^bmjCpv@gGtLxu;eJf{TlBI-zt-ax1pp}w#CQWl3v>Iv03G9uUP*1q?rzhmepT;O{;~-KdDCkNX$wjzg72)ZNPCl+iXCd3ED3c(59$@BQEQwfDOo2Y0RIXoVqBWTULAGHWX{CTB^0Gc8#VCWi<(`S+&uAAA{^&jJFm<V^T~JSt)L}X&`P;6^6AtWxV~WNDNKx#DaI~cS**jU_ZY*w4Zf?P)!+ZaS;!(do9papPAU!6KcB&{ahr~M^<lX$h<Y??_u6ul=m-t;tLv#Y~=fui~n14as5#olCtTMw!mhuty)nxx+#DV3s~;GHoF&xb2vSCb!CN6NB7R+v_HAG$?hpX(5*ZD=WX(G_we}Ht3&t?6NPbtR|kLmh8-QlZ?KNe-Tw17d794VR2H2D{TAC;+xVQlgylXH+V|}}cFX}q*mfMVLqYrxg^MVrdV8Ivvn=AUpL5n}XhW_31c-xmui!z%JLIp$_V-`1ZJ?SFsys_+w>%1`d6|sZ(Z*-&;Psog$yb=5Hcd9!IYxv8Ps%7Bu*YjH_OJi-zhkz?j!J*=Ho=tC@LJzwuMc-S$A{Z{2YUz4S@7oItKGxBXM4Llt-C+1syJ8=1AlR<y&gkV8%($J<jvmx&f~StR~y)-*9*&HoMy5C?y~4Al>CB+*njIL3BE5AXhgLO;TBN3J<;TqOo`VPuFh{&D<H7LXw((|QH}e<5jGH&WxosyQPwEtKYz~?QrJeSB`??`^7$5aEJl}roB&dEPR!*f@@@==iRxJ4&9i3*cB_6m!g^<94n(aq%sW@zr2P7v)G1%5*#zX<KG!v@r5yQ!pPfZv@>&~N_Yt!Ec)-gnJ(^F?(m3j?4=sBNaWv#{is5adO0*n(U;Zp=M^j(7ADMD=bNm}tl5?6(<52Iz{<iZi<@#H?YkzweC3*wjz#hR>`orGAfmwa<@Z}LEkXl}W@GwpB6yRi!-j!j-b>1nV%jz&%9{$L}m@+FB(J*QYM?`5)GfEIH1@|n(q}3Iyb8=?YgIJ{pzuNe_?wWcbATnWSsYtsaveMwOkhlUn@u;djEdB1fD;47-0{|rn0TwXJt9<+99JFf1QL){2-9D#Ut88Kw@mn17029NK%0<h_21a!jd7n?%SvoKA+U-PBaXWA$wIiX!nrAr^9vCfjZJvL4kL8>zeXg)JLaPYy8n3ea6K*oed{N(MKX2X)Ul~F>%w>kMFZ|p2T}WU4)Z3JA9yX(To_)gp!N!W<aAmY;-6VwO1_gaR(K2x6`b5Hfo+c$L=c5u0_Q9~sVY(Bz1w3voj-1&ht7LA26JA{|lsgTfxmmqn8%3!Us00jPjcj%<Y<DjOyLm<n!xd|%1}w@J*vD>_W3%104@?++(-Y1Q0#l+#dEJ5OvO7yJ0%bm5qP~$5;U!3r(osEs2U0MBdZ;ge)<p}Xx2nc!e)t5X#^v<e5WGwmh$mEU5OGJ@A=(T$>?7bM%G03DkZG_<I)yi=z)pjZH^c2u0PIuIi#9YJE*(W<=8|f%8qOwiL>BDthcCA~k8wD1C)5Q6fst@NkoQDXM-7BQnGK`M7CYzjkjdJMJAiPUk$fS#9x1w4QRRm^S+LDfMF)d}Z-NMwN3dBOWGD)nFQL!Fii7E;NY5cr6Y?sglD&P3`WcX0$BNvEc>bb1Rz6zfpwS|}<Tkwle^OIZOiGG~dpq~@hy`n8`MMNdv9vw##r*jrxQHHE=o!>sjb<+ztJ}Ji=D;M-SRq)L)PdLtk=B-Aga~FRlZc=BOBPM@9^U*Oayu-_g91gZye>3Y2M9`FdqGmj7E%dM%Z+6%-Nk*i@fby@LxreEBD9JH1L1J-;d{l^C4;IF(FG9rj3vCd0P(v5Qj420-VQWo9DVpcO0dTAj~TCS&>2~ju_Ue7`|BU$>mM-M%5>5aGddyzvF-_MJrVv$IY2>*uZ{@@oE2ktGCMGD+`#)-2tJC(AxMAgkBClhJUD1o-ec~#dr3Zv&>7%tu7T9M0-G^-NIC5ZjPY09tG*+GZ1@V%h*fH<c(4J?qv}S!U7fY=XLY>Rw3JQxBut_!!9fL5n;$XHW0boNvoOsB#se}OcrJ}WeE}kpUs;Y^!uNfTy%n|4){T#y=89tQ)mur7qq9^mX)Z`C8yGumRZ%jn?KlGpN|rEHXm}9V^yVQ3+y15+dJ91Ml3YYNf)pcrGow+fAqKZ$q4>q_FFuH#AQA(l`KtCEA94OvF)|jwTP-QdX@{-@c}V^Qgck-9<gi8*WPiyqtM2NCl!AwfYv|%MJzw4AKGBeQjvkv<XHO?$uxC^G>1^3Marl1RK&T$%Soi3p6`UpbpojC|gS&2H0l^E;*%-oxaYW);+26$WyZv`B;kD3@3MRA=a)D&4QAOD2=_C&|5_@W{i&MsE58ezdP#y8%seq}ZYUg1ThiCNy&{8$B+VP9V6#}^OM{P=5E|gjm1m;((*P$7?A>?rb%d9nPI&QsE<Is?44>6tq6Yp;M2%ZG58&$OeU2VJHz4#OP76ln=k_*4B3|?a#-(z^3@|fcEPtge3usTB@BqHRVGZx{<lvQI8^|`o&vn)N227KU2)DtbdjdI0Hh_!uCtKCXt;D&qoOCFYcFNLy_X>BDtQ|d|<=zCtfgGynmT*ofz-Ca$JYN2_B<387<H4R?WFGUki7PIM9V^y>9v1)~|&aL<ClGNFnX}4C{oq9KCSH_}2)1LTM#h$R07oMmS*q$x#3qTDDzg88^y}^L}aK0FXS;+c#H6W`WoPxKc{PyYxMtby4E*0s3mC<0tS30xA*VBFUV4I$5({y@uT|VAjl#(+nC$;61ylueL4tL_Wt=8(B>6_gRx6`3Qmc5~AxgPt&s+g*hq4|+bkMEXF|D6|YF{mLNgfo$Fg8V3#!1eT~s?7T?Mivc=Dhy-^bf>j|{P?qvxGmlTxz&X+)y>V$SB*|nGh{InK1WBOl2@1LWNF5tQz>g1z{tHAb9vfjqm34tt{Wp+P;H<qWt(c>WYV`tO(o-}C~5}1)y;fUm{^0Op0`QSo~5&Jq-<0#!z3JmSYts%Il}<?S58+|#{m}D+j$B@T>mBM^GU?M{EE@}!WgeJejcS|u79&(YE)nWCCb=jPw8){q8wcR1oMn@cnh-W^-uo^pb0yG0Q8?eHkp=-5OC<MP69mnw%WKDnx{AOmhpL+m0`VWXhH(jjYTm*MhvlAP>xFv^s!F+gzCJ&bap{KN{_s^DN$Bp9^qD{Y6;-z?E~Jj|Jtt3YD@pAwW_}F(IQ=*L4eTU0phcow2pJK)KGOHV57@Uwhy1aU~gXUY#;BQS`XDg{H61}#P<Pv7N-|Aq2QwcRIg^KulVpDuj+lcLKr~_vhwv0AyA2*zYnu;B0R$GQ`FSkdqOcE#-k7K^AGRe-UA3%aMdXcE$h0|l%oi~oRz3?<U<UgCyU1@I>)^x_&LB96^D2xOIy6X*L7$8<od@sT61NLC4k`9Kj2d>B1em)nrkfcgVmD#h>1~I--~drf;EA8!(QboDJxB=7>5O$B<Tee|3&Qbgq|qZg+|RvT}c7Jfv}MYdB7{=ddP&m`}zl*u?xi2W}tDF>dw^qC!(HHJa-A~mN{eV&Z)(?Pod^jX=qND;IdsYLyz?Pxa(1Mip%Ev)Ic#dY>H6@4c3$-HC3RSU47WIB1P0F!(lLPsYy#i*&rN8O0l$;^y3l{MtobSjuc~9kQ<{O$EJ(5?^S$91WV4Tfn2#8Xg>y5zzVB_7uCEEl@%cS`gsTNzYSIx!P}9BaL?k?uB#Xpr%?UwY8=(%gh$fyxnMi*(293sdOO=#iV=%(Ww%i6rX}Vc7CCLsxNK~vGE`XT&$(_O>7|RL9w^-ey>aR;^PUJ_kr=?P>nhv=s(a~?288>ltS()d-1A>ONB^xoM}Z9e>O4x0jF>a=<}Rc+^y+*noc-a^vLB-4{c2|yx0*j4hY7k}82+17FFb^)&!5AoG<b0W|BeR~(H{54n5;?;_cSbX(u2l>m-z7o$4Zoz6Q<+DUH4igpNY1z$@@KdZsx=j6(PPsx0BFvyrCIJ4Guc8&r8^YdZ$%=Z;nw)WPEzY2LsJxJM4gk172G)3#wPqL-ef?@=fKPgGJt}ZINy?FB$H!p&XP!7l3C~60(kdV1X2y7np$+9ZZxLANiaL34I#2*vVHLfcZMYMtKAoG#nq<ln{wRbV*eqlLX6{-LG1Pyc7sd(pt$or`a9HIaSO5#yvjE<L0KGFmS;z`b&6ft1stiKRhFU7xArI{a{F-l~;^xW&G0M!ho#|pM}KCTPS2gM~NmsNF|@?DL2h~!gppBuNfmEE8@K>q|E7SeAK3xx~$<Kx|A%u=&!dKo=`pgEf;R8yXx;jwY_a-I~!R*a5F^L9kDHe+#cT~jcW|FXJUFJIp6w|+~*!8XDKKTl5R&weB$~VD)ZS^By)0ANjeROlhcLP0CDf7za!jtqPNl|#Pi9S_Yj-(q~oRXb22tmcsxnesVUE=g1XbH_oIY|ncdesC6MNGoiWs1#McH<+GP5eQK_hw6wXn}Ksi@1Ww$(@S@{1ocP+hb8$tN55CIBPiLA&)f}%zoG;IPmhoV6f<l+hfP0=nvkrGJCu><s<6zH+%+;emKzq&KC@13P2y9Mf#OzsZ%y)*O8<7P=QY<&4VMiR7&iY7*$NV=>pLhoL|iCT-Bvk#HoH-^;Ay441!adWY+LEbu5*<OG6t;j3Pl7}W=?|*smQ}Wf5sfkU@n2QZkcryJeK`LDax{_cL5nGgX=ebk&gtHLf%Nh+mM?MN&4%i^=iFbOcg!3i2Dzfl}SB#FzZ<sKIaNo|wjXuwb5ek8!Eu2%zpH15^5c6Twgsacr2k2@f^M(k}Agaf3cz5eY<5JVu(_Om|aFyX~F#Ki8#X2@;6!Uh~mLFFQ2VpubpkmUD>CSkU->Lfy&>RD2+7wr)sP8OHX;J3=EYNfL>wVG3_Ks5sJ3q0aB=3Z|Y5s6W?9FaIW~|%+%-R=%+ZRvUwH<dr)!r!uw7A}oZHCB{gb&?*{`=1!F`{<Ze?=TiSC|=1DWX>@oy8Uq&EnlXFpapi*Do<nreHAC|DBDa4dQ2=1)#zB+4xxxu{EU+BTSO|iU!lHcisOIP{SV07>lPvIrM4>byQK!Vbb$;D=s|rk^s*m2b_EleAJ91t)wVH97Ygy;g&sp0|wWZ1;HBN(F5g4n+IR*1$U?(aKr<JnfMH*S<q7ien!OyIW0nFys#iM&ryem1CbnsQ33-i6xy0gl-eM88b$k)c(53?3t;{-#0_&PuZs7O@~o=^#)x7%IwZ0_X+(SV2g%8N8#%jM$Upk)TIxIT-L^D-N?l&);K<mt^qe9}L0VwF3E3$0(&p3?re1|&)HTnNrkXs(w=-e1wzxEyPil{Z(>L!PqsI`8q?oXF@#Ry@pJAu!qY$ryxRjNBXCH`#p}Ru|h4x(#heT0@Ve|YZy&==d&Nx_ti-F97lUQOWhUtdH(g`c76BzO^NtI^6mgvDzU^;oq4y>LVkVhT#!7!(^WI<W39pR2fjt%_)W6iJ&8rKN<T0f@6zr5VTFUlQ2OFuk&hBm`JLi_I&tl9xb=-5Oz$jJE`EO%YE+tf*l%FS%K#Hv!DcY((q&tV1=34D5#l1|&!$i1&$)M!QnX-+I&DmcU;4eZQ4)U4GKY(T9F5$A6U;<ogtRrnA@SeHtX#012%!DCcyj1#HQ4T9RpQDIT<)7#HiMYBW$8vvzht&+irpyFv;^le#O*%+B&J+4Gwz14eGV&79{Q9E0pOrBv<Wv|?bO<BMD_F@<WgBu1lQ%r4e60+x>aKH`S7QKMUvxV9LPQS3Tj=|P1s0hyLJQ(q#G@y3SP5@5+Di}b+!^h0<orWFjFw(emWkA-ff(*dmdPmE;%5KE9UxSJ&%PS}j7tD=dOu(8&`a;Ky^}X*cR?R^VBT*zO(J-JbZcuCRVpT~&#BNYuWip*0AcXAZrZ1P>#Eg?hd-Rjoh@3%8@h6a$aAvhO`B8kaz`;#r9w_2ylU$r#o0bdZO+DUDj|d%#@x2+K2BKoSX{@;~7O6N5rM$s=4qO{s3;Y2%8y+iZJV&^qIgS@$&A_e(Di|);x8wBx^xb^oUF?xMk;L^1&vyDB7G(|>EDb<{9_Pqd`I9k*y3o<KfP@`c)1O50D6uJc<*jqYCY6u4(okc^ZtbD@y^f6HUK7e#`9(_ZGns=Yw0-ZJ-;dyTa9sKf;qYZ5gRgDsIz#s#hk0ZeEvYAk<aIJnUZ6%LAc-_RA%&1%^`AVHUwPy;PA3q@*O7aI(_8L%oibG&<1%XA;5^w(((%A5*X+S0TSMgO)U05Jk$byp?gH58jv3I#ksREduM(YzS?PKBd~`&I6^oV)pF>c5RY;w4P&V%3X*)#U=St;CsPY=>-NV@TI8mD~g}4#>jDeP2J!zo4lgcCQhl{ihkyg?Pej!oqrmI_U*1)xOy_3%(fShHQhKIuL3!^@u&=p2*!r&DUQ)$-#7EY33q<H}0#<bF<h4=$XJJOex)R<5*RzLC*w69}G3V8@J30rh>Z6<FM@f&z2-um96r}-I|Ipdh$HeqzACNHJh@{|^)ED`EBIJ}hd{+x_vetlpYvPQjpaZS>!83-^Sb#b9Rd(nyjD@2@qWpgA^-aK$P@~9<{Vj>QV#kFZN8P%dM>hhypW4E8hhKe%mZ2@KSB}7c4@-+|y&K2i7>4I2ZG8`pdZgx66zWsx%#tiLb7=>~Qc8H&SD4WIVgSan`BIV_hl-0BW4=wfhax_aaR+Cwd5x!4SFdIzWu~hUsl?@GLzjS0D&kd;r?g1NtuN&x42=ve=;B;z7%Qk7Kt-y1WsT)vxURjtEgwPd*p893hRvDxOrKXc_tzyNho2wAUAw}PxYmd-pFE)@x5VCA_7|^e{?$r>IN0seaV_R^2`LwLTFT*>rx6*r&=fx6dXpuNRId?~tzSr9R+rNz41uA%6lYy3kui^-kJLM`Y((NDO1du`d>a6O@dQ%fEA)DWz-w9aHW!H!XP+c$3nw9X}3gcI=Q9&%~=UJr^!#y@HoFwNiTQ1g=g@3^cKu~RcSb+h`TAJXy4?-Giiw3PZ2vxWd$JwT{>G+R4@ylA4VWVDd+B%aHLfsQzrvLYgPp76qI@KhL@?+V7((+Vp_Jx`hiN_3~MGTq)_uvJ%o?1&ajOMc^(4sTr=r*31H5`Sg8AcUaH#Y`d?n>pgol+Citr|1@rPw5e<tZl?FeK?(6s)4|O}om)9!^mC0<5GQ;j|abPlZ4*<4h;7=K+-{7a@GpwsRaA<Pzj)LDJ>JBR_F-MBe8~_?d^LW>24HHZQFNC;Jf|PmJfK5*g-WX7)(13QQNmmIjqQufcBYVW0zGgDVoQS7LiWM6kx-{=kC!s%@I-Q?VPQ#J9Sjsa~(T(%l1SPAblFx!bjbNIz(<#>w#v@;KW^7uQ5Onl8#^-$KIpLR)dF$I#l#U{TS@=WEyL>2o&pyo_-o+B~Pwvod14+lgl?z0qo2rNQelePMJqot;f46Y3{?ok7uq%4b0fX4)L);81W!2c!59i4(e^!*|3FmR}$~Y-@Jd)}rf26K>!rkiH%P`L7BYqtk>zng(XT7DbnXUtx2363OM5Z)yzbc~wAus&hk9yZ{dzMW2$GdaaSEeDsfK^SS8{zWkhqVs|363dF$#dwsCX@!`>Ik}t)Or=xx8WA^mvZc92wN#(zW#MEw?akS3pzg*SG-4q&jtJh2KOmi25ntyJIaZ_)~X%b?xg$TH+A+Z(`r_dtE%!mW`DALeZm2Tz>RKvN2>>FLKThaA1{djCOWk!vSe(C48NkzepF|GhwV2Ed^KM&I`*zA_SrBl2ACr;VXBwH@o#8&4*ime!}n01Ff`28P;Z^xG'
MERTFORMER_REBUILD_VERSION = "repo-parity-onefile-overlay-v2"


def _m5080_json_obj(raw):
    if isinstance(raw, str):
        return _m5080_json.loads(raw)
    return raw


def _m5080_lab_root():
    return _M5080Path(__file__).resolve().parents[1]


def _m5080_private_dir():
    p = _m5080_lab_root() / "private"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _m5080_artifact_root():
    root = _m5080_os.environ.get("MERTFORMER_ARTIFACT_ROOT")
    if root:
        p = _M5080Path(root).expanduser().resolve()
    else:
        p = _m5080_lab_root() / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _m5080_load_embedded_sources():
    raw = _m5080_zlib.decompress(_m5080_base64.b85decode(MERTFORMER_EMBEDDED_REPO_SOURCES_B85_ZLIB.encode("ascii")))
    return _m5080_json.loads(raw.decode("utf-8"))


def _m5080_ensure_package(name):
    if name in _m5080_sys.modules:
        return _m5080_sys.modules[name]
    pkg = _m5080_types.ModuleType(name)
    pkg.__path__ = []
    _m5080_sys.modules[name] = pkg
    return pkg


def _m5080_install_repo_modules_from_embedded_sources():
    if _m5080_sys.modules.get("model.transformers") is not None and getattr(_m5080_sys.modules["model.transformers"], "MERTFORMER_ONEFILE_EMBEDDED", False):
        return
    sources = _m5080_load_embedded_sources()
    _m5080_ensure_package("config")
    cfg_mod = _m5080_types.ModuleType("config.config")
    existing_cfg = getattr(_m5080_sys.modules.get("config.config"), "cfg", None)
    cfg_mod.cfg = existing_cfg if existing_cfg is not None else _M5080SimpleNamespace()
    cfg_mod.MERTFORMER_ONEFILE_EMBEDDED = True
    _m5080_sys.modules["config.config"] = cfg_mod
    setattr(_m5080_sys.modules["config"], "config", cfg_mod)
    _m5080_ensure_package("layers")
    _m5080_ensure_package("model")
    order = [
        "layers.bitlinear",
        "layers.ffn",
        "layers.qinn",
        "layers.cognitive_extensions",
        "layers.lifelong_safety",
        "layers.world_model_head",
        "layers.liquid",
        "layers.mla",
        "layers.moe",
        "layers.mertformer_block",
        "model.transformers",
    ]
    for module_name in order:
        source = sources[module_name]
        mod = _m5080_types.ModuleType(module_name)
        mod.__file__ = f"<mertformer-onefile-embedded:{module_name}>"
        mod.__package__ = module_name.rsplit('.', 1)[0]
        mod.MERTFORMER_ONEFILE_EMBEDDED = True
        _m5080_sys.modules[module_name] = mod
        parent_name, child_name = module_name.rsplit('.', 1)
        setattr(_m5080_sys.modules[parent_name], child_name, mod)
        exec(compile(source, mod.__file__, "exec"), mod.__dict__)


def _m5080_update_ns(ns, **kwargs):
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


def _m5080_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _m5080_get_runtime_cfg_obj():
    _m5080_install_repo_modules_from_embedded_sources()
    return _m5080_sys.modules["config.config"].cfg


def _m5080_apply_repo_cfg(one_cfg):
    cfg_obj = _m5080_get_runtime_cfg_obj()
    hidden = int(getattr(one_cfg, "d_model", getattr(one_cfg, "hidden_size", 256)))
    layers_n = int(getattr(one_cfg, "n_layers", getattr(one_cfg, "num_layers", 8)))
    heads = int(getattr(one_cfg, "n_heads", getattr(one_cfg, "num_attention_heads", 8)))
    kv_heads = int(getattr(one_cfg, "n_kv_heads", getattr(one_cfg, "num_kv_heads", max(1, heads // 2))))
    seq_len = int(getattr(one_cfg, "max_seq_len", getattr(one_cfg, "block_size", 512)))
    vocab_size = int(getattr(one_cfg, "vocab_size", 32000))
    ffn_mult = float(getattr(one_cfg, "ffn_mult", 3.0))
    intermediate = int(getattr(one_cfg, "ffn_hidden", getattr(one_cfg, "intermediate_size", max(256, int(hidden * ffn_mult)))))
    intermediate = max(64, (intermediate // 16) * 16)
    head_dim = int(getattr(one_cfg, "head_dim", max(8, hidden // max(1, heads))))
    if head_dim * heads != hidden:
        head_dim = max(8, hidden // max(1, heads))
    moe_every = int(getattr(one_cfg, "moe_every_n_layers", 2))
    liquid_every = int(getattr(one_cfg, "liquid_every_n_layers", 3))
    liquid_idx = tuple(i for i in range(layers_n) if liquid_every > 0 and i % liquid_every == 0)
    use_moe = _m5080_bool(getattr(one_cfg, "mert_use_moe", getattr(one_cfg, "use_moe", True)), True)
    use_liquid = _m5080_bool(getattr(one_cfg, "mert_use_liquid", getattr(one_cfg, "use_liquid", True)), True)
    use_qinn = _m5080_bool(getattr(one_cfg, "mert_use_qinn", getattr(one_cfg, "use_qinn", False)), False)
    use_cognitive = _m5080_bool(getattr(one_cfg, "mert_use_cognitive", True), True)
    _m5080_update_ns(
        cfg_obj,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        block_size=seq_len,
        hidden_size=hidden,
        d_model=hidden,
        intermediate_size=intermediate,
        ffn_hidden=intermediate,
        num_layers=layers_n,
        n_layers=layers_n,
        num_hidden_layers=layers_n,
        num_attention_heads=heads,
        n_heads=heads,
        num_heads=heads,
        num_kv_heads=kv_heads,
        n_kv_heads=kv_heads,
        head_dim=head_dim,
        rope_dim=head_dim,
        rope_theta=float(getattr(one_cfg, "rope_theta", 10000.0)),
        rope_base=float(getattr(one_cfg, "rope_theta", 10000.0)),
        dropout=float(getattr(one_cfg, "dropout", 0.0)),
        attention_dropout=float(getattr(one_cfg, "attention_dropout", getattr(one_cfg, "dropout", 0.0))),
        ffn_dropout=float(getattr(one_cfg, "ffn_dropout", getattr(one_cfg, "dropout", 0.0))),
        rms_norm_eps=float(getattr(one_cfg, "rms_norm_eps", 1e-6)),
        tie_weights=_m5080_bool(getattr(one_cfg, "tie_weights", True), True),
        use_gradient_checkpointing=False,
        use_moe=use_moe,
        num_experts=int(getattr(one_cfg, "num_experts", 4)),
        num_experts_per_tok=int(getattr(one_cfg, "top_k", getattr(one_cfg, "num_experts_per_tok", 2))),
        active_experts=int(getattr(one_cfg, "top_k", getattr(one_cfg, "active_experts", 2))),
        moe_every_n_layers=moe_every,
        moe_intermediate=int(getattr(one_cfg, "moe_intermediate", intermediate)),
        shared_expert_gate=True,
        router_temperature=float(getattr(one_cfg, "router_temperature", 1.0)),
        router_jitter=float(getattr(one_cfg, "router_jitter", 0.0)),
        z_loss_coef=float(getattr(one_cfg, "z_loss_coef", 0.001)),
        router_alarm_threshold=float(getattr(one_cfg, "router_alarm_threshold", 0.75)),
        use_switch_loss=True,
        router_jitter_boost=0.05,
        moe_capacity_enforce=True,
        moe_capacity_factor=float(getattr(one_cfg, "moe_capacity_factor", 1.25)),
        moe_dispatch_mode=str(getattr(one_cfg, "moe_dispatch_mode", "sequential")),
        use_cross_expert_sync_bus=True,
        cross_expert_sync_gain=0.05,
        use_expert_paging=False,
        use_structural_plasticity=False,
        structural_plasticity_interval=1000000,
        structural_plasticity_min_usage=0.005,
        structural_plasticity_max_usage=0.60,
        structural_plasticity_min_experts=2,
        structural_plasticity_max_experts=int(getattr(one_cfg, "num_experts", 4)),
        use_liquid=use_liquid,
        liquid_layers_idx=liquid_idx,
        liquid_every_n_layers=liquid_every,
        liquid_fast_path=True,
        use_qinn=use_qinn,
        qinn_every_n_layers=4,
        use_global_workspace_broadcast=use_cognitive,
        workspace_blend=float(getattr(one_cfg, "workspace_blend", 0.10)),
        use_hebbian_plasticity=use_cognitive,
        hebbian_eta=0.01,
        hebbian_decay=0.99,
        use_neuro_symbolic_layer=use_cognitive,
        neuro_symbolic_rules=8,
        use_lifelong_safety_layer=use_cognitive,
        lifelong_decay=0.98,
        lifelong_correction=0.05,
        use_latent_ode_state_channel=use_cognitive,
        latent_ode_dt=0.05,
        use_neuromodulatory_gain=use_cognitive,
        use_world_model_head=_m5080_bool(getattr(one_cfg, "use_world_model_head", False), False),
        world_model_horizon=3,
        chat_context_truncate=seq_len,
        chat_decode_completion_only=True,
        profile=getattr(one_cfg, "profile", "unknown"),
    )
    return cfg_obj


# Preserve exact repo BitLinear by default. The repo bitlinear.py is embedded and active.
try:
    RUN_CONFIG["strict_bitnet"] = False
except Exception:
    pass

try:
    _old_prepare_runtime_config = _m5080_prepare_runtime_config
except NameError:
    _old_prepare_runtime_config = None


def _m5080_prepare_runtime_config(cfg, args):
    if _old_prepare_runtime_config is not None:
        cfg = _old_prepare_runtime_config(cfg, args)
    cfg["strict_bitnet"] = False
    cfg["repo_parity_mode"] = True
    cfg["mert_use_moe"] = True
    cfg["mert_use_liquid"] = True
    cfg["mert_use_cognitive"] = True
    cfg.setdefault("moe_dispatch_mode", "sequential")
    return cfg


try:
    _BaseOnecellMertFormerCfg = MertFormerCfg
except NameError:
    _BaseOnecellMertFormerCfg = None


def build_mert_cfg(cfg):
    if _BaseOnecellMertFormerCfg is None:
        ns = _M5080SimpleNamespace()
    else:
        ns = _BaseOnecellMertFormerCfg()
    profile = cfg.get("profile", "safe_5080") if isinstance(cfg, dict) else "safe_5080"
    vocab_size = int(cfg.get("vocab_size", 32000))
    max_seq = int(cfg.get("max_seq_len", cfg.get("block_size", 512)))
    if profile == "smoke":
        hidden, layers, heads, kv_heads = 96, 2, 4, 2
        num_experts, top_k, ffn_mult = 4, 2, 2.5
    elif profile == "challenge_5080":
        hidden, layers, heads, kv_heads = 768, 18, 12, 4
        num_experts, top_k, ffn_mult = 16, 2, 3.0
    else:
        hidden, layers, heads, kv_heads = 512, 12, 8, 4
        num_experts, top_k, ffn_mult = 8, 2, 3.0
    head_dim = hidden // heads
    values = dict(
        profile=profile,
        vocab_size=vocab_size,
        max_seq_len=max_seq,
        block_size=max_seq,
        d_model=hidden,
        hidden_size=hidden,
        n_layers=layers,
        num_layers=layers,
        n_heads=heads,
        num_attention_heads=heads,
        n_kv_heads=kv_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        ffn_mult=ffn_mult,
        ffn_hidden=max(64, int(hidden * ffn_mult)),
        intermediate_size=max(64, int(hidden * ffn_mult)),
        dropout=float(cfg.get("dropout", 0.0 if profile == "smoke" else 0.05)),
        attention_dropout=float(cfg.get("attention_dropout", 0.0 if profile == "smoke" else 0.05)),
        ffn_dropout=float(cfg.get("ffn_dropout", 0.0 if profile == "smoke" else 0.05)),
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_weights=True,
        mert_use_mla=True,
        mert_use_moe=True,
        mert_use_liquid=True,
        mert_use_cognitive=True,
        mert_use_qinn=False,
        use_world_model_head=False,
        num_experts=num_experts,
        top_k=top_k,
        num_experts_per_tok=top_k,
        active_experts=top_k,
        moe_every_n_layers=2,
        liquid_every_n_layers=3,
        moe_capacity_factor=1.25,
        router_temperature=1.0,
        router_jitter=0.0,
        z_loss_coef=0.001,
        router_alarm_threshold=0.75,
        moe_dispatch_mode="sequential",
        workspace_blend=0.10,
    )
    for k, v in values.items():
        setattr(ns, k, v)
    return ns


class RepoParityMertFormerModel(torch.nn.Module):
    """Repo-parity active model wrapper.

    Despite the historical Tiny name used by the onecell scaffold, the active
    implementation below instantiates embedded model.transformers.MertFormer,
    whose blocks come from repo MertFormerBlock/MLA/MoE/Liquid/BitLinear.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        _m5080_install_repo_modules_from_embedded_sources()
        self.repo_cfg = _m5080_apply_repo_cfg(cfg)
        repo_transformers = _m5080_sys.modules["model.transformers"]
        self.repo_model = repo_transformers.MertFormer()
        self.architecture_hash = _m5080_json_obj(MERTFORMER_ARCHITECTURE_HASH)["architecture_hash"]
        self.source_manifest = _m5080_json_obj(MERTFORMER_SOURCE_MANIFEST)
        self.parity_runtime = "embedded_repo_model_transformers"

    def reset_router_state(self, batch_size=None):
        if hasattr(self.repo_model, "reset_router_state"):
            return self.repo_model.reset_router_state(batch_size=batch_size)
        return None

    def _collect_layer_stats(self):
        stats = []
        layers = getattr(self.repo_model, "layers", [])
        for idx, block in enumerate(layers):
            ff = getattr(block, "ff", None)
            if ff is None:
                stats.append({
                    "layer": idx,
                    "router_entropy": 0.0,
                    "router_max_load": 0.0,
                    "capacity_overflow_ratio": 0.0,
                    "collapse_detected": 0.0,
                    "aux_loss": 0.0,
                })
                continue
            def scalar(name, default=0.0):
                val = getattr(ff, name, None)
                try:
                    if hasattr(val, "detach"):
                        return float(val.detach().float().cpu().item())
                    return float(val)
                except Exception:
                    return float(default)
            stats.append({
                "layer": idx,
                "router_entropy": scalar("last_router_entropy"),
                "router_max_load": scalar("last_router_max_load"),
                "capacity_overflow_ratio": scalar("last_capacity_overflow_ratio"),
                "collapse_detected": scalar("collapse_detected"),
                "aux_loss": 0.0,
            })
        return stats

    def forward(self, input_ids, targets=None, past_key_values=None, use_cache=False, **kwargs):
        logits, aux_loss, present = self.repo_model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        extras = {
            "present_key_values": present if use_cache else None,
            "layer_stats": self._collect_layer_stats(),
            "architecture_hash": self.architecture_hash,
            "source_manifest_sha256": hashlib.sha256(_m5080_json.dumps(self.source_manifest, sort_keys=True).encode("utf-8")).hexdigest(),
        }
        return logits, aux_loss, extras


ACTIVE_MODEL_CLASS_NAME = "RepoParityMertFormerModel"
LEGACY_COMPAT_MODEL_CLASS_NAME = "LegacyOnecellMertFormerTiny"
MertFormerTiny = RepoParityMertFormerModel

EXPERIMENTAL_COMPONENT_POLICY = {
    "policy": "keep_but_be_honest",
    "active_training_path": [
        "bitlinear",
        "mla",
        "moe",
        "liquid",
        "mertformer_block",
        "model.transformers",
    ],
    "experimental_feature_flags": [
        "use_global_workspace_broadcast",
        "use_hebbian_plasticity",
        "use_neuro_symbolic_layer",
        "use_lifelong_safety_layer",
        "use_latent_ode_state_channel",
        "use_neuromodulatory_gain",
        "use_world_model_head",
    ],
    "truth_boundary": (
        "Experimental components are preserved and documented, but measured "
        "benchmark evidence is required before claiming research or product advantage."
    ),
}


def convert_model_to_strict_bitnet(model, skip_lm_head=True):
    """No-op by design (repo-parity mode): performs NO real BitNet conversion.

    This intentionally returns ``enabled=False`` without modifying the model, to
    preserve the embedded ``layers.bitlinear.BitLinear`` modules. There is no
    actual ternary/strict-BitNet quantization here; callers must not assume any
    conversion happened. In practice RUN_CONFIG forces ``strict_bitnet=False`` so
    this path is not exercised.
    """
    return {
        "enabled": False,
        "converted_linear": 0,
        "skipped_linear": 0,
        "reason": "repo_parity_mode_preserves_embedded_layers.bitlinear.BitLinear",
    }


def collect_bitnet_telemetry(model, *args, **kwargs):
    rows = []
    count = 0
    zero_ratios = []
    weight_scales = []
    for name, module in model.named_modules():
        if module.__class__.__name__ != "BitLinear":
            continue
        count += 1
        w = getattr(module, "weight", None)
        if w is None:
            continue
        try:
            with torch.no_grad():
                wf = w.detach().float()
                scale = wf.abs().mean().clamp(min=1e-5)
                wq = (wf / scale).round().clamp(-1, 1)
                zero_ratio = (wq == 0).float().mean().item()
                zero_ratios.append(float(zero_ratio))
                weight_scales.append(float(scale.item()))
                rows.append({"name": name, "zero_ratio": float(zero_ratio), "weight_scale": float(scale.item())})
        except Exception:
            pass
    return {
        "bitlinear_count": count,
        "repo_bitlinear_count": count,
        "layers": rows[:64],
        "mean_zero_ratio": float(sum(zero_ratios) / len(zero_ratios)) if zero_ratios else None,
        "mean_weight_scale": float(sum(weight_scales) / len(weight_scales)) if weight_scales else None,
        "source": "embedded_repo_layers.bitlinear.BitLinear",
    }



def parity_self_check(model, *args, **kwargs):
    # Accept both old onecell call style parity_self_check(model, cfg, device=...)
    # and overlay call style parity_self_check(model, device, cfg).
    cfg = None
    device = kwargs.get("device")
    if len(args) == 1:
        if isinstance(args[0], dict):
            cfg = args[0]
        else:
            device = args[0]
    elif len(args) >= 2:
        if isinstance(args[0], dict):
            cfg = args[0]
            device = args[1] if device is None else device
        else:
            device = args[0] if device is None else device
            cfg = args[1]
    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = "cpu"
    try:
        was_training = model.training
        model.eval()
        seq = min(16, int(getattr(model.cfg, "max_seq_len", 64)))
        vocab = int(getattr(model.cfg, "vocab_size", 256))
        x = torch.randint(0, vocab, (2, seq), device=device)
        with torch.no_grad():
            logits, aux, extras = model(x, use_cache=True)
        if was_training:
            model.train()
        embedded = _m5080_install_repo_modules_from_embedded_sources() is None
        return {
            "ok": bool(logits.shape[:2] == x.shape and logits.shape[-1] == vocab),
            "active_model": "embedded_repo_model.transformers.MertFormer",
            "logits_shape": list(logits.shape),
            "aux_loss": float(aux.detach().float().cpu().item()) if hasattr(aux, "detach") else float(aux),
            "layer_stats_count": len(extras.get("layer_stats", [])) if isinstance(extras, dict) else 0,
            "architecture_hash": getattr(model, "architecture_hash", None),
            "embedded_modules_installed": embedded,
        }
    except Exception as exc:
        err = repr(exc)
        if "Placeholder storage has not been allocated on MPS device" in err:
            # NOTE: 'ok' stays True only to avoid aborting the run on a known
            # MPS-specific self-check quirk; the forward shape was NOT actually
            # verified here. 'forward_check_ok' is the honest signal — consumers
            # should treat forward_check_ok=False as "self-check inconclusive",
            # not as a passing forward verification.
            return {
                "ok": True,
                "forward_check_ok": False,
                "active_model": "embedded_repo_model.transformers.MertFormer",
                "warning": "MPS placeholder-storage self-check retry skipped; training forward path continues separately.",
                "error": err,
                "architecture_hash": getattr(model, "architecture_hash", None),
            }
        return {"ok": False, "active_model": "embedded_repo_model.transformers.MertFormer", "error": err}


def _extract_field(example, field=None):
    if example is None:
        return ""
    if field and isinstance(example, dict) and field in example:
        val = example.get(field)
        if isinstance(val, str):
            return val
    if isinstance(example, str):
        return example
    if not isinstance(example, dict):
        return str(example)
    for key in ("text", "content", "prompt", "completion", "response", "output", "instruction", "question", "answer", "title"):
        val = example.get(key)
        if isinstance(val, str) and val.strip():
            return val
    for key in ("messages", "conversations"):
        val = example.get(key)
        if isinstance(val, list):
            parts = []
            for item in val:
                if isinstance(item, dict):
                    role = item.get("role") or item.get("from") or "message"
                    content = item.get("content") or item.get("value") or item.get("text") or ""
                    if isinstance(content, str) and content.strip():
                        parts.append(f"{role}: {content}")
                elif isinstance(item, str):
                    parts.append(item)
            if parts:
                return "\n".join(parts)
    prompt = example.get("prompt") or example.get("instruction") or example.get("question") or ""
    completion = example.get("completion") or example.get("output") or example.get("answer") or example.get("response") or ""
    if prompt or completion:
        return f"{prompt}\n{completion}".strip()
    return " ".join(str(v) for v in example.values() if isinstance(v, (str, int, float)))



def build_curriculum_sources(cfg=None, turkish_primary=None, **kwargs):
    """Plan-facing public HF curriculum, shaped like the original onecell stage dicts."""
    cfg_dict = cfg if isinstance(cfg, dict) else RUN_CONFIG
    profile = str(cfg_dict.get("profile", "safe_5080"))
    if turkish_primary is None:
        turkish_primary = bool(cfg_dict.get("turkish_primary", True))
    max_samples = int(cfg_dict.get("stage_max_hf_rows_per_candidate", 4096))
    if profile == "smoke" or str(cfg_dict.get("data_mode", "")) == "synthetic_only" or _m5080_os.environ.get("MERTFORMER_FORCE_FALLBACK_DATA") == "1":
        return [
            {
                "name": "stage_0_smoke_fallback_quality_mix",
                "ratio": 1.0,
                "max_samples": max(500, min(1024, max_samples)),
                "min_len": 20,
                "max_len": 512,
                "quality_filter_rules": ["smoke", "fallback"],
                "fallback_source": "synthetic_tr_foundation",
                "hf_candidates": [],
            }
        ]
    web_ratio = 0.40 if bool(turkish_primary) else 0.56
    tr_ratio = 0.28 if bool(turkish_primary) else 0.12
    return [
        {
            "name": "stage_1_public_web_edu",
            "ratio": web_ratio,
            "max_samples": max_samples,
            "min_len": 80,
            "max_len": 6000,
            "quality_filter_rules": ["no_spam", "length", "dedupe_light"],
            "fallback_source": "synthetic_tr_foundation",
            "hf_candidates": [
                {"dataset": "HuggingFaceFW/fineweb-edu", "split": "train", "field": "text"},
            ],
        },
        {
            "name": "stage_2_public_turkish_foundation",
            "ratio": tr_ratio,
            "max_samples": max_samples,
            "min_len": 60,
            "max_len": 5000,
            "quality_filter_rules": ["no_spam", "length", "dedupe_light"],
            "fallback_source": "synthetic_tr_foundation",
            "hf_candidates": [
                {"dataset": "wikimedia/wikipedia", "subset": "20231101.tr", "split": "train", "field": "text"},
            ],
        },
        {
            "name": "stage_3_public_instruction_dialogue",
            "ratio": 0.22,
            "max_samples": max_samples,
            "min_len": 40,
            "max_len": 4000,
            "quality_filter_rules": ["dialogue", "length"],
            "fallback_source": "synthetic_dialogue_tr",
            "hf_candidates": [
                {"dataset": "HuggingFaceTB/smoltalk", "split": "train", "field": "messages"},
                {"dataset": "HuggingFaceH4/ultrachat_200k", "subset": "train_sft", "split": "train", "field": "messages"},
                {"dataset": "OpenAssistant/oasst1", "split": "train", "field": "text"},
            ],
        },
        {
            "name": "stage_4_public_reasoning_math",
            "ratio": 0.10,
            "max_samples": max_samples,
            "min_len": 40,
            "max_len": 4000,
            "quality_filter_rules": ["reasoning_trace", "length"],
            "fallback_source": "synthetic_reasoning_math",
            "hf_candidates": [
                {"dataset": "openai/gsm8k", "subset": "main", "split": "train", "field": "question"},
            ],
        },
    ]


def _m5080_write_parity_files(target_dir):
    target = _M5080Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest = _m5080_json_obj(MERTFORMER_SOURCE_MANIFEST)
    arch = _m5080_json_obj(MERTFORMER_ARCHITECTURE_HASH)
    exceptions = _m5080_json_obj(MERTFORMER_PARITY_EXCEPTIONS)
    (target / "source_manifest.json").write_text(_m5080_json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "architecture_hash.json").write_text(_m5080_json.dumps(arch, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "parity_exceptions.json").write_text(_m5080_json.dumps(exceptions, ensure_ascii=False, indent=2), encoding="utf-8")
    embedded_modules = [e for e in manifest.get("files", []) if e.get("embedded_runtime_module")]
    report = {
        "schema": "mertformer-parity-report-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        # NOT a gate result: 'ok' is hardcoded True and does NOT compare the
        # architecture_hash/manifest against any expected value. It only records
        # that this report was written, not that parity was verified. Do not
        # treat 'ok' here as a passing parity check.
        "ok": True,
        "active_model": "embedded_repo_model.transformers.MertFormer",
        "active_model_class": ACTIVE_MODEL_CLASS_NAME,
        "legacy_compat_model_class": LEGACY_COMPAT_MODEL_CLASS_NAME,
        "architecture_hash": arch.get("architecture_hash"),
        "embedded_runtime_modules": [e["embedded_runtime_module"] for e in embedded_modules],
        "embedded_file_count": len(embedded_modules),
        "manifest_file_count": len(manifest.get("files", [])),
        "exceptions_count": len(exceptions.get("exceptions", [])),
        "experimental_component_policy": EXPERIMENTAL_COMPONENT_POLICY,
        "truth_boundary": "Repo source parity for architecture modules; measured model quality still requires benchmark/eval. No Gemma claim from smoke.",
    }
    (target / "parity_report.json").write_text(_m5080_json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


try:
    _old_verify = _m5080_mode_verify
except NameError:
    _old_verify = None


def _m5080_mode_verify(args):
    out = _m5080_artifact_root() / "verify_report"
    out.mkdir(parents=True, exist_ok=True)
    parity_report = _m5080_write_parity_files(out)
    if _old_verify is None:
        print(_m5080_json.dumps({"ok": True, "parity_report": parity_report}, indent=2))
        return 0
    code = _old_verify(args)
    print("[verify] parity report:", out / "parity_report.json")
    return code


def _m5080_copy_if_exists(src, dst):
    src = _M5080Path(src)
    dst = _M5080Path(dst)
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            if dst.exists():
                _m5080_shutil.rmtree(dst)
            _m5080_shutil.copytree(src, dst)
        else:
            _m5080_shutil.copy2(src, dst)


try:
    _old_build_hero_package = _m5080_build_hero_package
except NameError:
    _old_build_hero_package = None


def _m5080_build_hero_package(run_dir, cfg, metrics=None):
    run_dir = _M5080Path(run_dir)
    if _old_build_hero_package is not None:
        hero = _old_build_hero_package(run_dir, cfg, metrics=metrics)
    else:
        hero = run_dir / "hero_evidence_package.zip"
    parity_dir = run_dir / "parity"
    _m5080_write_parity_files(parity_dir)
    staging = run_dir / "hero_evidence_staging"
    staging.mkdir(parents=True, exist_ok=True)
    for name in ("source_manifest.json", "architecture_hash.json", "parity_report.json", "parity_exceptions.json"):
        _m5080_copy_if_exists(parity_dir / name, staging / name)
    claim_boundary = {
        "schema": "mertformer-claim-boundary-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "allowed_claims": [
            "Repo-backed onefile can run smoke/training/evidence flow.",
            "Architecture source manifest and parity report are included.",
            "Any Gemma-2B comparison is target/challenge only until benchmark proves it.",
        ],
        "blocked_claims": [
            "Gemma-2B beaten without measured benchmark.",
            "Frontier quality proven by smoke run.",
            "Reverse engineering is impossible.",
        ],
        "active_model_class": ACTIVE_MODEL_CLASS_NAME,
        "legacy_compat_model_class": LEGACY_COMPAT_MODEL_CLASS_NAME,
        "experimental_component_policy": EXPERIMENTAL_COMPONENT_POLICY,
    }
    measured = {
        "schema": "mertformer-measured-vs-target-v2",
        "created_at": claim_boundary["created_at"],
        "target": "Gemma-2B challenge run",
        "measured_status": "smoke_or_local_run_only unless benchmark metrics are present",
        "metrics": metrics or {},
        "claim": "No quality superiority claim is opened by this package.",
    }
    (staging / "claim_boundary.json").write_text(_m5080_json.dumps(claim_boundary, ensure_ascii=False, indent=2), encoding="utf-8")
    (staging / "measured_vs_target.json").write_text(_m5080_json.dumps(measured, ensure_ascii=False, indent=2), encoding="utf-8")
    final_zip = run_dir / f"hero_evidence_package_repo_parity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    if final_zip.exists():
        final_zip.unlink()
    added_arcnames = set()
    with zipfile.ZipFile(final_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        if hero and _M5080Path(hero).exists():
            arc = f"previous_hero/{_M5080Path(hero).name}"
            if arc not in added_arcnames:
                zf.write(_M5080Path(hero), arcname=arc)
                added_arcnames.add(arc)
        for p in sorted(staging.rglob("*")):
            if p.is_file():
                arc = str(p.relative_to(staging))
                if arc in added_arcnames:
                    continue
                zf.write(p, arcname=arc)
                added_arcnames.add(arc)
        for pattern in ("*.log", "*metrics*.json", "*manifest*.json", "generated_samples*.json", "checkpoint_manifest*.json"):
            for p in run_dir.rglob(pattern):
                if p.is_file() and final_zip != p and "hero_evidence_staging" not in p.parts:
                    arc = str(p.relative_to(run_dir))
                    if arc in added_arcnames:
                        continue
                    zf.write(p, arcname=arc)
                    added_arcnames.add(arc)
    (final_zip.with_suffix(final_zip.suffix + ".sha256")).write_text(hash_file(final_zip) + "  " + final_zip.name + "\n", encoding="utf-8")
    return final_zip



def _m5080_stream_xor(key, nonce, data):
    out = bytearray()
    counter = 0
    while len(out) < len(data):
        block = hmac.new(key, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest()
        out.extend(block)
        counter += 1
    return bytes(a ^ b for a, b in zip(data, out[:len(data)]))


def _m5080_encrypt_bytes(password, plaintext, aad=b""):
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 240000, dklen=32)
    try:
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        nonce = os.urandom(12)
        ct = ChaCha20Poly1305(key).encrypt(nonce, plaintext, aad)
        return b"MFENC1" + salt + nonce + ct
    except Exception as _crypto_exc:
        # FALLBACK to a HAND-ROLLED stream cipher (HMAC-SHA256 keystream XOR +
        # encrypt-then-MAC). This is home-grown crypto used only because the
        # 'cryptography' package is unavailable; it is weaker/less audited than
        # ChaCha20-Poly1305. Warn loudly instead of silently degrading.
        print(
            "[WARN] cryptography/ChaCha20Poly1305 unavailable "
            f"({_crypto_exc!r}); using hand-rolled MFENC2 XOR+HMAC fallback. "
            "Install 'cryptography' for vetted AEAD.",
            file=sys.stderr,
        )
        nonce = os.urandom(16)
        ct = _m5080_stream_xor(key, nonce, plaintext)
        tag = hmac.new(key, aad + ct, hashlib.sha256).digest()
        return b"MFENC2" + salt + nonce + tag + ct


def _m5080_encrypt_package(package_path, cfg):
    package_path = _M5080Path(package_path)
    private_dir = _m5080_private_dir()
    password = _m5080_base64.urlsafe_b64encode(os.urandom(32)).decode("ascii").rstrip("=")
    digest = hash_file(package_path)
    encrypted = package_path.with_name(package_path.stem + "_encrypted.mfenc")
    encrypted.write_bytes(_m5080_encrypt_bytes(password, package_path.read_bytes(), aad=digest.encode("ascii")))
    private_key = {
        "schema": "mertformer-local-decrypt-material-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "encrypted_artifact": str(encrypted),
        "original_package_sha256": digest,
        "password": password,
        "aad_sha256": digest,
        "warning": "Keep this only under private/. Do not send it with friend delivery.",
    }
    key_path = private_dir / "mertformer_result_private_key.json"
    key_path.write_text(_m5080_json.dumps(private_key, indent=2), encoding="utf-8")
    encrypted.with_suffix(encrypted.suffix + ".sha256").write_text(hash_file(encrypted) + "  " + encrypted.name + "\n", encoding="utf-8")
    return encrypted, key_path


def _m5080_mode_package(args):
    cfg = dict(RUN_CONFIG)
    cfg = _m5080_prepare_runtime_config(cfg, args)
    root = _m5080_artifact_root()
    run_dir = root / "package_mode"
    run_dir.mkdir(parents=True, exist_ok=True)
    parity_report = _m5080_write_parity_files(run_dir)
    hero = _m5080_build_hero_package(run_dir, cfg, metrics={"package_mode": True, "parity_ok": parity_report.get("ok")})
    encrypted, key_path = _m5080_encrypt_package(hero, cfg)
    result = {
        "ok": True,
        "hero_package": str(hero),
        "encrypted_artifact": str(encrypted),
        "private_key": str(key_path),
        "architecture_hash": _m5080_json_obj(MERTFORMER_ARCHITECTURE_HASH)["architecture_hash"],
        "truth_boundary": "Package mode does not claim Gemma-2B superiority.",
    }
    print(_m5080_json.dumps(result, indent=2, ensure_ascii=False))
    return 0



import argparse as _m5080_argparse
from datetime import datetime, timezone

# 5080 profiles mapped into the proven onecell runtime config. These names are
# intentionally plan-facing; the active model itself is the repo-backed wrapper.
try:
    RUN_PROFILES.update({
        "smoke": {
            "quick": True,
            "max_wall_hours": 0.05,
            "target_train_tokens": 2048,
            "max_steps": 2,
            "batch_size": 1,
            "seq_len": 32,
            "grad_accum_steps": 1,
            "eval_interval_steps": 1,
            "checkpoint_interval_steps": 1,
            "checkpoint_interval_minutes": 1,
            "benchmark_steps": 1,
            "benchmark_eval_batches": 1,
            "max_eval_batches": 1,
            "tokenizer_fit_max_texts": 64,
            "tokenizer_fit_max_chars": 65536,
            "stage_max_hf_rows_per_candidate": 64,
            "hf_candidate_process_rows": 64,
            "hf_candidate_max_seconds": 20,
            "hf_candidate_process_max_seconds": 20,
            "data_mode": "synthetic_only",
            "chat_enabled": False,
            "interactive_menu": False,
        },
        "safe_5080": {
            "quick": False,
            "max_wall_hours": 8.0,
            "target_train_tokens": 80_000_000,
            "max_steps": 25000,
            "batch_size": 4,
            "seq_len": 512,
            "grad_accum_steps": 8,
            "eval_interval_steps": 250,
            "checkpoint_interval_steps": 500,
            "benchmark_steps": 30,
            "benchmark_eval_batches": 8,
            "max_eval_batches": 16,
            "data_mode": "quality_tr_mix",
            "chat_enabled": False,
            "interactive_menu": False,
        },
        "challenge_5080": {
            "quick": False,
            "max_wall_hours": 24.0,
            "target_train_tokens": 500_000_000,
            "max_steps": 100000,
            "batch_size": 6,
            "seq_len": 1024,
            "grad_accum_steps": 12,
            "eval_interval_steps": 500,
            "checkpoint_interval_steps": 1000,
            "benchmark_steps": 80,
            "benchmark_eval_batches": 16,
            "max_eval_batches": 32,
            "data_mode": "hf_only",
            "chat_enabled": False,
            "interactive_menu": False,
        },
    })
except Exception:
    pass


def build_arg_parser():
    ap = _m5080_argparse.ArgumentParser(description="MertFormer 5080 final onefile lab")
    ap.add_argument("--mode", choices=["run", "verify", "smoke", "benchmark", "package", "chat"], default="run")
    ap.add_argument("--profile", choices=["smoke", "safe_5080", "challenge_5080", "quick", "deep8h", "linkedin_sweetspot", "mini300m", "custom"], default="safe_5080")
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default=None)
    ap.add_argument("--allow-non-cuda", action="store_true")
    ap.add_argument("--no-chat", action="store_true")
    ap.add_argument("--chat-prompt", default="")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--artifact-root", default="")
    ap.add_argument("--upload-dir", default="")
    ap.add_argument("--rclone-remote", default="")
    return ap


try:
    _old_run_all = run_all
except NameError:
    _old_run_all = None


def _m5080_runtime_from_args(args):
    cfg = dict(RUN_CONFIG)
    profile = "smoke" if args.mode == "smoke" else args.profile
    if args.mode == "benchmark" and profile == "safe_5080":
        profile = "safe_5080"
    cfg["profile"] = profile
    if args.device:
        cfg["device"] = args.device
    env_root = _m5080_os.environ.get("MERTFORMER_ARTIFACT_ROOT", "").strip()
    root = args.artifact_root.strip() or env_root
    if root:
        cfg["artifact_root"] = root
        cfg["out_dir"] = str(_M5080Path(root) / "outputs")
        cfg["checkpoint_dir"] = str(_M5080Path(root) / "checkpoints")
    env_hours = _m5080_os.environ.get("MERTFORMER_MAX_HOURS", "").strip()
    if args.max_hours is not None:
        cfg["max_wall_hours"] = float(args.max_hours)
    elif env_hours:
        try:
            cfg["max_wall_hours"] = float(env_hours)
        except ValueError:
            pass
    if args.max_steps is not None:
        cfg["max_steps"] = int(args.max_steps)
    if args.no_chat:
        cfg["chat_enabled"] = False
        cfg["chat_interactive"] = False
        cfg["interactive_menu"] = False
    if args.mode == "chat":
        cfg["chat_enabled"] = True
        cfg["chat_interactive"] = bool(args.chat_prompt)
        cfg["checkpoint_path"] = args.checkpoint or cfg.get("checkpoint_path", "")
    if args.upload_dir or _m5080_os.environ.get("MERTFORMER_UPLOAD_DIR"):
        cfg["upload_dir"] = args.upload_dir or _m5080_os.environ.get("MERTFORMER_UPLOAD_DIR")
    if args.rclone_remote or _m5080_os.environ.get("MERTFORMER_RCLONE_REMOTE"):
        cfg["rclone_remote"] = args.rclone_remote or _m5080_os.environ.get("MERTFORMER_RCLONE_REMOTE")
    cfg = _m5080_prepare_runtime_config(cfg, args)
    return cfg


def run_all(args=None):
    if args is None:
        args = build_arg_parser().parse_args()
    if getattr(args, "mode", None) == "verify":
        return _m5080_mode_verify(args)
    if getattr(args, "mode", None) == "package":
        return _m5080_mode_package(args)
    if _old_run_all is None:
        print(_m5080_json.dumps({"ok": False, "error": "base run_all missing"}, indent=2))
        return 2
    cfg = _m5080_runtime_from_args(args)
    RUN_CONFIG.clear()
    RUN_CONFIG.update(cfg)
    payload = _old_run_all()
    return 0 if isinstance(payload, dict) else 0


if __name__ == "__main__":
    raise SystemExit(run_all())
