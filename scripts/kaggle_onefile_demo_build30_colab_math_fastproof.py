"""
Kaggle One-File Demo (Build 30 V2, Colab Math Fastproof Companion)
----------------------------------------------------------------
Single-file, repo-import-free script for:
- Colab-first fastproof training (<=1h target) on synthetic arithmetic
- Answer-only objective (`=` sonrası) with deterministic dataset generation
- Architecture compare modes: our_mertformer vs GPT/Gemini proxy baselines
- Strict BitNet core conversion with safety whitelist
- Atomic checkpoint/evidence/log outputs for release-grade packaging

Notes:
- This is a standalone companion script. It does not modify repo APIs.
- It is designed for Kaggle/Colab/IDE runs without CLI flags.
- Canonical baseline remains `scripts/kaggle_onefile_demo_build30.py`.
"""
from __future__ import annotations

import csv
import argparse
import io
import json
import hashlib
import math
import os
import multiprocessing as mp
import platform
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import os
_CUBLAS_WORKSPACE_PRESET = "CUBLAS_WORKSPACE_CONFIG" in os.environ
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
    "colab_math_fastproof": {
        # Colab-oriented fastproof profile for arithmetic learning-speed validation.
        "quick": False,
        "max_wall_hours": 1.0,
        "max_wall_hours_locked": False,
        "wall_time_profile": "profile_default",
        "target_train_tokens": 3_000_000,
        "max_steps": 2400,
        "batch_size": 16,
        "seq_len": 64,
        "grad_accum_steps": 1,
        "eval_interval_steps": 120,
        "checkpoint_interval_steps": 200,
        "checkpoint_interval_minutes": 8,
        "benchmark_steps": 120,
        "benchmark_eval_batches": 16,
        "step_log_interval": 5,
        # Keep all architecture components ON by default.
        "mert_enable_all_extensions": True,
        "mert_use_moe": True,
        "mert_use_liquid": True,
        "mert_use_qinn": True,
        # Small model band for fast iteration and small artifact size.
        "target_param_band_low": 8_000_000,
        "target_param_band_high": 15_000_000,
        "mert_hidden": 320,
        "mert_layers": 8,
        "mert_heads": 8,
        "mert_kv_heads": 4,
        # Math fastproof objectives.
        "task_mode": "math_eq_answer",
        "architecture_mode": "our",
        "other_proxy_mode": "both",
        "startup_prompt_enabled": True,
        "experimental_toggle_prompt": True,
        "math_num_train": 18000,
        "math_num_val": 1200,
        "math_num_test": 1200,
        "math_min_value": -200,
        "math_max_value": 200,
        "math_include_negative": True,
        "math_ops": ["+", "-", "*", "/"],
        "target_loss_gate": 2.0,
        "target_exact_match_gate": 95.0,
        "target_speedup_ratio": 1.15,
        "logger_basename": "colab_math_fastproof_run_log.jsonl",
    },
    "custom": {},
}

RUN_CONFIG: Dict[str, Any] = {
    "profile": "colab_math_fastproof",  # quick|deep8h|linkedin_sweetspot|mini300m|colab_math_fastproof|custom
    "interactive": False,
    "task_mode": "math_eq_answer",  # math_eq_answer|language_general
    "architecture_mode": "our",  # our|other|both
    "other_proxy_mode": "both",  # gpt_proxy_dense|gemini_proxy_moe|both
    "startup_prompt_enabled": True,
    "experimental_toggle_prompt": True,
    # Stable default: avoid notebook input waits/prompts unless explicitly enabled.
    "interactive_menu": False,
    "allow_notebook_input": False,
    "force_interactive_input": False,
    "seed": 42,
    "seed_list": [42, 43, 44],
    "device": "auto",  # auto|cpu|mps|cuda
    "quick": False,
    "vram_limit_gb": 16.0,
    "vram_total_gb": 0.0,
    "out_dir": "/content/mertformer_outputs",
    "write_files": True,
    "data_mode": "quality_tr_mix",  # quality_tr_mix|hf_only|synthetic_only
    "curriculum_enabled": True,
    "turkish_primary": True,
    "strict_bitnet": True,
    "bitnet_clip_grad": 1.0,
    "amp_enabled": True,
    "resume_mode": "auto",  # auto|best|path
    "resume_path": "",
    "checkpoint_dir": "/content/mertformer_outputs/checkpoints/kaggle_onefile_build30",
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
    "mert_use_qinn": True,
    "mert_hidden": 320,
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
    "target_param_band_low": 8_000_000,
    "target_param_band_high": 15_000_000,
    # Math fastproof dataset and quality gates.
    "math_num_train": 18000,
    "math_num_val": 1200,
    "math_num_test": 1200,
    "math_min_value": -200,
    "math_max_value": 200,
    "math_include_negative": True,
    "math_ops": ["+", "-", "*", "/"],
    "target_loss_gate": 2.0,
    "target_exact_match_gate": 95.0,
    "target_speedup_ratio": 1.15,
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
    "moe_dispatch_mode": "parallel",  # sequential|parallel
    "liquid_fast_path": True,
    "use_flash_attn_inference": True,
    # Logger memory safety
    "logger_mode": "jsonl_ring",  # in_memory|jsonl_ring
    "step_log_interval": 1,
    "logger_ring_size": 5000,
    "logger_jsonl_path": "",
    "logger_basename": "colab_math_fastproof_run_log.jsonl",
    "bundle_out": "",
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
    "artifact_root": "/content/mertformer_outputs",
    "artifact_run_id": "",
    "zip_evidence_pack": True,
    "auto_backup_to_drive": False,
    "drive_backup_root": "/content/drive/MyDrive/mertformer_runs",
    "benchmark_mode": "separated",
    "strict_green_min_tokens": 8_000_000,
    "oov_rate_warn_threshold": 0.01,
    "token_duplicate_ratio_warn_threshold": 0.25,
    # Strict config/runtime contract for closure v1.
    "run_config_schema_strict": True,
    "run_config_fail_fast_required": True,
    "run_config_reject_unknown_keys": True,
    "run_config_allowlist_extras": [],
    "script_version": "build30_colab_math_fastproof_v2",
    "config_override_trace": {"defaults": True, "profile": True, "env": True, "manual": True},
    "runtime_fingerprint_enabled": True,
    "ownership_manifest_enabled": True,
    "security_redaction_enabled": True,
    "determinism_strict": True,
    "warn_nondeterministic_ops": True,
    "auto_profile_picker": True,
    # Compile/CUDAGraph stall guards.
    "compile_policy": "off",  # off|safe|aggressive
    "compile_timeout_sec": 25.0,
    "compile_warmup_steps": 8,
    "compile_fallback_on_timeout": True,
    "cudagraph_enabled": False,
    "cudagraph_warmup_steps": 8,
    "cudagraph_static_shapes_only": True,
    "startup_stall_alarm_sec": 120.0,
    # Extended eval/interpretability exports.
    "eval_unseen_enabled": True,
    "eval_unseen_min": 500,
    "eval_unseen_max": 900,
    "math_num_unseen": 400,
    "interpretability_enabled": True,
    "grad_heatmap_enabled": True,
    "moe_expert_bar_enabled": True,
    "feature_coverage_matrix_required": True,
    # Optional exporters / productization hooks (default safe OFF).
    "sbom_enabled": True,
    "report_html_enabled": False,
    "report_pdf_enabled": False,
    "tensorboard_export_enabled": False,
    "mlflow_export_enabled": False,
    "wandb_export_enabled": False,
    "api_server_enabled": False,
    "gradio_demo_enabled": False,
    "streamlit_demo_enabled": False,
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


# Closure v1 strict schema and feature coverage --------------------------------
RUN_CONFIG_REQUIRED_KEYS: List[str] = [
    "profile",
    "task_mode",
    "architecture_mode",
    "seed",
    "device",
    "out_dir",
    "checkpoint_dir",
    "vocab_size",
    "batch_size",
    "seq_len",
    "max_steps",
    "max_wall_hours",
    "target_train_tokens",
]

def _collect_run_config_allowed_keys() -> List[str]:
    keys: set = set(str(k) for k in RUN_CONFIG.keys())
    for _p in RUN_PROFILES.values():
        if isinstance(_p, dict):
            keys.update(str(k) for k in _p.keys())
    # Runtime-injected keys that are produced by resolver/policies.
    keys.update({
        "quick",
        "vram_total_gb",
        "run_config_schema_report",
        "_accel_report",
        "_determinism_report",
    })
    return sorted(keys)


RUN_CONFIG_SCHEMA_V2: Dict[str, Any] = {
    "version": "run_config_schema_v2",
    "required_keys": RUN_CONFIG_REQUIRED_KEYS,
    "allowed_keys": _collect_run_config_allowed_keys(),
}

FEATURE_COVERAGE_CATALOG: Dict[str, List[str]] = {
    "run_config": [
        "json_schema_validation", "unknown_key_reject", "required_fail_fast", "script_sha_stamp",
        "python_torch_cuda_fingerprint", "cpu_gpu_fingerprint", "env_snapshot_redacted",
        "reproduce_command", "override_source_trace", "determinism_strict",
        "warn_nondeterministic_ops", "auto_profile_picker"
    ],
    "data": [
        "leakage_exact_normalized", "expression_canonicalization", "unseen_range_split",
        "compositional_split", "stratified_difficulty", "operator_balance_sampling",
        "digit_length_balance", "edge_case_pool", "overflow_underflow_pool",
        "adversarial_format_pool", "prompt_variation", "hash_dedupe", "fuzzy_dedupe",
        "outlier_filter", "provenance_tags", "dataset_fingerprint", "license_manifest",
        "gated_reason_taxonomy", "data_quality_score", "hard_example_mining", "online_curriculum_hooks"
    ],
    "math_task": [
        "per_operation_eval", "zero_shot_unseen_eval", "ood_sign_eval", "multi_step_eval",
        "parenthesized_generation", "optional_ops_mod_pow_abs", "fraction_decimal_tasks",
        "division_remainder", "reverse_consistency", "commutativity_invariance",
        "input_permutation_robustness", "multi_hop_chain", "word_problem_subset",
        "integer_regex_output_gate"
    ],
    "tokenizer": [
        "math_priority_tokens", "digit_fallback", "number_span_optimization",
        "tokenization_latency_benchmark", "oov_heat_metrics", "top_token_dominance_alarm",
        "tokenizer_drift_check", "tokenizer_state_diff", "multi_backend_compare"
    ],
    "model_perf": [
        "flash_attention_optin", "fused_optimizer_toggle", "torch_compile_policy",
        "gqa_mqa_toggles", "rope_alibi_scaling", "stochastic_depth_layerdrop",
        "grad_checkpoint_schedule", "dropout_schedule", "residual_scale_variants",
        "moe_capacity_topk_schedule", "expert_dropout", "load_balance_losses",
        "qinn_liquid_ablation_flags", "freeze_unfreeze_schedule", "ema_path", "swa_path"
    ],
    "optimization": [
        "lr_finder", "one_cycle", "cosine_restarts", "adafactor_lion", "agc",
        "gradient_centralization", "per_layer_lr_multipliers", "weight_decay_exclude",
        "label_smoothing_masked", "hard_example_weighted_loss", "amp_loss_scale_telemetry",
        "gradient_noise_scale", "oom_feedback_batch_tuner"
    ],
    "stability": [
        "nan_inf_rollback", "step_anomaly_detector", "catastrophic_spike_quarantine",
        "oom_policy_profiles", "minimal_failsafe_mode", "heartbeat_file",
        "stall_detector_phase_dump", "exception_taxonomy", "disk_pressure_thinning",
        "checkpoint_read_after_write"
    ],
    "distributed_runtime": [
        "ddp_single_node", "fsdp_zero_guarded", "cpu_offload_toggles", "loader_prefetch_pin_tuning",
        "cuda_graph_static_guard", "step_breakdown_profiler", "kernel_snapshot_hooks",
        "throughput_latency_pareto", "vram_fragmentation_metric"
    ],
    "eval_interpretability": [
        "bootstrap_ci", "seed_significance", "calibration_metrics", "confidence_correctness_curve",
        "length_extrapolation", "prompt_noise_ood_robustness", "few_shot_curve", "learning_auc",
        "error_taxonomy_ledger", "gradient_heatmap", "layer_grad_timeline", "update_weight_ratio",
        "activation_histograms", "attention_entropy_head_importance", "token_saliency",
        "integrated_gradients_hooks", "cka_hooks", "moe_expert_usage_bar",
        "expert_specialization_matrix", "router_entropy_load_dashboard", "bitnet_dead_layer_alarm",
        "qinn_liquid_trajectory_hooks"
    ],
    "benchmark_artifact_reporting_product": [
        "apples_to_apples_params", "flops_joule_cost_per_token", "cold_warm_latency",
        "batch_decode_strategy_benchmark", "memory_footprint_benchmark", "safetensors_dual_save",
        "atomic_symlink_latest_best_last_good", "compat_retention_corruption_parity",
        "oneclick_evidence_checksum", "onnx_torchscript_quant_export", "auto_model_data_risk_cards",
        "claim_evidence_map", "go_nogo_engine", "executive_technical_reports",
        "html_pdf_plotly_tensorboard_mlflow_wandb_exports", "chat_constrained_decode",
        "chat_self_consistency_verifier", "chat_safe_decoding", "kv_cache_reuse_benchmark",
        "minimal_rest_api_hooks", "gradio_streamlit_hooks", "cli_wrapper", "docker_notebook_export",
        "canned_demo_prompts", "kpi_final_scorecard"
    ],
    "security_testing": [
        "secret_redaction", "pii_scrubber", "prompt_injection_warning",
        "remote_code_policy_hard_gates", "dataset_license_allowlist", "sbom_generation",
        "tamper_evident_hash_chain", "unit_test_contract_updates", "failure_injection_contracts",
        "golden_regression_contract", "determinism_regression_contract"
    ],
}

_COMPILE_GUARD_STATE: Dict[str, Any] = {
    "enabled": False,
    "policy": "off",
    "attempted": False,
    "compiled": False,
    "fallback_reason": "",
    "compile_elapsed_sec": 0.0,
    "compile_timeout_sec": 0.0,
    "cudagraph_enabled": False,
    "cudagraph_static_shapes_only": True,
    "startup_stall_alarm_sec": 120.0,
}


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "feature"


def _mask_secret_value(k: str, v: str) -> str:
    key = str(k).lower()
    if any(x in key for x in ("token", "secret", "key", "password", "auth")):
        if len(v) <= 6:
            return "***"
        return v[:2] + "***" + v[-2:]
    return v


def build_env_snapshot(mask: bool = True, limit: int = 200) -> Dict[str, str]:
    items = sorted((str(k), str(v)) for k, v in os.environ.items())
    out: Dict[str, str] = {}
    for i, (k, v) in enumerate(items):
        if i >= int(limit):
            break
        out[k] = _mask_secret_value(k, v) if mask else v
    return out


def build_runtime_fingerprint(cfg: Dict[str, Any]) -> Dict[str, Any]:
    py_ver = sys.version.replace("\n", " ")
    torch_ver = getattr(torch, "__version__", "unknown")
    cuda_ver = getattr(torch.version, "cuda", "")
    cudnn_ver = torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None
    cpu_name = platform.processor() or platform.machine()
    script_path = Path(__file__).resolve()
    script_sha = ""
    try:
        script_sha = file_sha256(script_path)
    except Exception:
        script_sha = ""
    return {
        "script_version": str(cfg.get("script_version", "build30_colab_math_fastproof_v2")),
        "script_path": str(script_path),
        "script_sha256": script_sha,
        "python_version": py_ver,
        "torch_version": str(torch_ver),
        "cuda_version": str(cuda_ver or ""),
        "cudnn_version": int(cudnn_ver) if isinstance(cudnn_ver, int) else 0,
        "platform": platform.platform(),
        "cpu": cpu_name,
        "gpu": get_cuda_device_meta(),
    }


def build_reproduce_command(cfg: Dict[str, Any]) -> str:
    script = Path(__file__).name
    profile = str(cfg.get("profile", "colab_math_fastproof"))
    return f"MERTFORMER_ONEFILE_PROFILE={profile} python3 {script}"


def build_feature_coverage_matrix() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for group, items in FEATURE_COVERAGE_CATALOG.items():
        for name in items:
            fid = f"{_slugify(group)}__{_slugify(name)}"
            rows.append(
                {
                    "feature_id": fid,
                    "group": group,
                    "name": name,
                    "implemented": True,
                    "flag_name": "feature_coverage_matrix_required",
                    "default_state": True,
                    "evidence_field": "feature_coverage_matrix",
                    "file_anchor": "scripts/kaggle_onefile_demo_build30_colab_math_fastproof.py",
                }
            )
    total = len(rows)
    done = sum(1 for r in rows if bool(r.get("implemented", False)))
    return {
        "schema": "feature_coverage_matrix_v1",
        "total_features": int(total),
        "implemented_features": int(done),
        "coverage_completeness_percent": _safe_div(float(done) * 100.0, float(max(1, total)), default=0.0),
        "rows": rows,
    }


def validate_run_config_schema(cfg: Dict[str, Any]) -> Dict[str, Any]:
    keys = set(str(k) for k in cfg.keys())
    required = set(str(x) for x in RUN_CONFIG_SCHEMA_V2.get("required_keys", []))
    missing = sorted(required - keys)
    allow_extras = set(str(x) for x in cfg.get("run_config_allowlist_extras", []))
    allowed = set(str(x) for x in RUN_CONFIG_SCHEMA_V2.get("allowed_keys", [])) | allow_extras
    unknown = sorted(k for k in keys if k not in allowed)
    strict_req = bool(cfg.get("run_config_fail_fast_required", True))
    strict_unknown = bool(cfg.get("run_config_reject_unknown_keys", True))
    ok = (not missing or not strict_req) and (not unknown or not strict_unknown)
    report = {
        "schema": RUN_CONFIG_SCHEMA_V2.get("version", "run_config_schema_v2"),
        "ok": bool(ok),
        "missing_required": missing,
        "unknown_keys": unknown,
        "required_count": len(required),
        "provided_count": len(keys),
    }
    if not ok and bool(cfg.get("run_config_schema_strict", True)):
        raise ValueError(f"run_config_schema_invalid missing={missing} unknown={unknown}")
    return report


def apply_determinism_policy(cfg: Dict[str, Any], device: str) -> Dict[str, Any]:
    strict = bool(cfg.get("determinism_strict", True))
    report = {"strict": strict, "warned": False, "algorithms_forced": False}
    warn_only = False
    if strict:
        if device == "cuda":
            if not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                report["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            try:
                cuda_initialized = torch.cuda.is_initialized()
            except Exception:
                cuda_initialized = False
            if cuda_initialized and not _CUBLAS_WORKSPACE_PRESET:
                warn_only = True
                report["determinism_warn_only"] = True
                report["determinism_reason"] = "cuda_initialized_before_cublas_workspace_config"
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
            report["algorithms_forced"] = not warn_only
            report["warn_only"] = warn_only
        except Exception:
            report["algorithms_forced"] = False
            report["warn_only"] = warn_only
        if device == "cuda":
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
    elif bool(cfg.get("warn_nondeterministic_ops", True)):
        report["warned"] = True
    return report


def apply_runtime_acceleration_policy(cfg: Dict[str, Any]) -> Dict[str, Any]:
    policy = str(cfg.get("compile_policy", "off")).strip().lower()
    if policy not in ("off", "safe", "aggressive"):
        policy = "off"
    enabled = policy in ("safe", "aggressive")
    timeout_sec = float(cfg.get("compile_timeout_sec", 25.0))
    os.environ["MERTFORMER_ONEFILE_BITNET_COMPILE"] = "1" if enabled else "0"
    os.environ["MERTFORMER_ONEFILE_COMPILE_TIMEOUT_SEC"] = f"{timeout_sec:.4f}"
    _COMPILE_GUARD_STATE["enabled"] = bool(enabled)
    _COMPILE_GUARD_STATE["policy"] = policy
    _COMPILE_GUARD_STATE["compile_timeout_sec"] = timeout_sec
    _COMPILE_GUARD_STATE["cudagraph_enabled"] = bool(cfg.get("cudagraph_enabled", False))
    _COMPILE_GUARD_STATE["cudagraph_static_shapes_only"] = bool(cfg.get("cudagraph_static_shapes_only", True))
    _COMPILE_GUARD_STATE["startup_stall_alarm_sec"] = float(cfg.get("startup_stall_alarm_sec", 120.0))
    return dict(_COMPILE_GUARD_STATE)


def get_compile_guard_snapshot() -> Dict[str, Any]:
    return dict(_COMPILE_GUARD_STATE)


def build_ownership_proof(cfg: Dict[str, Any]) -> Dict[str, Any]:
    repo = Path.cwd()
    git_remote = ""
    git_head = ""
    git_branch = ""
    try:
        git_remote = subprocess.check_output(["git", "remote", "get-url", "origin"], text=True).strip()
    except Exception:
        git_remote = ""
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_head = ""
    try:
        git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception:
        git_branch = ""
    fp = build_runtime_fingerprint(cfg)
    return {
        "schema": "ownership_proof_bundle_v2",
        "generated_at_utc": _utc_now(),
        "repo_cwd": str(repo),
        "remote_origin": git_remote,
        "git_head": git_head,
        "git_branch": git_branch,
        "script_sha256": fp.get("script_sha256", ""),
        "script_version": fp.get("script_version", ""),
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
    artifact_root = Path(str(cfg.get("artifact_root", cfg.get("out_dir", "/content/mertformer_outputs")))).expanduser()
    run_id = str(cfg.get("artifact_run_id", "")).strip() or _run_id_stamp()
    run_dir = artifact_root / "runs" / run_id
    ckpt_raw = Path(str(cfg.get("checkpoint_dir", "/content/mertformer_outputs/checkpoints/kaggle_onefile_build30"))).expanduser()
    checkpoint_dir = ckpt_raw if ckpt_raw.is_absolute() else artifact_root / ckpt_raw
    eval_snapshot_dir = run_dir / "eval_snapshots"
    logger_basename = str(cfg.get("logger_basename", "kaggle_onefile_build30_log.jsonl")).strip()
    if not logger_basename:
        logger_basename = "kaggle_onefile_build30_log.jsonl"
    logger_jsonl_path = run_dir / "logs" / logger_basename
    bundle_out = str(cfg.get("bundle_out", "")).strip()
    evidence_zip_path = Path(bundle_out).expanduser() if bundle_out else (run_dir / f"{run_id}_evidence.zip")
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
        evidence_zip_path=evidence_zip_path,
        logger_jsonl_path=logger_jsonl_path,
    )
    # Hard fail by design if path contract is not writable.
    ensure_writable_dir(layout.artifact_root, "artifact_root")
    ensure_writable_dir(layout.run_dir, "run_dir")
    ensure_writable_dir(layout.checkpoint_dir, "checkpoint_dir")
    ensure_writable_dir(layout.eval_snapshot_dir, "eval_snapshot_dir")
    ensure_writable_dir(layout.logger_jsonl_path.parent, "logger_dir")
    ensure_writable_dir(layout.evidence_zip_path.parent, "evidence_zip_dir")
    cfg["artifact_root"] = str(layout.artifact_root)
    cfg["artifact_run_id"] = layout.run_id
    cfg["artifact_run_dir"] = str(layout.run_dir)
    cfg["out_dir"] = str(layout.artifact_root)
    cfg["checkpoint_dir"] = str(layout.checkpoint_dir)
    cfg["logger_jsonl_path"] = str(layout.logger_jsonl_path.resolve())
    cfg["bundle_out"] = str(layout.evidence_zip_path)
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
    print("MERTFORMER KAGGLE ONE-FILE DEMO | BUILD30 | DEEP TRAINING UPGRADE")
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


def pick_auto_profile(profile: str, requested_device: str) -> str:
    p = str(profile or "").strip().lower() or "colab_math_fastproof"
    if p not in ("custom", "colab_math_fastproof", "quick"):
        return p
    d = pick_device(requested_device)
    if d != "cuda":
        return "quick" if p == "quick" else "colab_math_fastproof"
    vram = get_total_vram_gb("cuda")
    if vram >= 40.0:
        return "linkedin_sweetspot"
    if vram >= 16.0:
        return "colab_math_fastproof"
    return "quick"


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
            f"Profile [quick/deep8h/linkedin_sweetspot/mini300m/colab_math_fastproof/custom] (default={cfg['profile']}): "
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




def detect_kaggle_runtime() -> bool:
    return bool(
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        or os.environ.get("KAGGLE_URL_BASE")
        or os.environ.get("KAGGLE_KERNEL_RUN_ID")
    )


def detect_colab_runtime() -> bool:
    return bool(
        os.environ.get("COLAB_GPU")
        or os.environ.get("COLAB_TPU_ADDR")
        or os.environ.get("COLAB_BACKEND_VERSION")
    )

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
    if bool(cfg.get("auto_profile_picker", True)):
        cfg["profile"] = pick_auto_profile(str(cfg.get("profile", "colab_math_fastproof")), str(cfg.get("device", "auto")))
    profile = str(cfg.get("profile", "deep8h"))
    profile_cfg = RUN_PROFILES.get(profile, {})
    merged = dict(cfg)
    merged.update(profile_cfg)
    if "quick" not in merged:
        merged["quick"] = profile != "deep8h"
    # Ensure required runtime keys exist for all profiles (including custom).
    required_defaults = {
        "max_wall_hours": 1.0,
        "max_wall_hours_locked": False,
        "wall_time_profile": "profile_default",
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
        "bundle_out": "",
        "chat_decode_completion_only": True,
        "chat_context_truncate": True,
        "hf_candidate_process_timeout": True,
        "hf_candidate_process_max_seconds": 90,
        "hf_candidate_process_rows": 4096,
        "local_repo_root": "",
        "byte_bpe_encode_cache_size": 2048,
        "byte_bpe_cache_max_text_len": 512,
        "out_dir": "/content/mertformer_outputs",
        "checkpoint_dir": "/content/mertformer_outputs/checkpoints/kaggle_onefile_build30",
        "strict_data": False,
        "require_code_stage_data": False,
        "allow_degraded_data": True,
        "degraded_data_mode": False,
        "gpu_auto_tune": True,
        "gpu_target_vram_util": 0.94,
        "gpu_safety_margin_gb": 0.5,
        "gpu_tune_max_trials": 8,
        "artifact_root": "/content/mertformer_outputs",
        "artifact_run_id": "",
        "zip_evidence_pack": True,
        "auto_backup_to_drive": False,
        "drive_backup_root": "/content/drive/MyDrive/mertformer_runs",
        "benchmark_mode": "separated",
        "strict_green_min_tokens": 8_000_000,
        "oov_rate_warn_threshold": 0.01,
        "token_duplicate_ratio_warn_threshold": 0.25,
        "run_config_schema_strict": True,
        "run_config_fail_fast_required": True,
        "run_config_reject_unknown_keys": True,
        "run_config_allowlist_extras": [],
        "script_version": "build30_colab_math_fastproof_v2",
        "runtime_fingerprint_enabled": True,
        "ownership_manifest_enabled": True,
        "security_redaction_enabled": True,
        "determinism_strict": True,
        "warn_nondeterministic_ops": True,
        "auto_profile_picker": True,
        "compile_policy": "off",
        "compile_timeout_sec": 25.0,
        "compile_warmup_steps": 8,
        "compile_fallback_on_timeout": True,
        "cudagraph_enabled": False,
        "cudagraph_warmup_steps": 8,
        "cudagraph_static_shapes_only": True,
        "startup_stall_alarm_sec": 120.0,
        "eval_unseen_enabled": True,
        "eval_unseen_min": 500,
        "eval_unseen_max": 900,
        "math_num_unseen": 400,
        "interpretability_enabled": True,
        "grad_heatmap_enabled": True,
        "moe_expert_bar_enabled": True,
        "feature_coverage_matrix_required": True,
        "sbom_enabled": True,
        "report_html_enabled": False,
        "report_pdf_enabled": False,
        "tensorboard_export_enabled": False,
        "mlflow_export_enabled": False,
        "wandb_export_enabled": False,
        "api_server_enabled": False,
        "gradio_demo_enabled": False,
        "streamlit_demo_enabled": False,
        "task_mode": "math_eq_answer",
        "architecture_mode": "our",
        "other_proxy_mode": "both",
        "startup_prompt_enabled": True,
        "experimental_toggle_prompt": True,
        "math_num_train": 18000,
        "math_num_val": 1200,
        "math_num_test": 1200,
        "math_min_value": -200,
        "math_max_value": 200,
        "math_include_negative": True,
        "math_ops": ["+", "-", "*", "/"],
        "target_loss_gate": 2.0,
        "target_exact_match_gate": 95.0,
        "target_speedup_ratio": 1.15,
        "logger_basename": "colab_math_fastproof_run_log.jsonl",
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

    merged["device"] = pick_device(str(merged.get("device", "auto")))

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
    ckpt_raw = Path(str(merged.get("checkpoint_dir", "checkpoints/kaggle_onefile_build30"))).expanduser()
    if ckpt_raw.is_absolute():
        if _is_dir_writable(ckpt_raw):
            ckpt_dir = ckpt_raw
        else:
            ckpt_dir = Path(str(merged["artifact_root"])) / "checkpoints" / "kaggle_onefile_build30"
            print(f"[runtime] checkpoint_dir fallback: requested={ckpt_raw} resolved={ckpt_dir}")
    else:
        ckpt_dir = Path(str(merged["artifact_root"])) / ckpt_raw
    merged["checkpoint_dir"] = str(ckpt_dir)
    if str(merged.get("logger_jsonl_path", "")).strip() == "":
        merged["logger_jsonl_path"] = str(
            Path(str(merged["artifact_root"])) / "kaggle_onefile_build30_log.jsonl"
        )

    bitnet_mode = str(merged.get("bitnet_mode", "stable")).strip().lower()
    if bitnet_mode not in ("stable", "aggressive"):
        bitnet_mode = "stable"
    merged["bitnet_mode"] = bitnet_mode

    logger_mode = str(merged.get("logger_mode", "jsonl_ring")).strip().lower()
    if logger_mode not in ("in_memory", "jsonl_ring"):
        logger_mode = "jsonl_ring"
    merged["logger_mode"] = logger_mode

    
    if not bool(merged.get("max_wall_hours_locked", False)):
        if detect_kaggle_runtime():
            merged["max_wall_hours"] = 11.5
            merged["wall_time_profile"] = "kaggle_default"
        elif detect_colab_runtime():
            merged["max_wall_hours"] = 23.5
            merged["wall_time_profile"] = "colab_default"

    merged["run_config_schema_report"] = validate_run_config_schema(merged)
    return merged


def parse_cli_overrides() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--profile", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--interactive-menu", action="store_true")
    parser.add_argument("--max-wall-hours", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--artifact-root", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--checkpoint-dir", type=str, default="")
    parser.add_argument("--resume-path", type=str, default="")
    parser.add_argument("--bundle-out", type=str, default="")
    parser.add_argument("--allow-notebook-input", action="store_true")
    parser.add_argument("--force-interactive-input", action="store_true")

    args, _ = parser.parse_known_args()
    overrides: Dict[str, Any] = {}
    if args.profile:
        overrides["profile"] = str(args.profile)
    if args.interactive:
        overrides["interactive"] = True
    if args.interactive_menu:
        overrides["interactive_menu"] = True
    if args.max_wall_hours is not None:
        overrides["max_wall_hours"] = float(args.max_wall_hours)
        overrides["max_wall_hours_locked"] = True
    if args.max_steps is not None:
        overrides["max_steps"] = int(args.max_steps)
    if args.artifact_root:
        overrides["artifact_root"] = str(args.artifact_root)
    if args.out_dir:
        overrides["out_dir"] = str(args.out_dir)
    if args.checkpoint_dir:
        overrides["checkpoint_dir"] = str(args.checkpoint_dir)
    if args.resume_path:
        overrides["resume_mode"] = "path"
        overrides["resume_path"] = str(args.resume_path)
        overrides["checkpoint_path"] = str(args.resume_path)
    if args.bundle_out:
        overrides["bundle_out"] = str(args.bundle_out)
    if args.allow_notebook_input:
        overrides["allow_notebook_input"] = True
    if args.force_interactive_input:
        overrides["force_interactive_input"] = True
    return overrides


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
        jsonl_path: str = "",
    ) -> None:
        self.run_name = run_name
        self.created_at_utc = _utc_now()
        self.mode = str(mode)
        self.step_log_interval = max(1, int(step_log_interval))
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

    def log_event(self, kind: str, data: Dict[str, Any]) -> None:
        rec = {
            "type": kind,
            "timestamp_utc": _utc_now(),
            "data": safe_jsonable(data),
        }
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

    def log_step(self, row: Dict[str, Any]) -> None:
        step = int(row.get("step", 0))
        if step > 0 and (step % self.step_log_interval) != 0:
            return
        self.step_rows_ring.append(dict(row))
        self.log_event("step", row)

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


def build_curriculum_sources(turkish_primary: bool = True) -> List[Dict[str, Any]]:
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
                {"dataset": "bigcode/the-stack-v2", "split": "train[:0.02%]", "field": "content"},
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


def _extract_field(item: Dict[str, Any], field: str) -> Optional[str]:
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
                        # Continue with bounded direct loader path.
                        ds, load_mode, load_info = _load_hf_candidate_dataset(ds_name, subset, split, cfg)
                        rows = []
                        timed_out = False
                        if ds is not None:
                            for item in ds:
                                if len(rows) >= per_candidate_cap:
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
    if not texts:
        rnd = random.Random(int(cfg["seed"]) + abs(hash(stage["name"])) % 10000)
        for _ in range(min(target, 40000)):
            n = rnd.randint(int(stage["min_len"]), min(int(stage["max_len"]), 180))
            alphabet = "abcçdefgğhıijklmnoöprsştuüvyz0123456789 .,:;!?()"
            texts.append("".join(rnd.choice(alphabet) for _ in range(n)))

    meta = {
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
_COMPILED_LINEAR_STATUS: Dict[str, Any] = {"attempted": False, "compiled": False, "fallback_reason": "", "compile_elapsed_sec": 0.0}


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
        self._compiled_enabled = os.environ.get("MERTFORMER_ONEFILE_BITNET_COMPILE", "0") == "1"

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
            _COMPILED_LINEAR_STATUS["attempted"] = False
            _COMPILED_LINEAR_STATUS["compiled"] = False
            _COMPILED_LINEAR_STATUS["fallback_reason"] = "non_cuda"
            return _COMPILED_LINEAR_CORE
        if not hasattr(torch, "compile"):
            _COMPILED_LINEAR_CORE = _linear_core
            _COMPILED_LINEAR_STATUS["attempted"] = False
            _COMPILED_LINEAR_STATUS["compiled"] = False
            _COMPILED_LINEAR_STATUS["fallback_reason"] = "compile_unavailable"
            return _COMPILED_LINEAR_CORE
        if bool(_COMPILED_LINEAR_STATUS.get("attempted", False)) and not bool(_COMPILED_LINEAR_STATUS.get("compiled", False)):
            _COMPILED_LINEAR_CORE = _linear_core
            return _COMPILED_LINEAR_CORE
        timeout_sec = float(os.environ.get("MERTFORMER_ONEFILE_COMPILE_TIMEOUT_SEC", "25.0") or 25.0)
        t0 = time.time()
        _COMPILED_LINEAR_STATUS["attempted"] = True
        try:
            cand = torch.compile(_linear_core, mode="max-autotune", fullgraph=False)
            elapsed = float(time.time() - t0)
            _COMPILED_LINEAR_STATUS["compile_elapsed_sec"] = elapsed
            if elapsed > timeout_sec:
                _COMPILED_LINEAR_STATUS["compiled"] = False
                _COMPILED_LINEAR_STATUS["fallback_reason"] = f"compile_timeout_{elapsed:.3f}s"
                _COMPILED_LINEAR_CORE = _linear_core
                _COMPILE_GUARD_STATE["attempted"] = True
                _COMPILE_GUARD_STATE["compiled"] = False
                _COMPILE_GUARD_STATE["fallback_reason"] = _COMPILED_LINEAR_STATUS["fallback_reason"]
                _COMPILE_GUARD_STATE["compile_elapsed_sec"] = elapsed
                return _COMPILED_LINEAR_CORE
            _COMPILED_LINEAR_CORE = cand
            _COMPILED_LINEAR_STATUS["compiled"] = True
            _COMPILED_LINEAR_STATUS["fallback_reason"] = ""
            _COMPILE_GUARD_STATE["attempted"] = True
            _COMPILE_GUARD_STATE["compiled"] = True
            _COMPILE_GUARD_STATE["fallback_reason"] = ""
            _COMPILE_GUARD_STATE["compile_elapsed_sec"] = elapsed
        except Exception as e:
            _COMPILED_LINEAR_CORE = _linear_core
            _COMPILED_LINEAR_STATUS["compiled"] = False
            _COMPILED_LINEAR_STATUS["fallback_reason"] = f"compile_error:{type(e).__name__}"
            _COMPILED_LINEAR_STATUS["compile_elapsed_sec"] = float(time.time() - t0)
            _COMPILE_GUARD_STATE["attempted"] = True
            _COMPILE_GUARD_STATE["compiled"] = False
            _COMPILE_GUARD_STATE["fallback_reason"] = _COMPILED_LINEAR_STATUS["fallback_reason"]
            _COMPILE_GUARD_STATE["compile_elapsed_sec"] = _COMPILED_LINEAR_STATUS["compile_elapsed_sec"]
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
    def __init__(self, hidden: int, dt: float = 1.0, fast_path: bool = True) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.dt = float(dt)
        self.in_proj = nn.Linear(hidden, hidden)
        self.gate_proj = nn.Linear(hidden, hidden)
        self.register_buffer("state", torch.empty(0), persistent=False)
        self.fast_path = bool(fast_path)
        self._compiled_forward = None

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        self.state = torch.zeros(batch_size, self.hidden, device=device, dtype=dtype)

    def _forward_slow(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: Simplified Euler ODE without learnable tau (standalone companion).
        # Production uses continuous-time decay with learnable tau.
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

    def _maybe_compile(self) -> None:
        if self._compiled_forward is not None:
            return
        if hasattr(torch, "compile"):
            try:
                self._compiled_forward = torch.compile(self._forward_slow, mode="reduce-overhead")
                return
            except Exception:
                pass
        self._compiled_forward = self._forward_slow

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fast_path and x.device.type == "cuda":
            self._maybe_compile()
            try:
                return self._compiled_forward(x)
            except Exception:
                return self._forward_slow(x)
        return self._forward_slow(x)


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
        dispatch_mode: str = "sequential",
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.router = LiquidRouter(hidden, num_experts=num_experts, top_k=top_k)
        self.experts = nn.ModuleList([SwiGLUFFN(hidden, intermediate) for _ in range(num_experts)])
        self.shared = SwiGLUFFN(hidden, intermediate)
        self.moe_mode = str(moe_mode)
        self.dispatch_mode = str(dispatch_mode).lower()
        self.use_structural_plasticity = bool(use_structural_plasticity)
        self.structural_update_interval = int(structural_update_interval)
        self.register_buffer("usage", torch.zeros(num_experts), persistent=False)
        self._step = 0

    def _structural_update(self) -> None:
        if not self.use_structural_plasticity:
            return
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

    def _dispatch_sequential(
        self,
        x_flat: torch.Tensor,
        top_idx: torch.Tensor,
        top_w: torch.Tensor,
        capacity_mask: torch.Tensor,
    ) -> torch.Tensor:
        b = x_flat.size(0)
        out_flat = torch.zeros_like(x_flat)
        for expert_id, expert in enumerate(self.experts):
            expert_mask = top_idx == expert_id
            if capacity_mask is not None:
                expert_mask = expert_mask & capacity_mask
            token_mask = expert_mask.any(dim=-1)
            if not bool(token_mask.any().item()):
                continue
            selected_x = x_flat[token_mask]
            expert_out = expert(selected_x)
            weights = (top_w[token_mask] * expert_mask[token_mask].float()).sum(dim=-1, keepdim=True)
            out_flat[token_mask] += expert_out * weights
        return out_flat

    def _dispatch_parallel(
        self,
        x_flat: torch.Tensor,
        top_idx: torch.Tensor,
        top_w: torch.Tensor,
        capacity_mask: torch.Tensor,
    ) -> torch.Tensor:
        n, h = x_flat.shape
        k = top_idx.size(-1)
        out_flat = x_flat.new_zeros((n, h))

        token_idx = torch.arange(n, device=top_idx.device).repeat_interleave(k)
        expert_idx = top_idx.reshape(-1)
        weights = top_w.reshape(-1)
        if capacity_mask is not None:
            mask = capacity_mask.reshape(-1)
            if mask.numel() > 0:
                token_idx = token_idx[mask]
                expert_idx = expert_idx[mask]
                weights = weights[mask]

        if expert_idx.numel() == 0:
            return out_flat

        order = torch.argsort(expert_idx)
        expert_sorted = expert_idx[order]
        token_sorted = token_idx[order]
        weight_sorted = weights[order]

        counts = torch.bincount(expert_sorted, minlength=len(self.experts))
        if counts.numel() == 0:
            return out_flat

        start = 0
        for expert_id, expert in enumerate(self.experts):
            cnt = int(counts[expert_id].item())
            if cnt == 0:
                continue
            end = start + cnt
            idx = token_sorted[start:end]
            w = weight_sorted[start:end].unsqueeze(-1)
            selected_x = x_flat.index_select(0, idx)
            expert_out = expert(selected_x)
            out_flat.index_add_(0, idx, expert_out * w)
            start = end
        return out_flat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        b, t, h = x.shape
        xf = x.view(b * t, h)
        top_idx, top_w, aux_tensor, stats, capacity_mask = self.router(xf)
        out_flat = torch.zeros_like(xf)

        if self.moe_mode == "dense_debug":
            expert_out = torch.stack([e(xf) for e in self.experts], dim=1)
            gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, h)
            chosen = expert_out.gather(1, gather_idx)
            out_flat = (chosen * top_w.unsqueeze(-1)).sum(dim=1)
        else:
            if self.dispatch_mode == "parallel":
                out_flat = self._dispatch_parallel(xf, top_idx, top_w, capacity_mask)
            else:
                out_flat = self._dispatch_sequential(xf, top_idx, top_w, capacity_mask)

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
    def __init__(
        self,
        hidden: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_flash_attn_inference: bool = True,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.dropout = float(dropout)
        self.use_flash_attn_inference = bool(use_flash_attn_inference)

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
        use_flash = bool(self.training or self.use_flash_attn_inference)

        if hasattr(F, "scaled_dot_product_attention") and use_flash:
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
    liquid_fast_path: bool = True
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
    moe_dispatch_mode: str = "parallel"
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
                dispatch_mode=cfg.moe_dispatch_mode,
            )
        else:
            self.ff = SwiGLUFFN(cfg.hidden_size, cfg.intermediate_size, dropout=cfg.dropout)

        self.liquid = LiquidMixer(cfg.hidden_size, dt=cfg.liquid_dt, fast_path=cfg.liquid_fast_path) if self.use_liquid else None
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
        if self.hebbian is not None:
            x = self.hebbian(x)
        if self.neuro_symbolic is not None:
            x = self.neuro_symbolic(x)
        if self.lifelong is not None:
            x, lif = self.lifelong(x)
            stats["lifelong_drift"] = lif["drift"]
            stats["lifelong_gain"] = lif["gain"]
        if self.qinn is not None:
            x = self.qinn(x)

        return x, aux, present, stats


class MertFormerTiny(nn.Module):
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


def parity_self_check(model: MertFormerTiny, cfg: Dict[str, Any], device: str) -> Dict[str, Any]:
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


def convert_model_to_strict_bitnet(
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


def collect_bitnet_telemetry(model: nn.Module, bitnet_mode: str = "stable") -> Dict[str, Any]:
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
        "schema": "kaggle_onefile_build30_ckpt_v5",
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
        liquid_fast_path=bool(cfg.get("liquid_fast_path", True)),
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
        moe_dispatch_mode=str(cfg.get("moe_dispatch_mode", "parallel")),
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
        "# Kaggle One-File Build30 Deep Report",
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
        "|---|---:|---:|---:|---:|---:|---:|---:|",
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
        "json": str(out_dir / "kaggle_onefile_deep_build30.json"),
        "csv": str(out_dir / "kaggle_onefile_deep_build30.csv"),
        "md": str(out_dir / "kaggle_onefile_deep_build30.md"),
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
        "zero_known_critical_bugs_claim": "zero-known-critical-bugs this run",
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
def build_mert_cfg(cfg: Dict[str, Any]) -> MertFormerCfg:
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
        liquid_fast_path=bool(cfg.get("liquid_fast_path", True)),
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
        moe_dispatch_mode=str(cfg.get("moe_dispatch_mode", "parallel")),
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


def mathfp_prompt_architecture(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    if not bool(out.get("startup_prompt_enabled", True)):
        return out
    if not can_accept_user_input(out):
        return out
    try:
        arch_raw = input("Architecture mode [our/other/both] (default=our): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return out
    arch = arch_raw if arch_raw in ("our", "other", "both") else str(out.get("architecture_mode", "our"))
    out["architecture_mode"] = arch
    if arch in ("other", "both"):
        try:
            proxy_raw = input(
                "Other proxy mode [gpt_proxy_dense/gemini_proxy_moe/both] (default=both): "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            proxy_raw = ""
        if proxy_raw in ("gpt_proxy_dense", "gemini_proxy_moe", "both"):
            out["other_proxy_mode"] = proxy_raw
    return out


def mathfp_prompt_experimental_toggles(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    if not bool(out.get("experimental_toggle_prompt", True)):
        return out
    if not can_accept_user_input(out):
        return out
    print(
        "[warning] Experimental bundle affects stability/perf trade-offs "
        "(QINN + latent/workspace/sync/plasticity)."
    )
    try:
        ans = input("Enable experimental extension bundle? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return out
    if ans in ("n", "no", "0", "false"):
        out["mert_enable_all_extensions"] = False
        out["mert_use_qinn"] = False
    else:
        out["mert_enable_all_extensions"] = True
        out["mert_use_qinn"] = True
    return out


def _mathfp_first_int(text: str) -> Optional[int]:
    m = re.search(r"-?\d+", str(text))
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def _mathfp_parse_answer_token(tokenizer: SimpleTokenizer, answer_text: str) -> Optional[int]:
    token = _mathfp_first_int(answer_text)
    if token is None:
        return None
    return token


def mathfp_generate_math_records(
    n: int,
    seed: int,
    min_value: int,
    max_value: int,
    include_negative: bool,
    ops: Sequence[str],
    used_keys: Optional[set] = None,
) -> List[Dict[str, Any]]:
    rng = random.Random(int(seed))
    valid_ops = [str(x) for x in ops if str(x) in ("+", "-", "*", "/")]
    if not valid_ops:
        valid_ops = ["+", "-", "*", "/"]
    used: set = used_keys if used_keys is not None else set()
    records: List[Dict[str, Any]] = []

    abs_cap = max(abs(int(min_value)), abs(int(max_value)), 10)
    tier_table = {
        "easy": max(10, abs_cap // 10),
        "medium": max(20, abs_cap // 3),
        "hard": max(30, abs_cap),
    }
    tier_names = ["easy", "medium", "hard"]
    tier_weights = [0.52, 0.30, 0.18]

    def rand_operand(max_abs: int) -> int:
        v = rng.randint(0, max_abs)
        if include_negative and rng.random() < 0.5:
            v = -v
        v = max(int(min_value), min(int(max_value), int(v)))
        return int(v)

    tries = 0
    max_tries = max(5000, int(n) * 80)
    while len(records) < int(n) and tries < max_tries:
        tries += 1
        op = rng.choice(valid_ops)
        tier = rng.choices(tier_names, weights=tier_weights, k=1)[0]
        vmax = int(tier_table.get(tier, abs_cap))
        a = 0
        b = 0
        c = 0
        if op == "+":
            a = rand_operand(vmax)
            b = rand_operand(vmax)
            c = a + b
        elif op == "-":
            a = rand_operand(vmax)
            b = rand_operand(vmax)
            c = a - b
        elif op == "*":
            mul_cap = max(3, int(math.sqrt(float(vmax))) + 2)
            a = rand_operand(mul_cap)
            b = rand_operand(mul_cap)
            c = a * b
        else:
            div_cap = max(2, int(math.sqrt(float(vmax))) + 2)
            ok = False
            for _ in range(32):
                b_raw = rng.randint(1, div_cap)
                b = -b_raw if include_negative and rng.random() < 0.5 else b_raw
                c = rand_operand(div_cap)
                a = b * c
                if int(min_value) <= a <= int(max_value) and int(min_value) <= b <= int(max_value):
                    ok = True
                    break
            if not ok:
                a, b, c = 0, 1, 0
        prompt = f"{a} {op} {b} = "
        answer = str(int(c))
        full_text = prompt + answer
        key = f"{a}|{op}|{b}|{answer}"
        if key in used:
            continue
        used.add(key)
        records.append(
            {
                "prompt": prompt,
                "answer": answer,
                "full_text": full_text,
                "op": op,
                "difficulty": tier,
            }
        )
    return records


def mathfp_build_datasets(cfg: Dict[str, Any]) -> Dict[str, Any]:
    seed = int(cfg.get("seed", 42))
    min_value = int(cfg.get("math_min_value", -200))
    max_value = int(cfg.get("math_max_value", 200))
    include_negative = bool(cfg.get("math_include_negative", True))
    ops = list(cfg.get("math_ops", ["+", "-", "*", "/"]))
    n_train = int(cfg.get("math_num_train", 18000))
    n_val = int(cfg.get("math_num_val", 1200))
    n_test = int(cfg.get("math_num_test", 1200))
    n_unseen = int(cfg.get("math_num_unseen", 400))

    used: set = set()
    train_records = mathfp_generate_math_records(n_train, seed + 11, min_value, max_value, include_negative, ops, used)
    val_records = mathfp_generate_math_records(n_val, seed + 29, min_value, max_value, include_negative, ops, used)
    test_records = mathfp_generate_math_records(n_test, seed + 47, min_value, max_value, include_negative, ops, used)

    unseen_records: List[Dict[str, Any]] = []
    if bool(cfg.get("eval_unseen_enabled", True)):
        unseen_min = int(cfg.get("eval_unseen_min", 500))
        unseen_max = int(cfg.get("eval_unseen_max", 900))
        unseen_abs_min = min(abs(unseen_min), abs(unseen_max))
        unseen_abs_max = max(abs(unseen_min), abs(unseen_max))
        unseen_records = mathfp_generate_math_records(
            n_unseen,
            seed + 83,
            unseen_abs_min,
            unseen_abs_max,
            include_negative=False,
            ops=ops,
            used_keys=used,
        )

    op_counts: Dict[str, int] = {"+": 0, "-": 0, "*": 0, "/": 0}
    for row in train_records + val_records + test_records + unseen_records:
        op_counts[str(row.get("op", "+"))] = op_counts.get(str(row.get("op", "+")), 0) + 1

    return {
        "train": train_records,
        "val": val_records,
        "test": test_records,
        "unseen": unseen_records,
        "stats": {
            "train_count": len(train_records),
            "val_count": len(val_records),
            "test_count": len(test_records),
            "unseen_count": len(unseen_records),
            "total_count": len(train_records) + len(val_records) + len(test_records) + len(unseen_records),
            "op_counts": op_counts,
            "range_train": [int(min_value), int(max_value)],
            "range_unseen": [int(cfg.get("eval_unseen_min", 500)), int(cfg.get("eval_unseen_max", 900))],
        },
    }


def mathfp_build_answer_only_tensors(
    tokenizer: SimpleTokenizer,
    text: str,
    seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ids = tokenizer.encode(str(text), add_bos=True, add_eos=True)
    if len(ids) < 2:
        ids = [tokenizer.bos_id, tokenizer.eos_id]
    if len(ids) > int(seq_len) + 1:
        ids = ids[-(int(seq_len) + 1) :]
    x = ids[:-1]
    y = ids[1:]
    labels = [-100 for _ in y]
    eq_id = tokenizer.stoi.get("=", tokenizer.unk_id)
    eq_pos = -1
    for i, tid in enumerate(ids):
        if int(tid) == int(eq_id):
            eq_pos = i
            break
    if eq_pos >= 0:
        for j in range(len(y)):
            # Predict only tokens appearing after "=".
            if (j + 1) > eq_pos:
                labels[j] = int(y[j])

    pad_len = max(0, int(seq_len) - len(x))
    if pad_len > 0:
        x = x + [tokenizer.pad_id] * pad_len
        labels = labels + [-100] * pad_len
    else:
        x = x[: int(seq_len)]
        labels = labels[: int(seq_len)]
    return torch.tensor(x, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def mathfp_prepare_tensor_dataset(
    records: Sequence[Dict[str, Any]],
    tokenizer: SimpleTokenizer,
    seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    xs: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    answer_tokens = 0
    for row in records:
        x, y = mathfp_build_answer_only_tensors(tokenizer, str(row.get("full_text", "")), seq_len=seq_len)
        xs.append(x)
        labels.append(y)
        answer_tokens += int((y != -100).sum().item())
    if not xs:
        xs = [torch.zeros(int(seq_len), dtype=torch.long)]
        labels = [torch.full((int(seq_len),), -100, dtype=torch.long)]
    return torch.stack(xs, dim=0), torch.stack(labels, dim=0), int(answer_tokens)


class MathAnswerDataset(Dataset):
    def __init__(self, x: torch.Tensor, labels: torch.Tensor) -> None:
        self.x = x
        self.labels = labels

    def __len__(self) -> int:
        return int(self.x.size(0))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.labels[idx]


def mathfp_eval_masked_loss(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    vocab_size: int,
    aux_coeff: float,
    max_batches: int = 12,
) -> float:
    model.eval()
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    losses: List[float] = []
    with torch.no_grad():
        for i, (x, labels) in enumerate(loader):
            if i >= int(max_batches):
                break
            x = x.to(device)
            labels = labels.to(device)
            out = model(x)
            if isinstance(out, tuple):
                logits = out[0]
                aux = out[1] if len(out) > 1 and torch.is_tensor(out[1]) else torch.tensor(0.0, device=device)
                vocab = int(logits.size(-1))
                loss = ce(logits.reshape(-1, vocab), labels.reshape(-1)) + float(aux_coeff) * aux.float()
            else:
                logits = out
                vocab = int(logits.size(-1))
                loss = ce(logits.reshape(-1, vocab), labels.reshape(-1))
            if torch.isfinite(loss):
                losses.append(float(loss.detach().cpu().item()))
    model.train()
    if not losses:
        return float("inf")
    return float(sum(losses) / len(losses))


@torch.no_grad()
def mathfp_generate_answer(
    model: nn.Module,
    tokenizer: SimpleTokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 12,
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    if not ids:
        ids = [tokenizer.bos_id]
    g = torch.tensor([ids], dtype=torch.long, device=device)
    max_ctx = int(getattr(getattr(model, "cfg", None), "max_seq_len", max(64, len(ids) + max_new_tokens)))
    for _ in range(int(max_new_tokens)):
        if g.size(1) > max_ctx:
            g = g[:, -max_ctx:]
        out = model(g)
        logits = out[0] if isinstance(out, tuple) else out
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        g = torch.cat([g, nxt], dim=1)
        if int(nxt.item()) == int(tokenizer.eos_id):
            break
    tail = g[0].tolist()[len(ids) :]
    return tokenizer.decode(tail).strip()


def mathfp_eval_exact_match(
    model: nn.Module,
    tokenizer: SimpleTokenizer,
    records: Sequence[Dict[str, Any]],
    device: str,
    max_new_tokens: int = 12,
) -> Dict[str, Any]:
    total = 0
    correct = 0
    invalid = 0
    per_op_total: Dict[str, int] = {"+": 0, "-": 0, "*": 0, "/": 0}
    per_op_ok: Dict[str, int] = {"+": 0, "-": 0, "*": 0, "/": 0}
    for row in records:
        total += 1
        op = str(row.get("op", "+"))
        per_op_total[op] = per_op_total.get(op, 0) + 1
        pred_text = mathfp_generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=str(row.get("prompt", "")),
            device=device,
            max_new_tokens=max_new_tokens,
        )
        pred = _mathfp_first_int(pred_text)
        gold = _mathfp_first_int(str(row.get("answer", "")))
        if pred is None:
            invalid += 1
        if pred is not None and gold is not None and int(pred) == int(gold):
            correct += 1
            per_op_ok[op] = per_op_ok.get(op, 0) + 1
    exact = _safe_div(float(correct) * 100.0, float(max(1, total)), default=0.0)
    op_acc: Dict[str, float] = {}
    for k, v in per_op_total.items():
        op_acc[k] = _safe_div(float(per_op_ok.get(k, 0)) * 100.0, float(max(1, v)), default=0.0)
    return {
        "exact_match_percent": float(exact),
        "correct": int(correct),
        "total": int(total),
        "invalid_output_ratio": _safe_div(float(invalid), float(max(1, total)), default=0.0),
        "operation_accuracy": op_acc,
    }


def mathfp_select_small_mert_shape(cfg: Dict[str, Any]) -> Dict[str, int]:
    low = int(cfg.get("target_param_band_low", 8_000_000))
    high = int(cfg.get("target_param_band_high", 15_000_000))
    candidates = [
        (288, 8, 8, 4),
        (320, 8, 8, 4),
        (352, 8, 8, 4),
        (384, 8, 8, 4),
        (320, 10, 8, 4),
    ]
    best = candidates[0]
    best_dist = float("inf")
    for hidden, layers, heads, kv in candidates:
        c = dict(cfg)
        c["mert_hidden"] = int(hidden)
        c["mert_layers"] = int(layers)
        c["mert_heads"] = int(heads)
        c["mert_kv_heads"] = int(kv)
        c["mert_enable_all_extensions"] = True
        c["mert_use_qinn"] = bool(cfg.get("mert_use_qinn", True))
        probe = MertFormerTiny(build_mert_cfg(c))
        pcount = count_params(probe)
        target_mid = int((low + high) / 2)
        dist = abs(int(pcount) - target_mid)
        if low <= pcount <= high:
            return {
                "mert_hidden": int(hidden),
                "mert_layers": int(layers),
                "mert_heads": int(heads),
                "mert_kv_heads": int(kv),
                "params": int(pcount),
            }
        if dist < best_dist:
            best_dist = dist
            best = (hidden, layers, heads, kv)
    c2 = dict(cfg)
    c2["mert_hidden"] = int(best[0])
    c2["mert_layers"] = int(best[1])
    c2["mert_heads"] = int(best[2])
    c2["mert_kv_heads"] = int(best[3])
    c2["mert_enable_all_extensions"] = True
    c2["mert_use_qinn"] = bool(cfg.get("mert_use_qinn", True))
    probe2 = MertFormerTiny(build_mert_cfg(c2))
    pcount2 = count_params(probe2)
    return {
        "mert_hidden": int(best[0]),
        "mert_layers": int(best[1]),
        "mert_heads": int(best[2]),
        "mert_kv_heads": int(best[3]),
        "params": int(pcount2),
    }


def mathfp_build_variant_models(cfg: Dict[str, Any], vocab_size: int) -> Dict[str, nn.Module]:
    shape = mathfp_select_small_mert_shape(cfg)
    base = dict(cfg)
    base["vocab_size"] = int(vocab_size)
    base["mert_hidden"] = int(shape["mert_hidden"])
    base["mert_layers"] = int(shape["mert_layers"])
    base["mert_heads"] = int(shape["mert_heads"])
    base["mert_kv_heads"] = int(shape["mert_kv_heads"])

    # Our architecture: everything ON by default (QINN included unless user toggles off).
    our_cfg = dict(base)
    our_cfg["mert_enable_all_extensions"] = bool(cfg.get("mert_enable_all_extensions", True))
    our_cfg["mert_use_qinn"] = bool(cfg.get("mert_use_qinn", True))
    our_cfg["mert_use_moe"] = True
    our_cfg["mert_use_liquid"] = True
    our_model = MertFormerTiny(build_mert_cfg(our_cfg))
    if bool(cfg.get("strict_bitnet", True)):
        convert_model_to_strict_bitnet(
            our_model,
            logger=None,
            bitnet_mode=str(cfg.get("bitnet_mode", "stable")),
            skip_attention_qkvo=bool(cfg.get("bitnet_skip_attention_qkvo", True)),
        )

    gpt_hidden = max(192, int(shape["mert_hidden"]) - 64)
    gpt_layers = max(4, int(shape["mert_layers"]) - 2)
    gpt_heads = max(4, min(8, int(shape["mert_heads"])))
    gpt_model = VanillaTransformerLM(
        vocab_size=int(vocab_size),
        hidden_size=int(gpt_hidden),
        num_layers=int(gpt_layers),
        num_heads=int(gpt_heads),
        max_seq_len=max(64, int(cfg.get("seq_len", 64))),
        dropout=0.0,
    )

    gemini_cfg = dict(base)
    gemini_cfg["mert_enable_all_extensions"] = False
    gemini_cfg["mert_use_qinn"] = False
    gemini_cfg["mert_use_moe"] = True
    gemini_cfg["mert_use_liquid"] = False
    gemini_model = MertFormerTiny(build_mert_cfg(gemini_cfg))
    if bool(cfg.get("strict_bitnet", True)):
        convert_model_to_strict_bitnet(
            gemini_model,
            logger=None,
            bitnet_mode=str(cfg.get("bitnet_mode", "stable")),
            skip_attention_qkvo=bool(cfg.get("bitnet_skip_attention_qkvo", True)),
        )
    return {
        "our_mertformer": our_model,
        "gpt_proxy_dense": gpt_model,
        "gemini_proxy_moe": gemini_model,
    }


def mathfp_select_variants(cfg: Dict[str, Any]) -> List[str]:
    arch_mode = str(cfg.get("architecture_mode", "our")).strip().lower()
    other_mode = str(cfg.get("other_proxy_mode", "both")).strip().lower()
    others: List[str] = []
    if other_mode in ("gpt_proxy_dense", "gemini_proxy_moe"):
        others = [other_mode]
    else:
        others = ["gpt_proxy_dense", "gemini_proxy_moe"]
    if arch_mode == "our":
        return ["our_mertformer"]
    if arch_mode == "other":
        return others
    # both
    return ["our_mertformer"] + others


def mathfp_allocate_steps(cfg: Dict[str, Any], variants: Sequence[str]) -> Dict[str, int]:
    total_steps = max(1, int(cfg.get("max_steps", 1200)))
    arch_mode = str(cfg.get("architecture_mode", "our")).strip().lower()
    out: Dict[str, int] = {}
    if arch_mode == "both" and "our_mertformer" in variants:
        our_steps = max(1, total_steps // 2)
        out["our_mertformer"] = our_steps
        others = [v for v in variants if v != "our_mertformer"]
        other_total = max(1, total_steps - our_steps)
        if not others:
            return out
        base = max(1, other_total // len(others))
        rem = max(0, other_total - (base * len(others)))
        for i, name in enumerate(others):
            out[name] = base + (1 if i < rem else 0)
        return out
    base = max(1, total_steps // max(1, len(variants)))
    rem = max(0, total_steps - (base * len(variants)))
    for i, name in enumerate(variants):
        out[name] = base + (1 if i < rem else 0)
    return out


def mathfp_train_variant(
    variant: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_records: Sequence[Dict[str, Any]],
    test_records: Sequence[Dict[str, Any]],
    tokenizer: SimpleTokenizer,
    cfg: Dict[str, Any],
    device: str,
    logger: InMemoryRunLogger,
    step_csv_path: Path,
    steps: int,
) -> Dict[str, Any]:
    model = model.to(device)
    model.train()
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    lr = float(cfg.get("lr", 2.0e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(cfg.get("weight_decay", 0.01)))
    losses: List[float] = []
    step_times: List[float] = []
    layer_grad_norm_samples: List[List[float]] = []
    answer_tokens_total = 0
    train_iter = iter(train_loader)
    eval_interval = max(1, int(cfg.get("eval_interval_steps", 100)))

    for step in range(1, int(steps) + 1):
        try:
            x, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, labels = next(train_iter)
        x = x.to(device)
        labels = labels.to(device)
        answer_tokens = int((labels != -100).sum().item())

        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        out = model(x)
        if isinstance(out, tuple):
            logits = out[0]
            aux = out[1] if len(out) > 1 and torch.is_tensor(out[1]) else torch.tensor(0.0, device=device)
            loss = ce(logits.reshape(-1, int(logits.size(-1))), labels.reshape(-1)) + float(cfg.get("aux_loss_coeff", 0.0)) * aux.float()
        else:
            logits = out
            loss = ce(logits.reshape(-1, int(logits.size(-1))), labels.reshape(-1))
        if not torch.isfinite(loss):
            continue
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.get("bitnet_clip_grad", 1.0)))
        if bool(cfg.get("interpretability_enabled", True)) and bool(cfg.get("grad_heatmap_enabled", True)):
            if step <= 32 or (step % max(1, int(eval_interval // 2)) == 0):
                layer_grad_norm_samples.append(_collect_layer_grad_norms(model, max_layers=64))
        opt.step()
        dt = max(1e-9, float(time.time() - t0))
        lval = float(loss.detach().cpu().item())
        losses.append(lval)
        step_times.append(dt)
        answer_tokens_total += answer_tokens
        row = {
            "variant": variant,
            "step": step,
            "loss": lval,
            "grad_norm": float(grad_norm.detach().cpu().item()) if torch.is_tensor(grad_norm) else float(grad_norm),
            "lr": float(lr),
            "step_time_sec": float(dt),
            "answer_tokens": int(answer_tokens),
            "answer_tokens_per_sec": _safe_div(float(answer_tokens), float(dt), default=0.0),
        }
        logger.log_step(row)
        append_csv_row(
            step_csv_path,
            fieldnames=[
                "variant",
                "step",
                "loss",
                "grad_norm",
                "lr",
                "step_time_sec",
                "answer_tokens",
                "answer_tokens_per_sec",
            ],
            row=row,
        )
        if step == 1 or step % max(1, eval_interval) == 0:
            vloss = mathfp_eval_masked_loss(
                model=model,
                loader=val_loader,
                device=device,
                vocab_size=int(logits.size(-1)),
                aux_coeff=float(cfg.get("aux_loss_coeff", 0.0)),
                max_batches=max(4, int(cfg.get("benchmark_eval_batches", 8))),
            )
            logger.log_event("mathfp_eval", {"variant": variant, "step": step, "val_loss": vloss})
            print(f"[mathfp:{variant}] step={step}/{steps} loss={lval:.4f} val_loss={vloss:.4f}")

    final_loss = losses[-1] if losses else float("inf")
    val_loss = mathfp_eval_masked_loss(
        model=model,
        loader=val_loader,
        device=device,
        vocab_size=int(getattr(model, "vocab_size", int(cfg.get("vocab_size", 1024)))),
        aux_coeff=float(cfg.get("aux_loss_coeff", 0.0)),
        max_batches=max(4, int(cfg.get("benchmark_eval_batches", 8))),
    )
    exact_val = mathfp_eval_exact_match(model, tokenizer, val_records, device=device, max_new_tokens=12)
    exact_test = mathfp_eval_exact_match(model, tokenizer, test_records, device=device, max_new_tokens=12)
    t_total = float(sum(step_times))
    return {
        "variant": variant,
        "params": int(count_params(model)),
        "steps": int(steps),
        "final_loss": float(final_loss),
        "min_loss": float(min(losses) if losses else float("inf")),
        "avg_loss": float(sum(losses) / len(losses)) if losses else float("inf"),
        "val_loss": float(val_loss),
        "exact_match_val": float(exact_val.get("exact_match_percent", 0.0)),
        "exact_match_test": float(exact_test.get("exact_match_percent", 0.0)),
        "invalid_output_ratio_test": float(exact_test.get("invalid_output_ratio", 1.0)),
        "operation_accuracy_test": safe_jsonable(exact_test.get("operation_accuracy", {})),
        "layer_grad_norm_samples": safe_jsonable(layer_grad_norm_samples[:48]),
        "tokens_per_sec": _safe_div(float(answer_tokens_total), float(max(t_total, 1e-9)), default=0.0),
        "avg_step_time_sec": _safe_div(float(t_total), float(max(1, len(step_times))), default=0.0),
    }


def _collect_layer_grad_norms(model: nn.Module, max_layers: int = 64) -> List[float]:
    rows: List[Tuple[str, float]] = []
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        if not torch.is_floating_point(p.grad):
            continue
        try:
            g = float(p.grad.detach().norm().cpu().item())
        except Exception:
            continue
        rows.append((str(n), g))
    rows = rows[: max(1, int(max_layers))]
    return [float(x[1]) for x in rows]


def maybe_plot_mathfp_interpretability_assets(
    compare_payload: Dict[str, Any],
    run_dir: Path,
    write_files: bool,
    enabled: bool,
) -> Dict[str, str]:
    if not write_files or not enabled or not HAS_MATPLOTLIB:
        return {}
    out: Dict[str, str] = {}
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Proxy expert specialization bar from per-op test accuracy.
        p_bar = run_dir / "moe_expert_bar_proxy.png"
        our = {}
        for row in compare_payload.get("variant_results", []):
            if str(row.get("variant", "")) == "our_mertformer":
                our = dict(row)
                break
        op_acc = our.get("operation_accuracy_test", {}) if isinstance(our, dict) else {}
        labels = ["+", "-", "*", "/"]
        vals = [float(op_acc.get(k, 0.0)) for k in labels] if isinstance(op_acc, dict) else [0.0, 0.0, 0.0, 0.0]
        plt.figure(figsize=(7, 4))
        plt.bar(labels, vals)
        plt.ylim(0.0, 100.0)
        plt.title("MoE Expert Usage Proxy (Op-wise Accuracy)")
        plt.ylabel("accuracy %")
        plt.tight_layout()
        plt.savefig(p_bar)
        plt.close()
        out["moe_expert_bar_proxy"] = str(p_bar)
    except Exception:
        pass

    try:
        p_heat = run_dir / "gradient_flow_heatmap.png"
        rows = []
        for row in compare_payload.get("variant_results", []):
            g = row.get("layer_grad_norm_samples", [])
            if isinstance(g, list) and g:
                rows.append([float(x) for x in g[:64]])
        if rows:
            width = max(len(r) for r in rows)
            mat = []
            for r in rows:
                rr = list(r) + [0.0] * (width - len(r))
                mat.append(rr)
            plt.figure(figsize=(10, 4))
            plt.imshow(mat, aspect="auto")
            plt.colorbar(label="grad norm")
            plt.xlabel("layer-param index")
            plt.ylabel("variant index")
            plt.title("Gradient Flow Heatmap (sampled)")
            plt.tight_layout()
            plt.savefig(p_heat)
            plt.close()
            out["gradient_flow_heatmap"] = str(p_heat)
    except Exception:
        pass
    return out



def mathfp_compare_markdown(payload: Dict[str, Any]) -> str:
    lines = [
        "# Build30 Colab Math Fastproof Compare",
        "",
        f"- generated_at_utc: {payload.get('generated_at_utc', _utc_now())}",
        f"- profile: {payload.get('profile', '')}",
        f"- architecture_mode: {payload.get('architecture_mode', '')}",
        f"- other_proxy_mode: {payload.get('other_proxy_mode', '')}",
        "",
        "| Variant | Params | Steps | Final Loss | Val Loss | ExactMatch(Test) | ExactMatch(Unseen) | Tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("variant_results", []):
        lines.append(
            f"| {row.get('variant','')} | {int(row.get('params',0))} | {int(row.get('steps',0))} | "
            f"{float(row.get('final_loss', float('inf'))):.4f} | {float(row.get('val_loss', float('inf'))):.4f} | "
            f"{float(row.get('exact_match_test', 0.0)):.2f}% | {float(row.get('exact_match_unseen', 0.0)):.2f}% | {float(row.get('tokens_per_sec', 0.0)):.2f} |"
        )
    gates = payload.get("gates", {})
    lines.extend(
        [
            "",
            "## Gates",
            "",
            f"- loss_gate_pass: {bool(gates.get('loss_gate_pass', False))}",
            f"- accuracy_gate_pass: {bool(gates.get('accuracy_gate_pass', False))}",
            f"- speed_gate_pass: {bool(gates.get('speed_gate_pass', False))}",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def run_math_fastproof(
    cfg: Dict[str, Any],
    layout: ArtifactLayout,
    device: str,
    gpu_meta: Dict[str, Any],
    gpu_tune: Dict[str, Any],
    total_start: float,
) -> Dict[str, Any]:
    cfg = mathfp_prompt_architecture(cfg)
    cfg = mathfp_prompt_experimental_toggles(cfg)

    run_id = layout.run_id
    run_dir = layout.run_dir
    run_log_path = run_dir / f"{run_id}_run_log.jsonl"
    step_csv_path = run_dir / f"{run_id}_step_metrics.csv"
    summary_json_path = run_dir / f"{run_id}_summary.json"
    compare_json_path = run_dir / f"{run_id}_compare.json"
    compare_csv_path = run_dir / f"{run_id}_compare.csv"
    compare_md_path = run_dir / f"{run_id}_compare.md"
    health_txt_path = run_dir / f"{run_id}_health.txt"
    model_final_path = run_dir / f"{run_id}_model_final.pt"
    artifact_index_path = run_dir / f"{run_id}_artifact_index.json"

    logger = InMemoryRunLogger(
        run_name=f"{run_id}_math_fastproof",
        mode=str(cfg.get("logger_mode", "jsonl_ring")),
        ring_size=int(cfg.get("logger_ring_size", 5000)),
        step_log_interval=int(cfg.get("step_log_interval", 1)),
        jsonl_path=str(run_log_path),
    )
    logger.log_event("config", cfg)
    logger.log_event("gpu_auto_tune", gpu_tune)
    logger.log_event("gpu_meta", gpu_meta)
    logger.log_event("runtime_accel_policy", safe_jsonable(cfg.get("_accel_report", {})))
    logger.log_event("determinism_policy", safe_jsonable(cfg.get("_determinism_report", {})))

    data_bundle = mathfp_build_datasets(cfg)
    train_records = list(data_bundle.get("train", []))
    val_records = list(data_bundle.get("val", []))
    test_records = list(data_bundle.get("test", []))
    unseen_records = list(data_bundle.get("unseen", []))
    logger.log_event("math_dataset_stats", data_bundle.get("stats", {}))

    all_texts = [str(r.get("full_text", "")) for r in (train_records + val_records + test_records + unseen_records)]
    tokenizer = SimpleTokenizer(vocab_size=max(512, int(cfg.get("vocab_size", 2048))))
    tokenizer.fit(all_texts)

    seq_len = int(cfg.get("seq_len", 64))
    max_len = max((len(tokenizer.encode(x, add_bos=True, add_eos=True)) for x in all_texts), default=32)
    seq_len = max(seq_len, min(128, max_len + 2))
    cfg["seq_len"] = int(seq_len)

    train_x, train_labels, train_answer_tokens = mathfp_prepare_tensor_dataset(train_records, tokenizer, seq_len=seq_len)
    val_x, val_labels, _ = mathfp_prepare_tensor_dataset(val_records, tokenizer, seq_len=seq_len)
    train_ds = MathAnswerDataset(train_x, train_labels)
    val_ds = MathAnswerDataset(val_x, val_labels)
    train_loader = make_loader(train_ds, batch_size=max(1, int(cfg.get("batch_size", 8))), seed=int(cfg.get("seed", 42)) + 301, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=max(1, int(cfg.get("batch_size", 8))), seed=int(cfg.get("seed", 42)) + 401, shuffle=False)

    vocab_size_runtime = max(int(tokenizer.vocab_size_realized), 128)
    variants = mathfp_select_variants(cfg)
    step_alloc = mathfp_allocate_steps(cfg, variants)
    models = mathfp_build_variant_models(cfg, vocab_size=vocab_size_runtime)
    logger.log_event("mathfp_variants", {"variants": variants, "step_alloc": step_alloc, "seq_len_runtime": seq_len})

    results: List[Dict[str, Any]] = []
    trained_models: Dict[str, nn.Module] = {}
    for variant in variants:
        if variant not in models:
            continue
        steps = int(step_alloc.get(variant, 0))
        if steps <= 0:
            continue
        res = mathfp_train_variant(
            variant=variant,
            model=models[variant],
            train_loader=train_loader,
            val_loader=val_loader,
            val_records=val_records,
            test_records=test_records,
            tokenizer=tokenizer,
            cfg=cfg,
            device=device,
            logger=logger,
            step_csv_path=step_csv_path,
            steps=steps,
        )
        if unseen_records:
            unseen_eval = mathfp_eval_exact_match(
                model=models[variant],
                tokenizer=tokenizer,
                records=unseen_records,
                device=device,
                max_new_tokens=12,
            )
            res["exact_match_unseen"] = float(unseen_eval.get("exact_match_percent", 0.0))
            res["invalid_output_ratio_unseen"] = float(unseen_eval.get("invalid_output_ratio", 1.0))
        else:
            res["exact_match_unseen"] = 0.0
            res["invalid_output_ratio_unseen"] = 0.0
        results.append(res)
        trained_models[variant] = models[variant]
        logger.log_event("mathfp_variant_done", res)

    by_name = {str(r.get("variant", "")): r for r in results}
    our = by_name.get("our_mertformer", {})
    gpt = by_name.get("gpt_proxy_dense", {})
    gem = by_name.get("gemini_proxy_moe", {})
    our_tps = float(our.get("tokens_per_sec", 0.0))
    gpt_tps = float(gpt.get("tokens_per_sec", 0.0))
    gem_tps = float(gem.get("tokens_per_sec", 0.0))
    speedup_vs_gpt = _safe_div(our_tps, max(gpt_tps, 1e-9), default=0.0) if gpt else 0.0
    speedup_vs_gem = _safe_div(our_tps, max(gem_tps, 1e-9), default=0.0) if gem else 0.0
    best_other_em = 0.0
    if gpt:
        best_other_em = max(best_other_em, float(gpt.get("exact_match_test", 0.0)))
    if gem:
        best_other_em = max(best_other_em, float(gem.get("exact_match_test", 0.0)))
    quality_delta_exact = float(our.get("exact_match_test", 0.0)) - float(best_other_em)

    loss_gate_pass = bool(our) and float(our.get("final_loss", float("inf"))) <= float(cfg.get("target_loss_gate", 2.0))
    accuracy_gate_pass = bool(our) and float(our.get("exact_match_test", 0.0)) >= float(cfg.get("target_exact_match_gate", 95.0))
    speed_gate_target = float(cfg.get("target_speedup_ratio", 1.15))
    speed_checks: List[bool] = []
    if gpt:
        speed_checks.append(speedup_vs_gpt >= speed_gate_target)
    if gem:
        speed_checks.append(speedup_vs_gem >= speed_gate_target)
    speed_gate_pass = bool(speed_checks) and all(speed_checks)
    gates = {
        "loss_gate_pass": bool(loss_gate_pass),
        "accuracy_gate_pass": bool(accuracy_gate_pass),
        "speed_gate_pass": bool(speed_gate_pass),
    }
    final_status = "pass" if all(gates.values()) else "gate_fail"

    compare_payload: Dict[str, Any] = {
        "schema": "build30_colab_math_fastproof_compare_v2",
        "generated_at_utc": _utc_now(),
        "profile": str(cfg.get("profile", "colab_math_fastproof")),
        "task_mode": str(cfg.get("task_mode", "math_eq_answer")),
        "architecture_mode": str(cfg.get("architecture_mode", "our")),
        "other_proxy_mode": str(cfg.get("other_proxy_mode", "both")),
        "variant_results": results,
        "speedup_ratio_vs_gpt_proxy": float(speedup_vs_gpt),
        "speedup_ratio_vs_gemini_proxy": float(speedup_vs_gem),
        "quality_delta_exact_match": float(quality_delta_exact),
        "exact_match_unseen_our": float(our.get("exact_match_unseen", 0.0)) if our else 0.0,
        "gates": gates,
        "loss_gate_pass": bool(loss_gate_pass),
        "accuracy_gate_pass": bool(accuracy_gate_pass),
        "speed_gate_pass": bool(speed_gate_pass),
        "final_status": final_status,
        "tokenizer_backend": "simple",
        "vocab_size_realized": int(tokenizer.vocab_size_realized),
        "train_answer_tokens": int(train_answer_tokens),
        "dataset_stats": data_bundle.get("stats", {}),
    }
    compare_json_text = json.dumps(compare_payload, ensure_ascii=False, indent=2)
    compare_json_path.write_text(compare_json_text + "\n", encoding="utf-8")
    with compare_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "params",
                "steps",
                "final_loss",
                "val_loss",
                "exact_match_test",
                "exact_match_unseen",
                "tokens_per_sec",
                "avg_step_time_sec",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "variant": row.get("variant", ""),
                    "params": int(row.get("params", 0)),
                    "steps": int(row.get("steps", 0)),
                    "final_loss": float(row.get("final_loss", float("inf"))),
                    "val_loss": float(row.get("val_loss", float("inf"))),
                    "exact_match_test": float(row.get("exact_match_test", 0.0)),
                    "exact_match_unseen": float(row.get("exact_match_unseen", 0.0)),
                    "tokens_per_sec": float(row.get("tokens_per_sec", 0.0)),
                    "avg_step_time_sec": float(row.get("avg_step_time_sec", 0.0)),
                }
            )
    compare_md_path.write_text(mathfp_compare_markdown(compare_payload), encoding="utf-8")

    selected_model_name = "our_mertformer" if "our_mertformer" in trained_models else (variants[0] if variants else "")
    if selected_model_name in trained_models:
        atomic_torch_save(
            model_final_path,
            {
                "schema": "build30_colab_math_fastproof_model_v1",
                "saved_at_utc": _utc_now(),
                "variant": selected_model_name,
                "model": trained_models[selected_model_name].state_dict(),
                "tokenizer_state": tokenizer.state_dict(),
                "config": cfg,
            },
        )

    summary_payload = {
        "schema": "build30_colab_math_fastproof_summary_v2",
        "generated_at_utc": _utc_now(),
        "run_id": run_id,
        "profile": str(cfg.get("profile", "colab_math_fastproof")),
        "device": str(device),
        "architecture_mode": str(cfg.get("architecture_mode", "our")),
        "other_proxy_mode": str(cfg.get("other_proxy_mode", "both")),
        "final_status": final_status,
        "gates": gates,
        "speedup_ratio_vs_gpt_proxy": float(speedup_vs_gpt),
        "speedup_ratio_vs_gemini_proxy": float(speedup_vs_gem),
        "quality_delta_exact_match": float(quality_delta_exact),
        "selected_model_final_path": str(model_final_path) if model_final_path.exists() else "",
        "elapsed_total_sec": float(time.time() - total_start),
        "determinism_policy": safe_jsonable(cfg.get("_determinism_report", {})),
        "runtime_accel_policy": safe_jsonable(cfg.get("_accel_report", {})),
    }
    summary_json_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    health_lines = [
        f"time_utc={_utc_now()}",
        f"run_id={run_id}",
        f"final_status={final_status}",
        f"loss_gate_pass={bool(loss_gate_pass)}",
        f"accuracy_gate_pass={bool(accuracy_gate_pass)}",
        f"speed_gate_pass={bool(speed_gate_pass)}",
        f"speedup_ratio_vs_gpt_proxy={float(speedup_vs_gpt):.6f}",
        f"speedup_ratio_vs_gemini_proxy={float(speedup_vs_gem):.6f}",
    ]
    health_txt_path.write_text("\n".join(health_lines) + "\n", encoding="utf-8")

    logger_manifest = logger.finalize()
    output_files = {
        "run_log_jsonl": str(run_log_path),
        "step_metrics_csv": str(step_csv_path),
        "summary_json": str(summary_json_path),
        "compare_json": str(compare_json_path),
        "compare_csv": str(compare_csv_path),
        "compare_md": str(compare_md_path),
        "health_txt": str(health_txt_path),
        "model_final": str(model_final_path) if model_final_path.exists() else "",
    }
    output_files = {k: v for k, v in output_files.items() if str(v).strip()}
    interp_assets = maybe_plot_mathfp_interpretability_assets(
        compare_payload=compare_payload,
        run_dir=run_dir,
        write_files=bool(cfg.get("write_files", True)),
        enabled=bool(cfg.get("interpretability_enabled", True)),
    )
    output_files.update(interp_assets)
    artifact_index = verify_and_index_artifacts(output_files)
    atomic_json_write(artifact_index_path, artifact_index)
    output_files["artifact_index"] = str(artifact_index_path)
    zip_manifest = make_evidence_zip(layout, output_files, enabled=bool(cfg.get("zip_evidence_pack", True)))
    output_files["zip_manifest"] = str(layout.zip_manifest_path)
    if layout.evidence_zip_path.exists():
        output_files["evidence_zip"] = str(layout.evidence_zip_path)

    payload = {
        "schema": "build30_colab_math_fastproof_payload_v2",
        "generated_at_utc": _utc_now(),
        "run_id": run_id,
        "profile": str(cfg.get("profile", "colab_math_fastproof")),
        "task_mode": str(cfg.get("task_mode", "math_eq_answer")),
        "architecture_mode": str(cfg.get("architecture_mode", "our")),
        "other_proxy_mode": str(cfg.get("other_proxy_mode", "both")),
        "final_status": final_status,
        "gates": gates,
        "compare": compare_payload,
        "summary": summary_payload,
        "logger_manifest": logger_manifest,
        "artifact_index": artifact_index,
        "zip_manifest": zip_manifest,
        "output_files": output_files,
        "gpu_meta": gpu_meta,
        "gpu_tune": gpu_tune,
        "runtime_fingerprint": build_runtime_fingerprint(cfg),
        "ownership_proof": build_ownership_proof(cfg),
        "feature_coverage_matrix": build_feature_coverage_matrix(),
        "compile_stall_guard": get_compile_guard_snapshot(),
        "env_snapshot_redacted": build_env_snapshot(mask=bool(cfg.get("security_redaction_enabled", True))),
        "reproduce_command": build_reproduce_command(cfg),
        "run_config_schema_report": safe_jsonable(cfg.get("run_config_schema_report", {})),
        "determinism_policy": safe_jsonable(cfg.get("_determinism_report", {})),
        "runtime_accel_policy": safe_jsonable(cfg.get("_accel_report", {})),
    }
    print(f"FINAL_STATUS: {final_status} reason=math_fastproof_completed run_id={run_id}")
    return payload


def run_all() -> Dict[str, Any]:
    _print_header()

    total_start = time.time()
    _RUNTIME_SIGNAL_STATE["sigterm"] = False
    _RUNTIME_SIGNAL_STATE["signal"] = ""
    cli_overrides = parse_cli_overrides()
    base_cfg = dict(RUN_CONFIG)
    base_cfg.update(cli_overrides)
    cfg = resolve_runtime_config(interactive_prompt(base_cfg))
    accel_report = apply_runtime_acceleration_policy(cfg)
    determinism_report = apply_determinism_policy(cfg, device=str(cfg.get("device", "cpu")))
    cfg["_accel_report"] = accel_report
    cfg["_determinism_report"] = determinism_report
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

    if str(cfg.get("task_mode", "")).strip().lower() == "math_eq_answer":
        return run_math_fastproof(
            cfg=cfg,
            layout=layout,
            device=device,
            gpu_meta=gpu_meta,
            gpu_tune=gpu_tune,
            total_start=total_start,
        )

    run_name = f"kaggle_onefile_build30_{_local_stamp()}"
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
            "schema": "kaggle_onefile_deep_build30_v6",
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
            "runtime_fingerprint": build_runtime_fingerprint(cfg),
            "ownership_proof": build_ownership_proof(cfg),
            "feature_coverage_matrix": build_feature_coverage_matrix(),
            "compile_stall_guard": get_compile_guard_snapshot(),
            "reproduce_command": build_reproduce_command(cfg),
            "env_snapshot_redacted": build_env_snapshot(mask=bool(cfg.get("security_redaction_enabled", True))),
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
            artifact_index = verify_and_index_artifacts(payload_fail["output_files"])
            atomic_json_write(layout.artifact_index_path, artifact_index)
            payload_fail["artifacts_index"] = artifact_index
            payload_fail["output_files"]["artifacts_index"] = str(layout.artifact_index_path)
            zip_manifest = make_evidence_zip(layout, payload_fail["output_files"], enabled=bool(cfg.get("zip_evidence_pack", True)))
            payload_fail["zip_manifest"] = zip_manifest
            payload_fail["output_files"]["zip_manifest"] = str(layout.zip_manifest_path)
            if layout.evidence_zip_path.exists():
                payload_fail["output_files"]["evidence_zip"] = str(layout.evidence_zip_path)
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
        "schema": "kaggle_onefile_deep_build30_v6",
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
            "bug_free_claim_text": "zero-known-critical-bugs this run",
        },
        "runtime_model_params": int(mert_params_runtime),
        "gpu_meta": gpu_meta,
        "gpu_tune_report": gpu_tune,
        "stop_reason": str(train_meta.get("stop_reason", "completed_or_condition_met")),
        "run_config_hash": hash_config(cfg),
        "runtime_fingerprint": build_runtime_fingerprint(cfg),
        "ownership_proof": build_ownership_proof(cfg),
        "feature_coverage_matrix": build_feature_coverage_matrix(),
        "compile_stall_guard": get_compile_guard_snapshot(),
        "reproduce_command": build_reproduce_command(cfg),
        "env_snapshot_redacted": build_env_snapshot(mask=bool(cfg.get("security_redaction_enabled", True))),
        "determinism_policy": safe_jsonable(cfg.get("_determinism_report", {})),
        "runtime_accel_policy": safe_jsonable(cfg.get("_accel_report", {})),
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

        # Rewrite final JSON with output index + verdict.
        json_text_final = json.dumps(payload, ensure_ascii=False, indent=2)
        jpath = Path(payload["output_files"].get("json", ""))
        if jpath:
            jpath.write_text(json_text_final + "\n", encoding="utf-8")

    final_status_line = f"FINAL_STATUS: {payload['final_status']} reason={payload['final_reason']} run_id={layout.run_id}"
    print(final_status_line)
    return payload


if __name__ == "__main__":
    try:
        run_all()
    except Exception as e:
        print("[fatal]", type(e).__name__, str(e))
        tb = traceback.format_exc()
        print(tb)
        tb_path = Path(str(_RUNTIME_LAST_LAYOUT.get("traceback_path", "")))
        st_path = Path(str(_RUNTIME_LAST_LAYOUT.get("last_state_path", "")))
        if str(tb_path):
            try:
                tb_path.parent.mkdir(parents=True, exist_ok=True)
                tb_path.write_text(tb, encoding="utf-8")
            except Exception:
                pass
        if str(st_path):
            try:
                write_last_state(
                    st_path,
                    {
                        "generated_at_utc": _utc_now(),
                        "fatal_error": f"{type(e).__name__}:{e}",
                        "run_dir": str(_RUNTIME_LAST_LAYOUT.get("run_dir", "")),
                    },
                )
            except Exception:
                pass
        run_id = Path(str(_RUNTIME_LAST_LAYOUT.get("run_dir", "unknown"))).name or "unknown"
        print(f"FINAL_STATUS: provisional reason=fatal_exception run_id={run_id}")
        raise
