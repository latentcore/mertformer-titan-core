#!/usr/bin/env python3
"""
MertFormer Chess RTX 5080 Onefile
---------------------------------
Single-file Windows-friendly chess proof lane for a single RTX 5080 desktop.

Goals:
- single-command PyCharm execution once dependencies are installed
- optional first-run dependency bootstrap with explicit operator opt-in
- deterministic multi-archive Lichess partial ingestion on the target machine
- legal-move-safe chess model training and evidence packaging
- internal benchmark/report artifacts that are stricter and more claim-safe than the original PoC

This file intentionally stays repo-owned and readable. A separate Windows delivery
build can compile a hardened standalone executable for external sharing, but the
proof lane here remains open and auditable.
"""
from __future__ import annotations

import argparse
import atexit
import contextlib
import csv
import ctypes
import enum
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import logging
import logging.handlers
import math
import os
import platform
import random
import re
import shutil
import socket
import struct
import subprocess
import sys
import textwrap
import threading
import time
import traceback
import urllib.error
import urllib.request
import warnings
import zipfile
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

SCRIPT_VERSION = "mertformer_chess_5080_onefile_v2"
SCRIPT_BASENAME = "mertformer_chess_5080_onefile"
RESULT_ZIP_PREFIX = "MertFormer_Chess_5080_Result"
DELIVERY_PREFIX = "MertFormer_Chess_5080_Delivery"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ALLOW_INSTALL_ENV = "MERTFORMER_CHESS_ALLOW_INSTALL"
DEFAULT_SKIP_BOOTSTRAP_ENV = "MERTFORMER_CHESS_SKIP_BOOTSTRAP"
DEFAULT_SHARE_MODE_ENV = "MERTFORMER_CHESS_SHARE_MODE"
DEFAULT_SELF_DELETE_ENV = "MERTFORMER_CHESS_SELF_DELETE"
DEFAULT_SELF_DELETE_TARGET_ENV = "MERTFORMER_CHESS_SELF_DELETE_TARGET"
DEFAULT_TEST_MODE_ENV = "MERTFORMER_CHESS_TEST_MODE"
DEFAULT_TORCH_INDEX_ENV = "MERTFORMER_CHESS_TORCH_INDEX_URL"
DEFAULT_ARCHIVE_PASSWORD_ENV = "MERTFORMER_CHESS_ARCHIVE_PASSWORD"
DEFAULT_ENCRYPT_OUTPUT_ENV = "MERTFORMER_CHESS_ENCRYPT_OUTPUT"
DEFAULT_ENCRYPTION_REQUIRED_ENV = "MERTFORMER_CHESS_ENCRYPTION_REQUIRED"
DEFAULT_CLEANUP_AFTER_BUNDLE_ENV = "MERTFORMER_CHESS_CLEANUP_AFTER_BUNDLE"
DEFAULT_SINGLE_OUTPUT_ENV = "MERTFORMER_CHESS_SINGLE_OUTPUT"
DEFAULT_PROFILE_ENV = "MERTFORMER_CHESS_PROFILE"
DEFAULT_ARTIFACT_ROOT_ENV = "MERTFORMER_CHESS_ARTIFACT_ROOT"
DEFAULT_CACHE_ROOT_ENV = "MERTFORMER_CHESS_CACHE_ROOT"
DEFAULT_STOCKFISH_CACHE_ROOT_ENV = "MERTFORMER_CHESS_STOCKFISH_CACHE_ROOT"
DEFAULT_STOCKFISH_AUTO_FETCH_ENV = "MERTFORMER_CHESS_STOCKFISH_AUTO_FETCH"
LOG_SCHEMA_VERSION = "2.0"
DEFAULT_LOG_MAX_BYTES = 2 * 1024 * 1024
DEFAULT_LOG_BACKUP_COUNT = 3
DEFAULT_LOG_PAYLOAD_PREVIEW_CHARS = 4000
DEFAULT_LOG_CONSOLE_LEVEL = "INFO"

EMBEDDED_SEED_PGN = textwrap.dedent(
    """
    [Event "Rated Seed Game 1"]
    [Site "Local"]
    [Date "2026.01.01"]
    [Round "-"]
    [White "SeedA"]
    [Black "SeedB"]
    [Result "1-0"]
    [WhiteElo "2150"]
    [BlackElo "2100"]
    [TimeControl "300+0"]
    [Termination "Normal"]

    1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. c4 b4 12. a3 bxa3 13. Nxa3 Bb7 14. Bc2 Re8 15. b4 Bf8 16. d5 c6 17. Be3 cxd5 18. cxd5 Nb6 19. Bd3 Nfd7 20. Nc4 Nxc4 21. Bxc4 Be7 22. Qa4 Rf8 23. Rec1 f5 24. exf5 Rxf5 25. Bd3 Rxf3 26. gxf3 Bxd5 27. Be4 Bxe4 28. fxe4 Nf6 29. Qc6 Kh8 30. Rxa6 Rxa6 31. Qxa6 Qd7 32. Qc8+ Qxc8 33. Rxc8+ Ng8 34. b5 1-0

    [Event "Rated Seed Game 2"]
    [Site "Local"]
    [Date "2026.01.02"]
    [Round "-"]
    [White "SeedC"]
    [Black "SeedD"]
    [Result "0-1"]
    [WhiteElo "2200"]
    [BlackElo "2230"]
    [TimeControl "300+3"]
    [Termination "Normal"]

    1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. e3 O-O 5. Bd3 d5 6. Nf3 c5 7. O-O dxc4 8. Bxc4 cxd4 9. exd4 b6 10. Bg5 Bb7 11. Re1 Nbd7 12. Rc1 Rc8 13. Bd3 h6 14. Bh4 Re8 15. Ne5 Be7 16. Bg3 Nxe5 17. Bxe5 Nd5 18. Qg4 Bf6 19. Nb5 Rxc1 20. Rxc1 Bxe5 21. dxe5 Re7 22. Nd6 Rc7 23. Rxc7 Qxc7 24. h4 Qc1+ 25. Bf1 Ba6 26. Qf3 Qxf1+ 27. Kh2 f6 28. exf6 Nxf6 29. Qa8+ Kh7 30. Qxa7 Be2 31. f3 Qf2 32. Qa4 Bxf3 33. Qc2+ Qxc2 34. Kg3 Qxg2+ 35. Kf4 Nd5+ 36. Ke5 Qe2+ 37. Kd4 Qd2+ 38. Ke5 Qf4+ 39. Kxe6 Bg4+ 40. Kxd5 Bf3+ 41. Ke6 Qf6+ 42. Kd7 Bg4+ 43. Kc7 Qe7+ 44. Kc6 Bf3+ 45. Kxb6 Qxd6+ 46. Ka5 Qc5+ 47. Ka4 Bd1+ 48. b3 Be2 49. b4 Qc3 50. Ka5 Bc4 0-1

    [Event "Rated Seed Game 3"]
    [Site "Local"]
    [Date "2026.01.03"]
    [Round "-"]
    [White "SeedE"]
    [Black "SeedF"]
    [Result "1-0"]
    [WhiteElo "2050"]
    [BlackElo "2080"]
    [TimeControl "600+0"]
    [Termination "Normal"]

    1. c4 e5 2. Nc3 Nf6 3. Nf3 Nc6 4. g3 d5 5. cxd5 Nxd5 6. Bg2 Nb6 7. O-O Be7 8. d3 O-O 9. Be3 Re8 10. Rc1 Bf8 11. a3 Nd4 12. Nd2 c6 13. b4 Bg4 14. h3 Bh5 15. g4 Bg6 16. Nce4 Nd5 17. Bg5 f6 18. Bh4 Nf4 19. Re1 a5 20. e3 Nxd3 21. exd4 Nxc1 22. dxe5 Rxe5 23. Qxc1 axb4 24. axb4 Bxb4 25. Qc4+ Bf7 26. Qxb4 Rb5 27. Qc3 Ra2 28. Nf3 Bd5 29. g5 Bxe4 30. Qc4+ Bd5 31. Qg4 fxg5 32. Bxg5 Qf8 33. Be7 Qf7 34. Qc8+ Qf8 35. Qxf8# 1-0
    """
).strip()

DEFAULT_LICHESS_URLS = [
    "https://database.lichess.org/standard/lichess_db_standard_rated_2026-03.pgn.zst",
    "https://database.lichess.org/standard/lichess_db_standard_rated_2026-02.pgn.zst",
    "https://database.lichess.org/standard/lichess_db_standard_rated_2026-01.pgn.zst",
    "https://database.lichess.org/standard/lichess_db_standard_rated_2025-12.pgn.zst",
    "https://database.lichess.org/standard/lichess_db_standard_rated_2025-11.pgn.zst",
    "https://database.lichess.org/standard/lichess_db_standard_rated_2025-10.pgn.zst",
]

OPENING_SEEDS = [
    ["e2e4", "e7e5", "g1f3", "b8c6"],
    ["d2d4", "g8f6", "c2c4", "e7e6"],
    ["c2c4", "e7e5", "g1f3", "b8c6"],
    ["e2e4", "c7c5", "g1f3", "d7d6"],
    ["d2d4", "d7d5", "c2c4", "e7e6"],
    ["g1f3", "d7d5", "c2c4", "d5d4"],
]

RUN_CONFIG: Dict[str, Any] = {
    "mode": "train",
    "profile": "production_5080",
    "baseline": "dense",
    "feature_bundle": "default",
    "enabled_features": [],
    "disabled_features": [],
    "seed": 42,
    "device": "auto",
    "artifact_root": "~/Desktop",
    "cache_root": "~/Desktop/mertformer_chess_cache",
    "delivery_mode": False,
    "result_prefix": RESULT_ZIP_PREFIX,
    "redact_paths": True,
    "allow_install": False,
    "share_mode": False,
    "enable_self_delete": False,
    "self_delete_target": "",
    "determinism_strict": True,
    "auto_download_enabled": True,
    "offline_seed_only": False,
    "test_mode": False,
    "encrypt_output": False,
    "archive_encryption_required": False,
    "archive_password_env": DEFAULT_ARCHIVE_PASSWORD_ENV,
    "single_output_only": False,
    "cleanup_after_bundle": False,
    "download_partial_mb": 768,
    "download_archive_count": 4,
    "download_timeout_sec": 60,
    "download_retries": 2,
    "download_retry_backoff_sec": 2.0,
    "download_content_type_allowlist": [
        "application/octet-stream",
        "application/zstd",
        "binary/octet-stream",
    ],
    "max_games": 120000,
    "max_positions": 480000,
    "max_positions_per_game": 8,
    "position_selection_strategy": "scored",
    "min_elo": 1900,
    "time_control_min_seconds": 180,
    "time_control_max_seconds": 900,
    "exclude_time_forfeit": True,
    "prefer_eval_positions": True,
    "dedupe_games": True,
    "dedupe_positions": True,
    "val_fraction": 0.12,
    "test_fraction": 0.08,
    "curriculum_enabled": True,
    "curriculum_stage_fracs": [0.20, 0.30, 0.50],
    "max_wall_hours": 4.0,
    "max_steps": 28000,
    "batch_size": 192,
    "eval_batch_size": 192,
    "learning_rate": 3.0e-4,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "grad_clip": 1.0,
    "grad_accum_steps": 1,
    "hidden_size": 384,
    "intermediate_size": 1536,
    "num_layers": 8,
    "num_heads": 8,
    "num_kv_heads": 4,
    "head_dim": 48,
    "max_seq_len": 80,
    "dropout": 0.10,
    "attention_dropout": 0.0,
    "ffn_dropout": 0.0,
    "rms_norm_eps": 1e-6,
    "use_moe": False,
    "moe_top_k": 2,
    "num_experts": 4,
    "moe_every_n_layers": 3,
    "moe_intermediate": 1536,
    "router_temperature": 1.0,
    "router_jitter": 0.02,
    "router_jitter_boost": 0.10,
    "router_alarm_threshold": 0.40,
    "shared_expert_gate": 0.0,
    "z_loss_coef": 1e-4,
    "use_switch_loss": True,
    "moe_capacity_enforce": True,
    "moe_capacity_factor": 1.25,
    "moe_dispatch_mode": "sequential",
    "use_expert_paging": False,
    "expert_paging_inference_only": True,
    "expert_paging_lazy_init": True,
    "expert_paging_cache_size": 2,
    "expert_paging_offload_device": "cpu",
    "expert_paging_verbose": False,
    "use_bitlinear": False,
    "use_liquid": False,
    "use_liquid_adapter": False,
    "liquid_layers_idx": [],
    "liquid_every_n_layers": 0,
    "liquid_fast_path": True,
    "use_qinn": False,
    "qinn_every_n_layers": 1,
    "rope_theta": 100000.0,
    "rope_base": 100000.0,
    "rope_dim": None,
    "use_flash_attn_inference": False,
    "use_hierarchical_kv_cache": False,
    "hkv_short_window": 512,
    "hkv_long_stride": 8,
    "hkv_max_long_blocks": 128,
    "use_global_workspace_broadcast": False,
    "workspace_blend": 0.7,
    "use_neuromodulatory_gain": False,
    "use_latent_ode_state_channel": False,
    "latent_ode_dt": 1.0,
    "use_cross_expert_sync_bus": False,
    "cross_expert_sync_gain": 0.05,
    "use_structural_plasticity": False,
    "structural_ema_decay": 0.98,
    "structural_prune_threshold": 0.02,
    "structural_grow_threshold": 0.60,
    "structural_update_interval": 100,
    "use_hebbian_plasticity": False,
    "hebbian_eta": 0.01,
    "hebbian_decay": 0.99,
    "use_neuro_symbolic_layer": False,
    "neuro_symbolic_rules": 8,
    "use_world_model_head": False,
    "world_model_horizon": 1,
    "use_phase_head": False,
    "phase_loss_coef": 0.05,
    "use_wdl_head": False,
    "wdl_loss_coef": 0.08,
    "wdl_draw_threshold": 0.20,
    "use_legality_head": False,
    "legality_loss_coef": 0.03,
    "legality_pos_weight_cap": 64.0,
    "use_lifelong_safety_layer": False,
    "lifelong_ema_decay": 0.99,
    "lifelong_max_adaptation_gain": 0.05,
    "lifelong_drift_threshold": 0.35,
    "use_gradient_checkpointing": False,
    "compile_policy": "off",
    "use_bf16": True,
    "num_workers": 0,
    "eval_interval": 400,
    "checkpoint_interval": 1000,
    "training_eval_batches": 16,
    "legal_move_sample_checks": 4096,
    "resume_from": "",
    "sample_replay_games": 3,
    "sample_replay_max_plies": 24,
    "selfplay_eval_enabled": False,
    "selfplay_games": 4,
    "selfplay_max_plies": 96,
    "selfplay_opening_prefix_plies": 2,
    "tournament_eval_enabled": False,
    "tournament_games": 6,
    "tournament_max_plies": 96,
    "replay_buffer_enabled": False,
    "replay_buffer_max_positions": 256,
    "include_curated_position_suites": True,
    "curated_position_repeat": 6,
    "synthetic_teaching_corpus_enabled": True,
    "curated_suite_eval_enabled": True,
    "stockfish_path": "",
    "stockfish_auto_fetch": True,
    "stockfish_cache_root": "~/Desktop/mertformer_chess_cache/stockfish",
    "stockfish_release_api": "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest",
    "stockfish_download_timeout_sec": 60,
    "stockfish_ladder": [
        {"label": "sf_skill4_nodes20k", "games": 12, "skill": 4, "nodes": 20000, "anchor_elo_proxy": 1100},
        {"label": "sf_skill8_nodes50k", "games": 12, "skill": 8, "nodes": 50000, "anchor_elo_proxy": 1400},
        {"label": "sf_skill12_nodes100k", "games": 16, "skill": 12, "nodes": 100000, "anchor_elo_proxy": 1700},
    ],
    "search_enabled": True,
    "search_candidate_topk": 4,
    "search_reply_topk": 3,
    "search_policy_blend": 0.35,
    "search_value_blend": 0.65,
    "search_tactical_bonus": 0.05,
    "search_auto_budget": True,
    "midrun_curated_snapshot_interval": 0,
    "midrun_stockfish_snapshot_interval": 0,
    "midrun_stockfish_snapshot_games": 4,
    "rating_target_proxy_threshold": 1600,
    "claim_min_benchmark_games": 40,
    "zip_outputs": True,
    "lichess_urls": DEFAULT_LICHESS_URLS,
}

FEATURE_FLAG_KEYS: Tuple[str, ...] = (
    "use_moe",
    "use_bitlinear",
    "use_liquid",
    "use_liquid_adapter",
    "use_qinn",
    "use_flash_attn_inference",
    "use_hierarchical_kv_cache",
    "use_global_workspace_broadcast",
    "use_neuromodulatory_gain",
    "use_latent_ode_state_channel",
    "use_cross_expert_sync_bus",
    "use_structural_plasticity",
    "use_hebbian_plasticity",
    "use_neuro_symbolic_layer",
    "use_world_model_head",
    "use_phase_head",
    "use_wdl_head",
    "use_legality_head",
    "use_lifelong_safety_layer",
    "use_expert_paging",
    "use_gradient_checkpointing",
    "search_enabled",
    "selfplay_eval_enabled",
    "tournament_eval_enabled",
    "replay_buffer_enabled",
    "curriculum_enabled",
    "curated_suite_eval_enabled",
    "synthetic_teaching_corpus_enabled",
)

FEATURE_BUNDLES: Dict[str, Dict[str, Any]] = {
    "default": {
        "description": "Canonical repo-safe defaults with explicit opt-in for advanced surfaces.",
        "overrides": {},
    },
    "routing_stack": {
        "description": "MoE routing, paging, and expert coordination surfaces.",
        "overrides": {
            "use_moe": True,
            "moe_dispatch_mode": "parallel",
            "use_expert_paging": True,
            "use_cross_expert_sync_bus": True,
            "use_structural_plasticity": True,
        },
    },
    "liquid_stack": {
        "description": "Liquid/CfC, adapter, and QINN surfaces.",
        "overrides": {
            "use_liquid": True,
            "use_liquid_adapter": True,
            "liquid_fast_path": True,
            "use_qinn": True,
        },
    },
    "memory_attention_stack": {
        "description": "Inference-memory and attention-side optimizations.",
        "overrides": {
            "use_flash_attn_inference": True,
            "use_hierarchical_kv_cache": True,
            "use_gradient_checkpointing": True,
        },
    },
    "cognitive_stack": {
        "description": "Workspace, modulation, latent state, symbolic bridge, and world-model side surfaces.",
        "overrides": {
            "use_global_workspace_broadcast": True,
            "use_neuromodulatory_gain": True,
            "use_latent_ode_state_channel": True,
            "use_hebbian_plasticity": True,
            "use_neuro_symbolic_layer": True,
            "use_world_model_head": True,
            "use_lifelong_safety_layer": True,
        },
    },
    "objective_stack": {
        "description": "Chess-side auxiliary heads for phase, WDL, and legality shaping.",
        "overrides": {
            "use_phase_head": True,
            "phase_loss_coef": 0.05,
            "use_wdl_head": True,
            "wdl_loss_coef": 0.08,
            "use_legality_head": True,
            "legality_loss_coef": 0.03,
        },
    },
    "postrun_analysis_stack": {
        "description": "Self-play, inference-mode tournament, and replay-buffer artifact surfaces.",
        "overrides": {
            "selfplay_eval_enabled": True,
            "tournament_eval_enabled": True,
            "replay_buffer_enabled": True,
        },
    },
    "all_stable_extensions": {
        "description": "Broad but relatively stable advanced stack for ambitious local runs.",
        "overrides": {
            "use_moe": True,
            "use_liquid": True,
            "use_liquid_adapter": True,
            "use_qinn": True,
            "use_flash_attn_inference": True,
            "use_hierarchical_kv_cache": True,
            "use_global_workspace_broadcast": True,
            "use_neuromodulatory_gain": True,
            "use_latent_ode_state_channel": True,
            "use_hebbian_plasticity": True,
            "use_neuro_symbolic_layer": True,
            "use_world_model_head": True,
            "use_phase_head": True,
            "use_wdl_head": True,
            "use_legality_head": True,
            "use_lifelong_safety_layer": True,
            "use_gradient_checkpointing": True,
            "search_enabled": True,
            "selfplay_eval_enabled": True,
            "tournament_eval_enabled": True,
            "replay_buffer_enabled": True,
            "curriculum_enabled": True,
            "curated_suite_eval_enabled": True,
            "synthetic_teaching_corpus_enabled": True,
            "moe_dispatch_mode": "parallel",
        },
    },
    "all_on_experimental": {
        "description": "Maximum onefile surface activation, including experimental and higher-risk combinations.",
        "overrides": {
            "use_moe": True,
            "use_bitlinear": True,
            "use_liquid": True,
            "use_liquid_adapter": True,
            "use_qinn": True,
            "use_flash_attn_inference": True,
            "use_hierarchical_kv_cache": True,
            "use_global_workspace_broadcast": True,
            "use_neuromodulatory_gain": True,
            "use_latent_ode_state_channel": True,
            "use_cross_expert_sync_bus": True,
            "use_structural_plasticity": True,
            "use_hebbian_plasticity": True,
            "use_neuro_symbolic_layer": True,
            "use_world_model_head": True,
            "use_phase_head": True,
            "use_wdl_head": True,
            "use_legality_head": True,
            "use_lifelong_safety_layer": True,
            "use_expert_paging": True,
            "use_gradient_checkpointing": True,
            "search_enabled": True,
            "selfplay_eval_enabled": True,
            "tournament_eval_enabled": True,
            "replay_buffer_enabled": True,
            "curriculum_enabled": True,
            "curated_suite_eval_enabled": True,
            "synthetic_teaching_corpus_enabled": True,
            "moe_dispatch_mode": "parallel",
        },
    },
}

RUN_PROFILES: Dict[str, Dict[str, Any]] = {
    "production_5080": {
        "download_partial_mb": 768,
        "download_archive_count": 4,
        "max_games": 120000,
        "max_positions": 480000,
        "max_positions_per_game": 8,
        "max_steps": 28000,
        "max_wall_hours": 4.0,
        "batch_size": 192,
        "eval_batch_size": 192,
        "use_moe": False,
        "use_bitlinear": False,
        "use_liquid_adapter": False,
        "curated_position_repeat": 6,
        "stockfish_ladder": [
            {"label": "sf_skill4_nodes20k", "games": 12, "skill": 4, "nodes": 20000, "anchor_elo_proxy": 1100},
            {"label": "sf_skill8_nodes50k", "games": 12, "skill": 8, "nodes": 50000, "anchor_elo_proxy": 1400},
            {"label": "sf_skill12_nodes100k", "games": 16, "skill": 12, "nodes": 100000, "anchor_elo_proxy": 1700},
        ],
        "claim_min_benchmark_games": 40,
    },
    "benchmark_5080": {
        "download_partial_mb": 896,
        "download_archive_count": 5,
        "max_games": 150000,
        "max_positions": 560000,
        "max_positions_per_game": 8,
        "max_steps": 32000,
        "max_wall_hours": 4.0,
        "batch_size": 192,
        "eval_batch_size": 192,
        "use_moe": False,
        "use_bitlinear": False,
        "use_liquid_adapter": False,
        "curated_position_repeat": 8,
        "stockfish_ladder": [
            {"label": "sf_skill4_nodes20k", "games": 14, "skill": 4, "nodes": 20000, "anchor_elo_proxy": 1100},
            {"label": "sf_skill8_nodes50k", "games": 14, "skill": 8, "nodes": 50000, "anchor_elo_proxy": 1400},
            {"label": "sf_skill12_nodes100k", "games": 14, "skill": 12, "nodes": 100000, "anchor_elo_proxy": 1700},
            {"label": "sf_skill16_nodes200k", "games": 14, "skill": 16, "nodes": 200000, "anchor_elo_proxy": 1900},
        ],
        "claim_min_benchmark_games": 56,
    },
    "strength_4060_24h": {
        "download_partial_mb": 3072,
        "download_archive_count": 8,
        "max_games": 400000,
        "max_positions": 2400000,
        "max_positions_per_game": 12,
        "min_elo": 2200,
        "max_steps": 120000,
        "max_wall_hours": 24.0,
        "batch_size": 128,
        "eval_batch_size": 128,
        "hidden_size": 448,
        "intermediate_size": 1792,
        "num_layers": 10,
        "num_heads": 8,
        "num_kv_heads": 4,
        "head_dim": 56,
        "curated_position_repeat": 10,
        "search_enabled": True,
        "search_candidate_topk": 5,
        "search_reply_topk": 4,
        "search_auto_budget": True,
        "midrun_curated_snapshot_interval": 4000,
        "midrun_stockfish_snapshot_interval": 12000,
        "midrun_stockfish_snapshot_games": 4,
        "stockfish_ladder": [
            {"label": "sf_skill4_nodes40k", "games": 24, "skill": 4, "nodes": 40000, "anchor_elo_proxy": 1100},
            {"label": "sf_skill8_nodes80k", "games": 24, "skill": 8, "nodes": 80000, "anchor_elo_proxy": 1400},
            {"label": "sf_skill12_nodes160k", "games": 28, "skill": 12, "nodes": 160000, "anchor_elo_proxy": 1700},
            {"label": "sf_skill16_nodes320k", "games": 28, "skill": 16, "nodes": 320000, "anchor_elo_proxy": 1900},
        ],
        "rating_target_proxy_threshold": 2000,
        "claim_min_benchmark_games": 100,
    },
    "strength_4060_24h_all_on_experimental": {
        "download_partial_mb": 3072,
        "download_archive_count": 8,
        "max_games": 400000,
        "max_positions": 2400000,
        "max_positions_per_game": 12,
        "min_elo": 2200,
        "max_steps": 120000,
        "max_wall_hours": 24.0,
        "batch_size": 96,
        "eval_batch_size": 96,
        "hidden_size": 448,
        "intermediate_size": 1792,
        "num_layers": 10,
        "num_heads": 8,
        "num_kv_heads": 4,
        "head_dim": 56,
        "num_experts": 8,
        "moe_top_k": 2,
        "moe_every_n_layers": 2,
        "moe_intermediate": 1792,
        "moe_dispatch_mode": "parallel",
        "moe_capacity_factor": 1.5,
        "feature_bundle": "all_on_experimental",
        "expert_paging_cache_size": 2,
        "curated_position_repeat": 12,
        "search_enabled": True,
        "search_candidate_topk": 6,
        "search_reply_topk": 5,
        "search_auto_budget": True,
        "selfplay_eval_enabled": True,
        "selfplay_games": 6,
        "selfplay_max_plies": 110,
        "tournament_eval_enabled": True,
        "tournament_games": 6,
        "tournament_max_plies": 110,
        "replay_buffer_enabled": True,
        "replay_buffer_max_positions": 384,
        "midrun_curated_snapshot_interval": 3000,
        "midrun_stockfish_snapshot_interval": 10000,
        "midrun_stockfish_snapshot_games": 6,
        "stockfish_ladder": [
            {"label": "sf_skill4_nodes40k", "games": 24, "skill": 4, "nodes": 40000, "anchor_elo_proxy": 1100},
            {"label": "sf_skill8_nodes80k", "games": 24, "skill": 8, "nodes": 80000, "anchor_elo_proxy": 1400},
            {"label": "sf_skill12_nodes160k", "games": 28, "skill": 12, "nodes": 160000, "anchor_elo_proxy": 1700},
            {"label": "sf_skill16_nodes320k", "games": 28, "skill": 16, "nodes": 320000, "anchor_elo_proxy": 1900},
        ],
        "rating_target_proxy_threshold": 2000,
        "claim_min_benchmark_games": 100,
    },
    "strength_4060_24h_omni_max": {
        "download_partial_mb": 3584,
        "download_archive_count": 9,
        "max_games": 440000,
        "max_positions": 2600000,
        "max_positions_per_game": 12,
        "min_elo": 2200,
        "max_steps": 126000,
        "max_wall_hours": 24.0,
        "batch_size": 88,
        "eval_batch_size": 88,
        "hidden_size": 480,
        "intermediate_size": 1920,
        "num_layers": 10,
        "num_heads": 8,
        "num_kv_heads": 4,
        "head_dim": 60,
        "num_experts": 10,
        "moe_top_k": 2,
        "moe_every_n_layers": 2,
        "moe_intermediate": 1920,
        "moe_dispatch_mode": "parallel",
        "moe_capacity_factor": 1.6,
        "feature_bundle": "all_on_experimental",
        "enabled_features": list(FEATURE_FLAG_KEYS),
        "expert_paging_cache_size": 3,
        "curated_position_repeat": 16,
        "search_enabled": True,
        "search_candidate_topk": 7,
        "search_reply_topk": 5,
        "search_auto_budget": True,
        "search_policy_blend": 0.32,
        "search_value_blend": 0.68,
        "selfplay_eval_enabled": True,
        "selfplay_games": 8,
        "selfplay_max_plies": 120,
        "tournament_eval_enabled": True,
        "tournament_games": 8,
        "tournament_max_plies": 120,
        "replay_buffer_enabled": True,
        "replay_buffer_max_positions": 512,
        "midrun_curated_snapshot_interval": 2500,
        "midrun_stockfish_snapshot_interval": 9000,
        "midrun_stockfish_snapshot_games": 8,
        "stockfish_ladder": [
            {"label": "sf_skill4_nodes40k", "games": 24, "skill": 4, "nodes": 40000, "anchor_elo_proxy": 1100},
            {"label": "sf_skill8_nodes100k", "games": 24, "skill": 8, "nodes": 100000, "anchor_elo_proxy": 1400},
            {"label": "sf_skill12_nodes200k", "games": 28, "skill": 12, "nodes": 200000, "anchor_elo_proxy": 1700},
            {"label": "sf_skill16_nodes400k", "games": 28, "skill": 16, "nodes": 400000, "anchor_elo_proxy": 1900},
        ],
        "rating_target_proxy_threshold": 2050,
        "claim_min_benchmark_games": 104,
    },
    "benchmark_4060_hard": {
        "download_partial_mb": 4096,
        "download_archive_count": 10,
        "max_games": 500000,
        "max_positions": 3200000,
        "max_positions_per_game": 12,
        "min_elo": 2250,
        "max_steps": 160000,
        "max_wall_hours": 24.0,
        "batch_size": 128,
        "eval_batch_size": 128,
        "curated_position_repeat": 12,
        "search_enabled": True,
        "search_candidate_topk": 5,
        "search_reply_topk": 4,
        "search_auto_budget": True,
        "midrun_curated_snapshot_interval": 3000,
        "midrun_stockfish_snapshot_interval": 10000,
        "midrun_stockfish_snapshot_games": 6,
        "stockfish_ladder": [
            {"label": "sf_skill4_nodes60k", "games": 28, "skill": 4, "nodes": 60000, "anchor_elo_proxy": 1100},
            {"label": "sf_skill8_nodes120k", "games": 28, "skill": 8, "nodes": 120000, "anchor_elo_proxy": 1400},
            {"label": "sf_skill12_nodes240k", "games": 32, "skill": 12, "nodes": 240000, "anchor_elo_proxy": 1700},
            {"label": "sf_skill16_nodes400k", "games": 32, "skill": 16, "nodes": 400000, "anchor_elo_proxy": 1900},
        ],
        "rating_target_proxy_threshold": 2100,
        "claim_min_benchmark_games": 120,
    },
    "delivery_windows_oneclick": {
        "artifact_root": "~/Desktop/MertFormer_Chess_5080_Runtime",
        "cache_root": "~/Desktop/MertFormer_Chess_5080_Runtime/cache",
        "delivery_mode": True,
        "single_output_only": True,
        "cleanup_after_bundle": True,
        "download_partial_mb": 2048,
        "download_archive_count": 6,
        "max_games": 220000,
        "max_positions": 1200000,
        "max_positions_per_game": 10,
        "max_steps": 48000,
        "max_wall_hours": 8.0,
        "batch_size": 128,
        "eval_batch_size": 128,
        "min_elo": 2100,
        "stockfish_auto_fetch": True,
        "search_enabled": True,
        "search_candidate_topk": 4,
        "search_reply_topk": 3,
        "search_auto_budget": True,
        "midrun_curated_snapshot_interval": 2000,
        "midrun_stockfish_snapshot_interval": 8000,
        "midrun_stockfish_snapshot_games": 4,
        "rating_target_proxy_threshold": 1900,
        "claim_min_benchmark_games": 80,
    },
    "smoke": {
        "offline_seed_only": True,
        "auto_download_enabled": False,
        "download_partial_mb": 0,
        "max_games": 6,
        "max_positions": 96,
        "max_positions_per_game": 4,
        "max_steps": 8,
        "max_wall_hours": 0.03,
        "batch_size": 8,
        "eval_batch_size": 8,
        "hidden_size": 128,
        "intermediate_size": 512,
        "num_layers": 2,
        "num_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 32,
        "num_experts": 2,
        "use_moe": False,
        "use_bitlinear": False,
        "use_liquid": False,
        "use_liquid_adapter": False,
        "compile_policy": "off",
        "use_bf16": False,
        "curriculum_enabled": False,
        "curated_position_repeat": 2,
        "sample_replay_games": 1,
        "sample_replay_max_plies": 8,
        "stockfish_ladder": [],
        "rating_target_proxy_threshold": 1600,
    },
}


class ExecutionStatus(str, enum.Enum):
    RAN = "ran"
    PARTIALLY_RAN = "partially_ran"
    FAILED = "failed"


class EvaluationStatus(str, enum.Enum):
    UNEVALUATED = "unevaluated"
    INTERNALLY_MEASURED = "internally_measured"
    EXTERNALLY_VERIFIED = "externally_verified"


class RatingClaimStatus(str, enum.Enum):
    NO_CLAIM = "no_claim"
    PROXY_ONLY = "proxy_only"
    TARGET_NOT_MET = "target_not_met"
    TARGET_MET_INTERNAL = "target_met_internal"
    TARGET_MET_EXTERNAL = "target_met_external"


class ChessOnefileError(RuntimeError):
    pass


class ConfigValidationError(ChessOnefileError):
    pass


class DependencyBootstrapRequired(ChessOnefileError):
    pass


class DownloadError(ChessOnefileError):
    pass


class DatasetEmptyError(ChessOnefileError):
    pass


class TrainingOOMError(ChessOnefileError):
    pass


class PackagingError(ChessOnefileError):
    pass


class NonFiniteLossError(ChessOnefileError):
    pass


class ResumeCheckpointError(ChessOnefileError):
    pass


@dataclass
class ArtifactLayout:
    run_id: str
    root: Path
    run_dir: Path
    logs_dir: Path
    reports_dir: Path
    checkpoints_dir: Path
    export_dir: Path
    benchmark_dir: Path
    desktop_dir: Path
    final_zip_path: Path
    final_sha_path: Path


@dataclass
class ChessExample:
    piece_ids: List[int]
    meta_ids: List[int]
    legal_move_ids: List[int]
    target_move_id: int
    value_target: float
    phase: int
    source_game_id: str
    ply: int
    total_plies: int
    turn: int
    has_eval: bool
    opening_prefix: str
    value_source: str
    source_archive: str
    position_hash: str
    move_uci: str


@dataclass(frozen=True)
class CuratedPositionSpec:
    label: str
    suite: str
    game_index: int
    ply_index: int
    expected_move_uci: str
    expected_tags: Tuple[str, ...]
    teaching_focus: str
    commentary_tr: str


@dataclass
class DownloadSlice:
    url: str
    requested_range: str
    path: Path
    bytes_written: int
    sha256: str
    response_headers: Dict[str, str]
    http_status: int
    content_type: str
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "requested_range": self.requested_range,
            "path": str(self.path),
            "bytes_written": self.bytes_written,
            "sha256": self.sha256,
            "response_headers": self.response_headers,
            "http_status": self.http_status,
            "content_type": self.content_type,
            "error": self.error,
        }


@dataclass
class ResumeState:
    step: int
    best_val_loss: float
    metrics: Dict[str, Any]
    checkpoint_path: Path


class JSONLLogger:
    _LEVEL_MAP = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    _EVENT_LEVELS = {
        "archive_read_error": "ERROR",
        "bundle_cleanup": "INFO",
        "compile_fallback": "WARNING",
        "download_error": "ERROR",
        "fatal_exception": "CRITICAL",
        "logger_finalize": "INFO",
        "oom_event": "ERROR",
        "package_only_complete": "INFO",
        "pgn_parse_error": "WARNING",
        "power_guard": "INFO",
        "run_complete": "INFO",
        "run_start": "INFO",
        "training_stop": "WARNING",
    }

    def __init__(
        self,
        path: Path,
        *,
        run_id: str = "",
        mode: str = "",
        profile: str = "",
        artifact_root: str = "",
        component: str = "chess_onefile",
        max_bytes: int = DEFAULT_LOG_MAX_BYTES,
        backup_count: int = DEFAULT_LOG_BACKUP_COUNT,
        console_level: str = DEFAULT_LOG_CONSOLE_LEVEL,
    ):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self.mode = str(mode)
        self.profile = str(profile)
        self.artifact_root = str(artifact_root)
        self.component = str(component)
        self.max_bytes = int(max(1024, max_bytes))
        self.backup_count = int(max(1, backup_count))
        self.console_level = str(console_level or DEFAULT_LOG_CONSOLE_LEVEL).upper()
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self._finalized = False
        self._event_count = 0
        self._events = Counter()
        self._levels = Counter()
        self._lock = threading.RLock()
        self._logger = logging.getLogger(f"mertformer.chess.{sha256_bytes(str(self.path).encode('utf-8'))[:12]}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        self._logger.handlers.clear()
        self._file_handler = logging.handlers.RotatingFileHandler(
            self.path,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding="utf-8",
        )
        self._file_handler.setLevel(logging.DEBUG)
        self._file_handler.setFormatter(_JSONLLogFormatter())
        self._logger.addHandler(self._file_handler)
        self._console_handler = logging.StreamHandler(sys.stdout)
        self._console_handler.setLevel(self._LEVEL_MAP.get(self.console_level, logging.INFO))
        self._console_handler.setFormatter(_ConsoleLogFormatter())
        self._logger.addHandler(self._console_handler)
        atexit.register(self._atexit_finalize)

    def bind_context(
        self,
        *,
        run_id: Optional[str] = None,
        mode: Optional[str] = None,
        profile: Optional[str] = None,
        artifact_root: Optional[str] = None,
        component: Optional[str] = None,
    ) -> None:
        with self._lock:
            if run_id is not None:
                self.run_id = str(run_id)
            if mode is not None:
                self.mode = str(mode)
            if profile is not None:
                self.profile = str(profile)
            if artifact_root is not None:
                self.artifact_root = str(artifact_root)
            if component is not None:
                self.component = str(component)

    def _infer_level(self, kind: str, payload: Dict[str, Any], explicit_level: Optional[str]) -> str:
        if explicit_level:
            return str(explicit_level).upper()
        if kind in self._EVENT_LEVELS:
            level = self._EVENT_LEVELS[kind]
            if kind == "bundle_cleanup":
                status = str(payload.get("status", "")).lower()
                if status == "failed":
                    return "ERROR"
                return "INFO"
            if kind == "power_guard":
                status = str(payload.get("status", "")).lower()
                if status.startswith("failed"):
                    return "WARNING"
                return level
            return level
        if "error" in payload and payload.get("error"):
            return "ERROR"
        return "INFO"

    def _sanitize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        safe_payload = _log_safe_json(payload)
        redacted = _redact_log_object(safe_payload)
        serialized = json.dumps(redacted, ensure_ascii=False, sort_keys=True)
        if len(serialized) <= self.max_bytes:
            return redacted
        return {
            "_truncated": True,
            "approx_chars": len(serialized),
            "sha256": sha256_bytes(serialized.encode("utf-8")),
            "preview": serialized[:DEFAULT_LOG_PAYLOAD_PREVIEW_CHARS],
        }

    def _emit(self, row: Dict[str, Any], level: str) -> None:
        numeric_level = self._LEVEL_MAP.get(level, logging.INFO)
        self._logger.log(numeric_level, row["event"], extra={"row": row})
        for handler in self._logger.handlers:
            with contextlib.suppress(Exception):
                handler.flush()

    def write(
        self,
        kind: str,
        payload: Dict[str, Any],
        *,
        level: Optional[str] = None,
        component: Optional[str] = None,
    ) -> Dict[str, Any]:
        safe_payload = self._sanitize_payload(payload)
        resolved_level = self._infer_level(kind, safe_payload, level)
        row = {
            "ts_utc": utc_now(),
            "schema_version": LOG_SCHEMA_VERSION,
            "level": resolved_level,
            "component": str(component or self.component),
            "event": kind,
            "kind": kind,
            "run_id": self.run_id,
            "mode": self.mode,
            "profile": self.profile,
            "pid": self.pid,
            "host": self.hostname,
            "artifact_root": self.artifact_root,
            "payload": safe_payload,
        }
        with self._lock:
            self._event_count += 1
            self._events[kind] += 1
            self._levels[resolved_level] += 1
            self._emit(row, resolved_level)
        return row

    def write_exception(
        self,
        kind: str,
        exc: BaseException,
        *,
        component: Optional[str] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
        level: str = "CRITICAL",
    ) -> Dict[str, Any]:
        payload = dict(extra_payload or {})
        payload.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return self.write(kind, payload, level=level, component=component)

    def contract(self) -> Dict[str, Any]:
        return {
            "schema_version": LOG_SCHEMA_VERSION,
            "log_path": str(self.path),
            "console_level": self.console_level,
            "rotation": {
                "enabled": True,
                "max_bytes": self.max_bytes,
                "backup_count": self.backup_count,
            },
            "redaction_policy": {
                "enabled": True,
                "patterns": [
                    "hf_[REDACTED]",
                    "wandb_[REDACTED]",
                    "sk-[REDACTED]",
                    "archive_password",
                ],
            },
            "required_fields": [
                "ts_utc",
                "schema_version",
                "level",
                "component",
                "event",
                "run_id",
                "mode",
                "profile",
                "pid",
                "host",
                "artifact_root",
                "payload",
            ],
        }

    def observability_report(self) -> Dict[str, Any]:
        log_files = []
        for path in sorted(self.path.parent.glob(f"{self.path.name}*")):
            if not path.is_file():
                continue
            log_files.append(
                {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": path_sha256(path),
                }
            )
        return {
            "status": "ok",
            "schema_version": LOG_SCHEMA_VERSION,
            "run_id": self.run_id,
            "mode": self.mode,
            "profile": self.profile,
            "artifact_root": self.artifact_root,
            "event_count": self._event_count,
            "events": dict(sorted(self._events.items())),
            "levels": dict(sorted(self._levels.items())),
            "retention": {
                "max_bytes": self.max_bytes,
                "backup_count": self.backup_count,
            },
            "log_files": log_files,
        }

    def finalize(self, status: str, extra: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            if self._finalized:
                return
            self.write(
                "logger_finalize",
                {
                    "status": status,
                    "event_count": self._event_count,
                    "events": dict(sorted(self._events.items())),
                    "levels": dict(sorted(self._levels.items())),
                    **(extra or {}),
                },
                level="INFO" if status == "completed" else "WARNING",
            )
            self._finalized = True
            for handler in list(self._logger.handlers):
                with contextlib.suppress(Exception):
                    handler.flush()
                with contextlib.suppress(Exception):
                    handler.close()
                with contextlib.suppress(Exception):
                    self._logger.removeHandler(handler)

    def _atexit_finalize(self) -> None:
        with contextlib.suppress(Exception):
            if not self._finalized:
                if self._console_handler in self._logger.handlers:
                    self._logger.removeHandler(self._console_handler)
                self.finalize("abrupt_exit")


class _JSONLLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        row = getattr(record, "row", None)
        if not isinstance(row, dict):
            row = {
                "ts_utc": utc_now(),
                "schema_version": LOG_SCHEMA_VERSION,
                "level": record.levelname,
                "component": "chess_onefile",
                "event": record.getMessage(),
                "kind": record.getMessage(),
                "run_id": "",
                "mode": "",
                "profile": "",
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "artifact_root": "",
                "payload": {"message": record.getMessage()},
            }
        return json.dumps(row, ensure_ascii=False, sort_keys=True)


class _ConsoleLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        row = getattr(record, "row", None)
        if not isinstance(row, dict):
            return super().format(record)
        payload = row.get("payload", {})
        summary = ""
        if isinstance(payload, dict):
            if "status" in payload:
                summary = f" status={payload['status']}"
            elif "error_type" in payload:
                summary = f" error_type={payload['error_type']}"
            elif "step" in payload:
                summary = f" step={payload['step']}"
        return (
            f"[chess-log] {row.get('ts_utc', '')} "
            f"{row.get('level', 'INFO')} {row.get('component', 'chess_onefile')}:{row.get('event', '')}{summary}"
        )


class WindowsExecutionGuard:
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002

    def __init__(self, logger: Optional[JSONLLogger], enabled: bool = True):
        self.logger = logger
        self.enabled = bool(enabled and platform.system() == "Windows")
        self._restore_value = self.ES_CONTINUOUS

    def __enter__(self):
        if self.enabled:
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(
                    self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED | self.ES_DISPLAY_REQUIRED
                )
                if self.logger is not None:
                    self.logger.write("power_guard", {"status": "enabled", "platform": platform.system()})
            except Exception as exc:  # pragma: no cover - Windows API only
                if self.logger is not None:
                    self.logger.write("power_guard", {"status": "failed_enable", "error": str(exc)})
        else:
            if self.logger is not None:
                self.logger.write("power_guard", {"status": "noop", "platform": platform.system()})
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(self._restore_value)
                if self.logger is not None:
                    self.logger.write("power_guard", {"status": "restored"})
            except Exception as restore_exc:  # pragma: no cover - Windows API only
                if self.logger is not None:
                    self.logger.write("power_guard", {"status": "failed_restore", "error": str(restore_exc)})
        return False


def _module_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _install_allowed() -> bool:
    if os.environ.get(DEFAULT_SKIP_BOOTSTRAP_ENV, "0") == "1":
        return False
    if os.environ.get(DEFAULT_ALLOW_INSTALL_ENV, "0") == "1":
        return True
    return "--allow-install" in sys.argv


def _pip_install(args: Sequence[str]) -> None:
    commands = [
        [sys.executable, "-m", "pip", "install", "--user", "--break-system-packages", *args],
        [sys.executable, "-m", "pip", "install", "--user", *args],
        [sys.executable, "-m", "pip", "install", *args],
    ]
    last_error: Optional[subprocess.CalledProcessError] = None
    for cmd in commands:
        try:
            subprocess.check_call(cmd)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error


def _bootstrap_if_needed() -> None:
    if __name__ != "__main__":
        return
    if os.environ.get(DEFAULT_SKIP_BOOTSTRAP_ENV, "0") == "1":
        return
    missing: List[Tuple[str, List[str]]] = []
    if not _module_exists("torch"):
        torch_args = ["torch>=2.6,<3"]
        if platform.system() == "Windows":
            index_url = os.environ.get(DEFAULT_TORCH_INDEX_ENV, "https://download.pytorch.org/whl/cu128")
            torch_args += ["--index-url", index_url]
        missing.append(("torch", torch_args))
    for mod_name, package_args in (
        ("numpy", ["numpy>=1.24,<3"]),
        ("zstandard", ["zstandard>=0.21,<1"]),
        ("chess", ["python-chess>=1.999,<2"]),
        ("psutil", ["psutil>=5.9,<8"]),
    ):
        if not _module_exists(mod_name):
            missing.append((mod_name, package_args))
    if not missing:
        return
    if not _install_allowed():
        names = ", ".join(name for name, _ in missing)
        raise SystemExit(
            "Missing required packages: "
            f"{names}. Re-run with --allow-install or set {DEFAULT_ALLOW_INSTALL_ENV}=1."
        )
    if os.environ.get("MERTFORMER_CHESS_BOOTSTRAP_DONE", "0") == "1":
        raise SystemExit(
            "Required packages are still missing after bootstrap attempt: "
            + ", ".join(name for name, _ in missing)
        )
    for _, package_args in missing:
        _pip_install(package_args)
    env = os.environ.copy()
    env["MERTFORMER_CHESS_BOOTSTRAP_DONE"] = "1"
    os.execve(sys.executable, [sys.executable, __file__, *sys.argv[1:]], env)


_bootstrap_if_needed()

np: Any = None
torch: Any = None
nn: Any = None
F: Any = None
zstd: Any = None
chess: Any = None
psutil: Any = None
pyzipper: Any = None


def _import_runtime_dependencies() -> None:
    globals_ns = globals()
    globals_ns["np"] = importlib.import_module("numpy")
    globals_ns["torch"] = importlib.import_module("torch")
    globals_ns["nn"] = importlib.import_module("torch.nn")
    globals_ns["F"] = importlib.import_module("torch.nn.functional")
    globals_ns["zstd"] = importlib.import_module("zstandard")
    try:
        chess_mod = importlib.import_module("chess")
        importlib.import_module("chess.engine")
        importlib.import_module("chess.pgn")
        globals_ns["chess"] = chess_mod
    except Exception:  # pragma: no cover - import guarded by bootstrap in __main__
        globals_ns["chess"] = None
    try:
        globals_ns["psutil"] = importlib.import_module("psutil")
    except Exception:  # pragma: no cover
        globals_ns["psutil"] = None
    try:
        globals_ns["pyzipper"] = importlib.import_module("pyzipper")
    except Exception:  # pragma: no cover - optional runtime dependency
        globals_ns["pyzipper"] = None


_import_runtime_dependencies()


if chess is None:  # pragma: no cover
    raise SystemExit("python-chess is required; bootstrap did not complete successfully")


LAST_RUNTIME_CFG: Optional[Dict[str, Any]] = None
LAST_FINAL_ZIP: Optional[Path] = None
LAST_RUN_SUCCESS = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def deterministic_seed(seed: int, strict: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with contextlib.suppress(Exception):
        torch.backends.cudnn.deterministic = bool(strict)
        torch.backends.cudnn.benchmark = not bool(strict)
    with contextlib.suppress(Exception):
        torch.use_deterministic_algorithms(bool(strict), warn_only=True)
    if hasattr(torch.backends, "cuda"):
        with contextlib.suppress(Exception):
            torch.backends.cuda.matmul.allow_tf32 = not bool(strict)
            torch.backends.cudnn.allow_tf32 = not bool(strict)
    if strict:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def sha256_bytes(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    finally:
        if tmp.exists():
            with contextlib.suppress(Exception):
                tmp.unlink()


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _log_safe_json(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _log_safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_log_safe_json(v) for v in value]
    if isinstance(value, bytes):
        return {
            "__bytes__": True,
            "len": len(value),
            "sha256": sha256_bytes(value),
        }
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    if shape is not None and dtype is not None:
        with contextlib.suppress(Exception):
            return {
                "__tensor__": True,
                "shape": [int(x) for x in shape],
                "dtype": str(dtype),
                "device": str(device) if device is not None else "",
            }
    return repr(value)


def _redact_log_text(text: str) -> str:
    redacted = text
    redacted = re.sub(r"sk-[A-Za-z0-9_-]+", "sk-[REDACTED]", redacted)
    redacted = re.sub(r"hf_[A-Za-z0-9_-]+", "hf_[REDACTED]", redacted)
    redacted = re.sub(r"wandb_[A-Za-z0-9_-]+", "wandb_[REDACTED]", redacted)
    return redacted


def _redact_log_object(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _redact_log_text(value)
    if isinstance(value, list):
        return [_redact_log_object(item) for item in value]
    if isinstance(value, dict):
        redacted: Dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if any(token in key_str.lower() for token in ("password", "secret", "token", "api_key", "apikey")):
                redacted[key_str] = "[REDACTED]"
            else:
                redacted[key_str] = _redact_log_object(item)
        return redacted
    return _redact_log_text(str(value))


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "artifact"


def redact_path(value: str) -> str:
    resolved = str(Path(value).expanduser())
    home = str(Path.home())
    if resolved.startswith(home):
        return resolved.replace(home, "~", 1)
    return resolved


def detect_desktop_dir() -> Path:
    desktop = Path.home() / "Desktop"
    desktop.mkdir(parents=True, exist_ok=True)
    return desktop


def disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return round(float(usage.free) / (1024 ** 3), 3)


def get_package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def get_nvidia_driver_version() -> str:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return "unavailable"
    try:
        output = subprocess.check_output(
            [binary, "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
    except Exception:
        return "unavailable"
    return output.splitlines()[0].strip() if output else "unavailable"


def env_snapshot(cfg: Dict[str, Any]) -> Dict[str, Any]:
    script_path = Path(__file__).resolve()
    root = Path(str(cfg["artifact_root"]))
    snap: Dict[str, Any] = {
        "platform": platform.platform(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "python": sys.version,
        "python_executable": redact_path(sys.executable) if bool(cfg.get("redact_paths", True)) else sys.executable,
        "cwd": redact_path(os.getcwd()) if bool(cfg.get("redact_paths", True)) else os.getcwd(),
        "script": redact_path(str(script_path)) if bool(cfg.get("redact_paths", True)) else str(script_path),
        "script_sha256": path_sha256(script_path) if script_path.exists() else "",
        "artifact_root": redact_path(str(root)) if bool(cfg.get("redact_paths", True)) else str(root),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": str(torch.backends.cudnn.version()) if hasattr(torch.backends, "cudnn") else "unknown",
        "driver_version": get_nvidia_driver_version(),
        "share_mode": bool(cfg.get("share_mode", False)),
        "delivery_mode": bool(cfg.get("delivery_mode", False)),
        "allow_install": bool(cfg.get("allow_install", False)),
        "stockfish_auto_fetch": bool(cfg.get("stockfish_auto_fetch", True)),
        "self_delete_target_configured": bool(str(cfg.get("self_delete_target", "")).strip()),
        "determinism_strict": bool(cfg.get("determinism_strict", True)),
        "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)) if hasattr(torch.backends, "cudnn") else False,
        "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)) if hasattr(torch.backends, "cudnn") else False,
        "disk_free_gb": disk_free_gb(root),
    }
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            snap["ram_total_gb"] = round(float(vm.total) / (1024 ** 3), 3)
            snap["cpu_count_logical"] = int(psutil.cpu_count(logical=True) or 0)
        except Exception:
            pass
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        snap["cuda_name"] = props.name
        snap["cuda_total_gb"] = round(float(props.total_memory) / (1024 ** 3), 3)
        snap["cuda_capability"] = f"{props.major}.{props.minor}"
    return snap


def collect_dependency_lock() -> Dict[str, Any]:
    return {
        "python": sys.version,
        "packages": {
            "torch": get_package_version("torch"),
            "numpy": get_package_version("numpy"),
            "zstandard": get_package_version("zstandard"),
            "python-chess": get_package_version("python-chess"),
            "psutil": get_package_version("psutil"),
            "pyzipper": get_package_version("pyzipper"),
        },
    }


def validate_enum_choice(value: str, choices: Sequence[str], field_name: str) -> None:
    if value not in choices:
        raise ConfigValidationError(f"{field_name} must be one of {choices}, got {value!r}")


def parse_feature_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",")]
    elif isinstance(raw, (list, tuple, set)):
        items = [str(item).strip() for item in raw]
    else:
        raise ConfigValidationError(f"Feature list must be string/list-like, got {type(raw).__name__}")
    normalized = []
    seen = set()
    for item in items:
        if not item:
            continue
        if item not in FEATURE_FLAG_KEYS:
            raise ConfigValidationError(f"Unknown feature flag: {item}")
        if item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized


def apply_feature_bundle(cfg: Dict[str, Any], bundle_name: str) -> Dict[str, Any]:
    validate_enum_choice(bundle_name, list(FEATURE_BUNDLES.keys()), "feature_bundle")
    merged = dict(cfg)
    bundle = FEATURE_BUNDLES[bundle_name]
    merged.update(dict(bundle.get("overrides", {})))
    merged["feature_bundle"] = bundle_name
    return merged


def apply_feature_flag_overrides(
    cfg: Dict[str, Any],
    enabled_features: Sequence[str],
    disabled_features: Sequence[str],
) -> Dict[str, Any]:
    merged = dict(cfg)
    enable_list = parse_feature_list(enabled_features)
    disable_list = parse_feature_list(disabled_features)
    overlap = sorted(set(enable_list) & set(disable_list))
    if overlap:
        raise ConfigValidationError(f"Feature flags cannot be enabled and disabled together: {overlap}")
    for flag_name in enable_list:
        merged[flag_name] = True
    for flag_name in disable_list:
        merged[flag_name] = False
    merged["enabled_features"] = enable_list
    merged["disabled_features"] = disable_list
    return merged


def build_feature_flag_report(cfg: Dict[str, Any]) -> Dict[str, Any]:
    enabled = sorted(flag_name for flag_name in FEATURE_FLAG_KEYS if bool(cfg.get(flag_name, False)))
    disabled = sorted(flag_name for flag_name in FEATURE_FLAG_KEYS if not bool(cfg.get(flag_name, False)))
    bundle_name = str(cfg.get("feature_bundle", "default"))
    bundle_payload = FEATURE_BUNDLES.get(bundle_name, {"description": "unknown", "overrides": {}})
    explicit_enable = parse_feature_list(cfg.get("enabled_features", []))
    explicit_disable = parse_feature_list(cfg.get("disabled_features", []))
    return {
        "schema": "chess_feature_flag_report_v1",
        "feature_bundle": bundle_name,
        "bundle_description": str(bundle_payload.get("description", "")),
        "bundle_override_count": len(dict(bundle_payload.get("overrides", {}))),
        "explicitly_enabled": explicit_enable,
        "explicitly_disabled": explicit_disable,
        "enabled_count": len(enabled),
        "disabled_count": len(disabled),
        "enabled_features": enabled,
        "disabled_features": disabled,
    }


def render_feature_flag_report_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Chess Feature Flag Report",
        "",
        f"- feature_bundle: `{report.get('feature_bundle', 'default')}`",
        f"- bundle_description: `{report.get('bundle_description', '')}`",
        f"- enabled_count: `{report.get('enabled_count', 0)}`",
        f"- disabled_count: `{report.get('disabled_count', 0)}`",
        "",
        "## Explicit Overrides",
    ]
    enabled_explicit = report.get("explicitly_enabled", [])
    disabled_explicit = report.get("explicitly_disabled", [])
    lines.append(f"- enabled: `{', '.join(enabled_explicit) if enabled_explicit else 'none'}`")
    lines.append(f"- disabled: `{', '.join(disabled_explicit) if disabled_explicit else 'none'}`")
    lines.extend(["", "## Enabled Features"])
    for flag_name in report.get("enabled_features", []):
        lines.append(f"- `{flag_name}`")
    lines.extend(["", "## Disabled Features"])
    for flag_name in report.get("disabled_features", []):
        lines.append(f"- `{flag_name}`")
    return "\n".join(lines) + "\n"


def apply_profile(cfg: Dict[str, Any], profile: str) -> Dict[str, Any]:
    if profile not in RUN_PROFILES:
        raise ConfigValidationError(f"Unknown profile: {profile}")
    merged = dict(cfg)
    merged.update(RUN_PROFILES[profile])
    merged["profile"] = profile
    return merged


def apply_baseline(cfg: Dict[str, Any], baseline: str) -> Dict[str, Any]:
    validate_enum_choice(baseline, ["dense", "moe", "moe_adapter"], "baseline")
    merged = dict(cfg)
    merged["baseline"] = baseline
    if baseline == "dense":
        merged["use_moe"] = False
        merged["use_liquid"] = False
        merged["use_liquid_adapter"] = False
        merged["use_bitlinear"] = False
    elif baseline == "moe":
        merged["use_moe"] = True
        merged["use_liquid"] = False
        merged["use_liquid_adapter"] = False
    elif baseline == "moe_adapter":
        merged["use_moe"] = True
        merged["use_liquid"] = True
        merged["use_liquid_adapter"] = True
    else:
        merged["use_liquid"] = bool(merged.get("use_liquid", False))
    if "use_liquid" not in merged:
        merged["use_liquid"] = bool(merged.get("use_liquid_adapter", False))
    return merged


def resolve_runtime_config(args: argparse.Namespace, base_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(base_cfg or RUN_CONFIG)
    env_profile = os.environ.get(DEFAULT_PROFILE_ENV, "").strip()
    profile = str(getattr(args, "profile", cfg["profile"]))
    if env_profile and profile == str((base_cfg or RUN_CONFIG)["profile"]):
        profile = env_profile
    cfg = apply_profile(cfg, profile)
    baseline = str(getattr(args, "baseline", cfg["baseline"]))
    cfg = apply_baseline(cfg, baseline)

    cfg["mode"] = str(getattr(args, "mode", cfg["mode"]))
    validate_enum_choice(cfg["mode"], ["train", "verify", "benchmark", "package", "resume", "arena"], "mode")

    if getattr(args, "artifact_root", None):
        cfg["artifact_root"] = args.artifact_root
    if getattr(args, "stockfish_path", None):
        cfg["stockfish_path"] = args.stockfish_path
    if getattr(args, "resume_from", None):
        cfg["resume_from"] = args.resume_from
    if getattr(args, "self_delete_target", None):
        cfg["self_delete_target"] = args.self_delete_target
    if getattr(args, "max_steps", None) is not None:
        cfg["max_steps"] = int(args.max_steps)
    if getattr(args, "max_wall_hours", None) is not None:
        cfg["max_wall_hours"] = float(args.max_wall_hours)
    if getattr(args, "batch_size", None) is not None:
        cfg["batch_size"] = int(args.batch_size)
        cfg["eval_batch_size"] = int(args.batch_size)

    if getattr(args, "no_download", False):
        cfg["auto_download_enabled"] = False
    if getattr(args, "offline_seed_only", False):
        cfg["offline_seed_only"] = True
        cfg["auto_download_enabled"] = False
    if getattr(args, "test_mode", False) or os.environ.get(DEFAULT_TEST_MODE_ENV, "0") == "1":
        cfg = apply_profile(cfg, "smoke")
        cfg = apply_baseline(cfg, "dense")
        cfg["test_mode"] = True
        cfg["offline_seed_only"] = True
        cfg["auto_download_enabled"] = False
    if getattr(args, "allow_install", False) or os.environ.get(DEFAULT_ALLOW_INSTALL_ENV, "0") == "1":
        cfg["allow_install"] = True
    if getattr(args, "share_mode", False) or os.environ.get(DEFAULT_SHARE_MODE_ENV, "0") == "1":
        cfg["share_mode"] = True
    if getattr(args, "enable_self_delete", False) or os.environ.get(DEFAULT_SELF_DELETE_ENV, "0") == "1":
        cfg["enable_self_delete"] = True
    env_self_delete_target = os.environ.get(DEFAULT_SELF_DELETE_TARGET_ENV, "").strip()
    if env_self_delete_target:
        cfg["self_delete_target"] = env_self_delete_target
    if os.environ.get(DEFAULT_ENCRYPT_OUTPUT_ENV, "0") == "1":
        cfg["encrypt_output"] = True
    if os.environ.get(DEFAULT_ENCRYPTION_REQUIRED_ENV, "0") == "1":
        cfg["archive_encryption_required"] = True
    if os.environ.get(DEFAULT_SINGLE_OUTPUT_ENV, "0") == "1":
        cfg["single_output_only"] = True
    if os.environ.get(DEFAULT_CLEANUP_AFTER_BUNDLE_ENV, "0") == "1":
        cfg["cleanup_after_bundle"] = True
    env_artifact_root = os.environ.get(DEFAULT_ARTIFACT_ROOT_ENV, "").strip()
    if env_artifact_root:
        cfg["artifact_root"] = env_artifact_root
    env_cache_root = os.environ.get(DEFAULT_CACHE_ROOT_ENV, "").strip()
    if env_cache_root:
        cfg["cache_root"] = env_cache_root
    env_stockfish_cache_root = os.environ.get(DEFAULT_STOCKFISH_CACHE_ROOT_ENV, "").strip()
    if env_stockfish_cache_root:
        cfg["stockfish_cache_root"] = env_stockfish_cache_root
    env_stockfish_auto_fetch = os.environ.get(DEFAULT_STOCKFISH_AUTO_FETCH_ENV, "").strip()
    if env_stockfish_auto_fetch:
        cfg["stockfish_auto_fetch"] = env_stockfish_auto_fetch == "1"

    bundle_arg = getattr(args, "feature_bundle", None)
    bundle_name = str(cfg.get("feature_bundle", RUN_CONFIG.get("feature_bundle", "default")))
    if bundle_arg not in (None, "", RUN_CONFIG.get("feature_bundle", "default")):
        bundle_name = str(bundle_arg)
    cfg = apply_feature_bundle(cfg, bundle_name)
    enable_features_arg = getattr(args, "enable_features", None)
    disable_features_arg = getattr(args, "disable_features", None)
    cfg = apply_feature_flag_overrides(
        cfg,
        cfg.get("enabled_features", []) if enable_features_arg in (None, "") else enable_features_arg,
        cfg.get("disabled_features", []) if disable_features_arg in (None, "") else disable_features_arg,
    )

    cfg["artifact_root"] = str(Path(str(cfg["artifact_root"])).expanduser())
    cfg["cache_root"] = str(Path(str(cfg["cache_root"])).expanduser())
    stockfish_cache_root = str(cfg.get("stockfish_cache_root", "")).strip()
    if not stockfish_cache_root:
        cfg["stockfish_cache_root"] = str(Path(str(cfg["cache_root"])) / "stockfish")
    else:
        cfg["stockfish_cache_root"] = str(Path(stockfish_cache_root).expanduser())
    cfg["resume_from"] = str(Path(str(cfg.get("resume_from", ""))).expanduser()) if str(cfg.get("resume_from", "")) else ""
    cfg["self_delete_target"] = str(Path(str(cfg.get("self_delete_target", ""))).expanduser()) if str(cfg.get("self_delete_target", "")) else ""
    cfg["use_liquid"] = bool(cfg.get("use_liquid", cfg.get("use_liquid_adapter", False)))
    cfg["use_liquid_adapter"] = bool(cfg.get("use_liquid_adapter", cfg.get("use_liquid", False)))

    if str(cfg.get("device", "auto")) == "auto":
        if torch.cuda.is_available():
            cfg["device"] = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            cfg["device"] = "mps"
        else:
            cfg["device"] = "cpu"

    if str(cfg["device"]) != "cuda":
        cfg["use_bf16"] = False
        cfg["compile_policy"] = "off"
        cfg["batch_size"] = min(int(cfg["batch_size"]), 32)
        cfg["eval_batch_size"] = min(int(cfg["eval_batch_size"]), 32)
    if str(cfg["device"]) == "mps":
        cfg["num_workers"] = 0
    if str(cfg["mode"]) == "verify":
        cfg["auto_download_enabled"] = False
        cfg["offline_seed_only"] = True

    validate_runtime_config(cfg)
    return cfg


def validate_runtime_config(cfg: Dict[str, Any]) -> None:
    validate_enum_choice(str(cfg["mode"]), ["train", "verify", "benchmark", "package", "resume", "arena"], "mode")
    validate_enum_choice(str(cfg["profile"]), list(RUN_PROFILES.keys()), "profile")
    validate_enum_choice(str(cfg["baseline"]), ["dense", "moe", "moe_adapter"], "baseline")
    validate_enum_choice(str(cfg.get("feature_bundle", "default")), list(FEATURE_BUNDLES.keys()), "feature_bundle")
    parse_feature_list(cfg.get("enabled_features", []))
    parse_feature_list(cfg.get("disabled_features", []))
    if float(cfg["val_fraction"]) < 0 or float(cfg["test_fraction"]) < 0:
        raise ConfigValidationError("Validation/test fractions must be non-negative")
    if float(cfg["val_fraction"]) + float(cfg["test_fraction"]) >= 1.0:
        raise ConfigValidationError("Validation + test fractions must be < 1.0")
    for field_name in (
        "seed",
        "download_partial_mb",
        "download_archive_count",
        "max_games",
        "max_positions",
        "max_positions_per_game",
        "min_elo",
        "time_control_min_seconds",
        "time_control_max_seconds",
        "max_steps",
        "batch_size",
        "eval_batch_size",
        "stockfish_download_timeout_sec",
        "search_candidate_topk",
        "search_reply_topk",
        "midrun_curated_snapshot_interval",
        "midrun_stockfish_snapshot_interval",
        "midrun_stockfish_snapshot_games",
        "hidden_size",
        "num_layers",
        "num_heads",
        "num_experts",
        "checkpoint_interval",
        "eval_interval",
        "grad_accum_steps",
    ):
        if int(cfg[field_name]) < 0:
            raise ConfigValidationError(f"{field_name} must be >= 0")
    if int(cfg["hidden_size"]) % max(1, int(cfg["num_heads"])) != 0:
        raise ConfigValidationError("hidden_size must be divisible by num_heads")
    num_kv_heads = max(1, int(cfg.get("num_kv_heads", cfg["num_heads"])))
    if num_kv_heads > int(cfg["num_heads"]) or int(cfg["num_heads"]) % num_kv_heads != 0:
        raise ConfigValidationError("num_kv_heads must be <= num_heads and divide num_heads evenly")
    if int(cfg["batch_size"]) < 1 or int(cfg["eval_batch_size"]) < 1:
        raise ConfigValidationError("batch sizes must be >= 1")
    if int(cfg["grad_accum_steps"]) < 1:
        raise ConfigValidationError("grad_accum_steps must be >= 1")
    if float(cfg["max_wall_hours"]) <= 0:
        raise ConfigValidationError("max_wall_hours must be > 0")
    if float(cfg["learning_rate"]) <= 0:
        raise ConfigValidationError("learning_rate must be > 0")
    if bool(cfg.get("enable_self_delete", False)) and not bool(cfg.get("share_mode", False)):
        raise ConfigValidationError("enable_self_delete requires share_mode")
    if bool(cfg.get("enable_self_delete", False)):
        target_value = str(cfg.get("self_delete_target", "")).strip()
        if not target_value:
            raise ConfigValidationError("enable_self_delete requires self_delete_target")
        target_path = Path(target_value).expanduser().resolve()
        if target_path == Path(__file__).resolve():
            raise ConfigValidationError("self_delete_target must not point to the canonical repo script")
    if bool(cfg.get("cleanup_after_bundle", False)) and not bool(cfg.get("zip_outputs", True)):
        raise ConfigValidationError("cleanup_after_bundle requires zip_outputs=True")
    if str(cfg["mode"]) in {"resume", "benchmark", "package"} and not str(cfg.get("resume_from", "")).strip():
        raise ConfigValidationError(f"mode={cfg['mode']} requires --resume-from")
    if str(cfg["mode"]) == "verify" and int(cfg["max_steps"]) == 0:
        return


@dataclass(frozen=True)
class MirrorModelConfig:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_hidden_layers: int
    num_heads: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    dropout: float
    attention_dropout: float
    ffn_dropout: float
    rms_norm_eps: float
    use_bitnet: bool
    use_moe: bool
    num_experts: int
    num_experts_per_tok: int
    active_experts: int
    moe_every_n_layers: int
    moe_intermediate: int
    router_temperature: float
    router_jitter: float
    router_jitter_boost: float
    router_alarm_threshold: float
    shared_expert_gate: float
    z_loss_coef: float
    use_switch_loss: bool
    moe_capacity_enforce: bool
    moe_capacity_factor: float
    moe_dispatch_mode: str
    use_expert_paging: bool
    expert_paging_inference_only: bool
    expert_paging_lazy_init: bool
    expert_paging_cache_size: int
    expert_paging_offload_device: str
    expert_paging_verbose: bool
    use_liquid: bool
    liquid_layers_idx: Tuple[int, ...]
    liquid_every_n_layers: int
    liquid_fast_path: bool
    use_qinn: bool
    qinn_every_n_layers: int
    rope_theta: float
    rope_base: float
    rope_dim: Optional[int]
    use_flash_attn_inference: bool
    use_hierarchical_kv_cache: bool
    hkv_short_window: int
    hkv_long_stride: int
    hkv_max_long_blocks: int
    use_global_workspace_broadcast: bool
    workspace_blend: float
    use_neuromodulatory_gain: bool
    use_latent_ode_state_channel: bool
    latent_ode_dt: float
    use_cross_expert_sync_bus: bool
    cross_expert_sync_gain: float
    use_structural_plasticity: bool
    structural_ema_decay: float
    structural_prune_threshold: float
    structural_grow_threshold: float
    structural_update_interval: int
    use_hebbian_plasticity: bool
    hebbian_eta: float
    hebbian_decay: float
    use_neuro_symbolic_layer: bool
    neuro_symbolic_rules: int
    use_world_model_head: bool
    world_model_horizon: int
    use_phase_head: bool
    phase_loss_coef: float
    use_wdl_head: bool
    wdl_loss_coef: float
    wdl_draw_threshold: float
    use_legality_head: bool
    legality_loss_coef: float
    legality_pos_weight_cap: float
    use_lifelong_safety_layer: bool
    lifelong_ema_decay: float
    lifelong_max_adaptation_gain: float
    lifelong_drift_threshold: float
    use_gradient_checkpointing: bool


def _normalize_liquid_layers_idx(indices: Sequence[int], num_layers: int) -> Tuple[int, ...]:
    if num_layers <= 0:
        return tuple()
    cleaned = sorted({idx for idx in (int(item) for item in indices) if 0 <= idx < num_layers})
    return tuple(cleaned)


def default_liquid_layers_idx(num_layers: int) -> Tuple[int, ...]:
    if num_layers <= 0:
        return tuple()
    template = (4.0 / 18.0, 10.0 / 18.0, 16.0 / 18.0)
    derived = [int(round((num_layers - 1) * ratio)) for ratio in template]
    return _normalize_liquid_layers_idx(derived, num_layers)


def build_mirror_model_config(run_cfg: Dict[str, Any]) -> MirrorModelConfig:
    hidden_size = max(1, int(run_cfg["hidden_size"]))
    num_layers = max(1, int(run_cfg["num_layers"]))
    num_heads = max(1, int(run_cfg["num_heads"]))
    derived_head_dim = max(1, hidden_size // num_heads)
    head_dim = int(run_cfg.get("head_dim", derived_head_dim) or derived_head_dim)
    if head_dim * num_heads != hidden_size:
        head_dim = derived_head_dim
    num_kv_heads = int(run_cfg.get("num_kv_heads", num_heads) or num_heads)
    if num_kv_heads <= 0 or num_kv_heads > num_heads or num_heads % num_kv_heads != 0:
        num_kv_heads = num_heads
    intermediate_size = max(1, int(run_cfg.get("intermediate_size", hidden_size * 4)))
    moe_intermediate = max(1, int(run_cfg.get("moe_intermediate", intermediate_size)))
    num_experts = max(1, int(run_cfg.get("num_experts", 4)))
    active_experts = max(1, min(num_experts, int(run_cfg.get("moe_top_k", 2))))
    use_liquid = bool(run_cfg.get("use_liquid", run_cfg.get("use_liquid_adapter", False)))
    raw_liquid_layers = run_cfg.get("liquid_layers_idx", [])
    if raw_liquid_layers:
        liquid_layers_idx = _normalize_liquid_layers_idx(raw_liquid_layers, num_layers)
    elif use_liquid:
        liquid_layers_idx = default_liquid_layers_idx(num_layers)
    else:
        liquid_layers_idx = tuple()
    return MirrorModelConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        num_hidden_layers=num_layers,
        num_heads=num_heads,
        num_attention_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=max(80, int(run_cfg.get("max_seq_len", 80))),
        dropout=float(run_cfg.get("dropout", 0.0)),
        attention_dropout=float(run_cfg.get("attention_dropout", 0.0)),
        ffn_dropout=float(run_cfg.get("ffn_dropout", 0.0)),
        rms_norm_eps=float(run_cfg.get("rms_norm_eps", 1e-6)),
        use_bitnet=bool(run_cfg.get("use_bitlinear", False)),
        use_moe=bool(run_cfg.get("use_moe", False)),
        num_experts=num_experts,
        num_experts_per_tok=active_experts,
        active_experts=active_experts,
        moe_every_n_layers=max(0, int(run_cfg.get("moe_every_n_layers", 3))),
        moe_intermediate=moe_intermediate,
        router_temperature=float(run_cfg.get("router_temperature", 1.0)),
        router_jitter=float(run_cfg.get("router_jitter", 0.02)),
        router_jitter_boost=float(run_cfg.get("router_jitter_boost", 0.10)),
        router_alarm_threshold=float(run_cfg.get("router_alarm_threshold", 0.40)),
        shared_expert_gate=float(run_cfg.get("shared_expert_gate", 0.0)),
        z_loss_coef=float(run_cfg.get("z_loss_coef", 1e-4)),
        use_switch_loss=bool(run_cfg.get("use_switch_loss", True)),
        moe_capacity_enforce=bool(run_cfg.get("moe_capacity_enforce", True)),
        moe_capacity_factor=float(run_cfg.get("moe_capacity_factor", 1.25)),
        moe_dispatch_mode=str(run_cfg.get("moe_dispatch_mode", "sequential")).lower(),
        use_expert_paging=bool(run_cfg.get("use_expert_paging", False)),
        expert_paging_inference_only=bool(run_cfg.get("expert_paging_inference_only", True)),
        expert_paging_lazy_init=bool(run_cfg.get("expert_paging_lazy_init", True)),
        expert_paging_cache_size=max(1, int(run_cfg.get("expert_paging_cache_size", active_experts))),
        expert_paging_offload_device=str(run_cfg.get("expert_paging_offload_device", "cpu")),
        expert_paging_verbose=bool(run_cfg.get("expert_paging_verbose", False)),
        use_liquid=use_liquid,
        liquid_layers_idx=liquid_layers_idx,
        liquid_every_n_layers=max(0, int(run_cfg.get("liquid_every_n_layers", 0))),
        liquid_fast_path=bool(run_cfg.get("liquid_fast_path", True)),
        use_qinn=bool(run_cfg.get("use_qinn", False)),
        qinn_every_n_layers=max(1, int(run_cfg.get("qinn_every_n_layers", 1))),
        rope_theta=float(run_cfg.get("rope_theta", 100000.0)),
        rope_base=float(run_cfg.get("rope_base", 100000.0)),
        rope_dim=run_cfg.get("rope_dim", None),
        use_flash_attn_inference=bool(run_cfg.get("use_flash_attn_inference", False)),
        use_hierarchical_kv_cache=bool(run_cfg.get("use_hierarchical_kv_cache", False)),
        hkv_short_window=max(1, int(run_cfg.get("hkv_short_window", 512))),
        hkv_long_stride=max(1, int(run_cfg.get("hkv_long_stride", 8))),
        hkv_max_long_blocks=max(1, int(run_cfg.get("hkv_max_long_blocks", 128))),
        use_global_workspace_broadcast=bool(run_cfg.get("use_global_workspace_broadcast", False)),
        workspace_blend=float(run_cfg.get("workspace_blend", 0.7)),
        use_neuromodulatory_gain=bool(run_cfg.get("use_neuromodulatory_gain", False)),
        use_latent_ode_state_channel=bool(run_cfg.get("use_latent_ode_state_channel", False)),
        latent_ode_dt=float(run_cfg.get("latent_ode_dt", 1.0)),
        use_cross_expert_sync_bus=bool(run_cfg.get("use_cross_expert_sync_bus", False)),
        cross_expert_sync_gain=float(run_cfg.get("cross_expert_sync_gain", 0.05)),
        use_structural_plasticity=bool(run_cfg.get("use_structural_plasticity", False)),
        structural_ema_decay=float(run_cfg.get("structural_ema_decay", 0.98)),
        structural_prune_threshold=float(run_cfg.get("structural_prune_threshold", 0.02)),
        structural_grow_threshold=float(run_cfg.get("structural_grow_threshold", 0.60)),
        structural_update_interval=max(1, int(run_cfg.get("structural_update_interval", 100))),
        use_hebbian_plasticity=bool(run_cfg.get("use_hebbian_plasticity", False)),
        hebbian_eta=float(run_cfg.get("hebbian_eta", 0.01)),
        hebbian_decay=float(run_cfg.get("hebbian_decay", 0.99)),
        use_neuro_symbolic_layer=bool(run_cfg.get("use_neuro_symbolic_layer", False)),
        neuro_symbolic_rules=max(1, int(run_cfg.get("neuro_symbolic_rules", 8))),
        use_world_model_head=bool(run_cfg.get("use_world_model_head", False)),
        world_model_horizon=max(1, int(run_cfg.get("world_model_horizon", 1))),
        use_phase_head=bool(run_cfg.get("use_phase_head", False)),
        phase_loss_coef=float(run_cfg.get("phase_loss_coef", 0.05)),
        use_wdl_head=bool(run_cfg.get("use_wdl_head", False)),
        wdl_loss_coef=float(run_cfg.get("wdl_loss_coef", 0.08)),
        wdl_draw_threshold=max(0.0, float(run_cfg.get("wdl_draw_threshold", 0.20))),
        use_legality_head=bool(run_cfg.get("use_legality_head", False)),
        legality_loss_coef=float(run_cfg.get("legality_loss_coef", 0.03)),
        legality_pos_weight_cap=max(1.0, float(run_cfg.get("legality_pos_weight_cap", 64.0))),
        use_lifelong_safety_layer=bool(run_cfg.get("use_lifelong_safety_layer", False)),
        lifelong_ema_decay=float(run_cfg.get("lifelong_ema_decay", 0.99)),
        lifelong_max_adaptation_gain=float(run_cfg.get("lifelong_max_adaptation_gain", 0.05)),
        lifelong_drift_threshold=float(run_cfg.get("lifelong_drift_threshold", 0.35)),
        use_gradient_checkpointing=bool(run_cfg.get("use_gradient_checkpointing", False)),
    )


_LOWBIT_KERNEL_ENABLED = os.getenv("MERTFORMER_LOWBIT_KERNEL", "0") == "1"
_TENSORCORE_ENABLED = os.getenv("MERTFORMER_TENSORCORE", "0") == "1"


def set_lowbit_kernel_enabled(enabled: bool) -> None:
    global _LOWBIT_KERNEL_ENABLED
    _LOWBIT_KERNEL_ENABLED = bool(enabled)


def _import_optional_sdk_module(module_name: str) -> Optional[Any]:
    if str(REPO_ROOT) not in sys.path and (REPO_ROOT / "mertformer_sdk").is_dir():
        sys.path.insert(0, str(REPO_ROOT))
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _try_lowbit_kernel(x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if not _LOWBIT_KERNEL_ENABLED:
        return None
    dispatcher_mod = _import_optional_sdk_module("mertformer_sdk.kernels.dispatcher")
    select_backend = getattr(dispatcher_mod, "select_backend", None) if dispatcher_mod is not None else None
    if select_backend is None:
        return None
    try:
        backend = select_backend(x, w)
        if backend == "triton_cuda":
            triton_mod = _import_optional_sdk_module("mertformer_sdk.kernels.triton_ternary")
            is_triton_available = getattr(triton_mod, "is_triton_available", None) if triton_mod is not None else None
            triton_ternary_linear = getattr(triton_mod, "triton_ternary_linear", None) if triton_mod is not None else None
            if is_triton_available is None or triton_ternary_linear is None or not is_triton_available():
                return None
            return triton_ternary_linear(x, w, bias, use_tensorcore=_TENSORCORE_ENABLED)
        if backend == "cpp_cpu":
            cpp_mod = _import_optional_sdk_module("mertformer_sdk.kernels.cpp.loader")
            bitnet_cpu_linear = getattr(cpp_mod, "bitnet_cpu_linear", None) if cpp_mod is not None else None
            if bitnet_cpu_linear is None:
                return None
            return bitnet_cpu_linear(x, w, bias)
        if backend == "metal_fallback":
            metal_mod = _import_optional_sdk_module("mertformer_sdk.kernels.metal.engine")
            metal_linear = getattr(metal_mod, "metal_linear", None) if metal_mod is not None else None
            if metal_linear is None:
                return None
            return metal_linear(activation_quant(x), weight_quant(w), bias)
        if backend == "vulkan_fallback":
            vulkan_mod = _import_optional_sdk_module("mertformer_sdk.kernels.vulkan.engine")
            vulkan_linear = getattr(vulkan_mod, "vulkan_linear", None) if vulkan_mod is not None else None
            if vulkan_linear is None:
                return None
            return vulkan_linear(activation_quant(x), weight_quant(w), bias)
        if backend == "npu_fallback":
            npu_mod = _import_optional_sdk_module("mertformer_sdk.kernels.npu.engine")
            npu_linear = getattr(npu_mod, "npu_linear", None) if npu_mod is not None else None
            if npu_linear is None:
                return None
            return npu_linear(activation_quant(x), weight_quant(w), bias)
        if backend == "mps_optimized":
            return F.linear(activation_quant(x), weight_quant(w), bias)
    except Exception:
        return None
    return None


def activation_quant(x: torch.Tensor) -> torch.Tensor:
    max_abs = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    scale = 127.0 / max_abs
    x_q = torch.round(x * scale).clamp(-127, 127) / scale
    return x + (x_q - x).detach()


def weight_quant(w: torch.Tensor) -> torch.Tensor:
    scale = torch.sqrt((w ** 2).mean(dim=1, keepdim=True)).clamp(min=1e-5)
    w_norm = w / scale
    w_q = torch.round(w_norm).clamp(-1.0, 1.0)
    w_q_real = w_q * scale
    return w + (w_q_real - w).detach()


def make_linear(use_bitnet: bool, in_features: int, out_features: int, bias: bool = True) -> nn.Module:
    if use_bitnet:
        return BitLinear(in_features, out_features, bias=bias, enabled=True)
    return nn.Linear(in_features, out_features, bias=bias)


class BitLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, enabled: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        self.enabled = bool(enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return F.linear(x, self.weight, self.bias)
        lowbit_out = _try_lowbit_kernel(x, self.weight, self.bias)
        if lowbit_out is not None:
            return lowbit_out
        return F.linear(activation_quant(x), weight_quant(self.weight), self.bias)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight


class _QKRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms.to(x.dtype) * self.weight.to(x.dtype)


try:
    from flash_attn import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


def _is_onnx_export() -> bool:
    onnx_mod = getattr(torch, "onnx", None)
    if onnx_mod is None:
        return False
    check_fn = getattr(onnx_mod, "is_in_onnx_export", None)
    if check_fn is None:
        return False
    try:
        return bool(check_fn())
    except Exception:
        return False


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 100000.0, device: Optional[torch.device] = None):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}")
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._update_cache(max_seq_len, device if device is not None else inv_freq.device)

    @torch.no_grad()
    def _update_cache(self, seq_len: int, device: Optional[torch.device]) -> None:
        seq_len = int(seq_len)
        if device is None:
            device = self.inv_freq.device
        self.max_seq_len = max(seq_len, self.max_seq_len)
        t = torch.arange(self.max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        self._buffers["cos_cached"] = cos
        self._buffers["sin_cached"] = sin

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[2]
        total_len = int(seq_len) + int(offset)
        if (
            self.cos_cached.numel() == 0
            or total_len > self.cos_cached.shape[2]
            or self.cos_cached.device != x.device
            or self.cos_cached.dtype != self.inv_freq.dtype
        ):
            self._update_cache(total_len, x.device)
        return (
            self.cos_cached[..., offset:total_len, :].to(dtype=x.dtype),
            self.sin_cached[..., offset:total_len, :].to(dtype=x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_optimized(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MLA(nn.Module):
    def __init__(self, arch_cfg: MirrorModelConfig) -> None:
        super().__init__()
        self.hidden_size = arch_cfg.hidden_size
        self.num_heads = arch_cfg.num_heads
        self.head_dim = arch_cfg.head_dim
        self.num_kv_heads = arch_cfg.num_kv_heads
        self.rope_theta = arch_cfg.rope_theta
        if self.num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be >= 1, got {self.num_kv_heads}")
        if self.num_kv_heads > self.num_heads:
            raise ValueError(
                f"num_kv_heads ({self.num_kv_heads}) must be <= num_heads ({self.num_heads}) for GQA."
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads}) for GQA."
            )
        self.rope_dim = arch_cfg.rope_dim
        self.rope_base = arch_cfg.rope_base
        rope_dim_eff = self.head_dim if self.rope_dim is None else int(self.rope_dim)
        if rope_dim_eff <= 0 or rope_dim_eff > self.head_dim:
            raise ValueError(
                f"rope_dim must be in (0, head_dim], got rope_dim={rope_dim_eff}, head_dim={self.head_dim}"
            )
        if rope_dim_eff % 2 != 0:
            raise ValueError(f"rope_dim must be even, got {rope_dim_eff}")
        self._rope_dim_eff = rope_dim_eff
        self.rotary_emb = RotaryEmbedding(
            dim=self._rope_dim_eff,
            max_seq_len=arch_cfg.max_seq_len,
            base=self.rope_base,
        )
        self.q_proj = make_linear(arch_cfg.use_bitnet, self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = make_linear(arch_cfg.use_bitnet, self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = make_linear(arch_cfg.use_bitnet, self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = make_linear(arch_cfg.use_bitnet, self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = _QKRMSNorm(self.head_dim)
        self.k_norm = _QKRMSNorm(self.head_dim)
        self.attn_dropout = nn.Dropout(arch_cfg.attention_dropout)
        self.use_flash_attn_inference = bool(arch_cfg.use_flash_attn_inference)
        self.max_seq = int(arch_cfg.max_seq_len)
        self.use_hierarchical_kv_cache = bool(arch_cfg.use_hierarchical_kv_cache)
        self.hkv_short_window = int(arch_cfg.hkv_short_window)
        self.hkv_long_stride = int(arch_cfg.hkv_long_stride)
        self.hkv_max_long_blocks = int(arch_cfg.hkv_max_long_blocks)

    def _pool_long_kv(self, tensor: torch.Tensor, stride: int, max_blocks: int) -> torch.Tensor:
        bsz, hk, slen, dim = tensor.shape
        if slen == 0:
            return tensor.new_zeros((bsz, hk, 0, dim))
        stride = max(1, int(stride))
        blocks = slen // stride
        pooled = tensor.new_zeros((bsz, hk, 0, dim))
        if blocks > 0:
            trimmed = tensor[:, :, : blocks * stride, :]
            pooled = trimmed.reshape(bsz, hk, blocks, stride, dim).mean(dim=3)
        rem = slen - blocks * stride
        if rem > 0:
            rem_chunk = tensor[:, :, blocks * stride :, :].mean(dim=2, keepdim=True)
            pooled = torch.cat([pooled, rem_chunk], dim=2)
        if max_blocks > 0 and pooled.size(2) > max_blocks:
            pooled = pooled[:, :, -max_blocks:, :]
        return pooled

    def _build_hierarchical_kv(self, k_full: torch.Tensor, v_full: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        total_len = k_full.size(2)
        short_window = max(1, min(self.hkv_short_window, total_len))
        long_len = total_len - short_window
        if long_len <= 0:
            return k_full, v_full
        k_long = k_full[:, :, :long_len, :]
        v_long = v_full[:, :, :long_len, :]
        k_short = k_full[:, :, long_len:, :]
        v_short = v_full[:, :, long_len:, :]
        k_long_pooled = self._pool_long_kv(k_long, self.hkv_long_stride, self.hkv_max_long_blocks)
        v_long_pooled = self._pool_long_kv(v_long, self.hkv_long_stride, self.hkv_max_long_blocks)
        return torch.cat([k_long_pooled, k_short], dim=2), torch.cat([v_long_pooled, v_short], dim=2)

    def forward(
        self,
        x: torch.Tensor,
        decoupled_rope: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch, seq_len, hidden = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        kv_seq_len = seq_len
        if past_key_value is not None:
            past_k, past_v = past_key_value
            kv_seq_len = past_k.shape[2] + seq_len
        if kv_seq_len > self.max_seq:
            raise ValueError(
                f"kv_seq_len ({kv_seq_len}) exceeds max_seq ({self.max_seq}). Increase cfg.max_seq_len."
            )
        cos, sin = self.rotary_emb(q, seq_len=seq_len, offset=kv_seq_len - seq_len)
        rope_dim = self._rope_dim_eff
        if decoupled_rope:
            q_rope = q[..., -rope_dim:]
            k_rope = k[..., -rope_dim:]
            q_rope, k_rope = apply_rope_optimized(q_rope, k_rope, cos, sin)
            if rope_dim < self.head_dim:
                q = torch.cat([q[..., :-rope_dim], q_rope], dim=-1)
                k = torch.cat([k[..., :-rope_dim], k_rope], dim=-1)
            else:
                q, k = q_rope, k_rope
        else:
            if rope_dim < self.head_dim:
                q_rope = q[..., :rope_dim]
                k_rope = k[..., :rope_dim]
                q_rope, k_rope = apply_rope_optimized(q_rope, k_rope, cos, sin)
                q = torch.cat([q_rope, q[..., rope_dim:]], dim=-1)
                k = torch.cat([k_rope, k[..., rope_dim:]], dim=-1)
            else:
                q, k = apply_rope_optimized(q, k, cos, sin)
        k_full, v_full = k, v
        if past_key_value is not None:
            k_full = torch.cat([past_k, k], dim=2)
            v_full = torch.cat([past_v, v], dim=2)
        present_key_value = (k_full, v_full) if use_cache else None
        if self.use_hierarchical_kv_cache and past_key_value is not None and seq_len == 1:
            k, v = self._build_hierarchical_kv(k_full, v_full)
        else:
            k, v = k_full, v_full
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        use_flash = (
            FLASH_ATTN_AVAILABLE
            and q.is_cuda
            and past_key_value is None
            and (self.training or self.use_flash_attn_inference)
            and not _is_onnx_export()
        )
        if use_flash:
            q_flash = q.transpose(1, 2).contiguous()
            k_flash = k.transpose(1, 2).contiguous()
            v_flash = v.transpose(1, 2).contiguous()
            out = flash_attn_func(
                q_flash,
                k_flash,
                v_flash,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                causal=True,
                softmax_scale=1.0 / math.sqrt(self.head_dim),
            ).transpose(1, 2)
        elif hasattr(F, "scaled_dot_product_attention") and not _is_onnx_export():
            dropout_p = self.attn_dropout.p if self.training else 0.0
            if past_key_value is None:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True)
            elif seq_len == 1:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=False)
            else:
                q_pos = torch.arange(kv_seq_len - seq_len, kv_seq_len, device=x.device)
                k_pos = torch.arange(kv_seq_len, device=x.device)
                causal_mask = q_pos[:, None] >= k_pos[None, :]
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask, dropout_p=dropout_p, is_causal=False)
        else:
            q_pos = torch.arange(kv_seq_len - seq_len, kv_seq_len, device=x.device)
            k_pos = torch.arange(kv_seq_len, device=x.device)
            causal_mask = q_pos[:, None] >= k_pos[None, :]
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            scores = scores.float()
            scores.masked_fill_(~causal_mask, float("-inf"))
            attn_weights = self.attn_dropout(F.softmax(scores, dim=-1).to(x.dtype))
            out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        return self.o_proj(out), present_key_value


class MertFormerFFN(nn.Module):
    def __init__(self, arch_cfg: MirrorModelConfig) -> None:
        super().__init__()
        hidden_size = arch_cfg.hidden_size
        intermediate_size = arch_cfg.intermediate_size
        self.gate_proj = make_linear(arch_cfg.use_bitnet, hidden_size, intermediate_size, bias=False)
        self.up_proj = make_linear(arch_cfg.use_bitnet, hidden_size, intermediate_size, bias=False)
        self.down_proj = make_linear(arch_cfg.use_bitnet, intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(arch_cfg.ffn_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_inter = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x_inter = self.dropout(x_inter)
        return self.down_proj(x_inter)


def _jit_script_if_supported(fn):
    if sys.version_info >= (3, 14):
        return fn
    try:
        return torch.jit.script(fn)
    except Exception as exc:
        warnings.warn(f"TorchScript disabled for liquid kernel due to: {exc}", RuntimeWarning)
        return fn


class LiquidCell(nn.Module):
    def __init__(self, h: int, use_bitnet: bool) -> None:
        super().__init__()
        self.input_w = make_linear(use_bitnet, h, h, bias=False)
        self.hidden_w = make_linear(use_bitnet, h, h, bias=False)
        self.tau_input_w = make_linear(use_bitnet, h, h, bias=False)
        self.tau_hidden_w = make_linear(use_bitnet, h, h, bias=False)
        self.tau_bias = nn.Parameter(torch.ones(1, h) * 0.5)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        val_in = self.input_w(x)
        val_rec = self.hidden_w(h_prev)
        state_target = torch.tanh(val_in + val_rec)
        tau_in = self.tau_input_w(x)
        tau_rec = self.tau_hidden_w(h_prev)
        time_decay = F.softplus(tau_in + tau_rec + self.tau_bias)
        decay = torch.exp(torch.clamp(-time_decay * dt, min=-20.0, max=20.0))
        return state_target + (h_prev - state_target) * decay


@_jit_script_if_supported
def jit_quant(w: torch.Tensor) -> torch.Tensor:
    w_f = w.float()
    scale = torch.sqrt((w_f * w_f).mean(dim=1, keepdim=True)).clamp(min=1e-5)
    w_q = torch.round(w_f / scale).clamp(-1.0, 1.0)
    return (w_q * scale).to(dtype=w.dtype)


@_jit_script_if_supported
def jit_liquid_loop_cached(
    input_seq: torch.Tensor,
    h_init: torch.Tensor,
    dt: float,
    input_w_q_t: torch.Tensor,
    hidden_w_q_t: torch.Tensor,
    tau_input_w_q_t: torch.Tensor,
    tau_hidden_w_q_t: torch.Tensor,
    tau_bias: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, hidden = input_seq.shape
    h = h_init
    out_seq = torch.zeros(batch, seq_len, hidden, device=input_seq.device, dtype=input_seq.dtype)
    for t in range(seq_len):
        x_t = input_seq[:, t, :]
        val_in = torch.matmul(x_t, input_w_q_t)
        val_rec = torch.matmul(h, hidden_w_q_t)
        state_target = torch.tanh(val_in + val_rec)
        tau_in = torch.matmul(x_t, tau_input_w_q_t)
        tau_rec = torch.matmul(h, tau_hidden_w_q_t)
        raw_tau = torch.nn.functional.softplus(tau_in + tau_rec + tau_bias)
        time_decay = torch.clamp(raw_tau, min=1e-4, max=5.0)
        decay = torch.exp(torch.clamp(-time_decay * dt, min=-20.0, max=20.0))
        h = state_target + (h - state_target) * decay
        out_seq[:, t, :] = h
    return out_seq, h


@_jit_script_if_supported
def jit_liquid_loop(
    input_seq: torch.Tensor,
    h_init: torch.Tensor,
    dt: float,
    input_w_weight: torch.Tensor,
    hidden_w_weight: torch.Tensor,
    tau_input_w_weight: torch.Tensor,
    tau_hidden_w_weight: torch.Tensor,
    tau_bias: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return jit_liquid_loop_cached(
        input_seq,
        h_init,
        dt,
        jit_quant(input_w_weight).t().contiguous(),
        jit_quant(hidden_w_weight).t().contiguous(),
        jit_quant(tau_input_w_weight).t().contiguous(),
        jit_quant(tau_hidden_w_weight).t().contiguous(),
        tau_bias.to(device=input_seq.device, dtype=input_seq.dtype),
    )


class LiquidMixer(nn.Module):
    def __init__(self, h: int, use_bitnet: bool, fast_path: bool = True) -> None:
        super().__init__()
        self.cell = LiquidCell(h, use_bitnet=use_bitnet)
        self.norm = nn.LayerNorm(h)
        self.fast_path = bool(fast_path)
        self._compiled_train_loop = None
        self.register_buffer("_q_input_w_t", torch.empty(0), persistent=False)
        self.register_buffer("_q_hidden_w_t", torch.empty(0), persistent=False)
        self.register_buffer("_q_tau_input_w_t", torch.empty(0), persistent=False)
        self.register_buffer("_q_tau_hidden_w_t", torch.empty(0), persistent=False)
        self.register_buffer("_q_tau_bias", torch.empty(0), persistent=False)
        self.register_buffer("_weight_version", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("_cached_weight_version", torch.full((), -1, dtype=torch.int64), persistent=False)
        self._cache_ready = False

    def _set_cache(self, name: str, value: torch.Tensor) -> None:
        self._buffers[name] = value.detach().contiguous()

    def _compute_weight_version(self) -> int:
        return int(
            self.cell.input_w.weight._version
            + self.cell.hidden_w.weight._version
            + self.cell.tau_input_w.weight._version
            + self.cell.tau_hidden_w.weight._version
            + self.cell.tau_bias._version
        )

    def reset_stream_state(self) -> None:
        for name in ("_q_input_w_t", "_q_hidden_w_t", "_q_tau_input_w_t", "_q_tau_hidden_w_t", "_q_tau_bias"):
            self._buffers[name] = self._buffers[name].new_empty((0,))
        self._cached_weight_version.fill_(-1)
        self._cache_ready = False

    def mark_weights_updated(self) -> None:
        self._weight_version += 1
        self._cache_ready = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.reset_stream_state()
        if not mode:
            self._weight_version.fill_(self._compute_weight_version())
        return self

    def _ensure_qcache(self, device: torch.device, dtype: torch.dtype) -> None:
        if self.training:
            return
        current_weight_version = self._compute_weight_version()
        if int(self._weight_version.item()) != current_weight_version:
            self._weight_version.fill_(current_weight_version)
            self._cache_ready = False
        if (
            self._cache_ready
            and self._q_input_w_t.numel() > 0
            and self._q_input_w_t.device == device
            and self._q_input_w_t.dtype == dtype
            and int(self._cached_weight_version.item()) == int(self._weight_version.item())
        ):
            return
        with torch.no_grad():
            self._set_cache("_q_input_w_t", jit_quant(self.cell.input_w.weight).to(device=device, dtype=dtype).t().contiguous())
            self._set_cache("_q_hidden_w_t", jit_quant(self.cell.hidden_w.weight).to(device=device, dtype=dtype).t().contiguous())
            self._set_cache("_q_tau_input_w_t", jit_quant(self.cell.tau_input_w.weight).to(device=device, dtype=dtype).t().contiguous())
            self._set_cache("_q_tau_hidden_w_t", jit_quant(self.cell.tau_hidden_w.weight).to(device=device, dtype=dtype).t().contiguous())
            self._set_cache("_q_tau_bias", self.cell.tau_bias.to(device=device, dtype=dtype).contiguous())
            self._cached_weight_version.copy_(self._weight_version)
            self._cache_ready = True

    def _train_loop(self, x: torch.Tensor, h: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, hidden = x.shape
        out_seq = torch.empty(batch, seq_len, hidden, device=x.device, dtype=x.dtype).contiguous()
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h, dt)
            out_seq[:, t, :] = h
        return out_seq, h

    def _train_loop_compiled(self, x: torch.Tensor, h: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._compiled_train_loop is None:
            try:
                self._compiled_train_loop = torch.compile(self._train_loop, mode="reduce-overhead")
            except Exception as exc:
                warnings.warn(f"Liquid fast path compile disabled: {exc}", RuntimeWarning)
                self._compiled_train_loop = self._train_loop
        return self._compiled_train_loop(x, h, dt)

    def forward(
        self,
        x: torch.Tensor,
        dt: float = 1.0,
        h_init: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ):
        batch, _, hidden = x.shape
        if h_init is None:
            h = torch.zeros(batch, hidden, device=x.device, dtype=x.dtype)
        else:
            if h_init.dim() != 2 or h_init.shape != (batch, hidden):
                raise ValueError(f"h_init must be [B,H] = [{batch},{hidden}], got {tuple(h_init.shape)}")
            if h_init.device != x.device or h_init.dtype != x.dtype:
                raise RuntimeError(
                    "h_init device/dtype mismatch with x. "
                    f"h_init={h_init.device}/{h_init.dtype}, x={x.device}/{x.dtype}."
                )
            h = h_init
        if self.training:
            if self.fast_path and x.device.type != "mps":
                try:
                    out_seq, h = self._train_loop_compiled(x, h, dt)
                except Exception as exc:
                    warnings.warn(f"Liquid fast path failed; falling back to eager: {exc}", RuntimeWarning)
                    out_seq, h = self._train_loop(x, h, dt)
            else:
                out_seq, h = self._train_loop(x, h, dt)
        else:
            self._ensure_qcache(device=x.device, dtype=x.dtype)
            out_seq, h = jit_liquid_loop_cached(
                x,
                h,
                dt,
                self._q_input_w_t,
                self._q_hidden_w_t,
                self._q_tau_input_w_t,
                self._q_tau_hidden_w_t,
                self._q_tau_bias,
            )
        y = self.norm(out_seq + x)
        if return_state:
            return y, h
        return y

    def load_state_dict(self, *args, **kwargs):
        out = super().load_state_dict(*args, **kwargs)
        self.reset_stream_state()
        self._weight_version.fill_(self._compute_weight_version())
        return out


class BitSwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, use_bitnet: bool) -> None:
        super().__init__()
        self.gate_proj = make_linear(use_bitnet, hidden_size, intermediate_size, bias=False)
        self.up_proj = make_linear(use_bitnet, hidden_size, intermediate_size, bias=False)
        self.down_proj = make_linear(use_bitnet, intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LiquidRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, use_bitnet: bool):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.main_proj = make_linear(use_bitnet, hidden_size, num_experts, bias=False)
        self.history_window = 4
        self.fluid_mixer = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=self.history_window,
            groups=hidden_size,
            padding=0,
            bias=False,
        )
        self.fluid_gate = make_linear(use_bitnet, hidden_size, num_experts, bias=False)
        nn.init.zeros_(self.fluid_gate.weight)
        self.register_buffer(
            "inference_state",
            torch.zeros(1, hidden_size, self.history_window - 1),
            persistent=False,
        )

    def _update_inference_state(self, state: torch.Tensor) -> None:
        self._buffers["inference_state"] = state.detach().clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_flat = False
        if x.dim() == 2:
            is_flat = True
            x = x.unsqueeze(1)
        batch, seq_len, hidden = x.shape
        logits_main = self.main_proj(x)
        if self.training or seq_len > 1:
            x_t = x.transpose(1, 2)
            x_t_padded = F.pad(x_t, (self.history_window - 1, 0))
            fluid_mem = self.fluid_mixer(x_t_padded).transpose(1, 2)
            logits_fluid = self.fluid_gate(F.silu(fluid_mem))
            if not self.training:
                last_tokens = x_t[..., -(self.history_window - 1) :]
                if last_tokens.size(2) < (self.history_window - 1):
                    pad = torch.zeros(
                        batch,
                        hidden,
                        (self.history_window - 1) - last_tokens.size(2),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    last_tokens = torch.cat([pad, last_tokens], dim=2)
                self._update_inference_state(last_tokens)
            out = logits_main + logits_fluid
        else:
            if self.inference_state.size(0) != batch:
                if self.inference_state.size(0) == 1:
                    self._update_inference_state(self.inference_state.expand(batch, -1, -1).contiguous())
                else:
                    self._update_inference_state(
                        torch.zeros(batch, hidden, self.history_window - 1, device=x.device, dtype=x.dtype)
                    )
            current_token = x.transpose(1, 2)
            context = torch.cat([self.inference_state.to(device=x.device, dtype=x.dtype), current_token], dim=2)
            context_padded = F.pad(context, (self.history_window - 1, 0))
            fluid_mem = self.fluid_mixer(context_padded)[..., -1:].transpose(1, 2)
            logits_fluid = self.fluid_gate(F.silu(fluid_mem))
            self._update_inference_state(context[..., 1:])
            out = logits_main + logits_fluid
        if is_flat:
            out = out.reshape(-1, self.num_experts)
        return out

    def get_state(self) -> torch.Tensor:
        return self.inference_state.clone()

    def set_state(self, state: torch.Tensor) -> None:
        if state.dim() != 3:
            raise ValueError(f"State must be 3D [Batch, Hidden, Window-1], got {state.shape}")
        if state.size(2) != self.history_window - 1:
            raise ValueError(f"State window size mismatch. Expected {self.history_window - 1}, got {state.size(2)}")
        self._update_inference_state(state)


class MoE(nn.Module):
    def __init__(self, arch_cfg: MirrorModelConfig) -> None:
        super().__init__()
        self.hidden_size = arch_cfg.hidden_size
        self.num_experts = arch_cfg.num_experts
        self.active_experts = arch_cfg.active_experts
        self.router = LiquidRouter(self.hidden_size, self.num_experts, use_bitnet=arch_cfg.use_bitnet)
        self.experts = nn.ModuleList(
            BitSwiGLU(self.hidden_size, arch_cfg.moe_intermediate, use_bitnet=arch_cfg.use_bitnet)
            for _ in range(self.num_experts)
        )
        self.shared_expert = BitSwiGLU(self.hidden_size, arch_cfg.moe_intermediate, use_bitnet=arch_cfg.use_bitnet)
        self.shared_gate = nn.Parameter(torch.tensor([arch_cfg.shared_expert_gate], dtype=torch.float32))
        self.router_temperature = float(arch_cfg.router_temperature)
        self.router_jitter = float(arch_cfg.router_jitter)
        self.router_z_loss_coef = float(arch_cfg.z_loss_coef)
        self.router_alarm_threshold = float(arch_cfg.router_alarm_threshold)
        self.use_cross_expert_sync_bus = bool(arch_cfg.use_cross_expert_sync_bus)
        self.cross_expert_sync_gain = float(arch_cfg.cross_expert_sync_gain)
        self.use_structural_plasticity = bool(arch_cfg.use_structural_plasticity)
        self.structural_ema_decay = float(arch_cfg.structural_ema_decay)
        self.structural_prune_threshold = float(arch_cfg.structural_prune_threshold)
        self.structural_grow_threshold = float(arch_cfg.structural_grow_threshold)
        self.structural_update_interval = int(arch_cfg.structural_update_interval)
        self.use_expert_paging = bool(arch_cfg.use_expert_paging)
        self.expert_paging_inference_only = bool(arch_cfg.expert_paging_inference_only)
        self.expert_paging_lazy_init = bool(arch_cfg.expert_paging_lazy_init)
        self.expert_paging_cache_size = max(1, int(arch_cfg.expert_paging_cache_size))
        self.expert_paging_offload_device = str(arch_cfg.expert_paging_offload_device)
        self.expert_paging_verbose = bool(arch_cfg.expert_paging_verbose)
        self._expert_lru: List[int] = []
        self._expert_resident: set[int] = set()
        self._paging_bootstrapped = False
        self.use_switch_loss = bool(arch_cfg.use_switch_loss)
        self.router_jitter_boost = float(arch_cfg.router_jitter_boost)
        self.collapse_threshold = 0.85
        self.moe_capacity_enforce = bool(arch_cfg.moe_capacity_enforce)
        self.moe_capacity_factor = float(arch_cfg.moe_capacity_factor)
        self.dispatch_mode = str(arch_cfg.moe_dispatch_mode).lower()
        self.register_buffer("last_expert_load", torch.zeros(self.num_experts))
        self.register_buffer("last_router_entropy", torch.tensor(0.0))
        self.register_buffer("last_router_max_load", torch.tensor(0.0))
        self.register_buffer("last_capacity_overflow_ratio", torch.tensor(0.0))
        self.register_buffer("collapse_detected", torch.tensor(False))
        self.register_buffer("expert_activity_mask", torch.ones(self.num_experts, dtype=torch.bool))
        self.register_buffer("expert_usage_ema", torch.zeros(self.num_experts))
        self.register_buffer("plasticity_step", torch.zeros((), dtype=torch.int64))
        self.register_buffer("expert_paging_swaps_in", torch.zeros((), dtype=torch.int64), persistent=False)
        self.register_buffer("expert_paging_swaps_out", torch.zeros((), dtype=torch.int64), persistent=False)
        self.sync_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.sync_load_proj = nn.Linear(self.num_experts, self.hidden_size, bias=False)
        self.sync_gate = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def _should_skip_expert_apply(self) -> bool:
        return bool(self.use_expert_paging and self.expert_paging_lazy_init)

    def _offload_device(self) -> torch.device:
        try:
            return torch.device(self.expert_paging_offload_device)
        except Exception:
            return torch.device("cpu")

    def _offload_all_experts(self, target_device: torch.device, target_dtype: Optional[torch.dtype] = None) -> None:
        with torch.no_grad():
            for expert in self.experts:
                if target_dtype is None:
                    expert.to(device=target_device)
                else:
                    expert.to(device=target_device, dtype=target_dtype)
        self._expert_resident.clear()
        self._expert_lru.clear()
        self._paging_bootstrapped = True

    def _apply(self, fn):
        if not self._should_skip_expert_apply():
            return super()._apply(fn)
        experts = self._modules.pop("experts")
        try:
            out = super()._apply(fn)
        finally:
            self._modules["experts"] = experts
        offload_device = self._offload_device()
        ref_param = next(self.shared_expert.parameters(), None)
        target_dtype = ref_param.dtype if ref_param is not None and torch.is_floating_point(ref_param) else None
        self._offload_all_experts(offload_device, target_dtype=target_dtype)
        return out

    def train(self, mode: bool = True):
        out = super().train(mode)
        if not self.use_expert_paging:
            return out
        ref_param = next(self.shared_expert.parameters(), None)
        if ref_param is None:
            return out
        ref_device = ref_param.device
        ref_dtype = ref_param.dtype if torch.is_floating_point(ref_param) else None
        if mode:
            with torch.no_grad():
                for expert in self.experts:
                    if ref_dtype is None:
                        expert.to(device=ref_device)
                    else:
                        expert.to(device=ref_device, dtype=ref_dtype)
            self._expert_resident = set(range(self.num_experts))
            self._expert_lru = list(range(self.num_experts))
            self._paging_bootstrapped = False
            return out
        if self.expert_paging_lazy_init:
            offload_device = self._offload_device()
            if offload_device != ref_device:
                self._offload_all_experts(offload_device, target_dtype=ref_dtype)
        return out

    def get_router_state(self) -> torch.Tensor:
        return self.router.get_state()

    def set_router_state(self, state: torch.Tensor) -> None:
        self.router.set_state(state)

    def get_expert_load(self) -> torch.Tensor:
        return self.last_expert_load

    def get_router_entropy(self) -> torch.Tensor:
        return self.last_router_entropy

    def get_router_max_load(self) -> torch.Tensor:
        return self.last_router_max_load

    def get_expert_paging_stats(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.use_expert_paging),
            "inference_only": bool(self.expert_paging_inference_only),
            "lazy_init": bool(self.expert_paging_lazy_init),
            "cache_size": int(self.expert_paging_cache_size),
            "offload_device": str(self.expert_paging_offload_device),
            "bootstrapped": bool(self._paging_bootstrapped),
            "swaps_in": int(self.expert_paging_swaps_in.item()),
            "swaps_out": int(self.expert_paging_swaps_out.item()),
            "resident_count": int(len(self._expert_resident)),
        }

    def _paging_active_for_step(self) -> bool:
        if not self.use_expert_paging:
            return False
        if self.training and self.expert_paging_inference_only:
            return False
        return not self.training

    def _expert_device(self, expert_id: int) -> torch.device:
        param = next(self.experts[expert_id].parameters(), None)
        return param.device if param is not None else torch.device("cpu")

    def _touch_lru(self, expert_id: int) -> None:
        if expert_id in self._expert_lru:
            self._expert_lru.remove(expert_id)
        self._expert_lru.append(expert_id)

    def _refresh_resident(self, compute_device: torch.device) -> None:
        self._expert_resident = {
            idx for idx in range(self.num_experts) if self._expert_device(idx) == compute_device
        }
        self._expert_lru = [idx for idx in self._expert_lru if idx in self._expert_resident]

    def _bootstrap_expert_paging(self, compute_device: torch.device) -> None:
        if self._paging_bootstrapped:
            return
        if compute_device.type == "cpu":
            self._paging_bootstrapped = True
            return
        offload_device = self._offload_device()
        if offload_device == compute_device:
            self._paging_bootstrapped = True
            return
        ref_param = next(self.shared_expert.parameters(), None)
        target_dtype = ref_param.dtype if ref_param is not None and torch.is_floating_point(ref_param) else None
        self._offload_all_experts(offload_device, target_dtype=target_dtype)

    def _page_in_active_experts(
        self,
        active_expert_ids: List[int],
        compute_device: torch.device,
        compute_dtype: torch.dtype,
    ) -> None:
        if not active_expert_ids:
            return
        self._bootstrap_expert_paging(compute_device)
        if compute_device.type == "cpu":
            return
        offload_device = self._offload_device()
        if offload_device == compute_device:
            return
        self._refresh_resident(compute_device)
        with torch.no_grad():
            for expert_id in active_expert_ids:
                if expert_id not in self._expert_resident:
                    self.experts[expert_id].to(device=compute_device, dtype=compute_dtype)
                    self._expert_resident.add(expert_id)
                    self.expert_paging_swaps_in.add_(1)
                self._touch_lru(expert_id)
            keep = set(active_expert_ids)
            max_resident = max(self.expert_paging_cache_size, len(keep))
            while len(self._expert_resident) > max_resident:
                evict_id = None
                for candidate in self._expert_lru:
                    if candidate not in keep:
                        evict_id = candidate
                        break
                if evict_id is None:
                    break
                self.experts[evict_id].to(device=offload_device, dtype=compute_dtype)
                self._expert_resident.discard(evict_id)
                self.expert_paging_swaps_out.add_(1)
                self._expert_lru = [idx for idx in self._expert_lru if idx != evict_id]

    def _apply_structural_plasticity(self, load: torch.Tensor) -> None:
        if not self.use_structural_plasticity or not self.training:
            return
        with torch.no_grad():
            self.expert_usage_ema.mul_(self.structural_ema_decay).add_(load.detach() * (1.0 - self.structural_ema_decay))
            self.plasticity_step.add_(1)
            if int(self.plasticity_step.item()) % max(1, self.structural_update_interval) != 0:
                return
            active_count = int(self.expert_activity_mask.sum().item())
            min_active = max(1, self.active_experts)
            if active_count > min_active:
                active_idx = torch.where(self.expert_activity_mask)[0]
                active_ema = self.expert_usage_ema[active_idx]
                prune_pos = torch.argmin(active_ema)
                prune_idx = active_idx[prune_pos]
                if active_ema[prune_pos].item() < self.structural_prune_threshold:
                    self.expert_activity_mask[prune_idx] = False
                    active_count -= 1
            inactive_idx = torch.where(~self.expert_activity_mask)[0]
            if inactive_idx.numel() > 0 and self.last_router_max_load.item() > self.structural_grow_threshold:
                self.expert_activity_mask[inactive_idx[0]] = True

    def _dispatch_sequential(self, x_flat: torch.Tensor, topk_idx: torch.Tensor, topk_vals: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden = x_flat.shape
        out_flat = x_flat.new_zeros((num_tokens, hidden))
        for expert_id_int, expert in enumerate(self.experts):
            expert_mask = topk_idx == expert_id_int
            token_mask = expert_mask.any(dim=-1)
            if not token_mask.any():
                continue
            selected_x = x_flat[token_mask]
            expert_param = next(expert.parameters(), None)
            if expert_param is not None and selected_x.dtype != expert_param.dtype:
                selected_x = selected_x.to(dtype=expert_param.dtype)
            expert_out = expert(selected_x)
            if expert_out.dtype != out_flat.dtype:
                expert_out = expert_out.to(dtype=out_flat.dtype)
            weights = (topk_vals[token_mask] * expert_mask[token_mask].float()).sum(dim=-1, keepdim=True)
            out_flat[token_mask] += expert_out * weights
        return out_flat

    def _dispatch_parallel(
        self,
        x_flat: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_vals: torch.Tensor,
        capacity_mask: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, hidden = x_flat.shape
        k = topk_idx.size(-1)
        out_flat = x_flat.new_zeros((num_tokens, hidden))
        token_idx = torch.arange(num_tokens, device=topk_idx.device).repeat_interleave(k)
        expert_idx = topk_idx.reshape(-1)
        weights = topk_vals.reshape(-1)
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
        counts = torch.bincount(expert_sorted, minlength=self.num_experts)
        start = 0
        for expert_id_int, expert in enumerate(self.experts):
            cnt = int(counts[expert_id_int].item())
            if cnt == 0:
                continue
            end = start + cnt
            idx = token_sorted[start:end]
            weight = weight_sorted[start:end].unsqueeze(-1)
            selected_x = x_flat.index_select(0, idx)
            expert_param = next(expert.parameters(), None)
            if expert_param is not None and selected_x.dtype != expert_param.dtype:
                selected_x = selected_x.to(dtype=expert_param.dtype)
            expert_out = expert(selected_x)
            if expert_out.dtype != out_flat.dtype:
                expert_out = expert_out.to(dtype=out_flat.dtype)
            out_flat.index_add_(0, idx, expert_out * weight)
            start = end
        return out_flat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, hidden = x.shape
        logits = self.router(x).reshape(-1, self.num_experts)
        x_flat = x.reshape(-1, hidden)
        num_tokens = x_flat.size(0)
        num_experts = self.num_experts
        logits_f = logits.float()
        active_mask = self.expert_activity_mask.to(device=logits_f.device)
        if active_mask.any():
            logits_f = logits_f.masked_fill(~active_mask.unsqueeze(0), float("-inf"))
        if self.router_temperature != 1.0:
            logits_f = logits_f / self.router_temperature
        applied_jitter = self.router_jitter_boost if self.training and self.collapse_detected.item() else self.router_jitter
        if self.training and applied_jitter > 0.0:
            logits_f = logits_f + torch.randn_like(logits_f) * applied_jitter
        k = min(self.active_experts, num_experts)
        topk_logits, topk_idx = torch.topk(logits_f, k=k, dim=-1)
        topk_vals = F.softmax(topk_logits, dim=-1)
        capacity_mask = torch.ones_like(topk_idx, dtype=torch.bool)
        overflow_ratio = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        if self.moe_capacity_enforce and self.moe_capacity_factor > 0.0:
            capacity = max(1, int(math.ceil(self.moe_capacity_factor * (num_tokens * k) / max(1, num_experts))))
            dropped = 0
            for expert_id_int in range(num_experts):
                hits = (topk_idx == expert_id_int).nonzero(as_tuple=False)
                if hits.size(0) > capacity:
                    overflow = hits[capacity:]
                    capacity_mask[overflow[:, 0], overflow[:, 1]] = False
                    dropped += int(overflow.size(0))
            topk_vals = topk_vals * capacity_mask.float()
            empty_rows = topk_vals.sum(dim=-1) <= 0
            if empty_rows.any():
                topk_vals[empty_rows, 0] = 1.0
                capacity_mask[empty_rows, 0] = True
            topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            overflow_ratio = torch.tensor(
                float(dropped) / float(max(1, num_tokens * k)),
                device=x.device,
                dtype=torch.float32,
            )
        flat_idx = topk_idx[capacity_mask].reshape(-1)
        counts = torch.zeros(num_experts, device=flat_idx.device, dtype=torch.float32)
        if flat_idx.numel() > 0:
            counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
        denom = float(max(1, int(flat_idx.numel())))
        load = counts / denom
        self.last_expert_load.copy_(load.detach())
        self.last_router_max_load.copy_(load.max().detach())
        norm = math.log(float(num_experts)) if num_experts > 1 else 1.0
        entropy = -(load.clamp(min=1e-8) * load.clamp(min=1e-8).log()).sum() / norm
        self.last_router_entropy.copy_(entropy.detach())
        self.last_capacity_overflow_ratio.copy_(overflow_ratio.detach())
        self._apply_structural_plasticity(load)
        if self._paging_active_for_step():
            active_expert_ids = torch.nonzero(counts > 0.0, as_tuple=False).flatten().tolist()
            self._page_in_active_experts([int(idx) for idx in active_expert_ids], x.device, x.dtype)
        if self.training:
            gates_full = F.softmax(logits_f, dim=-1)
            importance = gates_full.mean(dim=0)
            if self.use_switch_loss:
                load_balancing_loss = (importance * load).sum() * float(num_experts)
            else:
                load_balancing_loss = ((importance - load) ** 2).mean() * float(num_experts)
            max_load = load.max().item()
            if max_load > self.collapse_threshold:
                self.collapse_detected.fill_(True)
            elif self.collapse_detected.item() and max_load < 0.5:
                self.collapse_detected.fill_(False)
        else:
            load_balancing_loss = torch.tensor(0.0, device=x.device, dtype=logits_f.dtype)
        aux_loss = load_balancing_loss
        if self.router_z_loss_coef > 0.0:
            z = torch.logsumexp(logits_f, dim=-1)
            aux_loss = aux_loss + (z * z).mean() * self.router_z_loss_coef
        if self.dispatch_mode == "parallel":
            out_flat = self._dispatch_parallel(x_flat, topk_idx, topk_vals, capacity_mask)
        else:
            out_flat = self._dispatch_sequential(x_flat, topk_idx, topk_vals)
        shared_out = self.shared_expert(x_flat)
        gate_scale = torch.sigmoid(self.shared_gate).to(dtype=shared_out.dtype, device=shared_out.device)
        out_flat = out_flat + shared_out * gate_scale
        if self.use_cross_expert_sync_bus:
            token_sync = self.sync_proj(out_flat.mean(dim=0, keepdim=True))
            load_sync = self.sync_load_proj(load.unsqueeze(0).to(dtype=out_flat.dtype))
            sync = torch.tanh(token_sync + load_sync).expand_as(out_flat)
            out_flat = out_flat + sync * (torch.sigmoid(self.sync_gate) * self.cross_expert_sync_gain)
        return out_flat.reshape(batch, seq_len, hidden), aux_loss


def newton_schulz_inverse(mat: torch.Tensor, num_iters: int = 6) -> torch.Tensor:
    orig_dtype = mat.dtype
    mat = mat.float()
    *batch, n, m = mat.shape
    assert n == m, "newton_schulz_inverse expects square matrices."
    eye = torch.eye(n, device=mat.device, dtype=mat.dtype)
    if batch:
        eye = eye.expand(*batch, n, n)
    norm_inf = mat.abs().sum(dim=-1).max(dim=-1, keepdim=True)[0].unsqueeze(-1).clamp(min=1e-6)
    mat_scaled = mat / norm_inf
    x = mat_scaled.transpose(-1, -2)
    for _ in range(num_iters):
        ax = torch.matmul(mat_scaled, x)
        x = torch.matmul(x, (2.0 * eye - ax))
    return (x / norm_inf).to(orig_dtype)


class UnitaryQINN(nn.Module):
    def __init__(self, dim: int, num_iters: int = 6, enabled: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.num_iters = num_iters
        self.enabled = bool(enabled)
        self.A = nn.Parameter(torch.randn(dim, dim) * 1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        orig_dtype = x.dtype
        x = x.float()
        skew = self.A - self.A.t()
        eye = torch.eye(self.dim, device=skew.device, dtype=skew.dtype)
        m_inv = newton_schulz_inverse(eye - skew, num_iters=self.num_iters)
        unitary = torch.matmul(m_inv, eye + skew)
        if not torch.isfinite(unitary).all():
            unitary = torch.where(torch.isfinite(unitary), unitary, eye)
        return torch.matmul(x, unitary.t()).to(orig_dtype)


class GlobalWorkspaceBroadcast(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, workspace: Optional[torch.Tensor]) -> torch.Tensor:
        if workspace is None:
            return x
        workspace = workspace.to(device=x.device, dtype=x.dtype)
        signal = torch.tanh(self.proj(workspace)).unsqueeze(1)
        return x + signal * torch.sigmoid(self.gate)


class ContinuousLatentODEStateChannel(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.state_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.input_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.register_buffer("latent_state", torch.zeros(1, hidden_size), persistent=False)

    def reset_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        self._buffers["latent_state"] = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def _ensure_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        if (
            self.latent_state.numel() == 0
            or self.latent_state.shape[0] != batch_size
            or self.latent_state.device != device
            or self.latent_state.dtype != dtype
        ):
            self.reset_state(batch_size, device, dtype)

    def forward(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        batch_size = x.size(0)
        self._ensure_state(batch_size, x.device, x.dtype)
        summary = x.mean(dim=1)
        z = self.latent_state.detach().clone()
        dz = torch.tanh(self.state_proj(z) + self.input_proj(summary))
        z_next = z + float(dt) * dz
        with torch.no_grad():
            self.latent_state.copy_(z_next.detach())
        return x + self.out_proj(z_next).unsqueeze(1).to(dtype=x.dtype)


class NeuromodulatoryGainLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.gain_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.bias_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.gain_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, workspace: Optional[torch.Tensor]) -> torch.Tensor:
        if workspace is None:
            return x
        workspace = workspace.to(device=x.device, dtype=x.dtype)
        gain = torch.sigmoid(self.gain_proj(workspace)).unsqueeze(1)
        bias = torch.tanh(self.bias_proj(workspace)).unsqueeze(1)
        return x * (1.0 + gain * self.gain_scale) + bias * self.gain_scale


class HebbianPlasticityLayer(nn.Module):
    def __init__(self, hidden_size: int, eta: float = 0.01, decay: float = 0.99) -> None:
        super().__init__()
        self.eta = float(eta)
        self.decay = float(decay)
        self.register_buffer("trace", torch.zeros(hidden_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                activity = x.detach().pow(2).mean(dim=(0, 1)).to(self.trace.dtype)
                self.trace.mul_(self.decay).add_(activity * (1.0 - self.decay))
        gain = 1.0 + self.eta * torch.tanh(self.trace.to(device=x.device, dtype=x.dtype))
        return x * gain.view(1, 1, -1)


class NeuroSymbolicLayer(nn.Module):
    def __init__(self, hidden_size: int, num_rules: int = 8) -> None:
        super().__init__()
        self.num_rules = int(max(1, num_rules))
        self.rule_keys = nn.Parameter(torch.randn(self.num_rules, hidden_size) * 0.02)
        self.rule_values = nn.Parameter(torch.randn(self.num_rules, hidden_size) * 0.02)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rule_gain = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        summary = x.mean(dim=1)
        logits = torch.matmul(summary, self.rule_keys.t())
        weights = F.softmax(logits, dim=-1)
        rule_context = torch.matmul(weights, self.rule_values)
        residual = torch.tanh(self.out_proj(rule_context)).unsqueeze(1)
        return x + residual * torch.sigmoid(self.rule_gain)


class LifelongSafetyLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ema_decay: float = 0.99,
        max_adaptation_gain: float = 0.05,
        drift_threshold: float = 0.35,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.ema_decay = float(ema_decay)
        self.max_adaptation_gain = float(max_adaptation_gain)
        self.drift_threshold = float(drift_threshold)
        self.register_buffer("running_mean", torch.zeros(self.hidden_size), persistent=False)
        self.register_buffer("running_var", torch.ones(self.hidden_size), persistent=False)
        self.register_buffer("last_drift", torch.zeros(()), persistent=False)
        self.gain = nn.Parameter(torch.zeros(self.hidden_size))

    def _update_stats(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            mean = x.detach().mean(dim=(0, 1))
            var = x.detach().var(dim=(0, 1), unbiased=False)
            self.running_mean.mul_(self.ema_decay).add_(mean * (1.0 - self.ema_decay))
            self.running_var.mul_(self.ema_decay).add_(var * (1.0 - self.ema_decay))
            self.last_drift.copy_((mean - self.running_mean).abs().mean().detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"LifelongSafetyLayer expects [B,T,H], got {tuple(x.shape)}")
        self._update_stats(x)
        bounded = torch.tanh(self.gain).to(device=x.device, dtype=x.dtype)
        scale = torch.clamp(torch.tensor(self.max_adaptation_gain, device=x.device, dtype=x.dtype), min=0.0)
        if float(self.last_drift.item()) > self.drift_threshold:
            scale = scale * 0.5
        return x * (1.0 + bounded.view(1, 1, -1) * scale)

    def safety_metrics(self) -> Dict[str, float]:
        return {
            "last_drift": float(self.last_drift.item()),
            "ema_decay": self.ema_decay,
            "max_adaptation_gain": self.max_adaptation_gain,
            "drift_threshold": self.drift_threshold,
        }


@dataclass
class WorldModelOutput:
    dynamics_logits: torch.Tensor
    latent_state: torch.Tensor
    uncertainty: torch.Tensor
    counterfactual_logits: torch.Tensor
    risk_score: torch.Tensor

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "world_dynamics_logits": self.dynamics_logits,
            "world_latent_state": self.latent_state,
            "world_uncertainty": self.uncertainty,
            "world_counterfactual_logits": self.counterfactual_logits,
            "world_risk_score": self.risk_score,
        }


class CausalWorldModelHead(nn.Module):
    def __init__(self, hidden_size: int, horizon: int = 1) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.horizon = int(max(1, horizon))
        self.pre = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.dynamics = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.counterfactual = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.uncertainty = nn.Linear(self.hidden_size, 1, bias=True)
        self.risk = nn.Linear(self.hidden_size, 1, bias=True)

    def forward(self, x: torch.Tensor) -> WorldModelOutput:
        summary = x.mean(dim=1)
        latent = torch.tanh(self.pre(summary))
        dyn_steps = []
        state = latent
        for _ in range(self.horizon):
            state = torch.tanh(self.dynamics(state))
            dyn_steps.append(state)
        cf_steps = []
        cf_state = torch.tanh(self.counterfactual(-latent))
        for _ in range(self.horizon):
            cf_state = torch.tanh(self.dynamics(cf_state))
            cf_steps.append(cf_state)
        return WorldModelOutput(
            dynamics_logits=torch.stack(dyn_steps, dim=1),
            latent_state=latent,
            uncertainty=torch.sigmoid(self.uncertainty(latent)).squeeze(-1),
            counterfactual_logits=torch.stack(cf_steps, dim=1),
            risk_score=torch.sigmoid(self.risk(latent - cf_state)).squeeze(-1),
        )


class MertFormerBlock(nn.Module):
    def __init__(self, layer_id: int, arch_cfg: MirrorModelConfig) -> None:
        super().__init__()
        self.layer_id = int(layer_id)
        self.arch_cfg = arch_cfg
        hidden_size = arch_cfg.hidden_size
        self.norm1 = RMSNorm(hidden_size, eps=arch_cfg.rms_norm_eps)
        self.norm2 = RMSNorm(hidden_size, eps=arch_cfg.rms_norm_eps)
        self.residual_scale = (2 * arch_cfg.num_layers) ** -0.5
        self.attn = MLA(arch_cfg)
        self.use_moe = bool(arch_cfg.use_moe)
        every_n = int(arch_cfg.moe_every_n_layers)
        if self.use_moe and every_n > 0 and ((self.layer_id + 1) % every_n == 0):
            self.is_moe_layer = True
            self.ff = MoE(arch_cfg)
        else:
            self.is_moe_layer = False
            self.ff = MertFormerFFN(arch_cfg)
        self.qinn = None
        if arch_cfg.use_qinn and ((self.layer_id + 1) % max(1, arch_cfg.qinn_every_n_layers) == 0):
            self.qinn = UnitaryQINN(hidden_size, enabled=True)
        self.liquid = None
        if arch_cfg.use_liquid:
            if arch_cfg.liquid_layers_idx and self.layer_id in arch_cfg.liquid_layers_idx:
                self.liquid = LiquidMixer(hidden_size, use_bitnet=arch_cfg.use_bitnet, fast_path=arch_cfg.liquid_fast_path)
            elif arch_cfg.liquid_every_n_layers > 0 and ((self.layer_id + 1) % arch_cfg.liquid_every_n_layers == 0):
                self.liquid = LiquidMixer(hidden_size, use_bitnet=arch_cfg.use_bitnet, fast_path=arch_cfg.liquid_fast_path)
        self.workspace_layer = GlobalWorkspaceBroadcast(hidden_size) if arch_cfg.use_global_workspace_broadcast else None
        self.hebbian_layer = (
            HebbianPlasticityLayer(hidden_size, eta=arch_cfg.hebbian_eta, decay=arch_cfg.hebbian_decay)
            if arch_cfg.use_hebbian_plasticity
            else None
        )
        self.neuro_symbolic_layer = (
            NeuroSymbolicLayer(hidden_size, num_rules=arch_cfg.neuro_symbolic_rules)
            if arch_cfg.use_neuro_symbolic_layer
            else None
        )
        self.lifelong_safety_layer = (
            LifelongSafetyLayer(
                hidden_size,
                ema_decay=arch_cfg.lifelong_ema_decay,
                max_adaptation_gain=arch_cfg.lifelong_max_adaptation_gain,
                drift_threshold=arch_cfg.lifelong_drift_threshold,
            )
            if arch_cfg.use_lifelong_safety_layer
            else None
        )
        self.last_router_stats: Dict[str, Any] = {}

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        workspace: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        h = self.norm1(x)
        attn_out, present_key_value = self.attn(h, past_key_value=past_key_value, use_cache=use_cache)
        x = x + attn_out * self.residual_scale
        if self.liquid is not None:
            x = self.liquid(x)
        if self.workspace_layer is not None:
            x = self.workspace_layer(x, workspace)
        h = self.norm2(x)
        if self.is_moe_layer:
            ff_out, aux_loss = self.ff(h)
            self.last_router_stats = {
                "router_entropy": float(self.ff.get_router_entropy().item()),
                "router_max_load": float(self.ff.get_router_max_load().item()),
                "expert_load": [float(item) for item in self.ff.get_expert_load().tolist()],
                "capacity_overflow_ratio": float(self.ff.last_capacity_overflow_ratio.item()),
                "collapse_detected": bool(self.ff.collapse_detected.item()),
                "expert_paging": self.ff.get_expert_paging_stats(),
            }
        else:
            ff_out = self.ff(h)
            aux_loss = h.new_zeros(())
            self.last_router_stats = {}
        x = x + ff_out * self.residual_scale
        if self.hebbian_layer is not None:
            x = self.hebbian_layer(x)
        if self.neuro_symbolic_layer is not None:
            x = self.neuro_symbolic_layer(x)
        if self.lifelong_safety_layer is not None:
            x = self.lifelong_safety_layer(x)
        if self.qinn is not None:
            x = self.qinn(x)
        if aux_loss.ndim > 0:
            aux_loss = aux_loss.sum()
        return x, aux_loss, present_key_value


TransformerBlock = MertFormerBlock


class ChessPolicyValueNet(nn.Module):
    META_CARDINALITIES = [
        2,   # turn
        2,   # white king-side castling
        2,   # white queen-side castling
        2,   # black king-side castling
        2,   # black queen-side castling
        9,   # ep file
        16,  # halfmove bucket
        32,  # fullmove bucket
        2,   # in check
        32,  # legal move count bucket
        40,  # white material bucket
        40,  # black material bucket
    ]

    def __init__(self, cfg: Dict[str, Any], vocab_size: int):
        super().__init__()
        self.arch_cfg = build_mirror_model_config(cfg)
        hidden = self.arch_cfg.hidden_size
        dropout = self.arch_cfg.dropout
        self.piece_embed = nn.Embedding(13, hidden)
        self.square_embed = nn.Embedding(64, hidden)
        self.meta_type_embed = nn.Embedding(len(self.META_CARDINALITIES), hidden)
        self.meta_value_embeds = nn.ModuleList(nn.Embedding(card, hidden) for card in self.META_CARDINALITIES)
        self.blocks = nn.ModuleList(
            MertFormerBlock(layer_id=layer_idx, arch_cfg=self.arch_cfg)
            for layer_idx in range(self.arch_cfg.num_layers)
        )
        self.final_norm = RMSNorm(hidden, eps=self.arch_cfg.rms_norm_eps)
        self.drop = nn.Dropout(dropout)
        self.use_global_workspace_broadcast = bool(self.arch_cfg.use_global_workspace_broadcast)
        self.workspace_blend = float(self.arch_cfg.workspace_blend)
        self.latent_ode_channel = (
            ContinuousLatentODEStateChannel(hidden)
            if self.arch_cfg.use_latent_ode_state_channel
            else None
        )
        self.neuromod_gain_layer = (
            NeuromodulatoryGainLayer(hidden)
            if self.arch_cfg.use_neuromodulatory_gain
            else None
        )
        self.world_model_head = (
            CausalWorldModelHead(hidden, horizon=self.arch_cfg.world_model_horizon)
            if self.arch_cfg.use_world_model_head
            else None
        )
        self._last_world_model_outputs: Optional[Dict[str, torch.Tensor]] = None
        self.phase_head = (
            make_linear(self.arch_cfg.use_bitnet, hidden, PHASE_CLASS_COUNT, bias=True)
            if self.arch_cfg.use_phase_head
            else None
        )
        self.wdl_head = (
            make_linear(self.arch_cfg.use_bitnet, hidden, WDL_CLASS_COUNT, bias=True)
            if self.arch_cfg.use_wdl_head
            else None
        )
        self.legality_head = (
            make_linear(self.arch_cfg.use_bitnet, hidden, vocab_size, bias=True)
            if self.arch_cfg.use_legality_head
            else None
        )
        self._last_auxiliary_outputs: Optional[Dict[str, torch.Tensor]] = None
        self.policy_head = make_linear(self.arch_cfg.use_bitnet, hidden, vocab_size, bias=True)
        self.value_head = make_linear(self.arch_cfg.use_bitnet, hidden, 1, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.vocab_size = vocab_size

    def forward(self, piece_ids: torch.Tensor, meta_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        bsz = piece_ids.size(0)
        square_ids = torch.arange(64, device=piece_ids.device).unsqueeze(0).expand(bsz, -1)
        board = self.piece_embed(piece_ids) + self.square_embed(square_ids)
        meta_tokens: List[torch.Tensor] = []
        for meta_idx, embed in enumerate(self.meta_value_embeds):
            meta_val = meta_ids[:, meta_idx]
            type_tok = self.meta_type_embed(torch.full_like(meta_val, meta_idx))
            meta_tokens.append(embed(meta_val) + type_tok)
        meta = torch.stack(meta_tokens, dim=1)
        x = torch.cat([meta, board], dim=1)
        x = x * (self.arch_cfg.hidden_size ** 0.5)
        x = self.drop(x)
        workspace_state = x.mean(dim=1) if self.use_global_workspace_broadcast else None
        aux_loss = x.new_tensor(0.0)
        router_reports: Dict[str, Any] = {}
        use_gradient_checkpointing = bool(self.arch_cfg.use_gradient_checkpointing and self.training)
        for block_idx, block in enumerate(self.blocks):
            if self.latent_ode_channel is not None:
                x = self.latent_ode_channel(x, dt=self.arch_cfg.latent_ode_dt)
            if use_gradient_checkpointing:
                if workspace_state is None:
                    def _checkpoint_block(block_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                        block_out, block_aux, _ = block(block_input, workspace=None)
                        return block_out, block_aux

                    x, aux = torch.utils.checkpoint.checkpoint(_checkpoint_block, x, use_reentrant=False)
                else:
                    def _checkpoint_block_with_workspace(
                        block_input: torch.Tensor,
                        workspace_input: torch.Tensor,
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
                        block_out, block_aux, _ = block(block_input, workspace=workspace_input)
                        return block_out, block_aux

                    x, aux = torch.utils.checkpoint.checkpoint(
                        _checkpoint_block_with_workspace,
                        x,
                        workspace_state,
                        use_reentrant=False,
                    )
            else:
                x, aux, _ = block(x, workspace=workspace_state)
            aux_loss = aux_loss + aux
            if workspace_state is not None:
                token_summary = x.mean(dim=1)
                blend = min(max(self.workspace_blend, 0.0), 1.0)
                workspace_state = workspace_state * blend + token_summary * (1.0 - blend)
            if self.neuromod_gain_layer is not None:
                x = self.neuromod_gain_layer(x, workspace_state)
            if block.last_router_stats:
                router_reports[f"block_{block_idx}"] = dict(block.last_router_stats)
        x = self.final_norm(x)
        if self.world_model_head is not None:
            self._last_world_model_outputs = self.world_model_head(x).to_dict()
        else:
            self._last_world_model_outputs = None
        pooled = self.dropout(x.mean(dim=1))
        auxiliary_outputs: Dict[str, torch.Tensor] = {}
        if self.phase_head is not None:
            auxiliary_outputs["phase_logits"] = self.phase_head(pooled)
        if self.wdl_head is not None:
            auxiliary_outputs["wdl_logits"] = self.wdl_head(pooled)
        if self.legality_head is not None:
            auxiliary_outputs["legality_logits"] = self.legality_head(pooled)
        self._last_auxiliary_outputs = auxiliary_outputs or None
        policy_logits = self.policy_head(pooled)
        value = torch.tanh(self.value_head(pooled)).squeeze(-1)
        return policy_logits, value, aux_loss, router_reports

    def parameter_report(self) -> Dict[str, Any]:
        total = sum(param.numel() for param in self.parameters())
        trainable = sum(param.numel() for param in self.parameters() if param.requires_grad)
        auxiliary_heads: List[str] = []
        if self.phase_head is not None:
            auxiliary_heads.append("phase_head")
        if self.wdl_head is not None:
            auxiliary_heads.append("wdl_head")
        if self.legality_head is not None:
            auxiliary_heads.append("legality_head")
        return {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "policy_head_type": "pooled_global_move_classifier",
            "auxiliary_heads": auxiliary_heads,
        }

    def get_last_world_model_outputs(self) -> Optional[Dict[str, torch.Tensor]]:
        return self._last_world_model_outputs

    def get_last_auxiliary_outputs(self) -> Optional[Dict[str, torch.Tensor]]:
        return self._last_auxiliary_outputs

    def reset_router_state(self, batch_size: int = 1) -> None:
        if self.latent_ode_channel is not None:
            self.latent_ode_channel.reset_state(
                batch_size=batch_size,
                device=self.piece_embed.weight.device,
                dtype=self.piece_embed.weight.dtype,
            )
        for block in self.blocks:
            if getattr(block, "is_moe_layer", False):
                router = getattr(getattr(block, "ff", None), "router", None)
                if router is None:
                    continue
                state = torch.zeros(
                    batch_size,
                    router.hidden_size,
                    router.history_window - 1,
                    device=router.inference_state.device,
                    dtype=router.inference_state.dtype,
                )
                router.set_state(state)


class ChessExampleDataset(torch.utils.data.Dataset):
    def __init__(self, examples: Sequence[ChessExample]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> ChessExample:
        return self.examples[idx]


PHASE_NAMES = {0: "opening", 1: "middlegame", 2: "endgame"}
WDL_CLASS_NAMES = {0: "loss", 1: "draw", 2: "win"}
PHASE_CLASS_COUNT = len(PHASE_NAMES)
WDL_CLASS_COUNT = len(WDL_CLASS_NAMES)


def value_target_to_wdl_class(value: float, draw_threshold: float = 0.20) -> int:
    threshold = max(0.0, float(draw_threshold))
    if value <= -threshold:
        return 0
    if value >= threshold:
        return 2
    return 1


def value_targets_to_wdl_classes(values: torch.Tensor, draw_threshold: float = 0.20) -> torch.Tensor:
    threshold = max(0.0, float(draw_threshold))
    targets = torch.full_like(values, 1, dtype=torch.long)
    targets = targets.masked_fill(values <= -threshold, 0)
    targets = targets.masked_fill(values >= threshold, 2)
    return targets


def piece_to_id(piece: Optional[chess.Piece]) -> int:
    if piece is None:
        return 0
    offset = 0 if piece.color == chess.WHITE else 6
    return offset + piece.piece_type


def material_bucket(board: chess.Board, color: bool) -> int:
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    score = 0
    for piece_type, value in values.items():
        score += len(board.pieces(piece_type, color)) * value
    return min(39, score)


def encode_board_state(board: chess.Board, legal_move_count: Optional[int] = None) -> Tuple[List[int], List[int]]:
    if legal_move_count is None:
        legal_move_count = board.legal_moves.count()
    piece_ids = [piece_to_id(board.piece_at(square)) for square in chess.SQUARES]
    ep_square = board.ep_square
    ep_file = 0 if ep_square is None else chess.square_file(ep_square) + 1
    meta_ids = [
        int(board.turn),
        int(board.has_kingside_castling_rights(chess.WHITE)),
        int(board.has_queenside_castling_rights(chess.WHITE)),
        int(board.has_kingside_castling_rights(chess.BLACK)),
        int(board.has_queenside_castling_rights(chess.BLACK)),
        ep_file,
        min(15, board.halfmove_clock // 4),
        min(31, board.fullmove_number // 2),
        int(board.is_check()),
        min(31, legal_move_count // 2),
        material_bucket(board, chess.WHITE),
        material_bucket(board, chess.BLACK),
    ]
    return piece_ids, meta_ids


def infer_phase(board: chess.Board, ply_idx: int) -> int:
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
    non_pawn_non_king = 0
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        non_pawn_non_king += len(board.pieces(piece_type, chess.WHITE)) + len(board.pieces(piece_type, chess.BLACK))
    if ply_idx <= 18 and queens == 2 and non_pawn_non_king >= 10:
        return 0
    if queens == 0 or non_pawn_non_king <= 6:
        return 2
    return 1


def parse_time_control(tc: str) -> int:
    tc = (tc or "").strip()
    if tc in {"-", "?", ""}:
        return 0
    match = re.match(r"^(\d+)", tc)
    if not match:
        return 0
    return int(match.group(1))


def result_to_value(result: str, turn: bool, ply_idx: int, total_plies: int) -> float:
    raw = 0.0
    if result == "1-0":
        raw = 1.0 if turn == chess.WHITE else -1.0
    elif result == "0-1":
        raw = -1.0 if turn == chess.WHITE else 1.0
    progress = 0.0 if total_plies <= 1 else float(ply_idx) / float(total_plies - 1)
    weight = 0.35 + 0.65 * progress
    return float(raw * weight)


def parse_eval_comment(comment: str) -> Optional[float]:
    comment = comment or ""
    mate_match = re.search(r"\[%eval\s+#(-?\d+)\]", comment)
    if mate_match:
        mate_value = float(mate_match.group(1))
        return max(-1.0, min(1.0, mate_value / 6.0))
    cp_match = re.search(r"\[%eval\s+(-?\d+(?:\.\d+)?)\]", comment)
    if not cp_match:
        return None
    cp = float(cp_match.group(1))
    return max(-1.0, min(1.0, math.tanh(cp / 3.0)))


def build_move_vocab() -> List[str]:
    moves: List[str] = []
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            if from_sq == to_sq:
                continue
            moves.append(chess.square_name(from_sq) + chess.square_name(to_sq))
    promos: List[str] = []
    promo_pieces = ["q", "r", "b", "n"]
    for file_idx in range(8):
        white_from = chess.square(file_idx, 6)
        black_from = chess.square(file_idx, 1)
        for delta in (-1, 0, 1):
            to_file = file_idx + delta
            if 0 <= to_file < 8:
                white_to = chess.square(to_file, 7)
                black_to = chess.square(to_file, 0)
                for promo in promo_pieces:
                    promos.append(chess.square_name(white_from) + chess.square_name(white_to) + promo)
                    promos.append(chess.square_name(black_from) + chess.square_name(black_to) + promo)
    ordered: List[str] = []
    seen = set()
    for item in moves + promos:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    if len(ordered) != len(set(ordered)):
        raise RuntimeError("Move vocabulary contains duplicates")
    return ordered


MOVE_VOCAB = build_move_vocab()
MOVE_TO_ID = {uci: idx for idx, uci in enumerate(MOVE_VOCAB)}
ID_TO_MOVE = {idx: uci for uci, idx in MOVE_TO_ID.items()}
MOVE_VOCAB_HASH = sha256_bytes("\n".join(MOVE_VOCAB).encode("utf-8"))


if len(MOVE_VOCAB) != len(MOVE_TO_ID) or len(MOVE_TO_ID) != len(ID_TO_MOVE):
    raise RuntimeError("Move vocabulary mappings are inconsistent")


def legal_move_ids(board: chess.Board) -> List[int]:
    ids: List[int] = []
    missing: List[str] = []
    for move in board.legal_moves:
        move_id = MOVE_TO_ID.get(move.uci())
        if move_id is None:
            missing.append(move.uci())
        else:
            ids.append(move_id)
    if missing:
        raise RuntimeError(f"Encountered out-of-vocabulary legal moves: {missing[:5]}")
    return ids


def normalized_position_hash(board: chess.Board) -> str:
    fen_parts = board.fen().split(" ")
    normalized = " ".join(fen_parts[:4])
    return sha256_bytes(normalized.encode("utf-8"))


def normalized_game_hash(game: chess.pgn.Game, moves: Sequence[chess.Move]) -> str:
    parts = [
        game.headers.get("White", "?"),
        game.headers.get("Black", "?"),
        game.headers.get("Date", "?"),
        game.headers.get("Result", "?"),
        game.headers.get("TimeControl", "?"),
        " ".join(move.uci() for move in moves),
    ]
    return sha256_bytes("|".join(parts).encode("utf-8"))


def opening_prefix_from_moves(moves: Sequence[chess.Move]) -> str:
    return " ".join(move.uci() for move in moves[:4]) or "empty"


def comment_has_eval_tag(comment: str) -> bool:
    return "[%eval" in (comment or "")


def score_candidate_ply(
    board: chess.Board,
    move: chess.Move,
    comment: str,
    ply_idx: int,
    total_plies: int,
) -> float:
    score = 0.0
    if comment_has_eval_tag(comment):
        score += 4.0
    parsed_eval = parse_eval_comment(comment)
    if parsed_eval is not None:
        score += min(1.5, abs(float(parsed_eval)) * 1.5)
    if board.is_capture(move):
        score += 3.0
    if board.gives_check(move):
        score += 3.0
    if move.promotion is not None:
        score += 4.0
    if board.is_castling(move):
        score += 2.0
    if infer_phase(board, ply_idx) == 2:
        score += 1.25
    if board.legal_moves.count() <= 14:
        score += 1.0
    if ply_idx <= max(6, total_plies // 8):
        piece = board.piece_at(move.from_square)
        if piece is not None and piece.piece_type in {chess.KNIGHT, chess.BISHOP}:
            score += 0.75
    progress = 0.0 if total_plies <= 1 else float(ply_idx) / float(max(1, total_plies - 1))
    score += progress * 0.5
    return score


def select_ply_indices(
    moves: Sequence[Tuple[chess.Move, str]],
    limit: int,
    prefer_eval_positions: bool,
    board: Optional[chess.Board] = None,
) -> List[int]:
    total_plies = len(moves)
    if total_plies <= 0 or limit <= 0:
        return []
    if total_plies <= limit:
        return list(range(total_plies))
    board_state = board.copy(stack=False) if board is not None else chess.Board()
    scored_candidates: List[Tuple[float, int]] = []
    for idx, (move, comment) in enumerate(moves):
        scored_candidates.append((score_candidate_ply(board_state, move, comment, idx, total_plies), idx))
        board_state.push(move)
    picks: List[int] = []
    eval_indices = [idx for idx, (_, comment) in enumerate(moves) if comment_has_eval_tag(comment)] if prefer_eval_positions else []
    scored_indices = [idx for _, idx in sorted(scored_candidates, key=lambda item: (-item[0], item[1]))]
    evenly_spaced = sorted({
        int(round(position))
        for position in np.linspace(0, total_plies - 1, num=min(total_plies, max(limit * 2, limit + 2)))
    })
    for idx in eval_indices + scored_indices + evenly_spaced:
        if 0 <= idx < total_plies and idx not in picks:
            picks.append(idx)
        if len(picks) >= limit:
            break
    return sorted(picks[:limit])


def game_is_usable(game: chess.pgn.Game, cfg: Dict[str, Any]) -> Tuple[bool, str]:
    headers = game.headers
    variant = headers.get("Variant", "Standard").strip().lower()
    if variant not in {"", "standard"}:
        return False, "non_standard"
    if headers.get("WhiteTitle", "").strip().upper() == "BOT" or headers.get("BlackTitle", "").strip().upper() == "BOT":
        return False, "bot_game"
    event = headers.get("Event", "").strip().lower()
    if "rated" not in event:
        return False, "non_rated"
    try:
        white_elo = int(headers.get("WhiteElo", "0") or 0)
        black_elo = int(headers.get("BlackElo", "0") or 0)
    except ValueError:
        return False, "bad_elo"
    if min(white_elo, black_elo) < int(cfg["min_elo"]):
        return False, "low_elo"
    base_seconds = parse_time_control(headers.get("TimeControl", "0+0"))
    if base_seconds < int(cfg["time_control_min_seconds"]) or base_seconds > int(cfg["time_control_max_seconds"]):
        return False, "bad_time_control"
    termination = headers.get("Termination", "").strip().lower()
    if bool(cfg.get("exclude_time_forfeit", True)) and termination == "time forfeit":
        return False, "time_forfeit"
    result = headers.get("Result", "")
    if result not in {"1-0", "0-1", "1/2-1/2"}:
        return False, "bad_result"
    return True, "accepted"


def iter_games_from_pgn_text(text: str) -> Iterator[chess.pgn.Game]:
    handle = io.StringIO(text)
    while True:
        game = chess.pgn.read_game(handle)
        if game is None:
            break
        yield game


def embedded_seed_games() -> List[chess.pgn.Game]:
    return list(iter_games_from_pgn_text(EMBEDDED_SEED_PGN))


CURATED_POSITION_BLUEPRINTS: Tuple[CuratedPositionSpec, ...] = (
    CuratedPositionSpec(
        label="opening_ruy_lopez_development",
        suite="opening",
        game_index=1,
        ply_index=4,
        expected_move_uci="f1b5",
        expected_tags=("development",),
        teaching_focus="Açılışta gelişimi hızlandırıp baskı kurma.",
        commentary_tr="Bu pozisyon açılış gelişimi ve taş aktivasyonu için kullanılır.",
    ),
    CuratedPositionSpec(
        label="opening_king_safety_castle",
        suite="opening",
        game_index=1,
        ply_index=8,
        expected_move_uci="e1g1",
        expected_tags=("castle",),
        teaching_focus="Şah güvenliği ve hızlı kale bağlantısı.",
        commentary_tr="Erken rok örneği; öğretme modunda güvenlik vurgusu için kullanılır.",
    ),
    CuratedPositionSpec(
        label="opening_center_break",
        suite="opening",
        game_index=1,
        ply_index=18,
        expected_move_uci="d2d4",
        expected_tags=("center_control",),
        teaching_focus="Merkez kırışı ve alan kazanımı.",
        commentary_tr="Merkez alanını büyüten klasik piyon kırışı örneği.",
    ),
    CuratedPositionSpec(
        label="tactical_exchange_on_f5",
        suite="tactical",
        game_index=1,
        ply_index=46,
        expected_move_uci="e4f5",
        expected_tags=("capture",),
        teaching_focus="Taktik hesapta doğru alışı seçme.",
        commentary_tr="Taktik sekans içinde doğru taş alışını vurgulayan konum.",
    ),
    CuratedPositionSpec(
        label="endgame_runner_pawn_push",
        suite="endgame",
        game_index=1,
        ply_index=66,
        expected_move_uci="b4b5",
        expected_tags=("positional_choice",),
        teaching_focus="Geçer piyon sürüşü ve oyunsonu dönüşümü.",
        commentary_tr="Oyunsonunda piyon sürerek dönüşüm planı kurma örneği.",
    ),
    CuratedPositionSpec(
        label="blunder_correction_recapture",
        suite="blunder_correction",
        game_index=2,
        ply_index=40,
        expected_move_uci="d4e5",
        expected_tags=("capture",),
        teaching_focus="Rakip hatasını sade ve doğru geri alışla cezalandırma.",
        commentary_tr="Blunder correction seti için temiz geri alış örneği.",
    ),
    CuratedPositionSpec(
        label="endgame_king_recapture",
        suite="endgame",
        game_index=2,
        ply_index=78,
        expected_move_uci="e6d5",
        expected_tags=("capture",),
        teaching_focus="Aktif şah kullanımı ve materyal temizliği.",
        commentary_tr="Oyunsonunda şah aktivitesi ile materyal toplama örneği.",
    ),
    CuratedPositionSpec(
        label="opening_fianchetto_setup",
        suite="opening",
        game_index=3,
        ply_index=6,
        expected_move_uci="g2g3",
        expected_tags=("positional_choice",),
        teaching_focus="Fianchetto planı ve güvenli gelişim.",
        commentary_tr="Açılışta fianchetto planını anlatmak için kullanılır.",
    ),
    CuratedPositionSpec(
        label="center_capture_transition",
        suite="tactical",
        game_index=3,
        ply_index=8,
        expected_move_uci="c4d5",
        expected_tags=("capture",),
        teaching_focus="Merkezde zamanlamalı alış ve yapı değişimi.",
        commentary_tr="Merkezde doğru anda alma kararını örnekler.",
    ),
    CuratedPositionSpec(
        label="tactical_check_entry",
        suite="tactical",
        game_index=3,
        ply_index=66,
        expected_move_uci="g4c8",
        expected_tags=("check",),
        teaching_focus="Kazanan sekansa şah çekerek giriş.",
        commentary_tr="Taktik bitirişe check ile girilen örnek konum.",
    ),
    CuratedPositionSpec(
        label="mating_finish_qxf8",
        suite="tactical",
        game_index=3,
        ply_index=68,
        expected_move_uci="c8f8",
        expected_tags=("capture", "check", "checkmate"),
        teaching_focus="Net mat bitirişi ve forcing move disiplini.",
        commentary_tr="Mat bitirişini teaching ve benchmark yüzeyinde sabitler.",
    ),
)


def mainline_moves(game: chess.pgn.Game) -> List[Tuple[chess.Move, str]]:
    node = game
    moves: List[Tuple[chess.Move, str]] = []
    while node.variations:
        next_node = node.variation(0)
        moves.append((next_node.move, next_node.comment or ""))
        node = next_node
    return moves


def materialize_curated_position_bank() -> List[Dict[str, Any]]:
    seed_games = embedded_seed_games()
    bank: List[Dict[str, Any]] = []
    for blueprint in CURATED_POSITION_BLUEPRINTS:
        game = seed_games[blueprint.game_index - 1]
        moves = mainline_moves(game)
        if blueprint.ply_index >= len(moves):
            raise ChessOnefileError(f"Curated blueprint out of range: {blueprint.label}")
        board = game.board()
        raw_moves = [move for move, _ in moves]
        for ply_idx in range(blueprint.ply_index):
            board.push(moves[ply_idx][0])
        expected_move, comment = moves[blueprint.ply_index]
        if expected_move.uci() != blueprint.expected_move_uci:
            raise ChessOnefileError(
                f"Curated blueprint drift for {blueprint.label}: expected {blueprint.expected_move_uci}, "
                f"got {expected_move.uci()}"
            )
        bank.append(
            {
                "label": blueprint.label,
                "suite": blueprint.suite,
                "board": board.copy(stack=False),
                "expected_move_uci": blueprint.expected_move_uci,
                "expected_tags": list(blueprint.expected_tags),
                "teaching_focus": blueprint.teaching_focus,
                "commentary_tr": blueprint.commentary_tr,
                "phase": infer_phase(board, blueprint.ply_index),
                "opening_prefix": opening_prefix_from_moves(raw_moves),
                "source_game_id": normalized_game_hash(game, raw_moves),
                "source_archive": f"curated_suite::{blueprint.suite}",
                "position_hash": normalized_position_hash(board),
                "value_target": result_to_value(game.headers.get("Result", "1/2-1/2"), board.turn, blueprint.ply_index, len(moves)),
                "source_comment": comment,
            }
        )
    return bank


def build_curated_position_manifest(cfg: Dict[str, Any]) -> Dict[str, Any]:
    bank = materialize_curated_position_bank()
    suite_counts = Counter(item["suite"] for item in bank)
    phase_counts = Counter(PHASE_NAMES[int(item["phase"])] for item in bank)
    return {
        "schema": "chess_curated_position_manifest_v1",
        "script_version": SCRIPT_VERSION,
        "enabled": bool(cfg.get("include_curated_position_suites", True)),
        "repeat_factor": int(cfg.get("curated_position_repeat", 0)),
        "position_count": len(bank),
        "suite_counts": dict(sorted(suite_counts.items())),
        "phase_counts": dict(sorted(phase_counts.items())),
        "labels": [
            {
                "label": item["label"],
                "suite": item["suite"],
                "phase": PHASE_NAMES[int(item["phase"])],
                "expected_move_uci": item["expected_move_uci"],
                "expected_tags": item["expected_tags"],
                "opening_prefix": item["opening_prefix"],
                "teaching_focus": item["teaching_focus"],
            }
            for item in bank
        ],
        "notes": {
            "purpose": "High-signal internal augmentation/eval bank for opening, tactical, endgame, and blunder-correction surfaces.",
            "rating_note": "This bank is for internal augmentation and repeatable smoke/benchmark checks only.",
        },
    }


def build_curated_training_examples(cfg: Dict[str, Any]) -> Tuple[List[ChessExample], Dict[str, Any]]:
    if not bool(cfg.get("include_curated_position_suites", True)):
        return [], {
            "enabled": False,
            "repeat_factor": 0,
            "positions_total": 0,
            "examples_total": 0,
            "suite_counts": {},
        }
    repeat_factor = max(1, int(cfg.get("curated_position_repeat", 1)))
    bank = materialize_curated_position_bank()
    suite_counts = Counter(item["suite"] for item in bank)
    examples: List[ChessExample] = []
    for item in bank:
        board = item["board"]
        legal_ids = legal_move_ids(board)
        target_move_id = MOVE_TO_ID.get(str(item["expected_move_uci"]))
        if target_move_id is None or target_move_id not in legal_ids:
            raise ChessOnefileError(f"Curated training target is illegal or OOV: {item['label']}")
        piece_ids, meta_ids = encode_board_state(board, legal_move_count=len(legal_ids))
        for _ in range(repeat_factor):
            examples.append(
                ChessExample(
                    piece_ids=list(piece_ids),
                    meta_ids=list(meta_ids),
                    legal_move_ids=list(legal_ids),
                    target_move_id=int(target_move_id),
                    value_target=float(item["value_target"]),
                    phase=int(item["phase"]),
                    source_game_id=f"curated::{item['label']}",
                    ply=int(item["phase"]) + 1000,
                    total_plies=1,
                    turn=int(board.turn),
                    has_eval=False,
                    opening_prefix=str(item["opening_prefix"]),
                    value_source="curated_seed_suite",
                    source_archive=str(item["source_archive"]),
                    position_hash=f"{item['position_hash']}::{item['label']}",
                    move_uci=str(item["expected_move_uci"]),
                )
            )
    manifest = {
        "enabled": True,
        "repeat_factor": repeat_factor,
        "positions_total": len(bank),
        "examples_total": len(examples),
        "suite_counts": dict(sorted(suite_counts.items())),
        "labels": [item["label"] for item in bank],
        "notes": {
            "split_policy": "Curated suites are appended to the training split only; holdout and locked-test remain game-split only.",
            "intentional_repeat": True,
        },
    }
    return examples, manifest


def render_curated_position_manifest_md(manifest: Dict[str, Any]) -> str:
    lines = [
        "# Curated Chess Position Manifest",
        "",
        f"- enabled: `{manifest.get('enabled', False)}`",
        f"- repeat_factor: `{manifest.get('repeat_factor', 0)}`",
        f"- position_count: `{manifest.get('position_count', manifest.get('positions_total', 0))}`",
        f"- examples_total: `{manifest.get('examples_total', 0)}`",
        "",
        "## Suite Counts",
    ]
    suite_counts = manifest.get("suite_counts", {})
    for suite_name, count in sorted(suite_counts.items()):
        lines.append(f"- `{suite_name}`: `{count}`")
    lines.extend(["", "## Labels"])
    for item in manifest.get("labels", []):
        if isinstance(item, dict):
            lines.append(
                f"- `{item['label']}` | suite=`{item['suite']}` | phase=`{item['phase']}` | move=`{item['expected_move_uci']}`"
            )
        else:
            lines.append(f"- `{item}`")
    return "\n".join(lines) + "\n"


def choose_archive_urls(urls: Sequence[str], cfg: Dict[str, Any]) -> List[str]:
    count = max(1, min(len(urls), int(cfg.get("download_archive_count", 4))))
    if count >= len(urls):
        return list(urls)
    candidate_indices = sorted({
        int(round(position))
        for position in np.linspace(0, len(urls) - 1, num=count)
    })
    if len(candidate_indices) < count:
        for idx in range(len(urls)):
            if idx not in candidate_indices:
                candidate_indices.append(idx)
            if len(candidate_indices) >= count:
                break
    return [str(urls[idx]) for idx in sorted(candidate_indices[:count])]


def download_archive_slices(
    urls: Sequence[str],
    cfg: Dict[str, Any],
    logger: JSONLLogger,
    cache_root: Path,
) -> List[DownloadSlice]:
    cache_root.mkdir(parents=True, exist_ok=True)
    selected_urls = choose_archive_urls(urls, cfg)
    total_budget_bytes = max(0, int(cfg.get("download_partial_mb", 0))) * 1024 * 1024
    if total_budget_bytes <= 0:
        return []
    per_archive_budget = max(1, total_budget_bytes // max(1, len(selected_urls)))
    timeout = int(cfg.get("download_timeout_sec", 60))
    retries = int(cfg.get("download_retries", 2))
    backoff = float(cfg.get("download_retry_backoff_sec", 2.0))
    allowlist = [str(item).lower() for item in cfg.get("download_content_type_allowlist", [])]

    slices: List[DownloadSlice] = []
    for url_idx, url in enumerate(selected_urls):
        filename = safe_name(Path(url).name) + f".part{url_idx:02d}"
        target = cache_root / filename
        requested_range = f"bytes=0-{per_archive_budget - 1}"
        headers = {"User-Agent": f"{SCRIPT_BASENAME}/{SCRIPT_VERSION}", "Range": requested_range}
        if not bool(cfg.get("auto_download_enabled", True)):
            if target.exists() and target.stat().st_size > 0:
                slices.append(
                    DownloadSlice(
                        url=url,
                        requested_range=requested_range,
                        path=target,
                        bytes_written=target.stat().st_size,
                        sha256=path_sha256(target),
                        response_headers={"source": "cache_only"},
                        http_status=200,
                        content_type="cached",
                    )
                )
                continue
            raise DownloadError(f"Auto-download disabled and cached archive slice is missing: {target}")
        last_error = ""
        for attempt in range(1, retries + 2):
            logger.write("download_start", {"url": url, "target": str(target), "requested_range": requested_range, "attempt": attempt})
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    response_headers = {str(k): str(v) for k, v in response.info().items()}
                    content_type = response_headers.get("Content-Type", "").split(";")[0].strip().lower()
                    if content_type and allowlist and content_type not in allowlist:
                        raise DownloadError(f"Unexpected content type for {url}: {content_type}")
                    with target.open("wb") as handle:
                        bytes_written = 0
                        while True:
                            room = per_archive_budget - bytes_written
                            if room <= 0:
                                break
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            if len(chunk) > room:
                                handle.write(chunk[:room])
                                bytes_written += room
                                break
                            handle.write(chunk)
                            bytes_written += len(chunk)
                if target.stat().st_size <= 0:
                    raise DownloadError(f"Downloaded archive slice is empty: {url}")
                slice_info = DownloadSlice(
                    url=url,
                    requested_range=requested_range,
                    path=target,
                    bytes_written=target.stat().st_size,
                    sha256=path_sha256(target),
                    response_headers=response_headers,
                    http_status=getattr(response, "status", 200),
                    content_type=content_type,
                )
                logger.write(
                    "download_done",
                    {
                        "url": url,
                        "target": str(target),
                        "requested_range": requested_range,
                        "bytes_written": slice_info.bytes_written,
                    },
                )
                slices.append(slice_info)
                break
            except (urllib.error.URLError, TimeoutError, DownloadError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.write("download_error", {"url": url, "attempt": attempt, "error": last_error})
                if attempt <= retries:
                    time.sleep(backoff * attempt)
                    continue
                raise DownloadError(f"Unable to download archive slice from {url}: {last_error}")
    return slices


def iter_games_from_zstd(path: Path, logger: Optional[JSONLLogger] = None) -> Iterator[chess.pgn.Game]:
    try:
        with path.open("rb") as raw:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(raw) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore", newline="")
                while True:
                    try:
                        game = chess.pgn.read_game(text_stream)
                    except (ValueError, UnicodeDecodeError) as exc:
                        if logger is not None:
                            logger.write("pgn_parse_error", {"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
                        break
                    if game is None:
                        break
                    yield game
    except (OSError, zstd.ZstdError) as exc:
        if logger is not None:
            logger.write("archive_read_error", {"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
        return


def build_examples_from_games(
    named_game_sources: Sequence[Tuple[str, Iterable[chess.pgn.Game]]],
    cfg: Dict[str, Any],
    logger: JSONLLogger,
) -> Tuple[List[ChessExample], Dict[str, Any]]:
    examples: List[ChessExample] = []
    seen_game_hashes: set[str] = set()
    seen_position_hashes: set[str] = set()
    drop_reason_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    target_color_counts: Counter[str] = Counter()
    opening_prefix_counts: Counter[str] = Counter()
    value_source_counts: Counter[str] = Counter()
    move_class_counts: Counter[str] = Counter()
    archive_game_counts: Counter[str] = Counter()
    archive_position_counts: Counter[str] = Counter()
    eval_tag_seen = 0
    eval_parse_success = 0
    games_seen = 0
    games_accepted = 0
    duplicated_games = 0
    duplicated_positions = 0

    max_games = int(cfg["max_games"])
    max_positions = int(cfg["max_positions"])
    max_positions_per_game = int(cfg["max_positions_per_game"])

    for source_name, game_iter in named_game_sources:
        for game in game_iter:
            games_seen += 1
            usable, reason = game_is_usable(game, cfg)
            if not usable:
                drop_reason_counts[reason] += 1
                continue

            node = game
            moves: List[Tuple[chess.Move, str]] = []
            while node.variations:
                next_node = node.variation(0)
                moves.append((next_node.move, next_node.comment or ""))
                node = next_node
            if not moves:
                drop_reason_counts["empty_game"] += 1
                continue

            raw_moves = [move for move, _ in moves]
            game_hash = normalized_game_hash(game, raw_moves)
            if bool(cfg.get("dedupe_games", True)) and game_hash in seen_game_hashes:
                duplicated_games += 1
                drop_reason_counts["duplicate_game"] += 1
                continue
            seen_game_hashes.add(game_hash)
            games_accepted += 1
            archive_game_counts[source_name] += 1

            total_plies = len(moves)
            opening_prefix = opening_prefix_from_moves(raw_moves)
            selected_indices = select_ply_indices(
                moves,
                max_positions_per_game,
                bool(cfg.get("prefer_eval_positions", True)),
                board=game.board(),
            )
            board = game.board()
            for ply_idx, (move, comment) in enumerate(moves):
                if ply_idx not in selected_indices:
                    board.push(move)
                    continue

                legal_ids = legal_move_ids(board)
                target_id = MOVE_TO_ID.get(move.uci())
                if target_id is None or target_id not in legal_ids:
                    drop_reason_counts["target_not_legal_or_oov"] += 1
                    board.push(move)
                    continue

                pos_hash = normalized_position_hash(board)
                if bool(cfg.get("dedupe_positions", True)) and pos_hash in seen_position_hashes:
                    duplicated_positions += 1
                    drop_reason_counts["duplicate_position"] += 1
                    board.push(move)
                    continue
                seen_position_hashes.add(pos_hash)

                piece_ids, meta_ids = encode_board_state(board, legal_move_count=len(legal_ids))
                phase = infer_phase(board, ply_idx)
                target_color_counts["white" if board.turn == chess.WHITE else "black"] += 1
                phase_counts[PHASE_NAMES[phase]] += 1
                opening_prefix_counts[opening_prefix] += 1
                archive_position_counts[source_name] += 1

                raw_value = parse_eval_comment(comment)
                if comment_has_eval_tag(comment):
                    eval_tag_seen += 1
                has_eval = raw_value is not None
                if has_eval:
                    eval_parse_success += 1
                    value_target = float(raw_value)
                    value_source = "eval"
                else:
                    value_target = result_to_value(game.headers.get("Result", "1/2-1/2"), board.turn, ply_idx, total_plies)
                    value_source = "result_discounted"
                value_source_counts[value_source] += 1

                if move.promotion is not None:
                    move_class_counts["promotion"] += 1
                elif board.is_castling(move):
                    move_class_counts["castling"] += 1
                elif board.is_en_passant(move):
                    move_class_counts["en_passant"] += 1
                else:
                    move_class_counts["standard"] += 1

                examples.append(
                    ChessExample(
                        piece_ids=piece_ids,
                        meta_ids=meta_ids,
                        legal_move_ids=legal_ids,
                        target_move_id=target_id,
                        value_target=value_target,
                        phase=phase,
                        source_game_id=game_hash,
                        ply=ply_idx,
                        total_plies=total_plies,
                        turn=int(board.turn),
                        has_eval=has_eval,
                        opening_prefix=opening_prefix,
                        value_source=value_source,
                        source_archive=source_name,
                        position_hash=pos_hash,
                        move_uci=move.uci(),
                    )
                )
                if len(examples) >= max_positions:
                    logger.write("dataset_cap_reached", {"positions": len(examples), "games_accepted": games_accepted})
                    break
                board.push(move)
            else:
                if games_accepted >= max_games:
                    logger.write("game_cap_reached", {"games_accepted": games_accepted, "positions": len(examples)})
                    break
                continue
            if len(examples) >= max_positions or games_accepted >= max_games:
                break
        if len(examples) >= max_positions or games_accepted >= max_games:
            break

    if not examples:
        raise DatasetEmptyError("No usable training examples were produced from the configured data sources")

    total_positions = len(examples)
    move_targets_seen = len({item.target_move_id for item in examples})
    data_stats = {
        "games_seen": games_seen,
        "games_accepted": games_accepted,
        "games_rejected": max(0, games_seen - games_accepted),
        "duplicate_games": duplicated_games,
        "duplicate_positions": duplicated_positions,
        "positions_total": total_positions,
        "unique_games": len({item.source_game_id for item in examples}),
        "unique_positions": len({item.position_hash for item in examples}),
        "phase_distribution": dict(phase_counts),
        "target_color_distribution": dict(target_color_counts),
        "opening_distribution_top20": dict(opening_prefix_counts.most_common(20)),
        "value_source_distribution": dict(value_source_counts),
        "move_class_distribution": dict(move_class_counts),
        "drop_reason_counts": dict(drop_reason_counts),
        "archive_game_counts": dict(archive_game_counts),
        "archive_position_counts": dict(archive_position_counts),
        "eval_tag_seen": eval_tag_seen,
        "eval_parse_success": eval_parse_success,
        "eval_tag_seen_rate": round(eval_tag_seen / max(1, total_positions), 6),
        "eval_parse_success_rate": round(eval_parse_success / max(1, total_positions), 6),
        "move_vocab_size": len(MOVE_VOCAB),
        "move_targets_seen": move_targets_seen,
        "move_vocab_coverage_rate": round(move_targets_seen / max(1, len(MOVE_VOCAB)), 6),
    }
    return examples, data_stats


def maybe_collect_dataset(cfg: Dict[str, Any], layout: ArtifactLayout, logger: JSONLLogger) -> Tuple[List[ChessExample], Dict[str, Any]]:
    provenance: Dict[str, Any] = {
        "mode": "embedded_seed" if bool(cfg.get("offline_seed_only", False)) else "multi_archive_partial",
        "script_version": SCRIPT_VERSION,
        "move_vocab_hash": MOVE_VOCAB_HASH,
        "urls": list(cfg.get("lichess_urls", [])),
        "sampling_strategy": "multi_archive_spread_prefix_ranges_for_zstd_streamability",
    }
    if bool(cfg.get("offline_seed_only", False)):
        examples, stats = build_examples_from_games([("embedded_seed", embedded_seed_games())], cfg, logger)
        provenance.update({
            "embedded_seed": True,
            "download_slices": [],
            "data_stats": stats,
        })
        return examples, provenance

    cache_root = Path(str(cfg["cache_root"]))
    slices = download_archive_slices(cfg.get("lichess_urls", DEFAULT_LICHESS_URLS), cfg, logger, cache_root)
    named_sources: List[Tuple[str, Iterable[chess.pgn.Game]]] = []
    for item in slices:
        named_sources.append((Path(item.path).name, iter_games_from_zstd(item.path, logger)))
    examples, stats = build_examples_from_games(named_sources, cfg, logger)
    provenance.update({
        "embedded_seed": False,
        "download_slices": [item.to_dict() for item in slices],
        "data_stats": stats,
    })
    return examples, provenance


def split_examples_by_game(examples: Sequence[ChessExample], cfg: Dict[str, Any]) -> Tuple[Dict[str, List[ChessExample]], Dict[str, Any]]:
    grouped: DefaultDict[str, List[ChessExample]] = defaultdict(list)
    for example in examples:
        grouped[example.source_game_id].append(example)
    game_ids = list(grouped.keys())
    rng = random.Random(int(cfg["seed"]))
    rng.shuffle(game_ids)

    val_fraction = float(cfg.get("val_fraction", 0.12))
    test_fraction = float(cfg.get("test_fraction", 0.08))
    total_games = len(game_ids)
    test_count = int(round(total_games * test_fraction))
    val_count = int(round(total_games * val_fraction))

    if total_games >= 3:
        test_count = min(max(1, test_count), max(1, total_games - 2))
        val_count = min(max(1, val_count), max(1, total_games - test_count - 1))
    else:
        test_count = 0
        val_count = 1 if total_games > 1 else 0

    test_ids = set(game_ids[:test_count])
    val_ids = set(game_ids[test_count:test_count + val_count])
    train_ids = set(game_ids[test_count + val_count:])
    if not train_ids and val_ids:
        moved = next(iter(val_ids))
        val_ids.remove(moved)
        train_ids.add(moved)

    splits = {"train": [], "val": [], "locked_test": []}
    for game_id, items in grouped.items():
        if game_id in train_ids:
            splits["train"].extend(items)
        elif game_id in val_ids:
            splits["val"].extend(items)
        else:
            splits["locked_test"].extend(items)

    train_game_ids = {item.source_game_id for item in splits["train"]}
    val_game_ids = {item.source_game_id for item in splits["val"]}
    test_game_ids = {item.source_game_id for item in splits["locked_test"]}
    overlap = {
        "train_val": sorted(train_game_ids & val_game_ids),
        "train_test": sorted(train_game_ids & test_game_ids),
        "val_test": sorted(val_game_ids & test_game_ids),
    }
    manifest = {
        "counts": {
            "games_total": total_games,
            "games_train": len(train_game_ids),
            "games_val": len(val_game_ids),
            "games_locked_test": len(test_game_ids),
            "examples_train": len(splits["train"]),
            "examples_val": len(splits["val"]),
            "examples_locked_test": len(splits["locked_test"]),
        },
        "fractions": {"val_fraction": val_fraction, "test_fraction": test_fraction},
        "overlap": overlap,
        "train_game_ids_hash": sha256_bytes("\n".join(sorted(train_game_ids)).encode("utf-8")),
        "val_game_ids_hash": sha256_bytes("\n".join(sorted(val_game_ids)).encode("utf-8")),
        "locked_test_game_ids_hash": sha256_bytes("\n".join(sorted(test_game_ids)).encode("utf-8")),
    }
    return splits, manifest


def build_curriculum_stages(train_examples: Sequence[ChessExample], cfg: Dict[str, Any]) -> Tuple[List[Tuple[str, List[ChessExample]]], List[int]]:
    full_train = list(train_examples)
    if not full_train:
        return [("empty", [])], [0]
    if not bool(cfg.get("curriculum_enabled", True)):
        max_steps = int(cfg["max_steps"])
        return [("full_train", full_train)], [max_steps]

    stage1 = [item for item in full_train if item.phase == 0]
    stage2 = [item for item in full_train if item.phase in {0, 1}]
    stage3 = full_train
    stages: List[Tuple[str, List[ChessExample]]] = []
    for name, data in (("stage1_opening_clean", stage1), ("stage2_opening_middlegame", stage2), ("stage3_full_train", stage3)):
        if data:
            stages.append((name, data))
    if not stages:
        stages = [("full_train", full_train)]

    fracs = list(cfg.get("curriculum_stage_fracs", [0.20, 0.30, 0.50]))
    if len(fracs) < len(stages):
        fracs.extend([0.0] * (len(stages) - len(fracs)))
    fracs = fracs[:len(stages)]
    total_frac = sum(fracs) or 1.0
    fracs = [value / total_frac for value in fracs]
    max_steps = int(cfg["max_steps"])
    stage_caps: List[int] = []
    running = 0
    for idx, frac in enumerate(fracs):
        if idx == len(fracs) - 1:
            running = max_steps
        else:
            running += int(round(max_steps * frac))
        stage_caps.append(min(max_steps, running))
    if stage_caps:
        stage_caps[-1] = max_steps
    return stages, stage_caps


def pick_device(cfg: Dict[str, Any]) -> torch.device:
    return torch.device(str(cfg["device"]))


def maybe_enable_compile(model: nn.Module, cfg: Dict[str, Any], logger: JSONLLogger) -> Tuple[nn.Module, Dict[str, Any]]:
    policy = str(cfg.get("compile_policy", "off"))
    report = {"policy": policy, "attempted": False, "compiled": False, "reason": "disabled"}
    if policy == "off":
        return model, report
    if not hasattr(torch, "compile"):
        report["reason"] = "torch_compile_unavailable"
        return model, report
    if str(cfg.get("device")) != "cuda":
        report["reason"] = "non_cuda_device"
        return model, report
    report["attempted"] = True
    try:
        model = torch.compile(model, mode="max-autotune" if policy == "aggressive" else "default")  # type: ignore[attr-defined]
        report["compiled"] = True
        report["reason"] = "ok"
        return model, report
    except Exception as exc:  # pragma: no cover - compile availability varies
        logger.write("compile_fallback", {"error": str(exc)})
        report["reason"] = f"fallback:{type(exc).__name__}"
        return model, report


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        betas=(0.9, 0.95),
    )


def lr_for_step(step: int, cfg: Dict[str, Any]) -> float:
    warmup = max(1, int(cfg["warmup_steps"]))
    total = max(warmup + 1, int(cfg["max_steps"]))
    if step < warmup:
        return float(step + 1) / float(warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def apply_optimizer_lr(optimizer: torch.optim.Optimizer, factor: float, cfg: Dict[str, Any]) -> float:
    base_lr = float(cfg["learning_rate"])
    lr = base_lr * factor
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def collate_examples(batch: Sequence[ChessExample]) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    piece_ids = torch.tensor([item.piece_ids for item in batch], dtype=torch.long)
    meta_ids = torch.tensor([item.meta_ids for item in batch], dtype=torch.long)
    move_targets = torch.tensor([item.target_move_id for item in batch], dtype=torch.long)
    value_targets = torch.tensor([item.value_target for item in batch], dtype=torch.float32)
    phases = torch.tensor([item.phase for item in batch], dtype=torch.long)
    vocab_size = len(MOVE_VOCAB)
    legal_mask = torch.zeros(batch_size, vocab_size, dtype=torch.bool)
    for row_idx, item in enumerate(batch):
        legal_mask[row_idx, item.legal_move_ids] = True
    return {
        "piece_ids": piece_ids,
        "meta_ids": meta_ids,
        "move_targets": move_targets,
        "value_targets": value_targets,
        "legal_mask": legal_mask,
        "phases": phases,
    }


def compute_prediction_metrics(
    logits: torch.Tensor,
    masked_logits: torch.Tensor,
    legal_mask: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    raw_top1 = logits.argmax(dim=-1)
    raw_topk = torch.topk(logits, k=min(5, logits.size(-1)), dim=-1).indices
    masked_top1 = masked_logits.argmax(dim=-1)
    masked_topk = torch.topk(masked_logits, k=min(5, masked_logits.size(-1)), dim=-1).indices

    raw_top1_is_legal = legal_mask.gather(1, raw_top1.unsqueeze(-1)).squeeze(-1).float().mean().item()
    raw_topk_contains_legal = legal_mask.gather(1, raw_topk).any(dim=-1).float().mean().item()
    masked_policy_accuracy = (masked_top1 == targets).float().mean().item()
    masked_top5_accuracy = (masked_topk == targets.unsqueeze(-1)).any(dim=-1).float().mean().item()
    return {
        "raw_top1_is_legal_rate": float(raw_top1_is_legal),
        "raw_topk_contains_legal_rate": float(raw_topk_contains_legal),
        "masked_policy_accuracy": float(masked_policy_accuracy),
        "masked_top5_accuracy": float(masked_top5_accuracy),
    }


def compute_loss(
    model: ChessPolicyValueNet,
    batch: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
    loss, metrics, router_reports, _, _ = forward_batch_metrics(model, batch, cfg)
    return loss, metrics, router_reports


def forward_batch_metrics(
    model: ChessPolicyValueNet,
    batch: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any], torch.Tensor, torch.Tensor]:
    logits, value_pred, aux_loss, router_reports = model(batch["piece_ids"], batch["meta_ids"])
    auxiliary_outputs = model.get_last_auxiliary_outputs() if hasattr(model, "get_last_auxiliary_outputs") else {}
    masked_logits = logits.masked_fill(~batch["legal_mask"], -1e9)
    policy_loss = F.cross_entropy(masked_logits, batch["move_targets"])
    value_loss = F.mse_loss(value_pred, batch["value_targets"])
    aux_coeff = 0.01
    value_coeff = 0.25
    loss = policy_loss + value_coeff * value_loss + aux_coeff * aux_loss
    metrics = {
        "loss": float(loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "value_loss": float(value_loss.detach().item()),
        "aux_loss": float(aux_loss.detach().item()),
        **compute_prediction_metrics(logits.detach(), masked_logits.detach(), batch["legal_mask"], batch["move_targets"]),
    }
    phase_logits = auxiliary_outputs.get("phase_logits") if auxiliary_outputs else None
    if phase_logits is not None:
        phase_loss = F.cross_entropy(phase_logits, batch["phases"])
        loss = loss + float(model.arch_cfg.phase_loss_coef) * phase_loss
        phase_pred = phase_logits.argmax(dim=-1)
        metrics["phase_loss"] = float(phase_loss.detach().item())
        metrics["phase_accuracy"] = float((phase_pred == batch["phases"]).float().mean().item())
    wdl_logits = auxiliary_outputs.get("wdl_logits") if auxiliary_outputs else None
    if wdl_logits is not None:
        wdl_targets = value_targets_to_wdl_classes(batch["value_targets"], model.arch_cfg.wdl_draw_threshold)
        wdl_loss = F.cross_entropy(wdl_logits, wdl_targets)
        loss = loss + float(model.arch_cfg.wdl_loss_coef) * wdl_loss
        wdl_pred = wdl_logits.argmax(dim=-1)
        metrics["wdl_loss"] = float(wdl_loss.detach().item())
        metrics["wdl_accuracy"] = float((wdl_pred == wdl_targets).float().mean().item())
    legality_logits = auxiliary_outputs.get("legality_logits") if auxiliary_outputs else None
    if legality_logits is not None:
        legality_targets = batch["legal_mask"].float()
        positive_count = legality_targets.sum().clamp_min(1.0)
        total_count = legality_targets.new_tensor(float(legality_targets.numel()))
        negative_count = (total_count - positive_count).clamp_min(1.0)
        pos_weight_value = min(float((negative_count / positive_count).item()), float(model.arch_cfg.legality_pos_weight_cap))
        pos_weight = legality_targets.new_tensor(pos_weight_value)
        legality_loss = F.binary_cross_entropy_with_logits(legality_logits, legality_targets, pos_weight=pos_weight)
        loss = loss + float(model.arch_cfg.legality_loss_coef) * legality_loss
        legality_top1 = legality_logits.argmax(dim=-1)
        legality_top1_is_legal = batch["legal_mask"].gather(1, legality_top1.unsqueeze(-1)).squeeze(-1).float().mean().item()
        legality_probs = torch.sigmoid(legality_logits.detach())
        metrics["legality_loss"] = float(legality_loss.detach().item())
        metrics["legality_head_top1_is_legal_rate"] = float(legality_top1_is_legal)
        metrics["legality_head_legal_mean"] = float(legality_probs[batch["legal_mask"]].mean().item())
        metrics["legality_head_illegal_mean"] = float(legality_probs[~batch["legal_mask"]].mean().item())
    metrics["loss"] = float(loss.detach().item())
    return loss, metrics, router_reports, logits, masked_logits


def merge_metric_sums(sums: Dict[str, float], metrics: Dict[str, float]) -> None:
    for key, value in metrics.items():
        sums[key] = sums.get(key, 0.0) + float(value)


def merge_metric_sums_weighted(sums: Dict[str, float], metrics: Dict[str, float], weight: int) -> None:
    if weight <= 0:
        return
    for key, value in metrics.items():
        sums[key] = sums.get(key, 0.0) + float(value) * float(weight)


def summarize_metric_sums(sums: Dict[str, float], count: int) -> Dict[str, float]:
    if count <= 0:
        return {key: 0.0 for key in sums}
    return {key: value / count for key, value in sums.items()}


@torch.no_grad()
def evaluate_model(
    model: ChessPolicyValueNet,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: Dict[str, Any],
    max_batches: int = 0,
) -> Dict[str, Any]:
    model.eval()
    sums: Dict[str, float] = {}
    router_entropy_values: List[float] = []
    phase_sums: Dict[str, Dict[str, float]] = defaultdict(dict)
    phase_counts: Counter[str] = Counter()
    batch_count = 0
    example_count = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = batch_to_device(batch, device)
        _, metrics, router_reports, logits, masked_logits = forward_batch_metrics(model, batch, cfg)
        batch_examples = int(batch["piece_ids"].size(0))
        merge_metric_sums_weighted(sums, metrics, batch_examples)
        batch_count += 1
        example_count += batch_examples
        for router_stats in router_reports.values():
            if "router_entropy" in router_stats:
                router_entropy_values.append(float(router_stats["router_entropy"]))
        for phase_value in (0, 1, 2):
            phase_mask = batch["phases"] == phase_value
            if not bool(phase_mask.any()):
                continue
            phase_name = PHASE_NAMES[int(phase_value)]
            phase_examples = int(phase_mask.sum().item())
            phase_counts[phase_name] += phase_examples
            phase_logits = logits[phase_mask]
            phase_masked = masked_logits[phase_mask]
            phase_legal = batch["legal_mask"][phase_mask]
            phase_targets = batch["move_targets"][phase_mask]
            phase_metrics = compute_prediction_metrics(phase_logits, phase_masked, phase_legal, phase_targets)
            merge_metric_sums_weighted(phase_sums[phase_name], phase_metrics, phase_examples)
    model.train()
    overall = summarize_metric_sums(sums, example_count)
    per_phase = {phase_name: summarize_metric_sums(metrics, max(1, phase_counts[phase_name])) for phase_name, metrics in phase_sums.items()}
    return {
        "batches_evaluated": batch_count,
        "examples_evaluated": example_count,
        "metrics": overall,
        "per_phase": per_phase,
        "router_entropy_mean": float(sum(router_entropy_values) / len(router_entropy_values)) if router_entropy_values else 0.0,
    }


def extract_raw_vs_masked_metrics(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    metrics = dict(evaluation.get("metrics", {}))
    return {
        "checked_examples": int(evaluation.get("examples_evaluated", 0)),
        "raw_top1_is_legal_rate": float(metrics.get("raw_top1_is_legal_rate", 0.0)),
        "raw_topk_contains_legal_rate": float(metrics.get("raw_topk_contains_legal_rate", 0.0)),
        "masked_policy_accuracy": float(metrics.get("masked_policy_accuracy", 0.0)),
        "masked_top5_accuracy": float(metrics.get("masked_top5_accuracy", 0.0)),
        "per_phase": evaluation.get("per_phase", {}),
        "note": "Raw legality and masked accuracy are reported separately. Replay/demo output is not a strength claim.",
    }


def get_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    with contextlib.suppress(Exception):
        random.setstate(state["python"])
    with contextlib.suppress(Exception):
        np.random.set_state(state["numpy"])
    with contextlib.suppress(Exception):
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        with contextlib.suppress(Exception):
            torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    model: ChessPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    path: Path,
    step: int,
    cfg: Dict[str, Any],
    metrics: Dict[str, Any],
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "script_version": SCRIPT_VERSION,
        "step": step,
        "config": cfg,
        "metrics": metrics,
        "best_val_loss": best_val_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "move_vocab_hash": MOVE_VOCAB_HASH,
        "move_vocab_size": len(MOVE_VOCAB),
        "rng_state": get_rng_state(),
    }
    torch.save(payload, path)


def load_checkpoint(
    checkpoint_path: Path,
    model: ChessPolicyValueNet,
    optimizer: Optional[torch.optim.Optimizer] = None,
    restore_optimizer: bool = True,
) -> ResumeState:
    if not checkpoint_path.exists():
        raise ResumeCheckpointError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("move_vocab_hash") != MOVE_VOCAB_HASH:
        raise ResumeCheckpointError("Checkpoint move vocabulary hash does not match the current onefile")
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and restore_optimizer and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"])
    return ResumeState(
        step=int(checkpoint.get("step", 0)),
        best_val_loss=float(checkpoint.get("best_val_loss", float("inf"))),
        metrics=dict(checkpoint.get("metrics", {})),
        checkpoint_path=checkpoint_path,
    )


def infer_existing_run_dir_from_resume(resume_from: str) -> Optional[Path]:
    if not resume_from:
        return None
    checkpoint_path = Path(resume_from).expanduser().resolve()
    if checkpoint_path.parent.name != "checkpoints":
        return None
    run_dir = checkpoint_path.parent.parent
    if not run_dir.exists():
        return None
    return run_dir


def make_layout(cfg: Dict[str, Any], existing_run_dir: Optional[Path] = None) -> ArtifactLayout:
    desktop = detect_desktop_dir()
    root = Path(str(cfg["artifact_root"]))
    delivery_mode = bool(cfg.get("delivery_mode", False))
    if existing_run_dir is not None and existing_run_dir.parent.name == "runs":
        root = existing_run_dir.parent.parent
    root.mkdir(parents=True, exist_ok=True)
    runs_root = root / "runs" if delivery_mode else root
    final_root = root / "final" if delivery_mode else desktop
    runs_root.mkdir(parents=True, exist_ok=True)
    final_root.mkdir(parents=True, exist_ok=True)
    if existing_run_dir is not None:
        run_dir = existing_run_dir
        run_id = existing_run_dir.name.removeprefix(f"{DELIVERY_PREFIX}_") or datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = runs_root / f"{DELIVERY_PREFIX}_{run_id}"
    logs_dir = run_dir / "logs"
    reports_dir = run_dir / "reports"
    checkpoints_dir = run_dir / "checkpoints"
    export_dir = run_dir / "exports"
    benchmark_dir = run_dir / "benchmarks"
    final_zip = final_root / f"{cfg['result_prefix']}_{run_id}.zip"
    final_sha = final_root / f"{cfg['result_prefix']}_{run_id}.zip.sha256"
    for path in (run_dir, logs_dir, reports_dir, checkpoints_dir, export_dir, benchmark_dir):
        path.mkdir(parents=True, exist_ok=True)
    return ArtifactLayout(
        run_id=run_id,
        root=root,
        run_dir=run_dir,
        logs_dir=logs_dir,
        reports_dir=reports_dir,
        checkpoints_dir=checkpoints_dir,
        export_dir=export_dir,
        benchmark_dir=benchmark_dir,
        desktop_dir=desktop,
        final_zip_path=final_zip,
        final_sha_path=final_sha,
    )


def prepare_layout(cfg: Dict[str, Any]) -> ArtifactLayout:
    existing_run_dir: Optional[Path] = None
    if str(cfg.get("mode", "")) == "package":
        existing_run_dir = infer_existing_run_dir_from_resume(str(cfg.get("resume_from", "")))
    return make_layout(cfg, existing_run_dir=existing_run_dir)


def make_loader(
    examples: Sequence[ChessExample],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> torch.utils.data.DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        ChessExampleDataset(examples),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_examples,
        drop_last=False,
        generator=generator,
    )


def stage_index_for_step(step: int, stage_caps: Sequence[int]) -> int:
    for idx, cap in enumerate(stage_caps):
        if step < cap:
            return idx
    return max(0, len(stage_caps) - 1)


def training_loop(
    model: ChessPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    train_examples: Sequence[ChessExample],
    val_examples: Sequence[ChessExample],
    cfg: Dict[str, Any],
    layout: ArtifactLayout,
    logger: JSONLLogger,
    start_step: int = 0,
    best_val_loss: float = float("inf"),
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Path, Path]:
    device = pick_device(cfg)
    stage_data, stage_caps = build_curriculum_stages(train_examples, cfg)
    val_loader = make_loader(
        val_examples,
        batch_size=int(cfg["eval_batch_size"]),
        shuffle=False,
        num_workers=0,
        seed=int(cfg["seed"]) + 999,
    )

    latest_ckpt = layout.checkpoints_dir / "latest.pt"
    best_ckpt = layout.checkpoints_dir / "best_by_val_loss.pt"
    compatibility_best = layout.checkpoints_dir / "best_model.pt"
    curve_rows: List[Dict[str, Any]] = []

    bf16_autocast_enabled = bool(cfg.get("use_bf16", False)) and device.type == "cuda"
    autocast_device = device.type if device.type in {"cuda", "cpu"} else "cpu"
    grad_accum_steps = max(1, int(cfg.get("grad_accum_steps", 1)))
    max_steps = int(cfg["max_steps"])
    start_time = time.time()
    global_step = int(start_step)
    optimizer.zero_grad(set_to_none=True)
    last_checkpoint_at = start_step
    active_stage_index = -1
    active_stage_name = ""
    current_stage_examples: Sequence[ChessExample] = train_examples
    stage_loader: Optional[torch.utils.data.DataLoader] = None
    stage_iterator: Optional[Iterator[Dict[str, torch.Tensor]]] = None
    stage_epoch = 0
    last_val_eval: Dict[str, Any] = {"status": "not_run", "metrics": {}}
    last_train_row: Dict[str, Any] = {}

    while global_step < max_steps:
        elapsed_hours = (time.time() - start_time) / 3600.0
        if elapsed_hours >= float(cfg["max_wall_hours"]):
            logger.write("training_stop", {"reason": "wall_clock_limit", "step": global_step, "elapsed_hours": elapsed_hours})
            break

        stage_idx = stage_index_for_step(global_step, stage_caps)
        if stage_idx != active_stage_index or stage_iterator is None:
            active_stage_index = stage_idx
            active_stage_name, current_stage_examples = stage_data[stage_idx]
            stage_loader = make_loader(
                current_stage_examples,
                batch_size=int(cfg["batch_size"]),
                shuffle=True,
                num_workers=int(cfg.get("num_workers", 0)),
                seed=int(cfg["seed"]) + stage_idx + stage_epoch,
            )
            stage_iterator = iter(stage_loader)
            logger.write(
                "curriculum_stage",
                {
                    "stage_index": stage_idx,
                    "stage_name": active_stage_name,
                    "step": global_step,
                    "stage_cap": stage_caps[stage_idx],
                    "examples": len(current_stage_examples),
                },
            )

        assert stage_iterator is not None
        try:
            batch = next(stage_iterator)
        except StopIteration:
            stage_epoch += 1
            stage_loader = make_loader(
                current_stage_examples,
                batch_size=int(cfg["batch_size"]),
                shuffle=True,
                num_workers=int(cfg.get("num_workers", 0)),
                seed=int(cfg["seed"]) + stage_idx + stage_epoch,
            )
            stage_iterator = iter(stage_loader)
            batch = next(stage_iterator)

        batch = batch_to_device(batch, device)
        step_start = time.time()
        lr = apply_optimizer_lr(optimizer, lr_for_step(global_step, cfg), cfg)
        try:
            with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=bf16_autocast_enabled):
                loss, metrics, router_reports = compute_loss(model, batch, cfg)
            if not torch.isfinite(loss):
                raise NonFiniteLossError(f"Non-finite loss detected at step {global_step}: {float(loss.detach().item())}")
            (loss / grad_accum_steps).backward()
            do_optimizer_step = ((global_step + 1) % grad_accum_steps == 0)
            grad_norm_value = 0.0
            if do_optimizer_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
                grad_norm_value = float(grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            elapsed_sec = time.time() - step_start
            positions_in_batch = int(batch["piece_ids"].size(0))
            peak_vram_mb = 0.0
            if device.type == "cuda":
                peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
            row = {
                "step": global_step + 1,
                "split": "train",
                "stage": active_stage_name,
                "loss": metrics["loss"],
                "policy_loss": metrics["policy_loss"],
                "value_loss": metrics["value_loss"],
                "aux_loss": metrics["aux_loss"],
                "raw_top1_is_legal_rate": metrics["raw_top1_is_legal_rate"],
                "raw_topk_contains_legal_rate": metrics["raw_topk_contains_legal_rate"],
                "masked_policy_accuracy": metrics["masked_policy_accuracy"],
                "masked_top5_accuracy": metrics["masked_top5_accuracy"],
                "elapsed_sec": round(time.time() - start_time, 4),
                "lr": lr,
                "grad_norm": grad_norm_value,
                "steps_per_sec": 0.0 if elapsed_sec <= 0 else 1.0 / elapsed_sec,
                "examples_per_sec": 0.0 if elapsed_sec <= 0 else positions_in_batch / elapsed_sec,
                "peak_vram_mb": peak_vram_mb,
            }
            if router_reports:
                entropies = [float(item.get("router_entropy", 0.0)) for item in router_reports.values() if item]
                row["router_entropy"] = float(sum(entropies) / len(entropies)) if entropies else 0.0
            curve_rows.append(row)
            last_train_row = dict(row)
            if global_step == start_step or (global_step + 1) % 25 == 0:
                logger.write("train_step", row)
        except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - depends on hardware
            if torch.cuda.is_available():
                with contextlib.suppress(Exception):
                    torch.cuda.empty_cache()
            logger.write("oom_event", {"step": global_step + 1, "error": str(exc)})
            raise TrainingOOMError(f"CUDA OOM at step {global_step + 1}: {exc}") from exc

        global_step += 1

        if global_step % int(cfg["eval_interval"]) == 0 or global_step == 1:
            val_eval = evaluate_model(
                model,
                val_loader,
                device,
                cfg,
                max_batches=int(cfg.get("training_eval_batches", 16)),
            )
            val_row = {
                "step": global_step,
                "split": "val",
                "stage": active_stage_name,
                "elapsed_sec": round(time.time() - start_time, 4),
                **val_eval["metrics"],
                "lr": lr,
                "grad_norm": 0.0,
                "steps_per_sec": 0.0,
                "examples_per_sec": 0.0,
                "peak_vram_mb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0,
            }
            curve_rows.append(val_row)
            last_val_eval = dict(val_eval)
            logger.write("eval_step", {"step": global_step, **val_eval["metrics"]})
            current_val_loss = float(val_eval["metrics"].get("loss", 0.0))
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                save_checkpoint(model, optimizer, best_ckpt, global_step, cfg, val_eval, best_val_loss)
                shutil.copy2(best_ckpt, compatibility_best)
        if global_step - last_checkpoint_at >= int(cfg["checkpoint_interval"]):
            save_checkpoint(model, optimizer, latest_ckpt, global_step, cfg, {"type": "latest", "step": global_step}, best_val_loss)
            maybe_write_midrun_training_snapshots(
                model,
                cfg,
                device,
                layout,
                logger,
                step=global_step,
                latest_train_row=last_train_row,
                latest_val_eval=last_val_eval,
            )
            last_checkpoint_at = global_step

    save_checkpoint(model, optimizer, latest_ckpt, global_step, cfg, {"type": "latest", "step": global_step}, best_val_loss)
    if not best_ckpt.exists():
        shutil.copy2(latest_ckpt, best_ckpt)
        shutil.copy2(best_ckpt, compatibility_best)
    summary = {
        "steps_completed": global_step,
        "best_val_loss": best_val_loss,
        "latest_checkpoint": str(latest_ckpt),
        "best_checkpoint": str(best_ckpt),
    }
    return summary, curve_rows, latest_ckpt, best_ckpt


def _find_stockfish_binary(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    suffix = ".exe" if platform.system() == "Windows" else ""
    candidates: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if "stockfish" not in name:
            continue
        if suffix and not name.endswith(suffix):
            continue
        if path.suffix.lower() in {".zip", ".txt", ".json", ".md"}:
            continue
        candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda item: (len(item.parts), item.name.lower()))
    return candidates[0]


def _stockfish_asset_score(name: str, system_name: str) -> int:
    lowered = name.lower()
    if any(token in lowered for token in ("source", "src", "android", "wasm", "browser", "armv7", "appimage")):
        return -100
    score = 0
    if system_name == "Windows":
        if "windows" in lowered or "win" in lowered:
            score += 20
        if lowered.endswith(".zip"):
            score += 8
        if any(token in lowered for token in ("x64", "x86-64", "x86_64", "64")):
            score += 6
        if any(token in lowered for token in ("avx2", "modern", "bmi2")):
            score += 3
        if lowered.endswith(".exe"):
            score += 2
    elif system_name == "Linux":
        if "linux" in lowered or "ubuntu" in lowered:
            score += 20
        if lowered.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
            score += 8
    elif system_name == "Darwin":
        if any(token in lowered for token in ("mac", "macos", "osx")):
            score += 20
        if lowered.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
            score += 8
    return score


def _download_to_path(url: str, path: Path, timeout: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": f"{SCRIPT_BASENAME}/{SCRIPT_VERSION}"})
    with urllib.request.urlopen(request, timeout=timeout) as response, path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def _fetch_stockfish_binary(cfg: Dict[str, Any], logger: Optional[JSONLLogger] = None) -> Optional[str]:
    if not bool(cfg.get("stockfish_auto_fetch", True)):
        return None
    if not bool(cfg.get("auto_download_enabled", True)):
        return None
    cache_root = Path(str(cfg.get("stockfish_cache_root", Path(str(cfg["cache_root"])) / "stockfish")))
    current_binary = _find_stockfish_binary(cache_root / "current")
    if current_binary is not None:
        return str(current_binary)
    release_api = str(cfg.get("stockfish_release_api", "")).strip()
    if not release_api:
        return None
    timeout = max(5, int(cfg.get("stockfish_download_timeout_sec", 60)))
    try:
        request = urllib.request.Request(release_api, headers={"User-Agent": f"{SCRIPT_BASENAME}/{SCRIPT_VERSION}"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            release_payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        if logger is not None:
            logger.write("stockfish_fetch_error", {"stage": "release_metadata", "error": f"{type(exc).__name__}: {exc}"})
        return None
    system_name = platform.system()
    assets = release_payload.get("assets", []) if isinstance(release_payload, dict) else []
    best_asset: Optional[Dict[str, Any]] = None
    best_score = -10**9
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name", ""))
        score = _stockfish_asset_score(name, system_name)
        if score > best_score:
            best_score = score
            best_asset = asset
    if best_asset is None or best_score < 0:
        if logger is not None:
            logger.write("stockfish_fetch_error", {"stage": "asset_selection", "system": system_name, "asset_count": len(assets)})
        return None
    asset_name = str(best_asset.get("name", "")).strip() or "stockfish_download.zip"
    asset_url = str(best_asset.get("browser_download_url", "")).strip()
    if not asset_url:
        if logger is not None:
            logger.write("stockfish_fetch_error", {"stage": "asset_url_missing", "asset_name": asset_name})
        return None
    release_tag = str(release_payload.get("tag_name", "latest")).strip() or "latest"
    release_root = cache_root / "releases" / safe_name(release_tag)
    release_root.mkdir(parents=True, exist_ok=True)
    archive_path = release_root / asset_name
    extract_root = release_root / "extracted"
    try:
        if not archive_path.exists():
            if logger is not None:
                logger.write("stockfish_fetch_start", {"asset_name": asset_name, "asset_url": asset_url, "target": str(archive_path)})
            _download_to_path(asset_url, archive_path, timeout)
        if not extract_root.exists():
            extract_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive_path) as bundle:
                bundle.extractall(extract_root)
    except Exception as exc:
        if logger is not None:
            logger.write("stockfish_fetch_error", {"stage": "download_or_extract", "asset_name": asset_name, "error": f"{type(exc).__name__}: {exc}"})
        return None
    extracted_binary = _find_stockfish_binary(extract_root)
    if extracted_binary is None:
        if logger is not None:
            logger.write("stockfish_fetch_error", {"stage": "binary_discovery", "extract_root": str(extract_root)})
        return None
    current_root = cache_root / "current"
    current_root.mkdir(parents=True, exist_ok=True)
    current_copy = current_root / extracted_binary.name
    shutil.copy2(extracted_binary, current_copy)
    manifest = {
        "fetched_at_utc": utc_now(),
        "release_tag": release_tag,
        "asset_name": asset_name,
        "asset_url": asset_url,
        "archive_path": str(archive_path),
        "extracted_binary": str(extracted_binary),
        "current_binary": str(current_copy),
        "sha256": path_sha256(current_copy),
    }
    atomic_json(cache_root / "stockfish_manifest.json", manifest)
    if logger is not None:
        logger.write("stockfish_fetch_done", {"asset_name": asset_name, "binary": str(current_copy)})
    return str(current_copy)


def detect_stockfish_path(cfg: Dict[str, Any], logger: Optional[JSONLLogger] = None) -> Optional[str]:
    explicit = str(cfg.get("stockfish_path", "") or "").strip()
    if explicit and Path(explicit).exists():
        return explicit
    cached_binary = _find_stockfish_binary(Path(str(cfg.get("stockfish_cache_root", Path(str(cfg["cache_root"])) / "stockfish"))) / "current")
    if cached_binary is not None:
        return str(cached_binary)
    for candidate in (
        shutil.which("stockfish"),
        shutil.which("stockfish.exe"),
        str(Path.home() / "Desktop" / "stockfish" / "stockfish.exe"),
        str(Path.home() / "Downloads" / "stockfish" / "stockfish.exe"),
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return _fetch_stockfish_binary(cfg, logger)


CHESS_RESPONSE_CONTRACT_VERSION = "1.0"
CHESS_RESPONSE_MODES = {"play", "teach", "analyze", "turkish_teach", "benchmark"}
CHESS_TEACHING_LEVELS = {"basic", "club", "advanced"}


def normalize_chess_response_mode(mode: str) -> str:
    mode_norm = str(mode or "play").strip().lower()
    return mode_norm if mode_norm in CHESS_RESPONSE_MODES else "play"


def normalize_teaching_level(level: str) -> str:
    level_norm = str(level or "club").strip().lower()
    return level_norm if level_norm in CHESS_TEACHING_LEVELS else "club"


def classify_evaluation_label(value: float) -> str:
    if value >= 0.75:
        return "winning"
    if value >= 0.25:
        return "pressing"
    if value <= -0.75:
        return "losing"
    if value <= -0.25:
        return "under_pressure"
    return "balanced"


def build_evaluation_phrase_tr(value: float, level: str) -> str:
    label = classify_evaluation_label(value)
    phrases = {
        "winning": {
            "basic": "açık şekilde iyi görünüyor",
            "club": "eldeki konum belirgin biçimde iyi görünüyor",
            "advanced": "değerlendirme tarafında net üstünlük sinyali veriyor",
        },
        "pressing": {
            "basic": "hafif daha rahat görünüyor",
            "club": "konum tarafında küçük ama gerçek bir baskı avantajı var",
            "advanced": "motor-benzeri olmayan bu policy ölçümünde hafif artı bölgede kalıyor",
        },
        "balanced": {
            "basic": "yaklaşık dengeli görünüyor",
            "club": "konum büyük ölçüde dengeli, daha çok plan kalitesi fark yaratacak",
            "advanced": "değer başlığı tarafında keskin bir kopuş yok; plan ve uygulama öne çıkıyor",
        },
        "under_pressure": {
            "basic": "biraz baskı altında görünüyor",
            "club": "konum hafif eksi bölgede ve dikkatli savunma istiyor",
            "advanced": "değer başlığı tarafında eksi bölgede; tempo ve dayanıklılık önemli",
        },
        "losing": {
            "basic": "zor görünüyor",
            "club": "konum ciddi baskı altında ve hata payı daralmış durumda",
            "advanced": "değer başlığı tarafında ağır eksi sinyali var; savunma kaynakları sınırlı olabilir",
        },
    }
    return phrases[label][normalize_teaching_level(level)]


def build_confidence_payload(
    masked_logits: torch.Tensor,
    best_id: int,
    masked_topk_ids: Sequence[int],
) -> Dict[str, Any]:
    probs = torch.softmax(masked_logits, dim=-1)
    best_prob = float(probs[best_id].item())
    runner_prob = float(probs[masked_topk_ids[1]].item()) if len(masked_topk_ids) > 1 else 0.0
    gap = max(0.0, best_prob - runner_prob)
    if best_prob >= 0.55 or gap >= 0.25:
        tier = "high"
    elif best_prob >= 0.30 or gap >= 0.10:
        tier = "medium"
    else:
        tier = "low"
    return {
        "score": round(best_prob, 4),
        "gap": round(gap, 4),
        "tier": tier,
    }


def build_auxiliary_prediction_payload(
    auxiliary_outputs: Optional[Dict[str, torch.Tensor]],
    legal_ids: Sequence[int],
) -> Dict[str, Any]:
    if not auxiliary_outputs:
        return {}
    payload: Dict[str, Any] = {}
    phase_logits = auxiliary_outputs.get("phase_logits")
    if phase_logits is not None and phase_logits.ndim >= 2 and phase_logits.size(0) > 0:
        probs = torch.softmax(phase_logits[0], dim=-1)
        best_idx = int(probs.argmax().item())
        payload["phase_head"] = {
            "label": PHASE_NAMES.get(best_idx, str(best_idx)),
            "score": round(float(probs[best_idx].item()), 4),
            "distribution": {PHASE_NAMES.get(idx, str(idx)): round(float(prob.item()), 4) for idx, prob in enumerate(probs)},
        }
    wdl_logits = auxiliary_outputs.get("wdl_logits")
    if wdl_logits is not None and wdl_logits.ndim >= 2 and wdl_logits.size(0) > 0:
        probs = torch.softmax(wdl_logits[0], dim=-1)
        best_idx = int(probs.argmax().item())
        payload["wdl_head"] = {
            "label": WDL_CLASS_NAMES.get(best_idx, str(best_idx)),
            "score": round(float(probs[best_idx].item()), 4),
            "distribution": {WDL_CLASS_NAMES.get(idx, str(idx)): round(float(prob.item()), 4) for idx, prob in enumerate(probs)},
        }
    legality_logits = auxiliary_outputs.get("legality_logits")
    if legality_logits is not None and legality_logits.ndim >= 2 and legality_logits.size(0) > 0:
        scores = torch.sigmoid(legality_logits[0])
        legal_index_list = [int(item) for item in legal_ids]
        legal_mean = float(scores[legal_index_list].mean().item()) if legal_index_list else 0.0
        illegal_mask = torch.ones_like(scores, dtype=torch.bool)
        if legal_index_list:
            illegal_mask[legal_index_list] = False
        illegal_mean = float(scores[illegal_mask].mean().item()) if int(illegal_mask.sum().item()) > 0 else 0.0
        top1_id = int(legality_logits[0].argmax().item())
        payload["legality_head"] = {
            "top1_is_legal": top1_id in set(legal_index_list),
            "legal_mean_score": round(legal_mean, 4),
            "illegal_mean_score": round(illegal_mean, 4),
        }
    return payload


def unpack_model_outputs(model_output: Any, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if not isinstance(model_output, (tuple, list)) or len(model_output) < 2:
        raise RuntimeError("Model output must contain at least logits and value tensors")
    logits = model_output[0]
    value = model_output[1]
    aux_loss = model_output[2] if len(model_output) >= 3 else torch.tensor(0.0, device=device)
    router_reports = model_output[3] if len(model_output) >= 4 and isinstance(model_output[3], dict) else {}
    return logits, value, aux_loss, router_reports


def terminal_value_for_color(board: chess.Board, perspective_color: bool) -> Optional[float]:
    outcome = board.outcome()
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == perspective_color else -1.0


@torch.no_grad()
def evaluate_board_value(
    model: ChessPolicyValueNet,
    board: chess.Board,
    device: torch.device,
) -> float:
    legal_ids = legal_move_ids(board)
    if not legal_ids:
        terminal = terminal_value_for_color(board, board.turn)
        return float(terminal if terminal is not None else 0.0)
    piece_ids, meta_ids = encode_board_state(board, legal_move_count=len(legal_ids))
    piece = torch.tensor([piece_ids], dtype=torch.long, device=device)
    meta = torch.tensor([meta_ids], dtype=torch.long, device=device)
    _, value, _, _ = unpack_model_outputs(model(piece, meta), device)
    return float(value[0].item())


@torch.no_grad()
def policy_snapshot_for_board(
    model: ChessPolicyValueNet,
    board: chess.Board,
    device: torch.device,
    topk: int,
) -> Dict[str, Any]:
    legal_ids = legal_move_ids(board)
    if not legal_ids:
        raise RuntimeError("No legal moves available for policy snapshot")
    piece_ids, meta_ids = encode_board_state(board, legal_move_count=len(legal_ids))
    piece = torch.tensor([piece_ids], dtype=torch.long, device=device)
    meta = torch.tensor([meta_ids], dtype=torch.long, device=device)
    logits, value, _, _ = unpack_model_outputs(model(piece, meta), device)
    auxiliary_outputs = model.get_last_auxiliary_outputs() if hasattr(model, "get_last_auxiliary_outputs") else None
    logits = logits[0]
    raw_topk = torch.topk(logits, k=min(max(1, topk), logits.size(-1)), dim=-1)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[legal_ids] = True
    masked_logits = logits.masked_fill(~mask, -1e9)
    masked_topk = torch.topk(masked_logits, k=min(max(1, topk), masked_logits.size(-1)), dim=-1)
    return {
        "legal_ids": legal_ids,
        "piece_ids": piece_ids,
        "meta_ids": meta_ids,
        "logits": logits,
        "masked_logits": masked_logits,
        "value": float(value[0].item()),
        "raw_topk_ids": raw_topk.indices.tolist(),
        "raw_topk_scores": [round(float(item), 6) for item in raw_topk.values.tolist()],
        "masked_topk_ids": masked_topk.indices.tolist(),
        "masked_topk_scores": [round(float(item), 6) for item in masked_topk.values.tolist()],
        "auxiliary_predictions": build_auxiliary_prediction_payload(auxiliary_outputs, legal_ids),
    }


def is_tactically_forcing(board: chess.Board, move: chess.Move) -> bool:
    return bool(board.is_capture(move) or board.gives_check(move) or move.promotion is not None)


def infer_search_budget(board: chess.Board, confidence: Dict[str, Any], cfg: Dict[str, Any], topk: int) -> Dict[str, int]:
    candidate_topk = max(1, int(cfg.get("search_candidate_topk", topk)))
    reply_topk = max(1, int(cfg.get("search_reply_topk", 3)))
    if not bool(cfg.get("search_auto_budget", True)):
        return {"candidate_topk": candidate_topk, "reply_topk": reply_topk}
    legal_moves = list(board.legal_moves)
    forcing_count = sum(1 for move in legal_moves if is_tactically_forcing(board, move))
    if board.is_check() or forcing_count >= 5 or confidence.get("tier") == "low":
        candidate_topk = max(candidate_topk, 5)
        reply_topk = max(reply_topk, 4)
    elif confidence.get("tier") == "high" and float(confidence.get("gap", 0.0)) >= 0.20 and forcing_count == 0:
        candidate_topk = min(candidate_topk, max(2, topk))
        reply_topk = min(reply_topk, 2)
    return {"candidate_topk": candidate_topk, "reply_topk": reply_topk}


@torch.no_grad()
def score_move_with_shallow_search(
    model: ChessPolicyValueNet,
    board: chess.Board,
    move: chess.Move,
    device: torch.device,
    cfg: Dict[str, Any],
    reply_topk: int,
) -> Dict[str, Any]:
    perspective_color = board.turn
    board_after = board.copy(stack=False)
    board_after.push(move)
    terminal_after = terminal_value_for_color(board_after, perspective_color)
    if terminal_after is not None:
        return {
            "score": float(terminal_after),
            "immediate_score": float(terminal_after),
            "reply_move": "",
            "reply_scores": [],
        }
    immediate_score = -evaluate_board_value(model, board_after, device)
    snapshot = policy_snapshot_for_board(model, board_after, device, topk=max(1, reply_topk))
    worst_reply_score = immediate_score
    worst_reply_move = ""
    reply_scores: List[Dict[str, Any]] = []
    for reply_id in snapshot["masked_topk_ids"][: max(1, reply_topk)]:
        reply_uci = ID_TO_MOVE[int(reply_id)]
        reply_move = chess.Move.from_uci(reply_uci)
        if reply_move not in board_after.legal_moves:
            continue
        reply_board = board_after.copy(stack=False)
        reply_board.push(reply_move)
        terminal_reply = terminal_value_for_color(reply_board, perspective_color)
        reply_score = float(terminal_reply) if terminal_reply is not None else evaluate_board_value(model, reply_board, device)
        reply_scores.append({"move": reply_uci, "score": round(reply_score, 6)})
        if reply_score < worst_reply_score:
            worst_reply_score = reply_score
            worst_reply_move = reply_uci
    tactical_bonus = float(cfg.get("search_tactical_bonus", 0.0)) if is_tactically_forcing(board, move) else 0.0
    return {
        "score": float(worst_reply_score + tactical_bonus),
        "immediate_score": float(immediate_score),
        "reply_move": worst_reply_move,
        "reply_scores": reply_scores,
    }


def classify_teaching_tags(board: chess.Board, move: chess.Move) -> List[str]:
    piece = board.piece_at(move.from_square)
    if piece is None:
        return ["positional_choice"]
    tags: List[str] = []
    target_square = chess.square_name(move.to_square)
    source_square = chess.square_name(move.from_square)
    if board.is_capture(move):
        tags.append("capture")
    if board.is_castling(move):
        tags.append("castle")
    if move.promotion:
        tags.append("promotion")
    if board.gives_check(move):
        tags.append("check")
    board_after = board.copy(stack=False)
    board_after.push(move)
    if board_after.is_checkmate():
        tags.append("checkmate")
    if piece.piece_type == chess.PAWN and target_square in {"c4", "d4", "e4", "f4", "c5", "d5", "e5", "f5"}:
        tags.append("center_control")
    if piece.piece_type in {chess.KNIGHT, chess.BISHOP} and source_square in {"b1", "g1", "c1", "f1", "b8", "g8", "c8", "f8"}:
        tags.append("development")
    enemy_queen_squares = list(board_after.pieces(chess.QUEEN, not piece.color))
    if enemy_queen_squares and board_after.is_attacked_by(piece.color, enemy_queen_squares[0]):
        tags.append("queen_pressure")
    attacked_enemy_count = 0
    for square in board_after.attacks(move.to_square):
        attacked_piece = board_after.piece_at(square)
        if attacked_piece is not None and attacked_piece.color != piece.color:
            attacked_enemy_count += 1
    if attacked_enemy_count >= 2:
        tags.append("activity")
    if not tags:
        tags.append("positional_choice")
    return list(dict.fromkeys(tags))


def build_teaching_reasons_tr(tags: Sequence[str], level: str) -> List[str]:
    phrases = {
        "capture": "materyal dengesini etkileyen bir değişim yaratıyor",
        "castle": "şah güvenliğini artırıp kaleleri daha hızlı oyuna sokuyor",
        "promotion": "terfi ile taş gücünü ciddi biçimde yükseltiyor",
        "check": "rakip şahı hemen cevap vermeye zorluyor",
        "checkmate": "pozisyonu doğrudan mat ağına kapatıyor",
        "center_control": "merkez kareler üzerinde daha güçlü kontrol kuruyor",
        "development": "gelişimi hızlandırıp taş koordinasyonunu iyileştiriyor",
        "queen_pressure": "rakip vezire tempo kazandıran baskı uyguluyor",
        "activity": "tek hamlede birden fazla tehdit hattını canlandırıyor",
        "positional_choice": "konumun genel dengesini bozmadan oynanabilir bir plan seçiyor",
    }
    reasons = [phrases[tag] for tag in tags if tag in phrases]
    if not reasons:
        reasons = [phrases["positional_choice"]]
    if normalize_teaching_level(level) == "basic":
        return reasons[:2]
    if normalize_teaching_level(level) == "club":
        return reasons[:3]
    return reasons[:4]


def build_chess_response_contract(
    board: chess.Board,
    trace: Dict[str, Any],
    *,
    mode: str = "play",
    teaching_level: str = "club",
) -> Dict[str, Any]:
    mode_norm = normalize_chess_response_mode(mode)
    level_norm = normalize_teaching_level(teaching_level)
    move_uci = str(trace["move"])
    move = chess.Move.from_uci(move_uci)
    san = board.san(move)
    value = float(trace["value"])
    tags = classify_teaching_tags(board, move)
    reasons = build_teaching_reasons_tr(tags, level_norm)
    confidence_payload = dict(trace.get("confidence", {}))
    if not confidence_payload:
        confidence_payload = {"score": 0.0, "gap": 0.0, "tier": "low"}
    auxiliary_predictions = dict(trace.get("auxiliary_predictions", {}))
    evaluation_payload = {
        "value": round(value, 4),
        "label": classify_evaluation_label(value),
        "perspective": "side_to_move",
        "phrase_tr": build_evaluation_phrase_tr(value, level_norm),
    }
    alternatives = [item for item in trace.get("masked_topk", []) if item != move_uci][:2]
    prefix = {
        "play": "Oyun modu",
        "teach": "Öğretme modu",
        "analyze": "Analiz modu",
        "turkish_teach": "Türkçe öğretme modu",
        "benchmark": "Benchmark modu",
    }[mode_norm]
    short_reason = reasons[0]
    explanation_tr_short = f"{prefix}: {san} oynanıyor; {short_reason}. Konum {evaluation_payload['phrase_tr']}."
    long_parts = [
        f"{prefix} bu pozisyonda `{move_uci}` ({san}) hamlesini öne çıkarıyor.",
        f"Ana fikir: {', '.join(reasons)}.",
        f"Değer başlığına göre konum {evaluation_payload['phrase_tr']}.",
        f"Güven seviyesi `{confidence_payload['tier']}` ve üst aday farkı `{confidence_payload['gap']:.4f}`.",
    ]
    if alternatives:
        long_parts.append(f"Yakın alternatif adaylar: {', '.join(alternatives)}.")
    if auxiliary_predictions.get("phase_head"):
        phase_head = auxiliary_predictions["phase_head"]
        long_parts.append(
            f"Phase head bu konumu `{phase_head.get('label', 'unknown')}` olarak işaretliyor "
            f"(güven `{float(phase_head.get('score', 0.0)):.4f}`)."
        )
    if auxiliary_predictions.get("wdl_head"):
        wdl_head = auxiliary_predictions["wdl_head"]
        long_parts.append(
            f"WDL head kısa ufukta `{wdl_head.get('label', 'unknown')}` eğilimi veriyor "
            f"(güven `{float(wdl_head.get('score', 0.0)):.4f}`)."
        )
    if auxiliary_predictions.get("legality_head"):
        legality_head = auxiliary_predictions["legality_head"]
        long_parts.append(
            f"Legality head legal yüzeyde `{float(legality_head.get('legal_mean_score', 0.0)):.4f}`, "
            f"illegal yüzeyde `{float(legality_head.get('illegal_mean_score', 0.0)):.4f}` ortalama skor üretiyor."
        )
    long_parts.append("Buradaki principal variation search-derinliği değil, mevcut policy seçiminin tek hamlelik özetidir.")
    return {
        "contract_version": CHESS_RESPONSE_CONTRACT_VERSION,
        "best_move": move_uci,
        "best_move_san": san,
        "evaluation": evaluation_payload,
        "principal_variation": [move_uci],
        "confidence": confidence_payload,
        "auxiliary_predictions": auxiliary_predictions,
        "teaching_tags": tags,
        "explanation_tr_short": explanation_tr_short,
        "explanation_tr_long": " ".join(long_parts),
        "mode": mode_norm,
        "teaching_level": level_norm,
    }


def synthetic_trace_for_curated_position(item: Dict[str, Any], level: str) -> Dict[str, Any]:
    suite_value_hint = {
        "opening": 0.18,
        "tactical": 0.72,
        "endgame": 0.36,
        "blunder_correction": 0.48,
    }
    move = str(item["expected_move_uci"])
    confidence_score = {"basic": 0.58, "club": 0.64, "advanced": 0.7}[normalize_teaching_level(level)]
    return {
        "move": move,
        "value": suite_value_hint.get(str(item["suite"]), 0.22),
        "confidence": {
            "score": confidence_score,
            "gap": round(max(0.08, confidence_score - 0.22), 4),
            "tier": "medium" if confidence_score < 0.68 else "high",
        },
        "masked_topk": [move],
    }


def build_synthetic_teaching_corpus(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not bool(cfg.get("synthetic_teaching_corpus_enabled", True)):
        return {
            "schema": "chess_synthetic_teaching_corpus_v1",
            "enabled": False,
            "record_count": 0,
            "records": [],
        }
    bank = materialize_curated_position_bank()
    level_order = ["basic", "club", "advanced"]
    records: List[Dict[str, Any]] = []
    suite_counts: Counter[str] = Counter()
    level_counts: Counter[str] = Counter()
    for item in bank:
        board = item["board"]
        for level in level_order:
            trace = synthetic_trace_for_curated_position(item, level)
            contract = build_chess_response_contract(board, trace, mode="turkish_teach", teaching_level=level)
            records.append(
                {
                    "label": item["label"],
                    "suite": item["suite"],
                    "level": level,
                    "expected_move_uci": item["expected_move_uci"],
                    "expected_tags": item["expected_tags"],
                    "teaching_focus": item["teaching_focus"],
                    "commentary_tr": item["commentary_tr"],
                    "response_contract": contract,
                }
            )
            suite_counts[item["suite"]] += 1
            level_counts[level] += 1
    return {
        "schema": "chess_synthetic_teaching_corpus_v1",
        "enabled": True,
        "record_count": len(records),
        "suite_counts": dict(sorted(suite_counts.items())),
        "level_counts": dict(sorted(level_counts.items())),
        "records": records,
        "notes": {
            "purpose": "Internal Turkish teaching/explanation corpus for contract validation and future explanation training lanes.",
            "claim_note": "This corpus is synthetic and internal. It is not an externally verified explanation benchmark by itself.",
        },
    }


def render_synthetic_teaching_corpus_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Synthetic Chess Teaching Corpus",
        "",
        f"- enabled: `{report.get('enabled', False)}`",
        f"- record_count: `{report.get('record_count', 0)}`",
        "",
        "## Suite Counts",
    ]
    for suite_name, count in sorted(report.get("suite_counts", {}).items()):
        lines.append(f"- `{suite_name}`: `{count}`")
    lines.extend(["", "## Level Counts"])
    for level_name, count in sorted(report.get("level_counts", {}).items()):
        lines.append(f"- `{level_name}`: `{count}`")
    return "\n".join(lines) + "\n"


@torch.no_grad()
def choose_move_trace(
    model: ChessPolicyValueNet,
    board: chess.Board,
    device: torch.device,
    topk: int = 5,
    *,
    cfg: Optional[Dict[str, Any]] = None,
    mode: str = "play",
    teaching_level: str = "club",
) -> Dict[str, Any]:
    runtime_cfg = dict(RUN_CONFIG)
    if LAST_RUNTIME_CFG is not None:
        runtime_cfg.update(LAST_RUNTIME_CFG)
    if cfg is not None:
        runtime_cfg.update(cfg)
    search_snapshot_topk = max(
        int(topk),
        int(runtime_cfg.get("search_candidate_topk", topk)),
        int(runtime_cfg.get("search_reply_topk", 1)),
    )
    if not list(board.legal_moves):
        raise RuntimeError("No legal moves available for choose_move_trace")
    start = time.time()
    snapshot = policy_snapshot_for_board(model, board, device, topk=max(1, search_snapshot_topk))
    latency_ms = (time.time() - start) * 1000.0
    logits = snapshot["logits"]
    masked_logits = snapshot["masked_logits"]
    raw_topk_ids = snapshot["raw_topk_ids"][: max(1, topk)]
    raw_topk_scores = snapshot["raw_topk_scores"][: max(1, topk)]
    masked_topk_ids = snapshot["masked_topk_ids"][: max(1, topk)]
    masked_topk_scores = snapshot["masked_topk_scores"][: max(1, topk)]
    best_id = int(masked_logits.argmax().item())
    confidence = build_confidence_payload(masked_logits, best_id, snapshot["masked_topk_ids"])
    budget = infer_search_budget(board, confidence, runtime_cfg, topk=max(1, topk))
    probs = torch.softmax(masked_logits, dim=-1)
    candidate_ids = snapshot["masked_topk_ids"][: max(1, budget["candidate_topk"])]
    candidate_records: List[Dict[str, Any]] = []
    if bool(runtime_cfg.get("search_enabled", True)):
        policy_blend = float(runtime_cfg.get("search_policy_blend", 0.35))
        value_blend = float(runtime_cfg.get("search_value_blend", 0.65))
        for candidate_id in candidate_ids:
            move_uci = ID_TO_MOVE[int(candidate_id)]
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                continue
            search_info = score_move_with_shallow_search(
                model,
                board,
                move,
                device,
                runtime_cfg,
                reply_topk=max(1, budget["reply_topk"]),
            )
            policy_score = float(probs[int(candidate_id)].item())
            final_score = policy_blend * policy_score + value_blend * float(search_info["score"])
            candidate_records.append(
                {
                    "move": move_uci,
                    "policy_score": round(policy_score, 6),
                    "search_score": round(float(search_info["score"]), 6),
                    "immediate_score": round(float(search_info["immediate_score"]), 6),
                    "reply_move": str(search_info["reply_move"]),
                    "reply_scores": search_info["reply_scores"],
                    "final_score": round(final_score, 6),
                }
            )
    if candidate_records:
        candidate_records.sort(key=lambda item: (-float(item["final_score"]), -float(item["policy_score"]), item["move"]))
        move_uci = str(candidate_records[0]["move"])
        search_enabled = True
    else:
        move_uci = ID_TO_MOVE[best_id]
        search_enabled = False
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        raise RuntimeError(f"Masked policy selected illegal move: {move_uci}")
    raw_top1_id = int(logits.argmax().item())
    trace = {
        "move": move_uci,
        "value": float(snapshot["value"]),
        "latency_ms": round(latency_ms, 4),
        "raw_top1_is_legal": raw_top1_id in snapshot["legal_ids"],
        "raw_topk": [ID_TO_MOVE[idx] for idx in raw_topk_ids],
        "raw_topk_scores": raw_topk_scores,
        "masked_topk": [ID_TO_MOVE[idx] for idx in masked_topk_ids],
        "masked_topk_scores": masked_topk_scores,
        "confidence": confidence,
        "auxiliary_predictions": snapshot.get("auxiliary_predictions", {}),
        "search_enabled": search_enabled,
        "search_budget": budget,
        "search_candidates": candidate_records[: max(1, topk)],
    }
    trace["response_contract"] = build_chess_response_contract(
        board,
        trace,
        mode=mode,
        teaching_level=teaching_level,
    )
    return trace


def not_run_curated_position_eval(reason: str) -> Dict[str, Any]:
    return {
        "status": "not_run",
        "reason": reason,
        "position_count": 0,
        "exact_hit_rate": 0.0,
        "top3_hit_rate": 0.0,
        "expected_tag_coverage_rate": 0.0,
        "teaching_length_monotonic_rate": 0.0,
    }


@torch.no_grad()
def evaluate_curated_position_suites(model: ChessPolicyValueNet, cfg: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    if not bool(cfg.get("curated_suite_eval_enabled", True)):
        return not_run_curated_position_eval("disabled_by_config")
    was_training = model.training
    model.eval()
    bank = materialize_curated_position_bank()
    suite_totals: Counter[str] = Counter()
    suite_exact: Counter[str] = Counter()
    suite_top3: Counter[str] = Counter()
    suite_tags: Counter[str] = Counter()
    exact_hits = 0
    top3_hits = 0
    tag_hits = 0
    monotonic_hits = 0
    records: List[Dict[str, Any]] = []
    try:
        for item in bank:
            board = item["board"]
            trace = choose_move_trace(model, board, device, cfg=cfg, mode="teach", teaching_level="advanced")
            expected_move = str(item["expected_move_uci"])
            response_contract = dict(trace.get("response_contract", {}))
            observed_tags = set(response_contract.get("teaching_tags", []))
            expected_tags = list(item.get("expected_tags", []))
            exact_hit = trace["move"] == expected_move
            top3_hit = expected_move in trace.get("masked_topk", [])[:3]
            tag_hit = all(tag in observed_tags for tag in expected_tags)
            basic_contract = build_chess_response_contract(board, synthetic_trace_for_curated_position(item, "basic"), mode="turkish_teach", teaching_level="basic")
            advanced_contract = build_chess_response_contract(board, synthetic_trace_for_curated_position(item, "advanced"), mode="turkish_teach", teaching_level="advanced")
            monotonic_ok = len(advanced_contract["explanation_tr_long"]) >= len(basic_contract["explanation_tr_short"])
            suite_name = str(item["suite"])
            suite_totals[suite_name] += 1
            suite_exact[suite_name] += int(exact_hit)
            suite_top3[suite_name] += int(top3_hit)
            suite_tags[suite_name] += int(tag_hit)
            exact_hits += int(exact_hit)
            top3_hits += int(top3_hit)
            tag_hits += int(tag_hit)
            monotonic_hits += int(monotonic_ok)
            records.append(
                {
                    "label": item["label"],
                    "suite": suite_name,
                    "expected_move_uci": expected_move,
                    "predicted_move_uci": trace["move"],
                    "exact_hit": exact_hit,
                    "top3_hit": top3_hit,
                    "expected_tag_coverage": tag_hit,
                    "teaching_length_monotonic": monotonic_ok,
                    "expected_tags": expected_tags,
                    "predicted_tags": response_contract.get("teaching_tags", []),
                    "confidence": response_contract.get("confidence", {}),
                }
            )
        per_suite = {}
        for suite_name, total in sorted(suite_totals.items()):
            per_suite[suite_name] = {
                "positions": total,
                "exact_hit_rate": round(suite_exact[suite_name] / max(1, total), 6),
                "top3_hit_rate": round(suite_top3[suite_name] / max(1, total), 6),
                "expected_tag_coverage_rate": round(suite_tags[suite_name] / max(1, total), 6),
            }
        total_positions = len(bank)
        return {
            "status": "completed",
            "position_count": total_positions,
            "exact_hit_rate": round(exact_hits / max(1, total_positions), 6),
            "top3_hit_rate": round(top3_hits / max(1, total_positions), 6),
            "expected_tag_coverage_rate": round(tag_hits / max(1, total_positions), 6),
            "teaching_length_monotonic_rate": round(monotonic_hits / max(1, total_positions), 6),
            "per_suite": per_suite,
            "records": records,
            "notes": {
                "benchmark_scope": "Internal curated suite only; exact hits are against a small repeatable bank, not an external Elo pool.",
                "teaching_scope": "Tag coverage and length monotonicity are contract-faithfulness checks, not pedagogy-quality proof.",
            },
        }
    finally:
        model.train(was_training)


def render_curated_position_suite_report_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Curated Chess Position Suite Report",
        "",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- position_count: `{report.get('position_count', 0)}`",
        f"- exact_hit_rate: `{report.get('exact_hit_rate', 0.0)}`",
        f"- top3_hit_rate: `{report.get('top3_hit_rate', 0.0)}`",
        f"- expected_tag_coverage_rate: `{report.get('expected_tag_coverage_rate', 0.0)}`",
        f"- teaching_length_monotonic_rate: `{report.get('teaching_length_monotonic_rate', 0.0)}`",
        "",
        "## Per Suite",
    ]
    for suite_name, suite_payload in sorted(report.get("per_suite", {}).items()):
        lines.append(
            f"- `{suite_name}`: exact=`{suite_payload.get('exact_hit_rate', 0.0)}`, "
            f"top3=`{suite_payload.get('top3_hit_rate', 0.0)}`, "
            f"tags=`{suite_payload.get('expected_tag_coverage_rate', 0.0)}`"
        )
    return "\n".join(lines) + "\n"


@torch.no_grad()
def run_legality_report(model: ChessPolicyValueNet, examples: Sequence[ChessExample], device: torch.device, cfg: Dict[str, Any]) -> Dict[str, Any]:
    was_training = model.training
    model.eval()
    sample_limit = int(cfg.get("legal_move_sample_checks", 0))
    picks = list(examples)
    if sample_limit > 0 and len(picks) > sample_limit:
        rng = random.Random(int(cfg["seed"]) + 77)
        rng.shuffle(picks)
        picks = picks[:sample_limit]
    checked = 0
    raw_top1_legal = 0
    raw_topk_contains_legal = 0
    masked_correct = 0
    phase_checked: Counter[str] = Counter()
    phase_raw_top1_legal: Counter[str] = Counter()
    phase_raw_topk_contains_legal: Counter[str] = Counter()
    phase_masked_correct: Counter[str] = Counter()
    examples_out: List[Dict[str, Any]] = []
    for example in picks:
        piece = torch.tensor([example.piece_ids], dtype=torch.long, device=device)
        meta = torch.tensor([example.meta_ids], dtype=torch.long, device=device)
        logits, _, _, _ = model(piece, meta)
        logits = logits[0]
        raw_top1 = int(logits.argmax().item())
        raw_topk = torch.topk(logits, k=min(5, logits.size(-1)), dim=-1).indices.tolist()
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[example.legal_move_ids] = True
        masked_logits = logits.masked_fill(~mask, -1e9)
        masked_top1 = int(masked_logits.argmax().item())
        phase_name = PHASE_NAMES[int(example.phase)]
        phase_checked[phase_name] += 1
        checked += 1
        if raw_top1 in example.legal_move_ids:
            raw_top1_legal += 1
            phase_raw_top1_legal[phase_name] += 1
        if any(item in example.legal_move_ids for item in raw_topk):
            raw_topk_contains_legal += 1
            phase_raw_topk_contains_legal[phase_name] += 1
        if masked_top1 == example.target_move_id:
            masked_correct += 1
            phase_masked_correct[phase_name] += 1
        if len(examples_out) < 16:
            examples_out.append(
                {
                    "phase": phase_name,
                    "target": example.move_uci,
                    "raw_top1": ID_TO_MOVE[raw_top1],
                    "masked_top1": ID_TO_MOVE[masked_top1],
                    "raw_top1_is_legal": raw_top1 in example.legal_move_ids,
                }
            )
    try:
        report = {
            "checked_examples": checked,
            "raw_top1_is_legal_rate": round(raw_top1_legal / max(1, checked), 6),
            "raw_topk_contains_legal_rate": round(raw_topk_contains_legal / max(1, checked), 6),
            "masked_policy_accuracy": round(masked_correct / max(1, checked), 6),
            "per_phase": {},
            "example_rows": examples_out,
            "note": "Replay/demo output is demonstration material only. Raw legality and masked accuracy are intentionally separated.",
        }
        for phase_name in PHASE_NAMES.values():
            count = phase_checked[phase_name]
            if count <= 0:
                continue
            report["per_phase"][phase_name] = {
                "checked": count,
                "raw_top1_is_legal_rate": round(phase_raw_top1_legal[phase_name] / count, 6),
                "raw_topk_contains_legal_rate": round(phase_raw_topk_contains_legal[phase_name] / count, 6),
                "masked_policy_accuracy": round(phase_masked_correct[phase_name] / count, 6),
            }
        return report
    finally:
        model.train(was_training)


def ensure_interactive_console() -> None:
    if platform.system() != "Windows":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        if kernel32.GetConsoleWindow():
            return
        if kernel32.AllocConsole() == 0:
            return
        sys.stdin = open("CONIN$", "r", encoding="utf-8", buffering=1)
        sys.stdout = open("CONOUT$", "w", encoding="utf-8", buffering=1)
        sys.stderr = open("CONOUT$", "w", encoding="utf-8", buffering=1)
    except Exception:
        return


def play_human_vs_model_arena(
    model: ChessPolicyValueNet,
    cfg: Dict[str, Any],
    device: torch.device,
    logger: Optional[JSONLLogger] = None,
) -> Dict[str, Any]:
    ensure_interactive_console()
    was_training = model.training
    model.eval()
    board = chess.Board()
    transcript: List[Dict[str, Any]] = []
    human_color = chess.WHITE
    help_text = "Enter UCI moves such as e2e4. Commands: help, board, fen, quit."
    try:
        try:
            side_raw = input("Play as white or black? [w/b, Enter=white]: ").strip().lower()
        except EOFError:
            side_raw = ""
        if side_raw in {"b", "black"}:
            human_color = chess.BLACK
        print(help_text)
        print(board)
        if logger is not None:
            logger.write("arena_start", {"human_color": "white" if human_color == chess.WHITE else "black"})
        while not board.is_game_over():
            if board.turn == human_color:
                while True:
                    try:
                        move_raw = input("Your move> ").strip()
                    except EOFError:
                        move_raw = "quit"
                    lowered = move_raw.lower()
                    if lowered in {"quit", "exit"}:
                        report = {
                            "status": "aborted_by_user",
                            "interactive_only": True,
                            "human_color": "white" if human_color == chess.WHITE else "black",
                            "result": "*",
                            "plies_played": len(transcript),
                            "final_fen": board.fen(),
                            "transcript": transcript,
                            "note": "Arena session stopped by the user before game termination.",
                        }
                        if logger is not None:
                            logger.write("arena_stop", {"reason": "aborted_by_user", "plies_played": len(transcript)})
                        return report
                    if lowered == "help":
                        print(help_text)
                        continue
                    if lowered == "board":
                        print(board)
                        continue
                    if lowered == "fen":
                        print(board.fen())
                        continue
                    try:
                        move = chess.Move.from_uci(move_raw)
                    except ValueError:
                        print("Invalid move format. Use UCI like e2e4.")
                        continue
                    if move not in board.legal_moves:
                        print("Illegal move. Try again.")
                        continue
                    board.push(move)
                    transcript.append({"ply": len(transcript) + 1, "actor": "human", "move": move.uci(), "fen": board.fen()})
                    print(board)
                    break
            else:
                trace = choose_move_trace(model, board, device, cfg=cfg)
                move = chess.Move.from_uci(trace["move"])
                board.push(move)
                transcript.append(
                    {
                        "ply": len(transcript) + 1,
                        "actor": "model",
                        "move": move.uci(),
                        "fen": board.fen(),
                        "value": trace["value"],
                        "latency_ms": trace["latency_ms"],
                        "raw_top1_is_legal": trace["raw_top1_is_legal"],
                    }
                )
                print(f"Model move: {move.uci()} | value={trace['value']:.3f} | latency_ms={trace['latency_ms']:.2f}")
                print(board)
        outcome = board.outcome()
        report = {
            "status": "completed",
            "interactive_only": True,
            "human_color": "white" if human_color == chess.WHITE else "black",
            "result": outcome.result() if outcome is not None else "*",
            "termination": str(outcome.termination) if outcome is not None else "unknown",
            "winner": (
                "white"
                if outcome is not None and outcome.winner is chess.WHITE
                else "black"
                if outcome is not None and outcome.winner is chess.BLACK
                else "draw"
            ),
            "plies_played": len(transcript),
            "final_fen": board.fen(),
            "transcript": transcript,
            "note": "Arena mode is interactive and intended for human-vs-model inspection, not strength proof.",
        }
        if logger is not None:
            logger.write("arena_complete", {"result": report["result"], "plies_played": len(transcript)})
        return report
    finally:
        model.train(was_training)


def write_curve_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["step", "split", "loss"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    payload = tag + data
    return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)


def _write_simple_png(path: Path, width: int, height: int, rows: List[bytearray]) -> None:
    raw = bytearray()
    for row in rows:
        raw.append(0)
        raw.extend(row)
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def _set_pixel(rows: List[bytearray], x: int, y: int, color: Tuple[int, int, int]) -> None:
    height = len(rows)
    width = len(rows[0]) // 3 if rows else 0
    if 0 <= x < width and 0 <= y < height:
        offset = x * 3
        rows[y][offset:offset + 3] = bytes(color)


def _draw_line(rows: List[bytearray], x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int]) -> None:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        _set_pixel(rows, x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def write_curve_png(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    width = 900
    height = 460
    canvas = [bytearray([255, 255, 255] * width) for _ in range(height)]
    if not rows:
        _write_simple_png(path, width, height, canvas)
        return
    margin_left, margin_right, margin_top, margin_bottom = 60, 30, 30, 45
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    for x in range(margin_left, width - margin_right):
        _set_pixel(canvas, x, height - margin_bottom, (180, 180, 180))
    for y in range(margin_top, height - margin_bottom):
        _set_pixel(canvas, margin_left, y, (180, 180, 180))

    train_points = [(float(row.get("step", 0)), float(row.get("loss", 0.0))) for row in rows if row.get("split") == "train" and row.get("loss") is not None]
    val_points = [(float(row.get("step", 0)), float(row.get("loss", 0.0))) for row in rows if row.get("split") == "val" and row.get("loss") is not None]

    def downsample_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(points) <= plot_w:
            return points
        step = max(1, len(points) // plot_w)
        sampled = points[::step]
        if sampled[-1] != points[-1]:
            sampled.append(points[-1])
        return sampled

    train_points = downsample_points(train_points)
    val_points = downsample_points(val_points)
    all_points = train_points + val_points
    if not all_points:
        _write_simple_png(path, width, height, canvas)
        return
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0

    def project(point: Tuple[float, float]) -> Tuple[int, int]:
        x_value, y_value = point
        x = margin_left + int(round(((x_value - x_min) / (x_max - x_min)) * plot_w))
        y = margin_top + int(round((1.0 - ((y_value - y_min) / (y_max - y_min))) * plot_h))
        return x, y

    for grid_idx in range(5):
        grid_y = margin_top + int(round((grid_idx / 4.0) * plot_h))
        for grid_x in range(margin_left, width - margin_right):
            _set_pixel(canvas, grid_x, grid_y, (235, 235, 235))
    for grid_idx in range(5):
        grid_x = margin_left + int(round((grid_idx / 4.0) * plot_w))
        for grid_y in range(margin_top, height - margin_bottom):
            _set_pixel(canvas, grid_x, grid_y, (240, 240, 240))

    def draw_series(points: List[Tuple[float, float]], color: Tuple[int, int, int]) -> None:
        if len(points) == 1:
            x, y = project(points[0])
            _set_pixel(canvas, x, y, color)
            return
        for idx in range(1, len(points)):
            x0, y0 = project(points[idx - 1])
            x1, y1 = project(points[idx])
            _draw_line(canvas, x0, y0, x1, y1, color)

    draw_series(train_points, (54, 111, 207))
    draw_series(val_points, (214, 69, 65))
    _write_simple_png(path, width, height, canvas)


def compute_score_rate_ci(score_rate: float, games: int) -> Dict[str, float]:
    if games <= 0:
        return {"low": 0.0, "high": 0.0}
    z = 1.96
    n = float(games)
    p = min(max(float(score_rate), 0.0), 1.0)
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) / n) + ((z * z) / (4.0 * n * n))) / denom
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return {"low": round(low, 6), "high": round(high, 6)}


def elo_proxy_from_score(score_rate: float, anchor_elo: int) -> Optional[int]:
    if score_rate <= 0.0 or score_rate >= 1.0:
        return None
    try:
        diff = 400.0 * math.log10(score_rate / max(1e-9, 1.0 - score_rate))
    except ValueError:
        return None
    return int(round(anchor_elo + diff))


def build_benchmark_protocol(cfg: Dict[str, Any], engine_path: Optional[str]) -> Dict[str, Any]:
    engine_sha = path_sha256(Path(engine_path)) if engine_path and Path(engine_path).exists() else ""
    return {
        "protocol_name": "internal_stockfish_gauntlet_v2",
        "status": "configured" if engine_path else "engine_missing",
        "engine_path": redact_path(engine_path) if engine_path else "",
        "engine_sha256": engine_sha,
        "engine_acquisition": "auto_fetch_or_cached" if bool(cfg.get("stockfish_auto_fetch", True)) else "manual_only",
        "openings": OPENING_SEEDS,
        "ladder": cfg.get("stockfish_ladder", []),
        "rating_note": "This protocol emits elo_proxy_internal only. It does not emit a plain ELO claim.",
        "anchor_elo_proxy_note": "Anchor ELO values are internal approximations for Stockfish skill levels and are not calibrated against an external rating pool.",
    }


def build_pgn_from_moves(starting_moves: Sequence[str], played_moves: Sequence[str], result: str) -> str:
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "MertFormer Stockfish Gauntlet"
    game.headers["Site"] = "Local"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = "-"
    game.headers["Result"] = result
    node = game
    for move_uci in list(starting_moves) + list(played_moves):
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            break
        board.push(move)
        node = node.add_variation(move)
    return str(game)


def not_run_selfplay_report(reason: str) -> Dict[str, Any]:
    return {
        "status": "not_run",
        "reason": reason,
        "games_requested": 0,
        "games_played": 0,
        "result_counts": {"1-0": 0, "1/2-1/2": 0, "0-1": 0, "*": 0},
        "average_plies": 0.0,
        "games": [],
        "note": "Self-play artifact generation was skipped for this run.",
    }


def render_selfplay_report_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Self-Play Report",
        "",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- games_requested: `{report.get('games_requested', 0)}`",
        f"- games_played: `{report.get('games_played', 0)}`",
        f"- average_plies: `{report.get('average_plies', 0.0)}`",
        "",
        "## Result Counts",
    ]
    for result_name, count in sorted(report.get("result_counts", {}).items()):
        lines.append(f"- `{result_name}`: `{count}`")
    return "\n".join(lines) + "\n"


def not_run_tournament_report(reason: str) -> Dict[str, Any]:
    return {
        "status": "not_run",
        "reason": reason,
        "games_requested": 0,
        "games_played": 0,
        "players": {},
        "games": [],
        "note": "Inference-mode tournament generation was skipped for this run.",
    }


def render_tournament_report_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Inference-Mode Tournament Report",
        "",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- games_requested: `{report.get('games_requested', 0)}`",
        f"- games_played: `{report.get('games_played', 0)}`",
        "",
        "## Player Scores",
    ]
    for player_name, payload in sorted(report.get("players", {}).items()):
        lines.append(
            f"- `{player_name}`: wins=`{payload.get('wins', 0)}`, "
            f"draws=`{payload.get('draws', 0)}`, losses=`{payload.get('losses', 0)}`, "
            f"score_rate=`{payload.get('score_rate', 0.0)}`"
        )
    return "\n".join(lines) + "\n"


def not_run_replay_buffer_report(reason: str) -> Dict[str, Any]:
    return {
        "status": "not_run",
        "reason": reason,
        "positions": 0,
        "games_used": 0,
        "truncated": False,
        "records": [],
        "note": "Replay-buffer manifest generation was skipped for this run.",
    }


def render_replay_buffer_report_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Replay Buffer Manifest",
        "",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- positions: `{report.get('positions', 0)}`",
        f"- games_used: `{report.get('games_used', 0)}`",
        f"- truncated: `{report.get('truncated', False)}`",
    ]
    return "\n".join(lines) + "\n"


def _limited_opening_prefix(game_idx: int, cfg: Dict[str, Any]) -> List[str]:
    opening = list(OPENING_SEEDS[game_idx % len(OPENING_SEEDS)])
    prefix_plies = max(0, int(cfg.get("selfplay_opening_prefix_plies", 2)))
    if prefix_plies <= 0:
        return []
    return opening[: min(prefix_plies, len(opening))]


@torch.no_grad()
def generate_selfplay_report(
    model: ChessPolicyValueNet,
    cfg: Dict[str, Any],
    device: torch.device,
    layout: ArtifactLayout,
) -> Dict[str, Any]:
    if not bool(cfg.get("selfplay_eval_enabled", False)):
        return not_run_selfplay_report("disabled_by_config")
    was_training = model.training
    model.eval()
    pgn_dir = layout.benchmark_dir / "selfplay_pgns"
    pgn_dir.mkdir(parents=True, exist_ok=True)
    games_requested = max(0, int(cfg.get("selfplay_games", 0)))
    max_plies = max(1, int(cfg.get("selfplay_max_plies", 96)))
    report: Dict[str, Any] = {
        "status": "completed",
        "games_requested": games_requested,
        "games_played": 0,
        "result_counts": {"1-0": 0, "1/2-1/2": 0, "0-1": 0, "*": 0},
        "average_plies": 0.0,
        "games": [],
        "note": "Self-play uses the current in-memory model and remains an internal artifact, not an external strength claim.",
    }
    total_plies = 0
    try:
        for game_idx in range(games_requested):
            board = chess.Board()
            opening_prefix = _limited_opening_prefix(game_idx, cfg)
            for move_uci in opening_prefix:
                move = chess.Move.from_uci(move_uci)
                if move not in board.legal_moves:
                    break
                board.push(move)
            played_moves: List[str] = []
            move_rows: List[Dict[str, Any]] = []
            while not board.is_game_over() and len(played_moves) < max_plies:
                fen_before = board.fen()
                trace = choose_move_trace(model, board, device, cfg=cfg, mode="analyze", teaching_level="club")
                move = chess.Move.from_uci(trace["move"])
                if move not in board.legal_moves:
                    break
                played_moves.append(trace["move"])
                board.push(move)
                move_rows.append(
                    {
                        "ply": len(played_moves),
                        "move": trace["move"],
                        "fen_before": fen_before,
                        "fen_after": board.fen(),
                        "value": trace["value"],
                        "confidence": trace.get("confidence", {}),
                        "auxiliary_predictions": trace.get("auxiliary_predictions", {}),
                    }
                )
            outcome = board.outcome()
            result = outcome.result() if outcome is not None else "*"
            report["result_counts"][result] = report["result_counts"].get(result, 0) + 1
            pgn_path = pgn_dir / f"selfplay_{game_idx:03d}.pgn"
            atomic_write_text(pgn_path, build_pgn_from_moves(opening_prefix, played_moves, result))
            report["games"].append(
                {
                    "game_index": game_idx,
                    "opening_prefix": opening_prefix,
                    "result": result,
                    "plies": len(played_moves),
                    "completed": bool(outcome is not None),
                    "final_fen": board.fen(),
                    "pgn_path": str(pgn_path),
                    "moves": move_rows,
                }
            )
            total_plies += len(played_moves)
        report["games_played"] = len(report["games"])
        report["average_plies"] = round(total_plies / max(1, report["games_played"]), 4)
        return report
    finally:
        model.train(was_training)


def build_replay_buffer_report(selfplay_report: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not bool(cfg.get("replay_buffer_enabled", False)):
        return not_run_replay_buffer_report("disabled_by_config")
    if selfplay_report.get("status") != "completed":
        return not_run_replay_buffer_report(f"selfplay_unavailable:{selfplay_report.get('status', 'unknown')}")
    max_positions = max(1, int(cfg.get("replay_buffer_max_positions", 256)))
    records: List[Dict[str, Any]] = []
    games_used = 0
    truncated = False
    for game in selfplay_report.get("games", []):
        games_used += 1
        for row in game.get("moves", []):
            records.append(
                {
                    "game_index": game.get("game_index", 0),
                    "ply": row.get("ply", 0),
                    "fen_before": row.get("fen_before", ""),
                    "move": row.get("move", ""),
                    "value": row.get("value", 0.0),
                    "confidence": row.get("confidence", {}),
                    "auxiliary_predictions": row.get("auxiliary_predictions", {}),
                }
            )
            if len(records) >= max_positions:
                truncated = True
                break
        if truncated:
            break
    return {
        "status": "completed",
        "positions": len(records),
        "games_used": games_used,
        "truncated": truncated,
        "records": records,
        "note": "Replay-buffer manifest is derived from internal self-play only and is not external benchmark evidence.",
    }


@torch.no_grad()
def play_inference_mode_tournament(
    model: ChessPolicyValueNet,
    cfg: Dict[str, Any],
    device: torch.device,
    layout: ArtifactLayout,
) -> Dict[str, Any]:
    if not bool(cfg.get("tournament_eval_enabled", False)):
        return not_run_tournament_report("disabled_by_config")
    was_training = model.training
    model.eval()
    games_requested = max(0, int(cfg.get("tournament_games", 0)))
    max_plies = max(1, int(cfg.get("tournament_max_plies", cfg.get("selfplay_max_plies", 96))))
    pgn_dir = layout.benchmark_dir / "inference_mode_tournament_pgns"
    pgn_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "status": "completed",
        "games_requested": games_requested,
        "games_played": 0,
        "players": {
            "search_assisted": {"wins": 0, "draws": 0, "losses": 0, "score_rate": 0.0},
            "pure_policy": {"wins": 0, "draws": 0, "losses": 0, "score_rate": 0.0},
        },
        "games": [],
        "note": "This tournament compares inference modes of the same model. It is diagnostic only.",
    }
    try:
        for game_idx in range(games_requested):
            board = chess.Board()
            opening_prefix = _limited_opening_prefix(game_idx, cfg)
            for move_uci in opening_prefix:
                move = chess.Move.from_uci(move_uci)
                if move not in board.legal_moves:
                    break
                board.push(move)
            white_mode = "search_assisted" if game_idx % 2 == 0 else "pure_policy"
            black_mode = "pure_policy" if white_mode == "search_assisted" else "search_assisted"
            played_moves: List[str] = []
            while not board.is_game_over() and len(played_moves) < max_plies:
                current_mode = white_mode if board.turn == chess.WHITE else black_mode
                mode_cfg = dict(cfg)
                mode_cfg["search_enabled"] = current_mode == "search_assisted"
                trace = choose_move_trace(model, board, device, cfg=mode_cfg, mode="benchmark", teaching_level="club")
                move = chess.Move.from_uci(trace["move"])
                if move not in board.legal_moves:
                    break
                played_moves.append(trace["move"])
                board.push(move)
            outcome = board.outcome()
            result = outcome.result() if outcome is not None else "*"
            if result == "1-0":
                report["players"][white_mode]["wins"] += 1
                report["players"][black_mode]["losses"] += 1
            elif result == "0-1":
                report["players"][black_mode]["wins"] += 1
                report["players"][white_mode]["losses"] += 1
            else:
                report["players"][white_mode]["draws"] += 1
                report["players"][black_mode]["draws"] += 1
            pgn_path = pgn_dir / f"mode_tournament_{game_idx:03d}.pgn"
            atomic_write_text(pgn_path, build_pgn_from_moves(opening_prefix, played_moves, result))
            report["games"].append(
                {
                    "game_index": game_idx,
                    "white_mode": white_mode,
                    "black_mode": black_mode,
                    "result": result,
                    "plies": len(played_moves),
                    "pgn_path": str(pgn_path),
                }
            )
        report["games_played"] = len(report["games"])
        for player_name, player_report in report["players"].items():
            score = player_report["wins"] + 0.5 * player_report["draws"]
            player_report["score_rate"] = round(score / max(1, report["games_played"]), 6)
        return report
    finally:
        model.train(was_training)


def play_stockfish_gauntlet(
    model: ChessPolicyValueNet,
    cfg: Dict[str, Any],
    device: torch.device,
    layout: ArtifactLayout,
    logger: JSONLLogger,
) -> Dict[str, Any]:
    engine_path = detect_stockfish_path(cfg, logger)
    protocol = build_benchmark_protocol(cfg, engine_path)
    atomic_json(layout.reports_dir / "benchmark_protocol.json", protocol)
    if not engine_path or not cfg.get("stockfish_ladder"):
        report = {
            "status": "not_run",
            "reason": "stockfish_missing_or_disabled",
            "protocol": protocol,
        }
        atomic_json(layout.reports_dir / "stockfish_match_report.json", report)
        return report

    try:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    except Exception as exc:  # pragma: no cover - engine availability varies
        report = {
            "status": "not_run",
            "reason": f"engine_start_failed:{type(exc).__name__}",
            "protocol": protocol,
        }
        atomic_json(layout.reports_dir / "stockfish_match_report.json", report)
        return report

    gauntlet_dir = layout.benchmark_dir / "stockfish_gauntlet_pgns"
    gauntlet_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "status": "completed",
        "protocol": protocol,
        "levels": [],
        "games_total": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "elo_proxy_internal": None,
        "rating_note": "Internal gauntlet only. Any rating output is a proxy, not a verified external rating.",
        "anchor_elo_proxy_note": "Anchor ELO values are internal approximations for Stockfish skill levels and are not calibrated against an external rating pool.",
    }

    was_training = model.training
    model.eval()
    try:
        for level_idx, level in enumerate(cfg.get("stockfish_ladder", [])):
            games_requested = int(level.get("games", 0))
            if games_requested <= 0:
                continue
            games_requested += games_requested % 2
            openings = OPENING_SEEDS
            level_result = {
                "label": str(level.get("label", f"level_{level_idx}")),
                "skill": int(level.get("skill", 4)),
                "nodes": int(level.get("nodes", 20000)),
                "games_requested": games_requested,
                "games_played": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "score_rate": 0.0,
                "score_rate_ci": {"low": 0.0, "high": 0.0},
                "anchor_elo_proxy": int(level.get("anchor_elo_proxy", 1400)),
                "anchor_elo_proxy_note": "Internal approximation only; not externally calibrated.",
                "elo_proxy_internal": None,
                "opening_seed_hash": sha256_bytes(json.dumps(openings).encode("utf-8")),
                "games": [],
            }
            for game_idx in range(games_requested):
                opening_moves = openings[game_idx % len(openings)]
                board = chess.Board()
                for move_uci in opening_moves:
                    board.push(chess.Move.from_uci(move_uci))
                model_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK
                played_moves: List[str] = []
                while not board.is_game_over() and len(played_moves) < 180:
                    if board.turn == model_color:
                        trace = choose_move_trace(model, board, device, cfg=cfg)
                        move = chess.Move.from_uci(trace["move"])
                        if move not in board.legal_moves:
                            report = {
                                "status": "failed",
                                "reason": "illegal_move_generated",
                                "protocol": protocol,
                                "level": level_result["label"],
                                "game_index": game_idx,
                            }
                            atomic_json(layout.reports_dir / "stockfish_match_report.json", report)
                            return report
                        board.push(move)
                        played_moves.append(move.uci())
                    else:
                        result = engine.play(
                            board,
                            chess.engine.Limit(nodes=int(level.get("nodes", 20000))),
                            options={"Skill Level": int(level.get("skill", 4))},
                        )
                        board.push(result.move)
                        played_moves.append(result.move.uci())
                outcome = board.outcome()
                result_str = outcome.result() if outcome is not None else "1/2-1/2"
                level_result["games_played"] += 1
                report["games_total"] += 1
                if (result_str == "1-0" and model_color == chess.WHITE) or (result_str == "0-1" and model_color == chess.BLACK):
                    level_result["wins"] += 1
                    report["wins"] += 1
                elif result_str == "1/2-1/2":
                    level_result["draws"] += 1
                    report["draws"] += 1
                else:
                    level_result["losses"] += 1
                    report["losses"] += 1
                pgn_text = build_pgn_from_moves(opening_moves, played_moves, result_str)
                pgn_path = gauntlet_dir / f"{level_result['label']}_game_{game_idx:03d}.pgn"
                atomic_write_text(pgn_path, pgn_text + "\n")
                level_result["games"].append(
                    {
                        "game_index": game_idx,
                        "model_color": "white" if model_color == chess.WHITE else "black",
                        "result": result_str,
                        "plies": len(played_moves),
                        "pgn_path": str(pgn_path.relative_to(layout.run_dir)),
                    }
                )
            score_rate = (level_result["wins"] + 0.5 * level_result["draws"]) / max(1, level_result["games_played"])
            level_result["score_rate"] = round(score_rate, 6)
            level_result["score_rate_ci"] = compute_score_rate_ci(score_rate, int(level_result["games_played"]))
            level_result["elo_proxy_internal"] = elo_proxy_from_score(score_rate, int(level_result["anchor_elo_proxy"]))
            report["levels"].append(level_result)
        level_proxies = [item["elo_proxy_internal"] for item in report["levels"] if item.get("elo_proxy_internal") is not None]
        if level_proxies:
            report["elo_proxy_internal"] = int(round(sum(level_proxies) / len(level_proxies)))
        logger.write(
            "stockfish_eval",
            {
                "status": report["status"],
                "games_total": report["games_total"],
                "elo_proxy_internal": report.get("elo_proxy_internal"),
            },
        )
        atomic_json(layout.reports_dir / "stockfish_match_report.json", report)
        return report
    finally:
        model.train(was_training)
        with contextlib.suppress(Exception):
            engine.quit()


def build_midrun_stockfish_snapshot_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    snapshot_cfg = dict(cfg)
    games_per_level = max(1, int(cfg.get("midrun_stockfish_snapshot_games", 4)))
    ladder = []
    for level in cfg.get("stockfish_ladder", [])[:2]:
        level_copy = dict(level)
        level_copy["games"] = min(int(level_copy.get("games", games_per_level)), games_per_level)
        ladder.append(level_copy)
    snapshot_cfg["stockfish_ladder"] = ladder
    return snapshot_cfg


def build_snapshot_layout(layout: ArtifactLayout, snapshot_root: Path, step: int) -> ArtifactLayout:
    snapshot_run_dir = snapshot_root / f"step_{step:07d}"
    logs_dir = snapshot_run_dir / "logs"
    reports_dir = snapshot_run_dir / "reports"
    checkpoints_dir = snapshot_run_dir / "checkpoints"
    export_dir = snapshot_run_dir / "exports"
    benchmark_dir = snapshot_run_dir / "benchmarks"
    for path in (snapshot_run_dir, logs_dir, reports_dir, checkpoints_dir, export_dir, benchmark_dir):
        path.mkdir(parents=True, exist_ok=True)
    return ArtifactLayout(
        run_id=f"{layout.run_id}_snapshot_{step:07d}",
        root=layout.root,
        run_dir=snapshot_run_dir,
        logs_dir=logs_dir,
        reports_dir=reports_dir,
        checkpoints_dir=checkpoints_dir,
        export_dir=export_dir,
        benchmark_dir=benchmark_dir,
        desktop_dir=layout.desktop_dir,
        final_zip_path=layout.final_zip_path,
        final_sha_path=layout.final_sha_path,
    )


def maybe_write_midrun_training_snapshots(
    model: ChessPolicyValueNet,
    cfg: Dict[str, Any],
    device: torch.device,
    layout: ArtifactLayout,
    logger: JSONLLogger,
    *,
    step: int,
    latest_train_row: Dict[str, Any],
    latest_val_eval: Dict[str, Any],
) -> None:
    snapshot_root = layout.reports_dir / "midrun_snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    snapshot_layout = build_snapshot_layout(layout, snapshot_root, step)
    atomic_json(
        snapshot_layout.reports_dir / "training_progress.json",
        {
            "step": step,
            "train_row": latest_train_row,
            "latest_val_eval": latest_val_eval,
        },
    )
    curated_interval = max(0, int(cfg.get("midrun_curated_snapshot_interval", 0)))
    if curated_interval > 0 and step % curated_interval == 0:
        curated_report = evaluate_curated_position_suites(model, cfg, device)
        atomic_json(snapshot_layout.reports_dir / "curated_position_suite_report.json", curated_report)
        atomic_write_text(
            snapshot_layout.reports_dir / "curated_position_suite_report.md",
            render_curated_position_suite_report_md(curated_report),
        )
        logger.write(
            "midrun_snapshot_curated",
            {
                "step": step,
                "exact_hit_rate": curated_report.get("exact_hit_rate", 0.0),
                "top3_hit_rate": curated_report.get("top3_hit_rate", 0.0),
            },
        )
    stockfish_interval = max(0, int(cfg.get("midrun_stockfish_snapshot_interval", 0)))
    if stockfish_interval > 0 and step % stockfish_interval == 0:
        snapshot_cfg = build_midrun_stockfish_snapshot_cfg(cfg)
        if snapshot_cfg.get("stockfish_ladder"):
            stockfish_report = play_stockfish_gauntlet(model, snapshot_cfg, device, snapshot_layout, logger)
            atomic_json(snapshot_layout.reports_dir / "stockfish_snapshot_report.json", stockfish_report)
            logger.write(
                "midrun_snapshot_stockfish",
                {
                    "step": step,
                    "status": stockfish_report.get("status", "not_run"),
                    "elo_proxy_internal": stockfish_report.get("elo_proxy_internal"),
                },
            )


def generate_demo_replay(model: ChessPolicyValueNet, cfg: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    was_training = model.training
    model.eval()
    games: List[Dict[str, Any]] = []
    max_games = int(cfg.get("sample_replay_games", 3))
    max_plies = int(cfg.get("sample_replay_max_plies", 24))
    try:
        for game_idx in range(max_games):
            board = chess.Board()
            opening = OPENING_SEEDS[game_idx % len(OPENING_SEEDS)]
            for move_uci in opening[: min(2, len(opening))]:
                board.push(chess.Move.from_uci(move_uci))
            moves: List[Dict[str, Any]] = []
            while not board.is_game_over() and len(moves) < max_plies:
                trace = choose_move_trace(model, board, device, cfg=cfg)
                move = chess.Move.from_uci(trace["move"])
                if move not in board.legal_moves:
                    break
                board.push(move)
                moves.append({"ply": len(moves) + 1, **trace, "fen": board.fen()})
            games.append(
                {
                    "game_index": game_idx,
                    "opening_prefix": opening,
                    "moves": moves,
                    "final_fen": board.fen(),
                    "demonstration_only": True,
                }
            )
        return {
            "status": "completed",
            "demonstration_only": True,
            "note": "Replay output is demonstration material only and is not a strength proof.",
            "games": games,
        }
    finally:
        model.train(was_training)


def determine_statuses(cfg: Dict[str, Any], benchmark_report: Dict[str, Any]) -> Tuple[ExecutionStatus, EvaluationStatus, RatingClaimStatus]:
    execution_status = ExecutionStatus.RAN
    evaluation_status = EvaluationStatus.INTERNALLY_MEASURED
    if str(cfg.get("mode", "")) == "verify":
        return execution_status, EvaluationStatus.UNEVALUATED, RatingClaimStatus.NO_CLAIM
    if str(cfg.get("mode", "")) == "arena":
        return execution_status, EvaluationStatus.UNEVALUATED, RatingClaimStatus.NO_CLAIM
    if bool(cfg.get("test_mode", False)) or bool(cfg.get("offline_seed_only", False)):
        return execution_status, evaluation_status, RatingClaimStatus.NO_CLAIM
    if benchmark_report.get("status") != "completed":
        return execution_status, evaluation_status, RatingClaimStatus.NO_CLAIM
    total_games = int(benchmark_report.get("games_total", 0))
    elo_proxy_internal = benchmark_report.get("elo_proxy_internal")
    if elo_proxy_internal is None:
        return execution_status, evaluation_status, RatingClaimStatus.PROXY_ONLY
    if total_games < int(cfg.get("claim_min_benchmark_games", 40)):
        return execution_status, evaluation_status, RatingClaimStatus.PROXY_ONLY
    if int(elo_proxy_internal) >= int(cfg.get("rating_target_proxy_threshold", 1600)):
        return execution_status, evaluation_status, RatingClaimStatus.TARGET_MET_INTERNAL
    return execution_status, evaluation_status, RatingClaimStatus.TARGET_NOT_MET


MIRROR_REQUIRED_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "bitlinear": ("BitLinear", "activation_quant", "weight_quant", "set_lowbit_kernel_enabled"),
    "mla": ("MLA", "RotaryEmbedding", "apply_rope_optimized"),
    "liquid_cfc": ("LiquidCell", "LiquidMixer", "jit_liquid_loop_cached"),
    "moe_liquid_router": ("BitSwiGLU", "LiquidRouter", "MoE"),
    "qinn": ("UnitaryQINN", "newton_schulz_inverse"),
    "cognitive_extensions": (
        "GlobalWorkspaceBroadcast",
        "ContinuousLatentODEStateChannel",
        "NeuromodulatoryGainLayer",
        "HebbianPlasticityLayer",
        "NeuroSymbolicLayer",
        "LifelongSafetyLayer",
    ),
    "world_model": ("WorldModelOutput", "CausalWorldModelHead"),
    "transformer_assembly": ("MertFormerBlock", "ChessPolicyValueNet"),
}


def build_mirror_enabled_flags(cfg: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "use_bitnet": bool(cfg.get("use_bitlinear", False)),
        "use_moe": bool(cfg.get("use_moe", False)),
        "use_liquid": bool(cfg.get("use_liquid", cfg.get("use_liquid_adapter", False))),
        "use_qinn": bool(cfg.get("use_qinn", False)),
        "use_global_workspace_broadcast": bool(cfg.get("use_global_workspace_broadcast", False)),
        "use_neuromodulatory_gain": bool(cfg.get("use_neuromodulatory_gain", False)),
        "use_latent_ode_state_channel": bool(cfg.get("use_latent_ode_state_channel", False)),
        "use_cross_expert_sync_bus": bool(cfg.get("use_cross_expert_sync_bus", False)),
        "use_structural_plasticity": bool(cfg.get("use_structural_plasticity", False)),
        "use_hebbian_plasticity": bool(cfg.get("use_hebbian_plasticity", False)),
        "use_neuro_symbolic_layer": bool(cfg.get("use_neuro_symbolic_layer", False)),
        "use_world_model_head": bool(cfg.get("use_world_model_head", False)),
        "use_lifelong_safety_layer": bool(cfg.get("use_lifelong_safety_layer", False)),
        "use_hierarchical_kv_cache": bool(cfg.get("use_hierarchical_kv_cache", False)),
    }


def build_mirror_parity_report(cfg: Dict[str, Any]) -> Dict[str, Any]:
    missing_families = [
        family_name
        for family_name, symbols in MIRROR_REQUIRED_FAMILIES.items()
        if not all(symbol in globals() for symbol in symbols)
    ]
    enabled_flags = build_mirror_enabled_flags(cfg)
    return {
        "script_version": SCRIPT_VERSION,
        "parity_mode": "strict_onefile_mirror",
        "embedding_strategy": "onefile_only",
        "canonical_reference": {
            "layers_root": "layers/",
            "model_root": "model/transformers.py",
        },
        "required_families": sorted(MIRROR_REQUIRED_FAMILIES.keys()),
        "required_symbols": {family_name: list(symbols) for family_name, symbols in MIRROR_REQUIRED_FAMILIES.items()},
        "audit_status": "ok" if not missing_families else "failed",
        "missing_families": missing_families,
        "enabled_flags": enabled_flags,
        "available_but_disabled": sorted(flag for flag, enabled in enabled_flags.items() if not enabled),
        "exact_mirror_scope": [
            "BitLinear quantization surface",
            "MLA/GQA/RoPE attention surface",
            "CfC LiquidCell/LiquidMixer surface",
            "MoE/LiquidRouter routing surface",
            "QINN surface",
            "cognitive extension surfaces",
            "world-model head surface",
            "transformer block assembly order",
        ],
        "chess_specific_surface": [
            "board piece embeddings",
            "meta-token embeddings",
            "pooled policy/value heads",
            "legal move masking",
            "arena/train/package/report operators",
        ],
        "hardening_scope": {
            "stable_compiler_safe_only": True,
            "anti_debug_or_vm_protector": False,
        },
    }


def assert_mirror_surface_integrity(cfg: Dict[str, Any]) -> Dict[str, Any]:
    report = build_mirror_parity_report(cfg)
    if report["audit_status"] != "ok":
        raise ChessOnefileError(f"Strict mirror audit failed: missing families={report['missing_families']}")
    return report


def build_model_card(model: ChessPolicyValueNet, cfg: Dict[str, Any], checkpoint_path: Optional[Path]) -> Dict[str, Any]:
    report = model.parameter_report()
    checkpoint_size = checkpoint_path.stat().st_size if checkpoint_path and checkpoint_path.exists() else 0
    parity_report = build_mirror_parity_report(cfg)
    feature_report = build_feature_flag_report(cfg)
    report.update(
        {
            "script_version": SCRIPT_VERSION,
            "baseline": cfg.get("baseline", "dense"),
            "feature_bundle": cfg.get("feature_bundle", "default"),
            "feature_flags": feature_report,
            "hidden_size": int(cfg["hidden_size"]),
            "num_layers": int(cfg["num_layers"]),
            "num_heads": int(cfg["num_heads"]),
            "use_moe": bool(cfg.get("use_moe", False)),
            "use_bitlinear": bool(cfg.get("use_bitlinear", False)),
            "use_liquid": bool(cfg.get("use_liquid", cfg.get("use_liquid_adapter", False))),
            "use_qinn": bool(cfg.get("use_qinn", False)),
            "moe_top_k": int(cfg.get("moe_top_k", 2)),
            "enabled_auxiliary_heads": [
                head_name
                for head_name, enabled in (
                    ("phase_head", bool(cfg.get("use_phase_head", False))),
                    ("wdl_head", bool(cfg.get("use_wdl_head", False))),
                    ("legality_head", bool(cfg.get("use_legality_head", False))),
                )
                if enabled
            ],
            "checkpoint_size_bytes": int(checkpoint_size),
            "move_vocab_size": len(MOVE_VOCAB),
            "move_vocab_hash": MOVE_VOCAB_HASH,
            "mirror_parity": parity_report,
            "architecture_notes": [
                "Board attention is intentionally non-causal: the model sees the whole board state at once.",
                "The chess trunk is a strict onefile mirror of the canonical Build30 attention, CfC liquid, MoE, QINN, and extension families.",
                "Policy head remains a pooled global move-classifier over a fixed UCI vocabulary on top of the mirrored trunk.",
                "Optional phase/WDL/legality auxiliary heads can be enabled to shape chess-specific supervision without mutating the mirrored trunk contract.",
                "Chess-specific behavior is limited to board/meta tokenization, legality masking, and policy/value operator surfaces.",
            ],
        }
    )
    return report


def build_eval_card(
    cfg: Dict[str, Any],
    val_eval: Dict[str, Any],
    test_eval: Dict[str, Any],
    legality_report: Dict[str, Any],
    benchmark_report: Dict[str, Any],
    curated_position_suite_report: Dict[str, Any],
    selfplay_report: Dict[str, Any],
    tournament_report: Dict[str, Any],
    replay_buffer_report: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "mirror_parity": build_mirror_parity_report(cfg),
        "feature_flags": build_feature_flag_report(cfg),
        "holdout_validation": val_eval,
        "locked_test": test_eval,
        "raw_vs_masked_policy_metrics": legality_report,
        "benchmark_protocol": "internal_stockfish_gauntlet_v2",
        "benchmark_result": benchmark_report,
        "curated_position_suite": curated_position_suite_report,
        "selfplay_report": selfplay_report,
        "inference_mode_tournament": tournament_report,
        "replay_buffer_manifest": replay_buffer_report,
        "rating_note": "Strength outputs are internal-only unless externally verified.",
        "parity_scope_note": "Exact mirror parity covers the canonical trunk families; chess-specific heads and legality surfaces remain domain-specific.",
    }


def render_run_summary_md(payload: Dict[str, Any]) -> str:
    verify_mode = str(payload["config"].get("mode", "")) == "verify"
    arena_mode = str(payload["config"].get("mode", "")) == "arena"
    logging_report = payload.get("logging", {})
    lines = [
        "# MertFormer Chess Run Summary",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Mode: `{payload['config']['mode']}`",
        f"- Profile: `{payload['config']['profile']}`",
        f"- Baseline: `{payload['config']['baseline']}`",
        f"- Feature bundle: `{payload['config'].get('feature_bundle', 'default')}`",
        f"- Enabled feature flags: `{payload.get('feature_flags', {}).get('enabled_count', 0)}`",
        f"- Mirror parity: `{payload.get('mirror_parity', {}).get('parity_mode', 'unknown')}`",
        f"- Execution status: `{payload['execution_status']}`",
        f"- Evaluation status: `{payload['evaluation_status']}`",
        f"- Rating claim status: `{payload['rating_claim_status']}`",
        f"- Proxy threshold target: `{payload['rating_target_proxy_threshold']}`",
        f"- Logging schema version: `{logging_report.get('schema_version', LOG_SCHEMA_VERSION)}`",
        f"- Logged events: `{logging_report.get('event_count', 0)}`",
        "",
        "## What This Proves",
        "- The onefile can ingest bounded Lichess data, build a legal-move-safe supervised chess dataset, and package measurable artifacts.",
        "",
        "## What This Does Not Prove",
        "- This run does not prove frontier general-purpose LLM capability.",
        "- Replay/demo output is not strength proof.",
        "- Any `elo_proxy_internal` value is a proxy, not an externally verified rating.",
    ]
    if verify_mode:
        lines.extend(
            [
                "",
                "## Verify Scope",
                "- Verify mode is runtime-only: holdout evaluation, legality scoring, replay, and Stockfish benchmarking are intentionally skipped.",
                f"- Forward verify status: `{payload['forward_verify'].get('status', 'unknown')}`",
                f"- Forward verify batch size checked: `{payload['forward_verify'].get('checked', 0)}`",
            ]
        )
    elif arena_mode:
        arena_session = payload.get("arena_session", {})
        lines.extend(
            [
                "",
                "## Arena Scope",
                "- Arena mode skips dataset ingestion, training, holdout metrics, and Stockfish benchmarking.",
                f"- Arena status: `{arena_session.get('status', 'unknown')}`",
                f"- Arena result: `{arena_session.get('result', '*')}`",
                f"- Arena plies played: `{arena_session.get('plies_played', 0)}`",
            ]
        )
    else:
        lines.extend(
            [
                "- Holdout metrics, legality metrics, and optional internal Stockfish gauntlet results were generated from this run.",
                "",
                "## Key Metrics",
                f"- Validation masked policy accuracy: `{payload['holdout_validation']['metrics'].get('masked_policy_accuracy', 0.0):.4f}`",
                f"- Locked test masked policy accuracy: `{payload['locked_test']['metrics'].get('masked_policy_accuracy', 0.0):.4f}`",
                f"- Raw top-1 legality: `{payload['legality_report'].get('raw_top1_is_legal_rate', 0.0):.4f}`",
                f"- Raw top-k contains legal: `{payload['legality_report'].get('raw_topk_contains_legal_rate', 0.0):.4f}`",
            ]
        )
    benchmark = payload.get("stockfish", {})
    curated_suite = payload.get("curated_position_suite", {})
    training_augmentation = payload.get("training_augmentation", {})
    if verify_mode:
        lines.append("- Internal gauntlet: `not_run (verify mode)`")
    elif arena_mode:
        lines.append("- Internal gauntlet: `not_run (arena mode)`")
    elif benchmark.get("status") == "completed":
        lines.extend(
            [
                f"- Internal gauntlet games: `{benchmark.get('games_total', 0)}`",
                f"- Internal elo proxy: `{benchmark.get('elo_proxy_internal')}`",
            ]
        )
    else:
        lines.append(f"- Internal gauntlet: `{benchmark.get('status', 'not_run')}`")
    if curated_suite.get("status") == "completed":
        lines.extend(
            [
                f"- Curated suite exact-hit rate: `{curated_suite.get('exact_hit_rate', 0.0):.4f}`",
                f"- Curated suite top-3 hit rate: `{curated_suite.get('top3_hit_rate', 0.0):.4f}`",
                f"- Curated suite tag coverage: `{curated_suite.get('expected_tag_coverage_rate', 0.0):.4f}`",
            ]
        )
    else:
        lines.append(f"- Curated suite: `{curated_suite.get('status', 'not_run')}`")
    if training_augmentation.get("enabled", False):
        lines.append(f"- Curated training augmentation examples: `{training_augmentation.get('examples_total', 0)}`")
    bundle = payload.get("bundle", {})
    lines.extend(
        [
            "",
            "## Feature Flags",
            f"- Explicit enable overrides: `{', '.join(payload.get('feature_flags', {}).get('explicitly_enabled', [])) or 'none'}`",
            f"- Explicit disable overrides: `{', '.join(payload.get('feature_flags', {}).get('explicitly_disabled', [])) or 'none'}`",
            f"- Top enabled surfaces: `{', '.join(payload.get('feature_flags', {}).get('enabled_features', [])[:12]) or 'none'}`",
            "",
            "## Post-Run Analysis",
            f"- Self-play status: `{payload.get('selfplay_report', {}).get('status', 'unknown')}`",
            f"- Tournament status: `{payload.get('tournament_report', {}).get('status', 'unknown')}`",
            f"- Replay-buffer status: `{payload.get('replay_buffer_report', {}).get('status', 'unknown')}`",
            "",
            "## Bundle",
            f"- Output root: `{payload['output_root']}`",
            f"- Final zip: `{bundle.get('zip_path', '')}`",
            f"- Final sha256: `{bundle.get('sha256', '')}`",
            "",
            "## Observability",
            f"- Main log: `{payload['output_root']}/logs/run_log.jsonl`",
            f"- Contract: `{payload['output_root']}/reports/logging_contract.json`",
            f"- Report: `{payload['output_root']}/reports/observability_report.json`",
            "- Fatal exceptions are recorded in both the run log and the desktop FAILED artifact.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def render_proof_scope_md() -> str:
    return textwrap.dedent(
        """
        # Proof Scope

        ## What This Run Proves
        - A consumer-class single-machine pipeline can ingest bounded chess data, build a legality-safe dataset, train a supervised policy/value network, emit reproducible artifact packs, and run an interactive human-vs-model arena session.
        - The run records data provenance, split manifests, legality metrics, holdout metrics, and optional internal benchmark outputs.

        ## What This Run Does Not Prove
        - This is not a frontier general-purpose LLM benchmark.
        - This is not an externally verified chess rating.
        - Replay demonstrations are not performance proof.
        - If Stockfish benchmarking is absent or limited, rating claims remain `no_claim` or `proxy_only`.
        """
    ).strip() + "\n"


def render_repro_md(cfg: Dict[str, Any], layout: ArtifactLayout) -> str:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--mode", str(cfg["mode"]), "--profile", str(cfg["profile"]), "--baseline", str(cfg["baseline"])]
    feature_bundle = str(cfg.get("feature_bundle", "default")).strip()
    if feature_bundle and feature_bundle != "default":
        cmd.extend(["--feature-bundle", feature_bundle])
    enabled_features = parse_feature_list(cfg.get("enabled_features", []))
    disabled_features = parse_feature_list(cfg.get("disabled_features", []))
    if enabled_features:
        cmd.extend(["--enable-features", ",".join(enabled_features)])
    if disabled_features:
        cmd.extend(["--disable-features", ",".join(disabled_features)])
    if str(cfg.get("resume_from", "")).strip():
        cmd.extend(["--resume-from", str(cfg["resume_from"])])
    if bool(cfg.get("offline_seed_only", False)):
        cmd.append("--offline-seed-only")
    if bool(cfg.get("share_mode", False)):
        cmd.append("--share-mode")
    if bool(cfg.get("enable_self_delete", False)):
        cmd.append("--enable-self-delete")
    if str(cfg.get("self_delete_target", "")).strip():
        cmd.extend(["--self-delete-target", str(cfg["self_delete_target"])])
    return textwrap.dedent(
        f"""
        # Repro Instructions

        Run command:

        ```bash
        {' '.join(cmd)}
        ```

        Artifact root:
        - `{layout.run_dir}`

        Notes:
        - This onefile defaults to proof-safe behavior: no self-delete unless explicitly enabled and bound to an explicit shared-copy target.
        - Rating outputs are internal proxies unless externally verified.
        """
    ).strip() + "\n"


def render_third_party_licenses() -> str:
    return textwrap.dedent(
        """
        THIRD-PARTY DATA NOTICES
        =======================

        Source: Lichess database archives
        URL: https://database.lichess.org/

        Usage note:
        - This run may consume partial slices of Lichess standard rated game archives.
        - Operators should review the current Lichess database usage and licensing terms before external distribution.
        - This artifact pack records source URLs and archive checksums for auditability.
        """
    ).strip() + "\n"


def write_cards_and_reports(
    layout: ArtifactLayout,
    cfg: Dict[str, Any],
    payload: Dict[str, Any],
    data_card: Dict[str, Any],
    model_card: Dict[str, Any],
    eval_card: Dict[str, Any],
    benchmark_protocol: Dict[str, Any],
    dependency_lock: Dict[str, Any],
    env_info: Dict[str, Any],
    curve_rows: Sequence[Dict[str, Any]],
    logger: Optional[JSONLLogger] = None,
) -> None:
    reports = layout.reports_dir
    payload["mirror_parity"] = payload.get("mirror_parity", build_mirror_parity_report(cfg))
    payload["feature_flags"] = payload.get("feature_flags", build_feature_flag_report(cfg))
    if logger is not None:
        payload["logging"] = logger.observability_report()
    atomic_json(reports / "run_summary.json", payload)
    atomic_write_text(reports / "run_summary.md", render_run_summary_md(payload))
    atomic_json(reports / "data_card.json", data_card)
    atomic_json(reports / "model_card.json", model_card)
    atomic_json(reports / "eval_card.json", eval_card)
    atomic_json(reports / "mirror_parity_report.json", payload["mirror_parity"])
    atomic_json(reports / "feature_flag_report.json", payload["feature_flags"])
    atomic_write_text(reports / "feature_flag_report.md", render_feature_flag_report_md(payload["feature_flags"]))
    atomic_json(reports / "benchmark_protocol.json", benchmark_protocol)
    atomic_json(reports / "dependency_lock.json", dependency_lock)
    atomic_json(reports / "environment_snapshot.json", env_info)
    atomic_json(reports / "dataset_provenance.json", payload["dataset_provenance"])
    atomic_json(reports / "holdout_metrics.json", payload["holdout_validation"])
    atomic_json(reports / "locked_test_metrics.json", payload["locked_test"])
    atomic_json(reports / "legal_move_safety.json", payload["legality_report"])
    atomic_json(reports / "raw_vs_masked_policy_metrics.json", payload["legality_report"])
    atomic_json(reports / "opening_distribution.json", payload["dataset_provenance"]["data_stats"].get("opening_distribution_top20", {}))
    atomic_json(reports / "phase_distribution.json", payload["dataset_provenance"]["data_stats"].get("phase_distribution", {}))
    atomic_json(reports / "drop_reason_counts.json", payload["dataset_provenance"]["data_stats"].get("drop_reason_counts", {}))
    atomic_json(reports / "curated_position_manifest.json", payload["curated_position_manifest"])
    atomic_write_text(reports / "curated_position_manifest.md", render_curated_position_manifest_md(payload["curated_position_manifest"]))
    atomic_json(reports / "synthetic_teaching_corpus.json", payload["synthetic_teaching_corpus"])
    atomic_write_text(reports / "synthetic_teaching_corpus.md", render_synthetic_teaching_corpus_md(payload["synthetic_teaching_corpus"]))
    atomic_json(reports / "curated_position_suite_report.json", payload["curated_position_suite"])
    atomic_write_text(reports / "curated_position_suite_report.md", render_curated_position_suite_report_md(payload["curated_position_suite"]))
    atomic_json(reports / "selfplay_report.json", payload["selfplay_report"])
    atomic_write_text(reports / "selfplay_report.md", render_selfplay_report_md(payload["selfplay_report"]))
    atomic_json(reports / "inference_mode_tournament_report.json", payload["tournament_report"])
    atomic_write_text(reports / "inference_mode_tournament_report.md", render_tournament_report_md(payload["tournament_report"]))
    atomic_json(reports / "replay_buffer_manifest.json", payload["replay_buffer_report"])
    atomic_write_text(reports / "replay_buffer_manifest.md", render_replay_buffer_report_md(payload["replay_buffer_report"]))
    if "arena_session" in payload:
        atomic_json(reports / "arena_session.json", payload["arena_session"])
    atomic_write_text(reports / "PROOF_SCOPE.md", render_proof_scope_md())
    atomic_write_text(reports / "REPRO.md", render_repro_md(cfg, layout))
    atomic_write_text(reports / "THIRD_PARTY_DATA_LICENSES.txt", render_third_party_licenses())
    write_curve_csv(reports / "training_curve.csv", curve_rows)
    write_curve_png(reports / "training_curve.png", curve_rows)
    if logger is not None:
        atomic_json(reports / "logging_contract.json", logger.contract())
        atomic_json(reports / "observability_report.json", logger.observability_report())


def build_artifact_manifest(layout: ArtifactLayout) -> Dict[str, Any]:
    manifest_entries: List[Dict[str, Any]] = []
    manifest_path = layout.reports_dir / "artifact_manifest_with_hashes.json"
    for path in sorted(layout.run_dir.rglob("*")):
        if path.is_dir() or path == manifest_path:
            continue
        manifest_entries.append(
            {
                "relative_path": str(path.relative_to(layout.run_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": path_sha256(path),
            }
        )
    manifest = {
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": utc_now(),
        "entry_count": len(manifest_entries),
        "entries": manifest_entries,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def build_run_status_manifest(payload: Dict[str, Any]) -> Dict[str, Any]:
    config = dict(payload.get("config", {}))
    return {
        "schema": "chess_run_status_manifest_v1",
        "script_version": payload.get("script_version", SCRIPT_VERSION),
        "run_id": payload.get("run_id", ""),
        "mode": config.get("mode", ""),
        "profile": config.get("profile", ""),
        "baseline": config.get("baseline", ""),
        "feature_bundle": config.get("feature_bundle", "default"),
        "execution_status": payload.get("execution_status", "unknown"),
        "evaluation_status": payload.get("evaluation_status", "unknown"),
        "rating_claim_status": payload.get("rating_claim_status", "unknown"),
        "best_checkpoint": payload.get("best_checkpoint", ""),
        "latest_checkpoint": payload.get("latest_checkpoint", ""),
        "bundle": payload.get("bundle", {}),
        "status_surfaces": {
            "compile_report": payload.get("compile_report", {}).get("status", payload.get("compile_report", {}).get("mode", "unknown")),
            "forward_verify": payload.get("forward_verify", {}).get("status", "unknown"),
            "holdout_validation": payload.get("holdout_validation", {}).get("status", "completed"),
            "locked_test": payload.get("locked_test", {}).get("status", "completed"),
            "legality_report": payload.get("legality_report", {}).get("status", "completed"),
            "stockfish": payload.get("stockfish", {}).get("status", "unknown"),
            "curated_position_suite": payload.get("curated_position_suite", {}).get("status", "unknown"),
            "selfplay_report": payload.get("selfplay_report", {}).get("status", "unknown"),
            "tournament_report": payload.get("tournament_report", {}).get("status", "unknown"),
            "replay_buffer_report": payload.get("replay_buffer_report", {}).get("status", "unknown"),
        },
    }


def render_run_status_manifest_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Run Status Manifest",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- mode: `{report.get('mode', '')}`",
        f"- profile: `{report.get('profile', '')}`",
        f"- baseline: `{report.get('baseline', '')}`",
        f"- feature_bundle: `{report.get('feature_bundle', 'default')}`",
        f"- execution_status: `{report.get('execution_status', 'unknown')}`",
        f"- evaluation_status: `{report.get('evaluation_status', 'unknown')}`",
        f"- rating_claim_status: `{report.get('rating_claim_status', 'unknown')}`",
        "",
        "## Status Surfaces",
    ]
    for name, status in sorted(report.get("status_surfaces", {}).items()):
        lines.append(f"- `{name}`: `{status}`")
    return "\n".join(lines) + "\n"


def build_postrun_analysis_manifest(payload: Dict[str, Any]) -> Dict[str, Any]:
    selfplay_report = dict(payload.get("selfplay_report", {}))
    tournament_report = dict(payload.get("tournament_report", {}))
    replay_buffer_report = dict(payload.get("replay_buffer_report", {}))
    return {
        "schema": "chess_postrun_analysis_manifest_v1",
        "selfplay": {
            "status": selfplay_report.get("status", "unknown"),
            "games_played": int(selfplay_report.get("games_played", 0)),
            "average_plies": float(selfplay_report.get("average_plies", 0.0)),
        },
        "tournament": {
            "status": tournament_report.get("status", "unknown"),
            "games_played": int(tournament_report.get("games_played", 0)),
            "players": tournament_report.get("players", {}),
        },
        "replay_buffer": {
            "status": replay_buffer_report.get("status", "unknown"),
            "positions": int(replay_buffer_report.get("positions", 0)),
            "games_used": int(replay_buffer_report.get("games_used", 0)),
            "truncated": bool(replay_buffer_report.get("truncated", False)),
        },
        "curated_position_suite": {
            "status": payload.get("curated_position_suite", {}).get("status", "unknown"),
            "exact_hit_rate": float(payload.get("curated_position_suite", {}).get("exact_hit_rate", 0.0)),
            "top3_hit_rate": float(payload.get("curated_position_suite", {}).get("top3_hit_rate", 0.0)),
        },
        "stockfish": {
            "status": payload.get("stockfish", {}).get("status", "unknown"),
            "games_total": int(payload.get("stockfish", {}).get("games_total", 0)),
            "elo_proxy_internal": payload.get("stockfish", {}).get("elo_proxy_internal"),
        },
    }


def render_postrun_analysis_manifest_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Post-Run Analysis Manifest",
        "",
        f"- selfplay_status: `{report.get('selfplay', {}).get('status', 'unknown')}`",
        f"- tournament_status: `{report.get('tournament', {}).get('status', 'unknown')}`",
        f"- replay_buffer_status: `{report.get('replay_buffer', {}).get('status', 'unknown')}`",
        f"- curated_suite_status: `{report.get('curated_position_suite', {}).get('status', 'unknown')}`",
        f"- stockfish_status: `{report.get('stockfish', {}).get('status', 'unknown')}`",
        "",
        "## Counts",
        f"- selfplay_games_played: `{report.get('selfplay', {}).get('games_played', 0)}`",
        f"- tournament_games_played: `{report.get('tournament', {}).get('games_played', 0)}`",
        f"- replay_buffer_positions: `{report.get('replay_buffer', {}).get('positions', 0)}`",
        f"- stockfish_games_total: `{report.get('stockfish', {}).get('games_total', 0)}`",
    ]
    return "\n".join(lines) + "\n"


def build_artifact_truth_matrix(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    reports = layout.reports_dir
    entries: List[Dict[str, Any]] = []

    def add_entry(label: str, path: Path, *, required: bool, kind: str) -> None:
        entries.append(
            {
                "label": label,
                "kind": kind,
                "required": bool(required),
                "path": str(path),
                "exists": path.exists(),
            }
        )

    for label, filename in (
        ("resolved_config", "resolved_config.json"),
        ("run_summary_json", "run_summary.json"),
        ("run_summary_md", "run_summary.md"),
        ("data_card", "data_card.json"),
        ("model_card", "model_card.json"),
        ("eval_card", "eval_card.json"),
        ("benchmark_protocol", "benchmark_protocol.json"),
        ("feature_flag_report_json", "feature_flag_report.json"),
        ("feature_flag_report_md", "feature_flag_report.md"),
        ("observability_report", "observability_report.json"),
        ("artifact_manifest", "artifact_manifest_with_hashes.json"),
        ("run_status_manifest", "run_status_manifest.json"),
        ("postrun_analysis_manifest", "postrun_analysis_manifest.json"),
        ("artifact_truth_matrix", "artifact_truth_matrix.json"),
        ("run_contract", "run_contract.json"),
        ("release_snapshot", "release_snapshot.json"),
        ("evidence_pack_stub", "evidence_pack_stub.json"),
        ("final_truth_registry", "final_truth_registry.json"),
        ("claim_registry", "claim_registry.json"),
        ("known_limits", "known_limits.json"),
        ("support_matrix", "support_matrix.json"),
        ("release_gate_summary", "release_gate_summary.json"),
        ("rc_stub", "rc_stub.json"),
        ("golden_stub", "golden_stub.json"),
        ("handoff_pack_manifest", "handoff_pack_manifest.json"),
        ("operator_handoff_summary", "operator_handoff_summary.json"),
        ("external_repro_stub", "external_repro_stub.json"),
        ("pilot_stub", "pilot_stub.json"),
        ("security_stub", "security_stub.json"),
        ("legal_stub", "legal_stub.json"),
        ("operator_handbook_stub", "operator_handbook_stub.json"),
        ("dr_evidence_stub", "dr_evidence_stub.json"),
        ("backup_retention_stub", "backup_retention_stub.json"),
        ("blind_handoff_stub", "blind_handoff_stub.json"),
        ("release_notes_stub", "release_notes_stub.json"),
        ("freeze_manifest_stub", "freeze_manifest_stub.json"),
        ("changelog_snapshot", "changelog_snapshot.json"),
        ("maintenance_policy_stub", "maintenance_policy_stub.json"),
        ("export_truth_stub", "export_truth_stub.json"),
        ("device_validation_stub", "device_validation_stub.json"),
        ("packaging_closure_stub", "packaging_closure_stub.json"),
        ("installer_validation_stub", "installer_validation_stub.json"),
        ("benchmark_raw_outputs_stub", "benchmark_raw_outputs_stub.json"),
        ("benchmark_compare_report_stub", "benchmark_compare_report_stub.json"),
        ("benchmark_summary_stub", "benchmark_summary_stub.json"),
        ("benchmark_manifest_stub", "benchmark_manifest_stub.json"),
        ("training_report_stub", "training_report_stub.json"),
        ("token_accounting_stub", "token_accounting_stub.json"),
        ("compute_accounting_stub", "compute_accounting_stub.json"),
        ("cost_report_stub", "cost_report_stub.json"),
        ("final_weights_truth_stub", "final_weights_truth_stub.json"),
        ("best_checkpoint_truth_stub", "best_checkpoint_truth_stub.json"),
        ("latest_checkpoint_truth_stub", "latest_checkpoint_truth_stub.json"),
        ("trained_artifact_registry_stub", "trained_artifact_registry_stub.json"),
        ("core_complete_decision_stub", "core_complete_decision_stub.json"),
        ("research_continues_stub", "research_continues_stub.json"),
        ("product_maintenance_only_stub", "product_maintenance_only_stub.json"),
        ("closure_decision_record_stub", "closure_decision_record_stub.json"),
        ("master_closure_table", "master_closure_table.json"),
        ("remaining_core_blockers", "remaining_core_blockers.json"),
        ("repo_side_completion_summary", "repo_side_completion_summary.json"),
        ("readiness_snapshot", "readiness_snapshot.json"),
        ("aggregated_master_table", "aggregated_master_table.json"),
        ("real_remaining_core_work", "real_remaining_core_work.json"),
        ("repo_truth_inventory", "repo_truth_inventory.json"),
        ("closure_gap_summary", "closure_gap_summary.json"),
        ("project_master_truth_reference", "project_master_truth_reference.json"),
        ("project_remaining_real_blockers", "project_remaining_real_blockers.json"),
        ("truth_docs_index", "truth_docs_index.json"),
        ("truth_docs_drift_report", "truth_docs_drift_report.json"),
        ("project_blocker_action_plan", "project_blocker_action_plan.json"),
        ("generated_truth_consistency_report", "generated_truth_consistency_report.json"),
        ("project_blocker_dependency_graph", "project_blocker_dependency_graph.json"),
        ("project_execution_sequence", "project_execution_sequence.json"),
        ("project_lane_status_board", "project_lane_status_board.json"),
        ("project_closure_phase_plan", "project_closure_phase_plan.json"),
        ("project_phase_readiness_scoreboard", "project_phase_readiness_scoreboard.json"),
        ("project_owner_accountability_matrix", "project_owner_accountability_matrix.json"),
        ("project_owner_work_queue", "project_owner_work_queue.json"),
        ("project_critical_path_report", "project_critical_path_report.json"),
        ("project_owner_next_actions_summary", "project_owner_next_actions_summary.json"),
        ("project_ready_now_board", "project_ready_now_board.json"),
        ("project_unlock_impact_report", "project_unlock_impact_report.json"),
        ("generated_truth_crosscheck_matrix", "generated_truth_crosscheck_matrix.json"),
        ("selfplay_report", "selfplay_report.json"),
        ("tournament_report", "inference_mode_tournament_report.json"),
        ("replay_buffer_manifest", "replay_buffer_manifest.json"),
    ):
        add_entry(label, reports / filename, required=True, kind="report")

    add_entry("run_log", layout.logs_dir / "run_log.jsonl", required=True, kind="log")
    best_checkpoint = str(payload.get("best_checkpoint", "")).strip()
    latest_checkpoint = str(payload.get("latest_checkpoint", "")).strip()
    if best_checkpoint:
        add_entry("best_checkpoint", Path(best_checkpoint), required=False, kind="checkpoint")
    if latest_checkpoint and latest_checkpoint != best_checkpoint:
        add_entry("latest_checkpoint", Path(latest_checkpoint), required=False, kind="checkpoint")
    bundle = dict(payload.get("bundle", {}))
    zip_path = str(bundle.get("zip_path", "")).strip()
    sha_path = str(bundle.get("sha256_path", "")).strip()
    if zip_path:
        add_entry("bundle_zip", Path(zip_path), required=True, kind="bundle")
    if sha_path:
        add_entry("bundle_sha", Path(sha_path), required=False, kind="bundle")
    return {
        "schema": "chess_artifact_truth_matrix_v1",
        "run_id": payload.get("run_id", ""),
        "required_count": sum(1 for entry in entries if entry["required"]),
        "present_required_count": sum(1 for entry in entries if entry["required"] and entry["exists"]),
        "entries": entries,
    }


def render_artifact_truth_matrix_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Artifact Truth Matrix",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- required_count: `{report.get('required_count', 0)}`",
        f"- present_required_count: `{report.get('present_required_count', 0)}`",
        "",
        "## Entries",
    ]
    for entry in report.get("entries", []):
        lines.append(
            f"- `{entry.get('label', '')}`: exists=`{entry.get('exists', False)}` "
            f"required=`{entry.get('required', False)}` kind=`{entry.get('kind', '')}` path=`{entry.get('path', '')}`"
        )
    return "\n".join(lines) + "\n"


def write_closure_manifests(layout: ArtifactLayout, payload: Dict[str, Any]) -> None:
    run_status = build_run_status_manifest(payload)
    atomic_json(layout.reports_dir / "run_status_manifest.json", run_status)
    atomic_write_text(layout.reports_dir / "run_status_manifest.md", render_run_status_manifest_md(run_status))
    postrun = build_postrun_analysis_manifest(payload)
    atomic_json(layout.reports_dir / "postrun_analysis_manifest.json", postrun)
    atomic_write_text(layout.reports_dir / "postrun_analysis_manifest.md", render_postrun_analysis_manifest_md(postrun))
    # Emit the upstream closure manifests first so the truth matrix can account for
    # them, then rewrite once so the matrix also records its own JSON artifact.
    truth = build_artifact_truth_matrix(layout, payload)
    atomic_json(layout.reports_dir / "artifact_truth_matrix.json", truth)
    atomic_write_text(layout.reports_dir / "artifact_truth_matrix.md", render_artifact_truth_matrix_md(truth))
    truth = build_artifact_truth_matrix(layout, payload)
    atomic_json(layout.reports_dir / "artifact_truth_matrix.json", truth)
    atomic_write_text(layout.reports_dir / "artifact_truth_matrix.md", render_artifact_truth_matrix_md(truth))


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def build_run_contract(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(payload.get("config", {}))
    notes = dict(payload.get("notes", {}))
    provenance = dict(payload.get("dataset_provenance", {}))
    return {
        "schema": "chess_run_contract_v1",
        "script_version": payload.get("script_version", SCRIPT_VERSION),
        "run_id": payload.get("run_id", ""),
        "mode": cfg.get("mode", ""),
        "profile": cfg.get("profile", ""),
        "baseline": cfg.get("baseline", ""),
        "feature_bundle": cfg.get("feature_bundle", "default"),
        "artifact_root": str(layout.run_dir),
        "operator_boundaries": {
            "package_only": bool(notes.get("package_only", False)),
            "replay_is_demo_only": bool(notes.get("replay_is_demo_only", True)),
            "internal_proxy_only": bool(notes.get("internal_proxy_only", True)),
        },
        "dataset_contract": {
            "offline_seed_only": bool(cfg.get("offline_seed_only", False)),
            "sampling_strategy": provenance.get("sampling_strategy", "unknown"),
            "source_mode": provenance.get("source_mode", provenance.get("mode", "unknown")),
        },
        "required_core_reports": [
            "run_summary.json",
            "model_card.json",
            "eval_card.json",
            "feature_flag_report.json",
            "run_status_manifest.json",
            "postrun_analysis_manifest.json",
            "artifact_truth_matrix.json",
        ],
        "checkpoint_contract": {
            "best_checkpoint": payload.get("best_checkpoint", ""),
            "latest_checkpoint": payload.get("latest_checkpoint", ""),
        },
        "claim_boundary": {
            "execution_status": payload.get("execution_status", "unknown"),
            "evaluation_status": payload.get("evaluation_status", "unknown"),
            "rating_claim_status": payload.get("rating_claim_status", "unknown"),
            "what_this_proves": notes.get("what_this_proves", ""),
            "what_this_does_not_prove": notes.get("what_this_does_not_prove", ""),
        },
    }


def render_run_contract_md(report: Dict[str, Any]) -> str:
    dataset_contract = dict(report.get("dataset_contract", {}))
    operator_boundaries = dict(report.get("operator_boundaries", {}))
    claim_boundary = dict(report.get("claim_boundary", {}))
    lines = [
        "# Run Contract",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- mode: `{report.get('mode', '')}`",
        f"- profile: `{report.get('profile', '')}`",
        f"- baseline: `{report.get('baseline', '')}`",
        f"- feature_bundle: `{report.get('feature_bundle', 'default')}`",
        f"- artifact_root: `{report.get('artifact_root', '')}`",
        "",
        "## Dataset Contract",
        f"- offline_seed_only: `{dataset_contract.get('offline_seed_only', False)}`",
        f"- sampling_strategy: `{dataset_contract.get('sampling_strategy', 'unknown')}`",
        f"- source_mode: `{dataset_contract.get('source_mode', 'unknown')}`",
        "",
        "## Operator Boundaries",
        f"- package_only: `{operator_boundaries.get('package_only', False)}`",
        f"- replay_is_demo_only: `{operator_boundaries.get('replay_is_demo_only', True)}`",
        f"- internal_proxy_only: `{operator_boundaries.get('internal_proxy_only', True)}`",
        "",
        "## Claim Boundary",
        f"- execution_status: `{claim_boundary.get('execution_status', 'unknown')}`",
        f"- evaluation_status: `{claim_boundary.get('evaluation_status', 'unknown')}`",
        f"- rating_claim_status: `{claim_boundary.get('rating_claim_status', 'unknown')}`",
        f"- what_this_proves: {claim_boundary.get('what_this_proves', '')}",
        f"- what_this_does_not_prove: {claim_boundary.get('what_this_does_not_prove', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_release_snapshot(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    truth = _read_json_if_exists(layout.reports_dir / "artifact_truth_matrix.json")
    bundle = dict(payload.get("bundle", {}))
    notes = dict(payload.get("notes", {}))
    zip_path = str(bundle.get("zip_path", "")).strip()
    sha_path = str(bundle.get("sha256_path", "")).strip()
    best_checkpoint = str(payload.get("best_checkpoint", "")).strip()
    latest_checkpoint = str(payload.get("latest_checkpoint", "")).strip()
    checkpoint_ready = bool(best_checkpoint or latest_checkpoint or notes.get("package_only", False))
    bundle_exists = Path(zip_path).exists() if zip_path else False
    sha_exists = Path(sha_path).exists() if sha_path else False
    required_count = int(truth.get("required_count", 0))
    present_required_count = int(truth.get("present_required_count", 0))
    core_reports_ready = required_count > 0 and present_required_count == required_count
    release_surface_status = "candidate_internal_only" if core_reports_ready and checkpoint_ready and bundle_exists else "incomplete"
    return {
        "schema": "chess_release_snapshot_v1",
        "script_version": payload.get("script_version", SCRIPT_VERSION),
        "run_id": payload.get("run_id", ""),
        "mode": payload.get("config", {}).get("mode", ""),
        "profile": payload.get("config", {}).get("profile", ""),
        "feature_bundle": payload.get("config", {}).get("feature_bundle", "default"),
        "execution_status": payload.get("execution_status", "unknown"),
        "evaluation_status": payload.get("evaluation_status", "unknown"),
        "rating_claim_status": payload.get("rating_claim_status", "unknown"),
        "required_count": required_count,
        "present_required_count": present_required_count,
        "core_reports_ready": core_reports_ready,
        "checkpoint_ready": checkpoint_ready,
        "bundle": {
            "zip_path": zip_path,
            "zip_exists": bundle_exists,
            "sha256_path": sha_path,
            "sha256_exists": sha_exists,
            "encrypted": bool(bundle.get("encrypted", False)),
        },
        "best_checkpoint": best_checkpoint,
        "latest_checkpoint": latest_checkpoint,
        "release_surface_status": release_surface_status,
        "external_release_grade": False,
        "external_release_reason": "Chess onefile run artifacts remain internal-only unless separately benchmarked and externally validated.",
    }


def render_release_snapshot_md(report: Dict[str, Any]) -> str:
    bundle = dict(report.get("bundle", {}))
    lines = [
        "# Release Snapshot",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- mode: `{report.get('mode', '')}`",
        f"- profile: `{report.get('profile', '')}`",
        f"- feature_bundle: `{report.get('feature_bundle', 'default')}`",
        f"- execution_status: `{report.get('execution_status', 'unknown')}`",
        f"- evaluation_status: `{report.get('evaluation_status', 'unknown')}`",
        f"- rating_claim_status: `{report.get('rating_claim_status', 'unknown')}`",
        f"- release_surface_status: `{report.get('release_surface_status', 'unknown')}`",
        f"- external_release_grade: `{report.get('external_release_grade', False)}`",
        "",
        "## Artifact Surface",
        f"- required_count: `{report.get('required_count', 0)}`",
        f"- present_required_count: `{report.get('present_required_count', 0)}`",
        f"- core_reports_ready: `{report.get('core_reports_ready', False)}`",
        f"- checkpoint_ready: `{report.get('checkpoint_ready', False)}`",
        f"- bundle_zip_exists: `{bundle.get('zip_exists', False)}`",
        f"- bundle_sha_exists: `{bundle.get('sha256_exists', False)}`",
        "",
        f"- external_release_reason: {report.get('external_release_reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_evidence_pack_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    truth = _read_json_if_exists(layout.reports_dir / "artifact_truth_matrix.json")
    truth_entries = {entry.get("label", ""): entry for entry in truth.get("entries", [])}
    notes = dict(payload.get("notes", {}))
    core_labels = [
        "run_summary_json",
        "model_card",
        "eval_card",
        "feature_flag_report_json",
        "run_status_manifest",
        "postrun_analysis_manifest",
        "artifact_truth_matrix",
        "artifact_manifest",
        "run_log",
    ]
    core_items = [
        {
            "label": label,
            "exists": bool(truth_entries.get(label, {}).get("exists", False)),
            "path": truth_entries.get(label, {}).get("path", ""),
        }
        for label in core_labels
    ]
    internal_items = [
        {"label": "selfplay_report", "status": payload.get("selfplay_report", {}).get("status", "unknown"), "scope": "internal_only"},
        {"label": "tournament_report", "status": payload.get("tournament_report", {}).get("status", "unknown"), "scope": "internal_only"},
        {"label": "replay_buffer_report", "status": payload.get("replay_buffer_report", {}).get("status", "unknown"), "scope": "internal_only"},
        {"label": "stockfish", "status": payload.get("stockfish", {}).get("status", "unknown"), "scope": "internal_only"},
    ]
    missing_for_external_release: List[str] = []
    if not str(payload.get("best_checkpoint", "")).strip() and not str(payload.get("latest_checkpoint", "")).strip() and not notes.get("package_only", False):
        missing_for_external_release.append("trained checkpoint or explicit package-only provenance")
    if payload.get("stockfish", {}).get("status") != "completed":
        missing_for_external_release.append("completed stockfish benchmark evidence")
    missing_for_external_release.append("external benchmark reproduction")
    status = "partial_internal_only" if all(item["exists"] for item in core_items) else "incomplete"
    return {
        "schema": "chess_evidence_pack_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": status,
        "core_items": core_items,
        "internal_only_items": internal_items,
        "missing_for_external_release": missing_for_external_release,
    }


def render_evidence_pack_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Evidence Pack Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        "",
        "## Core Items",
    ]
    for item in report.get("core_items", []):
        lines.append(f"- `{item.get('label', '')}`: exists=`{item.get('exists', False)}` path=`{item.get('path', '')}`")
    lines.append("")
    lines.append("## Internal-Only Items")
    for item in report.get("internal_only_items", []):
        lines.append(f"- `{item.get('label', '')}`: status=`{item.get('status', 'unknown')}` scope=`{item.get('scope', 'internal_only')}`")
    lines.append("")
    lines.append("## Missing For External Release")
    for item in report.get("missing_for_external_release", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def build_final_truth_registry(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    truth = _read_json_if_exists(layout.reports_dir / "artifact_truth_matrix.json")
    required_count = int(truth.get("required_count", 0))
    present_required_count = int(truth.get("present_required_count", 0))
    claims = [
        {
            "label": "runtime_execution",
            "classification": "measured" if str(payload.get("execution_status", "unknown")) != "failed" else "not_met",
            "status": payload.get("execution_status", "unknown"),
        },
        {
            "label": "artifact_chain_presence",
            "classification": "measured" if required_count > 0 and present_required_count == required_count else "partial",
            "status": f"{present_required_count}/{required_count}",
        },
        {
            "label": "feature_surface_auditable",
            "classification": "measured" if (layout.reports_dir / "feature_flag_report.json").exists() else "not_met",
            "status": "present" if (layout.reports_dir / "feature_flag_report.json").exists() else "missing",
        },
        {
            "label": "selfplay_diagnostic",
            "classification": "internal_only" if payload.get("selfplay_report", {}).get("status") == "completed" else "not_run",
            "status": payload.get("selfplay_report", {}).get("status", "unknown"),
        },
        {
            "label": "tournament_diagnostic",
            "classification": "internal_only" if payload.get("tournament_report", {}).get("status") == "completed" else "not_run",
            "status": payload.get("tournament_report", {}).get("status", "unknown"),
        },
        {
            "label": "replay_buffer_diagnostic",
            "classification": "internal_only" if payload.get("replay_buffer_report", {}).get("status") == "completed" else "not_run",
            "status": payload.get("replay_buffer_report", {}).get("status", "unknown"),
        },
        {
            "label": "external_strength_claim",
            "classification": "not_eligible",
            "status": "not_proven_by_onefile_artifacts_alone",
        },
        {
            "label": "external_release_grade",
            "classification": "not_ready",
            "status": "separate_release_validation_required",
        },
    ]
    return {
        "schema": "chess_final_truth_registry_v1",
        "run_id": payload.get("run_id", ""),
        "claim_count": len(claims),
        "claims": claims,
    }


def render_final_truth_registry_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Final Truth Registry",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- claim_count: `{report.get('claim_count', 0)}`",
        "",
        "## Claims",
    ]
    for claim in report.get("claims", []):
        lines.append(
            f"- `{claim.get('label', '')}`: classification=`{claim.get('classification', 'unknown')}` "
            f"status=`{claim.get('status', 'unknown')}`"
        )
    return "\n".join(lines) + "\n"


def build_claim_registry(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    truth = _read_json_if_exists(layout.reports_dir / "artifact_truth_matrix.json")
    truth_entries = {entry.get("label", ""): entry for entry in truth.get("entries", [])}

    def evidence_for(*labels: str) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for label in labels:
            entry = truth_entries.get(label, {})
            items.append(
                {
                    "label": label,
                    "exists": bool(entry.get("exists", False)),
                    "path": entry.get("path", ""),
                }
            )
        return items

    claims = [
        {
            "label": "runtime_execution",
            "classification": "measured" if str(payload.get("execution_status", "unknown")) != "failed" else "not_met",
            "status": payload.get("execution_status", "unknown"),
            "evidence": evidence_for("run_summary_json", "run_log"),
        },
        {
            "label": "artifact_chain_presence",
            "classification": "measured" if truth.get("required_count", 0) and truth.get("present_required_count", 0) == truth.get("required_count", 0) else "partial",
            "status": f"{truth.get('present_required_count', 0)}/{truth.get('required_count', 0)}",
            "evidence": evidence_for("artifact_truth_matrix", "artifact_manifest"),
        },
        {
            "label": "feature_surface_auditable",
            "classification": "measured" if truth_entries.get("feature_flag_report_json", {}).get("exists", False) else "not_met",
            "status": "present" if truth_entries.get("feature_flag_report_json", {}).get("exists", False) else "missing",
            "evidence": evidence_for("feature_flag_report_json", "run_contract"),
        },
        {
            "label": "stockfish_proxy_benchmark",
            "classification": "internal_only" if payload.get("stockfish", {}).get("status") == "completed" else "not_run",
            "status": payload.get("stockfish", {}).get("status", "unknown"),
            "evidence": evidence_for("release_snapshot"),
        },
        {
            "label": "selfplay_diagnostic",
            "classification": "internal_only" if payload.get("selfplay_report", {}).get("status") == "completed" else "not_run",
            "status": payload.get("selfplay_report", {}).get("status", "unknown"),
            "evidence": evidence_for("selfplay_report"),
        },
        {
            "label": "tournament_diagnostic",
            "classification": "internal_only" if payload.get("tournament_report", {}).get("status") == "completed" else "not_run",
            "status": payload.get("tournament_report", {}).get("status", "unknown"),
            "evidence": evidence_for("tournament_report"),
        },
        {
            "label": "replay_buffer_diagnostic",
            "classification": "internal_only" if payload.get("replay_buffer_report", {}).get("status") == "completed" else "not_run",
            "status": payload.get("replay_buffer_report", {}).get("status", "unknown"),
            "evidence": evidence_for("replay_buffer_manifest"),
        },
        {
            "label": "external_strength_claim",
            "classification": "not_eligible",
            "status": "not_proven_by_onefile_artifacts_alone",
            "evidence": evidence_for("claim_registry", "known_limits", "release_gate_summary"),
        },
    ]
    return {
        "schema": "chess_claim_registry_v1",
        "run_id": payload.get("run_id", ""),
        "claim_count": len(claims),
        "claims": claims,
    }


def render_claim_registry_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Claim Registry",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- claim_count: `{report.get('claim_count', 0)}`",
        "",
        "## Claims",
    ]
    for claim in report.get("claims", []):
        lines.append(
            f"- `{claim.get('label', '')}`: classification=`{claim.get('classification', 'unknown')}` "
            f"status=`{claim.get('status', 'unknown')}`"
        )
    return "\n".join(lines) + "\n"


def build_known_limits(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    drift_report = _read_json_if_exists(layout.reports_dir / "truth_docs_drift_report.json")
    cfg = dict(payload.get("config", {}))
    limits: List[Dict[str, Any]] = [
        {
            "label": "external_strength_unproven",
            "severity": "high",
            "status": "active",
            "detail": "Chess onefile artifacts alone do not prove externally validated chess strength.",
        },
        {
            "label": "diagnostic_surfaces_internal_only",
            "severity": "medium",
            "status": "active",
            "detail": "Self-play, inference-mode tournament, replay buffer, and proxy score surfaces remain internal diagnostics unless separately validated.",
        },
        {
            "label": "release_surface_not_external_grade",
            "severity": "high",
            "status": "active",
            "detail": "Internal release-surface readiness is not the same thing as external release-grade verification.",
        },
        {
            "label": "external_reproduction_pending",
            "severity": "high",
            "status": "active",
            "detail": "External reproducibility confirmation remains pending even when internal artifacts are complete.",
        },
        {
            "label": "security_legal_pilot_pending",
            "severity": "high",
            "status": "active",
            "detail": "Security, legal, and pilot closures remain separate external work streams.",
        },
        {
            "label": "operator_handoff_dr_pending",
            "severity": "high",
            "status": "active",
            "detail": "Operator handbook, DR evidence, backup retention, and blind handoff closures remain separate operational work streams.",
        },
        {
            "label": "release_governance_pending",
            "severity": "medium",
            "status": "active",
            "detail": "Release notes, freeze manifest, changelog snapshot review, and maintenance policy still require formal release governance closure.",
        },
        {
            "label": "device_export_packaging_pending",
            "severity": "high",
            "status": "active",
            "detail": "Export truth, device validation, packaging closure, and installer validation remain separate release work streams.",
        },
        {
            "label": "benchmark_closure_pending",
            "severity": "medium",
            "status": "active",
            "detail": "Benchmark raw outputs, compare reports, summaries, and benchmark manifests still require formal benchmark closure.",
        },
        {
            "label": "training_accounting_pending",
            "severity": "medium",
            "status": "active",
            "detail": "Training report, token accounting, compute accounting, and cost reporting still require formal closure.",
        },
        {
            "label": "trained_artifact_truth_pending",
            "severity": "high",
            "status": "active",
            "detail": "Final weights truth, best/latest checkpoint truth, and trained artifact registry still require formal artifact closure.",
        },
        {
            "label": "management_closure_pending",
            "severity": "high",
            "status": "active",
            "detail": "Core-complete, research-separate, maintenance-only, and final closure decisions still require management closure.",
        },
    ]
    if cfg.get("mode") == "verify":
        limits.append(
            {
                "label": "verify_mode_runtime_only",
                "severity": "medium",
                "status": "active",
                "detail": "Verify mode proves runtime integrity and artifact packaging, not strength or benchmark quality.",
            }
        )
    if bool(payload.get("notes", {}).get("package_only", False)):
        limits.append(
            {
                "label": "package_only_repackaging",
                "severity": "medium",
                "status": "active",
                "detail": "Package mode repackages an existing run and does not create fresh training or evaluation evidence.",
            }
        )
    if payload.get("stockfish", {}).get("status") != "completed":
        limits.append(
            {
                "label": "stockfish_benchmark_missing_or_not_run",
                "severity": "medium",
                "status": "active",
                "detail": "Stockfish benchmark evidence is absent or incomplete for this exact run.",
            }
        )
    if drift_report.get("status") not in {"", "in_sync"}:
        limits.append(
            {
                "label": "truth_docs_drift_pending",
                "severity": "medium",
                "status": "active",
                "detail": "Canonical chess/project truth docs and generated reports are not fully aligned yet.",
            }
        )
    consistency_report = _read_json_if_exists(layout.reports_dir / "generated_truth_consistency_report.json")
    if consistency_report.get("status") not in {"", "consistent"}:
        limits.append(
            {
                "label": "generated_truth_consistency_pending",
                "severity": "medium",
                "status": "active",
                "detail": "Generated truth reports are present but not fully consistent with each other yet.",
            }
        )
    crosscheck_report = _read_json_if_exists(layout.reports_dir / "generated_truth_crosscheck_matrix.json")
    if crosscheck_report.get("status") not in {"", "consistent"}:
        limits.append(
            {
                "label": "generated_truth_crosscheck_pending",
                "severity": "medium",
                "status": "active",
                "detail": "Generated truth crosscheck matrix still sees mismatches between blockers, execution order, lane board, or summary layers.",
            }
        )
    return {
        "schema": "chess_known_limits_v1",
        "run_id": payload.get("run_id", ""),
        "limit_count": len(limits),
        "limits": limits,
    }


def render_known_limits_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Known Limits",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- limit_count: `{report.get('limit_count', 0)}`",
        "",
        "## Limits",
    ]
    for item in report.get("limits", []):
        lines.append(
            f"- `{item.get('label', '')}`: severity=`{item.get('severity', 'unknown')}` "
            f"status=`{item.get('status', 'unknown')}` detail={item.get('detail', '')}"
        )
    return "\n".join(lines) + "\n"


def build_support_matrix(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    truth = _read_json_if_exists(layout.reports_dir / "artifact_truth_matrix.json")
    truth_entries = {entry.get("label", ""): entry for entry in truth.get("entries", [])}
    cfg = dict(payload.get("config", {}))
    profiles = [
        {"label": "production_5080", "support_level": "baseline_supported", "active": cfg.get("profile") == "production_5080"},
        {"label": "strength_4060_24h_all_on_experimental", "support_level": "experimental", "active": cfg.get("profile") == "strength_4060_24h_all_on_experimental"},
        {"label": "strength_4060_24h_omni_max", "support_level": "experimental_high_risk", "active": cfg.get("profile") == "strength_4060_24h_omni_max"},
    ]
    modes = [
        {"label": "verify", "support_level": "supported", "active": cfg.get("mode") == "verify"},
        {"label": "arena", "support_level": "supported", "active": cfg.get("mode") == "arena"},
        {"label": "train", "support_level": "supported", "active": cfg.get("mode") == "train"},
        {"label": "resume", "support_level": "supported", "active": cfg.get("mode") == "resume"},
        {"label": "benchmark", "support_level": "supported", "active": cfg.get("mode") == "benchmark"},
        {"label": "package", "support_level": "supported", "active": cfg.get("mode") == "package"},
    ]
    artifact_surfaces = [
        {"label": "closure_manifests", "support_level": "supported", "present": truth_entries.get("run_status_manifest", {}).get("exists", False)},
        {"label": "release_evidence_registry", "support_level": "supported", "present": truth_entries.get("run_contract", {}).get("exists", False)},
        {"label": "diagnostic_selfplay", "support_level": "flagged_internal", "present": truth_entries.get("selfplay_report", {}).get("exists", False)},
        {"label": "diagnostic_tournament", "support_level": "flagged_internal", "present": truth_entries.get("tournament_report", {}).get("exists", False)},
        {"label": "diagnostic_replay_buffer", "support_level": "flagged_internal", "present": truth_entries.get("replay_buffer_manifest", {}).get("exists", False)},
    ]
    return {
        "schema": "chess_support_matrix_v1",
        "run_id": payload.get("run_id", ""),
        "profiles": profiles,
        "modes": modes,
        "artifact_surfaces": artifact_surfaces,
    }


def render_support_matrix_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Support Matrix",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        "",
        "## Profiles",
    ]
    for item in report.get("profiles", []):
        lines.append(
            f"- `{item.get('label', '')}`: support_level=`{item.get('support_level', 'unknown')}` active=`{item.get('active', False)}`"
        )
    lines.append("")
    lines.append("## Modes")
    for item in report.get("modes", []):
        lines.append(
            f"- `{item.get('label', '')}`: support_level=`{item.get('support_level', 'unknown')}` active=`{item.get('active', False)}`"
        )
    lines.append("")
    lines.append("## Artifact Surfaces")
    for item in report.get("artifact_surfaces", []):
        lines.append(
            f"- `{item.get('label', '')}`: support_level=`{item.get('support_level', 'unknown')}` present=`{item.get('present', False)}`"
        )
    return "\n".join(lines) + "\n"


def build_release_gate_summary(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    truth = _read_json_if_exists(layout.reports_dir / "artifact_truth_matrix.json")
    truth_docs_drift_report = _read_json_if_exists(layout.reports_dir / "truth_docs_drift_report.json")
    generated_truth_consistency_report = _read_json_if_exists(layout.reports_dir / "generated_truth_consistency_report.json")
    generated_truth_crosscheck_matrix = _read_json_if_exists(layout.reports_dir / "generated_truth_crosscheck_matrix.json")
    truth_entries = {entry.get("label", ""): entry for entry in truth.get("entries", [])}
    notes = dict(payload.get("notes", {}))
    bundle = dict(payload.get("bundle", {}))
    core_artifacts_present = truth.get("required_count", 0) > 0 and truth.get("present_required_count", 0) == truth.get("required_count", 0)
    checkpoint_or_provenance = bool(str(payload.get("best_checkpoint", "")).strip() or str(payload.get("latest_checkpoint", "")).strip() or notes.get("package_only", False))
    bundle_present = bool(truth_entries.get("bundle_zip", {}).get("exists", False) or str(bundle.get("zip_path", "")).strip())
    run_log_present = bool(truth_entries.get("run_log", {}).get("exists", False))
    stockfish_completed = payload.get("stockfish", {}).get("status") == "completed"
    internal_claim_boundary_preserved = payload.get("rating_claim_status", "") != RatingClaimStatus.TARGET_MET_EXTERNAL.value
    release_registry_present = bool(truth_entries.get("run_contract", {}).get("exists", False)) and bool(truth_entries.get("release_snapshot", {}).get("exists", False))
    handoff_surfaces_present = bool(truth_entries.get("handoff_pack_manifest", {}).get("exists", False)) and bool(truth_entries.get("operator_handoff_summary", {}).get("exists", False))
    external_closure_stubs_present = (
        bool(truth_entries.get("external_repro_stub", {}).get("exists", False))
        and bool(truth_entries.get("pilot_stub", {}).get("exists", False))
        and bool(truth_entries.get("security_stub", {}).get("exists", False))
        and bool(truth_entries.get("legal_stub", {}).get("exists", False))
    )
    operational_stub_surfaces_present = (
        bool(truth_entries.get("operator_handbook_stub", {}).get("exists", False))
        and bool(truth_entries.get("dr_evidence_stub", {}).get("exists", False))
        and bool(truth_entries.get("backup_retention_stub", {}).get("exists", False))
        and bool(truth_entries.get("blind_handoff_stub", {}).get("exists", False))
    )
    release_governance_surfaces_present = (
        bool(truth_entries.get("release_notes_stub", {}).get("exists", False))
        and bool(truth_entries.get("freeze_manifest_stub", {}).get("exists", False))
        and bool(truth_entries.get("changelog_snapshot", {}).get("exists", False))
        and bool(truth_entries.get("maintenance_policy_stub", {}).get("exists", False))
    )
    device_packaging_surfaces_present = (
        bool(truth_entries.get("export_truth_stub", {}).get("exists", False))
        and bool(truth_entries.get("device_validation_stub", {}).get("exists", False))
        and bool(truth_entries.get("packaging_closure_stub", {}).get("exists", False))
        and bool(truth_entries.get("installer_validation_stub", {}).get("exists", False))
    )
    benchmark_closure_surfaces_present = (
        bool(truth_entries.get("benchmark_raw_outputs_stub", {}).get("exists", False))
        and bool(truth_entries.get("benchmark_compare_report_stub", {}).get("exists", False))
        and bool(truth_entries.get("benchmark_summary_stub", {}).get("exists", False))
        and bool(truth_entries.get("benchmark_manifest_stub", {}).get("exists", False))
    )
    training_accounting_surfaces_present = (
        bool(truth_entries.get("training_report_stub", {}).get("exists", False))
        and bool(truth_entries.get("token_accounting_stub", {}).get("exists", False))
        and bool(truth_entries.get("compute_accounting_stub", {}).get("exists", False))
        and bool(truth_entries.get("cost_report_stub", {}).get("exists", False))
    )
    trained_artifact_surfaces_present = (
        bool(truth_entries.get("final_weights_truth_stub", {}).get("exists", False))
        and bool(truth_entries.get("best_checkpoint_truth_stub", {}).get("exists", False))
        and bool(truth_entries.get("latest_checkpoint_truth_stub", {}).get("exists", False))
        and bool(truth_entries.get("trained_artifact_registry_stub", {}).get("exists", False))
    )
    management_closure_surfaces_present = (
        bool(truth_entries.get("core_complete_decision_stub", {}).get("exists", False))
        and bool(truth_entries.get("research_continues_stub", {}).get("exists", False))
        and bool(truth_entries.get("product_maintenance_only_stub", {}).get("exists", False))
        and bool(truth_entries.get("closure_decision_record_stub", {}).get("exists", False))
    )
    master_summary_surfaces_present = (
        bool(truth_entries.get("master_closure_table", {}).get("exists", False))
        and bool(truth_entries.get("remaining_core_blockers", {}).get("exists", False))
        and bool(truth_entries.get("repo_side_completion_summary", {}).get("exists", False))
        and bool(truth_entries.get("readiness_snapshot", {}).get("exists", False))
    )
    aggregate_truth_surfaces_present = (
        bool(truth_entries.get("aggregated_master_table", {}).get("exists", False))
        and bool(truth_entries.get("real_remaining_core_work", {}).get("exists", False))
        and bool(truth_entries.get("repo_truth_inventory", {}).get("exists", False))
        and bool(truth_entries.get("closure_gap_summary", {}).get("exists", False))
    )
    project_truth_surfaces_present = (
        bool(truth_entries.get("project_master_truth_reference", {}).get("exists", False))
        and bool(truth_entries.get("project_remaining_real_blockers", {}).get("exists", False))
        and bool(truth_entries.get("truth_docs_index", {}).get("exists", False))
        and bool(truth_entries.get("truth_docs_drift_report", {}).get("exists", False))
    )
    project_actionability_surfaces_present = (
        bool(truth_entries.get("project_blocker_action_plan", {}).get("exists", False))
        and bool(truth_entries.get("project_blocker_dependency_graph", {}).get("exists", False))
        and bool(truth_entries.get("project_execution_sequence", {}).get("exists", False))
        and bool(truth_entries.get("project_lane_status_board", {}).get("exists", False))
        and bool(truth_entries.get("project_closure_phase_plan", {}).get("exists", False))
        and bool(truth_entries.get("project_phase_readiness_scoreboard", {}).get("exists", False))
        and bool(truth_entries.get("project_owner_accountability_matrix", {}).get("exists", False))
        and bool(truth_entries.get("project_owner_work_queue", {}).get("exists", False))
        and bool(truth_entries.get("project_critical_path_report", {}).get("exists", False))
        and bool(truth_entries.get("project_owner_next_actions_summary", {}).get("exists", False))
        and bool(truth_entries.get("project_ready_now_board", {}).get("exists", False))
        and bool(truth_entries.get("project_unlock_impact_report", {}).get("exists", False))
    )
    generated_truth_consistency_present = bool(truth_entries.get("generated_truth_consistency_report", {}).get("exists", False))
    generated_truth_crosscheck_present = bool(truth_entries.get("generated_truth_crosscheck_matrix", {}).get("exists", False))
    truth_docs_drift_clear = truth_docs_drift_report.get("status") == "in_sync"
    generated_truth_consistency_clear = generated_truth_consistency_report.get("status") == "consistent"
    generated_truth_crosscheck_clear = generated_truth_crosscheck_matrix.get("status") == "consistent"
    gates = [
        {"label": "core_artifacts_present", "passed": core_artifacts_present},
        {"label": "checkpoint_or_package_provenance", "passed": checkpoint_or_provenance},
        {"label": "bundle_present", "passed": bundle_present},
        {"label": "run_log_present", "passed": run_log_present},
        {"label": "stockfish_completed", "passed": stockfish_completed},
        {"label": "internal_claim_boundary_preserved", "passed": internal_claim_boundary_preserved},
        {"label": "release_registry_present", "passed": release_registry_present},
        {"label": "handoff_surfaces_present", "passed": handoff_surfaces_present},
        {"label": "external_closure_stubs_present", "passed": external_closure_stubs_present},
        {"label": "operational_stub_surfaces_present", "passed": operational_stub_surfaces_present},
        {"label": "release_governance_surfaces_present", "passed": release_governance_surfaces_present},
        {"label": "device_packaging_surfaces_present", "passed": device_packaging_surfaces_present},
        {"label": "benchmark_closure_surfaces_present", "passed": benchmark_closure_surfaces_present},
        {"label": "training_accounting_surfaces_present", "passed": training_accounting_surfaces_present},
        {"label": "trained_artifact_surfaces_present", "passed": trained_artifact_surfaces_present},
        {"label": "management_closure_surfaces_present", "passed": management_closure_surfaces_present},
        {"label": "master_summary_surfaces_present", "passed": master_summary_surfaces_present},
        {"label": "aggregate_truth_surfaces_present", "passed": aggregate_truth_surfaces_present},
        {"label": "project_truth_surfaces_present", "passed": project_truth_surfaces_present},
        {"label": "project_actionability_surfaces_present", "passed": project_actionability_surfaces_present},
        {"label": "generated_truth_consistency_present", "passed": generated_truth_consistency_present},
        {"label": "generated_truth_crosscheck_present", "passed": generated_truth_crosscheck_present},
        {"label": "truth_docs_drift_clear", "passed": truth_docs_drift_clear},
        {"label": "generated_truth_consistency_clear", "passed": generated_truth_consistency_clear},
        {"label": "generated_truth_crosscheck_clear", "passed": generated_truth_crosscheck_clear},
    ]
    overall_internal_ready = all(gate["passed"] for gate in gates if gate["label"] != "stockfish_completed")
    overall_external_ready = all(gate["passed"] for gate in gates) and payload.get("rating_claim_status") == RatingClaimStatus.TARGET_MET_EXTERNAL.value
    return {
        "schema": "chess_release_gate_summary_v1",
        "run_id": payload.get("run_id", ""),
        "gate_count": len(gates),
        "gates": gates,
        "overall_internal_ready": overall_internal_ready,
        "overall_external_ready": overall_external_ready,
    }


def render_release_gate_summary_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Release Gate Summary",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- gate_count: `{report.get('gate_count', 0)}`",
        f"- overall_internal_ready: `{report.get('overall_internal_ready', False)}`",
        f"- overall_external_ready: `{report.get('overall_external_ready', False)}`",
        "",
        "## Gates",
    ]
    for gate in report.get("gates", []):
        lines.append(f"- `{gate.get('label', '')}`: passed=`{gate.get('passed', False)}`")
    return "\n".join(lines) + "\n"


def build_rc_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    truth = _read_json_if_exists(layout.reports_dir / "artifact_truth_matrix.json")
    gate = _read_json_if_exists(layout.reports_dir / "release_gate_summary.json")
    return {
        "schema": "chess_rc_stub_v1",
        "run_id": payload.get("run_id", ""),
        "candidate_type": "internal_rc_stub",
        "status": "candidate_internal_only" if gate.get("overall_internal_ready", False) else "not_ready",
        "required_count": int(truth.get("required_count", 0)),
        "present_required_count": int(truth.get("present_required_count", 0)),
        "overall_internal_ready": bool(gate.get("overall_internal_ready", False)),
        "overall_external_ready": bool(gate.get("overall_external_ready", False)),
    }


def render_rc_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# RC Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- candidate_type: `{report.get('candidate_type', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- required_count: `{report.get('required_count', 0)}`",
        f"- present_required_count: `{report.get('present_required_count', 0)}`",
        f"- overall_internal_ready: `{report.get('overall_internal_ready', False)}`",
        f"- overall_external_ready: `{report.get('overall_external_ready', False)}`",
    ]
    return "\n".join(lines) + "\n"


def build_golden_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    gate = _read_json_if_exists(layout.reports_dir / "release_gate_summary.json")
    return {
        "schema": "chess_golden_stub_v1",
        "run_id": payload.get("run_id", ""),
        "candidate_type": "golden_stub",
        "status": "not_ready",
        "overall_external_ready": bool(gate.get("overall_external_ready", False)),
        "reason": "Golden release requires external verification and final release closure beyond internal onefile artifacts.",
    }


def render_golden_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Golden Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- candidate_type: `{report.get('candidate_type', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- overall_external_ready: `{report.get('overall_external_ready', False)}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_handoff_pack_manifest(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    truth = _read_json_if_exists(layout.reports_dir / "artifact_truth_matrix.json")
    truth_entries = {entry.get("label", ""): entry for entry in truth.get("entries", [])}
    labels = [
        "run_summary_json",
        "model_card",
        "eval_card",
        "run_contract",
        "release_snapshot",
        "evidence_pack_stub",
        "final_truth_registry",
        "claim_registry",
        "known_limits",
        "support_matrix",
        "release_gate_summary",
        "external_repro_stub",
        "pilot_stub",
        "security_stub",
        "legal_stub",
        "operator_handbook_stub",
        "dr_evidence_stub",
        "backup_retention_stub",
        "blind_handoff_stub",
        "release_notes_stub",
        "freeze_manifest_stub",
        "changelog_snapshot",
        "maintenance_policy_stub",
        "export_truth_stub",
        "device_validation_stub",
        "packaging_closure_stub",
        "installer_validation_stub",
        "benchmark_raw_outputs_stub",
        "benchmark_compare_report_stub",
        "benchmark_summary_stub",
        "benchmark_manifest_stub",
        "training_report_stub",
        "token_accounting_stub",
        "compute_accounting_stub",
        "cost_report_stub",
        "final_weights_truth_stub",
        "best_checkpoint_truth_stub",
        "latest_checkpoint_truth_stub",
        "trained_artifact_registry_stub",
        "core_complete_decision_stub",
        "research_continues_stub",
        "product_maintenance_only_stub",
        "closure_decision_record_stub",
        "master_closure_table",
        "remaining_core_blockers",
        "repo_side_completion_summary",
        "readiness_snapshot",
        "aggregated_master_table",
        "real_remaining_core_work",
        "repo_truth_inventory",
        "closure_gap_summary",
        "project_master_truth_reference",
        "project_remaining_real_blockers",
        "truth_docs_index",
        "truth_docs_drift_report",
        "project_blocker_action_plan",
        "project_blocker_dependency_graph",
        "project_execution_sequence",
        "project_lane_status_board",
        "project_closure_phase_plan",
        "project_phase_readiness_scoreboard",
        "project_owner_accountability_matrix",
        "project_owner_work_queue",
        "project_critical_path_report",
        "project_owner_next_actions_summary",
        "project_ready_now_board",
        "project_unlock_impact_report",
        "generated_truth_consistency_report",
        "generated_truth_crosscheck_matrix",
        "run_log",
    ]
    items = [
        {
            "label": label,
            "exists": bool(truth_entries.get(label, {}).get("exists", False)),
            "path": truth_entries.get(label, {}).get("path", ""),
        }
        for label in labels
    ]
    return {
        "schema": "chess_handoff_pack_manifest_v1",
        "run_id": payload.get("run_id", ""),
        "item_count": len(items),
        "items": items,
    }


def render_handoff_pack_manifest_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Handoff Pack Manifest",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- item_count: `{report.get('item_count', 0)}`",
        "",
        "## Items",
    ]
    for item in report.get("items", []):
        lines.append(f"- `{item.get('label', '')}`: exists=`{item.get('exists', False)}` path=`{item.get('path', '')}`")
    return "\n".join(lines) + "\n"


def build_operator_handoff_summary(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    handoff = _read_json_if_exists(layout.reports_dir / "handoff_pack_manifest.json")
    release_gate = _read_json_if_exists(layout.reports_dir / "release_gate_summary.json")
    items = handoff.get("items", [])
    existing_items = sum(1 for item in items if item.get("exists", False))
    total_items = len(items)
    external_stub_count = sum(
        1
        for item in items
        if item.get("label") in {"external_repro_stub", "pilot_stub", "security_stub", "legal_stub"} and item.get("exists", False)
    )
    operational_stub_count = sum(
        1
        for item in items
        if item.get("label") in {"operator_handbook_stub", "dr_evidence_stub", "backup_retention_stub", "blind_handoff_stub"} and item.get("exists", False)
    )
    release_governance_count = sum(
        1
        for item in items
        if item.get("label") in {"release_notes_stub", "freeze_manifest_stub", "changelog_snapshot", "maintenance_policy_stub"} and item.get("exists", False)
    )
    device_packaging_count = sum(
        1
        for item in items
        if item.get("label") in {"export_truth_stub", "device_validation_stub", "packaging_closure_stub", "installer_validation_stub"} and item.get("exists", False)
    )
    benchmark_closure_count = sum(
        1
        for item in items
        if item.get("label") in {"benchmark_raw_outputs_stub", "benchmark_compare_report_stub", "benchmark_summary_stub", "benchmark_manifest_stub"} and item.get("exists", False)
    )
    training_accounting_count = sum(
        1
        for item in items
        if item.get("label") in {"training_report_stub", "token_accounting_stub", "compute_accounting_stub", "cost_report_stub"} and item.get("exists", False)
    )
    trained_artifact_count = sum(
        1
        for item in items
        if item.get("label") in {"final_weights_truth_stub", "best_checkpoint_truth_stub", "latest_checkpoint_truth_stub", "trained_artifact_registry_stub"} and item.get("exists", False)
    )
    management_closure_count = sum(
        1
        for item in items
        if item.get("label") in {"core_complete_decision_stub", "research_continues_stub", "product_maintenance_only_stub", "closure_decision_record_stub"} and item.get("exists", False)
    )
    master_summary_count = sum(
        1
        for item in items
        if item.get("label") in {"master_closure_table", "remaining_core_blockers", "repo_side_completion_summary", "readiness_snapshot"} and item.get("exists", False)
    )
    aggregate_truth_count = sum(
        1
        for item in items
        if item.get("label") in {"aggregated_master_table", "real_remaining_core_work", "repo_truth_inventory", "closure_gap_summary"} and item.get("exists", False)
    )
    truth_docs_count = sum(
        1
        for item in items
        if item.get("label") in {"project_master_truth_reference", "project_remaining_real_blockers", "truth_docs_index", "truth_docs_drift_report"} and item.get("exists", False)
    )
    project_actionability_count = sum(
        1
        for item in items
        if item.get("label") in {
            "project_blocker_action_plan",
            "project_blocker_dependency_graph",
            "project_execution_sequence",
            "project_lane_status_board",
            "project_closure_phase_plan",
            "project_phase_readiness_scoreboard",
            "project_owner_accountability_matrix",
            "project_owner_work_queue",
            "project_critical_path_report",
            "project_owner_next_actions_summary",
            "project_ready_now_board",
            "project_unlock_impact_report",
        } and item.get("exists", False)
    )
    generated_truth_count = sum(
        1
        for item in items
        if item.get("label") in {"generated_truth_consistency_report", "generated_truth_crosscheck_matrix"} and item.get("exists", False)
    )
    return {
        "schema": "chess_operator_handoff_summary_v1",
        "run_id": payload.get("run_id", ""),
        "handoff_surface_status": "internal_ready" if total_items > 0 and existing_items == total_items else "incomplete",
        "existing_items": existing_items,
        "total_items": total_items,
        "external_stub_count": external_stub_count,
        "operational_stub_count": operational_stub_count,
        "release_governance_count": release_governance_count,
        "device_packaging_count": device_packaging_count,
        "benchmark_closure_count": benchmark_closure_count,
        "training_accounting_count": training_accounting_count,
        "trained_artifact_count": trained_artifact_count,
        "management_closure_count": management_closure_count,
        "master_summary_count": master_summary_count,
        "aggregate_truth_count": aggregate_truth_count,
        "truth_docs_count": truth_docs_count,
        "project_actionability_count": project_actionability_count,
        "generated_truth_count": generated_truth_count,
        "overall_internal_ready": bool(release_gate.get("overall_internal_ready", False)),
        "overall_external_ready": bool(release_gate.get("overall_external_ready", False)),
        "operator_note": "Operator handoff can be internally complete while external release readiness remains false.",
    }


def render_operator_handoff_summary_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Operator Handoff Summary",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- handoff_surface_status: `{report.get('handoff_surface_status', 'unknown')}`",
        f"- existing_items: `{report.get('existing_items', 0)}`",
        f"- total_items: `{report.get('total_items', 0)}`",
        f"- external_stub_count: `{report.get('external_stub_count', 0)}`",
        f"- operational_stub_count: `{report.get('operational_stub_count', 0)}`",
        f"- release_governance_count: `{report.get('release_governance_count', 0)}`",
        f"- device_packaging_count: `{report.get('device_packaging_count', 0)}`",
        f"- benchmark_closure_count: `{report.get('benchmark_closure_count', 0)}`",
        f"- training_accounting_count: `{report.get('training_accounting_count', 0)}`",
        f"- trained_artifact_count: `{report.get('trained_artifact_count', 0)}`",
        f"- management_closure_count: `{report.get('management_closure_count', 0)}`",
        f"- master_summary_count: `{report.get('master_summary_count', 0)}`",
        f"- aggregate_truth_count: `{report.get('aggregate_truth_count', 0)}`",
        f"- truth_docs_count: `{report.get('truth_docs_count', 0)}`",
        f"- project_actionability_count: `{report.get('project_actionability_count', 0)}`",
        f"- generated_truth_count: `{report.get('generated_truth_count', 0)}`",
        f"- overall_internal_ready: `{report.get('overall_internal_ready', False)}`",
        f"- overall_external_ready: `{report.get('overall_external_ready', False)}`",
        f"- operator_note: {report.get('operator_note', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_external_repro_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    gate = _read_json_if_exists(layout.reports_dir / "release_gate_summary.json")
    return {
        "schema": "chess_external_repro_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_external_reproduction",
        "overall_internal_ready": bool(gate.get("overall_internal_ready", False)),
        "reason": "External reproducibility requires third-party rerun or independent confirmation outside this onefile artifact chain.",
    }


def render_external_repro_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# External Repro Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- overall_internal_ready: `{report.get('overall_internal_ready', False)}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_pilot_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    return {
        "schema": "chess_pilot_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_pilot_validation",
        "reason": "Pilot validation requires real operator or user deployment outside internal artifact generation.",
    }


def render_pilot_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Pilot Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_security_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    return {
        "schema": "chess_security_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_security_review",
        "reason": "Security closure requires external review, threat assessment, and deployment-specific checks beyond onefile artifacts.",
    }


def render_security_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Security Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_legal_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    return {
        "schema": "chess_legal_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_legal_review",
        "reason": "Legal closure requires external licensing, data, and deployment review beyond this internal run record.",
    }


def render_legal_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Legal Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_operator_handbook_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    return {
        "schema": "chess_operator_handbook_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_operator_handbook_validation",
        "reason": "Operator handbook closure requires rehearsed operator-facing documentation beyond local artifact generation.",
    }


def render_operator_handbook_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Operator Handbook Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_dr_evidence_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    return {
        "schema": "chess_dr_evidence_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_dr_validation",
        "reason": "Disaster-recovery closure requires restore rehearsal and cross-machine evidence beyond onefile-local outputs.",
    }


def render_dr_evidence_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# DR Evidence Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_backup_retention_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    return {
        "schema": "chess_backup_retention_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_retention_policy_finalization",
        "reason": "Backup retention closure requires explicit retention windows and operator policy outside this onefile run artifact chain.",
    }


def render_backup_retention_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Backup Retention Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_blind_handoff_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    return {
        "schema": "chess_blind_handoff_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_blind_handoff_rehearsal",
        "reason": "Blind handoff closure requires a fresh operator rehearsal beyond internally generated onefile evidence.",
    }


def render_blind_handoff_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Blind Handoff Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_release_notes_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    return {
        "schema": "chess_release_notes_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_release_note_curation",
        "reason": "Final release notes require curated human review beyond automatically generated onefile evidence.",
    }


def render_release_notes_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Release Notes Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_freeze_manifest_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    return {
        "schema": "chess_freeze_manifest_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_freeze_signoff",
        "reason": "Freeze manifest closure requires final release governance signoff beyond local onefile artifact generation.",
    }


def render_freeze_manifest_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Freeze Manifest Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_changelog_snapshot(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    truth = _read_json_if_exists(layout.reports_dir / "artifact_truth_matrix.json")
    release_gate = _read_json_if_exists(layout.reports_dir / "release_gate_summary.json")
    truth_entries = {entry.get("label", ""): entry for entry in truth.get("entries", [])}
    included_labels = [
        label
        for label in (
            "run_contract",
            "release_snapshot",
            "claim_registry",
            "known_limits",
            "release_gate_summary",
            "handoff_pack_manifest",
            "operator_handoff_summary",
        )
        if truth_entries.get(label, {}).get("exists", False)
    ]
    return {
        "schema": "chess_changelog_snapshot_v1",
        "run_id": payload.get("run_id", ""),
        "execution_status": payload.get("execution_status", "unknown"),
        "evaluation_status": payload.get("evaluation_status", "unknown"),
        "included_label_count": len(included_labels),
        "included_labels": included_labels,
        "overall_internal_ready": bool(release_gate.get("overall_internal_ready", False)),
        "overall_external_ready": bool(release_gate.get("overall_external_ready", False)),
    }


def render_changelog_snapshot_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Changelog Snapshot",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- execution_status: `{report.get('execution_status', 'unknown')}`",
        f"- evaluation_status: `{report.get('evaluation_status', 'unknown')}`",
        f"- included_label_count: `{report.get('included_label_count', 0)}`",
        f"- overall_internal_ready: `{report.get('overall_internal_ready', False)}`",
        f"- overall_external_ready: `{report.get('overall_external_ready', False)}`",
        "",
        "## Included Labels",
    ]
    for label in report.get("included_labels", []):
        lines.append(f"- `{label}`")
    return "\n".join(lines) + "\n"


def build_maintenance_policy_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    return {
        "schema": "chess_maintenance_policy_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_maintenance_policy_finalization",
        "reason": "Maintenance/support policy requires explicit governance and release support decisions beyond onefile-local artifact generation.",
    }


def render_maintenance_policy_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Maintenance Policy Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_export_truth_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    gate = _read_json_if_exists(layout.reports_dir / "release_gate_summary.json")
    return {
        "schema": "chess_export_truth_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_export_truth_validation",
        "overall_internal_ready": bool(gate.get("overall_internal_ready", False)),
        "reason": "Export truth closure requires parity and packaged export validation beyond internal onefile artifact generation.",
    }


def render_export_truth_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Export Truth Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- overall_internal_ready: `{report.get('overall_internal_ready', False)}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_device_validation_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    return {
        "schema": "chess_device_validation_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_device_validation",
        "reason": "Device validation closure requires latency, RAM, thermal, and runtime checks outside internal onefile artifact generation.",
    }


def render_device_validation_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Device Validation Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_packaging_closure_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    return {
        "schema": "chess_packaging_closure_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_packaging_closure",
        "reason": "Packaging closure requires finalized packaging validation beyond locally generated onefile artifacts.",
    }


def render_packaging_closure_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Packaging Closure Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_installer_validation_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    return {
        "schema": "chess_installer_validation_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_installer_validation",
        "reason": "Installer validation closure requires clean install and restore checks beyond internal onefile artifact generation.",
    }


def render_installer_validation_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Installer Validation Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_benchmark_raw_outputs_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    return {
        "schema": "chess_benchmark_raw_outputs_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_benchmark_raw_output_capture",
        "reason": "Benchmark closure requires preserved raw outputs beyond summary-level internal artifacts.",
    }


def render_benchmark_raw_outputs_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Benchmark Raw Outputs Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_benchmark_compare_report_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    return {
        "schema": "chess_benchmark_compare_report_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_benchmark_compare_report",
        "reason": "Benchmark closure requires before/after or baseline compare reporting beyond internal run-local evidence.",
    }


def render_benchmark_compare_report_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Benchmark Compare Report Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_benchmark_summary_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    return {
        "schema": "chess_benchmark_summary_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_benchmark_summary_closure",
        "reason": "Benchmark closure requires a curated benchmark summary beyond isolated internal artifacts.",
    }


def render_benchmark_summary_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Benchmark Summary Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_benchmark_manifest_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    return {
        "schema": "chess_benchmark_manifest_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_benchmark_manifest_lock",
        "reason": "Benchmark closure requires a locked benchmark manifest beyond ad hoc internal run-local reporting.",
    }


def render_benchmark_manifest_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Benchmark Manifest Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_training_report_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    return {
        "schema": "chess_training_report_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_training_report_closure",
        "reason": "Training closure requires a curated training report beyond local onefile artifact generation.",
    }


def render_training_report_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Training Report Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_token_accounting_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    return {
        "schema": "chess_token_accounting_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_token_accounting",
        "reason": "Training closure requires explicit token accounting beyond local onefile artifact generation.",
    }


def render_token_accounting_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Token Accounting Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_compute_accounting_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    return {
        "schema": "chess_compute_accounting_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_compute_accounting",
        "reason": "Training closure requires explicit compute accounting beyond local onefile artifact generation.",
    }


def render_compute_accounting_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Compute Accounting Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_cost_report_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    return {
        "schema": "chess_cost_report_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_cost_report",
        "reason": "Training closure requires explicit cost reporting beyond local onefile artifact generation.",
    }


def render_cost_report_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Cost Report Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_final_weights_truth_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    bundle = dict(payload.get("bundle", {}))
    final_zip = str(bundle.get("zip_path", "")).strip()
    return {
        "schema": "chess_final_weights_truth_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_final_weights_truth",
        "bundle_zip_present": bool(final_zip),
        "reason": "Final weights truth requires explicit trained-weight provenance and validation beyond internal onefile artifact generation.",
    }


def render_final_weights_truth_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Final Weights Truth Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- bundle_zip_present: `{report.get('bundle_zip_present', False)}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_best_checkpoint_truth_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    best_checkpoint = str(payload.get("best_checkpoint", "")).strip()
    return {
        "schema": "chess_best_checkpoint_truth_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_best_checkpoint_truth",
        "best_checkpoint_present": bool(best_checkpoint),
        "reason": "Best-checkpoint truth requires measured checkpoint selection and validation beyond local artifact generation.",
    }


def render_best_checkpoint_truth_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Best Checkpoint Truth Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- best_checkpoint_present: `{report.get('best_checkpoint_present', False)}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_latest_checkpoint_truth_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    latest_checkpoint = str(payload.get("latest_checkpoint", "")).strip()
    return {
        "schema": "chess_latest_checkpoint_truth_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_latest_checkpoint_truth",
        "latest_checkpoint_present": bool(latest_checkpoint),
        "reason": "Latest-checkpoint truth requires explicit artifact validation beyond local onefile artifact generation.",
    }


def render_latest_checkpoint_truth_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Latest Checkpoint Truth Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- latest_checkpoint_present: `{report.get('latest_checkpoint_present', False)}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_trained_artifact_registry_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    truth = _read_json_if_exists(layout.reports_dir / "artifact_truth_matrix.json")
    truth_entries = {entry.get("label", ""): entry for entry in truth.get("entries", [])}
    tracked_labels = [
        label
        for label in ("best_checkpoint", "latest_checkpoint", "bundle_zip", "bundle_sha")
        if truth_entries.get(label, {}).get("exists", False)
    ]
    return {
        "schema": "chess_trained_artifact_registry_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_trained_artifact_registry_lock",
        "tracked_label_count": len(tracked_labels),
        "tracked_labels": tracked_labels,
        "reason": "Trained artifact registry closure requires a locked trained-artifact registry beyond local onefile artifact generation.",
    }


def render_trained_artifact_registry_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Trained Artifact Registry Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- tracked_label_count: `{report.get('tracked_label_count', 0)}`",
        f"- reason: {report.get('reason', '')}",
        "",
        "## Tracked Labels",
    ]
    for label in report.get("tracked_labels", []):
        lines.append(f"- `{label}`")
    return "\n".join(lines) + "\n"


def build_core_complete_decision_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    return {
        "schema": "chess_core_complete_decision_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_core_complete_decision",
        "reason": "Core-complete status still requires management decision beyond local onefile artifact generation.",
    }


def render_core_complete_decision_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Core Complete Decision Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_research_continues_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    return {
        "schema": "chess_research_continues_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_research_separation_decision",
        "reason": "Research-continuation separation still requires management decision beyond local onefile artifact generation.",
    }


def render_research_continues_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Research Continues Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_product_maintenance_only_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    return {
        "schema": "chess_product_maintenance_only_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_maintenance_only_decision",
        "reason": "Maintenance-only posture still requires management decision beyond local onefile artifact generation.",
    }


def render_product_maintenance_only_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Product Maintenance Only Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- reason: {report.get('reason', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_closure_decision_record_stub(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    truth = _read_json_if_exists(layout.reports_dir / "artifact_truth_matrix.json")
    truth_entries = {entry.get("label", ""): entry for entry in truth.get("entries", [])}
    tracked_labels = [
        label
        for label in (
            "release_gate_summary",
            "handoff_pack_manifest",
            "operator_handoff_summary",
            "trained_artifact_registry_stub",
        )
        if truth_entries.get(label, {}).get("exists", False)
    ]
    return {
        "schema": "chess_closure_decision_record_stub_v1",
        "run_id": payload.get("run_id", ""),
        "status": "pending_management_closure_record",
        "tracked_label_count": len(tracked_labels),
        "tracked_labels": tracked_labels,
        "reason": "Final closure record still requires management signoff beyond local onefile artifact generation.",
    }


def render_closure_decision_record_stub_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Closure Decision Record Stub",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- tracked_label_count: `{report.get('tracked_label_count', 0)}`",
        f"- reason: {report.get('reason', '')}",
        "",
        "## Tracked Labels",
    ]
    for label in report.get("tracked_labels", []):
        lines.append(f"- `{label}`")
    return "\n".join(lines) + "\n"


def build_master_closure_table(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    truth = _read_json_if_exists(layout.reports_dir / "artifact_truth_matrix.json")
    gate = _read_json_if_exists(layout.reports_dir / "release_gate_summary.json")
    known_limits = _read_json_if_exists(layout.reports_dir / "known_limits.json")
    truth_entries = {entry.get("label", ""): entry for entry in truth.get("entries", [])}

    groups = {
        "release_registry": ["run_contract", "release_snapshot", "evidence_pack_stub", "final_truth_registry"],
        "external_closure": ["external_repro_stub", "pilot_stub", "security_stub", "legal_stub"],
        "operational_closure": ["operator_handbook_stub", "dr_evidence_stub", "backup_retention_stub", "blind_handoff_stub"],
        "release_governance": ["release_notes_stub", "freeze_manifest_stub", "changelog_snapshot", "maintenance_policy_stub"],
        "device_packaging": ["export_truth_stub", "device_validation_stub", "packaging_closure_stub", "installer_validation_stub"],
        "benchmark_closure": ["benchmark_raw_outputs_stub", "benchmark_compare_report_stub", "benchmark_summary_stub", "benchmark_manifest_stub"],
        "training_accounting": ["training_report_stub", "token_accounting_stub", "compute_accounting_stub", "cost_report_stub"],
        "trained_artifact_truth": ["final_weights_truth_stub", "best_checkpoint_truth_stub", "latest_checkpoint_truth_stub", "trained_artifact_registry_stub"],
        "management_closure": ["core_complete_decision_stub", "research_continues_stub", "product_maintenance_only_stub", "closure_decision_record_stub"],
        "truth_docs_alignment": ["project_master_truth_reference", "project_remaining_real_blockers", "truth_docs_index", "truth_docs_drift_report"],
        "project_actionability": [
            "project_blocker_action_plan",
            "project_blocker_dependency_graph",
            "project_execution_sequence",
            "project_lane_status_board",
            "project_closure_phase_plan",
            "project_phase_readiness_scoreboard",
            "project_owner_accountability_matrix",
            "project_owner_work_queue",
            "project_critical_path_report",
            "project_owner_next_actions_summary",
            "project_ready_now_board",
            "project_unlock_impact_report",
        ],
        "generated_truth_consistency": ["generated_truth_consistency_report", "generated_truth_crosscheck_matrix"],
    }
    rows: List[Dict[str, Any]] = []
    for label, members in groups.items():
        present = sum(1 for member in members if truth_entries.get(member, {}).get("exists", False))
        rows.append(
            {
                "label": label,
                "present": present,
                "total": len(members),
                "complete": present == len(members),
            }
        )
    return {
        "schema": "chess_master_closure_table_v1",
        "run_id": payload.get("run_id", ""),
        "row_count": len(rows),
        "rows": rows,
        "overall_internal_ready": bool(gate.get("overall_internal_ready", False)),
        "overall_external_ready": bool(gate.get("overall_external_ready", False)),
        "active_limit_count": len(known_limits.get("limits", [])),
    }


def render_master_closure_table_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Master Closure Table",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- row_count: `{report.get('row_count', 0)}`",
        f"- overall_internal_ready: `{report.get('overall_internal_ready', False)}`",
        f"- overall_external_ready: `{report.get('overall_external_ready', False)}`",
        f"- active_limit_count: `{report.get('active_limit_count', 0)}`",
        "",
        "| Label | Present | Total | Complete |",
        "|---|---:|---:|---|",
    ]
    for row in report.get("rows", []):
        lines.append(f"| `{row.get('label', '')}` | `{row.get('present', 0)}` | `{row.get('total', 0)}` | `{row.get('complete', False)}` |")
    return "\n".join(lines) + "\n"


def build_remaining_core_blockers(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    known_limits = _read_json_if_exists(layout.reports_dir / "known_limits.json")
    blockers = [
        {
            "label": item.get("label", ""),
            "severity": item.get("severity", "unknown"),
            "detail": item.get("detail", ""),
        }
        for item in known_limits.get("limits", [])
        if item.get("status") == "active"
    ]
    return {
        "schema": "chess_remaining_core_blockers_v1",
        "run_id": payload.get("run_id", ""),
        "blocker_count": len(blockers),
        "blockers": blockers,
    }


def render_remaining_core_blockers_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Remaining Core Blockers",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- blocker_count: `{report.get('blocker_count', 0)}`",
        "",
        "## Blockers",
    ]
    for blocker in report.get("blockers", []):
        lines.append(
            f"- `{blocker.get('label', '')}`: severity=`{blocker.get('severity', 'unknown')}` detail={blocker.get('detail', '')}"
        )
    return "\n".join(lines) + "\n"


def build_repo_side_completion_summary(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    truth = _read_json_if_exists(layout.reports_dir / "artifact_truth_matrix.json")
    gate = _read_json_if_exists(layout.reports_dir / "release_gate_summary.json")
    gates = gate.get("gates", [])
    passed = sum(1 for item in gates if item.get("passed", False))
    total = len(gates)
    required = int(truth.get("required_count", 0))
    present = int(truth.get("present_required_count", 0))
    return {
        "schema": "chess_repo_side_completion_summary_v1",
        "run_id": payload.get("run_id", ""),
        "required_count": required,
        "present_required_count": present,
        "missing_required_count": max(0, required - present),
        "gate_pass_count": passed,
        "gate_total_count": total,
        "repo_side_complete": required > 0 and present == required,
    }


def render_repo_side_completion_summary_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Repo Side Completion Summary",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- required_count: `{report.get('required_count', 0)}`",
        f"- present_required_count: `{report.get('present_required_count', 0)}`",
        f"- missing_required_count: `{report.get('missing_required_count', 0)}`",
        f"- gate_pass_count: `{report.get('gate_pass_count', 0)}`",
        f"- gate_total_count: `{report.get('gate_total_count', 0)}`",
        f"- repo_side_complete: `{report.get('repo_side_complete', False)}`",
    ]
    return "\n".join(lines) + "\n"


def build_readiness_snapshot(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    release_snapshot = _read_json_if_exists(layout.reports_dir / "release_snapshot.json")
    rc_stub = _read_json_if_exists(layout.reports_dir / "rc_stub.json")
    golden_stub = _read_json_if_exists(layout.reports_dir / "golden_stub.json")
    gate = _read_json_if_exists(layout.reports_dir / "release_gate_summary.json")
    return {
        "schema": "chess_readiness_snapshot_v1",
        "run_id": payload.get("run_id", ""),
        "execution_status": payload.get("execution_status", "unknown"),
        "evaluation_status": payload.get("evaluation_status", "unknown"),
        "release_surface_status": release_snapshot.get("release_surface_status", "unknown"),
        "rc_status": rc_stub.get("status", "unknown"),
        "golden_status": golden_stub.get("status", "unknown"),
        "overall_internal_ready": bool(gate.get("overall_internal_ready", False)),
        "overall_external_ready": bool(gate.get("overall_external_ready", False)),
    }


def render_readiness_snapshot_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Readiness Snapshot",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- execution_status: `{report.get('execution_status', 'unknown')}`",
        f"- evaluation_status: `{report.get('evaluation_status', 'unknown')}`",
        f"- release_surface_status: `{report.get('release_surface_status', 'unknown')}`",
        f"- rc_status: `{report.get('rc_status', 'unknown')}`",
        f"- golden_status: `{report.get('golden_status', 'unknown')}`",
        f"- overall_internal_ready: `{report.get('overall_internal_ready', False)}`",
        f"- overall_external_ready: `{report.get('overall_external_ready', False)}`",
    ]
    return "\n".join(lines) + "\n"


def build_aggregated_master_table(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    master = _read_json_if_exists(layout.reports_dir / "master_closure_table.json")
    blockers = _read_json_if_exists(layout.reports_dir / "remaining_core_blockers.json")
    blocker_labels = {item.get("label", "") for item in blockers.get("blockers", [])}
    group_to_blockers = {
        "release_registry": {"release_surface_not_external_grade"},
        "external_closure": {"external_reproduction_pending", "security_legal_pilot_pending"},
        "operational_closure": {"operator_handoff_dr_pending"},
        "release_governance": {"release_governance_pending"},
        "device_packaging": {"device_export_packaging_pending"},
        "benchmark_closure": {"benchmark_closure_pending"},
        "training_accounting": {"training_accounting_pending"},
        "trained_artifact_truth": {"trained_artifact_truth_pending"},
        "management_closure": {"management_closure_pending"},
        "truth_docs_alignment": {"truth_docs_drift_pending"},
        "project_actionability": set(),
        "generated_truth_consistency": {"generated_truth_consistency_pending", "generated_truth_crosscheck_pending"},
    }
    rows: List[Dict[str, Any]] = []
    for row in master.get("rows", []):
        label = row.get("label", "")
        linked = sorted(group_to_blockers.get(label, set()) & blocker_labels)
        rows.append(
            {
                "label": label,
                "repo_side_complete": bool(row.get("complete", False)),
                "present": int(row.get("present", 0)),
                "total": int(row.get("total", 0)),
                "real_closure_blocked": bool(linked),
                "linked_blockers": linked,
            }
        )
    return {
        "schema": "chess_aggregated_master_table_v1",
        "run_id": payload.get("run_id", ""),
        "row_count": len(rows),
        "rows": rows,
    }


def render_aggregated_master_table_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Aggregated Master Table",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- row_count: `{report.get('row_count', 0)}`",
        "",
        "| Label | Repo Side Complete | Present | Total | Real Closure Blocked | Linked Blockers |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in report.get("rows", []):
        blockers = ", ".join(f"`{label}`" for label in row.get("linked_blockers", []))
        lines.append(
            f"| `{row.get('label', '')}` | `{row.get('repo_side_complete', False)}` | `{row.get('present', 0)}` | "
            f"`{row.get('total', 0)}` | `{row.get('real_closure_blocked', False)}` | {blockers} |"
        )
    return "\n".join(lines) + "\n"


def build_real_remaining_core_work(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    blockers = _read_json_if_exists(layout.reports_dir / "remaining_core_blockers.json")
    items = [
        {
            "label": blocker.get("label", ""),
            "severity": blocker.get("severity", "unknown"),
            "detail": blocker.get("detail", ""),
        }
        for blocker in blockers.get("blockers", [])
    ]
    return {
        "schema": "chess_real_remaining_core_work_v1",
        "run_id": payload.get("run_id", ""),
        "item_count": len(items),
        "items": items,
    }


def render_real_remaining_core_work_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Real Remaining Core Work",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- item_count: `{report.get('item_count', 0)}`",
        "",
        "## Remaining Work",
    ]
    for item in report.get("items", []):
        lines.append(
            f"- `{item.get('label', '')}`: severity=`{item.get('severity', 'unknown')}` detail={item.get('detail', '')}"
        )
    return "\n".join(lines) + "\n"


def build_repo_truth_inventory(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    truth = _read_json_if_exists(layout.reports_dir / "artifact_truth_matrix.json")
    entries = truth.get("entries", [])
    by_kind: DefaultDict[str, Dict[str, int]] = defaultdict(lambda: {"present": 0, "missing": 0})
    missing_required_labels: List[str] = []
    for entry in entries:
        kind = str(entry.get("kind", "unknown"))
        if entry.get("exists", False):
            by_kind[kind]["present"] += 1
        else:
            by_kind[kind]["missing"] += 1
            if entry.get("required", False):
                missing_required_labels.append(str(entry.get("label", "")))
    kind_rows = [
        {"kind": kind, "present": counts["present"], "missing": counts["missing"]}
        for kind, counts in sorted(by_kind.items())
    ]
    return {
        "schema": "chess_repo_truth_inventory_v1",
        "run_id": payload.get("run_id", ""),
        "entry_count": len(entries),
        "kind_rows": kind_rows,
        "missing_required_labels": missing_required_labels,
    }


def render_repo_truth_inventory_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Repo Truth Inventory",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- entry_count: `{report.get('entry_count', 0)}`",
        "",
        "| Kind | Present | Missing |",
        "|---|---:|---:|",
    ]
    for row in report.get("kind_rows", []):
        lines.append(f"| `{row.get('kind', '')}` | `{row.get('present', 0)}` | `{row.get('missing', 0)}` |")
    lines.append("")
    lines.append("## Missing Required Labels")
    for label in report.get("missing_required_labels", []):
        lines.append(f"- `{label}`")
    return "\n".join(lines) + "\n"


def build_closure_gap_summary(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    repo_summary = _read_json_if_exists(layout.reports_dir / "repo_side_completion_summary.json")
    blockers = _read_json_if_exists(layout.reports_dir / "remaining_core_blockers.json")
    readiness = _read_json_if_exists(layout.reports_dir / "readiness_snapshot.json")
    truth_docs_drift_report = _read_json_if_exists(layout.reports_dir / "truth_docs_drift_report.json")
    project_lane_status_board = _read_json_if_exists(layout.reports_dir / "project_lane_status_board.json")
    project_closure_phase_plan = _read_json_if_exists(layout.reports_dir / "project_closure_phase_plan.json")
    project_phase_readiness_scoreboard = _read_json_if_exists(layout.reports_dir / "project_phase_readiness_scoreboard.json")
    project_owner_accountability_matrix = _read_json_if_exists(layout.reports_dir / "project_owner_accountability_matrix.json")
    project_owner_work_queue = _read_json_if_exists(layout.reports_dir / "project_owner_work_queue.json")
    project_critical_path_report = _read_json_if_exists(layout.reports_dir / "project_critical_path_report.json")
    project_owner_next_actions_summary = _read_json_if_exists(layout.reports_dir / "project_owner_next_actions_summary.json")
    project_ready_now_board = _read_json_if_exists(layout.reports_dir / "project_ready_now_board.json")
    project_unlock_impact_report = _read_json_if_exists(layout.reports_dir / "project_unlock_impact_report.json")
    return {
        "schema": "chess_closure_gap_summary_v1",
        "run_id": payload.get("run_id", ""),
        "repo_side_complete": bool(repo_summary.get("repo_side_complete", False)),
        "missing_required_count": int(repo_summary.get("missing_required_count", 0)),
        "blocker_count": int(blockers.get("blocker_count", 0)),
        "overall_internal_ready": bool(readiness.get("overall_internal_ready", False)),
        "overall_external_ready": bool(readiness.get("overall_external_ready", False)),
        "truth_docs_status": truth_docs_drift_report.get("status", "unknown"),
        "project_actionability_status": project_lane_status_board.get("status", "unknown"),
        "project_phase_status": project_closure_phase_plan.get("status", "unknown"),
        "project_phase_readiness_status": project_phase_readiness_scoreboard.get("status", "unknown"),
        "project_owner_status": project_owner_accountability_matrix.get("status", "unknown"),
        "project_owner_queue_status": project_owner_work_queue.get("status", "unknown"),
        "project_critical_path_status": project_critical_path_report.get("status", "unknown"),
        "project_owner_next_actions_status": project_owner_next_actions_summary.get("status", "unknown"),
        "project_ready_now_status": project_ready_now_board.get("status", "unknown"),
        "project_unlock_impact_status": project_unlock_impact_report.get("status", "unknown"),
        "generated_truth_status": _read_json_if_exists(layout.reports_dir / "generated_truth_consistency_report.json").get("status", "unknown"),
        "generated_truth_crosscheck_status": _read_json_if_exists(layout.reports_dir / "generated_truth_crosscheck_matrix.json").get("status", "unknown"),
    }


def render_closure_gap_summary_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Closure Gap Summary",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- repo_side_complete: `{report.get('repo_side_complete', False)}`",
        f"- missing_required_count: `{report.get('missing_required_count', 0)}`",
        f"- blocker_count: `{report.get('blocker_count', 0)}`",
        f"- overall_internal_ready: `{report.get('overall_internal_ready', False)}`",
        f"- overall_external_ready: `{report.get('overall_external_ready', False)}`",
        f"- truth_docs_status: `{report.get('truth_docs_status', 'unknown')}`",
        f"- project_actionability_status: `{report.get('project_actionability_status', 'unknown')}`",
        f"- project_phase_status: `{report.get('project_phase_status', 'unknown')}`",
        f"- project_phase_readiness_status: `{report.get('project_phase_readiness_status', 'unknown')}`",
        f"- project_owner_status: `{report.get('project_owner_status', 'unknown')}`",
        f"- project_owner_queue_status: `{report.get('project_owner_queue_status', 'unknown')}`",
        f"- project_critical_path_status: `{report.get('project_critical_path_status', 'unknown')}`",
        f"- project_owner_next_actions_status: `{report.get('project_owner_next_actions_status', 'unknown')}`",
        f"- project_ready_now_status: `{report.get('project_ready_now_status', 'unknown')}`",
        f"- project_unlock_impact_status: `{report.get('project_unlock_impact_status', 'unknown')}`",
        f"- generated_truth_status: `{report.get('generated_truth_status', 'unknown')}`",
        f"- generated_truth_crosscheck_status: `{report.get('generated_truth_crosscheck_status', 'unknown')}`",
    ]
    return "\n".join(lines) + "\n"


def build_project_master_truth_reference(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    rows = [
        {"label": "governance_and_repo_contracts", "repo_side_state": "repo-side strong", "real_closure_blocked": True},
        {"label": "train_readiness_45k", "repo_side_state": "repo-side strong", "real_closure_blocked": True},
        {"label": "chess_onefile_closure", "repo_side_state": "repo-side strong", "real_closure_blocked": True},
        {"label": "release_process_integrity", "repo_side_state": "repo-side strong", "real_closure_blocked": True},
        {"label": "kernel_and_runtime_paths", "repo_side_state": "repo-side strong", "real_closure_blocked": True},
        {"label": "product_modes_offline_rag_assistant", "repo_side_state": "repo-side partial", "real_closure_blocked": True},
        {"label": "device_export_packaging_truth", "repo_side_state": "repo-side partial", "real_closure_blocked": True},
        {"label": "benchmark_and_claim_safety", "repo_side_state": "repo-side strong", "real_closure_blocked": True},
        {"label": "security_legal_pilot_external", "repo_side_state": "repo-side partial", "real_closure_blocked": True},
        {"label": "management_finalization", "repo_side_state": "repo-side partial", "real_closure_blocked": True},
    ]
    doc_path = REPO_ROOT / "docs" / "PROJECT_MASTER_TRUTH.md"
    doc_tr_path = REPO_ROOT / "docs" / "PROJECT_MASTER_TRUTH_TR.md"
    return {
        "schema": "chess_project_master_truth_reference_v1",
        "run_id": payload.get("run_id", ""),
        "doc_path": str(doc_path),
        "doc_exists": doc_path.exists(),
        "doc_tr_path": str(doc_tr_path),
        "doc_tr_exists": doc_tr_path.exists(),
        "row_count": len(rows),
        "rows": rows,
    }


def render_project_master_truth_reference_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Project Master Truth Reference",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- doc_exists: `{report.get('doc_exists', False)}`",
        f"- doc_tr_exists: `{report.get('doc_tr_exists', False)}`",
        f"- row_count: `{report.get('row_count', 0)}`",
        "",
        "| Lane | Repo-Side State | Real Closure Blocked |",
        "|---|---|---|",
    ]
    for row in report.get("rows", []):
        lines.append(
            f"| `{row.get('label', '')}` | `{row.get('repo_side_state', 'unknown')}` | `{row.get('real_closure_blocked', False)}` |"
        )
    return "\n".join(lines) + "\n"


def build_project_remaining_real_blockers(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    items = [
        {"label": "external_strength_unproven", "severity": "high", "detail": "Measured external strength remains unproven until real runs and benchmark evidence exist."},
        {"label": "real_training_outputs_pending", "severity": "high", "detail": "The repository still needs real 24h and 45K training outputs, not just readiness and packaging surfaces."},
        {"label": "trained_artifact_truth_pending", "severity": "high", "detail": "Final weights truth and best/latest checkpoint truth still require validated trained artifacts."},
        {"label": "benchmark_evidence_pending", "severity": "high", "detail": "Benchmark raw outputs, compare reports, summaries, and locked manifests still require measured closure."},
        {"label": "export_device_packaging_pending", "severity": "high", "detail": "Export parity, device validation, packaging validation, and installer proof still require real target validation."},
        {"label": "external_reproduction_pending", "severity": "high", "detail": "Independent reproduction remains a distinct external confirmation lane."},
        {"label": "security_legal_pilot_pending", "severity": "high", "detail": "Security, legal, and pilot sign-off remain external closure work streams."},
        {"label": "operator_handoff_dr_pending", "severity": "high", "detail": "Operator rehearsal, disaster recovery, backup retention, and blind handoff still require operational proof."},
        {"label": "rc_golden_final_release_pending", "severity": "high", "detail": "RC, golden release, and final release still require real trained artifacts and formal sign-off."},
        {"label": "management_closure_pending", "severity": "medium", "detail": "Final core-complete and maintenance posture decisions still require explicit management closure."},
    ]
    return {
        "schema": "chess_project_remaining_real_blockers_v1",
        "run_id": payload.get("run_id", ""),
        "item_count": len(items),
        "items": items,
    }


def render_project_remaining_real_blockers_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Project Remaining Real Blockers",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- item_count: `{report.get('item_count', 0)}`",
        "",
        "## Blockers",
    ]
    for item in report.get("items", []):
        lines.append(
            f"- `{item.get('label', '')}`: severity=`{item.get('severity', 'unknown')}` detail={item.get('detail', '')}"
        )
    return "\n".join(lines) + "\n"


def _project_blocker_specs() -> Dict[str, Dict[str, Any]]:
    return {
        "external_strength_unproven": {
            "owner_domain": "training_eval",
            "closure_surface": "measured_benchmark",
            "next_action": "Run real chess/main-model training and publish measured benchmark evidence.",
            "required_evidence": ["trained checkpoints", "benchmark raw outputs", "compare report"],
            "depends_on": ["trained_artifact_truth_pending", "benchmark_evidence_pending"],
            "lane_labels": ["chess_onefile_closure", "benchmark_and_claim_safety"],
            "phase": "external_validation",
            "phase_order": 3,
        },
        "real_training_outputs_pending": {
            "owner_domain": "training_ops",
            "closure_surface": "real_training_run",
            "next_action": "Execute the planned 24h and 45K training runs on target hardware.",
            "required_evidence": ["training logs", "training report", "token/compute accounting"],
            "depends_on": [],
            "lane_labels": ["train_readiness_45k", "chess_onefile_closure"],
            "phase": "foundation_runs",
            "phase_order": 1,
        },
        "trained_artifact_truth_pending": {
            "owner_domain": "artifact_governance",
            "closure_surface": "trained_artifact_registry",
            "next_action": "Lock final weights and best/latest checkpoint truth against the actual trained run.",
            "required_evidence": ["final weights truth", "best/latest checkpoint truth", "artifact registry"],
            "depends_on": ["real_training_outputs_pending"],
            "lane_labels": ["chess_onefile_closure", "release_process_integrity"],
            "phase": "measured_internal_closure",
            "phase_order": 2,
        },
        "benchmark_evidence_pending": {
            "owner_domain": "evaluation",
            "closure_surface": "benchmark_closure",
            "next_action": "Preserve raw benchmark outputs and publish compare/summary/manifest artifacts.",
            "required_evidence": ["benchmark raw outputs", "benchmark compare report", "locked benchmark manifest"],
            "depends_on": ["real_training_outputs_pending"],
            "lane_labels": ["chess_onefile_closure", "benchmark_and_claim_safety"],
            "phase": "measured_internal_closure",
            "phase_order": 2,
        },
        "export_device_packaging_pending": {
            "owner_domain": "device_release",
            "closure_surface": "device_validation",
            "next_action": "Validate export, package, installer, and real target device behavior.",
            "required_evidence": ["export parity report", "device validation", "installer validation"],
            "depends_on": ["real_training_outputs_pending"],
            "lane_labels": ["device_export_packaging_truth", "release_process_integrity"],
            "phase": "measured_internal_closure",
            "phase_order": 2,
        },
        "external_reproduction_pending": {
            "owner_domain": "external_validation",
            "closure_surface": "third_party_repro",
            "next_action": "Get an external rerun or independent reproduction note.",
            "required_evidence": ["external repro note", "reproduction logs"],
            "depends_on": ["benchmark_evidence_pending", "external_strength_unproven"],
            "lane_labels": ["security_legal_pilot_external", "benchmark_and_claim_safety"],
            "phase": "external_validation",
            "phase_order": 3,
        },
        "security_legal_pilot_pending": {
            "owner_domain": "security_legal_ops",
            "closure_surface": "external_signoff",
            "next_action": "Complete legal, security, and pilot reviews outside the local repo boundary.",
            "required_evidence": ["security review", "legal review", "pilot sign-off"],
            "depends_on": ["benchmark_evidence_pending", "export_device_packaging_pending"],
            "lane_labels": ["security_legal_pilot_external"],
            "phase": "external_validation",
            "phase_order": 3,
        },
        "operator_handoff_dr_pending": {
            "owner_domain": "ops_handoff",
            "closure_surface": "operational_rehearsal",
            "next_action": "Run operator handoff, DR restore, backup retention, and blind handoff rehearsals.",
            "required_evidence": ["operator rehearsal", "DR evidence", "blind handoff report"],
            "depends_on": ["real_training_outputs_pending"],
            "lane_labels": ["security_legal_pilot_external", "release_process_integrity"],
            "phase": "external_validation",
            "phase_order": 3,
        },
        "rc_golden_final_release_pending": {
            "owner_domain": "release_governance",
            "closure_surface": "release_decision",
            "next_action": "Cut RC, then golden release, then final release after all external gates are closed.",
            "required_evidence": ["RC package", "golden release bundle", "final release note"],
            "depends_on": [
                "external_strength_unproven",
                "trained_artifact_truth_pending",
                "benchmark_evidence_pending",
                "export_device_packaging_pending",
                "external_reproduction_pending",
                "security_legal_pilot_pending",
                "operator_handoff_dr_pending",
            ],
            "lane_labels": ["release_process_integrity", "management_finalization"],
            "phase": "release_finalization",
            "phase_order": 4,
        },
        "management_closure_pending": {
            "owner_domain": "management",
            "closure_surface": "closure_decision",
            "next_action": "Record final core-complete and maintenance posture decisions.",
            "required_evidence": ["closure decision record", "management sign-off"],
            "depends_on": ["rc_golden_final_release_pending"],
            "lane_labels": ["management_finalization"],
            "phase": "governance_closeout",
            "phase_order": 5,
        },
    }


def build_truth_docs_index(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del layout
    readme = _read_text_if_exists(REPO_ROOT / "README.md")
    readme_tr = _read_text_if_exists(REPO_ROOT / "README_TR.md")
    items = [
        {
            "label": "chess_master_truth_en",
            "kind": "doc",
            "path": str(REPO_ROOT / "docs" / "CHESS_ONEFILE_MASTER_TRUTH.md"),
            "exists": (REPO_ROOT / "docs" / "CHESS_ONEFILE_MASTER_TRUTH.md").exists(),
        },
        {
            "label": "chess_master_truth_tr",
            "kind": "doc",
            "path": str(REPO_ROOT / "docs" / "CHESS_ONEFILE_MASTER_TRUTH_TR.md"),
            "exists": (REPO_ROOT / "docs" / "CHESS_ONEFILE_MASTER_TRUTH_TR.md").exists(),
        },
        {
            "label": "project_master_truth_en",
            "kind": "doc",
            "path": str(REPO_ROOT / "docs" / "PROJECT_MASTER_TRUTH.md"),
            "exists": (REPO_ROOT / "docs" / "PROJECT_MASTER_TRUTH.md").exists(),
        },
        {
            "label": "project_master_truth_tr",
            "kind": "doc",
            "path": str(REPO_ROOT / "docs" / "PROJECT_MASTER_TRUTH_TR.md"),
            "exists": (REPO_ROOT / "docs" / "PROJECT_MASTER_TRUTH_TR.md").exists(),
        },
        {
            "label": "readme_chess_master_truth_link_en",
            "kind": "readme_link",
            "path": str(REPO_ROOT / "README.md"),
            "exists": "docs/CHESS_ONEFILE_MASTER_TRUTH.md" in readme,
        },
        {
            "label": "readme_project_master_truth_link_en",
            "kind": "readme_link",
            "path": str(REPO_ROOT / "README.md"),
            "exists": "docs/PROJECT_MASTER_TRUTH.md" in readme,
        },
        {
            "label": "readme_chess_master_truth_link_tr",
            "kind": "readme_link",
            "path": str(REPO_ROOT / "README_TR.md"),
            "exists": "docs/CHESS_ONEFILE_MASTER_TRUTH_TR.md" in readme_tr,
        },
        {
            "label": "readme_project_master_truth_link_tr",
            "kind": "readme_link",
            "path": str(REPO_ROOT / "README_TR.md"),
            "exists": "docs/PROJECT_MASTER_TRUTH_TR.md" in readme_tr,
        },
    ]
    return {
        "schema": "chess_truth_docs_index_v1",
        "run_id": payload.get("run_id", ""),
        "item_count": len(items),
        "items": items,
    }


def render_truth_docs_index_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Truth Docs Index",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- item_count: `{report.get('item_count', 0)}`",
        "",
        "## Items",
    ]
    for item in report.get("items", []):
        lines.append(
            f"- `{item.get('label', '')}`: kind=`{item.get('kind', 'unknown')}` exists=`{item.get('exists', False)}` path=`{item.get('path', '')}`"
        )
    return "\n".join(lines) + "\n"


def build_truth_docs_drift_report(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    master = _read_json_if_exists(layout.reports_dir / "master_closure_table.json")
    blockers = _read_json_if_exists(layout.reports_dir / "remaining_core_blockers.json")
    project_master = _read_json_if_exists(layout.reports_dir / "project_master_truth_reference.json")
    project_blockers = _read_json_if_exists(layout.reports_dir / "project_remaining_real_blockers.json")
    truth_docs_index = _read_json_if_exists(layout.reports_dir / "truth_docs_index.json")

    chess_doc_en = _read_text_if_exists(REPO_ROOT / "docs" / "CHESS_ONEFILE_MASTER_TRUTH.md")
    chess_doc_tr = _read_text_if_exists(REPO_ROOT / "docs" / "CHESS_ONEFILE_MASTER_TRUTH_TR.md")
    project_doc_en = _read_text_if_exists(REPO_ROOT / "docs" / "PROJECT_MASTER_TRUTH.md")
    project_doc_tr = _read_text_if_exists(REPO_ROOT / "docs" / "PROJECT_MASTER_TRUTH_TR.md")

    chess_lane_labels = [str(row.get("label", "")) for row in master.get("rows", [])]
    documented_chess_blocker_labels = {
        "external_strength_unproven",
        "release_surface_not_external_grade",
        "external_reproduction_pending",
        "security_legal_pilot_pending",
        "operator_handoff_dr_pending",
        "release_governance_pending",
        "device_export_packaging_pending",
        "benchmark_closure_pending",
        "training_accounting_pending",
        "trained_artifact_truth_pending",
        "management_closure_pending",
    }
    chess_blocker_labels = [
        str(item.get("label", ""))
        for item in blockers.get("blockers", [])
        if str(item.get("label", "")) in documented_chess_blocker_labels
    ]
    project_lane_labels = [str(row.get("label", "")) for row in project_master.get("rows", [])]
    project_blocker_labels = [str(item.get("label", "")) for item in project_blockers.get("items", [])]

    missing_chess_lanes_en = [label for label in chess_lane_labels if label and label not in chess_doc_en]
    missing_chess_lanes_tr = [label for label in chess_lane_labels if label and label not in chess_doc_tr]
    missing_chess_blockers_en = [label for label in chess_blocker_labels if label and label not in chess_doc_en]
    missing_chess_blockers_tr = [label for label in chess_blocker_labels if label and label not in chess_doc_tr]
    missing_project_lanes_en = [label for label in project_lane_labels if label and label not in project_doc_en]
    missing_project_lanes_tr = [label for label in project_lane_labels if label and label not in project_doc_tr]
    missing_project_blockers_en = [label for label in project_blocker_labels if label and label not in project_doc_en]
    missing_project_blockers_tr = [label for label in project_blocker_labels if label and label not in project_doc_tr]
    missing_truth_index_items = [
        str(item.get("label", ""))
        for item in truth_docs_index.get("items", [])
        if not item.get("exists", False)
    ]
    missing = (
        missing_chess_lanes_en
        + missing_chess_lanes_tr
        + missing_chess_blockers_en
        + missing_chess_blockers_tr
        + missing_project_lanes_en
        + missing_project_lanes_tr
        + missing_project_blockers_en
        + missing_project_blockers_tr
        + missing_truth_index_items
    )
    return {
        "schema": "chess_truth_docs_drift_report_v1",
        "run_id": payload.get("run_id", ""),
        "status": "in_sync" if not missing else "drift_detected",
        "missing_chess_lanes_en": missing_chess_lanes_en,
        "missing_chess_lanes_tr": missing_chess_lanes_tr,
        "missing_chess_blockers_en": missing_chess_blockers_en,
        "missing_chess_blockers_tr": missing_chess_blockers_tr,
        "missing_project_lanes_en": missing_project_lanes_en,
        "missing_project_lanes_tr": missing_project_lanes_tr,
        "missing_project_blockers_en": missing_project_blockers_en,
        "missing_project_blockers_tr": missing_project_blockers_tr,
        "missing_truth_index_items": missing_truth_index_items,
    }


def render_truth_docs_drift_report_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Truth Docs Drift Report",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        "",
        "## Missing Chess Lanes (EN)",
    ]
    for label in report.get("missing_chess_lanes_en", []):
        lines.append(f"- `{label}`")
    lines.append("")
    lines.append("## Missing Chess Lanes (TR)")
    for label in report.get("missing_chess_lanes_tr", []):
        lines.append(f"- `{label}`")
    lines.append("")
    lines.append("## Missing Chess Blockers (EN)")
    for label in report.get("missing_chess_blockers_en", []):
        lines.append(f"- `{label}`")
    lines.append("")
    lines.append("## Missing Chess Blockers (TR)")
    for label in report.get("missing_chess_blockers_tr", []):
        lines.append(f"- `{label}`")
    lines.append("")
    lines.append("## Missing Project Lanes (EN)")
    for label in report.get("missing_project_lanes_en", []):
        lines.append(f"- `{label}`")
    lines.append("")
    lines.append("## Missing Project Lanes (TR)")
    for label in report.get("missing_project_lanes_tr", []):
        lines.append(f"- `{label}`")
    lines.append("")
    lines.append("## Missing Project Blockers (EN)")
    for label in report.get("missing_project_blockers_en", []):
        lines.append(f"- `{label}`")
    lines.append("")
    lines.append("## Missing Project Blockers (TR)")
    for label in report.get("missing_project_blockers_tr", []):
        lines.append(f"- `{label}`")
    lines.append("")
    lines.append("## Missing Truth Index Items")
    for label in report.get("missing_truth_index_items", []):
        lines.append(f"- `{label}`")
    return "\n".join(lines) + "\n"


def build_project_blocker_action_plan(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    action_map = _project_blocker_specs()
    blocker_report = _read_json_if_exists(layout.reports_dir / "project_remaining_real_blockers.json")
    items = []
    for blocker in blocker_report.get("items", []):
        label = str(blocker.get("label", ""))
        mapped = action_map.get(label, {})
        items.append(
            {
                "label": label,
                "severity": blocker.get("severity", "unknown"),
                "owner_domain": mapped.get("owner_domain", "unknown"),
                "closure_surface": mapped.get("closure_surface", "unknown"),
                "next_action": mapped.get("next_action", "Define the next closure step explicitly."),
                "required_evidence": mapped.get("required_evidence", []),
            }
        )
    return {
        "schema": "chess_project_blocker_action_plan_v1",
        "run_id": payload.get("run_id", ""),
        "item_count": len(items),
        "items": items,
    }


def render_project_blocker_action_plan_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Project Blocker Action Plan",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- item_count: `{report.get('item_count', 0)}`",
        "",
        "## Actions",
    ]
    for item in report.get("items", []):
        evidence = ", ".join(f"`{entry}`" for entry in item.get("required_evidence", []))
        lines.append(
            f"- `{item.get('label', '')}`: severity=`{item.get('severity', 'unknown')}` "
            f"owner_domain=`{item.get('owner_domain', 'unknown')}` closure_surface=`{item.get('closure_surface', 'unknown')}` "
            f"next_action={item.get('next_action', '')} required_evidence={evidence}"
        )
    return "\n".join(lines) + "\n"


def build_project_blocker_dependency_graph(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    blocker_report = _read_json_if_exists(layout.reports_dir / "project_remaining_real_blockers.json")
    action_plan = _read_json_if_exists(layout.reports_dir / "project_blocker_action_plan.json")
    action_by_label = {str(item.get("label", "")): item for item in action_plan.get("items", [])}
    specs = _project_blocker_specs()
    blocker_labels = [str(item.get("label", "")) for item in blocker_report.get("items", []) if str(item.get("label", ""))]
    severity_by_label = {
        str(item.get("label", "")): item.get("severity", "unknown")
        for item in blocker_report.get("items", [])
        if str(item.get("label", ""))
    }
    nodes = []
    edges = []
    for label in blocker_labels:
        spec = specs.get(label, {})
        depends_on = [dep for dep in spec.get("depends_on", []) if dep in blocker_labels]
        nodes.append(
            {
                "label": label,
                "severity": severity_by_label.get(label, "unknown"),
                "owner_domain": action_by_label.get(label, {}).get("owner_domain", spec.get("owner_domain", "unknown")),
                "closure_surface": action_by_label.get(label, {}).get("closure_surface", spec.get("closure_surface", "unknown")),
                "depends_on": depends_on,
                "ready_now": not depends_on,
            }
        )
        for dep in depends_on:
            edges.append({"from": dep, "to": label})
    incoming = {node["label"]: 0 for node in nodes}
    outgoing = {node["label"]: 0 for node in nodes}
    for edge in edges:
        incoming[edge["to"]] += 1
        outgoing[edge["from"]] += 1
    return {
        "schema": "chess_project_blocker_dependency_graph_v1",
        "run_id": payload.get("run_id", ""),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "root_count": sum(1 for label in incoming if incoming[label] == 0),
        "terminal_count": sum(1 for label in outgoing if outgoing[label] == 0),
        "nodes": nodes,
        "edges": edges,
    }


def render_project_blocker_dependency_graph_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Project Blocker Dependency Graph",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- node_count: `{report.get('node_count', 0)}`",
        f"- edge_count: `{report.get('edge_count', 0)}`",
        f"- root_count: `{report.get('root_count', 0)}`",
        f"- terminal_count: `{report.get('terminal_count', 0)}`",
        "",
        "## Nodes",
    ]
    for node in report.get("nodes", []):
        deps = ", ".join(f"`{entry}`" for entry in node.get("depends_on", []))
        lines.append(
            f"- `{node.get('label', '')}`: severity=`{node.get('severity', 'unknown')}` "
            f"owner_domain=`{node.get('owner_domain', 'unknown')}` closure_surface=`{node.get('closure_surface', 'unknown')}` "
            f"ready_now=`{node.get('ready_now', False)}` depends_on={deps}"
        )
    lines.append("")
    lines.append("## Edges")
    for edge in report.get("edges", []):
        lines.append(f"- `{edge.get('from', '')}` -> `{edge.get('to', '')}`")
    return "\n".join(lines) + "\n"


def build_project_execution_sequence(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del payload
    graph = _read_json_if_exists(layout.reports_dir / "project_blocker_dependency_graph.json")
    nodes = list(graph.get("nodes", []))
    label_order = [str(node.get("label", "")) for node in nodes]
    node_map = {str(node.get("label", "")): node for node in nodes}
    remaining_incoming = {label: len(node_map[label].get("depends_on", [])) for label in label_order}
    ready = [label for label in label_order if remaining_incoming.get(label, 0) == 0]
    sequence: List[Dict[str, Any]] = []
    emitted: set[str] = set()
    step = 1
    while ready:
        label = ready.pop(0)
        if label in emitted:
            continue
        node = node_map[label]
        sequence.append(
            {
                "step": step,
                "label": label,
                "owner_domain": node.get("owner_domain", "unknown"),
                "closure_surface": node.get("closure_surface", "unknown"),
                "depends_on": list(node.get("depends_on", [])),
                "status": "ready" if not node.get("depends_on", []) else "dependency_ordered",
            }
        )
        emitted.add(label)
        step += 1
        for candidate in label_order:
            if candidate in emitted:
                continue
            if label in node_map[candidate].get("depends_on", []):
                remaining_incoming[candidate] = max(0, remaining_incoming.get(candidate, 0) - 1)
                if remaining_incoming[candidate] == 0 and candidate not in ready:
                    ready.append(candidate)
    cycle_detected = len(emitted) != len(label_order)
    if cycle_detected:
        for label in label_order:
            if label not in emitted:
                node = node_map[label]
                sequence.append(
                    {
                        "step": step,
                        "label": label,
                        "owner_domain": node.get("owner_domain", "unknown"),
                        "closure_surface": node.get("closure_surface", "unknown"),
                        "depends_on": list(node.get("depends_on", [])),
                        "status": "cycle_fallback",
                    }
                )
                step += 1
    return {
        "schema": "chess_project_execution_sequence_v1",
        "run_id": graph.get("run_id", ""),
        "item_count": len(sequence),
        "cycle_detected": cycle_detected,
        "items": sequence,
    }


def render_project_execution_sequence_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Project Execution Sequence",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- item_count: `{report.get('item_count', 0)}`",
        f"- cycle_detected: `{report.get('cycle_detected', False)}`",
        "",
        "## Sequence",
    ]
    for item in report.get("items", []):
        deps = ", ".join(f"`{entry}`" for entry in item.get("depends_on", []))
        lines.append(
            f"- step=`{item.get('step', 0)}` `{item.get('label', '')}`: owner_domain=`{item.get('owner_domain', 'unknown')}` "
            f"closure_surface=`{item.get('closure_surface', 'unknown')}` status=`{item.get('status', 'unknown')}` depends_on={deps}"
        )
    return "\n".join(lines) + "\n"


def build_project_lane_status_board(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del payload
    project_master = _read_json_if_exists(layout.reports_dir / "project_master_truth_reference.json")
    project_blockers = _read_json_if_exists(layout.reports_dir / "project_remaining_real_blockers.json")
    action_plan = _read_json_if_exists(layout.reports_dir / "project_blocker_action_plan.json")
    action_by_label = {str(item.get("label", "")): item for item in action_plan.get("items", [])}
    specs = _project_blocker_specs()
    blocker_labels = [str(item.get("label", "")) for item in project_blockers.get("items", [])]
    rows = []
    covered_blockers: set[str] = set()
    for lane in project_master.get("rows", []):
        lane_label = str(lane.get("label", ""))
        lane_blockers = [label for label in blocker_labels if lane_label in specs.get(label, {}).get("lane_labels", [])]
        covered_blockers.update(lane_blockers)
        primary = action_by_label.get(lane_blockers[0], {}) if lane_blockers else {}
        rows.append(
            {
                "label": lane_label,
                "repo_side_state": lane.get("repo_side_state", "unknown"),
                "real_closure_blocked": bool(lane.get("real_closure_blocked", False)),
                "blocker_count": len(lane_blockers),
                "blockers": lane_blockers,
                "primary_owner_domain": primary.get("owner_domain", ""),
                "next_action_summary": primary.get("next_action", "Maintain repo-side state and wait for downstream closure work."),
            }
        )
    return {
        "schema": "chess_project_lane_status_board_v1",
        "run_id": project_master.get("run_id", ""),
        "lane_count": len(rows),
        "covered_blocker_count": len(covered_blockers),
        "status": "ready" if rows and len(covered_blockers) == len(blocker_labels) else "incomplete",
        "rows": rows,
    }


def render_project_lane_status_board_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Project Lane Status Board",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- lane_count: `{report.get('lane_count', 0)}`",
        f"- covered_blocker_count: `{report.get('covered_blocker_count', 0)}`",
        f"- status: `{report.get('status', 'unknown')}`",
        "",
        "| Lane | Repo-Side State | Blocker Count | Primary Owner |",
        "|---|---|---:|---|",
    ]
    for row in report.get("rows", []):
        lines.append(
            f"| `{row.get('label', '')}` | `{row.get('repo_side_state', 'unknown')}` | `{row.get('blocker_count', 0)}` | `{row.get('primary_owner_domain', '')}` |"
        )
    lines.append("")
    lines.append("## Next Actions")
    for row in report.get("rows", []):
        blockers = ", ".join(f"`{entry}`" for entry in row.get("blockers", []))
        lines.append(f"- `{row.get('label', '')}`: blockers={blockers} next_action={row.get('next_action_summary', '')}")
    return "\n".join(lines) + "\n"


def build_project_closure_phase_plan(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del payload
    blocker_report = _read_json_if_exists(layout.reports_dir / "project_remaining_real_blockers.json")
    action_plan = _read_json_if_exists(layout.reports_dir / "project_blocker_action_plan.json")
    action_by_label = {str(item.get("label", "")): item for item in action_plan.get("items", [])}
    specs = _project_blocker_specs()
    phase_labels = {
        "foundation_runs": "Foundation Runs",
        "measured_internal_closure": "Measured Internal Closure",
        "external_validation": "External Validation",
        "release_finalization": "Release Finalization",
        "governance_closeout": "Governance Closeout",
    }
    phases: Dict[str, Dict[str, Any]] = {}
    for blocker in blocker_report.get("items", []):
        label = str(blocker.get("label", ""))
        spec = specs.get(label, {})
        phase = str(spec.get("phase", "unassigned"))
        phase_order = int(spec.get("phase_order", 999))
        phase_entry = phases.setdefault(
            phase,
            {
                "phase": phase,
                "phase_label": phase_labels.get(phase, phase.replace("_", " ").title()),
                "phase_order": phase_order,
                "blockers": [],
                "owner_domains": set(),
                "ready_now_count": 0,
            },
        )
        phase_entry["blockers"].append(label)
        owner_domain = action_by_label.get(label, {}).get("owner_domain", spec.get("owner_domain", "unknown"))
        phase_entry["owner_domains"].add(owner_domain)
        if not spec.get("depends_on", []):
            phase_entry["ready_now_count"] += 1
    rows = []
    for phase in sorted(phases.values(), key=lambda item: (item["phase_order"], item["phase"])):
        rows.append(
            {
                "phase": phase["phase"],
                "phase_label": phase["phase_label"],
                "phase_order": phase["phase_order"],
                "blocker_count": len(phase["blockers"]),
                "blockers": sorted(phase["blockers"]),
                "owner_domains": sorted(phase["owner_domains"]),
                "ready_now_count": int(phase["ready_now_count"]),
                "status": "ready" if phase["blockers"] else "empty",
            }
        )
    return {
        "schema": "chess_project_closure_phase_plan_v1",
        "run_id": blocker_report.get("run_id", ""),
        "phase_count": len(rows),
        "status": "ready" if rows else "incomplete",
        "rows": rows,
    }


def render_project_closure_phase_plan_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Project Closure Phase Plan",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- phase_count: `{report.get('phase_count', 0)}`",
        f"- status: `{report.get('status', 'unknown')}`",
        "",
        "| Phase | Order | Blocker Count | Ready Now |",
        "|---|---:|---:|---:|",
    ]
    for row in report.get("rows", []):
        lines.append(
            f"| `{row.get('phase_label', '')}` | `{row.get('phase_order', 0)}` | `{row.get('blocker_count', 0)}` | `{row.get('ready_now_count', 0)}` |"
        )
    lines.append("")
    lines.append("## Phase Details")
    for row in report.get("rows", []):
        blockers = ", ".join(f"`{entry}`" for entry in row.get("blockers", []))
        owners = ", ".join(f"`{entry}`" for entry in row.get("owner_domains", []))
        lines.append(
            f"- `{row.get('phase_label', '')}`: blockers={blockers} owner_domains={owners} status=`{row.get('status', 'unknown')}`"
        )
    return "\n".join(lines) + "\n"


def build_project_phase_readiness_scoreboard(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del payload
    phase_plan = _read_json_if_exists(layout.reports_dir / "project_closure_phase_plan.json")
    rows = []
    for phase in phase_plan.get("rows", []):
        blocker_count = int(phase.get("blocker_count", 0))
        ready_now_count = int(phase.get("ready_now_count", 0))
        blocked_count = max(0, blocker_count - ready_now_count)
        rows.append(
            {
                "phase": phase.get("phase", ""),
                "phase_label": phase.get("phase_label", ""),
                "phase_order": int(phase.get("phase_order", 0)),
                "blocker_count": blocker_count,
                "ready_now_count": ready_now_count,
                "blocked_count": blocked_count,
                "readiness_ratio": (float(ready_now_count) / float(blocker_count)) if blocker_count else 0.0,
                "status": "ready_now" if blocker_count and ready_now_count == blocker_count else ("partially_ready" if ready_now_count > 0 else "dependency_blocked"),
            }
        )
    return {
        "schema": "chess_project_phase_readiness_scoreboard_v1",
        "run_id": phase_plan.get("run_id", ""),
        "phase_count": len(rows),
        "status": "ready" if rows else "incomplete",
        "rows": rows,
    }


def render_project_phase_readiness_scoreboard_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Project Phase Readiness Scoreboard",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- phase_count: `{report.get('phase_count', 0)}`",
        f"- status: `{report.get('status', 'unknown')}`",
        "",
        "| Phase | Blockers | Ready Now | Blocked | Ratio | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in report.get("rows", []):
        lines.append(
            f"| `{row.get('phase_label', '')}` | `{row.get('blocker_count', 0)}` | `{row.get('ready_now_count', 0)}` | "
            f"`{row.get('blocked_count', 0)}` | `{row.get('readiness_ratio', 0.0):.2f}` | `{row.get('status', 'unknown')}` |"
        )
    return "\n".join(lines) + "\n"


def build_project_owner_accountability_matrix(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del payload
    blocker_report = _read_json_if_exists(layout.reports_dir / "project_remaining_real_blockers.json")
    action_plan = _read_json_if_exists(layout.reports_dir / "project_blocker_action_plan.json")
    specs = _project_blocker_specs()
    owners: Dict[str, Dict[str, Any]] = {}
    for blocker in blocker_report.get("items", []):
        label = str(blocker.get("label", ""))
        action = next((item for item in action_plan.get("items", []) if item.get("label") == label), {})
        owner_domain = str(action.get("owner_domain", specs.get(label, {}).get("owner_domain", "unknown")))
        entry = owners.setdefault(
            owner_domain,
            {
                "owner_domain": owner_domain,
                "blockers": [],
                "phases": set(),
                "lane_labels": set(),
                "closure_surfaces": set(),
            },
        )
        spec = specs.get(label, {})
        entry["blockers"].append(label)
        entry["phases"].add(str(spec.get("phase", "unassigned")))
        for lane_label in spec.get("lane_labels", []):
            entry["lane_labels"].add(str(lane_label))
        entry["closure_surfaces"].add(str(action.get("closure_surface", spec.get("closure_surface", "unknown"))))
    rows = []
    for owner in sorted(owners.values(), key=lambda item: item["owner_domain"]):
        rows.append(
            {
                "owner_domain": owner["owner_domain"],
                "blocker_count": len(owner["blockers"]),
                "blockers": sorted(owner["blockers"]),
                "phases": sorted(owner["phases"]),
                "lane_labels": sorted(owner["lane_labels"]),
                "closure_surfaces": sorted(owner["closure_surfaces"]),
                "status": "covered" if owner["blockers"] else "empty",
            }
        )
    covered_blockers = sum(row["blocker_count"] for row in rows)
    return {
        "schema": "chess_project_owner_accountability_matrix_v1",
        "run_id": blocker_report.get("run_id", ""),
        "owner_count": len(rows),
        "covered_blocker_count": covered_blockers,
        "status": "ready" if rows else "incomplete",
        "rows": rows,
    }


def render_project_owner_accountability_matrix_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Project Owner Accountability Matrix",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- owner_count: `{report.get('owner_count', 0)}`",
        f"- covered_blocker_count: `{report.get('covered_blocker_count', 0)}`",
        f"- status: `{report.get('status', 'unknown')}`",
        "",
        "| Owner | Blocker Count | Phases |",
        "|---|---:|---|",
    ]
    for row in report.get("rows", []):
        phases = ", ".join(f"`{entry}`" for entry in row.get("phases", []))
        lines.append(f"| `{row.get('owner_domain', '')}` | `{row.get('blocker_count', 0)}` | {phases} |")
    lines.append("")
    lines.append("## Ownership Details")
    for row in report.get("rows", []):
        blockers = ", ".join(f"`{entry}`" for entry in row.get("blockers", []))
        lanes = ", ".join(f"`{entry}`" for entry in row.get("lane_labels", []))
        surfaces = ", ".join(f"`{entry}`" for entry in row.get("closure_surfaces", []))
        lines.append(
            f"- `{row.get('owner_domain', '')}`: blockers={blockers} lane_labels={lanes} closure_surfaces={surfaces} status=`{row.get('status', 'unknown')}`"
        )
    return "\n".join(lines) + "\n"


def build_project_owner_work_queue(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del payload
    action_plan = _read_json_if_exists(layout.reports_dir / "project_blocker_action_plan.json")
    execution_sequence = _read_json_if_exists(layout.reports_dir / "project_execution_sequence.json")
    specs = _project_blocker_specs()
    seq_map = {str(item.get("label", "")): item for item in execution_sequence.get("items", [])}
    owners: Dict[str, Dict[str, Any]] = {}
    for item in action_plan.get("items", []):
        label = str(item.get("label", ""))
        owner = str(item.get("owner_domain", "unknown"))
        spec = specs.get(label, {})
        queue = owners.setdefault(
            owner,
            {
                "owner_domain": owner,
                "items": [],
            },
        )
        seq_item = seq_map.get(label, {})
        queue["items"].append(
            {
                "label": label,
                "step": int(seq_item.get("step", 999)),
                "phase": str(spec.get("phase", "unassigned")),
                "phase_order": int(spec.get("phase_order", 999)),
                "closure_surface": str(item.get("closure_surface", spec.get("closure_surface", "unknown"))),
                "next_action": str(item.get("next_action", "")),
                "status": str(seq_item.get("status", "unknown")),
            }
        )
    rows = []
    for owner_domain in sorted(owners):
        queue_items = sorted(owners[owner_domain]["items"], key=lambda item: (item["step"], item["phase_order"], item["label"]))
        rows.append(
            {
                "owner_domain": owner_domain,
                "item_count": len(queue_items),
                "next_up": queue_items[0]["label"] if queue_items else "",
                "items": queue_items,
                "status": "ready" if queue_items else "empty",
            }
        )
    return {
        "schema": "chess_project_owner_work_queue_v1",
        "run_id": action_plan.get("run_id", ""),
        "owner_count": len(rows),
        "status": "ready" if rows else "incomplete",
        "rows": rows,
    }


def render_project_owner_work_queue_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Project Owner Work Queue",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- owner_count: `{report.get('owner_count', 0)}`",
        f"- status: `{report.get('status', 'unknown')}`",
        "",
        "## Owners",
    ]
    for row in report.get("rows", []):
        lines.append(
            f"- `{row.get('owner_domain', '')}`: item_count=`{row.get('item_count', 0)}` next_up=`{row.get('next_up', '')}` status=`{row.get('status', 'unknown')}`"
        )
        for item in row.get("items", []):
            lines.append(
                f"  - step=`{item.get('step', 0)}` `{item.get('label', '')}` phase=`{item.get('phase', '')}` "
                f"closure_surface=`{item.get('closure_surface', '')}` status=`{item.get('status', 'unknown')}`"
            )
    return "\n".join(lines) + "\n"


def build_project_critical_path_report(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del payload
    graph = _read_json_if_exists(layout.reports_dir / "project_blocker_dependency_graph.json")
    sequence = _read_json_if_exists(layout.reports_dir / "project_execution_sequence.json")
    node_map = {str(node.get("label", "")): node for node in graph.get("nodes", [])}
    outgoing: Dict[str, List[str]] = {label: [] for label in node_map}
    for edge in graph.get("edges", []):
        source = str(edge.get("from", ""))
        target = str(edge.get("to", ""))
        if source in outgoing:
            outgoing[source].append(target)
    best_length: Dict[str, int] = {}
    best_parent: Dict[str, str] = {}
    for item in sequence.get("items", []):
        label = str(item.get("label", ""))
        if not label:
            continue
        deps = [str(dep) for dep in node_map.get(label, {}).get("depends_on", []) if str(dep)]
        if not deps:
            best_length[label] = 1
            continue
        parent = max(deps, key=lambda dep: (best_length.get(dep, 0), dep))
        best_length[label] = best_length.get(parent, 0) + 1
        best_parent[label] = parent
    terminals = [label for label, targets in outgoing.items() if not targets] or list(node_map)
    if terminals:
        terminal = max(terminals, key=lambda label: (best_length.get(label, 0), label))
    else:
        terminal = ""
    path_labels: List[str] = []
    cursor = terminal
    seen: set[str] = set()
    while cursor and cursor not in seen:
        path_labels.append(cursor)
        seen.add(cursor)
        cursor = best_parent.get(cursor, "")
    path_labels.reverse()
    path_nodes = [node_map.get(label, {}) for label in path_labels]
    owner_domains = sorted({str(node.get("owner_domain", "")) for node in path_nodes if str(node.get("owner_domain", ""))})
    closure_surfaces = sorted({str(node.get("closure_surface", "")) for node in path_nodes if str(node.get("closure_surface", ""))})
    return {
        "schema": "chess_project_critical_path_report_v1",
        "run_id": graph.get("run_id", ""),
        "path_length": len(path_labels),
        "terminal_label": terminal,
        "owner_domains": owner_domains,
        "closure_surfaces": closure_surfaces,
        "path_labels": path_labels,
        "status": "ready" if path_labels else "incomplete",
    }


def render_project_critical_path_report_md(report: Dict[str, Any]) -> str:
    owners = ", ".join(f"`{entry}`" for entry in report.get("owner_domains", []))
    surfaces = ", ".join(f"`{entry}`" for entry in report.get("closure_surfaces", []))
    path_labels = ", ".join(f"`{entry}`" for entry in report.get("path_labels", []))
    lines = [
        "# Project Critical Path Report",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- path_length: `{report.get('path_length', 0)}`",
        f"- terminal_label: `{report.get('terminal_label', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        f"- owner_domains: {owners}",
        f"- closure_surfaces: {surfaces}",
        "",
        "## Path",
        f"- labels: {path_labels}",
    ]
    return "\n".join(lines) + "\n"


def build_project_owner_next_actions_summary(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del payload
    owner_work_queue = _read_json_if_exists(layout.reports_dir / "project_owner_work_queue.json")
    rows = []
    for row in owner_work_queue.get("rows", []):
        items = row.get("items", [])
        next_item = items[0] if items else {}
        rows.append(
            {
                "owner_domain": row.get("owner_domain", ""),
                "item_count": int(row.get("item_count", 0)),
                "next_up": row.get("next_up", ""),
                "next_step": int(next_item.get("step", 0)) if next_item else 0,
                "next_phase": next_item.get("phase", ""),
                "next_closure_surface": next_item.get("closure_surface", ""),
                "next_action": next_item.get("next_action", ""),
                "status": row.get("status", "unknown"),
            }
        )
    return {
        "schema": "chess_project_owner_next_actions_summary_v1",
        "run_id": owner_work_queue.get("run_id", ""),
        "owner_count": len(rows),
        "status": "ready" if rows else "incomplete",
        "rows": rows,
    }


def render_project_owner_next_actions_summary_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Project Owner Next Actions Summary",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- owner_count: `{report.get('owner_count', 0)}`",
        f"- status: `{report.get('status', 'unknown')}`",
        "",
        "## Owner Next Actions",
    ]
    for row in report.get("rows", []):
        lines.append(
            f"- `{row.get('owner_domain', '')}`: next_up=`{row.get('next_up', '')}` step=`{row.get('next_step', 0)}` "
            f"phase=`{row.get('next_phase', '')}` closure_surface=`{row.get('next_closure_surface', '')}` "
            f"item_count=`{row.get('item_count', 0)}` status=`{row.get('status', 'unknown')}` next_action={row.get('next_action', '')}"
        )
    return "\n".join(lines) + "\n"


def build_project_ready_now_board(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del payload
    graph = _read_json_if_exists(layout.reports_dir / "project_blocker_dependency_graph.json")
    action_plan = _read_json_if_exists(layout.reports_dir / "project_blocker_action_plan.json")
    action_by_label = {str(item.get("label", "")): item for item in action_plan.get("items", [])}
    direct_unlocks: Dict[str, int] = {}
    for edge in graph.get("edges", []):
        source = str(edge.get("from", ""))
        if source:
            direct_unlocks[source] = direct_unlocks.get(source, 0) + 1
    rows = []
    for node in graph.get("nodes", []):
        if not bool(node.get("ready_now", False)):
            continue
        label = str(node.get("label", ""))
        action = action_by_label.get(label, {})
        rows.append(
            {
                "label": label,
                "owner_domain": str(node.get("owner_domain", "")),
                "closure_surface": str(node.get("closure_surface", "")),
                "direct_unlock_count": int(direct_unlocks.get(label, 0)),
                "next_action": str(action.get("next_action", "")),
                "status": "ready_now",
            }
        )
    rows.sort(key=lambda item: (-item["direct_unlock_count"], item["label"]))
    return {
        "schema": "chess_project_ready_now_board_v1",
        "run_id": graph.get("run_id", ""),
        "item_count": len(rows),
        "status": "ready" if rows else "incomplete",
        "rows": rows,
    }


def render_project_ready_now_board_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Project Ready-Now Board",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- item_count: `{report.get('item_count', 0)}`",
        f"- status: `{report.get('status', 'unknown')}`",
        "",
        "## Ready Now",
    ]
    for row in report.get("rows", []):
        lines.append(
            f"- `{row.get('label', '')}`: owner_domain=`{row.get('owner_domain', '')}` "
            f"closure_surface=`{row.get('closure_surface', '')}` direct_unlock_count=`{row.get('direct_unlock_count', 0)}` "
            f"status=`{row.get('status', 'unknown')}` next_action={row.get('next_action', '')}"
        )
    return "\n".join(lines) + "\n"


def build_project_unlock_impact_report(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del payload
    graph = _read_json_if_exists(layout.reports_dir / "project_blocker_dependency_graph.json")
    outgoing: Dict[str, List[str]] = {}
    node_map = {str(node.get("label", "")): node for node in graph.get("nodes", [])}
    for label in node_map:
        outgoing[label] = []
    for edge in graph.get("edges", []):
        source = str(edge.get("from", ""))
        target = str(edge.get("to", ""))
        if source in outgoing and target:
            outgoing[source].append(target)

    memo: Dict[str, set[str]] = {}

    def descendants(label: str) -> set[str]:
        if label in memo:
            return memo[label]
        seen: set[str] = set()
        for child in outgoing.get(label, []):
            seen.add(child)
            seen.update(descendants(child))
        memo[label] = seen
        return seen

    rows = []
    for label, node in node_map.items():
        all_desc = descendants(label)
        rows.append(
            {
                "label": label,
                "owner_domain": str(node.get("owner_domain", "")),
                "closure_surface": str(node.get("closure_surface", "")),
                "direct_unlock_count": len(outgoing.get(label, [])),
                "total_unlock_count": len(all_desc),
                "status": "ranked",
            }
        )
    rows.sort(key=lambda item: (-item["total_unlock_count"], -item["direct_unlock_count"], item["label"]))
    return {
        "schema": "chess_project_unlock_impact_report_v1",
        "run_id": graph.get("run_id", ""),
        "item_count": len(rows),
        "top_unlock_label": rows[0]["label"] if rows else "",
        "status": "ready" if rows else "incomplete",
        "rows": rows,
    }


def render_project_unlock_impact_report_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Project Unlock Impact Report",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- item_count: `{report.get('item_count', 0)}`",
        f"- top_unlock_label: `{report.get('top_unlock_label', '')}`",
        f"- status: `{report.get('status', 'unknown')}`",
        "",
        "## Unlock Ranking",
    ]
    for row in report.get("rows", []):
        lines.append(
            f"- `{row.get('label', '')}`: total_unlock_count=`{row.get('total_unlock_count', 0)}` "
            f"direct_unlock_count=`{row.get('direct_unlock_count', 0)}` owner_domain=`{row.get('owner_domain', '')}` "
            f"closure_surface=`{row.get('closure_surface', '')}` status=`{row.get('status', 'unknown')}`"
        )
    return "\n".join(lines) + "\n"


def build_generated_truth_consistency_report(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del payload
    truth = _read_json_if_exists(layout.reports_dir / "artifact_truth_matrix.json")
    repo_summary = _read_json_if_exists(layout.reports_dir / "repo_side_completion_summary.json")
    master = _read_json_if_exists(layout.reports_dir / "master_closure_table.json")
    aggregated = _read_json_if_exists(layout.reports_dir / "aggregated_master_table.json")
    blockers = _read_json_if_exists(layout.reports_dir / "remaining_core_blockers.json")
    real_remaining = _read_json_if_exists(layout.reports_dir / "real_remaining_core_work.json")
    closure_gap = _read_json_if_exists(layout.reports_dir / "closure_gap_summary.json")
    truth_docs_drift = _read_json_if_exists(layout.reports_dir / "truth_docs_drift_report.json")
    project_master = _read_json_if_exists(layout.reports_dir / "project_master_truth_reference.json")
    project_blockers = _read_json_if_exists(layout.reports_dir / "project_remaining_real_blockers.json")
    project_blocker_action_plan = _read_json_if_exists(layout.reports_dir / "project_blocker_action_plan.json")
    project_blocker_dependency_graph = _read_json_if_exists(layout.reports_dir / "project_blocker_dependency_graph.json")
    project_execution_sequence = _read_json_if_exists(layout.reports_dir / "project_execution_sequence.json")
    project_lane_status_board = _read_json_if_exists(layout.reports_dir / "project_lane_status_board.json")
    project_closure_phase_plan = _read_json_if_exists(layout.reports_dir / "project_closure_phase_plan.json")
    project_phase_readiness_scoreboard = _read_json_if_exists(layout.reports_dir / "project_phase_readiness_scoreboard.json")
    project_owner_accountability_matrix = _read_json_if_exists(layout.reports_dir / "project_owner_accountability_matrix.json")
    project_owner_work_queue = _read_json_if_exists(layout.reports_dir / "project_owner_work_queue.json")
    project_critical_path_report = _read_json_if_exists(layout.reports_dir / "project_critical_path_report.json")
    project_owner_next_actions_summary = _read_json_if_exists(layout.reports_dir / "project_owner_next_actions_summary.json")
    project_ready_now_board = _read_json_if_exists(layout.reports_dir / "project_ready_now_board.json")
    project_unlock_impact_report = _read_json_if_exists(layout.reports_dir / "project_unlock_impact_report.json")

    checks = [
        {
            "label": "master_row_count_matches_aggregated",
            "passed": int(master.get("row_count", 0)) == int(aggregated.get("row_count", -1)) and int(master.get("row_count", 0)) > 0,
        },
        {
            "label": "remaining_blockers_match_real_remaining",
            "passed": int(blockers.get("blocker_count", 0)) == int(real_remaining.get("item_count", -1)),
        },
        {
            "label": "repo_summary_matches_truth_matrix",
            "passed": (
                int(repo_summary.get("required_count", -1)) == int(truth.get("required_count", -2))
                and int(repo_summary.get("present_required_count", -1)) == int(truth.get("present_required_count", -2))
            ),
        },
        {
            "label": "closure_gap_matches_truth_docs_status",
            "passed": str(closure_gap.get("truth_docs_status", "")) == str(truth_docs_drift.get("status", "")),
        },
        {
            "label": "project_reference_and_blockers_present",
            "passed": int(project_master.get("row_count", 0)) > 0 and int(project_blockers.get("item_count", 0)) > 0,
        },
        {
            "label": "project_blocker_action_plan_complete",
            "passed": int(project_blocker_action_plan.get("item_count", 0)) == int(project_blockers.get("item_count", -1)),
        },
        {
            "label": "project_actionability_surfaces_present",
            "passed": (
                int(project_blocker_dependency_graph.get("node_count", 0)) > 0
                and int(project_execution_sequence.get("item_count", 0)) > 0
                and int(project_lane_status_board.get("lane_count", 0)) > 0
                and int(project_closure_phase_plan.get("phase_count", 0)) > 0
                and int(project_phase_readiness_scoreboard.get("phase_count", 0)) > 0
                and int(project_owner_accountability_matrix.get("owner_count", 0)) > 0
                and int(project_owner_work_queue.get("owner_count", 0)) > 0
                and int(project_critical_path_report.get("path_length", 0)) > 0
                and int(project_owner_next_actions_summary.get("owner_count", 0)) > 0
                and int(project_ready_now_board.get("item_count", 0)) > 0
                and int(project_unlock_impact_report.get("item_count", 0)) > 0
            ),
        },
        {
            "label": "truth_docs_are_in_sync",
            "passed": truth_docs_drift.get("status") == "in_sync",
        },
    ]
    failed = [item["label"] for item in checks if not item["passed"]]
    return {
        "schema": "chess_generated_truth_consistency_report_v1",
        "run_id": truth.get("run_id", ""),
        "check_count": len(checks),
        "checks": checks,
        "failed_checks": failed,
        "status": "consistent" if not failed else "inconsistent",
    }


def render_generated_truth_consistency_report_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Generated Truth Consistency Report",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- check_count: `{report.get('check_count', 0)}`",
        f"- status: `{report.get('status', 'unknown')}`",
        "",
        "## Checks",
    ]
    for item in report.get("checks", []):
        lines.append(f"- `{item.get('label', '')}`: passed=`{item.get('passed', False)}`")
    lines.append("")
    lines.append("## Failed Checks")
    for label in report.get("failed_checks", []):
        lines.append(f"- `{label}`")
    return "\n".join(lines) + "\n"


def build_generated_truth_crosscheck_matrix(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    del payload
    project_master = _read_json_if_exists(layout.reports_dir / "project_master_truth_reference.json")
    project_blockers = _read_json_if_exists(layout.reports_dir / "project_remaining_real_blockers.json")
    project_blocker_action_plan = _read_json_if_exists(layout.reports_dir / "project_blocker_action_plan.json")
    project_blocker_dependency_graph = _read_json_if_exists(layout.reports_dir / "project_blocker_dependency_graph.json")
    project_execution_sequence = _read_json_if_exists(layout.reports_dir / "project_execution_sequence.json")
    project_lane_status_board = _read_json_if_exists(layout.reports_dir / "project_lane_status_board.json")
    project_closure_phase_plan = _read_json_if_exists(layout.reports_dir / "project_closure_phase_plan.json")
    project_phase_readiness_scoreboard = _read_json_if_exists(layout.reports_dir / "project_phase_readiness_scoreboard.json")
    project_owner_accountability_matrix = _read_json_if_exists(layout.reports_dir / "project_owner_accountability_matrix.json")
    project_owner_work_queue = _read_json_if_exists(layout.reports_dir / "project_owner_work_queue.json")
    project_critical_path_report = _read_json_if_exists(layout.reports_dir / "project_critical_path_report.json")
    project_owner_next_actions_summary = _read_json_if_exists(layout.reports_dir / "project_owner_next_actions_summary.json")
    project_ready_now_board = _read_json_if_exists(layout.reports_dir / "project_ready_now_board.json")
    project_unlock_impact_report = _read_json_if_exists(layout.reports_dir / "project_unlock_impact_report.json")
    generated_truth_consistency_report = _read_json_if_exists(layout.reports_dir / "generated_truth_consistency_report.json")
    specs = _project_blocker_specs()

    blocker_labels = {str(item.get("label", "")) for item in project_blockers.get("items", [])}
    sequence_labels = {str(item.get("label", "")) for item in project_execution_sequence.get("items", [])}
    lane_covered_labels = {
        label
        for row in project_lane_status_board.get("rows", [])
        for label in row.get("blockers", [])
        if isinstance(label, str)
    }
    phase_covered_labels = {
        label
        for row in project_closure_phase_plan.get("rows", [])
        for label in row.get("blockers", [])
        if isinstance(label, str)
    }
    owner_covered_labels = {
        label
        for row in project_owner_accountability_matrix.get("rows", [])
        for label in row.get("blockers", [])
        if isinstance(label, str)
    }
    owner_queue_labels = {
        str(item.get("label", ""))
        for row in project_owner_work_queue.get("rows", [])
        for item in row.get("items", [])
        if isinstance(item, dict)
    }
    next_actions_by_owner = {
        str(row.get("owner_domain", "")): str(row.get("next_up", ""))
        for row in project_owner_next_actions_summary.get("rows", [])
    }
    queue_next_by_owner = {
        str(row.get("owner_domain", "")): str(row.get("next_up", ""))
        for row in project_owner_work_queue.get("rows", [])
    }
    ready_now_labels = {str(row.get("label", "")) for row in project_ready_now_board.get("rows", [])}
    root_labels = {str(node.get("label", "")) for node in project_blocker_dependency_graph.get("nodes", []) if not node.get("depends_on", [])}
    sequence_phase_orders = [int(specs.get(str(item.get("label", "")), {}).get("phase_order", 999)) for item in project_execution_sequence.get("items", [])]
    checks = [
        {
            "label": "action_plan_matches_blockers",
            "passed": int(project_blocker_action_plan.get("item_count", 0)) == int(project_blockers.get("item_count", -1)),
        },
        {
            "label": "dependency_graph_matches_blockers",
            "passed": int(project_blocker_dependency_graph.get("node_count", 0)) == int(project_blockers.get("item_count", -1)),
        },
        {
            "label": "dependency_graph_has_roots_and_terminals",
            "passed": int(project_blocker_dependency_graph.get("root_count", 0)) > 0 and int(project_blocker_dependency_graph.get("terminal_count", 0)) > 0,
        },
        {
            "label": "execution_sequence_covers_blockers",
            "passed": blocker_labels == sequence_labels and not bool(project_execution_sequence.get("cycle_detected", True)),
        },
        {
            "label": "lane_status_board_matches_reference",
            "passed": int(project_lane_status_board.get("lane_count", 0)) == int(project_master.get("row_count", -1)),
        },
        {
            "label": "lane_status_board_covers_all_blockers",
            "passed": blocker_labels == lane_covered_labels,
        },
        {
            "label": "phase_plan_covers_all_blockers",
            "passed": blocker_labels == phase_covered_labels and project_closure_phase_plan.get("status") == "ready",
        },
        {
            "label": "phase_readiness_matches_phase_plan",
            "passed": (
                int(project_phase_readiness_scoreboard.get("phase_count", 0)) == int(project_closure_phase_plan.get("phase_count", -1))
                and project_phase_readiness_scoreboard.get("status") == "ready"
            ),
        },
        {
            "label": "owner_matrix_covers_all_blockers",
            "passed": blocker_labels == owner_covered_labels and project_owner_accountability_matrix.get("status") == "ready",
        },
        {
            "label": "owner_work_queue_matches_owner_matrix",
            "passed": (
                blocker_labels == owner_queue_labels
                and int(project_owner_work_queue.get("owner_count", 0)) == int(project_owner_accountability_matrix.get("owner_count", -1))
                and project_owner_work_queue.get("status") == "ready"
            ),
        },
        {
            "label": "critical_path_is_valid",
            "passed": (
                int(project_critical_path_report.get("path_length", 0)) > 0
                and all(label in blocker_labels for label in project_critical_path_report.get("path_labels", []))
                and str(project_critical_path_report.get("terminal_label", "")) in blocker_labels
                and project_critical_path_report.get("status") == "ready"
            ),
        },
        {
            "label": "owner_next_actions_match_queue",
            "passed": (
                int(project_owner_next_actions_summary.get("owner_count", 0)) == int(project_owner_work_queue.get("owner_count", -1))
                and next_actions_by_owner == queue_next_by_owner
                and project_owner_next_actions_summary.get("status") == "ready"
            ),
        },
        {
            "label": "ready_now_board_matches_roots",
            "passed": (
                ready_now_labels == root_labels
                and int(project_ready_now_board.get("item_count", 0)) == len(root_labels)
                and project_ready_now_board.get("status") == "ready"
            ),
        },
        {
            "label": "unlock_impact_matches_graph",
            "passed": (
                int(project_unlock_impact_report.get("item_count", 0)) == int(project_blocker_dependency_graph.get("node_count", -1))
                and str(project_unlock_impact_report.get("top_unlock_label", "")) in blocker_labels
                and project_unlock_impact_report.get("status") == "ready"
            ),
        },
        {
            "label": "execution_sequence_respects_phase_order",
            "passed": sequence_phase_orders == sorted(sequence_phase_orders),
        },
        {
            "label": "generated_truth_consistency_report_clear",
            "passed": generated_truth_consistency_report.get("status") == "consistent",
        },
    ]
    failed = [item["label"] for item in checks if not item["passed"]]
    return {
        "schema": "chess_generated_truth_crosscheck_matrix_v1",
        "run_id": project_master.get("run_id", ""),
        "check_count": len(checks),
        "checks": checks,
        "failed_checks": failed,
        "status": "consistent" if not failed else "inconsistent",
    }


def render_generated_truth_crosscheck_matrix_md(report: Dict[str, Any]) -> str:
    lines = [
        "# Generated Truth Crosscheck Matrix",
        "",
        f"- run_id: `{report.get('run_id', '')}`",
        f"- check_count: `{report.get('check_count', 0)}`",
        f"- status: `{report.get('status', 'unknown')}`",
        "",
        "## Checks",
    ]
    for item in report.get("checks", []):
        lines.append(f"- `{item.get('label', '')}`: passed=`{item.get('passed', False)}`")
    lines.append("")
    lines.append("## Failed Checks")
    for label in report.get("failed_checks", []):
        lines.append(f"- `{label}`")
    return "\n".join(lines) + "\n"


def _write_release_evidence_reports_once(layout: ArtifactLayout, payload: Dict[str, Any]) -> None:
    run_contract = build_run_contract(layout, payload)
    atomic_json(layout.reports_dir / "run_contract.json", run_contract)
    atomic_write_text(layout.reports_dir / "run_contract.md", render_run_contract_md(run_contract))
    release_snapshot = build_release_snapshot(layout, payload)
    atomic_json(layout.reports_dir / "release_snapshot.json", release_snapshot)
    atomic_write_text(layout.reports_dir / "release_snapshot.md", render_release_snapshot_md(release_snapshot))
    evidence_pack = build_evidence_pack_stub(layout, payload)
    atomic_json(layout.reports_dir / "evidence_pack_stub.json", evidence_pack)
    atomic_write_text(layout.reports_dir / "evidence_pack_stub.md", render_evidence_pack_stub_md(evidence_pack))
    truth_registry = build_final_truth_registry(layout, payload)
    atomic_json(layout.reports_dir / "final_truth_registry.json", truth_registry)
    atomic_write_text(layout.reports_dir / "final_truth_registry.md", render_final_truth_registry_md(truth_registry))
    claim_registry = build_claim_registry(layout, payload)
    atomic_json(layout.reports_dir / "claim_registry.json", claim_registry)
    atomic_write_text(layout.reports_dir / "claim_registry.md", render_claim_registry_md(claim_registry))
    known_limits = build_known_limits(layout, payload)
    atomic_json(layout.reports_dir / "known_limits.json", known_limits)
    atomic_write_text(layout.reports_dir / "known_limits.md", render_known_limits_md(known_limits))
    support_matrix = build_support_matrix(layout, payload)
    atomic_json(layout.reports_dir / "support_matrix.json", support_matrix)
    atomic_write_text(layout.reports_dir / "support_matrix.md", render_support_matrix_md(support_matrix))
    release_gate_summary = build_release_gate_summary(layout, payload)
    atomic_json(layout.reports_dir / "release_gate_summary.json", release_gate_summary)
    atomic_write_text(layout.reports_dir / "release_gate_summary.md", render_release_gate_summary_md(release_gate_summary))
    rc_stub = build_rc_stub(layout, payload)
    atomic_json(layout.reports_dir / "rc_stub.json", rc_stub)
    atomic_write_text(layout.reports_dir / "rc_stub.md", render_rc_stub_md(rc_stub))
    golden_stub = build_golden_stub(layout, payload)
    atomic_json(layout.reports_dir / "golden_stub.json", golden_stub)
    atomic_write_text(layout.reports_dir / "golden_stub.md", render_golden_stub_md(golden_stub))
    handoff_pack_manifest = build_handoff_pack_manifest(layout, payload)
    atomic_json(layout.reports_dir / "handoff_pack_manifest.json", handoff_pack_manifest)
    atomic_write_text(layout.reports_dir / "handoff_pack_manifest.md", render_handoff_pack_manifest_md(handoff_pack_manifest))
    operator_handoff_summary = build_operator_handoff_summary(layout, payload)
    atomic_json(layout.reports_dir / "operator_handoff_summary.json", operator_handoff_summary)
    atomic_write_text(layout.reports_dir / "operator_handoff_summary.md", render_operator_handoff_summary_md(operator_handoff_summary))
    external_repro_stub = build_external_repro_stub(layout, payload)
    atomic_json(layout.reports_dir / "external_repro_stub.json", external_repro_stub)
    atomic_write_text(layout.reports_dir / "external_repro_stub.md", render_external_repro_stub_md(external_repro_stub))
    pilot_stub = build_pilot_stub(layout, payload)
    atomic_json(layout.reports_dir / "pilot_stub.json", pilot_stub)
    atomic_write_text(layout.reports_dir / "pilot_stub.md", render_pilot_stub_md(pilot_stub))
    security_stub = build_security_stub(layout, payload)
    atomic_json(layout.reports_dir / "security_stub.json", security_stub)
    atomic_write_text(layout.reports_dir / "security_stub.md", render_security_stub_md(security_stub))
    legal_stub = build_legal_stub(layout, payload)
    atomic_json(layout.reports_dir / "legal_stub.json", legal_stub)
    atomic_write_text(layout.reports_dir / "legal_stub.md", render_legal_stub_md(legal_stub))
    operator_handbook_stub = build_operator_handbook_stub(layout, payload)
    atomic_json(layout.reports_dir / "operator_handbook_stub.json", operator_handbook_stub)
    atomic_write_text(layout.reports_dir / "operator_handbook_stub.md", render_operator_handbook_stub_md(operator_handbook_stub))
    dr_evidence_stub = build_dr_evidence_stub(layout, payload)
    atomic_json(layout.reports_dir / "dr_evidence_stub.json", dr_evidence_stub)
    atomic_write_text(layout.reports_dir / "dr_evidence_stub.md", render_dr_evidence_stub_md(dr_evidence_stub))
    backup_retention_stub = build_backup_retention_stub(layout, payload)
    atomic_json(layout.reports_dir / "backup_retention_stub.json", backup_retention_stub)
    atomic_write_text(layout.reports_dir / "backup_retention_stub.md", render_backup_retention_stub_md(backup_retention_stub))
    blind_handoff_stub = build_blind_handoff_stub(layout, payload)
    atomic_json(layout.reports_dir / "blind_handoff_stub.json", blind_handoff_stub)
    atomic_write_text(layout.reports_dir / "blind_handoff_stub.md", render_blind_handoff_stub_md(blind_handoff_stub))
    release_notes_stub = build_release_notes_stub(layout, payload)
    atomic_json(layout.reports_dir / "release_notes_stub.json", release_notes_stub)
    atomic_write_text(layout.reports_dir / "release_notes_stub.md", render_release_notes_stub_md(release_notes_stub))
    freeze_manifest_stub = build_freeze_manifest_stub(layout, payload)
    atomic_json(layout.reports_dir / "freeze_manifest_stub.json", freeze_manifest_stub)
    atomic_write_text(layout.reports_dir / "freeze_manifest_stub.md", render_freeze_manifest_stub_md(freeze_manifest_stub))
    changelog_snapshot = build_changelog_snapshot(layout, payload)
    atomic_json(layout.reports_dir / "changelog_snapshot.json", changelog_snapshot)
    atomic_write_text(layout.reports_dir / "changelog_snapshot.md", render_changelog_snapshot_md(changelog_snapshot))
    maintenance_policy_stub = build_maintenance_policy_stub(layout, payload)
    atomic_json(layout.reports_dir / "maintenance_policy_stub.json", maintenance_policy_stub)
    atomic_write_text(layout.reports_dir / "maintenance_policy_stub.md", render_maintenance_policy_stub_md(maintenance_policy_stub))
    export_truth_stub = build_export_truth_stub(layout, payload)
    atomic_json(layout.reports_dir / "export_truth_stub.json", export_truth_stub)
    atomic_write_text(layout.reports_dir / "export_truth_stub.md", render_export_truth_stub_md(export_truth_stub))
    device_validation_stub = build_device_validation_stub(layout, payload)
    atomic_json(layout.reports_dir / "device_validation_stub.json", device_validation_stub)
    atomic_write_text(layout.reports_dir / "device_validation_stub.md", render_device_validation_stub_md(device_validation_stub))
    packaging_closure_stub = build_packaging_closure_stub(layout, payload)
    atomic_json(layout.reports_dir / "packaging_closure_stub.json", packaging_closure_stub)
    atomic_write_text(layout.reports_dir / "packaging_closure_stub.md", render_packaging_closure_stub_md(packaging_closure_stub))
    installer_validation_stub = build_installer_validation_stub(layout, payload)
    atomic_json(layout.reports_dir / "installer_validation_stub.json", installer_validation_stub)
    atomic_write_text(layout.reports_dir / "installer_validation_stub.md", render_installer_validation_stub_md(installer_validation_stub))
    benchmark_raw_outputs_stub = build_benchmark_raw_outputs_stub(layout, payload)
    atomic_json(layout.reports_dir / "benchmark_raw_outputs_stub.json", benchmark_raw_outputs_stub)
    atomic_write_text(layout.reports_dir / "benchmark_raw_outputs_stub.md", render_benchmark_raw_outputs_stub_md(benchmark_raw_outputs_stub))
    benchmark_compare_report_stub = build_benchmark_compare_report_stub(layout, payload)
    atomic_json(layout.reports_dir / "benchmark_compare_report_stub.json", benchmark_compare_report_stub)
    atomic_write_text(layout.reports_dir / "benchmark_compare_report_stub.md", render_benchmark_compare_report_stub_md(benchmark_compare_report_stub))
    benchmark_summary_stub = build_benchmark_summary_stub(layout, payload)
    atomic_json(layout.reports_dir / "benchmark_summary_stub.json", benchmark_summary_stub)
    atomic_write_text(layout.reports_dir / "benchmark_summary_stub.md", render_benchmark_summary_stub_md(benchmark_summary_stub))
    benchmark_manifest_stub = build_benchmark_manifest_stub(layout, payload)
    atomic_json(layout.reports_dir / "benchmark_manifest_stub.json", benchmark_manifest_stub)
    atomic_write_text(layout.reports_dir / "benchmark_manifest_stub.md", render_benchmark_manifest_stub_md(benchmark_manifest_stub))
    training_report_stub = build_training_report_stub(layout, payload)
    atomic_json(layout.reports_dir / "training_report_stub.json", training_report_stub)
    atomic_write_text(layout.reports_dir / "training_report_stub.md", render_training_report_stub_md(training_report_stub))
    token_accounting_stub = build_token_accounting_stub(layout, payload)
    atomic_json(layout.reports_dir / "token_accounting_stub.json", token_accounting_stub)
    atomic_write_text(layout.reports_dir / "token_accounting_stub.md", render_token_accounting_stub_md(token_accounting_stub))
    compute_accounting_stub = build_compute_accounting_stub(layout, payload)
    atomic_json(layout.reports_dir / "compute_accounting_stub.json", compute_accounting_stub)
    atomic_write_text(layout.reports_dir / "compute_accounting_stub.md", render_compute_accounting_stub_md(compute_accounting_stub))
    cost_report_stub = build_cost_report_stub(layout, payload)
    atomic_json(layout.reports_dir / "cost_report_stub.json", cost_report_stub)
    atomic_write_text(layout.reports_dir / "cost_report_stub.md", render_cost_report_stub_md(cost_report_stub))
    final_weights_truth_stub = build_final_weights_truth_stub(layout, payload)
    atomic_json(layout.reports_dir / "final_weights_truth_stub.json", final_weights_truth_stub)
    atomic_write_text(layout.reports_dir / "final_weights_truth_stub.md", render_final_weights_truth_stub_md(final_weights_truth_stub))
    best_checkpoint_truth_stub = build_best_checkpoint_truth_stub(layout, payload)
    atomic_json(layout.reports_dir / "best_checkpoint_truth_stub.json", best_checkpoint_truth_stub)
    atomic_write_text(layout.reports_dir / "best_checkpoint_truth_stub.md", render_best_checkpoint_truth_stub_md(best_checkpoint_truth_stub))
    latest_checkpoint_truth_stub = build_latest_checkpoint_truth_stub(layout, payload)
    atomic_json(layout.reports_dir / "latest_checkpoint_truth_stub.json", latest_checkpoint_truth_stub)
    atomic_write_text(layout.reports_dir / "latest_checkpoint_truth_stub.md", render_latest_checkpoint_truth_stub_md(latest_checkpoint_truth_stub))
    trained_artifact_registry_stub = build_trained_artifact_registry_stub(layout, payload)
    atomic_json(layout.reports_dir / "trained_artifact_registry_stub.json", trained_artifact_registry_stub)
    atomic_write_text(layout.reports_dir / "trained_artifact_registry_stub.md", render_trained_artifact_registry_stub_md(trained_artifact_registry_stub))
    core_complete_decision_stub = build_core_complete_decision_stub(layout, payload)
    atomic_json(layout.reports_dir / "core_complete_decision_stub.json", core_complete_decision_stub)
    atomic_write_text(layout.reports_dir / "core_complete_decision_stub.md", render_core_complete_decision_stub_md(core_complete_decision_stub))
    research_continues_stub = build_research_continues_stub(layout, payload)
    atomic_json(layout.reports_dir / "research_continues_stub.json", research_continues_stub)
    atomic_write_text(layout.reports_dir / "research_continues_stub.md", render_research_continues_stub_md(research_continues_stub))
    product_maintenance_only_stub = build_product_maintenance_only_stub(layout, payload)
    atomic_json(layout.reports_dir / "product_maintenance_only_stub.json", product_maintenance_only_stub)
    atomic_write_text(layout.reports_dir / "product_maintenance_only_stub.md", render_product_maintenance_only_stub_md(product_maintenance_only_stub))
    closure_decision_record_stub = build_closure_decision_record_stub(layout, payload)
    atomic_json(layout.reports_dir / "closure_decision_record_stub.json", closure_decision_record_stub)
    atomic_write_text(layout.reports_dir / "closure_decision_record_stub.md", render_closure_decision_record_stub_md(closure_decision_record_stub))
    master_closure_table = build_master_closure_table(layout, payload)
    atomic_json(layout.reports_dir / "master_closure_table.json", master_closure_table)
    atomic_write_text(layout.reports_dir / "master_closure_table.md", render_master_closure_table_md(master_closure_table))
    remaining_core_blockers = build_remaining_core_blockers(layout, payload)
    atomic_json(layout.reports_dir / "remaining_core_blockers.json", remaining_core_blockers)
    atomic_write_text(layout.reports_dir / "remaining_core_blockers.md", render_remaining_core_blockers_md(remaining_core_blockers))
    repo_side_completion_summary = build_repo_side_completion_summary(layout, payload)
    atomic_json(layout.reports_dir / "repo_side_completion_summary.json", repo_side_completion_summary)
    atomic_write_text(layout.reports_dir / "repo_side_completion_summary.md", render_repo_side_completion_summary_md(repo_side_completion_summary))
    readiness_snapshot = build_readiness_snapshot(layout, payload)
    atomic_json(layout.reports_dir / "readiness_snapshot.json", readiness_snapshot)
    atomic_write_text(layout.reports_dir / "readiness_snapshot.md", render_readiness_snapshot_md(readiness_snapshot))
    aggregated_master_table = build_aggregated_master_table(layout, payload)
    atomic_json(layout.reports_dir / "aggregated_master_table.json", aggregated_master_table)
    atomic_write_text(layout.reports_dir / "aggregated_master_table.md", render_aggregated_master_table_md(aggregated_master_table))
    real_remaining_core_work = build_real_remaining_core_work(layout, payload)
    atomic_json(layout.reports_dir / "real_remaining_core_work.json", real_remaining_core_work)
    atomic_write_text(layout.reports_dir / "real_remaining_core_work.md", render_real_remaining_core_work_md(real_remaining_core_work))
    repo_truth_inventory = build_repo_truth_inventory(layout, payload)
    atomic_json(layout.reports_dir / "repo_truth_inventory.json", repo_truth_inventory)
    atomic_write_text(layout.reports_dir / "repo_truth_inventory.md", render_repo_truth_inventory_md(repo_truth_inventory))
    closure_gap_summary = build_closure_gap_summary(layout, payload)
    atomic_json(layout.reports_dir / "closure_gap_summary.json", closure_gap_summary)
    atomic_write_text(layout.reports_dir / "closure_gap_summary.md", render_closure_gap_summary_md(closure_gap_summary))
    project_master_truth_reference = build_project_master_truth_reference(layout, payload)
    atomic_json(layout.reports_dir / "project_master_truth_reference.json", project_master_truth_reference)
    atomic_write_text(layout.reports_dir / "project_master_truth_reference.md", render_project_master_truth_reference_md(project_master_truth_reference))
    project_remaining_real_blockers = build_project_remaining_real_blockers(layout, payload)
    atomic_json(layout.reports_dir / "project_remaining_real_blockers.json", project_remaining_real_blockers)
    atomic_write_text(layout.reports_dir / "project_remaining_real_blockers.md", render_project_remaining_real_blockers_md(project_remaining_real_blockers))
    project_blocker_action_plan = build_project_blocker_action_plan(layout, payload)
    atomic_json(layout.reports_dir / "project_blocker_action_plan.json", project_blocker_action_plan)
    atomic_write_text(layout.reports_dir / "project_blocker_action_plan.md", render_project_blocker_action_plan_md(project_blocker_action_plan))
    project_blocker_dependency_graph = build_project_blocker_dependency_graph(layout, payload)
    atomic_json(layout.reports_dir / "project_blocker_dependency_graph.json", project_blocker_dependency_graph)
    atomic_write_text(layout.reports_dir / "project_blocker_dependency_graph.md", render_project_blocker_dependency_graph_md(project_blocker_dependency_graph))
    project_execution_sequence = build_project_execution_sequence(layout, payload)
    atomic_json(layout.reports_dir / "project_execution_sequence.json", project_execution_sequence)
    atomic_write_text(layout.reports_dir / "project_execution_sequence.md", render_project_execution_sequence_md(project_execution_sequence))
    project_lane_status_board = build_project_lane_status_board(layout, payload)
    atomic_json(layout.reports_dir / "project_lane_status_board.json", project_lane_status_board)
    atomic_write_text(layout.reports_dir / "project_lane_status_board.md", render_project_lane_status_board_md(project_lane_status_board))
    project_closure_phase_plan = build_project_closure_phase_plan(layout, payload)
    atomic_json(layout.reports_dir / "project_closure_phase_plan.json", project_closure_phase_plan)
    atomic_write_text(layout.reports_dir / "project_closure_phase_plan.md", render_project_closure_phase_plan_md(project_closure_phase_plan))
    project_phase_readiness_scoreboard = build_project_phase_readiness_scoreboard(layout, payload)
    atomic_json(layout.reports_dir / "project_phase_readiness_scoreboard.json", project_phase_readiness_scoreboard)
    atomic_write_text(
        layout.reports_dir / "project_phase_readiness_scoreboard.md",
        render_project_phase_readiness_scoreboard_md(project_phase_readiness_scoreboard),
    )
    project_owner_accountability_matrix = build_project_owner_accountability_matrix(layout, payload)
    atomic_json(layout.reports_dir / "project_owner_accountability_matrix.json", project_owner_accountability_matrix)
    atomic_write_text(
        layout.reports_dir / "project_owner_accountability_matrix.md",
        render_project_owner_accountability_matrix_md(project_owner_accountability_matrix),
    )
    project_owner_work_queue = build_project_owner_work_queue(layout, payload)
    atomic_json(layout.reports_dir / "project_owner_work_queue.json", project_owner_work_queue)
    atomic_write_text(
        layout.reports_dir / "project_owner_work_queue.md",
        render_project_owner_work_queue_md(project_owner_work_queue),
    )
    project_critical_path_report = build_project_critical_path_report(layout, payload)
    atomic_json(layout.reports_dir / "project_critical_path_report.json", project_critical_path_report)
    atomic_write_text(
        layout.reports_dir / "project_critical_path_report.md",
        render_project_critical_path_report_md(project_critical_path_report),
    )
    project_owner_next_actions_summary = build_project_owner_next_actions_summary(layout, payload)
    atomic_json(layout.reports_dir / "project_owner_next_actions_summary.json", project_owner_next_actions_summary)
    atomic_write_text(
        layout.reports_dir / "project_owner_next_actions_summary.md",
        render_project_owner_next_actions_summary_md(project_owner_next_actions_summary),
    )
    project_ready_now_board = build_project_ready_now_board(layout, payload)
    atomic_json(layout.reports_dir / "project_ready_now_board.json", project_ready_now_board)
    atomic_write_text(
        layout.reports_dir / "project_ready_now_board.md",
        render_project_ready_now_board_md(project_ready_now_board),
    )
    project_unlock_impact_report = build_project_unlock_impact_report(layout, payload)
    atomic_json(layout.reports_dir / "project_unlock_impact_report.json", project_unlock_impact_report)
    atomic_write_text(
        layout.reports_dir / "project_unlock_impact_report.md",
        render_project_unlock_impact_report_md(project_unlock_impact_report),
    )
    truth_docs_index = build_truth_docs_index(layout, payload)
    atomic_json(layout.reports_dir / "truth_docs_index.json", truth_docs_index)
    atomic_write_text(layout.reports_dir / "truth_docs_index.md", render_truth_docs_index_md(truth_docs_index))
    truth_docs_drift_report = build_truth_docs_drift_report(layout, payload)
    atomic_json(layout.reports_dir / "truth_docs_drift_report.json", truth_docs_drift_report)
    atomic_write_text(layout.reports_dir / "truth_docs_drift_report.md", render_truth_docs_drift_report_md(truth_docs_drift_report))
    generated_truth_consistency_report = build_generated_truth_consistency_report(layout, payload)
    atomic_json(layout.reports_dir / "generated_truth_consistency_report.json", generated_truth_consistency_report)
    atomic_write_text(layout.reports_dir / "generated_truth_consistency_report.md", render_generated_truth_consistency_report_md(generated_truth_consistency_report))
    generated_truth_crosscheck_matrix = build_generated_truth_crosscheck_matrix(layout, payload)
    atomic_json(layout.reports_dir / "generated_truth_crosscheck_matrix.json", generated_truth_crosscheck_matrix)
    atomic_write_text(layout.reports_dir / "generated_truth_crosscheck_matrix.md", render_generated_truth_crosscheck_matrix_md(generated_truth_crosscheck_matrix))


def write_release_evidence_reports(layout: ArtifactLayout, payload: Dict[str, Any]) -> None:
    _write_release_evidence_reports_once(layout, payload)
    truth = build_artifact_truth_matrix(layout, payload)
    atomic_json(layout.reports_dir / "artifact_truth_matrix.json", truth)
    atomic_write_text(layout.reports_dir / "artifact_truth_matrix.md", render_artifact_truth_matrix_md(truth))
    _write_release_evidence_reports_once(layout, payload)
    # The release/evidence chain is self-referential: later generated summaries can
    # change gate readiness, so we refresh the truth matrix once more and run a final
    # pass to converge the derived reports onto the last consistent state.
    truth = build_artifact_truth_matrix(layout, payload)
    atomic_json(layout.reports_dir / "artifact_truth_matrix.json", truth)
    atomic_write_text(layout.reports_dir / "artifact_truth_matrix.md", render_artifact_truth_matrix_md(truth))
    _write_release_evidence_reports_once(layout, payload)


def resolve_archive_password(cfg: Dict[str, Any]) -> str:
    env_name = str(cfg.get("archive_password_env", DEFAULT_ARCHIVE_PASSWORD_ENV)).strip() or DEFAULT_ARCHIVE_PASSWORD_ENV
    return os.environ.get(env_name, "")


def _write_bundle_zip(
    zip_path: Path,
    run_dir: Path,
    password: str,
    require_encryption: bool,
    password_env_name: str,
) -> bool:
    file_paths = [path for path in sorted(run_dir.rglob("*")) if path.is_file() and path != zip_path]
    if password:
        if pyzipper is None:
            raise PackagingError(
                "Encrypted output requested but pyzipper is not installed in this runtime. "
                f"Set {DEFAULT_ENCRYPT_OUTPUT_ENV}=0 or install pyzipper in the delivery build environment."
            )
        with pyzipper.AESZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, encryption=pyzipper.WZ_AES) as zf:
            zf.setpassword(password.encode("utf-8"))
            zf.setencryption(pyzipper.WZ_AES, nbits=256)
            for path in file_paths:
                zf.write(path, arcname=str(path.relative_to(run_dir)))
        return True
    if require_encryption:
        raise PackagingError(
            f"Encrypted output is required but no password was provided in {password_env_name}"
        )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in file_paths:
            zf.write(path, arcname=str(path.relative_to(run_dir)))
    return False


def create_result_bundle(layout: ArtifactLayout, payload: Dict[str, Any]) -> Dict[str, Any]:
    if layout.final_zip_path.exists():
        layout.final_zip_path.unlink()
    if bool(payload["config"].get("zip_outputs", True)):
        password = resolve_archive_password(payload["config"])
        encrypted = _write_bundle_zip(
            zip_path=layout.final_zip_path,
            run_dir=layout.run_dir,
            password=password,
            require_encryption=bool(payload["config"].get("archive_encryption_required", False)),
            password_env_name=str(payload["config"].get("archive_password_env", DEFAULT_ARCHIVE_PASSWORD_ENV)),
        )
        sha = path_sha256(layout.final_zip_path)
        if not bool(payload["config"].get("single_output_only", False)):
            atomic_write_text(layout.final_sha_path, f"{sha}  {layout.final_zip_path.name}\n")
        elif layout.final_sha_path.exists():
            layout.final_sha_path.unlink()
        return {
            "zip_path": str(layout.final_zip_path),
            "sha256_path": "" if bool(payload["config"].get("single_output_only", False)) else str(layout.final_sha_path),
            "sha256": sha,
            "size_bytes": layout.final_zip_path.stat().st_size,
            "encrypted": encrypted,
        }
    return {"zip_path": "", "sha256_path": "", "sha256": "", "size_bytes": 0, "encrypted": False}


def cleanup_after_bundle_if_needed(cfg: Dict[str, Any], layout: ArtifactLayout, logger: Optional[JSONLLogger] = None) -> None:
    if not bool(cfg.get("cleanup_after_bundle", False)):
        return
    if not layout.run_dir.exists():
        return
    try:
        shutil.rmtree(layout.run_dir)
        if logger is not None:
            logger.write("bundle_cleanup", {"status": "removed_run_dir", "path": str(layout.run_dir)})
    except Exception as exc:
        if logger is not None:
            logger.write("bundle_cleanup", {"status": "failed", "path": str(layout.run_dir), "error": str(exc)})


def not_run_evaluation(reason: str) -> Dict[str, Any]:
    return {
        "status": "not_run",
        "reason": reason,
        "batches_evaluated": 0,
        "examples_evaluated": 0,
        "metrics": {},
        "per_phase": {},
        "router_entropy_mean": 0.0,
    }


def not_run_legality_report(reason: str) -> Dict[str, Any]:
    return {
        "status": "not_run",
        "reason": reason,
        "checked_examples": 0,
        "raw_top1_is_legal_rate": 0.0,
        "raw_topk_contains_legal_rate": 0.0,
        "masked_policy_accuracy": 0.0,
        "per_phase": {},
        "example_rows": [],
        "note": "Legality scoring is skipped in verify mode because verify is runtime-only.",
    }


def not_run_demo_replay(reason: str) -> Dict[str, Any]:
    return {
        "status": "not_run",
        "reason": reason,
        "demonstration_only": True,
        "note": "Replay is skipped in verify mode because verify is runtime-only.",
        "games": [],
    }


def not_run_arena_session(reason: str) -> Dict[str, Any]:
    return {
        "status": "not_run",
        "reason": reason,
        "interactive_only": True,
        "result": "*",
        "plies_played": 0,
        "transcript": [],
        "note": "Arena mode was not executed in this run.",
    }


def schedule_self_delete_if_needed(cfg: Dict[str, Any], success: bool, final_zip: Optional[Path]) -> None:
    if not success:
        return
    share_mode = bool(cfg.get("share_mode", False)) or os.environ.get(DEFAULT_SHARE_MODE_ENV, "0") == "1"
    enable_self_delete = bool(cfg.get("enable_self_delete", False)) or os.environ.get(DEFAULT_SELF_DELETE_ENV, "0") == "1"
    if not share_mode or not enable_self_delete:
        return
    target_value = str(cfg.get("self_delete_target", "")).strip() or os.environ.get(DEFAULT_SELF_DELETE_TARGET_ENV, "").strip()
    if not target_value:
        return
    script_path = Path(__file__).resolve()
    target_path = Path(target_value).expanduser().resolve()
    if target_path == script_path:
        return
    if target_path.suffix.lower() not in {".py", ".pyw"}:
        return
    if not target_path.exists():
        return
    if platform.system() == "Windows":
        cmd_path = target_path.with_suffix(".cleanup.cmd")
        cmd_path.write_text(
            "@echo off\n"
            "setlocal\n"
            "ping 127.0.0.1 -n 3 > nul\n"
            f"del /f /q \"{target_path}\" > nul 2>&1\n"
            f"del /f /q \"{cmd_path}\" > nul 2>&1\n",
            encoding="utf-8",
        )
        subprocess.Popen(["cmd.exe", "/c", str(cmd_path)], creationflags=0x08000000)
    else:  # pragma: no cover - share mode primarily targets Windows
        zip_label = final_zip.name if final_zip is not None else "artifact.zip"
        subprocess.Popen(
            ["bash", "-lc", f"sleep 2; rm -f '{target_path}' >/dev/null 2>&1 # {zip_label}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def verify_forward_pass(
    model: ChessPolicyValueNet,
    examples: Sequence[ChessExample],
    device: torch.device,
) -> Dict[str, Any]:
    if not examples:
        return {"status": "empty", "checked": 0}
    sample = examples[: min(4, len(examples))]
    batch = collate_examples(sample)
    batch = batch_to_device(batch, device)
    logits, value, aux_loss, _ = model(batch["piece_ids"], batch["meta_ids"])
    return {
        "status": "ok",
        "checked": len(sample),
        "logits_shape": list(logits.shape),
        "value_shape": list(value.shape),
        "aux_loss": float(aux_loss.detach().item()),
    }


def prepare_model_and_optimizer(cfg: Dict[str, Any], layout: ArtifactLayout, logger: JSONLLogger) -> Tuple[ChessPolicyValueNet, torch.optim.Optimizer, Dict[str, Any]]:
    device = pick_device(cfg)
    parity_report = assert_mirror_surface_integrity(cfg)
    atomic_json(layout.reports_dir / "mirror_parity_report.json", parity_report)
    model = ChessPolicyValueNet(cfg, len(MOVE_VOCAB)).to(device)
    optimizer = build_optimizer(model, cfg)
    model, compile_report = maybe_enable_compile(model, cfg, logger)
    atomic_json(layout.reports_dir / "compile_report.json", compile_report)
    return model, optimizer, compile_report


def collect_verify_examples(cfg: Dict[str, Any], layout: ArtifactLayout, logger: JSONLLogger) -> Tuple[List[ChessExample], Dict[str, Any]]:
    verify_cfg = dict(cfg)
    verify_cfg["offline_seed_only"] = True
    verify_cfg["auto_download_enabled"] = False
    examples, provenance = maybe_collect_dataset(verify_cfg, layout, logger)
    provenance["mode"] = "verify_embedded_seed"
    provenance["verification_only"] = True
    return examples, provenance


def package_existing_run(
    cfg: Dict[str, Any],
    layout: ArtifactLayout,
    logger: JSONLLogger,
) -> Dict[str, Any]:
    summary_path = layout.reports_dir / "run_summary.json"
    payload: Dict[str, Any]
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        checkpoint = torch.load(Path(str(cfg["resume_from"])), map_location="cpu", weights_only=False)
        checkpoint_metrics = dict(checkpoint.get("metrics", {}))
        payload = {
            "script_version": checkpoint.get("script_version", SCRIPT_VERSION),
            "run_id": layout.run_id,
            "config": checkpoint.get("config", cfg),
            "execution_status": ExecutionStatus.RAN.value,
            "evaluation_status": EvaluationStatus.UNEVALUATED.value,
            "rating_claim_status": RatingClaimStatus.NO_CLAIM.value,
            "claim_status": RatingClaimStatus.NO_CLAIM.value,
            "rating_target_proxy_threshold": int(cfg["rating_target_proxy_threshold"]),
            "dataset_provenance": {"mode": "package_only", "data_stats": {}, "sampling_strategy": "not_rebuilt"},
            "holdout_validation": checkpoint_metrics,
            "locked_test": {},
            "legality_report": {},
            "stockfish": {"status": "not_run", "reason": "package_only"},
            "curated_position_suite": {"status": "not_run", "reason": "package_only"},
            "selfplay_report": {"status": "not_run", "reason": "package_only"},
            "tournament_report": {"status": "not_run", "reason": "package_only"},
            "replay_buffer_report": {"status": "not_run", "reason": "package_only"},
            "curated_position_manifest": build_curated_position_manifest(cfg),
            "synthetic_teaching_corpus": build_synthetic_teaching_corpus(cfg),
            "training_augmentation": {"enabled": False, "reason": "package_only"},
            "training_summary": {
                "steps_completed": int(checkpoint.get("step", 0)),
                "best_val_loss": checkpoint.get("best_val_loss"),
                "package_only": True,
            },
            "compile_report": {"status": "not_run", "reason": "package_only"},
            "forward_verify": {"status": "not_run", "reason": "package_only"},
            "best_checkpoint": str(cfg["resume_from"]),
            "latest_checkpoint": str(cfg["resume_from"]),
            "output_root": str(layout.run_dir),
            "notes": {
                "package_only": True,
                "replay_is_demo_only": True,
                "internal_proxy_only": True,
                "what_this_proves": "Repackaging of an existing run directory without rebuilding the dataset or retraining.",
                "what_this_does_not_prove": "A new training/evaluation run.",
            },
        }
    payload["config"] = dict(payload.get("config", cfg))
    payload["config"]["mode"] = "package"
    payload["repackaged_at_utc"] = utc_now()
    payload["repackaged_from_checkpoint"] = str(cfg["resume_from"])
    payload["mirror_parity"] = build_mirror_parity_report(payload["config"])
    payload["feature_flags"] = build_feature_flag_report(payload["config"])
    payload["logging"] = logger.observability_report()
    manifest = build_artifact_manifest(layout)
    payload["artifact_manifest"] = manifest
    bundle = create_result_bundle(layout, payload)
    payload["bundle"] = bundle
    atomic_json(summary_path, payload)
    atomic_write_text(layout.reports_dir / "run_summary.md", render_run_summary_md(payload))
    atomic_json(layout.reports_dir / "feature_flag_report.json", payload["feature_flags"])
    atomic_write_text(layout.reports_dir / "feature_flag_report.md", render_feature_flag_report_md(payload["feature_flags"]))
    atomic_json(layout.reports_dir / "logging_contract.json", logger.contract())
    atomic_json(layout.reports_dir / "observability_report.json", logger.observability_report())
    write_closure_manifests(layout, payload)
    write_release_evidence_reports(layout, payload)
    logger.write("package_only_complete", {"checkpoint": str(cfg["resume_from"]), **bundle})
    cleanup_after_bundle_if_needed(cfg, layout, logger)
    return payload


def run_pipeline(
    cfg: Dict[str, Any],
    layout: Optional[ArtifactLayout] = None,
    logger: Optional[JSONLLogger] = None,
) -> Dict[str, Any]:
    deterministic_seed(int(cfg["seed"]), strict=bool(cfg.get("determinism_strict", True)))
    layout = layout or prepare_layout(cfg)
    logger = logger or JSONLLogger(layout.logs_dir / "run_log.jsonl")
    logger.bind_context(
        run_id=layout.run_id,
        mode=str(cfg["mode"]),
        profile=str(cfg["profile"]),
        artifact_root=str(layout.run_dir),
    )
    logger.write("config_resolved", {"config": cfg, "layout": {"run_dir": str(layout.run_dir), "logs_dir": str(layout.logs_dir)}})
    logger.write("run_start", {"script_version": SCRIPT_VERSION, "config": cfg})

    env_info = env_snapshot(cfg)
    dependency_lock = collect_dependency_lock()
    curated_position_manifest = build_curated_position_manifest(cfg)
    synthetic_teaching_corpus = build_synthetic_teaching_corpus(cfg)
    feature_flag_report = build_feature_flag_report(cfg)
    atomic_json(layout.reports_dir / "environment_snapshot.json", env_info)
    atomic_json(layout.reports_dir / "dependency_lock.json", dependency_lock)
    atomic_json(layout.reports_dir / "resolved_config.json", cfg)
    atomic_json(layout.reports_dir / "feature_flag_report.json", feature_flag_report)
    atomic_write_text(layout.reports_dir / "feature_flag_report.md", render_feature_flag_report_md(feature_flag_report))
    atomic_json(layout.reports_dir / "curated_position_manifest.json", curated_position_manifest)
    atomic_write_text(layout.reports_dir / "curated_position_manifest.md", render_curated_position_manifest_md(curated_position_manifest))
    atomic_json(layout.reports_dir / "synthetic_teaching_corpus.json", synthetic_teaching_corpus)
    atomic_write_text(layout.reports_dir / "synthetic_teaching_corpus.md", render_synthetic_teaching_corpus_md(synthetic_teaching_corpus))

    if cfg["mode"] == "package":
        return package_existing_run(cfg, layout, logger)
    if cfg["mode"] == "arena":
        model, optimizer, compile_report = prepare_model_and_optimizer(cfg, layout, logger)
        del optimizer
        device = pick_device(cfg)
        checkpoint_path: Optional[Path] = None
        if str(cfg.get("resume_from", "")).strip():
            checkpoint_path = Path(str(cfg["resume_from"]))
            resume_state = load_checkpoint(checkpoint_path, model, optimizer=None, restore_optimizer=False)
            logger.write("arena_checkpoint_loaded", {"checkpoint": str(checkpoint_path), "step": resume_state.step})
        arena_session = play_human_vs_model_arena(model, cfg, device, logger)
        benchmark_protocol = build_benchmark_protocol(cfg, detect_stockfish_path(cfg))
        holdout_validation = not_run_evaluation("arena_mode_interactive_only")
        locked_test = not_run_evaluation("arena_mode_interactive_only")
        legality_report = not_run_legality_report("arena_mode_interactive_only")
        demo_replay = not_run_demo_replay("arena_mode_interactive_only")
        stockfish_report = {"status": "not_run", "reason": "arena_mode_interactive_only"}
        curated_position_suite_report = not_run_curated_position_eval("arena_mode_interactive_only")
        selfplay_report = not_run_selfplay_report("arena_mode_interactive_only")
        tournament_report = not_run_tournament_report("arena_mode_interactive_only")
        replay_buffer_report = not_run_replay_buffer_report("arena_mode_interactive_only")
        atomic_json(layout.reports_dir / "model_replay.json", demo_replay)
        atomic_json(layout.reports_dir / "stockfish_match_report.json", stockfish_report)
        model_card = build_model_card(model, cfg, checkpoint_path)
        data_card = {
            "script_version": SCRIPT_VERSION,
            "dataset_provenance": {"mode": "arena_only", "data_stats": {}, "sampling_strategy": "not_used"},
            "split_manifest": {"status": "not_run", "reason": "arena_mode_interactive_only"},
            "curated_position_manifest": curated_position_manifest,
            "notes": {
                "train_val_test_split": "Arena mode does not build train/val/test splits.",
                "eval_signal": "Arena mode is interactive and uses the current in-memory model state.",
                "sampling_strategy": "not_used",
            },
        }
        eval_card = build_eval_card(
            cfg,
            holdout_validation,
            locked_test,
            legality_report,
            stockfish_report,
            curated_position_suite_report,
            selfplay_report,
            tournament_report,
            replay_buffer_report,
        )
        payload = {
            "script_version": SCRIPT_VERSION,
            "run_id": layout.run_id,
            "config": cfg,
            "execution_status": ExecutionStatus.RAN.value,
            "evaluation_status": EvaluationStatus.UNEVALUATED.value,
            "rating_claim_status": RatingClaimStatus.NO_CLAIM.value,
            "claim_status": RatingClaimStatus.NO_CLAIM.value,
            "rating_target_proxy_threshold": int(cfg["rating_target_proxy_threshold"]),
            "dataset_provenance": data_card["dataset_provenance"],
            "holdout_validation": holdout_validation,
            "locked_test": locked_test,
            "legality_report": legality_report,
            "stockfish": stockfish_report,
            "curated_position_suite": curated_position_suite_report,
            "selfplay_report": selfplay_report,
            "tournament_report": tournament_report,
            "replay_buffer_report": replay_buffer_report,
            "curated_position_manifest": curated_position_manifest,
            "synthetic_teaching_corpus": synthetic_teaching_corpus,
            "training_augmentation": {"enabled": False, "reason": "arena_mode_interactive_only"},
            "training_summary": {
                "steps_completed": 0,
                "best_val_loss": None,
                "arena_only": True,
                "checkpoint_loaded": checkpoint_path is not None,
            },
            "feature_flags": feature_flag_report,
            "compile_report": compile_report,
            "forward_verify": {"status": "not_run", "reason": "arena_mode_interactive_only"},
            "best_checkpoint": str(checkpoint_path) if checkpoint_path is not None else "",
            "latest_checkpoint": str(checkpoint_path) if checkpoint_path is not None else "",
            "output_root": str(layout.run_dir),
            "arena_session": arena_session,
            "notes": {
                "replay_is_demo_only": True,
                "internal_proxy_only": True,
                "what_this_proves": "Interactive human-vs-model arena execution with legal move masking and artifact packaging.",
                "what_this_does_not_prove": "Training quality, holdout metrics, or externally verified strength.",
            },
        }
        write_cards_and_reports(
            layout=layout,
            cfg=cfg,
            payload=payload,
            data_card=data_card,
            model_card=model_card,
            eval_card=eval_card,
            benchmark_protocol=benchmark_protocol,
            dependency_lock=dependency_lock,
            env_info=env_info,
            curve_rows=[],
            logger=logger,
        )
        build_artifact_manifest(layout)
        bundle = create_result_bundle(layout, payload)
        payload["bundle"] = bundle
        payload["logging"] = logger.observability_report()
        atomic_json(layout.reports_dir / "run_summary.json", payload)
        atomic_write_text(layout.reports_dir / "run_summary.md", render_run_summary_md(payload))
        write_closure_manifests(layout, payload)
        write_release_evidence_reports(layout, payload)
        logger.write("run_complete", {"execution_status": payload["execution_status"], "rating_claim_status": payload["rating_claim_status"], **bundle})
        cleanup_after_bundle_if_needed(cfg, layout, logger)
        return payload

    if cfg["mode"] == "verify":
        examples, provenance = collect_verify_examples(cfg, layout, logger)
    else:
        examples, provenance = maybe_collect_dataset(cfg, layout, logger)
    splits, split_manifest = split_examples_by_game(examples, cfg)
    train_examples = list(splits["train"])
    val_examples = splits["val"]
    test_examples = splits["locked_test"]
    curated_training_examples, curated_training_manifest = build_curated_training_examples(cfg)
    train_examples.extend(curated_training_examples)
    split_manifest["training_augmentation"] = curated_training_manifest
    split_manifest["counts"]["examples_train_before_augmentation"] = len(splits["train"])
    split_manifest["counts"]["examples_train_after_augmentation"] = len(train_examples)
    atomic_json(layout.reports_dir / "split_manifest.json", split_manifest)

    model, optimizer, compile_report = prepare_model_and_optimizer(cfg, layout, logger)
    device = pick_device(cfg)

    if not train_examples:
        raise DatasetEmptyError("Training split is empty after game-level split")
    if cfg["mode"] != "verify" and not val_examples:
        raise DatasetEmptyError("Validation split is empty after game-level split")

    forward_verify = verify_forward_pass(model, train_examples, device)
    atomic_json(layout.reports_dir / "verify_forward_pass.json", forward_verify)

    curve_rows: List[Dict[str, Any]] = []
    latest_ckpt = layout.checkpoints_dir / "latest.pt"
    best_ckpt = layout.checkpoints_dir / "best_by_val_loss.pt"
    training_summary: Dict[str, Any] = {"steps_completed": 0, "best_val_loss": float("inf")}

    if cfg["mode"] in {"train", "resume"}:
        resume_state: Optional[ResumeState] = None
        if str(cfg.get("resume_from", "")).strip():
            resume_path = Path(str(cfg["resume_from"]))
            resume_state = load_checkpoint(resume_path, model, optimizer=optimizer, restore_optimizer=True)
            logger.write("resume_loaded", {"checkpoint": str(resume_path), "step": resume_state.step, "best_val_loss": resume_state.best_val_loss})
        training_summary, curve_rows, latest_ckpt, best_ckpt = training_loop(
            model=model,
            optimizer=optimizer,
            train_examples=train_examples,
            val_examples=val_examples,
            cfg=cfg,
            layout=layout,
            logger=logger,
            start_step=resume_state.step if resume_state is not None else 0,
            best_val_loss=resume_state.best_val_loss if resume_state is not None else float("inf"),
        )
    elif cfg["mode"] == "benchmark":
        resume_path = Path(str(cfg["resume_from"]))
        resume_state = load_checkpoint(resume_path, model, optimizer=None, restore_optimizer=False)
        latest_ckpt = resume_path
        best_ckpt = resume_path
        logger.write("benchmark_checkpoint_loaded", {"checkpoint": str(resume_path), "step": resume_state.step})
    elif cfg["mode"] == "verify":
        verify_checkpoint = ""
        if str(cfg.get("resume_from", "")).strip():
            resume_path = Path(str(cfg["resume_from"]))
            resume_state = load_checkpoint(resume_path, model, optimizer=None, restore_optimizer=False)
            latest_ckpt = resume_path
            best_ckpt = resume_path
            verify_checkpoint = str(resume_path)
            logger.write("verify_checkpoint_loaded", {"checkpoint": str(resume_path), "step": resume_state.step})
        training_summary = {
            "steps_completed": 0,
            "best_val_loss": None,
            "verify_only": True,
            "verify_scope": "runtime_pipeline_only",
            "verify_checkpoint": verify_checkpoint,
        }
    else:
        raise ConfigValidationError(f"Unhandled mode: {cfg['mode']}")

    if cfg["mode"] != "verify" and best_ckpt.exists():
        load_checkpoint(best_ckpt, model, optimizer=None, restore_optimizer=False)
        logger.write("best_checkpoint_reloaded", {"checkpoint": str(best_ckpt)})

    if cfg["mode"] == "verify":
        holdout_validation = not_run_evaluation("verify_mode_runtime_only")
        locked_test = not_run_evaluation("verify_mode_runtime_only")
        legality_report = not_run_legality_report("verify_mode_runtime_only")
        demo_replay = not_run_demo_replay("verify_mode_runtime_only")
        selfplay_report = not_run_selfplay_report("verify_mode_runtime_only")
        tournament_report = not_run_tournament_report("verify_mode_runtime_only")
        replay_buffer_report = not_run_replay_buffer_report("verify_mode_runtime_only")
    else:
        val_loader = make_loader(val_examples, batch_size=int(cfg["eval_batch_size"]), shuffle=False, num_workers=0, seed=int(cfg["seed"]) + 123)
        test_loader = make_loader(test_examples if test_examples else val_examples, batch_size=int(cfg["eval_batch_size"]), shuffle=False, num_workers=0, seed=int(cfg["seed"]) + 124)
        holdout_validation = evaluate_model(model, val_loader, device, cfg, max_batches=0)
        locked_test = evaluate_model(model, test_loader, device, cfg, max_batches=0)
        model.eval()
        legality_report = run_legality_report(model, val_examples, device, cfg)
        demo_replay = generate_demo_replay(model, cfg, device)
        selfplay_report = generate_selfplay_report(model, cfg, device, layout)
        tournament_report = play_inference_mode_tournament(model, cfg, device, layout)
        replay_buffer_report = build_replay_buffer_report(selfplay_report, cfg)
    atomic_json(layout.reports_dir / "model_replay.json", demo_replay)
    atomic_json(layout.reports_dir / "selfplay_report.json", selfplay_report)
    atomic_write_text(layout.reports_dir / "selfplay_report.md", render_selfplay_report_md(selfplay_report))
    atomic_json(layout.reports_dir / "inference_mode_tournament_report.json", tournament_report)
    atomic_write_text(layout.reports_dir / "inference_mode_tournament_report.md", render_tournament_report_md(tournament_report))
    atomic_json(layout.reports_dir / "replay_buffer_manifest.json", replay_buffer_report)
    atomic_write_text(layout.reports_dir / "replay_buffer_manifest.md", render_replay_buffer_report_md(replay_buffer_report))

    benchmark_protocol = build_benchmark_protocol(cfg, detect_stockfish_path(cfg))
    stockfish_report = {"status": "not_run", "reason": "mode_disabled"}
    if cfg["mode"] == "verify":
        stockfish_report = {"status": "not_run", "reason": "verify_mode_runtime_only"}
    elif cfg["mode"] in {"train", "resume", "benchmark"}:
        model.eval()
        stockfish_report = play_stockfish_gauntlet(model, cfg, device, layout, logger)
    atomic_json(layout.reports_dir / "stockfish_match_report.json", stockfish_report)
    if cfg["mode"] == "verify":
        curated_position_suite_report = not_run_curated_position_eval("verify_mode_runtime_only")
    elif cfg["mode"] in {"train", "resume", "benchmark"}:
        model.eval()
        curated_position_suite_report = evaluate_curated_position_suites(model, cfg, device)
    else:
        curated_position_suite_report = not_run_curated_position_eval("mode_disabled")
    atomic_json(layout.reports_dir / "curated_position_suite_report.json", curated_position_suite_report)
    atomic_write_text(layout.reports_dir / "curated_position_suite_report.md", render_curated_position_suite_report_md(curated_position_suite_report))

    execution_status, evaluation_status, rating_claim_status = determine_statuses(cfg, stockfish_report)
    claim_status = rating_claim_status.value

    model_card = build_model_card(model, cfg, best_ckpt if best_ckpt.exists() else latest_ckpt)
    data_card = {
        "script_version": SCRIPT_VERSION,
        "dataset_provenance": provenance,
        "split_manifest": split_manifest,
        "curated_position_manifest": curated_position_manifest,
        "notes": {
            "train_val_test_split": "Game-level split with locked test set.",
            "eval_signal": "Eval-tagged positions are preferred when present; otherwise discounted result targets are used.",
            "sampling_strategy": provenance.get("sampling_strategy", "unknown"),
        },
    }
    eval_card = build_eval_card(
        cfg,
        holdout_validation,
        locked_test,
        legality_report,
        stockfish_report,
        curated_position_suite_report,
        selfplay_report,
        tournament_report,
        replay_buffer_report,
    )

    payload = {
        "script_version": SCRIPT_VERSION,
        "run_id": layout.run_id,
        "config": cfg,
        "execution_status": execution_status.value,
        "evaluation_status": evaluation_status.value,
        "rating_claim_status": rating_claim_status.value,
        "claim_status": claim_status,
        "rating_target_proxy_threshold": int(cfg["rating_target_proxy_threshold"]),
        "dataset_provenance": provenance,
        "holdout_validation": holdout_validation,
        "locked_test": locked_test,
        "legality_report": legality_report,
        "stockfish": stockfish_report,
        "curated_position_suite": curated_position_suite_report,
        "selfplay_report": selfplay_report,
        "tournament_report": tournament_report,
        "replay_buffer_report": replay_buffer_report,
        "curated_position_manifest": curated_position_manifest,
        "synthetic_teaching_corpus": synthetic_teaching_corpus,
        "training_augmentation": curated_training_manifest,
        "training_summary": training_summary,
        "feature_flags": feature_flag_report,
        "compile_report": compile_report,
        "forward_verify": forward_verify,
        "best_checkpoint": str(best_ckpt) if best_ckpt.exists() else "",
        "latest_checkpoint": str(latest_ckpt) if latest_ckpt.exists() else "",
        "output_root": str(layout.run_dir),
        "notes": {
            "replay_is_demo_only": True,
            "internal_proxy_only": True,
            "what_this_proves": (
                "Verify mode proves runtime/data-pipeline integrity and artifact packaging."
                if cfg["mode"] == "verify"
                else "Single-machine bounded chess data ingestion, supervised training, legality-safe inference, and artifact packaging."
            ),
            "what_this_does_not_prove": (
                "Verify mode does not provide holdout strength metrics, legality scoring, replay evidence, or rating evidence."
                if cfg["mode"] == "verify"
                else "External rating verification or frontier general-purpose LLM capability."
            ),
        },
    }

    write_cards_and_reports(
        layout=layout,
        cfg=cfg,
        payload=payload,
        data_card=data_card,
        model_card=model_card,
        eval_card=eval_card,
        benchmark_protocol=benchmark_protocol,
        dependency_lock=dependency_lock,
        env_info=env_info,
        curve_rows=curve_rows,
        logger=logger,
    )
    build_artifact_manifest(layout)
    bundle = create_result_bundle(layout, payload)
    payload["bundle"] = bundle
    payload["logging"] = logger.observability_report()
    atomic_json(layout.reports_dir / "run_summary.json", payload)
    atomic_write_text(layout.reports_dir / "run_summary.md", render_run_summary_md(payload))
    write_closure_manifests(layout, payload)
    write_release_evidence_reports(layout, payload)
    logger.write("run_complete", {"execution_status": payload["execution_status"], "rating_claim_status": payload["rating_claim_status"], **bundle})
    cleanup_after_bundle_if_needed(cfg, layout, logger)
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MertFormer Chess RTX 5080 onefile")
    parser.add_argument("--mode", default=RUN_CONFIG["mode"], choices=["train", "verify", "benchmark", "package", "resume", "arena"])
    parser.add_argument("--profile", default=RUN_CONFIG["profile"], choices=list(RUN_PROFILES.keys()))
    parser.add_argument("--baseline", default=RUN_CONFIG["baseline"], choices=["dense", "moe", "moe_adapter"])
    parser.add_argument("--feature-bundle", choices=list(FEATURE_BUNDLES.keys()), help="Named feature bundle overlay for advanced architecture surfaces.")
    parser.add_argument("--enable-features", help="Comma-separated feature flags to force-enable on top of the selected profile/bundle.")
    parser.add_argument("--disable-features", help="Comma-separated feature flags to force-disable on top of the selected profile/bundle.")
    parser.add_argument("--resume-from", help="Load a checkpoint for resume/benchmark/package modes. Verify and arena modes can optionally load one without retraining.")
    parser.add_argument("--artifact-root", help="Override artifact root.")
    parser.add_argument("--stockfish-path", help="Optional Stockfish executable override.")
    parser.add_argument("--no-download", action="store_true", help="Do not attempt network download; use cache or fail.")
    parser.add_argument("--allow-install", action="store_true", help="Allow runtime dependency installation if packages are missing.")
    parser.add_argument("--share-mode", action="store_true", help="Enable share-facing behavior. Self-delete remains opt-in.")
    parser.add_argument("--enable-self-delete", action="store_true", help="Delete only an explicit shared script copy after success. Requires share mode and --self-delete-target.")
    parser.add_argument("--self-delete-target", help="Explicit shared script-copy path eligible for opt-in self-delete.")
    parser.add_argument("--offline-seed-only", action="store_true", help="Skip network and use embedded seed PGN only.")
    parser.add_argument("--test-mode", action="store_true", help="Force tiny embedded-seed smoke mode.")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-wall-hours", type=float)
    parser.add_argument("--batch-size", type=int)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    global LAST_FINAL_ZIP, LAST_RUNTIME_CFG, LAST_RUN_SUCCESS
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    LAST_FINAL_ZIP = None
    LAST_RUNTIME_CFG = None
    LAST_RUN_SUCCESS = False
    logger_for_guard: Optional[JSONLLogger] = None
    layout: Optional[ArtifactLayout] = None
    try:
        cfg = resolve_runtime_config(args, RUN_CONFIG)
        LAST_RUNTIME_CFG = cfg
        layout = prepare_layout(cfg)
        logger_for_guard = JSONLLogger(
            layout.logs_dir / "run_log.jsonl",
            run_id=layout.run_id,
            mode=str(cfg["mode"]),
            profile=str(cfg["profile"]),
            artifact_root=str(layout.run_dir),
        )
        with WindowsExecutionGuard(logger_for_guard, enabled=True):
            payload = run_pipeline(cfg, layout=layout, logger=logger_for_guard)
        LAST_RUN_SUCCESS = True
        if payload.get("bundle", {}).get("zip_path"):
            LAST_FINAL_ZIP = Path(str(payload["bundle"]["zip_path"]))
        logger_for_guard.finalize(
            "completed",
            extra={
                "execution_status": payload["execution_status"],
                "rating_claim_status": payload["rating_claim_status"],
                "bundle_path": payload.get("bundle", {}).get("zip_path", ""),
            },
        )
        print(
            json.dumps(
                {
                    "status": "completed",
                    "execution_status": payload["execution_status"],
                    "evaluation_status": payload["evaluation_status"],
                    "rating_claim_status": payload["rating_claim_status"],
                    "bundle": payload.get("bundle", {}),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    except ChessOnefileError as exc:
        if logger_for_guard is not None:
            logger_for_guard.write_exception(
                "fatal_exception",
                exc,
                extra_payload={"scope": "controlled_failure", "layout_ready": layout is not None},
            )
        err = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    except Exception as exc:  # pragma: no cover - last-resort crash boundary
        if logger_for_guard is not None:
            logger_for_guard.write_exception(
                "fatal_exception",
                exc,
                extra_payload={"scope": "unhandled_exception", "layout_ready": layout is not None},
            )
        err = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    if logger_for_guard is not None:
        logger_for_guard.finalize(
            "failed",
            extra={"error_type": err["error_type"], "error": err["error"]},
        )
    desktop = detect_desktop_dir()
    err_path = desktop / f"{RESULT_ZIP_PREFIX}_FAILED_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    atomic_write_text(err_path, json.dumps(err, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(err, indent=2, ensure_ascii=False), file=sys.stderr)
    return 1
    

if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = main()
    finally:
        with contextlib.suppress(Exception):
            cfg_for_delete = LAST_RUNTIME_CFG
            if cfg_for_delete is None:
                parsed_args = build_argument_parser().parse_known_args()[0]
                cfg_for_delete = resolve_runtime_config(parsed_args, RUN_CONFIG)
            schedule_self_delete_if_needed(cfg_for_delete, exit_code == 0 and LAST_RUN_SUCCESS, LAST_FINAL_ZIP)
    raise SystemExit(exit_code)
