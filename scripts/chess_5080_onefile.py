#!/usr/bin/env python3
"""
MertFormer Chess RTX 5080 Onefile
---------------------------------
Single-file Windows-friendly chess proof lane for a single RTX 5080 desktop.

Goals:
- one-click PyCharm execution after dependencies are present
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
import contextlib
import csv
import ctypes
import enum
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import platform
import random
import re
import shutil
import struct
import subprocess
import sys
import textwrap
import time
import traceback
import urllib.error
import urllib.request
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
DEFAULT_ALLOW_INSTALL_ENV = "MERTFORMER_CHESS_ALLOW_INSTALL"
DEFAULT_SKIP_BOOTSTRAP_ENV = "MERTFORMER_CHESS_SKIP_BOOTSTRAP"
DEFAULT_SHARE_MODE_ENV = "MERTFORMER_CHESS_SHARE_MODE"
DEFAULT_SELF_DELETE_ENV = "MERTFORMER_CHESS_SELF_DELETE"
DEFAULT_TEST_MODE_ENV = "MERTFORMER_CHESS_TEST_MODE"
DEFAULT_TORCH_INDEX_ENV = "MERTFORMER_CHESS_TORCH_INDEX_URL"
DEFAULT_ARCHIVE_PASSWORD_ENV = "MERTFORMER_CHESS_ARCHIVE_PASSWORD"
DEFAULT_ENCRYPT_OUTPUT_ENV = "MERTFORMER_CHESS_ENCRYPT_OUTPUT"
DEFAULT_ENCRYPTION_REQUIRED_ENV = "MERTFORMER_CHESS_ENCRYPTION_REQUIRED"
DEFAULT_CLEANUP_AFTER_BUNDLE_ENV = "MERTFORMER_CHESS_CLEANUP_AFTER_BUNDLE"
DEFAULT_SINGLE_OUTPUT_ENV = "MERTFORMER_CHESS_SINGLE_OUTPUT"

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
    "seed": 42,
    "device": "auto",
    "artifact_root": "~/Desktop",
    "cache_root": "~/Desktop/mertformer_chess_cache",
    "result_prefix": RESULT_ZIP_PREFIX,
    "redact_paths": True,
    "allow_install": False,
    "share_mode": False,
    "enable_self_delete": False,
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
    "num_layers": 8,
    "num_heads": 8,
    "dropout": 0.10,
    "use_moe": False,
    "moe_top_k": 2,
    "num_experts": 4,
    "use_bitlinear": False,
    "use_liquid_adapter": False,
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
    "stockfish_path": "",
    "stockfish_ladder": [
        {"label": "sf_skill4_nodes20k", "games": 12, "skill": 4, "nodes": 20000, "anchor_elo_proxy": 1100},
        {"label": "sf_skill8_nodes50k", "games": 12, "skill": 8, "nodes": 50000, "anchor_elo_proxy": 1400},
        {"label": "sf_skill12_nodes100k", "games": 16, "skill": 12, "nodes": 100000, "anchor_elo_proxy": 1700},
    ],
    "rating_target_proxy_threshold": 1600,
    "claim_min_benchmark_games": 40,
    "zip_outputs": True,
    "lichess_urls": DEFAULT_LICHESS_URLS,
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
        "stockfish_ladder": [
            {"label": "sf_skill4_nodes20k", "games": 14, "skill": 4, "nodes": 20000, "anchor_elo_proxy": 1100},
            {"label": "sf_skill8_nodes50k", "games": 14, "skill": 8, "nodes": 50000, "anchor_elo_proxy": 1400},
            {"label": "sf_skill12_nodes100k", "games": 14, "skill": 12, "nodes": 100000, "anchor_elo_proxy": 1700},
            {"label": "sf_skill16_nodes200k", "games": 14, "skill": 16, "nodes": 200000, "anchor_elo_proxy": 1900},
        ],
        "claim_min_benchmark_games": 56,
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
        "num_layers": 2,
        "num_heads": 4,
        "num_experts": 2,
        "use_moe": False,
        "use_bitlinear": False,
        "use_liquid_adapter": False,
        "compile_policy": "off",
        "use_bf16": False,
        "curriculum_enabled": False,
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
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, payload: Dict[str, Any]) -> None:
        row = {"ts_utc": utc_now(), "kind": kind, **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zstandard as zstd
try:
    import chess
    import chess.engine
    import chess.pgn
except Exception:  # pragma: no cover - import guarded by bootstrap in __main__
    chess = None  # type: ignore
try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore
try:
    import pyzipper
except Exception:  # pragma: no cover - optional runtime dependency
    pyzipper = None  # type: ignore


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
        "allow_install": bool(cfg.get("allow_install", False)),
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
        merged["use_liquid_adapter"] = False
        merged["use_bitlinear"] = False
    elif baseline == "moe":
        merged["use_moe"] = True
        merged["use_liquid_adapter"] = False
    elif baseline == "moe_adapter":
        merged["use_moe"] = True
        merged["use_liquid_adapter"] = True
    return merged


def resolve_runtime_config(args: argparse.Namespace, base_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(base_cfg or RUN_CONFIG)
    profile = str(getattr(args, "profile", cfg["profile"]))
    cfg = apply_profile(cfg, profile)
    baseline = str(getattr(args, "baseline", cfg["baseline"]))
    cfg = apply_baseline(cfg, baseline)

    cfg["mode"] = str(getattr(args, "mode", cfg["mode"]))
    validate_enum_choice(cfg["mode"], ["train", "verify", "benchmark", "package", "resume"], "mode")

    if getattr(args, "artifact_root", None):
        cfg["artifact_root"] = args.artifact_root
    if getattr(args, "stockfish_path", None):
        cfg["stockfish_path"] = args.stockfish_path
    if getattr(args, "resume_from", None):
        cfg["resume_from"] = args.resume_from
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
    if os.environ.get(DEFAULT_ENCRYPT_OUTPUT_ENV, "0") == "1":
        cfg["encrypt_output"] = True
    if os.environ.get(DEFAULT_ENCRYPTION_REQUIRED_ENV, "0") == "1":
        cfg["archive_encryption_required"] = True
    if os.environ.get(DEFAULT_SINGLE_OUTPUT_ENV, "0") == "1":
        cfg["single_output_only"] = True
    if os.environ.get(DEFAULT_CLEANUP_AFTER_BUNDLE_ENV, "0") == "1":
        cfg["cleanup_after_bundle"] = True

    cfg["artifact_root"] = str(Path(str(cfg["artifact_root"])).expanduser())
    cfg["cache_root"] = str(Path(str(cfg["cache_root"])).expanduser())
    cfg["resume_from"] = str(Path(str(cfg.get("resume_from", ""))).expanduser()) if str(cfg.get("resume_from", "")) else ""

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
    validate_enum_choice(str(cfg["mode"]), ["train", "verify", "benchmark", "package", "resume"], "mode")
    validate_enum_choice(str(cfg["profile"]), list(RUN_PROFILES.keys()), "profile")
    validate_enum_choice(str(cfg["baseline"]), ["dense", "moe", "moe_adapter"], "baseline")
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
    if bool(cfg.get("cleanup_after_bundle", False)) and not bool(cfg.get("zip_outputs", True)):
        raise ConfigValidationError("cleanup_after_bundle requires zip_outputs=True")
    if str(cfg["mode"]) in {"resume", "benchmark", "package"} and not str(cfg.get("resume_from", "")).strip():
        raise ConfigValidationError(f"mode={cfg['mode']} requires --resume-from")
    if str(cfg["mode"]) == "verify" and int(cfg["max_steps"]) == 0:
        return


class BitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, enabled: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.enabled = enabled
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def _quantize(self, weight: torch.Tensor) -> torch.Tensor:
        scale = torch.sqrt((weight ** 2).mean(dim=1, keepdim=True)).clamp(min=1e-5)
        normalized = weight / scale
        quantized = torch.round(normalized).clamp(-1.0, 1.0) * scale
        return weight + (quantized - weight).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._quantize(self.weight) if self.enabled else self.weight
        return F.linear(x, weight, self.bias)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float, use_bitlinear: bool = False):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        linear = BitLinear if use_bitlinear else nn.Linear
        kwargs = {"enabled": True} if use_bitlinear else {}
        self.q_proj = linear(hidden_size, hidden_size, bias=False, **kwargs)  # type: ignore[arg-type]
        self.k_proj = linear(hidden_size, hidden_size, bias=False, **kwargs)  # type: ignore[arg-type]
        self.v_proj = linear(hidden_size, hidden_size, bias=False, **kwargs)  # type: ignore[arg-type]
        self.o_proj = linear(hidden_size, hidden_size, bias=False, **kwargs)  # type: ignore[arg-type]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        probs = F.softmax(attn, dim=-1)
        probs = self.dropout(probs)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
        return self.o_proj(out)


class DenseFeedForward(nn.Module):
    def __init__(self, hidden_size: int, dropout: float, use_bitlinear: bool = False):
        super().__init__()
        inner = hidden_size * 4
        linear = BitLinear if use_bitlinear else nn.Linear
        kwargs = {"enabled": True} if use_bitlinear else {}
        self.fc1 = linear(hidden_size, inner, **kwargs)  # type: ignore[arg-type]
        self.fc2 = linear(inner, hidden_size, **kwargs)  # type: ignore[arg-type]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class SparseMoE(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int, dropout: float, use_bitlinear: bool = False):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = max(1, min(top_k, num_experts))
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            DenseFeedForward(hidden_size, dropout, use_bitlinear=use_bitlinear) for _ in range(num_experts)
        )
        self.last_router_stats: Dict[str, Any] = {}

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        router_logits = self.router(x)
        weights = F.softmax(router_logits, dim=-1)
        top_weights, top_indices = torch.topk(weights, k=self.top_k, dim=-1)
        normalized_top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        flat_x = x.reshape(-1, x.size(-1))
        flat_indices = top_indices.reshape(-1, self.top_k)
        flat_weights = normalized_top_weights.reshape(-1, self.top_k)
        out = torch.zeros_like(flat_x)
        usage = torch.zeros(self.num_experts, dtype=torch.float32, device=x.device)

        for expert_idx, expert in enumerate(self.experts):
            contrib = flat_indices == expert_idx
            if not bool(contrib.any()):
                continue
            token_mask = contrib.any(dim=-1)
            token_positions = token_mask.nonzero(as_tuple=False).squeeze(-1)
            expert_in = flat_x[token_positions]
            expert_out = expert(expert_in)
            weight_rows = flat_weights[token_positions] * contrib[token_positions].float()
            token_weights = weight_rows.sum(dim=-1, keepdim=True)
            out[token_positions] += expert_out * token_weights
            usage[expert_idx] = float(token_mask.float().sum().item())

        load = weights.mean(dim=(0, 1))
        aux = ((load - (1.0 / self.num_experts)) ** 2).mean()
        entropy = -(weights * weights.clamp_min(1e-9).log()).sum(dim=-1).mean()
        usage_total = float(usage.sum().item())
        usage_pct = (usage / usage.sum().clamp_min(1.0)).tolist()
        dead_experts = [idx for idx, value in enumerate(usage_pct) if value < 1e-4]
        self.last_router_stats = {
            "router_entropy": float(entropy.detach().item()),
            "tokens_per_expert": [int(round(v)) for v in usage.tolist()],
            "tokens_per_expert_fraction": [float(v) for v in usage_pct],
            "dead_experts": dead_experts,
            "active_tokens": int(usage_total),
            "top_k": int(self.top_k),
        }
        return out.reshape_as(x), aux, self.last_router_stats


class GatedResidualAdapter(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(x))
        delta = torch.tanh(self.proj(x))
        return residual + gate * delta


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        num_layers: int,
        use_moe: bool,
        num_experts: int,
        moe_top_k: int,
        use_bitlinear: bool,
        use_gated_adapter: bool,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads, dropout, use_bitlinear=use_bitlinear)
        self.use_moe = use_moe
        if use_moe:
            self.ff = SparseMoE(hidden_size, num_experts, moe_top_k, dropout, use_bitlinear=use_bitlinear)
        else:
            self.ff = DenseFeedForward(hidden_size, dropout, use_bitlinear=use_bitlinear)
        self.use_gated_adapter = use_gated_adapter
        self.adapter = GatedResidualAdapter(hidden_size) if use_gated_adapter else None
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = (2 * num_layers) ** -0.5
        self.last_aux = 0.0
        self.last_router_stats: Dict[str, Any] = {}

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        residual = x
        x = residual + self.dropout(self.attn(self.norm1(x))) * self.residual_scale
        aux = x.new_tensor(0.0)
        router_stats: Dict[str, Any] = {}
        ff_in = self.norm2(x)
        if self.use_moe:
            ff_out, aux, router_stats = self.ff(ff_in)
        else:
            ff_out = self.ff(ff_in)
        if self.use_gated_adapter and self.adapter is not None:
            x = self.adapter(ff_out, x)
        else:
            x = x + self.dropout(ff_out) * self.residual_scale
        self.last_aux = float(aux.detach().item())
        self.last_router_stats = router_stats
        return x, aux, router_stats


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
        hidden = int(cfg["hidden_size"])
        layers = int(cfg["num_layers"])
        heads = int(cfg["num_heads"])
        dropout = float(cfg["dropout"])
        use_bitlinear = bool(cfg.get("use_bitlinear", False))
        use_moe = bool(cfg.get("use_moe", False))
        num_experts = int(cfg.get("num_experts", 4))
        moe_top_k = int(cfg.get("moe_top_k", 2))
        use_gated_adapter = bool(cfg.get("use_liquid_adapter", False))

        self.piece_embed = nn.Embedding(13, hidden)
        self.square_embed = nn.Embedding(64, hidden)
        self.meta_type_embed = nn.Embedding(len(self.META_CARDINALITIES), hidden)
        self.meta_value_embeds = nn.ModuleList(nn.Embedding(card, hidden) for card in self.META_CARDINALITIES)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                hidden_size=hidden,
                num_heads=heads,
                dropout=dropout,
                num_layers=layers,
                use_moe=use_moe and (layer_idx % 2 == 1),
                num_experts=num_experts,
                moe_top_k=moe_top_k,
                use_bitlinear=use_bitlinear,
                use_gated_adapter=use_gated_adapter,
            )
            for layer_idx in range(layers)
        )
        self.norm = RMSNorm(hidden)
        linear = BitLinear if use_bitlinear else nn.Linear
        kwargs = {"enabled": True} if use_bitlinear else {}
        self.policy_head = linear(hidden, vocab_size, **kwargs)  # type: ignore[arg-type]
        self.value_head = linear(hidden, 1, **kwargs)  # type: ignore[arg-type]
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
        aux_loss = x.new_tensor(0.0)
        router_reports: Dict[str, Any] = {}
        for block_idx, block in enumerate(self.blocks):
            x, aux, router_stats = block(x)
            aux_loss = aux_loss + aux
            if router_stats:
                router_reports[f"block_{block_idx}"] = router_stats
        x = self.norm(x)
        pooled = self.dropout(x.mean(dim=1))
        policy_logits = self.policy_head(pooled)
        value = torch.tanh(self.value_head(pooled)).squeeze(-1)
        return policy_logits, value, aux_loss, router_reports

    def parameter_report(self) -> Dict[str, Any]:
        total = sum(param.numel() for param in self.parameters())
        trainable = sum(param.numel() for param in self.parameters() if param.requires_grad)
        return {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "policy_head_type": "pooled_global_move_classifier",
        }


class ChessExampleDataset(torch.utils.data.Dataset):
    def __init__(self, examples: Sequence[ChessExample]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> ChessExample:
        return self.examples[idx]


PHASE_NAMES = {0: "opening", 1: "middlegame", 2: "endgame"}


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


def select_ply_indices(moves: Sequence[Tuple[chess.Move, str]], limit: int, prefer_eval_positions: bool) -> List[int]:
    total_plies = len(moves)
    if total_plies <= 0 or limit <= 0:
        return []
    if total_plies <= limit:
        return list(range(total_plies))
    picks: List[int] = []
    eval_indices = [idx for idx, (_, comment) in enumerate(moves) if comment_has_eval_tag(comment)] if prefer_eval_positions else []
    evenly_spaced = sorted({
        int(round(position))
        for position in np.linspace(0, total_plies - 1, num=min(total_plies, max(limit * 2, limit + 2)))
    })
    for idx in eval_indices + evenly_spaced:
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
                            chunk = response.read(1024 * 1024)
                            if not chunk:
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
            selected_indices = select_ply_indices(moves, max_positions_per_game, bool(cfg.get("prefer_eval_positions", True)))
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
    logits, value_pred, aux_loss, router_reports = model(batch["piece_ids"], batch["meta_ids"])
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
    return loss, metrics, router_reports


def merge_metric_sums(sums: Dict[str, float], metrics: Dict[str, float]) -> None:
    for key, value in metrics.items():
        sums[key] = sums.get(key, 0.0) + float(value)


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
        _, metrics, router_reports = compute_loss(model, batch, cfg)
        merge_metric_sums(sums, metrics)
        batch_count += 1
        example_count += int(batch["piece_ids"].size(0))
        for router_stats in router_reports.values():
            if "router_entropy" in router_stats:
                router_entropy_values.append(float(router_stats["router_entropy"]))
        logits, value_pred, aux_loss, _ = model(batch["piece_ids"], batch["meta_ids"])
        masked_logits = logits.masked_fill(~batch["legal_mask"], -1e9)
        for phase_value in (0, 1, 2):
            phase_mask = batch["phases"] == phase_value
            if not bool(phase_mask.any()):
                continue
            phase_name = PHASE_NAMES[int(phase_value)]
            phase_counts[phase_name] += int(phase_mask.sum().item())
            phase_logits = logits[phase_mask]
            phase_masked = masked_logits[phase_mask]
            phase_legal = batch["legal_mask"][phase_mask]
            phase_targets = batch["move_targets"][phase_mask]
            phase_metrics = compute_prediction_metrics(phase_logits, phase_masked, phase_legal, phase_targets)
            merge_metric_sums(phase_sums[phase_name], phase_metrics)
    model.train()
    overall = summarize_metric_sums(sums, batch_count)
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
    root.mkdir(parents=True, exist_ok=True)
    if existing_run_dir is not None:
        run_dir = existing_run_dir
        run_id = existing_run_dir.name.removeprefix(f"{DELIVERY_PREFIX}_") or datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = root / f"{DELIVERY_PREFIX}_{run_id}"
    logs_dir = run_dir / "logs"
    reports_dir = run_dir / "reports"
    checkpoints_dir = run_dir / "checkpoints"
    export_dir = run_dir / "exports"
    benchmark_dir = run_dir / "benchmarks"
    final_zip = desktop / f"{cfg['result_prefix']}_{run_id}.zip"
    final_sha = desktop / f"{cfg['result_prefix']}_{run_id}.zip.sha256"
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
    stage_loader: Optional[torch.utils.data.DataLoader] = None
    stage_iterator: Optional[Iterator[Dict[str, torch.Tensor]]] = None
    stage_epoch = 0

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
                stage_data[stage_idx][1],
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
            logger.write("eval_step", {"step": global_step, **val_eval["metrics"]})
            current_val_loss = float(val_eval["metrics"].get("loss", 0.0))
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                save_checkpoint(model, optimizer, best_ckpt, global_step, cfg, val_eval, best_val_loss)
                shutil.copy2(best_ckpt, compatibility_best)
        if global_step - last_checkpoint_at >= int(cfg["checkpoint_interval"]):
            save_checkpoint(model, optimizer, latest_ckpt, global_step, cfg, {"type": "latest", "step": global_step}, best_val_loss)
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


def detect_stockfish_path(cfg: Dict[str, Any]) -> Optional[str]:
    explicit = str(cfg.get("stockfish_path", "") or "").strip()
    if explicit and Path(explicit).exists():
        return explicit
    for candidate in (
        shutil.which("stockfish"),
        shutil.which("stockfish.exe"),
        str(Path.home() / "Desktop" / "stockfish" / "stockfish.exe"),
        str(Path.home() / "Downloads" / "stockfish" / "stockfish.exe"),
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def choose_move_trace(model: ChessPolicyValueNet, board: chess.Board, device: torch.device, topk: int = 5) -> Dict[str, Any]:
    legal_ids = legal_move_ids(board)
    if not legal_ids:
        raise RuntimeError("No legal moves available for choose_move_trace")
    piece_ids, meta_ids = encode_board_state(board, legal_move_count=len(legal_ids))
    piece = torch.tensor([piece_ids], dtype=torch.long, device=device)
    meta = torch.tensor([meta_ids], dtype=torch.long, device=device)
    start = time.time()
    logits, value, _, _ = model(piece, meta)
    latency_ms = (time.time() - start) * 1000.0
    logits = logits[0]
    raw_topk_ids = torch.topk(logits, k=min(topk, logits.size(-1)), dim=-1).indices.tolist()
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[legal_ids] = True
    masked_logits = logits.masked_fill(~mask, -1e9)
    masked_topk_ids = torch.topk(masked_logits, k=min(topk, masked_logits.size(-1)), dim=-1).indices.tolist()
    best_id = int(masked_logits.argmax().item())
    move_uci = ID_TO_MOVE[best_id]
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        raise RuntimeError(f"Masked policy selected illegal move: {move_uci}")
    raw_top1_id = int(logits.argmax().item())
    return {
        "move": move_uci,
        "value": float(value[0].item()),
        "latency_ms": round(latency_ms, 4),
        "raw_top1_is_legal": raw_top1_id in legal_ids,
        "raw_topk": [ID_TO_MOVE[idx] for idx in raw_topk_ids],
        "masked_topk": [ID_TO_MOVE[idx] for idx in masked_topk_ids],
    }


def run_legality_report(model: ChessPolicyValueNet, examples: Sequence[ChessExample], device: torch.device, cfg: Dict[str, Any]) -> Dict[str, Any]:
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
    stderr = math.sqrt(max(score_rate * (1.0 - score_rate), 0.0) / games)
    low = max(0.0, score_rate - 1.96 * stderr)
    high = min(1.0, score_rate + 1.96 * stderr)
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
        "openings": OPENING_SEEDS,
        "ladder": cfg.get("stockfish_ladder", []),
        "rating_note": "This protocol emits elo_proxy_internal only. It does not emit a plain ELO claim.",
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


def play_stockfish_gauntlet(
    model: ChessPolicyValueNet,
    cfg: Dict[str, Any],
    device: torch.device,
    layout: ArtifactLayout,
    logger: JSONLLogger,
) -> Dict[str, Any]:
    engine_path = detect_stockfish_path(cfg)
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
    }

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
                        trace = choose_move_trace(model, board, device)
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
        with contextlib.suppress(Exception):
            engine.quit()


def generate_demo_replay(model: ChessPolicyValueNet, cfg: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    games: List[Dict[str, Any]] = []
    max_games = int(cfg.get("sample_replay_games", 3))
    max_plies = int(cfg.get("sample_replay_max_plies", 24))
    for game_idx in range(max_games):
        board = chess.Board()
        opening = OPENING_SEEDS[game_idx % len(OPENING_SEEDS)]
        for move_uci in opening[: min(2, len(opening))]:
            board.push(chess.Move.from_uci(move_uci))
        moves: List[Dict[str, Any]] = []
        while not board.is_game_over() and len(moves) < max_plies:
            trace = choose_move_trace(model, board, device)
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


def determine_statuses(cfg: Dict[str, Any], benchmark_report: Dict[str, Any]) -> Tuple[ExecutionStatus, EvaluationStatus, RatingClaimStatus]:
    execution_status = ExecutionStatus.RAN
    evaluation_status = EvaluationStatus.INTERNALLY_MEASURED
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


def build_model_card(model: ChessPolicyValueNet, cfg: Dict[str, Any], checkpoint_path: Optional[Path]) -> Dict[str, Any]:
    report = model.parameter_report()
    checkpoint_size = checkpoint_path.stat().st_size if checkpoint_path and checkpoint_path.exists() else 0
    report.update(
        {
            "script_version": SCRIPT_VERSION,
            "baseline": cfg.get("baseline", "dense"),
            "hidden_size": int(cfg["hidden_size"]),
            "num_layers": int(cfg["num_layers"]),
            "num_heads": int(cfg["num_heads"]),
            "use_moe": bool(cfg.get("use_moe", False)),
            "use_bitlinear": bool(cfg.get("use_bitlinear", False)),
            "use_gated_residual_adapter": bool(cfg.get("use_liquid_adapter", False)),
            "moe_top_k": int(cfg.get("moe_top_k", 2)),
            "checkpoint_size_bytes": int(checkpoint_size),
            "move_vocab_size": len(MOVE_VOCAB),
            "move_vocab_hash": MOVE_VOCAB_HASH,
            "architecture_notes": [
                "Board attention is intentionally non-causal: the model sees the whole board state at once.",
                "Policy head is a pooled global move-classifier over a fixed UCI vocabulary.",
                "GatedResidualAdapter is a gated residual adapter, not a full CfC liquid cell.",
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
) -> Dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "holdout_validation": val_eval,
        "locked_test": test_eval,
        "raw_vs_masked_policy_metrics": legality_report,
        "benchmark_protocol": "internal_stockfish_gauntlet_v2",
        "benchmark_result": benchmark_report,
        "rating_note": "Strength outputs are internal-only unless externally verified.",
    }


def render_run_summary_md(payload: Dict[str, Any]) -> str:
    lines = [
        "# MertFormer Chess Run Summary",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Mode: `{payload['config']['mode']}`",
        f"- Profile: `{payload['config']['profile']}`",
        f"- Baseline: `{payload['config']['baseline']}`",
        f"- Execution status: `{payload['execution_status']}`",
        f"- Evaluation status: `{payload['evaluation_status']}`",
        f"- Rating claim status: `{payload['rating_claim_status']}`",
        f"- Proxy threshold target: `{payload['rating_target_proxy_threshold']}`",
        "",
        "## What This Proves",
        "- The onefile can ingest bounded Lichess data, build a legal-move-safe supervised chess dataset, train a policy/value model, and package measurable artifacts.",
        "- Holdout metrics, legality metrics, and optional internal Stockfish gauntlet results were generated from this run.",
        "",
        "## What This Does Not Prove",
        "- This run does not prove frontier general-purpose LLM capability.",
        "- Replay/demo output is not strength proof.",
        "- Any `elo_proxy_internal` value is a proxy, not an externally verified rating.",
        "",
        "## Key Metrics",
        f"- Validation masked policy accuracy: `{payload['holdout_validation']['metrics'].get('masked_policy_accuracy', 0.0):.4f}`",
        f"- Locked test masked policy accuracy: `{payload['locked_test']['metrics'].get('masked_policy_accuracy', 0.0):.4f}`",
        f"- Raw top-1 legality: `{payload['legality_report'].get('raw_top1_is_legal_rate', 0.0):.4f}`",
        f"- Raw top-k contains legal: `{payload['legality_report'].get('raw_topk_contains_legal_rate', 0.0):.4f}`",
    ]
    benchmark = payload.get("stockfish", {})
    if benchmark.get("status") == "completed":
        lines.extend(
            [
                f"- Internal gauntlet games: `{benchmark.get('games_total', 0)}`",
                f"- Internal elo proxy: `{benchmark.get('elo_proxy_internal')}`",
            ]
        )
    else:
        lines.append(f"- Internal gauntlet: `{benchmark.get('status', 'not_run')}`")
    bundle = payload.get("bundle", {})
    lines.extend(
        [
            "",
            "## Bundle",
            f"- Output root: `{payload['output_root']}`",
            f"- Final zip: `{bundle.get('zip_path', '')}`",
            f"- Final sha256: `{bundle.get('sha256', '')}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def render_proof_scope_md() -> str:
    return textwrap.dedent(
        """
        # Proof Scope

        ## What This Run Proves
        - A consumer-class single-machine pipeline can ingest bounded chess data, build a legality-safe dataset, train a supervised policy/value network, and emit reproducible artifact packs.
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
    if bool(cfg.get("offline_seed_only", False)):
        cmd.append("--offline-seed-only")
    if bool(cfg.get("share_mode", False)):
        cmd.append("--share-mode")
    if bool(cfg.get("enable_self_delete", False)):
        cmd.append("--enable-self-delete")
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
        - This onefile defaults to proof-safe behavior: no self-delete unless explicitly enabled.
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
) -> None:
    reports = layout.reports_dir
    atomic_json(reports / "run_summary.json", payload)
    atomic_write_text(reports / "run_summary.md", render_run_summary_md(payload))
    atomic_json(reports / "data_card.json", data_card)
    atomic_json(reports / "model_card.json", model_card)
    atomic_json(reports / "eval_card.json", eval_card)
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
    atomic_write_text(reports / "PROOF_SCOPE.md", render_proof_scope_md())
    atomic_write_text(reports / "REPRO.md", render_repro_md(cfg, layout))
    atomic_write_text(reports / "THIRD_PARTY_DATA_LICENSES.txt", render_third_party_licenses())
    write_curve_csv(reports / "training_curve.csv", curve_rows)
    write_curve_png(reports / "training_curve.png", curve_rows)


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


def schedule_self_delete_if_needed(cfg: Dict[str, Any], success: bool, final_zip: Optional[Path]) -> None:
    if not success:
        return
    share_mode = bool(cfg.get("share_mode", False)) or os.environ.get(DEFAULT_SHARE_MODE_ENV, "0") == "1"
    enable_self_delete = bool(cfg.get("enable_self_delete", False)) or os.environ.get(DEFAULT_SELF_DELETE_ENV, "0") == "1"
    if not share_mode or not enable_self_delete:
        return
    script_path = Path(__file__).resolve()
    if script_path.suffix.lower() not in {".py", ".pyw"}:
        return
    if platform.system() == "Windows":
        cmd_path = script_path.with_suffix(".cleanup.cmd")
        cmd_path.write_text(
            "@echo off\n"
            "setlocal\n"
            "ping 127.0.0.1 -n 3 > nul\n"
            f"del /f /q \"{script_path}\" > nul 2>&1\n"
            f"del /f /q \"{cmd_path}\" > nul 2>&1\n",
            encoding="utf-8",
        )
        subprocess.Popen(["cmd.exe", "/c", str(cmd_path)], creationflags=0x08000000)
    else:  # pragma: no cover - share mode primarily targets Windows
        zip_label = final_zip.name if final_zip is not None else "artifact.zip"
        subprocess.Popen(
            ["bash", "-lc", f"sleep 2; rm -f '{script_path}' >/dev/null 2>&1 # {zip_label}"],
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
    manifest = build_artifact_manifest(layout)
    payload["artifact_manifest"] = manifest
    bundle = create_result_bundle(layout, payload)
    payload["bundle"] = bundle
    atomic_json(summary_path, payload)
    atomic_write_text(layout.reports_dir / "run_summary.md", render_run_summary_md(payload))
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
    logger.write("run_start", {"script_version": SCRIPT_VERSION, "config": cfg})

    env_info = env_snapshot(cfg)
    dependency_lock = collect_dependency_lock()
    atomic_json(layout.reports_dir / "environment_snapshot.json", env_info)
    atomic_json(layout.reports_dir / "dependency_lock.json", dependency_lock)
    atomic_json(layout.reports_dir / "resolved_config.json", cfg)

    if cfg["mode"] == "package":
        return package_existing_run(cfg, layout, logger)

    if cfg["mode"] == "verify":
        examples, provenance = collect_verify_examples(cfg, layout, logger)
    else:
        examples, provenance = maybe_collect_dataset(cfg, layout, logger)
    splits, split_manifest = split_examples_by_game(examples, cfg)
    atomic_json(layout.reports_dir / "split_manifest.json", split_manifest)

    model, optimizer, compile_report = prepare_model_and_optimizer(cfg, layout, logger)
    device = pick_device(cfg)
    train_examples = splits["train"]
    val_examples = splits["val"]
    test_examples = splits["locked_test"]

    if not train_examples:
        raise DatasetEmptyError("Training split is empty after game-level split")
    if not val_examples:
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
        training_summary = {"steps_completed": 0, "best_val_loss": None, "verify_only": True}
    else:
        raise ConfigValidationError(f"Unhandled mode: {cfg['mode']}")

    if best_ckpt.exists():
        load_checkpoint(best_ckpt, model, optimizer=None, restore_optimizer=False)
        logger.write("best_checkpoint_reloaded", {"checkpoint": str(best_ckpt)})

    val_loader = make_loader(val_examples, batch_size=int(cfg["eval_batch_size"]), shuffle=False, num_workers=0, seed=int(cfg["seed"]) + 123)
    test_loader = make_loader(test_examples if test_examples else val_examples, batch_size=int(cfg["eval_batch_size"]), shuffle=False, num_workers=0, seed=int(cfg["seed"]) + 124)
    holdout_validation = evaluate_model(model, val_loader, device, cfg, max_batches=0)
    locked_test = evaluate_model(model, test_loader, device, cfg, max_batches=0)
    legality_report = run_legality_report(model, val_examples, device, cfg)
    demo_replay = generate_demo_replay(model, cfg, device)
    atomic_json(layout.reports_dir / "model_replay.json", demo_replay)

    benchmark_protocol = build_benchmark_protocol(cfg, detect_stockfish_path(cfg))
    stockfish_report = {"status": "not_run", "reason": "mode_disabled"}
    if cfg["mode"] in {"train", "resume", "benchmark"}:
        stockfish_report = play_stockfish_gauntlet(model, cfg, device, layout, logger)
    atomic_json(layout.reports_dir / "stockfish_match_report.json", stockfish_report)

    execution_status, evaluation_status, rating_claim_status = determine_statuses(cfg, stockfish_report)
    claim_status = rating_claim_status.value

    model_card = build_model_card(model, cfg, best_ckpt if best_ckpt.exists() else latest_ckpt)
    data_card = {
        "script_version": SCRIPT_VERSION,
        "dataset_provenance": provenance,
        "split_manifest": split_manifest,
        "notes": {
            "train_val_test_split": "Game-level split with locked test set.",
            "eval_signal": "Eval-tagged positions are preferred when present; otherwise discounted result targets are used.",
            "sampling_strategy": provenance.get("sampling_strategy", "unknown"),
        },
    }
    eval_card = build_eval_card(cfg, holdout_validation, locked_test, legality_report, stockfish_report)

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
        "training_summary": training_summary,
        "compile_report": compile_report,
        "forward_verify": forward_verify,
        "best_checkpoint": str(best_ckpt) if best_ckpt.exists() else "",
        "latest_checkpoint": str(latest_ckpt) if latest_ckpt.exists() else "",
        "output_root": str(layout.run_dir),
        "notes": {
            "replay_is_demo_only": True,
            "internal_proxy_only": True,
            "what_this_proves": "Single-machine bounded chess data ingestion, supervised training, legality-safe inference, and artifact packaging.",
            "what_this_does_not_prove": "External rating verification or frontier general-purpose LLM capability.",
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
    )
    build_artifact_manifest(layout)
    bundle = create_result_bundle(layout, payload)
    payload["bundle"] = bundle
    atomic_json(layout.reports_dir / "run_summary.json", payload)
    atomic_write_text(layout.reports_dir / "run_summary.md", render_run_summary_md(payload))
    logger.write("run_complete", {"execution_status": payload["execution_status"], "rating_claim_status": payload["rating_claim_status"], **bundle})
    cleanup_after_bundle_if_needed(cfg, layout, logger)
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MertFormer Chess RTX 5080 onefile")
    parser.add_argument("--mode", default=RUN_CONFIG["mode"], choices=["train", "verify", "benchmark", "package", "resume"])
    parser.add_argument("--profile", default=RUN_CONFIG["profile"], choices=list(RUN_PROFILES.keys()))
    parser.add_argument("--baseline", default=RUN_CONFIG["baseline"], choices=["dense", "moe", "moe_adapter"])
    parser.add_argument("--resume-from", help="Load a checkpoint for resume/benchmark/package modes.")
    parser.add_argument("--artifact-root", help="Override artifact root.")
    parser.add_argument("--stockfish-path", help="Optional Stockfish executable override.")
    parser.add_argument("--no-download", action="store_true", help="Do not attempt network download; use cache or fail.")
    parser.add_argument("--allow-install", action="store_true", help="Allow runtime dependency installation if packages are missing.")
    parser.add_argument("--share-mode", action="store_true", help="Enable share-facing behavior. Self-delete remains opt-in.")
    parser.add_argument("--enable-self-delete", action="store_true", help="Delete only the shared script copy after success. Requires share mode.")
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
    try:
        cfg = resolve_runtime_config(args, RUN_CONFIG)
        LAST_RUNTIME_CFG = cfg
        layout = prepare_layout(cfg)
        logger_for_guard: Optional[JSONLLogger] = JSONLLogger(layout.logs_dir / "run_log.jsonl")
        with WindowsExecutionGuard(logger_for_guard, enabled=True):
            payload = run_pipeline(cfg, layout=layout, logger=logger_for_guard)
        LAST_RUN_SUCCESS = True
        if payload.get("bundle", {}).get("zip_path"):
            LAST_FINAL_ZIP = Path(str(payload["bundle"]["zip_path"]))
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
        err = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    except Exception as exc:  # pragma: no cover - last-resort crash boundary
        err = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
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
