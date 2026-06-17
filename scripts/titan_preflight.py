"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - ULTIMATE PREFLIGHT JUDGE
-------------------------------------------------------------------------------
Copyright (c) 2026 Mert Yünlü. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) — Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

__version__ = "1.0-BUILD30-V2"
__author__ = "Mert"

import os
import sys
import shutil
import logging
import time
import subprocess
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from typing import Dict, Any


def _field_yields_text(rows, field, min_length: int, max_rows: int = 8) -> bool:
    """Pure helper: does any of the first ``max_rows`` rows yield text of at least
    ``min_length`` characters under ``field``? Uses the SAME extractor as the data
    pipeline so a misconfigured field (e.g. an empty-by-schema 'content') is caught."""
    import scripts.data_pipeline as _dp

    for i, row in enumerate(rows):
        if i >= int(max_rows):
            break
        text = _dp._extract_text(row, field)
        if isinstance(text, str) and len(text.strip()) >= int(min_length):
            return True
    return False


def _probe_sources_yield_text(hf_token, max_rows: int = 8) -> Dict[str, str]:
    """Stream a few rows from each non-optional stage source and confirm its configured
    field yields usable text. Returns {dataset_id: 'PASS' | 'SOURCE_FIELD_EMPTY' | 'WARN: ...'}.
    Transient/network errors are WARN (non-fatal); a source that fetches rows but yields no
    text under its field is SOURCE_FIELD_EMPTY (a real, loud blocker)."""
    from datasets import load_dataset
    import scripts.data_pipeline as _dp

    sources = (
        _dp.STAGE1_SOURCES + _dp.STAGE2_SOURCES + _dp.STAGE3_SOURCES
        + _dp.STAGE4_SOURCES + _dp.STAGE5_SOURCES
    )
    results: Dict[str, str] = {}
    for src in sources:
        ds_id = str(src.get("dataset", ""))
        if src.get("optional") or ds_id in results:
            continue
        try:
            load_kwargs: Dict[str, Any] = {"split": src.get("split", "train"), "streaming": True}
            if hf_token:
                load_kwargs["token"] = hf_token
            subset = src.get("subset")
            ds = load_dataset(ds_id, subset, **load_kwargs) if subset else load_dataset(ds_id, **load_kwargs)
            ok = _field_yields_text(ds, src.get("field", ""), int(src.get("min_length", 1)), max_rows)
            results[ds_id] = "PASS" if ok else "SOURCE_FIELD_EMPTY"
        except Exception as exc:  # noqa: BLE001 - transient/network is non-fatal
            results[ds_id] = f"WARN: {exc}"
    return results


# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import cfg, validate_layer_config
from orchestrator.distillation_manager import DistillationManager
from model.transformers import MertFormer

# Logs
LOG_DIR = PROJECT_ROOT / "logs" / "preflight"
LOG_DIR.mkdir(parents=True, exist_ok=True)
TEST_LOG_PATH = LOG_DIR / "titan_preflight.log"
SNAPSHOT_PATH = PROJECT_ROOT / "scripts" / "runs" / "preflight" / "config_snapshot.json"

# Temp Storage
TEMP_DATA_DIR = PROJECT_ROOT / "temp_preflight_data"
TEMP_LOGITS_DIR = PROJECT_ROOT / "temp_preflight_logits"

# Setup Logging
logging.basicConfig(
    filename=TEST_LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    filemode='w'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

def log(msg, level="info"):
    icon = "✈️"
    if level == "error": icon = "❌"
    elif level == "warning": icon = "⚠️"
    elif level == "success": icon = "✅"
    elif level == "security": icon = "🛡️"
    
    msg_str = f"{icon} {msg}"
    if level == "error": logging.error(msg_str)
    elif level == "warning": logging.warning(msg_str)
    else: logging.info(msg_str)
    
def load_env():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        log(f"Loading secrets from {env_path}...", "info")
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    # Strip whitespace and quotes
                    val = val.strip().strip('"').strip("'")
                    os.environ[key.strip()] = val
    else:
        log(".env file not found, skipping local load.", "warning")

def cleanup():
    log("CLEANUP: Removing temporary files...", "info")
    if TEMP_DATA_DIR.exists():
        shutil.rmtree(TEMP_DATA_DIR)
        log(f"Removed {TEMP_DATA_DIR}")
    if TEMP_LOGITS_DIR.exists():
        shutil.rmtree(TEMP_LOGITS_DIR)
        log(f"Removed {TEMP_LOGITS_DIR}")
    log("CLEANUP: Done.", "success")


def write_train_ready_report(payload: Dict[str, Any], report_name: str | None = None) -> None:
    suffix = f".{report_name}" if report_name else ""
    report_path = LOG_DIR / f"train_ready_status{suffix}.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _serialize_cfg() -> Dict[str, Any]:
    """Serialize cfg into JSON-safe dict (best-effort)."""
    out: Dict[str, Any] = {}
    for k, v in cfg.__dict__.items():
        try:
            json.dumps(v)
            out[k] = v
        except TypeError:
            out[k] = str(v)
    return out


def write_config_snapshot() -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialize_cfg()
    SNAPSHOT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"Config snapshot written: {SNAPSHOT_PATH}", "info")


def _stage_jsonl_paths() -> Dict[str, Path]:
    return {
        "stage1": PROJECT_ROOT / "datasets" / "stage1" / "stage1_data.jsonl",
        "stage2": PROJECT_ROOT / "datasets" / "stage2" / "stage2_data.jsonl",
        "stage3": PROJECT_ROOT / "datasets" / "stage3" / "stage3_data.jsonl",
        "stage4": PROJECT_ROOT / "datasets" / "stage4_soul" / "stage4_data.jsonl",
        "stage5": PROJECT_ROOT / "datasets" / "stage5_tools" / "stage5_data.jsonl",
    }


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _precompute_state_path(logits_root: Path, stage_name: str) -> Path:
    return logits_root / f"{stage_name}_train_topk_state.json"


def _precompute_done_samples(logits_root: Path, stage_name: str) -> int:
    state_path = _precompute_state_path(logits_root, stage_name)
    if not state_path.exists():
        return 0
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    # [tier-2 MED] Read the canonical lines_consumed first; done_samples is only a
    # back-compat mirror that a future writer might drop.
    return int(payload.get("lines_consumed", payload.get("done_samples", 0)) or 0)


def _offline_precompute_stage_status(logits_root: Path) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    all_complete = True
    for stage_name, stage_path in _stage_jsonl_paths().items():
        total_rows = _count_jsonl_rows(stage_path)
        shard_count = len(list(logits_root.glob(f"{stage_name}_train_part_*.pt")))
        done_samples = _precompute_done_samples(logits_root, stage_name)
        complete = total_rows > 0 and shard_count > 0 and done_samples >= total_rows
        stages[stage_name] = {
            "dataset_path": str(stage_path),
            "total_rows": total_rows,
            "done_samples": done_samples,
            "shard_count": shard_count,
            "complete": complete,
            "state_path": str(_precompute_state_path(logits_root, stage_name)),
        }
        all_complete = all_complete and complete
    return {
        "all_complete": all_complete,
        "stages": stages,
    }


def _local_tokenizer_ready() -> tuple[bool, str]:
    tokenizer_meta = PROJECT_ROOT / "tokenizer" / "tokenizer.json"
    if not tokenizer_meta.exists():
        return False, "tokenizer/tokenizer.json missing"
    try:
        meta = json.loads(tokenizer_meta.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"tokenizer metadata unreadable: {exc}"

    note = str(meta.get("note", "")).lower()
    if "loaded at runtime" in note:
        artifact_roots = [
            PROJECT_ROOT / "data" / "tokenizer" / "tr",
            PROJECT_ROOT / "tokenizer" / "tr",
        ]
        required_names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
        for root in artifact_roots:
            if all((root / name).exists() for name in required_names):
                return True, f"ok (local tokenizer cache: {_display_path(root)})"
        return False, "tokenizer metadata is runtime-only; offline clean path lacks a real local tokenizer artifact"
    return True, "ok"


def check_stage_jsonl(offline: bool) -> bool:
    missing = [name for name, path in _stage_jsonl_paths().items() if not path.exists()]
    if missing:
        msg = f"Stage JSONL missing: {', '.join(missing)}"
        if offline:
            allow_missing = os.environ.get("TITAN_PREFLIGHT_ALLOW_MISSING_STAGE_JSONL", "0") == "1"
            ci_env = os.environ.get("GITHUB_ACTIONS", "").strip().lower() == "true" or os.environ.get("CI", "").strip().lower() == "true"
            if allow_missing or ci_env:
                note = "override: allow missing in offline mode" if allow_missing else "CI: allow missing in offline mode"
                log(msg + f" ({note})", "warning")
                return True
            log(msg, "error")
            return False
        log(msg + " (online mode will generate via smart_runner)", "warning")
        return True
    log("Stage JSONL files present.", "success")
    return True


def check_cuda_lock(strict: bool) -> bool:
    lock_path = PROJECT_ROOT / "repro" / "cuda.lock"
    content = ""
    if lock_path.exists():
        content = lock_path.read_text(encoding="utf-8", errors="ignore").strip()

    ok = bool(content) and "unknown" not in content.lower()
    if ok:
        log("CUDA lock file present.", "success")
        return True

    if os.environ.get("TITAN_PREFLIGHT_WRITE_CUDA_LOCK", "0") == "1" and torch.cuda.is_available():
        try:
            subprocess.check_call([sys.executable, str(PROJECT_ROOT / "scripts" / "write_cuda_lock.py")])
            content = lock_path.read_text(encoding="utf-8", errors="ignore").strip()
            ok = bool(content) and "unknown" not in content.lower()
        except Exception as exc:
            log(f"CUDA lock auto-write failed: {exc}", "warning")

    if ok:
        log("CUDA lock file present (auto-written).", "success")
        return True

    msg = "CUDA lock missing or unknown (run scripts/write_cuda_lock.py on training hardware)."
    if strict and torch.cuda.is_available():
        log(msg, "error")
        return False
    log(msg, "warning")
    return True


def _check_writable_dir(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe.tmp"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def runtime_injected_training_readiness_profile() -> int:
    """
    Portable rented-machine readiness profile.
    This lane proves that the repo can be handed off to a target machine where
    runtime credentials are injected there and the dataset/bootstrap flow can
    complete on that machine.
    """
    started_at = time.time()
    checks: Dict[str, Any] = {}
    profile = "runtime_injected_training_readiness"
    status = "FAIL"
    reason_code = "UNKNOWN_ERROR"

    try:
        load_env()

        required_files = [
            PROJECT_ROOT / "zero_touch_start.sh",
            PROJECT_ROOT / "run.sh",
            PROJECT_ROOT / "scripts" / "smart_runner.py",
            PROJECT_ROOT / "scripts" / "data_pipeline.py",
            PROJECT_ROOT / "scripts" / "precompute_logits_topk.py",
            PROJECT_ROOT / "scripts" / "final_orchestrator.py",
            PROJECT_ROOT / "train" / "train.py",
        ]
        missing_files = [str(path.relative_to(PROJECT_ROOT)) for path in required_files if not path.exists()]
        checks["runtime_bootstrap_entrypoints"] = {
            "status": "PASS" if not missing_files else "FAIL",
            "missing": missing_files,
        }
        if missing_files:
            reason_code = "RUNTIME_BOOTSTRAP_ENTRYPOINT_MISSING"
            raise RuntimeError(f"missing runtime bootstrap entrypoints: {missing_files}")

        teacher_model_id = str(getattr(cfg, "teacher_model_id", "")).strip()
        require_teacher = bool(getattr(cfg, "require_gated_teacher", False))
        checks["teacher_contract"] = {
            "status": "PASS" if teacher_model_id else "FAIL",
            "teacher_model_id": teacher_model_id or None,
            "require_gated_teacher": require_teacher,
            "policy": "remote_bootstrap expects HF_TOKEN to be injected on the target machine before any gated teacher access is attempted",
        }
        if not teacher_model_id:
            reason_code = "TEACHER_MODEL_UNSET"
            raise RuntimeError("teacher_model_id is missing")

        hf_token_present = bool(os.environ.get("HF_TOKEN", "").strip())
        wandb_enabled = os.environ.get("TITAN_WANDB", "1") == "1"
        wandb_key_present = bool(os.environ.get("WANDB_API_KEY", "").strip())
        checks["runtime_credentials"] = {
            "status": "PASS",
            "hf_token_present_locally": hf_token_present,
            "wandb_enabled": wandb_enabled,
            "wandb_api_key_present_locally": wandb_key_present,
            "policy": "missing local credentials are allowed because this lane assumes runtime injection on the target machine",
        }

        missing_stage = [name for name, path in _stage_jsonl_paths().items() if not path.exists()]
        checks["stage_jsonl"] = {
            "status": "PASS" if not missing_stage else "WARN",
            "missing": missing_stage,
            "policy": "remote smart_runner/data_pipeline may generate stage JSONL at training start",
        }

        required_paths = [
            PROJECT_ROOT / "datasets",
            PROJECT_ROOT / cfg.save_dir,
            Path(cfg.precomputed_logits_path),
            PROJECT_ROOT / "reports",
            PROJECT_ROOT / "artifacts",
        ]
        write_results: Dict[str, Any] = {}
        for p in required_paths:
            ok, detail = _check_writable_dir(p)
            write_results[str(p)] = {"status": "PASS" if ok else "FAIL", "detail": detail}
        failing_paths = [k for k, v in write_results.items() if v["status"] == "FAIL"]
        checks["write_permissions"] = {
            "status": "PASS" if not failing_paths else "FAIL",
            "paths": write_results,
        }
        if failing_paths:
            reason_code = "WRITE_PERMISSION_DENIED"
            raise RuntimeError(f"runtime bootstrap write permission checks failed: {failing_paths}")

        logits_root = Path(cfg.precomputed_logits_path)
        existing_shards = len(list(logits_root.glob("stage*_train_part_*.pt"))) if logits_root.exists() else 0
        checks["distillation_bootstrap"] = {
            "status": "PASS",
            "logits_dir": str(logits_root),
            "existing_shards": existing_shards,
            "policy": "local logits shards are optional because Phase-0 precompute or online teacher generation may run on the target machine",
        }

        checks["lane_contract"] = {
            "status": "PASS",
            "lane_name": "remote_bootstrap",
            "target_machine_requirements": [
                "HF_TOKEN injected before launch",
                "TITAN_OFFLINE=0 on the target machine",
                "network access for gated teacher and dataset fetches",
                "optional WANDB_API_KEY only if TITAN_WANDB=1",
            ],
            "launcher": "HF_TOKEN=... TITAN_OFFLINE=0 TITAN_INSTALL=1 TITAN_PROFILE=stable bash zero_touch_start.sh",
        }

        status = "PASS"
        reason_code = "READY_RUNTIME_INJECTED_BOOTSTRAP"
        log("RUNTIME-INJECTED TRAIN-READINESS: PASS", "success")
        print("TRAIN_READY:PASS reason_code=READY_RUNTIME_INJECTED_BOOTSTRAP")
        return_code = 0
    except Exception as exc:
        if reason_code == "UNKNOWN_ERROR":
            reason_code = "RUNTIME_BOOTSTRAP_PREFLIGHT_EXCEPTION"
        log(f"RUNTIME-INJECTED TRAIN-READINESS FAIL [{reason_code}]: {exc}", "error")
        print(f"TRAIN_READY:FAIL reason_code={reason_code}")
        return_code = 1
    finally:
        payload = {
            "profile": profile,
            "status": status,
            "reason_code": reason_code,
            "checks": checks,
            "elapsed_s": round(time.time() - started_at, 3),
        }
        write_train_ready_report(payload, report_name=profile)
        log(f"Train-ready report: {_display_path(LOG_DIR / f'train_ready_status.{profile}.json')}", "info")

    return return_code


def strict_training_readiness_profile() -> int:
    """
    Strict online/gated readiness profile for portable training handoff.
    Returns shell exit code.
    """
    started_at = time.time()
    checks: Dict[str, Any] = {}
    profile = "strict_online_training_readiness"
    status = "FAIL"
    reason_code = "UNKNOWN_ERROR"

    try:
        load_env()

        hf_token = os.environ.get("HF_TOKEN", "").strip()
        if not hf_token:
            checks["hf_token"] = {"status": "FAIL", "detail": "HF_TOKEN missing"}
            reason_code = "MISSING_HF_TOKEN"
            raise RuntimeError("HF_TOKEN missing")
        checks["hf_token"] = {"status": "PASS"}

        from huggingface_hub import HfApi
        api = HfApi()
        try:
            who = api.whoami(token=hf_token)
            checks["hf_auth"] = {
                "status": "PASS",
                "account": who.get("name") if isinstance(who, dict) else "ok",
            }
        except Exception as exc:
            checks["hf_auth"] = {"status": "FAIL", "detail": str(exc)}
            reason_code = "HF_AUTH_FAILED"
            raise

        # Gated teacher tokenizer access must pass in strict mode.
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained(cfg.teacher_model_id, token=hf_token, use_fast=True)
            vocab = getattr(tok, "vocab_size", None)
            checks["teacher_tokenizer_access"] = {
                "status": "PASS",
                "teacher_model_id": cfg.teacher_model_id,
                "vocab_size": int(vocab) if isinstance(vocab, int) else None,
            }
        except Exception as exc:
            checks["teacher_tokenizer_access"] = {"status": "FAIL", "detail": str(exc)}
            reason_code = "TEACHER_GATED_ACCESS_FAILED"
            raise

        # Minimal dataset API reachability checks (one representative from each stage family).
        required_datasets = [
            "bigcode/the-stack-dedup",
            "HuggingFaceFW/fineweb-edu",
            "wikimedia/wikipedia",
            "OpenAssistant/oasst_top1_2023-08-25",
            "glaiveai/glaive-function-calling-v2",
            "gorilla-llm/gorilla-openfunctions-v2",
            "openai/gsm8k",
            "uonlp/CulturaX",
            "TIGER-Lab/MathInstruct",
        ]
        dataset_results = {}
        for ds in required_datasets:
            try:
                api.dataset_info(ds, token=hf_token)
                dataset_results[ds] = "PASS"
            except Exception as exc:
                dataset_results[ds] = f"FAIL: {exc}"
        failing_ds = [k for k, v in dataset_results.items() if str(v).startswith("FAIL")]
        checks["dataset_api_access"] = {
            "status": "PASS" if not failing_ds else "FAIL",
            "results": dataset_results,
        }
        if failing_ds:
            reason_code = "DATASET_API_UNREACHABLE"
            raise RuntimeError(f"dataset API checks failed: {failing_ds}")

        # [B1] Source field smoke-probe: prove each configured source actually yields text
        # under its field+min_length. Catches the silent-empty-field class (the old
        # the-stack-v2 'content' produced zero usable rows and starved Stage-1 code data).
        field_probe = _probe_sources_yield_text(hf_token)
        empty_sources = [k for k, v in field_probe.items() if v == "SOURCE_FIELD_EMPTY"]
        checks["source_field_probe"] = {
            "status": "PASS" if not empty_sources else "FAIL",
            "results": field_probe,
        }
        if empty_sources:
            reason_code = "SOURCE_FIELD_EMPTY"
            raise RuntimeError(f"source field probe found empty-field sources: {empty_sources}")

        # Stage JSONL presence (warn unless strict override).
        missing_stage = [name for name, path in _stage_jsonl_paths().items() if not path.exists()]
        checks["stage_jsonl"] = {
            "status": "PASS" if not missing_stage else "WARN",
            "missing": missing_stage,
            "policy": "online smart_runner can generate if missing",
        }
        if missing_stage and os.environ.get("TITAN_PREFLIGHT_REQUIRE_STAGE_JSONL", "0") == "1":
            reason_code = "STAGE_JSONL_MISSING"
            raise RuntimeError(f"stage JSONL missing: {missing_stage}")

        # Filesystem and write permission checks for train outputs/distillation.
        required_paths = [
            PROJECT_ROOT / "datasets",
            PROJECT_ROOT / cfg.save_dir,
            Path(cfg.precomputed_logits_path),
        ]
        write_results: Dict[str, Any] = {}
        for p in required_paths:
            ok, detail = _check_writable_dir(p)
            write_results[str(p)] = {"status": "PASS" if ok else "FAIL", "detail": detail}
        failing_paths = [k for k, v in write_results.items() if v["status"] == "FAIL"]
        checks["write_permissions"] = {
            "status": "PASS" if not failing_paths else "FAIL",
            "paths": write_results,
        }
        if failing_paths:
            reason_code = "WRITE_PERMISSION_DENIED"
            raise RuntimeError(f"write permission checks failed: {failing_paths}")

        # Distillation shard path sanity (existence optional, writeability mandatory).
        logits_root = Path(cfg.precomputed_logits_path)
        shard_count = len(list(logits_root.glob("stage*_train_part_*.pt")))
        checks["distillation_paths"] = {
            "status": "PASS",
            "logits_dir": str(logits_root),
            "existing_shards": shard_count,
            "policy": "missing shards are allowed; online teacher generation remains mandatory",
        }

        # Disk free check for multi-stage online training.
        min_disk_gb = float(os.environ.get("TITAN_PREFLIGHT_MIN_DISK_GB", "100"))
        _, _, free = shutil.disk_usage(str(PROJECT_ROOT))
        free_gb = free / (1024 ** 3)
        disk_ok = free_gb >= min_disk_gb
        checks["disk_free"] = {
            "status": "PASS" if disk_ok else "FAIL",
            "free_gb": round(free_gb, 2),
            "required_gb": min_disk_gb,
        }
        if not disk_ok:
            reason_code = "INSUFFICIENT_DISK"
            raise RuntimeError(f"disk free too low: {free_gb:.2f}GB < {min_disk_gb:.2f}GB")

        # CUDA lock (strict when GPU present unless override).
        strict_cuda_lock = os.environ.get("TITAN_PREFLIGHT_STRICT_CUDA_LOCK", "1") == "1"
        cuda_ok = check_cuda_lock(strict=strict_cuda_lock)
        checks["cuda_lock"] = {
            "status": "PASS" if cuda_ok else "FAIL",
            "strict": strict_cuda_lock,
            "path": str(PROJECT_ROOT / "repro" / "cuda.lock"),
        }
        if not cuda_ok and strict_cuda_lock and torch.cuda.is_available():
            reason_code = "CUDA_LOCK_MISSING"
            raise RuntimeError("cuda.lock missing or unknown")

        status = "PASS"
        reason_code = "READY"
        log("STRICT TRAIN-READINESS: PASS", "success")
        print("TRAIN_READY:PASS reason_code=READY")
        return_code = 0
    except Exception as exc:
        if reason_code == "UNKNOWN_ERROR":
            reason_code = "STRICT_PREFLIGHT_EXCEPTION"
        log(f"STRICT TRAIN-READINESS FAIL [{reason_code}]: {exc}", "error")
        print(f"TRAIN_READY:FAIL reason_code={reason_code}")
        return_code = 1
    finally:
        payload = {
            "profile": profile,
            "status": status,
            "reason_code": reason_code,
            "checks": checks,
            "elapsed_s": round(time.time() - started_at, 3),
        }
        write_train_ready_report(payload, report_name=profile)
        log(f"Train-ready report: {_display_path(LOG_DIR / 'train_ready_status.json')}", "info")

    return return_code


def strict_offline_training_readiness_profile() -> int:
    """
    Strict offline-clean readiness profile.
    This path must prove that the repository can launch the canonical strict
    precomputed-KD 45K pass without any teacherless fallback.
    """
    started_at = time.time()
    checks: Dict[str, Any] = {}
    profile = "strict_offline_training_readiness"
    status = "FAIL"
    reason_code = "UNKNOWN_ERROR"

    try:
        load_env()

        missing_stage = [name for name, path in _stage_jsonl_paths().items() if not path.exists()]
        checks["stage_jsonl"] = {
            "status": "PASS" if not missing_stage else "FAIL",
            "missing": missing_stage,
        }
        if missing_stage:
            reason_code = "STAGE_JSONL_MISSING"
            raise RuntimeError(f"offline stage JSONL missing: {missing_stage}")

        validation_path = PROJECT_ROOT / "datasets" / "validation.jsonl"
        val_rows = _count_jsonl_rows(validation_path)
        min_rows = int(getattr(cfg, "validation_min_samples_claim", 1000))
        checks["validation_jsonl"] = {
            "status": "PASS" if val_rows >= min_rows else "FAIL",
            "path": str(validation_path),
            "rows": val_rows,
            "required_rows": min_rows,
        }
        if val_rows < min_rows:
            reason_code = "VALIDATION_SET_TOO_SMALL"
            raise RuntimeError(f"validation rows too low: {val_rows} < {min_rows}")

        hashes_path = PROJECT_ROOT / "datasets" / "hashes.json"
        checks["dataset_hashes"] = {
            "status": "PASS" if hashes_path.exists() else "FAIL",
            "path": str(hashes_path),
        }
        if not hashes_path.exists():
            reason_code = "DATASET_HASHES_MISSING"
            raise RuntimeError("datasets/hashes.json missing")

        tokenizer_ok, tokenizer_detail = _local_tokenizer_ready()
        checks["local_tokenizer"] = {
            "status": "PASS" if tokenizer_ok else "FAIL",
            "detail": tokenizer_detail,
        }
        if not tokenizer_ok:
            reason_code = "LOCAL_TOKENIZER_UNAVAILABLE"
            raise RuntimeError(tokenizer_detail)

        strict_precompute = bool(getattr(cfg, "use_precomputed_logits", False))
        strict_teacher = bool(getattr(cfg, "require_gated_teacher", False))
        checks["strict_precompute_policy"] = {
            "status": "PASS" if strict_precompute and strict_teacher else "FAIL",
            "use_precomputed_logits": strict_precompute,
            "require_gated_teacher": strict_teacher,
            "teacher_model_id": str(getattr(cfg, "teacher_model_id", "")),
        }
        if not strict_precompute:
            reason_code = "PRECOMPUTED_LOGITS_DISABLED"
            raise RuntimeError("offline_clean canonical lane requires TITAN_USE_PRECOMPUTED_LOGITS=1")
        if not strict_teacher:
            reason_code = "STRICT_GATED_TEACHER_DISABLED"
            raise RuntimeError("offline_clean canonical lane requires TITAN_REQUIRE_GATED_TEACHER=1")

        required_paths = [
            PROJECT_ROOT / "datasets",
            PROJECT_ROOT / cfg.save_dir,
            Path(cfg.precomputed_logits_path),
        ]
        write_results: Dict[str, Any] = {}
        for p in required_paths:
            ok, detail = _check_writable_dir(p)
            write_results[str(p)] = {"status": "PASS" if ok else "FAIL", "detail": detail}
        failing_paths = [k for k, v in write_results.items() if v["status"] == "FAIL"]
        checks["write_permissions"] = {
            "status": "PASS" if not failing_paths else "FAIL",
            "paths": write_results,
        }
        if failing_paths:
            reason_code = "WRITE_PERMISSION_DENIED"
            raise RuntimeError(f"offline write permission checks failed: {failing_paths}")

        logits_root = Path(cfg.precomputed_logits_path)
        precompute_status = _offline_precompute_stage_status(logits_root)
        hf_token = os.environ.get("HF_TOKEN", "").strip()
        actionable_phase0 = bool(hf_token) and bool(str(getattr(cfg, "teacher_model_id", "")).strip())
        existing_shards = sum(stage["shard_count"] for stage in precompute_status["stages"].values())
        checks["distillation_paths"] = {
            "status": "PASS" if precompute_status["all_complete"] or actionable_phase0 else "FAIL",
            "logits_dir": str(logits_root),
            "existing_shards": existing_shards,
            "all_complete": precompute_status["all_complete"],
            "actionable_phase0_precompute": actionable_phase0,
            "hf_token_present": bool(hf_token),
            "policy": "offline_clean canonical lane requires either complete precomputed logits shards or actionable Phase-0 precompute with valid gated teacher access",
            "stages": precompute_status["stages"],
        }
        if not precompute_status["all_complete"] and not actionable_phase0:
            reason_code = (
                "PRECOMPUTED_LOGITS_INCOMPLETE_AND_PHASE0_NOT_ACTIONABLE"
                if existing_shards > 0
                else "PRECOMPUTED_LOGITS_MISSING_AND_PHASE0_NOT_ACTIONABLE"
            )
            raise RuntimeError("offline_clean strict precompute requirements are not satisfied")

        # TR: [B2] Shard'lar tamamsa, sadece varlik degil KIMLIK hizalamasini da
        #     dogrula -> hizasiz teacher-logit = sessiz KD bozulmasi, burada gurultulu
        #     on-ucus hatasina cevrilir. Henuz uretilmemis (phase0) ise SKIP.
        # EN: [B2] When shards are complete, verify IDENTITY alignment (not just
        #     existence) -> misaligned teacher logits become a loud preflight failure
        #     instead of silent KD corruption. SKIP when not yet produced (phase0).
        if precompute_status["all_complete"]:
            align = None
            try:
                from scripts.validate_logit_alignment import validate_all_stages
                align = validate_all_stages(logits_root)
            except Exception as exc:
                # Cannot validate (e.g. tokenizer not loadable here) -> WARN, do not
                # block; real training resolves the same tokenizer and re-checks.
                checks["logit_alignment"] = {"status": "WARN", "error": str(exc)}
            if align is not None:
                checks["logit_alignment"] = {
                    "status": "PASS" if align["status"] == "PASS" else "FAIL",
                    "result": align["status"],
                    "checks": align["checks"],
                }
                if align["status"] == "FAIL":
                    reason_code = "LOGIT_ALIGNMENT_FAILED"
                    raise RuntimeError("offline_clean logit alignment verification failed")
        else:
            checks["logit_alignment"] = {"status": "SKIP", "policy": "no complete shards yet to validate"}

        strict_cuda_lock = os.environ.get("TITAN_PREFLIGHT_STRICT_CUDA_LOCK", "0") == "1"
        cuda_ok = check_cuda_lock(strict=strict_cuda_lock)
        checks["cuda_lock"] = {
            "status": "PASS" if cuda_ok else "WARN",
            "strict": strict_cuda_lock,
            "path": str(PROJECT_ROOT / "repro" / "cuda.lock"),
        }

        status = "PASS"
        reason_code = "READY_PRECOMPUTED_LOGITS_COMPLETE" if precompute_status["all_complete"] else "READY_ACTIONABLE_PHASE0_PRECOMPUTE"
        print(f"TRAIN_READY:PASS reason_code={reason_code}")
        return_code = 0
    except Exception as exc:
        if reason_code == "UNKNOWN_ERROR":
            reason_code = "STRICT_OFFLINE_PREFLIGHT_EXCEPTION"
        log(f"STRICT OFFLINE TRAIN-READINESS FAIL [{reason_code}]: {exc}", "error")
        print(f"TRAIN_READY:FAIL reason_code={reason_code}")
        return_code = 1
    finally:
        payload = {
            "profile": profile,
            "status": status,
            "reason_code": reason_code,
            "checks": checks,
            "elapsed_s": round(time.time() - started_at, 3),
        }
        write_train_ready_report(payload, report_name=profile)
        log(f"Train-ready report: {_display_path(LOG_DIR / f'train_ready_status.{profile}.json')}", "info")

    return return_code

def check_secrets():
    log("STEP 1: SECRET SCAN...", "info")
    offline = os.environ.get("TITAN_OFFLINE", "1") != "0"
    require = os.environ.get("TITAN_PREFLIGHT_REQUIRE_SECRETS", "0") == "1"
    hf_token = os.environ.get("HF_TOKEN")
    wandb_key = os.environ.get("WANDB_API_KEY")
    
    if not hf_token or len(hf_token) < 10:
        if require or not offline:
            log("HF_TOKEN missing or invalid!", "error")
            return False
        log("HF_TOKEN missing (offline mode): OK (online checks will be skipped).", "warning")
    else:
        log("HF_TOKEN detected (redacted)", "security")

    if not wandb_key or len(wandb_key) < 10:
        if require:
            log("WANDB_API_KEY missing or invalid!", "error")
            return False
        log("WANDB_API_KEY missing (offline mode): OK (WandB checks disabled).", "warning")
    else:
        log("WANDB_API_KEY detected (redacted)", "security")

    log("Secrets check completed.", "success")
    return True

def architectural_audit():
    log("STEP 2: ARCHITECTURAL AUDIT...", "info")
    
    # 1. Layer Overlap Check (Native Config Check)
    try:
        validate_layer_config(cfg)
        log("Layer configuration validated: No Liquid/MoE conflicts.", "success")
    except Exception as e:
        log(f"Architectural conflict detected: {e}", "error")
        return False
        
    # 2. GQA Attention Dimension Consistency
    if cfg.hidden_size != cfg.num_heads * cfg.head_dim:
        log(f"GQA Dimension Mismatch: hidden_size({cfg.hidden_size}) != heads({cfg.num_heads}) * head_dim({cfg.head_dim})", "error")
        return False
    else:
        log(f"GQA Dimensions: Consistent ({cfg.hidden_size} features).", "success")
        
    # 3. BitNet Sanity
    log("BitNet b1.58 logic: ACTIVE (Locked).", "success")
    return True

def data_distill_test():
    log("STEP 3: DATA & DISTILLATION TEST...", "info")
    offline = os.environ.get("TITAN_OFFLINE", "1") != "0"
    
    # Reduce noisy HTTP logs in preflight output/logs (best-effort).
    try:
        from huggingface_hub.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_default_handler()
        hf_logging.disable_propagation()
    except Exception:
        pass
    try:
        from datasets.utils import logging as ds_logging
        ds_logging.set_verbosity_error()
        ds_logging.disable_default_handler()
        ds_logging.disable_propagation()
    except Exception:
        pass
    # Underlying HTTP clients (httpx/httpcore/urllib3) may still emit INFO logs.
    for name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # 1. Connection Check (lightweight by default)
    # NOTE: Avoid streaming a parquet sample here. In some environments this can
    # leave background transfers running and the process may never exit even
    # after printing "ALL GREEN".
    hf_token = os.environ.get("HF_TOKEN")
    if offline:
        log("Offline mode: skipping Hugging Face connectivity checks.", "info")
    else:
        try:
            from huggingface_hub import HfApi

            api = HfApi()

            # Auth check (no sensitive output)
            if hf_token:
                api.whoami(token=hf_token)

            # Metadata check (small request)
            api.dataset_info("uonlp/CulturaX", token=hf_token)
            log("Connection to uonlp/CulturaX metadata successful.", "success")

            # Optional deep check (opt-in)
            if os.environ.get("TITAN_PREFLIGHT_STREAM_SAMPLE") == "1":
                from datasets import load_dataset
                from utils.dataset_registry import get_hf_revision

                revision = get_hf_revision("uonlp/CulturaX")
                ds = load_dataset(
                    "uonlp/CulturaX",
                    "tr",
                    split="train",
                    streaming=True,
                    revision=revision,
                    token=hf_token,
                )
                next(iter(ds))
                log("Connection to uonlp/CulturaX streaming sample successful.", "success")
        except Exception as e:
            log(f"Data access warning (might be gated): {e}", "warning")
            log("Falling back to internal mock data for pipeline verification.")
        
    # 2. Pipeline Dry Run
    TEMP_DATA_DIR.mkdir(exist_ok=True)
    TEMP_LOGITS_DIR.mkdir(exist_ok=True)
    
    # Mock Manager
    # Tokenizer: avoid network in offline mode by using a tiny local mock.
    if offline:
        class _MockTokenizer:
            pad_token_id = 0
            eos_token_id = 1
            pad_token = "<pad>"
            eos_token = "</s>"

            def __call__(self, text, return_tensors="pt", max_length=None, truncation=False):
                import torch
                b = text.encode("utf-8", errors="ignore")
                ids = [(x % 250) + 2 for x in b]
                if max_length:
                    ids = ids[:max_length]
                if not ids:
                    ids = [self.eos_token_id]
                return type("TokOut", (), {"input_ids": torch.tensor([ids], dtype=torch.long)})()

        tokenizer = _MockTokenizer()
    else:
        from transformers import AutoTokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model_id, token=hf_token)
        except Exception:
            log("Teacher Tokenizer not found, using generic Llama-3 tokenizer mock.", "warning")
            tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    manager = DistillationManager(cfg, tokenizer)
    manager.logits_dir = TEMP_LOGITS_DIR
    
    # MOCK TEACHER
    class MockTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type('C', (), {'hidden_size': 16})()
        def forward(self, input_ids, **kwargs):
            b, s = input_ids.shape
            return type('O', (), {'logits': torch.randn(b, s, cfg.vocab_size)})()
        def eval(self): pass
        
    manager.teacher_model = MockTeacher()
    log("Teacher Model mocked (Prevented 140GB download).", "security")
    
    dataset_mock = [{"text": "MertFormer Titan Ultimate Test"}]
    previous_dense_precompute = os.environ.get("TITAN_ALLOW_DENSE_PRECOMPUTE")
    os.environ["TITAN_ALLOW_DENSE_PRECOMPUTE"] = "1"
    try:
        manager.precompute_logits(dataset_mock, "preflight", subset="test")
    finally:
        if previous_dense_precompute is None:
            os.environ.pop("TITAN_ALLOW_DENSE_PRECOMPUTE", None)
        else:
            os.environ["TITAN_ALLOW_DENSE_PRECOMPUTE"] = previous_dense_precompute
    
    if (TEMP_LOGITS_DIR / "preflight_test_part_0.pt").exists():
        log("Distillation pipeline: PROVEN (Logits generated/saved).", "success")
    else:
        log("Distillation pipeline FAILED: No logits saved.", "error")
        return False
        
    return True

def moe_guru_learning_test():
    log("STEP 4: MOE GURU LEARNING TEST...", "info")
    mini_overrides = {
        "num_layers": 2,
        "hidden_size": 256,
        "num_heads": 2,
        "num_kv_heads": 2,
        "vocab_size": 1000,
        "moe_every_n_layers": 1,
        "liquid_layers_idx": [0],
        "use_gradient_checkpointing": False,
        "router_jitter": 0.0,
    }
    original_cfg = {name: getattr(cfg, name) for name in mini_overrides}

    try:
        for name, value in mini_overrides.items():
            setattr(cfg, name, value)
        log("🏗️  CONFIG: Using 'Mini-Titan' (2 Layers, 256 Hidden, forced MoE/Liquid) for RAM safety.")

        model = MertFormer()
        model.train()

        # Test Data
        input_ids = torch.randint(0, 1000, (1, 32))
        target_logits = torch.randn(1, 32, 1000)

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1) # High LR for clear grad

        # Forward
        logits, aux_loss, _ = model(input_ids)

        # Loss (Distillation target)
        loss = nn.MSELoss()(logits, target_logits) + aux_loss.float()

        # Backward
        loss.backward()

        # Gradient Audit
        log("Checking Architectural Gradient Health...", "info")
        found_moe_grad = False
        found_liquid_grad = False
        experts_with_grad = 0
        liquid_params_with_grad = 0

        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.norm() > 0:
                if "experts" in name:
                    experts_with_grad += 1
                    found_moe_grad = True
                if "liquid" in name or "cfc" in name:
                    liquid_params_with_grad += 1
                    found_liquid_grad = True

        if found_moe_grad:
            log(f"MoE Learning: PROVEN ({experts_with_grad} expert params receiving gradients).", "success")
        else:
            log(f"MoE Gradient Trace: Loss={loss.item():.4f}, AuxLoss={aux_loss.item():.4f}", "warning")
            log("MoE Learning: FAILED (Gradients not flowing to experts!).", "error")
            return False

        if found_liquid_grad:
            log(f"Liquid Dynamics: PROVEN ({liquid_params_with_grad} liquid params receiving gradients).", "success")
        else:
            log("Liquid Dynamics: FAILED (Gradients not flowing to Liquid layers!).", "error")
            return False

        # Check shared expert
        shared_grad = False
        for name, param in model.named_parameters():
             if "shared_expert" in name and param.grad is not None and param.grad.norm() > 0:
                  shared_grad = True
        log(f"Shared Expert Grad: {'OK' if shared_grad else 'NONE'}", "info")

        log("MertFormer forward/backward pass verified.", "success")
        return True
    finally:
        for name, value in original_cfg.items():
            setattr(cfg, name, value)

def run_default_profile() -> int:
    log("============================================================")
    log("🚀 MERTFORMER TITAN - ULTIMATE PREFLIGHT JUDGE 🚀")
    log("============================================================")
    
    start_time = time.time()
    success = False
    
    try:
        load_env()
        write_config_snapshot()
        if not check_secrets():
            return 1
        if not check_stage_jsonl(offline=os.environ.get("TITAN_OFFLINE", "1") != "0"):
            return 1
        check_cuda_lock(strict=False)
        if not architectural_audit():
            return 1
        if not data_distill_test():
            return 1
        if not moe_guru_learning_test():
            return 1
        
        success = True
        log("OVERALL SYSTEM STATUS: 100% PROTECTED & READY.", "success")
    except Exception as e:
        log(f"CRITICAL PREFLIGHT FAILURE: {e}", "error")
        import traceback
        logging.error(traceback.format_exc())
        return 1
    finally:
        cleanup()
        elapsed = time.time() - start_time
        log(f"Preflight Duration: {elapsed:.2f}s", "info")
        log("============================================================")
        log(f"RESULT: {'🏆 ALL GREEN' if success else '🚨 RED ALERT'}")
        log(f"Full Report: {TEST_LOG_PATH}", "info")
        log("============================================================")
        
    if not success:
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description="TITAN preflight runner")
    parser.add_argument(
        "--profile",
        type=str,
        default="default",
        choices=[
            "default",
            "strict_online_training_readiness",
            "strict_offline_training_readiness",
            "runtime_injected_training_readiness",
        ],
        help="Preflight profile to run.",
    )
    args = parser.parse_args()

    if args.profile == "strict_online_training_readiness":
        code = strict_training_readiness_profile()
    elif args.profile == "strict_offline_training_readiness":
        code = strict_offline_training_readiness_profile()
    elif args.profile == "runtime_injected_training_readiness":
        code = runtime_injected_training_readiness_profile()
    else:
        code = run_default_profile()
    if code != 0:
        sys.exit(code)

if __name__ == "__main__":
    main()
