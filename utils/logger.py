"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - RUN LOGGER
-------------------------------------------------------------------------------
Copyright (c) 2026 Mert Yünlü. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v1.0 (Build 30) - Pre-Training
Status : PRE-TRAINING (UNVERIFIED)
==============================================================================
"""

from __future__ import annotations

from config.build_label import BUILD_LABEL as __version__
__author__ = "Mert Yünlü"

import os
import sys
import csv
import json
import re
import time
import hashlib
import platform
import socket
import subprocess
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Iterable, Union


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _local_stamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def _safe_json(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes__": True, "len": len(obj), "sha256": hashlib.sha256(obj).hexdigest()}

    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(x) for x in obj]

    if is_dataclass(obj):
        return _safe_json(asdict(obj))

    tname = type(obj).__name__.lower()
    if "tensor" in tname:
        try:
            shape = getattr(obj, "shape", None)
            dtype = getattr(obj, "dtype", None)
            device = getattr(obj, "device", None)
            return {
                "__tensor__": True,
                "shape": [int(x) for x in shape] if shape is not None else None,
                "dtype": str(dtype) if dtype is not None else None,
                "device": str(device) if device is not None else None,
            }
        except Exception:
            return {"__tensor__": True}

    return {"__repr__": repr(obj), "__type__": type(obj).__name__}


def sha256_file(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def try_git_commit(repo_dir: Union[str, Path]) -> Optional[str]:
    try:
        repo_dir = str(Path(repo_dir).resolve())
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode("utf-8", "ignore").strip()
    except Exception:
        return None


def atomic_write_json(path: Union[str, Path], data: Dict[str, Any]) -> None:
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


# KEEP IN SYNC with scripts/logbook_build.py REDACT_PATTERNS.
# (Same cross-file-sync convention as the bitlinear.py <-> liquid.py "quant parity note".)
# These two lists are maintained independently and had drifted: logbook_build.py carried a
# 40-hex catch-all that this list did not, so a WandB API key written through the live
# logger was NOT redacted while the same key was redacted on the logbook rebuild path.
SECRET_PATTERNS = [
    re.compile(r"\bhf_[A-Za-z0-9_\-]{8,}\b"),
    re.compile(r"\bwandb_[A-Za-z0-9_\-]{8,}\b"),
    # WandB API keys are often 40 hex chars; also catches accidental full git SHAs in logs
    # (deliberate false-positive tolerance — same trade-off as logbook_build.py).
    re.compile(r"\b[0-9a-fA-F]{40}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{10,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b"),
]

SENSITIVE_KEY_PATTERN = re.compile(
    r"(^|[_\-])(api[_\-]?key|access[_\-]?key|secret|password|passwd|credential|private[_\-]?key|auth[_\-]?token|hf[_\-]?token|wandb[_\-]?api[_\-]?key)($|[_\-])",
    re.IGNORECASE,
)


def _is_sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_").lower()
    if normalized in {"token", "secret", "password", "passwd", "api_key", "access_key", "private_key", "credential"}:
        return True
    return bool(SENSITIVE_KEY_PATTERN.search(key))


def _redact_text(text: str) -> str:
    out = text
    for pattern in SECRET_PATTERNS:
        out = pattern.sub("[REDACTED]", out)
    return out


def _redact_obj(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, str):
        return _redact_text(obj)
    if isinstance(obj, (int, float, bool)):
        return obj
    if isinstance(obj, dict):
        redacted = {}
        for k, v in obj.items():
            key = str(k)
            if _is_sensitive_key(key):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = _redact_obj(v)
        return redacted
    if isinstance(obj, (list, tuple)):
        return [_redact_obj(v) for v in obj]
    return _redact_text(str(obj))


def _ensure_logbook_header(path: Path) -> None:
    header = {
        "type": "logbook_header",
        "title": "MertFormer Unified Logbook",
        "schema_version": "1.0",
        "created_at_utc": _utc_iso(),
        "note": "Unified logbook for all logs under logs/. New entries append automatically.",
        "redaction_policy": "Best-effort value and sensitive-key redaction for common API tokens/secrets.",
    }
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(header, ensure_ascii=False) + "\n", encoding="utf-8")
        return

    try:
        with path.open("r", encoding="utf-8") as f:
            first = f.readline().strip()
        obj = json.loads(first) if first else {}
    except Exception:
        obj = {}

    if obj.get("type") != "logbook_header":
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as out:
            out.write(json.dumps(header, ensure_ascii=False) + "\n")
            with path.open("r", encoding="utf-8") as src:
                for line in src:
                    out.write(line)
        tmp.replace(path)


DEFAULT_STEP_CSV_FIELDS = [
    "timestamp_utc",
    "step",
    "curriculum_stage",
    "lr",
    "loss",
    "ce",
    "kd",
    "aux",
    "grad_norm",
    "max_grad_norm",
    "tok_s",
    "interval_sec",
    "step_wall_sec",
    "cpu_percent",
    "ram_used_gb",
    "ram_total_gb",
    "gpu_device",
    "gpu_allocated_gb",
    "gpu_reserved_gb",
    "gpu_total_gb",
    "gpu_util_percent",
    "gpu_temp_c",
    "gpu_power_w",
    "gpu_power_limit_w",
    "gpu_mem_used_gb_smi",
    "gpu_mem_total_gb_smi",
    "disk_used_gb",
    "disk_free_gb",
    "disk_total_gb",
    "continual_ema_loss",
    "continual_replay_size",
    "continual_drift_alert",
    "moe_max_load",
    "moe_avg_std",
    "moe_load_entropy",
    "moe_capacity_overflow",
    "router_collapse",
    # Legacy columns kept for compatibility with older report consumers.
    "cbd",
    "lsep",
    "ent",
    "mode",
    "mem",
]


class RunLogger:
    def __init__(
        self,
        cfg: Any,
        log_dir: Union[str, Path] = "logs",
        run_name: Optional[str] = None,
        also_csv: bool = False,
        csv_fields: Optional[Iterable[str]] = None,
        flush_every: int = 10,
        fsync_every: int = 100,
        fail_safe: bool = True,
        project_root: Optional[Union[str, Path]] = None,
        train_path: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        extra_files: Optional[Iterable[Union[str, Path]]] = None,
    ):
        self.cfg = cfg
        self.fail_safe = bool(fail_safe)
        self.flush_every = max(1, int(flush_every))
        self.fsync_every = max(1, int(fsync_every))

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logbook_path = self.log_dir / "ALL_LOGS.jsonl"
        self._logbook_ready = False
        self._logbook_enabled = os.environ.get("MERTFORMER_LOGBOOK", "1") != "0"

        stamp = _local_stamp()
        self.run_id = run_name or f"run_{stamp}"
        self.jsonl_path = self.log_dir / f"{self.run_id}.jsonl"
        self.manifest_path = self.log_dir / f"{self.run_id}.manifest.json"
        self.csv_path = self.log_dir / f"{self.run_id}.csv" if also_csv else None

        self._fh = None
        self._csv_fh = None
        self._csv_fields = list(csv_fields) if csv_fields else None

        self._genesis_hash = hashlib.sha256(b"").hexdigest()
        self._prev_hash = self._genesis_hash
        self._line_count = 0
        self._step_count = 0
        self._opened_at = _utc_iso()

        self.project_root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
        self.train_path = Path(train_path).resolve() if train_path else None
        self.config_path = Path(config_path).resolve() if config_path else None
        self.extra_files = [Path(x).resolve() for x in (extra_files or [])]

        self._finalized = False
        self._open_files()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._finalized:
            return
        if exc is None:
            self.finalize(status="completed")
        else:
            self.finalize(status="failed", extra={"exc_type": str(exc_type), "exc": repr(exc)})
        return None

    def _warn(self, msg: str) -> None:
        print(f"[LOGGER][WARN] {msg}", file=sys.stderr)

    def _open_files(self) -> None:
        try:
            self._fh = self.jsonl_path.open("a", encoding="utf-8", buffering=1)
            if self.csv_path is not None:
                self._csv_fh = self.csv_path.open("a", encoding="utf-8", buffering=1)
        except Exception as e:
            if self.fail_safe:
                self._warn(f"Log dosyaları açılamadı: {e}")
                self._fh = None
                self._csv_fh = None
            else:
                raise

    def _write_line(self, rec: Dict[str, Any]) -> None:
        if self._fh is None:
            return
        try:
            safe = _redact_obj(_safe_json(rec))
            if not isinstance(safe, dict):
                safe = {"value": safe}

            safe["_chain"] = {"prev": self._prev_hash, "n": self._line_count + 1}
            line = json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            line_bytes = (line + "\n").encode("utf-8")

            h = hashlib.sha256()
            h.update(self._prev_hash.encode("utf-8"))
            h.update(line_bytes)
            line_hash = h.hexdigest()

            safe["_chain"]["hash"] = line_hash

            line2 = json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            self._fh.write(line2 + "\n")

            self._prev_hash = line_hash
            self._line_count += 1

        except Exception as e:
            if self.fail_safe:
                self._warn(f"JSONL yazılamadı: {e}")
            else:
                raise

    def _append_logbook(self, entry_type: str, payload: Dict[str, Any], source_kind: str) -> None:
        if not self._logbook_enabled:
            return
        try:
            if not self._logbook_ready:
                _ensure_logbook_header(self.logbook_path)
                self._logbook_ready = True

            record = {
                "type": entry_type,
                "timestamp_utc": _utc_iso(),
                "run_id": self.run_id,
                "source_kind": source_kind,
                "source_file": str(self.jsonl_path),
                # [2026-07-08] Truth-in-labeling fix. This used to say
                # `"source_sha256_status": "pending"`, which implied something would later
                # resolve it. Nothing ever does, and nothing CAN: scripts/logbook_build.py
                # only fills source_sha256 for its own log_import_start/log_import_end
                # records. A live-appending JSONL cannot embed its own final SHA256 in lines
                # that are themselves part of the hash chain being appended to — hashing it
                # would change it. "pending" was a permanent lie; this is the honest label.
                # Zero behavior change.
                "source_sha256": None,
                "source_sha256_status": "not_applicable_live_stream",
                "payload": _redact_obj(payload),
            }
            with self.logbook_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            if self.fail_safe:
                self._warn(f"logbook append failed: {e}")
            else:
                raise

    def _write_csv(self, step_rec: Dict[str, Any]) -> None:
        if self._csv_fh is None or self.csv_path is None:
            return
        try:
            if self._csv_fields is None:
                self._csv_fields = list(DEFAULT_STEP_CSV_FIELDS)

            try:
                is_empty = (os.stat(self.csv_path).st_size == 0)
            except OSError:
                is_empty = True

            if is_empty:
                self._csv_fh.write(",".join(self._csv_fields) + "\n")

            writer = csv.writer(self._csv_fh, lineterminator="\n")
            row = []
            for k in self._csv_fields:
                v = step_rec.get(k, "")
                if isinstance(v, float):
                    s = f"{v:.6g}"
                else:
                    s = str(v)
                row.append("[REDACTED]" if _is_sensitive_key(str(k)) else _redact_text(s))
            writer.writerow(row)

        except Exception as e:
            if self.fail_safe:
                self._warn(f"CSV yazılamadı: {e}")
            else:
                raise

    def log_meta(self, extra: Optional[Dict[str, Any]] = None) -> None:
        files = []
        if self.train_path and self.train_path.is_file():
            files.append(self.train_path)
        if self.config_path and self.config_path.is_file():
            files.append(self.config_path)
        for p in self.extra_files:
            if p.is_file():
                files.append(p)

        file_hashes = {}
        for p in files:
            try:
                file_hashes[str(p)] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            except Exception as e:
                file_hashes[str(p)] = {"error": str(e)}

        try:
            import torch
            torch_ver = getattr(torch, "__version__", None)
        except Exception:
            torch_ver = None

        meta = {
            "type": "meta",
            "timestamp_utc": _utc_iso(),
            "opened_at_utc": self._opened_at,
            "run_id": self.run_id,
            "host": socket.gethostname(),
            "user": os.environ.get("USER") or os.environ.get("USERNAME"),
            "python": sys.version.split()[0],
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "torch": torch_ver,
            "git_commit": try_git_commit(self.project_root),
            "project_root": str(self.project_root),
            "cfg": _safe_json(self.cfg),
            "source_hashes": file_hashes,
        }
        if extra:
            meta["extra"] = _safe_json(extra)

        self._write_line(meta)
        self._append_logbook("live_meta", meta, "meta")
        self._maybe_flush(force=True)

    def log_event(self, name: str, data: Optional[Dict[str, Any]] = None) -> None:
        rec = {
            "type": "event",
            "timestamp_utc": _utc_iso(),
            "name": str(name),
            "data": _safe_json(data or {}),
        }
        self._write_line(rec)
        self._maybe_flush(force=False)
        self._append_logbook("live_event", rec, "event")

    def log_step(self, metrics: Dict[str, Any]) -> None:
        self._step_count += 1

        rec = dict(metrics)
        rec.setdefault("type", "step")
        rec.setdefault("timestamp_utc", _utc_iso())

        step_val = rec.get("global_step", rec.get("step", self._step_count))
        rec["step"] = step_val
        if "global_step" in rec:
            rec["global_step"] = step_val

        self._write_line(rec)
        self._write_csv(rec)
        self._maybe_flush(force=False)
        self._append_logbook("live_step", rec, "step")

    def _maybe_flush(self, force: bool) -> None:
        if self._fh is None:
            return
        try:
            if force or (self._line_count % self.flush_every == 0):
                self._fh.flush()
                if self._csv_fh:
                    self._csv_fh.flush()

            if force or (self._line_count % self.fsync_every == 0):
                os.fsync(self._fh.fileno())
                if self._csv_fh:
                    os.fsync(self._csv_fh.fileno())

        except Exception as e:
            if self.fail_safe:
                self._warn(f"flush/fsync başarısız: {e}")
            else:
                raise

    def finalize(self, status: str = "completed", extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self._finalized:
            return {
                "run_id": self.run_id,
                "status": "already_finalized",
                "jsonl_path": str(self.jsonl_path),
                "manifest_path": str(self.manifest_path),
                "final_chain_hash": self._prev_hash,
                "lines": self._line_count,
            }

        end = {
            "type": "final",
            "timestamp_utc": _utc_iso(),
            "status": str(status),
            "lines": self._line_count,
            "pre_final_chain_hash": self._prev_hash,
        }
        if extra:
            end["extra"] = _safe_json(extra)

        self._write_line(end)
        self._maybe_flush(force=True)
        self._append_logbook("live_final", end, "final")

        try:
            size = self.jsonl_path.stat().st_size if self.jsonl_path.exists() else None
        except Exception:
            size = None

        manifest = {
            "run_id": self.run_id,
            "opened_at_utc": self._opened_at,
            "closed_at_utc": _utc_iso(),
            "status": str(status),
            "jsonl_path": str(self.jsonl_path),
            "jsonl_bytes": size,
            "lines": self._line_count,
            "chain_genesis": self._genesis_hash,
            "final_chain_hash": self._prev_hash,
            "csv_path": str(self.csv_path) if self.csv_path else None,
        }
        if extra:
            manifest["extra"] = _redact_obj(_safe_json(extra))

        try:
            atomic_write_json(self.manifest_path, manifest)
        except Exception as e:
            if self.fail_safe:
                self._warn(f"manifest yazılamadı: {e}")
            else:
                raise

        try:
            if self._fh:
                self._fh.close()
            if self._csv_fh:
                self._csv_fh.close()
        except Exception:
            pass

        self._finalized = True
        return manifest
