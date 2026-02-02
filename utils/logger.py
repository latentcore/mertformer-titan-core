from __future__ import annotations

"""
==============================================================================
MERTFORMER TITAN (ONYX STORM) - LOCKED RUN LOGGER
-------------------------------------------------------------------------------
Copyright (c) 2026 MertFormer AI Team. All Rights Reserved.
Proprietary - All Rights Reserved.

Project: Mobile-First LLM Architecture for Samsung S25 NPU
Version: v27.0-FINAL (Locked & Sealed)
Status : PRODUCTION READY (LOCKED)
==============================================================================
"""

__version__ = "27.0-FINAL"
__author__ = "Mert"

import os
import io
import sys
import json
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
            safe = _safe_json(rec)
            if not isinstance(safe, dict):
                safe = {"value": safe}

            safe["_chain"] = {"prev": self._prev_hash}
            line = json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            line_bytes = (line + "\n").encode("utf-8")

            h = hashlib.sha256()
            h.update(self._prev_hash.encode("utf-8"))
            h.update(line_bytes)
            line_hash = h.hexdigest()

            safe["_chain"]["hash"] = line_hash
            safe["_chain"]["n"] = self._line_count + 1

            line2 = json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            self._fh.write(line2 + "\n")

            self._prev_hash = line_hash
            self._line_count += 1

        except Exception as e:
            if self.fail_safe:
                self._warn(f"JSONL yazılamadı: {e}")
            else:
                raise

    def _write_csv(self, step_rec: Dict[str, Any]) -> None:
        if self._csv_fh is None or self.csv_path is None:
            return
        try:
            if self._csv_fields is None:
                # >>> YALNIZCA BURASI GÜNCELLENDİ <<<
                self._csv_fields = [
                    "timestamp_utc", "step", "lr", "loss", "ce", "kd", "cbd",
                    "lsep", "aux", "ent", "grad_norm", "tok_s", "mode",
                    "mem", "moe_max_load", "moe_avg_std"
                ]

            try:
                is_empty = (os.stat(self.csv_path).st_size == 0)
            except OSError:
                is_empty = True

            if is_empty:
                self._csv_fh.write(",".join(self._csv_fields) + "\n")

            row = []
            for k in self._csv_fields:
                v = step_rec.get(k, "")
                if isinstance(v, float):
                    s = f"{v:.6g}"
                else:
                    s = str(v)
                if "," in s or "\n" in s or '"' in s:
                    s = '"' + s.replace('"', '""') + '"'
                row.append(s)
            self._csv_fh.write(",".join(row) + "\n")

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
            "final_chain_hash": self._prev_hash,
        }
        if extra:
            end["extra"] = _safe_json(extra)

        self._write_line(end)
        self._maybe_flush(force=True)

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
            manifest["extra"] = _safe_json(extra)

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
