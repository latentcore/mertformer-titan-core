"""Unified logbook builder.

Collects all log artifacts under logs/ and writes a single JSONL logbook:
logs/ALL_LOGS.jsonl

Usage:
  python3 scripts/logbook_build.py            # default append
  python3 scripts/logbook_build.py --append   # append only new files
  python3 scripts/logbook_build.py --rebuild  # rebuild from scratch
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable

LOG_DIR = Path("logs")
LOGBOOK_PATH = LOG_DIR / "ALL_LOGS.jsonl"

REDACT_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{8,}"),
    re.compile(r"wandb_[A-Za-z0-9]{8,}"),
    re.compile(r"sk-[A-Za-z0-9]{10,}"),
]


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _redact_text(text: str) -> str:
    out = text
    for pat in REDACT_PATTERNS:
        out = pat.sub("REDACTED", out)
    return out


def _redact(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, str):
        return _redact_text(obj)
    if isinstance(obj, (int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _redact(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_redact(v) for v in obj]
    return _redact_text(str(obj))


def _ensure_logbook_header(path: Path) -> None:
    header = {
        "type": "logbook_header",
        "title": "MertFormer Unified Logbook",
        "schema_version": "1.0",
        "created_at_utc": _utc_iso(),
        "note": "Unified logbook for all logs under logs/. New entries append automatically.",
        "redaction_policy": "Simple token redaction for hf_/wandb_/sk- patterns.",
    }
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(header, ensure_ascii=False) + "\n", encoding="utf-8")
        return

    # Ensure first line is a header; if not, prepend safely.
    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
    try:
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


def _iter_log_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path == LOGBOOK_PATH:
            continue
        if path.name == "README.md":
            continue
        if path.suffix == ".jsonl":
            yield path
            continue
        if path.suffix == ".csv":
            yield path
            continue
        if path.suffix == ".log":
            yield path
            continue
        if path.name.endswith(".manifest.json"):
            yield path
            continue


def _load_seen_hashes(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") in {"log_import_start", "log_import_end"}:
                sha = obj.get("source_sha256")
                if sha:
                    seen.add(sha)
    return seen


def _write_line(out, record: Dict[str, Any]) -> None:
    out.write(json.dumps(record, ensure_ascii=False) + "\n")


def _import_jsonl(out, path: Path, meta: Dict[str, Any]) -> int:
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:
                payload = {"raw": line.strip()}
            record = {
                "type": "log_entry",
                "timestamp_utc": _utc_iso(),
                **meta,
                "source_line": idx,
                "payload": _redact(payload),
            }
            _write_line(out, record)
            count += 1
    return count


def _import_csv(out, path: Path, meta: Dict[str, Any]) -> int:
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=2):
            record = {
                "type": "log_entry",
                "timestamp_utc": _utc_iso(),
                **meta,
                "source_line": idx,
                "payload": _redact(row),
            }
            _write_line(out, record)
            count += 1
    return count


def _import_log(out, path: Path, meta: Dict[str, Any]) -> int:
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = {
                "type": "log_entry",
                "timestamp_utc": _utc_iso(),
                **meta,
                "source_line": idx,
                "payload": {"message": _redact_text(line.rstrip("\n"))},
            }
            _write_line(out, record)
            count += 1
    return count


def _import_manifest(out, path: Path, meta: Dict[str, Any]) -> int:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        payload = {"error": f"failed to parse manifest: {exc}"}
    record = {
        "type": "log_manifest",
        "timestamp_utc": _utc_iso(),
        **meta,
        "source_line": 1,
        "payload": _redact(payload),
    }
    _write_line(out, record)
    return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--append", action="store_true", help="Append only new sources")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild logbook from scratch")
    args = parser.parse_args()

    if not args.append and not args.rebuild:
        args.append = True

    if not LOG_DIR.exists():
        raise SystemExit("logs/ directory not found")

    if args.rebuild and LOGBOOK_PATH.exists():
        LOGBOOK_PATH.unlink()

    _ensure_logbook_header(LOGBOOK_PATH)
    seen_hashes = _load_seen_hashes(LOGBOOK_PATH) if args.append else set()

    total_entries = 0
    imported_files = 0

    with LOGBOOK_PATH.open("a", encoding="utf-8") as out:
        for path in _iter_log_files(LOG_DIR):
            sha = _sha256_file(path)
            if sha in seen_hashes:
                continue

            rel = path.relative_to(LOG_DIR.parent)
            kind = "jsonl"
            if path.suffix == ".csv":
                kind = "csv"
            elif path.suffix == ".log":
                kind = "log"
            elif path.name.endswith(".manifest.json"):
                kind = "manifest"

            meta = {
                "source_file": str(rel),
                "source_kind": kind,
                "source_sha256": sha,
            }

            start_rec = {
                "type": "log_import_start",
                "timestamp_utc": _utc_iso(),
                **meta,
                "source_bytes": path.stat().st_size,
            }
            _write_line(out, start_rec)

            if kind == "jsonl":
                count = _import_jsonl(out, path, meta)
            elif kind == "csv":
                count = _import_csv(out, path, meta)
            elif kind == "log":
                count = _import_log(out, path, meta)
            else:
                count = _import_manifest(out, path, meta)

            end_rec = {
                "type": "log_import_end",
                "timestamp_utc": _utc_iso(),
                **meta,
                "entries": count,
            }
            _write_line(out, end_rec)

            total_entries += count
            imported_files += 1

    print(f"Logbook updated. Imported files: {imported_files}, entries: {total_entries}")


if __name__ == "__main__":
    main()
