"""Resumable prefix downloads for the Lichess CC0 databases.

We never need the whole 21.4 GB evaluation archive. At roughly 54 compressed
bytes per position (21,368,367,967 B / 394,669,566 positions, both measured
from the server), a few gigabytes of prefix already covers tens of millions of
positions. Downloads use HTTP Range and resume from whatever is already on
disk, so an interrupted fetch costs nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

USER_AGENT = "ChessFormerAI/1.0 (+local research build)"
CHUNK = 1 << 20  # 1 MiB

ProgressFn = Callable[[Dict[str, object]], None]


@dataclass
class DownloadResult:
    url: str
    path: Path
    bytes_on_disk: int
    total_bytes: Optional[int]
    complete: bool
    sha256: str
    elapsed_sec: float
    resumed_from: int = 0
    attempts: int = 1
    notes: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "url": self.url,
            "path": str(self.path),
            "bytes_on_disk": int(self.bytes_on_disk),
            "total_bytes": int(self.total_bytes) if self.total_bytes else None,
            "complete": bool(self.complete),
            "sha256": self.sha256,
            "elapsed_sec": round(self.elapsed_sec, 3),
            "resumed_from": int(self.resumed_from),
            "attempts": int(self.attempts),
            "notes": list(self.notes),
        }


def head(url: str, timeout: int = 60) -> Dict[str, object]:
    """Server-reported size and type. Used for planning and provenance."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        length = resp.headers.get("Content-Length")
        return {
            "url": url,
            "status": int(getattr(resp, "status", 200)),
            "content_length": int(length) if length else None,
            "content_type": resp.headers.get("Content-Type", ""),
            "last_modified": resp.headers.get("Last-Modified", ""),
            "accept_ranges": resp.headers.get("Accept-Ranges", ""),
        }


def sha256_file(path: Path, limit_bytes: Optional[int] = None) -> str:
    digest = hashlib.sha256()
    read = 0
    with Path(path).open("rb") as fh:
        while True:
            if limit_bytes is not None and read >= limit_bytes:
                break
            block = fh.read(CHUNK)
            if not block:
                break
            if limit_bytes is not None and read + len(block) > limit_bytes:
                block = block[: limit_bytes - read]
            digest.update(block)
            read += len(block)
    return digest.hexdigest()


def download_prefix(
    url: str,
    dest: Path,
    max_bytes: Optional[int] = None,
    *,
    timeout: int = 120,
    retries: int = 4,
    backoff_sec: float = 2.0,
    progress: Optional[ProgressFn] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> DownloadResult:
    """Download at most ``max_bytes`` of ``url`` into ``dest``, resuming if partial.

    ``max_bytes=None`` fetches the whole file. Returns provenance (byte count,
    sha256, server total) that goes straight into the data card.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    meta = head(url, timeout=timeout)
    total = meta.get("content_length")
    target = int(min(max_bytes, total)) if (max_bytes and total) else (max_bytes or total)

    existing = dest.stat().st_size if dest.exists() else 0
    resumed_from = existing
    started = time.time()
    notes = []
    attempts = 0

    if target is not None and existing >= target:
        notes.append("already satisfied from cache")
        return DownloadResult(
            url=url, path=dest, bytes_on_disk=existing, total_bytes=total,
            complete=True, sha256=sha256_file(dest, target), elapsed_sec=0.0,
            resumed_from=resumed_from, attempts=0, notes=notes,
        )

    last_error: Optional[str] = None
    while attempts <= retries:
        attempts += 1
        try:
            headers = {"User-Agent": USER_AGENT}
            if existing:
                headers["Range"] = f"bytes={existing}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = int(getattr(resp, "status", 200))
                if existing and status != 206:
                    # Server ignored the range: restart cleanly rather than
                    # appending a second copy of the head of the file.
                    notes.append(f"range not honored (status {status}); restarting")
                    existing = 0
                    dest.unlink(missing_ok=True)
                mode = "ab" if existing else "wb"
                with dest.open(mode) as fh:
                    while True:
                        if should_stop is not None and should_stop():
                            notes.append("stopped by caller")
                            break
                        if target is not None and existing >= target:
                            break
                        want = CHUNK
                        if target is not None:
                            want = min(CHUNK, target - existing)
                        block = resp.read(want)
                        if not block:
                            break
                        fh.write(block)
                        existing += len(block)
                        if progress is not None:
                            progress({
                                "stage": "download",
                                "url": url,
                                "bytes": existing,
                                "target": target,
                                "elapsed_sec": time.time() - started,
                            })
            break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            notes.append(f"attempt {attempts} failed: {last_error}")
            existing = dest.stat().st_size if dest.exists() else 0
            if attempts > retries:
                raise RuntimeError(f"download failed after {attempts} attempts: {last_error}") from exc
            time.sleep(backoff_sec * attempts)

    final_size = dest.stat().st_size if dest.exists() else 0
    complete = target is None or final_size >= target
    return DownloadResult(
        url=url, path=dest, bytes_on_disk=final_size, total_bytes=total,
        complete=complete, sha256=sha256_file(dest), elapsed_sec=time.time() - started,
        resumed_from=resumed_from, attempts=attempts, notes=notes,
    )


def estimate_bytes_for_positions(
    positions: int,
    total_bytes: int = 21_368_367_967,
    total_positions: int = 394_669_566,
    safety: float = 1.35,
) -> int:
    """Compressed prefix size needed for ``positions`` records.

    Both totals are the values the server actually reports for
    ``lichess_db_eval.jsonl.zst`` (checked via HTTP HEAD and the database
    index page). ``safety`` covers records we drop for low search depth.
    """
    per_position = total_bytes / float(total_positions)
    return int(positions * per_position * safety)
