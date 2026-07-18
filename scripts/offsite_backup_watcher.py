"""Off-site checkpoint backup watcher (BACKLOG "auto-wiring still open" item).

The 2026-05-14 2xH200 partial run's checkpoint was permanently lost -- no off-site
copy existed, and the rented machine's storage was reclaimed before it could be
retrieved. `runbooks/checkpoint_offsite_backup.md` documents the manual procedure
this watcher automates: poll the checkpoint directory, and once its newest file has
been stable (untouched) for a configurable window -- i.e. very likely not mid-write --
sync it out to an off-site destination, retrying with backoff on failure.

No-op by design unless TITAN_OFFSITE_BACKUP_DEST is set. Does not pick a cloud
provider for you (same reasoning as the runbook) -- it dispatches to aws/gsutil/rsync
based on the destination string's scheme, so any of the three already-documented
options in the runbook works without further code changes.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SOURCE_DIR = "checkpoints/mertformer_titan_prod"
DEFAULT_INTERVAL_SECONDS = 300.0
DEFAULT_STABILITY_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE_SECONDS = 5.0


def newest_mtime(checkpoint_dir: Path, exclude_suffixes: tuple[str, ...] = (".tmp",)) -> float | None:
    """Newest mtime among real (non-excluded) files under checkpoint_dir, or None
    if the directory doesn't exist or has no matching files yet."""
    if not checkpoint_dir.exists():
        return None
    newest: float | None = None
    for path in checkpoint_dir.rglob("*"):
        if not path.is_file():
            continue
        if any(path.name.endswith(suffix) for suffix in exclude_suffixes):
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if newest is None or mtime > newest:
            newest = mtime
    return newest


def is_safe_to_sync(checkpoint_dir: Path, stability_seconds: float = DEFAULT_STABILITY_SECONDS, now: float | None = None) -> bool:
    """True only if the newest file in checkpoint_dir has been untouched for at
    least stability_seconds -- a best-effort guard against syncing a checkpoint
    that is still mid-write. A partially-synced checkpoint is worse than no
    checkpoint (silent corruption vs. an honest "nothing here"), per the runbook."""
    now = time.time() if now is None else now
    newest = newest_mtime(checkpoint_dir)
    if newest is None:
        return False  # nothing to sync yet
    return (now - newest) >= stability_seconds


def build_sync_command(dest: str, source_dir: Path) -> list[str]:
    """Dispatch to aws/gsutil/rsync based on the destination scheme -- matches the
    three options already documented in runbooks/checkpoint_offsite_backup.md
    (a cloud bucket you control, your own machine over SSH, or a second rented
    storage volume). No provider is chosen here; the operator's TITAN_OFFSITE_BACKUP_DEST
    value decides."""
    source = str(source_dir)
    if dest.startswith("s3://"):
        return ["aws", "s3", "sync", source, dest, "--exclude", "*.tmp"]
    if dest.startswith("gs://"):
        return ["gsutil", "-m", "rsync", "-r", "-x", r".*\.tmp$", source, dest]
    # Anything else is treated as an rsync target: a local path or user@host:/path.
    src = source if source.endswith("/") else source + "/"
    dst = dest if dest.endswith("/") else dest + "/"
    return ["rsync", "-avz", "--exclude", "*.tmp", src, dst]


def sync_with_retry(
    cmd: list[str],
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
    run_fn=None,
    sleep_fn=None,
) -> bool:
    """Run cmd, retrying with exponential backoff on non-zero exit or exception.
    run_fn/sleep_fn are injectable so this is unit-testable without a real
    subprocess or a real sleep."""
    run_fn = run_fn or subprocess.run
    sleep_fn = sleep_fn or time.sleep
    last_error = ""
    for attempt in range(max_retries):
        try:
            result = run_fn(cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                return True
            stderr_tail = (result.stderr or "")[-500:]
            last_error = f"exit code {result.returncode}: {stderr_tail}"
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: any sync-tool failure is retryable
            last_error = str(exc)
        if attempt < max_retries - 1:
            sleep_fn(backoff_base_seconds * (2**attempt))
    print(f"[offsite-backup] sync failed after {max_retries} attempts: {last_error}", file=sys.stderr)
    return False


def main() -> int:
    dest = os.environ.get("TITAN_OFFSITE_BACKUP_DEST", "").strip()
    if not dest:
        print(
            "[offsite-backup] TITAN_OFFSITE_BACKUP_DEST not set -- watcher is a no-op. "
            "See runbooks/checkpoint_offsite_backup.md to configure a destination."
        )
        return 0

    source_dir = Path(os.environ.get("TITAN_OFFSITE_BACKUP_SRC", DEFAULT_SOURCE_DIR))
    interval = float(os.environ.get("TITAN_OFFSITE_BACKUP_INTERVAL_SECONDS", str(DEFAULT_INTERVAL_SECONDS)))
    stability = float(os.environ.get("TITAN_OFFSITE_BACKUP_STABILITY_SECONDS", str(DEFAULT_STABILITY_SECONDS)))

    print(
        f"[offsite-backup] watching {source_dir} -> {dest} every {interval:.0f}s "
        f"(stability window {stability:.0f}s)"
    )
    while True:
        if is_safe_to_sync(source_dir, stability_seconds=stability):
            cmd = build_sync_command(dest, source_dir)
            ok = sync_with_retry(cmd)
            status = "OK" if ok else "FAILED"
            print(f"[offsite-backup] sync {status} @ {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
        else:
            print("[offsite-backup] skip cycle: checkpoint dir empty or newest file still within stability window")
        time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main())
