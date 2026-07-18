"""
[2026-07-19] Tests for scripts/offsite_backup_watcher.py (BACKLOG "off-site checkpoint
backup -- auto-wiring still open" item).

Covers the pure/injectable logic only (stability/partial-write detection, sync-command
construction per destination scheme, retry/backoff) -- no real subprocess, no real
network, no real sleep. The watcher's main() loop itself is not unit-tested (it's an
infinite poll loop by design); its building blocks are.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from scripts.offsite_backup_watcher import (
    build_sync_command,
    is_safe_to_sync,
    newest_mtime,
    sync_with_retry,
)


def test_newest_mtime_none_when_dir_missing(tmp_path):
    assert newest_mtime(tmp_path / "does_not_exist") is None


def test_newest_mtime_none_when_dir_empty(tmp_path):
    assert newest_mtime(tmp_path) is None


def test_newest_mtime_ignores_tmp_files(tmp_path):
    (tmp_path / "best.pt").write_text("x")
    (tmp_path / "latest.pt.tmp").write_text("y")
    import os

    # Make the .tmp file's mtime clearly newer so ignoring it is actually exercised.
    real_mtime = (tmp_path / "best.pt").stat().st_mtime
    os.utime(tmp_path / "latest.pt.tmp", (real_mtime + 100, real_mtime + 100))
    assert newest_mtime(tmp_path) == pytest.approx(real_mtime, abs=1.0)


def test_is_safe_to_sync_false_when_empty(tmp_path):
    assert is_safe_to_sync(tmp_path, stability_seconds=30.0) is False


def test_is_safe_to_sync_false_when_recently_written(tmp_path):
    (tmp_path / "step_1000.pt").write_text("x")
    now = time.time()
    # Written "now" -- well within the 30s stability window -- must be treated as
    # possibly-mid-write and skipped.
    assert is_safe_to_sync(tmp_path, stability_seconds=30.0, now=now) is False


def test_is_safe_to_sync_true_once_stable(tmp_path):
    path = tmp_path / "step_1000.pt"
    path.write_text("x")
    real_mtime = path.stat().st_mtime
    # Simulate polling 60 seconds later -- well past the 30s stability window.
    later = real_mtime + 60
    assert is_safe_to_sync(tmp_path, stability_seconds=30.0, now=later) is True


def test_build_sync_command_s3():
    cmd = build_sync_command("s3://my-bucket/prefix", Path("checkpoints/mertformer_titan_prod"))
    assert cmd[:3] == ["aws", "s3", "sync"]
    assert "s3://my-bucket/prefix" in cmd
    assert "--exclude" in cmd and "*.tmp" in cmd


def test_build_sync_command_gcs():
    cmd = build_sync_command("gs://my-bucket/prefix", Path("checkpoints/mertformer_titan_prod"))
    assert cmd[0] == "gsutil"
    assert "gs://my-bucket/prefix" in cmd


def test_build_sync_command_rsync_ssh_target():
    cmd = build_sync_command("you@home-machine:/backup/mertformer", Path("checkpoints/mertformer_titan_prod"))
    assert cmd[0] == "rsync"
    assert cmd[-1] == "you@home-machine:/backup/mertformer/"
    assert cmd[-2].endswith("mertformer_titan_prod/")


def test_build_sync_command_rsync_local_path(tmp_path):
    dest = str(tmp_path / "backup_dest")
    cmd = build_sync_command(dest, Path("checkpoints/mertformer_titan_prod"))
    assert cmd[0] == "rsync"
    assert cmd[-1] == dest + "/"


class _FakeResult:
    def __init__(self, returncode: int, stderr: str = ""):
        self.returncode = returncode
        self.stderr = stderr


def test_sync_with_retry_succeeds_first_try():
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _FakeResult(0)

    ok = sync_with_retry(["echo", "hi"], run_fn=fake_run, sleep_fn=lambda s: None)
    assert ok is True
    assert len(calls) == 1


def test_sync_with_retry_retries_then_succeeds():
    attempts = {"n": 0}
    sleeps = []

    def fake_run(cmd, **kwargs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return _FakeResult(1, stderr="transient network error")
        return _FakeResult(0)

    ok = sync_with_retry(
        ["rsync", "x", "y"],
        max_retries=3,
        backoff_base_seconds=1.0,
        run_fn=fake_run,
        sleep_fn=lambda s: sleeps.append(s),
    )
    assert ok is True
    assert attempts["n"] == 3
    # Exponential backoff: 1.0 * 2**0, then 1.0 * 2**1 between the two failed attempts.
    assert sleeps == [1.0, 2.0]


def test_sync_with_retry_exhausts_and_fails():
    def fake_run(cmd, **kwargs):
        return _FakeResult(1, stderr="permanent failure")

    ok = sync_with_retry(
        ["rsync", "x", "y"],
        max_retries=2,
        backoff_base_seconds=0.01,
        run_fn=fake_run,
        sleep_fn=lambda s: None,
    )
    assert ok is False


def test_sync_with_retry_handles_exception_from_run_fn():
    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("aws: command not found")

    ok = sync_with_retry(
        ["aws", "s3", "sync", "x", "y"],
        max_retries=2,
        backoff_base_seconds=0.01,
        run_fn=fake_run,
        sleep_fn=lambda s: None,
    )
    assert ok is False
