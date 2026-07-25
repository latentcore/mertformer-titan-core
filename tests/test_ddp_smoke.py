"""
Tests for scripts/ddp_smoke.py -- the standalone, pre-spend 2-GPU DDP smoke test used
by scripts/pre45k_gate.py. Mirrors the decision-scenario coverage already proven for
scripts/kaggle_batch_runner.py::ddp_smoke_test() (see tests/test_kaggle_batch_runner.py),
adapted to this module's own (parameterized, dependency-light) signature.

The `accelerate launch` subprocess is faked throughout (no GPU/accelerate dependency in
CI); the polling/decision LOGIC around it is exercised for real.
"""
from __future__ import annotations

import sys
import time

import pytest

from scripts import ddp_smoke


class _FakeProc:
    """Stands in for subprocess.Popen. `poll_sequence` controls what proc.poll()
    returns across successive calls (None = still alive); the last non-None value
    becomes `returncode`."""

    def __init__(self, poll_sequence):
        self._seq = list(poll_sequence)
        self.returncode = None

    def poll(self):
        if not self._seq:
            return self.returncode
        val = self._seq.pop(0)
        if val is not None:
            self.returncode = val
        return val

    def send_signal(self, sig):
        pass

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


@pytest.fixture(autouse=True)
def _fast_smoke(monkeypatch):
    """Keep every test fast regardless of scenario -- no real waiting."""
    monkeypatch.setattr(ddp_smoke.time, "sleep", lambda s: None)


# ---------------------------------------------------------------------------
# (a) gpu_count gate -- the check is only meaningful for exactly 2 GPUs
# ---------------------------------------------------------------------------
def test_skips_entirely_when_not_2_gpus():
    for n in (0, 1, 3, 4, 8):
        result = ddp_smoke.run_ddp_smoke_test(gpu_count=n)
        assert result["skipped"] is True
        assert result["attempted"] is False
        assert result["status"] == "skipped_not_2_gpu"
        assert result["ok"] is False


def test_detect_gpu_count_falls_back_to_zero_with_no_torch_no_nvidia_smi(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, *a, **k):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    monkeypatch.setattr(
        ddp_smoke.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no nvidia-smi")),
    )
    assert ddp_smoke.detect_gpu_count() == 0


# ---------------------------------------------------------------------------
# (b) The three real decision scenarios, with a faked accelerate-launch subprocess
# ---------------------------------------------------------------------------
def test_falls_back_when_gpus_never_active(monkeypatch, tmp_path):
    # Real, recorded incident this pins (same class as kaggle_batch_runner's own fix):
    # samples all [0, 0] must NOT be reported as confirmed DDP.
    monkeypatch.setattr(ddp_smoke.subprocess, "Popen", lambda *a, **k: _FakeProc([None, None, None]))
    monkeypatch.setattr(ddp_smoke, "gpu_utilization_snapshot", lambda: [0, 0])
    result = ddp_smoke.run_ddp_smoke_test(gpu_count=2, budget_seconds=1, cwd=tmp_path)
    assert result["skipped"] is False
    assert result["attempted"] is True
    assert result["ok"] is False
    assert result["both_active_any_sample"] is False


def test_confirms_ddp_when_a_genuine_dual_gpu_sample_seen(monkeypatch, tmp_path):
    samples = iter([[0, 0], [0, 0], [55, 61], [0, 0]])
    monkeypatch.setattr(ddp_smoke.subprocess, "Popen", lambda *a, **k: _FakeProc([None] * 4))
    monkeypatch.setattr(ddp_smoke, "gpu_utilization_snapshot", lambda: next(samples, [0, 0]))
    result = ddp_smoke.run_ddp_smoke_test(gpu_count=2, budget_seconds=1, cwd=tmp_path)
    assert result["ok"] is True
    assert result["both_active_any_sample"] is True
    assert result["status"] == "timed_out"  # budget expired without the process exiting on its own


def test_falls_back_when_process_exits_with_error(monkeypatch, tmp_path):
    # Even if GPUs looked active, a nonzero exit must never yield a confirmed pass.
    monkeypatch.setattr(ddp_smoke.subprocess, "Popen", lambda *a, **k: _FakeProc([1]))
    monkeypatch.setattr(ddp_smoke, "gpu_utilization_snapshot", lambda: [70, 65])
    result = ddp_smoke.run_ddp_smoke_test(gpu_count=2, budget_seconds=1, cwd=tmp_path)
    assert result["ok"] is False
    assert result["status"] == "exited_error"


def test_falls_back_on_exception(monkeypatch, tmp_path):
    def _raise(*a, **k):
        raise RuntimeError("accelerate not installed")

    monkeypatch.setattr(ddp_smoke.subprocess, "Popen", _raise)
    result = ddp_smoke.run_ddp_smoke_test(gpu_count=2, budget_seconds=1, cwd=tmp_path)
    assert result["ok"] is False
    assert result["status"] == "exception"
    assert "RuntimeError" in result["error"]


def test_completed_process_with_zero_exit_and_no_active_sample_is_not_confirmed(monkeypatch, tmp_path):
    monkeypatch.setattr(ddp_smoke.subprocess, "Popen", lambda *a, **k: _FakeProc([0]))
    monkeypatch.setattr(ddp_smoke, "gpu_utilization_snapshot", lambda: [0, 0])
    result = ddp_smoke.run_ddp_smoke_test(gpu_count=2, budget_seconds=1, cwd=tmp_path)
    assert result["status"] == "completed"
    assert result["ok"] is False


# ---------------------------------------------------------------------------
# (c) Optional log_path plumbing
# ---------------------------------------------------------------------------
def test_writes_log_file_when_log_path_given(monkeypatch, tmp_path):
    monkeypatch.setattr(ddp_smoke.subprocess, "Popen", lambda *a, **k: _FakeProc([0]))
    monkeypatch.setattr(ddp_smoke, "gpu_utilization_snapshot", lambda: [])
    log_path = tmp_path / "nested" / "smoke.log"
    result = ddp_smoke.run_ddp_smoke_test(gpu_count=2, budget_seconds=1, cwd=tmp_path, log_path=log_path)
    assert result["attempted"] is True
    assert log_path.exists()
    assert "cmd:" in log_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# (d) CLI smoke -- exercised as a real subprocess to prove argparse/module wiring works
# ---------------------------------------------------------------------------
def test_cli_runs_and_reports_skip_on_non_2_gpu_host(tmp_path):
    import subprocess as real_subprocess
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    out_path = tmp_path / "ddp_smoke_cli.json"
    proc = real_subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.ddp_smoke",
            "--json-out",
            str(out_path),
            "--budget-seconds",
            "2",
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=30,
    )
    # A controlled, decisive exit (0=skip/confirmed, 1=not confirmed) -- never a crash.
    assert proc.returncode in (0, 1)
    assert out_path.exists()
    import json

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "gpu_count" in payload
    assert "ok" in payload
