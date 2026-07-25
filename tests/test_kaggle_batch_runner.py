"""
[2026-07-25] Tests for scripts/kaggle_batch_runner.py -- the standalone Kaggle-Dataset
orchestrator added to the repo after a real 2026-07-25 Kaggle run produced the N3/N4/
36M/171M results recorded in BACKLOG.md. This file turns the ad-hoc verifications done
during that build (lock exclusivity, SIGTERM->grace->SIGKILL, patch_constant, the DDP
smoke test's three decision paths, and a full ordering/budget/manifest dry-run) into
permanent regression tests.

Real subprocesses are used where the behavior being tested IS subprocess/signal
handling (SIGTERM->grace->SIGKILL, lock exclusivity). The DDP smoke test's
`accelerate launch` subprocess is faked (no GPU/accelerate dependency in CI), but the
polling/decision LOGIC around it is exercised for real.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts import kaggle_batch_runner as kbr


# ---------------------------------------------------------------------------
# (a) Lock mechanism
# ---------------------------------------------------------------------------
def test_acquire_lock_succeeds_then_rejects_second_call(tmp_path, monkeypatch):
    monkeypatch.setattr(kbr, "WORKING_DIR", tmp_path)
    monkeypatch.setattr(kbr, "LOCK_PATH", tmp_path / "batch_runner.lock")

    assert kbr.acquire_lock() is True
    # A second call, before release, must be rejected -- this is the exact
    # scenario that caused a real duplicate-run incident on Kaggle.
    assert kbr.acquire_lock() is False
    assert kbr.LOCK_PATH.exists()


def test_release_lock_then_acquire_succeeds_again(tmp_path, monkeypatch):
    monkeypatch.setattr(kbr, "WORKING_DIR", tmp_path)
    monkeypatch.setattr(kbr, "LOCK_PATH", tmp_path / "batch_runner.lock")

    assert kbr.acquire_lock() is True
    kbr.release_lock()
    assert not kbr.LOCK_PATH.exists()
    assert kbr.acquire_lock() is True


def test_release_lock_is_safe_when_never_acquired(tmp_path, monkeypatch):
    monkeypatch.setattr(kbr, "WORKING_DIR", tmp_path)
    monkeypatch.setattr(kbr, "LOCK_PATH", tmp_path / "batch_runner.lock")
    kbr.release_lock()  # must not raise


# ---------------------------------------------------------------------------
# (b) SIGTERM -> grace -> SIGKILL, using real subprocesses
# ---------------------------------------------------------------------------
def test_run_timeboxed_well_behaved_process_completes_normally(tmp_path):
    result = kbr.run_timeboxed(
        cmd=[sys.executable, "-c", "print('hi')"],
        cwd=tmp_path,
        log_path=tmp_path / "run.log",
        budget_seconds=10.0,
    )
    assert result["status"] == "completed"
    assert result["returncode"] == 0
    assert result["error"] is None


def test_run_timeboxed_kills_a_stubborn_process_that_ignores_sigterm(tmp_path, monkeypatch):
    # Shorten the grace period so the test is fast; the stubborn process below
    # traps SIGTERM (ignores it) so the SIGKILL branch is genuinely exercised,
    # not just the SIGTERM path.
    monkeypatch.setattr(kbr, "SIGTERM_GRACE_SECONDS", 2)
    stubborn_script = (
        "import signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(30)"
    )
    t0 = time.time()
    result = kbr.run_timeboxed(
        cmd=[sys.executable, "-c", stubborn_script],
        cwd=tmp_path,
        log_path=tmp_path / "run.log",
        budget_seconds=0.5,
    )
    wall = time.time() - t0
    assert result["status"] == "timed_out"
    # Must have actually died within budget + grace + a small buffer, not run the full 30s sleep.
    assert wall < 10.0


def test_run_timeboxed_well_behaved_process_respects_sigterm(tmp_path, monkeypatch):
    monkeypatch.setattr(kbr, "SIGTERM_GRACE_SECONDS", 5)
    # No SIGTERM handler installed -> default Python behavior is to exit on SIGTERM,
    # so this should stop well within the grace period (no SIGKILL needed).
    result = kbr.run_timeboxed(
        cmd=[sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        log_path=tmp_path / "run.log",
        budget_seconds=0.5,
    )
    assert result["status"] == "timed_out"
    assert result["wall_seconds"] < 5.0


def test_run_timeboxed_nonzero_exit_is_failed_not_timed_out(tmp_path):
    result = kbr.run_timeboxed(
        cmd=[sys.executable, "-c", "import sys; sys.exit(1)"],
        cwd=tmp_path,
        log_path=tmp_path / "run.log",
        budget_seconds=10.0,
    )
    assert result["status"] == "failed"
    assert result["returncode"] == 1
    assert "nonzero exit 1" in result["error"]


# ---------------------------------------------------------------------------
# (c) patch_constant
# ---------------------------------------------------------------------------
def test_patch_constant_replaces_the_correct_line(tmp_path):
    f = tmp_path / "train_nutrition5k.py"
    f.write_text("X = 1\nLIQUID_LAYER_IDS = (5,)\nY = 2\n")
    kbr.patch_constant(f, r"LIQUID_LAYER_IDS = \(5,\)", "LIQUID_LAYER_IDS = ()")
    text = f.read_text()
    assert "LIQUID_LAYER_IDS = ()" in text
    assert "X = 1" in text and "Y = 2" in text  # untouched neighbors


def test_patch_constant_raises_when_pattern_missing(tmp_path):
    f = tmp_path / "train_nutrition5k.py"
    f.write_text("X = 1\n")
    with pytest.raises(RuntimeError, match="found 0"):
        kbr.patch_constant(f, r"LIQUID_LAYER_IDS = \(5,\)", "LIQUID_LAYER_IDS = ()")


def test_patch_constant_raises_when_pattern_matches_more_than_once(tmp_path):
    f = tmp_path / "dup.py"
    f.write_text("A = (5,)\nA = (5,)\n")
    with pytest.raises(RuntimeError, match="found 2"):
        kbr.patch_constant(f, r"A = \(5,\)", "A = ()")


def test_patch_constant_against_the_real_repo_scripts(tmp_path):
    """Regression pin: the exact patterns run_nutrition5k_ablation() uses against
    the real, tracked scripts/train_nutrition5k.py must still match exactly once."""
    real = Path(__file__).resolve().parent.parent / "scripts" / "train_nutrition5k.py"
    text = real.read_text(encoding="utf-8")
    import re

    assert len(re.findall(r"LIQUID_LAYER_IDS = \(5,\)", text)) == 1
    assert len(re.findall(r"MOE_LAYER_IDS = \(3, 6\)", text)) == 1


# ---------------------------------------------------------------------------
# (d) DDP smoke test -- three decision scenarios, real polling logic, faked subprocess
# ---------------------------------------------------------------------------
class _FakeProc:
    """Stands in for subprocess.Popen. `poll_sequence` controls what proc.poll()
    returns across successive calls (None = still alive); the last non-None value
    becomes `returncode`."""

    def __init__(self, poll_sequence):
        self._seq = list(poll_sequence)
        self.returncode = None
        self._sent_sigterm = False

    def poll(self):
        if not self._seq:
            return self.returncode
        val = self._seq.pop(0)
        if val is not None:
            self.returncode = val
        return val

    def send_signal(self, sig):
        self._sent_sigterm = True

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


@pytest.fixture
def ddp_env_stub(tmp_path, monkeypatch):
    """Common wiring so ddp_smoke_test()'s stage_job_repo()/JOBS_DIR calls succeed
    without touching a real repo snapshot."""
    snapshot = tmp_path / "repo_snapshot"
    (snapshot / "scripts").mkdir(parents=True)
    (snapshot / "scripts" / "preflight_run.py").write_text("# stub\n")
    monkeypatch.setattr(kbr, "REPO_SNAPSHOT_SRC", snapshot)
    monkeypatch.setattr(kbr, "JOBS_DIR", tmp_path / "output")
    monkeypatch.setattr(kbr, "DDP_SMOKE_TEST_SECONDS", 1)  # keep the test fast
    return tmp_path


def test_ddp_smoke_test_skips_entirely_when_not_2_gpus(ddp_env_stub):
    assert kbr.ddp_smoke_test({"gpu_count": 1}) is False
    assert kbr.ddp_smoke_test({"gpu_count": 0}) is False


def test_ddp_smoke_test_falls_back_when_gpus_never_active(ddp_env_stub, monkeypatch):
    # Real, recorded incident this test pins: samples all [0, 0] must NOT be "USE DDP".
    monkeypatch.setattr(kbr.subprocess, "Popen", lambda *a, **k: _FakeProc([None, None, None]))
    monkeypatch.setattr(kbr, "gpu_utilization_snapshot", lambda: [0, 0])
    monkeypatch.setattr(kbr.time, "sleep", lambda s: None)  # don't actually wait in the test
    assert kbr.ddp_smoke_test({"gpu_count": 2}) is False


def test_ddp_smoke_test_uses_ddp_when_a_genuine_dual_gpu_sample_seen(ddp_env_stub, monkeypatch):
    samples = iter([[0, 0], [0, 0], [55, 61], [0, 0]])
    monkeypatch.setattr(kbr.subprocess, "Popen", lambda *a, **k: _FakeProc([None] * 4))
    monkeypatch.setattr(kbr, "gpu_utilization_snapshot", lambda: next(samples, [0, 0]))
    monkeypatch.setattr(kbr.time, "sleep", lambda s: None)
    assert kbr.ddp_smoke_test({"gpu_count": 2}) is True


def test_ddp_smoke_test_falls_back_when_process_exits_with_error(ddp_env_stub, monkeypatch):
    # Even if GPUs looked active, a nonzero exit must never yield "USE DDP".
    monkeypatch.setattr(kbr.subprocess, "Popen", lambda *a, **k: _FakeProc([1]))
    monkeypatch.setattr(kbr, "gpu_utilization_snapshot", lambda: [70, 65])
    monkeypatch.setattr(kbr.time, "sleep", lambda s: None)
    assert kbr.ddp_smoke_test({"gpu_count": 2}) is False


def test_ddp_smoke_test_falls_back_on_exception(ddp_env_stub, monkeypatch):
    def _raise(*a, **k):
        raise RuntimeError("accelerate not installed")

    monkeypatch.setattr(kbr.subprocess, "Popen", _raise)
    assert kbr.ddp_smoke_test({"gpu_count": 2}) is False


# ---------------------------------------------------------------------------
# (e) Full dry-run: ordering + budget + incomplete-job-excluded + manifest integrity
# ---------------------------------------------------------------------------
def test_full_batch_dry_run_all_jobs_in_manifest_regardless_of_outcome(tmp_path, monkeypatch):
    """Pins the real 2026-07-25 bug fix: an early `continue` on a skipped job used to
    bypass the incremental build_final_zip() call, so a skipped LAST job never made it
    into the final manifest at all. This test asserts all 5 jobs always appear."""
    monkeypatch.setattr(kbr, "WORKING_DIR", tmp_path)
    monkeypatch.setattr(kbr, "JOBS_DIR", tmp_path / "mertformer_batch_output")
    monkeypatch.setattr(kbr, "LOCK_PATH", tmp_path / "batch_runner.lock")
    monkeypatch.setattr(kbr, "RUN_STARTED_AT", time.time())
    # A tiny total budget so job 5 (budget_hours=None) is deliberately skipped
    # ("insufficient time remaining") -- the scenario that exposed the original bug.
    monkeypatch.setattr(kbr, "TOTAL_BUDGET_HOURS", 0.05)  # 3 minutes
    monkeypatch.setattr(kbr, "SAFETY_MARGIN_MINUTES", 0.0)
    monkeypatch.setattr(kbr, "MIN_LAST_JOB_MINUTES", 15.0)

    monkeypatch.setattr(kbr, "detect_environment", lambda: {
        "generated_utc": "2026-01-01T00:00:00Z", "is_kaggle": False, "gpu_count": 0,
        "working_dir": str(tmp_path), "repo_snapshot_src": "", "total_budget_hours": 0.05,
        "safety_margin_minutes": 0.0,
    })
    monkeypatch.setattr(kbr, "ddp_smoke_test", lambda env_info: False)

    call_order = []

    def _fake_nutrition(job, budget_seconds):
        call_order.append(job["name"])
        return {"status": "completed", "wall_seconds": 1.0, "key_metrics": {"calorie_mae": 1.0},
                "output_source": None, "repo_dir": tmp_path / "unused"}

    def _fake_lm(job, budget_seconds, use_ddp):
        call_order.append(job["name"])
        # 36M "fails" (mirrors the real 2026-07-25 divergence/safety-brake outcome);
        # 171M never gets a chance in this tiny-budget scenario in practice, but if it
        # runs, behave the same way.
        return {"status": "failed", "wall_seconds": 1.0, "error": "nonzero exit 1",
                "key_metrics": None, "output_source": None, "repo_dir": tmp_path / "unused",
                "used_ddp": use_ddp}

    monkeypatch.setattr(kbr, "run_nutrition5k_ablation", _fake_nutrition)
    monkeypatch.setattr(kbr, "run_lm_preflight", _fake_lm)
    # run_chess should never even be reached if the budget truly runs out first;
    # if it is reached, it must not raise.
    monkeypatch.setattr(kbr, "run_chess", lambda job, budget_seconds: {
        "status": "skipped", "wall_seconds": 0.0, "error": "not reached in this test",
        "key_metrics": None,
    })

    rc = kbr._run_batch()
    assert rc == 0

    manifest_path = tmp_path / "MertFormer_Kaggle_Batch_Output_manifest.json"
    assert manifest_path.exists()
    import json

    manifest = json.loads(manifest_path.read_text())
    names = [j["name"] for j in manifest["jobs"]]
    # All 5 jobs must appear, in JOBS order, regardless of completed/failed/skipped.
    assert names == [j["name"] for j in kbr.JOBS]
    assert len(manifest["jobs"]) == 5

    # A "failed" job must never be marked included_in_zip.
    for entry in manifest["jobs"]:
        if entry["status"] != "completed":
            assert entry["included_in_zip"] is False

    # Call order (for the jobs that did run) must match JOBS order, not be reordered.
    assert call_order == sorted(call_order, key=lambda n: [j["name"] for j in kbr.JOBS].index(n))


def test_collect_job_output_excludes_non_completed_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(kbr, "JOBS_DIR", tmp_path)
    src = tmp_path / "source_dir"
    src.mkdir()
    (src / "REPORT.md").write_text("hi")

    job = {"name": "some_job"}
    failed_result = {"status": "failed", "output_source": src}
    assert kbr.collect_job_output(job, failed_result) is None

    completed_result = {"status": "completed", "output_source": src}
    dest = kbr.collect_job_output(job, completed_result)
    assert dest is not None
    assert (dest / "REPORT.md").exists()
