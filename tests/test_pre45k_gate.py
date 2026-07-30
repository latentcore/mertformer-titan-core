"""
Tests for scripts/pre45k_gate.py -- the report-builder and verdict-combination logic
that chains the offline preflight, the dry-run preview, and the DDP smoke test (see
scripts/ddp_smoke.py). Every step here is injected as a fake dict (no real subprocess,
GPU, or network I/O) so these tests are fast and hardware-independent; the real
end-to-end wiring was verified manually (see BACKLOG.md B8) and is additionally covered
by test_ddp_smoke.py's own real-subprocess CLI test and the pre-existing
test_titan_preflight_contract.py / zero_touch_start.sh coverage.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import pre45k_gate as gate


def _ok_step(cmd="stub cmd"):
    return {"cmd": cmd, "return_code": 0, "stdout_tail": "", "stderr_tail": "", "ok": True}


def _fail_step(cmd="stub cmd"):
    return {"cmd": cmd, "return_code": 1, "stdout_tail": "", "stderr_tail": "boom", "ok": False}


def _ddp_skipped():
    return {
        "gpu_count": 0,
        "skipped": True,
        "attempted": False,
        "status": "skipped_not_2_gpu",
        "both_active_any_sample": False,
        "samples": [],
        "wall_seconds": 0.0,
        "ok": False,
        "error": None,
    }


def _ddp_confirmed():
    return {
        "gpu_count": 2,
        "skipped": False,
        "attempted": True,
        "status": "timed_out",
        "both_active_any_sample": True,
        "samples": [[55, 61]],
        "wall_seconds": 12.3,
        "ok": True,
        "error": None,
    }


def _ddp_unconfirmed():
    return {
        "gpu_count": 2,
        "skipped": False,
        "attempted": True,
        "status": "completed",
        "both_active_any_sample": False,
        "samples": [[0, 0]],
        "wall_seconds": 3.1,
        "ok": False,
        "error": None,
    }


# ---------------------------------------------------------------------------
# (a) sanitize_text -- redact absolute repo root and home-relative paths
# ---------------------------------------------------------------------------
def test_sanitize_text_redacts_repo_root():
    text = f"some output at {gate.ROOT}/scripts/foo.py failed"
    out = gate.sanitize_text(text)
    assert str(gate.ROOT) not in out
    assert "<REPO_ROOT>" in out


def test_sanitize_text_redacts_home_paths():
    text = "leaked path: /Users/someone/Downloads/secret_notes.md here"
    out = gate.sanitize_text(text)
    assert "someone" not in out
    assert "<HOME_PATH>" in out


# ---------------------------------------------------------------------------
# (b) combine_verdict -- pure decision logic, every branch
# ---------------------------------------------------------------------------
def test_verdict_blocked_when_offline_preflight_fails():
    ok, verdict = gate.combine_verdict(
        offline_preflight=_fail_step(), dry_run_preview=_ok_step(), ddp=_ddp_skipped(), strict_ddp=False
    )
    assert ok is False
    assert verdict == "BLOCKED"


def test_verdict_blocked_when_dry_run_preview_fails():
    ok, verdict = gate.combine_verdict(
        offline_preflight=_ok_step(), dry_run_preview=_fail_step(), ddp=_ddp_skipped(), strict_ddp=False
    )
    assert ok is False
    assert verdict == "BLOCKED"


def test_verdict_pass_not_applicable_when_ddp_skipped():
    ok, verdict = gate.combine_verdict(
        offline_preflight=_ok_step(), dry_run_preview=_ok_step(), ddp=_ddp_skipped(), strict_ddp=False
    )
    assert ok is True
    assert verdict == "PASS_DDP_NOT_APPLICABLE"


def test_verdict_pass_confirmed_when_ddp_confirmed():
    ok, verdict = gate.combine_verdict(
        offline_preflight=_ok_step(), dry_run_preview=_ok_step(), ddp=_ddp_confirmed(), strict_ddp=False
    )
    assert ok is True
    assert verdict == "PASS_DDP_CONFIRMED"


def test_verdict_pass_unconfirmed_non_strict_is_not_blocking():
    # Structural checks pass, DDP is inconclusive, but strict_ddp is off -> still ok=True.
    ok, verdict = gate.combine_verdict(
        offline_preflight=_ok_step(), dry_run_preview=_ok_step(), ddp=_ddp_unconfirmed(), strict_ddp=False
    )
    assert ok is True
    assert verdict == "PASS_DDP_UNCONFIRMED"


def test_verdict_blocked_ddp_when_strict_and_unconfirmed():
    ok, verdict = gate.combine_verdict(
        offline_preflight=_ok_step(), dry_run_preview=_ok_step(), ddp=_ddp_unconfirmed(), strict_ddp=True
    )
    assert ok is False
    assert verdict == "BLOCKED_DDP"


def test_verdict_strict_but_ddp_skipped_is_not_blocking():
    # strict_ddp should never punish a machine that genuinely doesn't have 2 GPUs.
    ok, verdict = gate.combine_verdict(
        offline_preflight=_ok_step(), dry_run_preview=_ok_step(), ddp=_ddp_skipped(), strict_ddp=True
    )
    assert ok is True
    assert verdict == "PASS_DDP_NOT_APPLICABLE"


def test_verdict_strict_and_confirmed_is_pass():
    ok, verdict = gate.combine_verdict(
        offline_preflight=_ok_step(), dry_run_preview=_ok_step(), ddp=_ddp_confirmed(), strict_ddp=True
    )
    assert ok is True
    assert verdict == "PASS_DDP_CONFIRMED"


# ---------------------------------------------------------------------------
# (c) build_report / build_report_md -- full report assembly with injected steps
# ---------------------------------------------------------------------------
def test_build_report_with_injected_steps_matches_combine_verdict():
    report = gate.build_report(
        python_bin="python3",
        strict_ddp=False,
        offline_preflight=_ok_step(),
        dry_run_preview=_ok_step(),
        ddp=_ddp_skipped(),
    )
    assert report["ok"] is True
    assert report["verdict"] == "PASS_DDP_NOT_APPLICABLE"
    assert report["strict_ddp"] is False
    assert "generated_utc" in report
    assert report["steps"]["offline_preflight"]["ok"] is True
    assert report["steps"]["ddp_smoke"]["skipped"] is True


def test_build_report_md_contains_verdict_and_claim_boundary():
    report = gate.build_report(
        python_bin="python3",
        strict_ddp=False,
        offline_preflight=_ok_step(),
        dry_run_preview=_ok_step(),
        ddp=_ddp_confirmed(),
    )
    md = gate.build_report_md(report)
    assert "PASS_DDP_CONFIRMED" in md
    assert "Claim boundary" in md
    assert "not a pass for DDP correctness" in md


# ---------------------------------------------------------------------------
# (d) main() end-to-end with injected steps -- via monkeypatched module functions
# ---------------------------------------------------------------------------
def test_main_writes_report_files_and_returns_zero_on_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "run_offline_preflight", lambda python_bin: _ok_step())
    monkeypatch.setattr(gate, "run_dry_run_preview", lambda: _ok_step())
    monkeypatch.setattr(gate, "run_ddp_smoke_test", lambda **kw: _ddp_skipped())

    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "pre45k_gate.py",
            "--python",
            "python3",
            "--report-json",
            str(report_json),
            "--report-md",
            str(report_md),
        ],
    )
    rc = gate.main()
    assert rc == 0
    assert report_json.exists()
    assert report_md.exists()
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["verdict"] == "PASS_DDP_NOT_APPLICABLE"


def test_main_returns_nonzero_when_structural_step_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "run_offline_preflight", lambda python_bin: _fail_step())
    monkeypatch.setattr(gate, "run_dry_run_preview", lambda: _ok_step())
    monkeypatch.setattr(gate, "run_ddp_smoke_test", lambda **kw: _ddp_skipped())

    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "pre45k_gate.py",
            "--report-json",
            str(report_json),
            "--report-md",
            str(report_md),
        ],
    )
    rc = gate.main()
    assert rc == 1
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["verdict"] == "BLOCKED"


def test_main_strict_ddp_flag_blocks_on_unconfirmed_ddp(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "run_offline_preflight", lambda python_bin: _ok_step())
    monkeypatch.setattr(gate, "run_dry_run_preview", lambda: _ok_step())
    monkeypatch.setattr(gate, "run_ddp_smoke_test", lambda **kw: _ddp_unconfirmed())

    report_json = tmp_path / "report.json"
    report_md = tmp_path / "report.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "pre45k_gate.py",
            "--strict-ddp",
            "--report-json",
            str(report_json),
            "--report-md",
            str(report_md),
        ],
    )
    rc = gate.main()
    assert rc == 1
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["verdict"] == "BLOCKED_DDP"


# ---------------------------------------------------------------------------
# (e) run_offline_preflight / run_dry_run_preview -- real subprocess against the
#     actual repo (proves the wiring genuinely works, not just the mocked logic above).
# ---------------------------------------------------------------------------
def _resolve_repo_python() -> str:
    """Mirrors scripts/pre45k_gate.sh's own interpreter selection: prefer the venv
    python (which has the repo importable as editable), fall back to plain python3
    only if the venv is absent. A bare system `python3` without the repo installed
    editable cannot resolve `from config.build_label import ...` when the script is
    invoked script-style (its own directory lands on sys.path[0], not the repo root)."""
    venv_python = gate.ROOT / ".titan-venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return "python3"


def _corpus_is_materialized() -> bool:
    """True when the (gitignored) stage corpus exists locally.

    Asks scripts/titan_preflight._stage_jsonl_paths() rather than hardcoding paths: the
    stage directories are NOT uniformly named (stage1..stage3, but `stage4_soul` and
    `stage5_tools`), and a hand-written path list here silently returned False on a machine
    that did have the corpus. Reading the preflight's own source of truth means this can
    never disagree with the check it is guarding.
    """
    from scripts import titan_preflight

    return all(path.exists() for path in titan_preflight._stage_jsonl_paths().values())


def test_run_offline_preflight_against_real_repo():
    """[2026-07-30] Skips without the corpus instead of failing.

    Found by the mandatory clean-clone verification: this test asserted `ok is True`
    unconditionally, but `scripts/titan_preflight.py` correctly FAILS with
    "Stage JSONL missing: stage1..stage5" when the corpus is absent -- which is the state of
    every fresh clone, every CI runner and every new contributor's machine, since the corpus
    is gitignored and cannot be committed (23.59B tokens). It passed only here, on the
    machine that happens to hold the local data.

    That made it a public-release blocker rather than a cosmetic issue: the repository's own
    PR rule is "`bash scripts/verify_all.sh` must pass with zero regressions", so a first-time
    contributor would have failed it through no fault of their own, on their first run.

    The real assertion is preserved wherever the corpus IS present, so this keeps its value
    on the training machine, which is the only place the check is meaningful.
    """
    if not _corpus_is_materialized():
        pytest.skip("stage corpus not materialized locally (datasets/ is gitignored)")
    result = gate.run_offline_preflight(_resolve_repo_python())
    assert result["ok"] is True
    assert "titan_preflight.py" in result["cmd"]


def test_offline_preflight_reports_the_missing_corpus_rather_than_passing(monkeypatch):
    """The other half: without the corpus the gate must FAIL, loudly and for that reason.

    A gate that went green on an empty corpus would be far worse than one that skips, so pin
    the honest-failure direction too. This runs everywhere, corpus present or not.

    [2026-07-31] Must also neutralize the two deliberate offline-preflight escape hatches for
    its OWN subprocess call, or this test's premise is defeated by its own environment:
    `scripts/titan_preflight.py::check_stage_jsonl()` returns `ok: True` on a missing corpus
    when either `GITHUB_ACTIONS`/`CI` is `"true"` (deliberate, CI runners never have the real
    corpus) or `TITAN_PREFLIGHT_ALLOW_MISSING_STAGE_JSONL=1` is set (deliberate, exported by
    `scripts/verify_all.sh` whenever `TITAN_OFFLINE=1`, which is CI's own default).
    `gate.run_offline_preflight()` does `env = dict(os.environ)`, so this test's own subprocess
    inherits whatever of those is ambient. Locally that's nothing, so this passed on a bare
    checkout -- but under real GitHub Actions CI (which runs via `bash scripts/verify_all.sh`)
    both were ambient and the subprocess got `ok: True`, failing this test's own assertion.
    Both escape hatches remain untouched and correct in production; only this test's own view
    of its environment is neutralized here, so it genuinely exercises "a real machine, no
    corpus, no override" regardless of what is ambient around it.
    """
    if _corpus_is_materialized():
        pytest.skip("corpus present; the missing-corpus path cannot be exercised here")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("TITAN_PREFLIGHT_ALLOW_MISSING_STAGE_JSONL", raising=False)
    result = gate.run_offline_preflight(_resolve_repo_python())
    assert result["ok"] is False, "preflight passed with no corpus on disk"
    combined = f"{result.get('stdout_tail', '')}{result.get('stderr_tail', '')}"
    assert "Stage JSONL missing" in combined, combined[-400:]


def test_run_dry_run_preview_against_real_repo():
    result = gate.run_dry_run_preview()
    assert result["ok"] is True
    assert "--dry-run" in result["cmd"]
