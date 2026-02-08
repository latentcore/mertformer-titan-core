from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mertformer_sdk import cli
from mertformer_sdk import pilot


def test_cli_verify_forces_offline(monkeypatch, capsys):
    called = {"offline": None}

    def _fake_verify(*, project_root=None, offline=True):
        called["offline"] = offline
        return {"status": "pass", "exit_code": 0}

    monkeypatch.setattr(cli, "run_verify_all", _fake_verify)
    monkeypatch.setattr(sys, "argv", ["mertformer", "verify"])

    cli.main()
    out = capsys.readouterr().out
    assert called["offline"] is True
    assert '"status": "pass"' in out


def test_pilot_report_fields_match_verify_summary(tmp_path: Path):
    verify_output = """
[verify] Secret scan ...
OK: no secret patterns detected in tracked files.
[verify] Pytest ...
21 passed, 4 skipped in 9.56s
[verify] Preflight (offline) ...
RESULT: 🏆 ALL GREEN
{
  "status": "completed",
  "results": [
    {"step": "nan_kill_switch", "status": "pass"},
    {"step": "checkpoint_restore_drill", "status": "pass"},
    {"step": "failure_budget_drill", "status": "pass"},
    {"step": "overfit_gate", "status": "pass_fast"},
    {"step": "golden_samples", "status": "pass"}
  ]
}
[verify] OK
""".strip()

    summary = pilot.parse_verify_output(verify_output, 0)
    report = pilot.build_pilot_report(project_root=tmp_path, verify_summary=summary)

    verify_gate = report["gate_results"]["verify_all"]
    assert verify_gate["status"] == summary["status"]
    assert verify_gate["secret_scan_pass"] == summary["secret_scan_pass"]
    assert verify_gate["pytest_pass"] == summary["pytest_pass"]
    assert verify_gate["preflight_pass"] == summary["preflight_pass"]
    assert verify_gate["operator_gate_pass"] == summary["operator_gate_pass"]

    operator_steps = report["gate_results"]["operator_mode_steps"]
    assert operator_steps["nan_kill_switch"] == "pass"
    assert operator_steps["overfit_gate"] == "pass_fast"


def test_parse_verify_output_marks_pytest_failures_false():
    verify_output = """
[verify] Pytest ...
27 passed, 1 failed, 4 skipped in 6.48s
[verify] Preflight (offline) ...
RESULT: 🏆 ALL GREEN
[verify] OK
""".strip()
    summary = pilot.parse_verify_output(verify_output, 1)
    assert summary["pytest_summary"]["passed"] == 27
    assert summary["pytest_summary"]["failed"] == 1
    assert summary["pytest_pass"] is False


def test_cli_pilot_report_writes_output(monkeypatch, tmp_path: Path):
    fake_summary = {
        "status": "pass",
        "exit_code": 0,
        "secret_scan_pass": True,
        "pytest_pass": True,
        "pytest_summary": {"passed": 21, "skipped": 4, "warnings": 0},
        "preflight_pass": True,
        "operator_gate_pass": True,
        "verify_script_pass": True,
        "operator_steps": {"nan_kill_switch": "pass"},
    }
    monkeypatch.setattr(cli, "run_verify_all", lambda **_: fake_summary)

    out_path = tmp_path / "pilot_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["mertformer", "pilot-report", "--out", str(out_path)],
    )
    cli.main()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "pilot_report_v1"
    assert payload["gate_results"]["verify_all"]["status"] == "pass"


def test_pilot_report_skip_verify_reason(tmp_path: Path):
    skipped_summary = {
        "status": "skipped",
        "exit_code": 0,
        "secret_scan_pass": None,
        "pytest_pass": None,
        "pytest_summary": {},
        "preflight_pass": None,
        "operator_gate_pass": None,
        "verify_script_pass": None,
        "operator_steps": {},
    }

    payload = pilot.build_pilot_report(project_root=tmp_path, verify_summary=skipped_summary)
    reasons = payload["benchmark_eligibility"]["reasons"]

    assert "verify_all skipped" in reasons
    assert "verify_all failed" not in reasons


def test_no_desktop_paths_in_tracked_files():
    result = subprocess.run(
        [
            "git",
            "grep",
            "-n",
            "/Users/mertyunlu/Desktop/",
            "--",
            ".",
            ":(exclude)tests/test_sdk_pilot_cli.py",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout


def test_readme_config_sample_matches_current_defaults():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "use_torch_compile = False" in readme
    assert "gradient_checkpoint_policy" not in readme
