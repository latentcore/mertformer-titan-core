from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_post_train_autorun_plan_only_writes_contracts(tmp_path: Path):
    report_out = tmp_path / "post_train_status.json"
    result = subprocess.run(
        [
            "python3",
            "scripts/post_train_autorun.py",
            "--plan-only",
            "--allow-missing-checkpoint",
            "--reports-dir",
            str(tmp_path),
            "--report-out",
            str(report_out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    payload = json.loads(report_out.read_text(encoding="utf-8"))
    assert payload["mode"] == "plan-only"
    assert payload["status"] == "planned"
    assert any(step["name"] == "training_outputs_bundle" for step in payload["steps"])
    assert (tmp_path / "post_train_automation_contract.md").exists()
    assert (tmp_path / "post_train_state_machine.md").exists()


def test_post_train_autorun_demo_only_allows_missing_checkpoint(tmp_path: Path):
    report_out = tmp_path / "post_train_status.json"
    result = subprocess.run(
        [
            "python3",
            "scripts/post_train_autorun.py",
            "--demo-only",
            "--allow-missing-checkpoint",
            "--reports-dir",
            str(tmp_path),
            "--report-out",
            str(report_out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    payload = json.loads(report_out.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert (tmp_path / "demo_bundle_manifest.json").exists()
    assert (tmp_path / "final_evidence_pack.md").exists()
