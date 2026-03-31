from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _load_final_orchestrator_module():
    script_path = ROOT / "scripts" / "final_orchestrator.py"
    spec = importlib.util.spec_from_file_location("final_orchestrator_module", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_final_orchestrator_plan_only_writes_contracts(tmp_path: Path):
    report_out = tmp_path / "final_orchestrator_status.json"
    result = subprocess.run(
        [
            "python3",
            "scripts/final_orchestrator.py",
            "--plan-only",
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
    assert payload["schema"] == "run_manifest_v1"
    assert payload["mode"] == "plan-only"
    assert (tmp_path / "run_contract.md").exists()
    assert (tmp_path / "expected_artifacts_list.md").exists()
    assert (tmp_path / "exit_code_standard.md").exists()
    assert (tmp_path / "post_train_automation_contract.md").exists()
    assert (tmp_path / "post_train_state_machine.md").exists()
    assert (ROOT / "interfaces" / "run_manifest_v1.schema.json").exists()


def test_final_orchestrator_lock_helpers(tmp_path: Path):
    module = _load_final_orchestrator_module()
    lock_path = tmp_path / "final_orchestrator.lock.json"

    acquired, existing = module.acquire_lock(lock_path, {"run_id": "lock-a"})
    assert acquired is True
    assert existing is None

    acquired_again, existing_again = module.acquire_lock(lock_path, {"run_id": "lock-b"})
    assert acquired_again is False
    assert existing_again["run_id"] == "lock-a"

    module.release_lock(lock_path)
    assert not lock_path.exists()


def test_build_train_command_uses_accelerate_entrypoint_from_venv():
    module = _load_final_orchestrator_module()
    cmd = module.build_train_command(str(ROOT / ".titan-venv" / "bin" / "python"), 29501)
    assert cmd[:3] == [str(ROOT / ".titan-venv" / "bin" / "python"), "-m", "accelerate.commands.launch"]
    assert "train/train.py" in cmd


def test_build_training_env_prefers_offline_clean_lane():
    module = _load_final_orchestrator_module()
    env = module.build_training_env(
        "auto",
        {
            "decision_reason_code": "READY_OFFLINE_CLEAN",
        },
    )
    assert env["TITAN_OFFLINE"] == "1"
    assert env["TITAN_REQUIRE_GATED_TEACHER"] == "0"
    assert env["TITAN_USE_TR_TOKENIZER"] == "1"
