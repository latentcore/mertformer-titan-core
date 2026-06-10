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


def test_build_train_command_uses_accelerate_entrypoint_from_venv(monkeypatch):
    monkeypatch.delenv("ACCELERATE_CONFIG_FILE", raising=False)
    module = _load_final_orchestrator_module()
    cmd = module.build_train_command(str(ROOT / ".titan-venv" / "bin" / "python"), 29501)
    assert cmd[:3] == [str(ROOT / ".titan-venv" / "bin" / "python"), "-m", "accelerate.commands.launch"]
    assert "train/train.py" in cmd


def test_build_train_command_uses_accelerate_config_file(monkeypatch):
    monkeypatch.setenv("ACCELERATE_CONFIG_FILE", "repro/accelerate_2xh200.yaml")
    module = _load_final_orchestrator_module()
    monkeypatch.setattr(module, "detect_num_processes", lambda: 2)
    cmd = module.build_train_command("python3", 29501)
    assert cmd[:5] == [
        "python3",
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        "repro/accelerate_2xh200.yaml",
    ]
    assert cmd[5:7] == ["--num_processes", "2"]


def test_run_training_with_batch_fallback_retries_clear_oom_only(monkeypatch):
    module = _load_final_orchestrator_module()
    attempts: list[int] = []

    def fake_run_command(root, cmd, env=None):
        batch_size = int(env["TITAN_BATCH_SIZE"])
        attempts.append(batch_size)
        if batch_size == 1024:
            return {
                "cmd": "train",
                "return_code": 0,
                "ok": True,
                "stdout_tail": "Safety brake stop finalized. reason=oom_backoff_exhausted",
                "stderr_tail": "",
            }
        return {"cmd": "train", "return_code": 0, "ok": True, "stdout_tail": "done", "stderr_tail": ""}

    monkeypatch.setattr(module, "run_command", fake_run_command)
    result = module.run_training_with_batch_fallback(
        ROOT,
        ["python3", "train/train.py"],
        {
            "TITAN_BATCH_SIZE": "1024",
            "TITAN_BATCH_SIZE_FALLBACKS": "1024,512,256",
        },
    )
    assert result["ok"] is True
    assert attempts == [1024, 512]
    assert result["batch_size_attempted"] == 512
    assert result["batch_fallback_used"] is True
    assert [item["batch_size"] for item in result["batch_fallback_attempts"]] == [1024, 512]
    assert result["fallback_policy"] == "clear_oom_only"


def test_run_training_with_batch_fallback_stops_on_non_oom(monkeypatch):
    module = _load_final_orchestrator_module()
    attempts: list[int] = []

    def fake_run_command(root, cmd, env=None):
        attempts.append(int(env["TITAN_BATCH_SIZE"]))
        return {
            "cmd": "train",
            "return_code": 17,
            "ok": False,
            "stdout_tail": "",
            "stderr_tail": "teacher artifact missing",
        }

    monkeypatch.setattr(module, "run_command", fake_run_command)
    result = module.run_training_with_batch_fallback(
        ROOT,
        ["python3", "train/train.py"],
        {
            "TITAN_BATCH_SIZE": "1024",
            "TITAN_BATCH_SIZE_FALLBACKS": "1024,512,256",
        },
    )
    assert result["ok"] is False
    assert attempts == [1024]
    assert result["batch_size_attempted"] == 1024
    assert result["batch_fallback_used"] is False
    assert result["batch_fallback_attempts"][0]["clear_oom"] is False


def test_build_training_env_prefers_offline_clean_lane():
    module = _load_final_orchestrator_module()
    env = module.build_training_env(
        "auto",
        {
            "decision_reason_code": "READY_OFFLINE_CLEAN",
        },
    )
    assert env["TITAN_OFFLINE"] == "1"
    assert env["TITAN_REQUIRE_GATED_TEACHER"] == "1"
    assert env["TITAN_USE_PRECOMPUTED_LOGITS"] == "1"
    # [tier-2 BLOCKER fix] offline_clean is teacher-tokenizer KD; it must NOT force
    # the TR tokenizer (precompute stamps the teacher identity and refuses TR, so
    # TR=1 here would trip the logit-alignment gate and block every canonical run).
    assert env.get("TITAN_USE_TR_TOKENIZER") != "1"


def test_build_training_env_supports_remote_bootstrap_lane():
    module = _load_final_orchestrator_module()
    env = module.build_training_env(
        "auto",
        {
            "decision_reason_code": "READY_REMOTE_BOOTSTRAP",
            "recommended_path": "remote_bootstrap",
        },
    )
    assert env["TITAN_OFFLINE"] == "0"
    assert env["TITAN_RUNTIME_INJECTED_BOOTSTRAP"] == "1"


def test_run_start_gate_can_skip_verify_all(tmp_path: Path):
    module = _load_final_orchestrator_module()
    calls: dict[str, object] = {}

    def fake_run_command(root, cmd, env=None):
        calls["cmd"] = cmd
        out_path = tmp_path / "start_gate_report.json"
        out_path.write_text(
            json.dumps(
                {
                    "train_allowed": True,
                    "train_readiness_status": "TRAIN_ALLOWED",
                    "decision_reason_code": "READY_OFFLINE_CLEAN",
                    "recommended_path": "offline_clean",
                }
            ),
            encoding="utf-8",
        )
        return {"cmd": " ".join(cmd), "return_code": 0, "ok": True, "stdout_tail": "", "stderr_tail": ""}

    module.run_command = fake_run_command
    result, payload = module.run_start_gate(ROOT, sys.executable, tmp_path, skip_verify_all=True)
    assert result["ok"] is True
    assert payload["train_allowed"] is True
    assert "--skip-verify-all" in calls["cmd"]


def test_final_orchestrator_check_only_releases_lock_and_stays_fast(tmp_path: Path, monkeypatch):
    module = _load_final_orchestrator_module()
    report_out = tmp_path / "final_orchestrator_status.json"
    calls: dict[str, object] = {}

    def fake_build_contract_outputs(root, reports_dir):
        (reports_dir / "run_contract.md").write_text("contract\n", encoding="utf-8")
        (reports_dir / "expected_artifacts_list.md").write_text("artifacts\n", encoding="utf-8")
        (reports_dir / "exit_code_standard.md").write_text("codes\n", encoding="utf-8")
        (root / "interfaces").mkdir(parents=True, exist_ok=True)
        (root / "interfaces" / "run_manifest_v1.schema.json").write_text("{}", encoding="utf-8")

    def fake_run_post_plan(root, py, reports_dir):
        (reports_dir / "post_train_automation_contract.md").write_text("pta\n", encoding="utf-8")
        (reports_dir / "post_train_state_machine.md").write_text("pts\n", encoding="utf-8")
        return {"cmd": "post-plan", "return_code": 0, "ok": True, "stdout_tail": "planned", "stderr_tail": ""}

    def fake_run_start_gate(root, py, reports_dir, *, skip_verify_all=False):
        calls["skip_verify_all"] = skip_verify_all
        return (
            {"cmd": "start-gate", "return_code": 0, "ok": True, "stdout_tail": "", "stderr_tail": ""},
            {
                "train_allowed": True,
                "train_readiness_status": "TRAIN_ALLOWED",
                "decision_reason_code": "READY_OFFLINE_CLEAN",
                "recommended_path": "offline_clean",
            },
        )

    monkeypatch.setattr(module, "build_contract_outputs", fake_build_contract_outputs)
    monkeypatch.setattr(module, "run_post_plan", fake_run_post_plan)
    monkeypatch.setattr(module, "run_start_gate", fake_run_start_gate)
    monkeypatch.setattr(module, "detect_python", lambda root, bootstrap: sys.executable)
    monkeypatch.setattr(module, "build_train_command", lambda py, port: [py, "train/train.py"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "final_orchestrator.py",
            "--check-only",
            "--reports-dir",
            str(tmp_path),
            "--report-out",
            str(report_out),
        ],
    )

    rc = module.main()
    assert rc == module.EXIT_OK
    assert calls["skip_verify_all"] is True
    assert not (tmp_path / "final_orchestrator.lock.json").exists()

    payload = json.loads(report_out.read_text(encoding="utf-8"))
    assert payload["mode"] == "check-only"
    assert payload["status"] == "completed"
    assert payload["decision_reason_code"] == "READY_OFFLINE_CLEAN"
