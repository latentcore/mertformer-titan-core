"""Pilot readiness helpers for commercial B2B workflows."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

_PYTEST_COUNT_RE = re.compile(r"(?P<count>\d+)\s+(?P<label>passed|failed|errors?|skipped|warnings?)")


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git_sha(root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def _extract_operator_summary_json(output: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    summary: dict[str, Any] | None = None
    for i, ch in enumerate(output):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(output[i:])
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("results"), list) and "status" in obj:
            summary = obj
    return summary


def _extract_pytest_summary(output: str) -> dict[str, int]:
    result = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "warnings": 0}
    for line in output.splitlines():
        if " passed" not in line and " failed" not in line and " error" not in line:
            continue
        counts = _PYTEST_COUNT_RE.findall(line)
        if not counts:
            continue
        for count_s, label in counts:
            count = int(count_s)
            key = label.rstrip("s")
            if key == "warning":
                key = "warnings"
            elif key == "error":
                key = "errors"
            result[key] = count
    return result


def parse_verify_output(output: str, exit_code: int) -> dict[str, Any]:
    """Parse `scripts/verify_all.sh` output into gate-level structured data."""
    pytest_summary = _extract_pytest_summary(output)
    operator_json = _extract_operator_summary_json(output)

    operator_steps: dict[str, str] = {}
    if operator_json:
        for item in operator_json.get("results", []):
            if not isinstance(item, dict):
                continue
            step = item.get("step")
            status = item.get("status")
            if step and status:
                operator_steps[str(step)] = str(status)

    required_operator_steps = {
        "nan_kill_switch",
        "checkpoint_restore_drill",
        "failure_budget_drill",
        "overfit_gate",
        "golden_samples",
    }
    operator_gate_pass = all(step in operator_steps for step in required_operator_steps)
    if operator_gate_pass:
        operator_gate_pass = (
            operator_steps.get("nan_kill_switch") == "pass"
            and operator_steps.get("checkpoint_restore_drill") == "pass"
            and operator_steps.get("failure_budget_drill") == "pass"
            and operator_steps.get("golden_samples") == "pass"
            and operator_steps.get("overfit_gate") in {"pass", "pass_fast"}
        )

    if not operator_steps:
        # Fallback parsing for logs without final JSON payload.
        operator_gate_pass = (
            "Checkpoint restore drill: PASS" in output
            and "Failure budget drill: PASS" in output
            and "Golden sample eval: PASS" in output
            and "Overfit gate: PASS" in output
        )

    return {
        "status": "pass" if exit_code == 0 else "fail",
        "exit_code": int(exit_code),
        "secret_scan_pass": "OK: no secret patterns detected in tracked files." in output,
        "pytest_pass": (
            pytest_summary["passed"] > 0
            and pytest_summary["failed"] == 0
            and pytest_summary["errors"] == 0
        ),
        "pytest_summary": pytest_summary,
        "preflight_pass": (
            "RESULT: 🏆 ALL GREEN" in output
            or "OVERALL SYSTEM STATUS: 100% PROTECTED & READY." in output
        ),
        "operator_gate_pass": operator_gate_pass,
        "operator_steps": operator_steps,
        "verify_script_pass": "[verify] OK" in output,
    }


def run_verify_all(*, project_root: Path | None = None, offline: bool = True) -> dict[str, Any]:
    """Run `scripts/verify_all.sh` and return structured gate summary."""
    root = (project_root or PROJECT_ROOT).resolve()
    # Use a cwd-relative command to avoid leaking absolute workstation paths
    # into tracked reports / shared artifacts.
    cmd = ["bash", "scripts/verify_all.sh"]

    env = os.environ.copy()
    if offline:
        env["TITAN_OFFLINE"] = "1"
        env.setdefault("TITAN_WANDB", "0")

    result = subprocess.run(
        cmd,
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}".strip()

    summary = parse_verify_output(output, result.returncode)
    summary["command"] = " ".join(cmd)
    summary["offline_mode"] = bool(offline)
    tail = output[-2000:]
    # Sanitize absolute project root paths for portability/privacy.
    try:
        tail = tail.replace(str(root), "<PROJECT_ROOT>")
        # Extra scrub: remove any leaked macOS home paths (Desktop/Downloads/
        # Documents), including those appearing inside test error messages.
        tail = re.sub(r"/Users/[^/]+/(Desktop|Downloads|Documents)/", r"<HOME>/\1/", tail)
    except Exception as exc:
        # Sanitization is privacy-critical; keep the original tail as fallback
        # but surface the failure so leaked absolute paths can be diagnosed.
        print(f"[pilot] warning: output_tail sanitize failed: {exc}", file=sys.stderr)
    summary["output_tail"] = tail
    return summary


def collect_risk_flags(*, project_root: Path | None = None) -> dict[str, Any]:
    root = (project_root or PROJECT_ROOT).resolve()

    checkpoint_files = sorted(root.glob("checkpoints/**/*.pt"))
    has_checkpoint = len(checkpoint_files) > 0

    required_stage_files = [
        "datasets/stage1/stage1_data.jsonl",
        "datasets/stage2/stage2_data.jsonl",
        "datasets/stage3/stage3_data.jsonl",
        "datasets/stage4_soul/stage4_data.jsonl",
        "datasets/stage5_tools/stage5_data.jsonl",
    ]
    missing_stage_files = [p for p in required_stage_files if not (root / p).exists()]

    return {
        "missing_checkpoint": not has_checkpoint,
        "missing_stage_datasets": len(missing_stage_files) > 0,
        "missing_stage_dataset_files": missing_stage_files,
        "benchmark_not_eligible_for_claim": not has_checkpoint,
    }


def build_pilot_report(
    *,
    project_root: Path | None = None,
    verify_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build `pilot_report_v1` payload."""
    root = (project_root or PROJECT_ROOT).resolve()
    verify = verify_summary or run_verify_all(project_root=root, offline=True)
    risks = collect_risk_flags(project_root=root)

    eligible_for_claim = bool(
        verify.get("status") == "pass"
        and verify.get("verify_script_pass")
        and not risks.get("missing_checkpoint")
    )
    reasons: list[str] = []
    verify_status = verify.get("status")
    if risks.get("missing_checkpoint"):
        reasons.append("trained checkpoint missing")
    if verify_status == "fail":
        reasons.append("verify_all failed")
    elif verify_status == "skipped":
        reasons.append("verify_all skipped")
    elif verify_status == "pass" and not verify.get("verify_script_pass"):
        reasons.append("verify_all completion marker missing")
    if not verify.get("operator_gate_pass"):
        reasons.append("operator gate incomplete/failed")
    if not reasons:
        reasons.append("all required evidence gates passed")

    run_id = time.strftime("pilot_%Y-%m-%d_%H-%M-%S", time.localtime())

    return {
        "schema": "pilot_report_v1",
        "generated_at_utc": _utc_iso(),
        "run_id": run_id,
        "git_sha": _git_sha(root),
        # Keep report portable (no absolute workstation paths).
        "project_root": "<PROJECT_ROOT>",
        "gate_results": {
            "verify_all": {
                "status": verify.get("status"),
                "exit_code": verify.get("exit_code"),
                "secret_scan_pass": verify.get("secret_scan_pass"),
                "pytest_pass": verify.get("pytest_pass"),
                "pytest_summary": verify.get("pytest_summary"),
                "preflight_pass": verify.get("preflight_pass"),
                "operator_gate_pass": verify.get("operator_gate_pass"),
                "verify_script_pass": verify.get("verify_script_pass"),
            },
            "operator_mode_steps": verify.get("operator_steps", {}),
        },
        "risk_flags": risks,
        "benchmark_eligibility": {
            "eligible_for_claim": eligible_for_claim,
            "reasons": reasons,
        },
    }


def write_pilot_report(path: str | Path, payload: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return out
