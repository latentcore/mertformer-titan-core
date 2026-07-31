#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS = ROOT / "reports"
DECISION_JSON = DEFAULT_REPORTS / "start_gate_operator_decision.json"
DECISION_MD = DEFAULT_REPORTS / "start_gate_operator_decision.md"


def sanitize_text(text: str) -> str:
    out = text
    out = out.replace(str(ROOT), "<REPO_ROOT>")
    out = re.sub(r"/Users/[^/]+/(?:Desktop|Downloads|Documents)/[^\s\"']+", "<HOME_PATH>", out)
    return out


def sanitize_value(value):
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, dict):
        return {sanitize_value(key): sanitize_value(item) for key, item in value.items()}
    return value


def run(cmd: list[str], py: str | None = None) -> dict:
    resolved = [py if token == "<PY>" and py else token for token in cmd]
    p = subprocess.run(resolved, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    return {
        "cmd": sanitize_text(" ".join(resolved)),
        "return_code": p.returncode,
        "stdout_tail": sanitize_text(p.stdout[-4000:]),
        "stderr_tail": sanitize_text(p.stderr[-4000:]),
        "ok": p.returncode == 0,
    }


def git_available() -> bool:
    p = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    return p.returncode == 0 and p.stdout.strip() == "true"


def skipped_step(cmd: list[str], reason: str) -> dict:
    return {
        "cmd": sanitize_text(" ".join(cmd)),
        "return_code": 0,
        "stdout_tail": reason,
        "stderr_tail": "",
        "ok": True,
        "skipped": True,
        "non_blocking": True,
    }


def git_gate_steps(*, package_mode: bool, strict_git: bool) -> dict:
    if package_mode and not strict_git:
        reason = "PACKAGE_MODE_GIT_CHECKS_NON_BLOCKING"
        return {
            "git_status": skipped_step(["git", "status", "--short", "--branch"], reason),
            "git_remote": skipped_step(["git", "remote", "-v"], reason),
        }
    return {
        "git_status": run(["git", "status", "--short", "--branch"]),
        "git_remote": run(["git", "remote", "-v"]),
    }


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def transfer_file_candidates() -> list[str]:
    candidates = [
        "zero_touch_start.sh",
        "run.sh",
        "launch_mertformer_kaggle_closure.command",
        "scripts/data_pipeline.py",
        "scripts/smart_runner.py",
        "scripts/precompute_logits_topk.py",
        "scripts/precompute_logits_parallel.py",
        "scripts/validate_logit_alignment.py",
        "train/packing.py",
        "scripts/kaggle_onefile_closure_build30.py",
        "scripts/kaggle_onecell_t4_build30.py",
        "scripts/macos_keepawake.sh",
        "scripts/final_orchestrator.py",
        "scripts/start_gate.py",
        "scripts/build_train_readiness_contract.py",
        "reports/train_readiness_decision.json",
        "reports/train_readiness_decision.md",
        "reports/start_gate_report.json",
        "reports/start_gate_operator_decision.json",
        "reports/start_gate_operator_decision.md",
        "reports/repo_external_handoff.md",
    ]
    always_present_outputs = {
        "reports/start_gate_operator_decision.json",
        "reports/start_gate_operator_decision.md",
    }
    return [path for path in candidates if path in always_present_outputs or (ROOT / path).exists()]


def build_operator_decision(
    structural_ok: bool,
    train_allowed: bool,
    readiness: dict,
    steps: dict,
    *,
    package_mode: bool = False,
    git_metadata_available: bool = True,
) -> dict:
    blockers = list(readiness.get("blockers", []))
    recommended_path = readiness.get("recommended_path")
    decision_reason_code = readiness.get("decision_reason_code")
    verify_ok = bool(steps.get("verify_all", {}).get("ok", True))

    if structural_ok and train_allowed:
        next_action = "ALLOCATE_TARGET_MACHINE_AND_START"
        if recommended_path == "remote_bootstrap":
            operator_message = (
                "Repo-side gate is green via the rented-machine bootstrap lane. Allocate or rent the target training machine, "
                "inject `HF_TOKEN` there (and `WANDB_API_KEY` only if needed), rerun `bash zero_touch_start.sh --check-only`, "
                "and start immediately if the target-machine gate remains green."
            )
        else:
            operator_message = (
                "Repo-side gate is green. Allocate or rent the target training machine, transfer the canonical files, "
                "rerun `bash zero_touch_start.sh --check-only` there, and start training immediately if it remains green."
            )
        if package_mode:
            operator_message += (
                " Package-mode was active because git metadata was unavailable or explicitly bypassed; provenance must be "
                "anchored by package manifests, checksums, and the generated start-gate report rather than live git status."
            )
    elif train_allowed and not verify_ok:
        next_action = "DO_NOT_RENT_YET_FIX_START_GATE"
        operator_message = (
            "Train readiness is green, but the start gate is not fully clean. Do not rent or allocate the expensive machine yet; "
            "fix the failing gate, keep the log, and rerun the canonical check."
        )
    else:
        next_action = "DO_NOT_RENT_YET_FIX_REPO_BLOCKERS"
        operator_message = (
            "Do not rent or allocate the expensive machine yet. Fix the exact repo-side blockers first, keep this decision log, "
            "then rerun the canonical start gate."
        )

    return {
        "schema": "start_gate_operator_decision_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "next_action": next_action,
        "operator_message": operator_message,
        "train_allowed": train_allowed,
        "structural_ok": structural_ok,
        "recommended_path": recommended_path,
        "decision_reason_code": decision_reason_code,
        "start_gate_reason_code": "START_ALLOWED" if structural_ok and train_allowed else (
            "TRAIN_ALLOWED_BUT_STRUCTURAL_BLOCKED" if train_allowed else decision_reason_code
        ),
        "package_mode": package_mode,
        "git_metadata_available": git_metadata_available,
        "blockers": blockers,
        "required_transfer_files": transfer_file_candidates() if train_allowed else [],
    }


def build_operator_decision_md(payload: dict) -> str:
    lines = [
        "# Start Gate Operator Decision",
        "",
        f"- next_action: `{payload['next_action']}`",
        f"- train_allowed: `{payload['train_allowed']}`",
        f"- structural_ok: `{payload['structural_ok']}`",
        f"- recommended_path: `{payload.get('recommended_path') or 'none'}`",
        f"- decision_reason_code: `{payload.get('decision_reason_code') or 'none'}`",
        f"- start_gate_reason_code: `{payload.get('start_gate_reason_code') or 'none'}`",
        f"- package_mode: `{payload.get('package_mode', False)}`",
        f"- git_metadata_available: `{payload.get('git_metadata_available', True)}`",
        "",
        "## Operator Message",
        payload["operator_message"],
        "",
        "## Blockers",
    ]
    if payload["blockers"]:
        lines.extend(f"- `{item}`" for item in payload["blockers"])
    else:
        lines.append("- none")
    lines.extend(["", "## Required Transfer Files"])
    if payload["required_transfer_files"]:
        lines.extend(f"- `{item}`" for item in payload["required_transfer_files"])
    else:
        lines.append("- none")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Closure start gate with exact readiness reporting.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--report-out", default=str(ROOT / "reports" / "start_gate_report.json"))
    parser.add_argument("--skip-verify-all", action="store_true")
    parser.add_argument("--allow-not-ready", action="store_true")
    parser.add_argument("--require-train-allowed", action="store_true")
    parser.add_argument("--package-mode", action="store_true", help="Treat missing git metadata as non-blocking package provenance mode.")
    parser.add_argument("--strict-git", action="store_true", help="Keep git status/remote checks blocking even in package mode.")
    args = parser.parse_args()

    git_metadata_available = git_available()
    package_mode = args.package_mode or not git_metadata_available
    secret_scan_cmd = ["<PY>", "scripts/secret_scan.py"]
    if package_mode:
        secret_scan_cmd.append("--package-mode")

    steps = {
        "secret_scan": run(secret_scan_cmd, py=args.python),
        "check_57_matrix": run(["<PY>", "scripts/check_57_matrix.py"], py=args.python),
        "train_readiness_contract": run(
            ["<PY>", "scripts/build_train_readiness_contract.py", "--allow-not-ready"],
            py=args.python,
        ),
    }
    steps.update(git_gate_steps(package_mode=package_mode, strict_git=args.strict_git))
    if not args.skip_verify_all:
        steps["verify_all"] = run(["bash", "scripts/verify_all.sh"])

    readiness = load_json(ROOT / "reports" / "train_readiness_decision.json")
    structural_ok = all(step["ok"] for step in steps.values())
    train_allowed = readiness.get("final_status") == "TRAIN_ALLOWED"
    start_allowed = structural_ok and train_allowed

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ok": structural_ok,
        "start_allowed": start_allowed,
        "train_allowed": train_allowed,
        "train_readiness_status": readiness.get("final_status"),
        "decision_reason_code": readiness.get("decision_reason_code"),
        "start_gate_reason_code": "START_ALLOWED" if start_allowed else (
            "TRAIN_ALLOWED_BUT_STRUCTURAL_BLOCKED" if train_allowed else readiness.get("decision_reason_code")
        ),
        "package_mode": package_mode,
        "git_metadata_available": git_metadata_available,
        "blockers": readiness.get("blockers", []),
        "steps": steps,
    }
    operator_decision = build_operator_decision(
        structural_ok,
        train_allowed,
        readiness,
        steps,
        package_mode=package_mode,
        git_metadata_available=git_metadata_available,
    )

    out = Path(args.report_out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sanitize_value(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    DECISION_JSON.write_text(json.dumps(sanitize_value(operator_decision), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_text(DECISION_MD, build_operator_decision_md(sanitize_value(operator_decision)))
    print(
        json.dumps(
            {
                "ok": structural_ok,
                "train_allowed": train_allowed,
                "decision_reason_code": readiness.get("decision_reason_code"),
                "start_gate_reason_code": operator_decision["start_gate_reason_code"],
                "package_mode": package_mode,
                "next_action": operator_decision["next_action"],
            },
            ensure_ascii=False,
        )
    )

    if not structural_ok:
        return 1
    if args.require_train_allowed and not train_allowed and not args.allow_not_ready:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
