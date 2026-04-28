#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS = ROOT / "reports"
DEFAULT_LOCK = DEFAULT_REPORTS / "final_orchestrator.lock.json"
DEFAULT_STATUS = DEFAULT_REPORTS / "final_orchestrator_status.json"
RUN_MANIFEST_SCHEMA = ROOT / "interfaces" / "run_manifest_v1.schema.json"

EXIT_OK = 0
EXIT_LOCKED = 2
EXIT_NOT_ALLOWED = 3
EXIT_GATE_FAILED = 4
EXIT_POST_FAILED = 5


@dataclass(frozen=True)
class Step:
    name: str
    kind: str
    description: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sanitize_text(text: str, root: Path) -> str:
    cleaned = text.replace(str(root), "<REPO_ROOT>")
    cleaned = cleaned.replace(str(Path.home() / "Desktop"), "<DESKTOP_PATH>")
    return cleaned


def sanitize_value(value, root: Path):
    if isinstance(value, str):
        return sanitize_text(value, root)
    if isinstance(value, list):
        return [sanitize_value(item, root) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_value(item, root) for key, item in value.items()}
    return value


def detect_python(root: Path, *, bootstrap: bool) -> str:
    env_py = os.environ.get("TITAN_PYTHON", "").strip()
    if env_py:
        if os.path.sep in env_py or env_py.startswith("."):
            return env_py
        resolved = shutil.which(env_py)
        if resolved:
            return resolved
        return env_py

    venv_py = root / ".titan-venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)

    if bootstrap and (root / "scripts" / "bootstrap_venv.sh").exists():
        proc = subprocess.run(["bash", "scripts/bootstrap_venv.sh"], cwd=root, check=False)
        if proc.returncode == 0 and venv_py.exists():
            return str(venv_py)

    return sys.executable or "python3"


def run_command(root: Path, cmd: list[str], env: dict[str, str] | None = None) -> dict:
    started = time.time()
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, env=env, check=False)
    return {
        "cmd": sanitize_text(" ".join(cmd), root),
        "return_code": proc.returncode,
        "ok": proc.returncode == 0,
        "elapsed_sec": round(time.time() - started, 3),
        "stdout_tail": sanitize_text(proc.stdout[-4000:], root),
        "stderr_tail": sanitize_text(proc.stderr[-4000:], root),
    }


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def detect_num_processes() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        parts = [part for part in visible.split(",") if part.strip() and part.strip() != "-1"]
        if parts:
            return len(parts)

    if shutil.which("nvidia-smi"):
        proc = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            lines = [line for line in proc.stdout.splitlines() if line.strip()]
            if lines:
                return len(lines)
    return 1


def build_train_command(py: str, port: int) -> list[str]:
    num_processes = detect_num_processes()
    launcher = [py, "-m", "accelerate.commands.launch"]
    return launcher + [
        "--num_processes",
        str(num_processes),
        "--num_machines",
        "1",
        "--mixed_precision",
        "bf16",
        "--main_process_port",
        str(port),
        "train/train.py",
    ]


def acquire_lock(lock_path: Path, payload: dict) -> tuple[bool, dict | None]:
    ensure_parent(lock_path)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing = load_json(lock_path)
        return False, existing or None
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return True, None


def release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def build_run_manifest_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "run_manifest_v1.schema.json",
        "title": "Run Manifest V1",
        "type": "object",
        "required": [
            "schema",
            "run_id",
            "generated_utc",
            "mode",
            "status",
            "steps",
        ],
        "properties": {
            "schema": {"const": "run_manifest_v1"},
            "run_id": {"type": "string"},
            "generated_utc": {"type": "string"},
            "mode": {"type": "string"},
            "status": {"type": "string"},
            "decision_reason_code": {"type": ["string", "null"]},
            "train_readiness_status": {"type": ["string", "null"]},
            "training_lane": {"type": ["string", "null"]},
            "resume_policy": {"type": "string"},
            "train_command": {"type": ["string", "null"]},
            "post_mode": {"type": ["string", "null"]},
            "lock_file": {"type": ["string", "null"]},
            "reports_dir": {"type": "string"},
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "status"],
                    "properties": {
                        "name": {"type": "string"},
                        "status": {"type": "string"},
                        "return_code": {"type": ["integer", "null"]},
                        "note": {"type": ["string", "null"]},
                        "cmd": {"type": ["string", "null"]},
                    },
                    "additionalProperties": True,
                },
            },
        },
        "additionalProperties": True,
    }


def build_run_contract() -> str:
    return dedent(
        """
        # Run Contract

        This file is the canonical runtime contract for the current 45K closure path.

        ## Canonical Entrypoint
        - `bash zero_touch_start.sh`

        ## Modes
        - `--check-only`: run the fast target-machine start gate and exact readiness contract without launching training.
        - `--plan-only`: emit the contracts and planned steps only.
        - `--dry-run`: emit the plan plus resolved train command, but do not launch training.
        - `--post-only`: skip training and run the post-train state machine.
        - `--no-post`: skip the post-train state machine after a successful training run.
        - `--bench-only`, `--demo-only`, `--export-only`, `--readme-update-only`: select the post-train subset.

        ## Start Rules
        - Training start is allowed only when `reports/train_readiness_decision.json` says `TRAIN_ALLOWED`.
        - The start gate must produce exact blocker reason codes before any full training launch.
        - The canonical `offline_clean` lane is strict precomputed KD and keeps `meta-llama/Llama-3.3-70B-Instruct` as the fixed teacher surface.
        - `--check-only` intentionally skips the heavyweight `verify_all.sh` sweep and behaves as a target-machine readiness gate.
        - This orchestrator uses a JSON lock file to prevent overlapping train-end launches.

        ## Resume Rules
        - `--resume auto`: enable auto-discovery via `TITAN_AUTO_RESUME=1`.
        - `--resume off`: disable resume via `TITAN_AUTO_RESUME=0`.
        - `--resume /path/to/checkpoint.pt`: set `TITAN_RESUME_FROM` explicitly.

        ## Post-Train Rule
        - Post-train closeout is delegated to `scripts/post_train_autorun.py`.
        - Post-train closeout now refreshes both release-side zips and the dedicated training outputs bundle zip for target-machine retrieval.
        - No trained evidence claim becomes true unless a real checkpoint is resolved and the downstream artifacts are refreshed from that checkpoint.
        """
    ).strip()


def build_expected_artifacts() -> str:
    return dedent(
        """
        # Expected Artifacts List

        ## Always Refreshed
        - `reports/run_contract.md`
        - `reports/expected_artifacts_list.md`
        - `reports/exit_code_standard.md`
        - `reports/post_train_automation_contract.md`
        - `reports/post_train_state_machine.md`
        - `interfaces/run_manifest_v1.schema.json`

        ## Check-Only / Start-Gate Path
        - `reports/start_gate_report.json`
        - `reports/train_readiness_decision.json`
        - `reports/train_readiness_decision.md`
        - `reports/training_readiness_manifest.json`
        - `reports/final_orchestrator_status.json`

        ## Full Train Path (after success)
        - `logs/production_run.log` or equivalent training stdout capture
        - checkpoints under `cfg.save_dir`
        - refreshed readiness/runtime manifests from the training path
        - `reports/post_train_autorun_status.json`
        - `reports/demo_bundle_manifest.json`
        - `reports/training_outputs_bundle_manifest.json`
        - `reports/training_outputs_bundle_manifest.md`
        - `reports/final_evidence_pack.md`
        - `artifacts/mertformer_training_outputs_bundle.zip`
        - `artifacts/mertformer_training_outputs_bundle.zip.sha256`

        ## Post-Only Path
        - `reports/post_train_autorun_status.json`
        - `reports/demo_bundle_manifest.json`
        - `reports/demo_bundle.md`
        - `reports/training_outputs_bundle_manifest.json`
        - `reports/training_outputs_bundle_manifest.md`
        - `reports/final_evidence_pack.md`
        - `artifacts/mertformer_training_outputs_bundle.zip`
        - `artifacts/mertformer_training_outputs_bundle.zip.sha256`
        """
    ).strip()


def build_exit_code_standard() -> str:
    lines = [
        "# Exit Code Standard",
        "",
        "| Code | Meaning |",
        "| --- | --- |",
        f"| `{EXIT_OK}` | success, planned, or dry-run completion |",
        f"| `{EXIT_LOCKED}` | orchestrator lock already exists |",
        f"| `{EXIT_NOT_ALLOWED}` | readiness blocked; training not allowed |",
        f"| `{EXIT_GATE_FAILED}` | start gate or training launch failed |",
        f"| `{EXIT_POST_FAILED}` | post-train state machine failed |",
    ]
    return "\n".join(lines)


def summarize_md(payload: dict) -> str:
    lines = [
        "# Final Orchestrator Status",
        "",
        f"- run_id: `{payload['run_id']}`",
        f"- mode: `{payload['mode']}`",
        f"- status: `{payload['status']}`",
        f"- generated_utc: `{payload['generated_utc']}`",
        f"- train_readiness_status: `{payload.get('train_readiness_status') or 'none'}`",
        f"- decision_reason_code: `{payload.get('decision_reason_code') or 'none'}`",
        f"- training_lane: `{payload.get('training_lane') or 'none'}`",
        f"- resume_policy: `{payload['resume_policy']}`",
        f"- train_command: `{payload.get('train_command') or 'none'}`",
        f"- post_mode: `{payload.get('post_mode') or 'none'}`",
        "",
        "## Steps",
        "",
        "| Step | Status | Return Code | Notes |",
        "| --- | --- | --- | --- |",
    ]
    for step in payload["steps"]:
        lines.append(
            f"| `{step['name']}` | `{step['status']}` | `{step.get('return_code', 'n/a')}` | {step.get('note', '') or ' '} |"
        )
    return "\n".join(lines)


def post_mode_from_args(args: argparse.Namespace) -> str:
    if args.bench_only:
        return "bench-only"
    if args.export_only:
        return "export-only"
    if args.demo_only:
        return "demo-only"
    if args.readme_update_only:
        return "readme-update-only"
    return "full"


def resolve_mode(args: argparse.Namespace) -> str:
    if args.post_only or args.bench_only or args.export_only or args.demo_only or args.readme_update_only:
        return "post-only"
    if args.check_only:
        return "check-only"
    return "full"


def run_post_plan(root: Path, py: str, reports_dir: Path) -> dict:
    return run_command(
        root,
        [
            py,
            "scripts/post_train_autorun.py",
            "--plan-only",
            "--allow-missing-checkpoint",
            "--reports-dir",
            str(reports_dir),
            "--report-out",
            str(reports_dir / "post_train_autorun_status.json"),
        ],
    )


def build_contract_outputs(root: Path, reports_dir: Path) -> None:
    write_text(reports_dir / "run_contract.md", build_run_contract())
    write_text(reports_dir / "expected_artifacts_list.md", build_expected_artifacts())
    write_text(reports_dir / "exit_code_standard.md", build_exit_code_standard())
    write_json(root / "interfaces" / "run_manifest_v1.schema.json", build_run_manifest_schema())


def run_start_gate(root: Path, py: str, reports_dir: Path, *, skip_verify_all: bool = False) -> tuple[dict, dict]:
    out_path = reports_dir / "start_gate_report.json"
    cmd = [
        py,
        "scripts/start_gate.py",
        "--python",
        py,
        "--report-out",
        str(out_path),
        "--allow-not-ready",
    ]
    if skip_verify_all:
        cmd.append("--skip-verify-all")
    result = run_command(
        root,
        cmd,
    )
    return result, load_json(out_path)


def resolve_training_lane(start_gate_payload: dict | None) -> str:
    payload = start_gate_payload or {}
    if payload.get("recommended_path") == "offline_clean":
        return "offline_clean"
    if payload.get("decision_reason_code") == "READY_OFFLINE_CLEAN":
        return "offline_clean"
    return "online_teacher"


def build_training_env(resume_policy: str, start_gate_payload: dict | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("TITAN_OFFLINE", "0")

    if resolve_training_lane(start_gate_payload) == "offline_clean":
        env["TITAN_OFFLINE"] = "1"
        env["TITAN_REQUIRE_GATED_TEACHER"] = "1"
        env["TITAN_USE_PRECOMPUTED_LOGITS"] = "1"
        env["TITAN_USE_TR_TOKENIZER"] = "1"

    if resume_policy == "off":
        env["TITAN_AUTO_RESUME"] = "0"
        env.pop("TITAN_RESUME_FROM", None)
    elif resume_policy == "auto":
        env["TITAN_AUTO_RESUME"] = "1"
        env.pop("TITAN_RESUME_FROM", None)
    else:
        env["TITAN_AUTO_RESUME"] = "1"
        env["TITAN_RESUME_FROM"] = resume_policy
    return env


def run_post_chain(root: Path, py: str, reports_dir: Path, args: argparse.Namespace, post_mode: str) -> dict:
    cmd = [
        py,
        "scripts/post_train_autorun.py",
        "--reports-dir",
        str(reports_dir),
        "--report-out",
        str(reports_dir / "post_train_autorun_status.json"),
    ]
    if args.allow_missing_checkpoint:
        cmd.append("--allow-missing-checkpoint")
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    if post_mode == "bench-only":
        cmd.append("--bench-only")
    elif post_mode == "export-only":
        cmd.append("--export-only")
    elif post_mode == "demo-only":
        cmd.append("--demo-only")
    elif post_mode == "readme-update-only":
        cmd.append("--readme-update-only")
    else:
        cmd.append("--post-only")
    return run_command(root, cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical zero-touch train-end orchestrator.")
    parser.add_argument("--python", help="Python interpreter for child commands.")
    parser.add_argument("--project-root", help="Override project root.")
    parser.add_argument("--reports-dir", help="Override reports directory.")
    parser.add_argument("--report-out", help="Write JSON status here instead of reports/final_orchestrator_status.json.")
    parser.add_argument("--plan-only", action="store_true", help="Only write contracts and planned steps.")
    parser.add_argument("--dry-run", action="store_true", help="Emit resolved commands without running mutating steps.")
    parser.add_argument("--check-only", action="store_true", help="Run the start gate and readiness contract only.")
    parser.add_argument("--post-only", action="store_true", help="Skip training and run only the post-train chain.")
    parser.add_argument("--no-post", action="store_true", help="Skip post-train closeout after training succeeds.")
    parser.add_argument("--bench-only", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--demo-only", action="store_true")
    parser.add_argument("--readme-update-only", action="store_true")
    parser.add_argument("--allow-missing-checkpoint", action="store_true", help="Allow post-only flows to continue without a checkpoint.")
    parser.add_argument("--checkpoint", help="Explicit checkpoint for post-only flows.")
    parser.add_argument("--resume", default="auto", help="Resume policy: auto | off | /path/to/checkpoint.pt")
    parser.add_argument("--force-lock-clear", action="store_true", help="Clear an existing orchestrator lock before proceeding.")
    parser.add_argument("--main-process-port", type=int, default=29501)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).resolve() if args.project_root else ROOT
    reports_dir = Path(args.reports_dir).resolve() if args.reports_dir else DEFAULT_REPORTS
    report_out = Path(args.report_out).resolve() if args.report_out else DEFAULT_STATUS
    lock_path = reports_dir / DEFAULT_LOCK.name
    py = args.python or detect_python(root, bootstrap=not args.plan_only and not args.dry_run)
    mode = resolve_mode(args)
    post_mode = post_mode_from_args(args)
    run_id = f"zero_touch_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    train_cmd = build_train_command(py, args.main_process_port)

    build_contract_outputs(root, reports_dir)

    payload = {
        "schema": "run_manifest_v1",
        "run_id": run_id,
        "generated_utc": utc_now(),
        "mode": "plan-only" if args.plan_only else "dry-run" if args.dry_run else mode,
        "status": "planned" if args.plan_only else "dry-run" if args.dry_run else "completed",
        "reports_dir": str(reports_dir),
        "lock_file": str(lock_path),
        "resume_policy": args.resume,
        "train_command": " ".join(train_cmd) if mode == "full" else None,
        "post_mode": post_mode if (mode == "post-only" or not args.no_post) else None,
        "training_lane": None,
        "train_readiness_status": None,
        "decision_reason_code": None,
        "steps": [],
    }

    payload["steps"].append({
        "name": "contract_outputs",
        "status": "completed",
        "return_code": 0,
        "note": "Run contract, artifact list, exit code standard, and run manifest schema refreshed.",
    })

    post_plan = run_post_plan(root, py, reports_dir)
    payload["steps"].append({
        "name": "post_train_plan_refresh",
        "status": "completed" if post_plan["ok"] else "failed",
        "return_code": post_plan["return_code"],
        "note": post_plan["stderr_tail"] or post_plan["stdout_tail"] or "Post-train plan refreshed.",
        "cmd": post_plan["cmd"],
    })
    if not post_plan["ok"]:
        payload["status"] = "failed"

    if args.plan_only or args.dry_run:
        planned = [
            Step("start_gate", "command", "Refresh verify/readiness gate and exact blocker report."),
            Step("training", "command", "Launch accelerate training path with resume policy applied."),
            Step("post_train", "command", f"Run post-train state machine in {post_mode} mode."),
        ]
        for step in planned:
            payload["steps"].append({
                "name": step.name,
                "status": "planned",
                "return_code": 0,
                "note": step.description,
            })
        sanitized = sanitize_value(payload, root)
        write_json(report_out, sanitized)
        write_text(report_out.with_suffix(".md"), summarize_md(sanitized))
        print(json.dumps({"status": payload["status"], "mode": payload["mode"]}, ensure_ascii=False))
        return EXIT_OK

    if args.force_lock_clear:
        release_lock(lock_path)

    lock_payload = {
        "run_id": run_id,
        "created_utc": utc_now(),
        "mode": mode,
        "pid": os.getpid(),
    }
    acquired, existing = acquire_lock(lock_path, lock_payload)
    if not acquired:
        payload["status"] = "failed"
        payload["steps"].append({
            "name": "run_lock",
            "status": "failed",
            "return_code": EXIT_LOCKED,
            "note": f"Existing lock: {existing or {'path': str(lock_path)}}",
        })
        sanitized = sanitize_value(payload, root)
        write_json(report_out, sanitized)
        write_text(report_out.with_suffix(".md"), summarize_md(sanitized))
        print(json.dumps({"status": payload["status"], "reason": "lock_exists"}, ensure_ascii=False))
        return EXIT_LOCKED

    exit_code = EXIT_OK
    try:
        if payload["status"] == "failed":
            exit_code = EXIT_GATE_FAILED
            return exit_code

        if mode in {"full", "check-only"}:
            start_gate_result, start_gate_payload = run_start_gate(
                root,
                py,
                reports_dir,
                skip_verify_all=(mode == "check-only"),
            )
            payload["train_readiness_status"] = start_gate_payload.get("train_readiness_status")
            payload["decision_reason_code"] = start_gate_payload.get("decision_reason_code")
            payload["training_lane"] = resolve_training_lane(start_gate_payload)
            payload["steps"].append({
                "name": "start_gate",
                "status": "completed" if start_gate_result["ok"] else "failed",
                "return_code": start_gate_result["return_code"],
                "note": start_gate_result["stderr_tail"] or start_gate_result["stdout_tail"] or "start gate finished",
                "cmd": start_gate_result["cmd"],
            })
            if not start_gate_result["ok"]:
                payload["status"] = "failed"
                exit_code = EXIT_GATE_FAILED
                return exit_code
            if not bool(start_gate_payload.get("train_allowed")):
                payload["status"] = "blocked"
                exit_code = EXIT_NOT_ALLOWED
                return exit_code
            if mode == "check-only":
                payload["status"] = "completed"
                exit_code = EXIT_OK
                return exit_code

        if mode == "post-only":
            post_result = run_post_chain(root, py, reports_dir, args, post_mode)
            payload["steps"].append({
                "name": "post_train",
                "status": "completed" if post_result["ok"] else "failed",
                "return_code": post_result["return_code"],
                "note": post_result["stderr_tail"] or post_result["stdout_tail"] or "post-train flow executed",
                "cmd": post_result["cmd"],
            })
            payload["status"] = "completed" if post_result["ok"] else "failed"
            exit_code = EXIT_OK if post_result["ok"] else EXIT_POST_FAILED
            return exit_code

        env = build_training_env(args.resume, start_gate_payload)
        train_result = run_command(root, train_cmd, env=env)
        payload["steps"].append({
            "name": "training",
            "status": "completed" if train_result["ok"] else "failed",
            "return_code": train_result["return_code"],
            "note": train_result["stderr_tail"] or train_result["stdout_tail"] or f"training flow executed ({resolve_training_lane(start_gate_payload)})",
            "cmd": train_result["cmd"],
        })
        if not train_result["ok"]:
            payload["status"] = "failed"
            exit_code = EXIT_GATE_FAILED
            return exit_code

        if args.no_post:
            payload["steps"].append({
                "name": "post_train",
                "status": "skipped",
                "return_code": 0,
                "note": "Skipped because --no-post was set.",
            })
            payload["status"] = "completed"
            exit_code = EXIT_OK
            return exit_code

        post_result = run_post_chain(root, py, reports_dir, args, post_mode)
        payload["steps"].append({
            "name": "post_train",
            "status": "completed" if post_result["ok"] else "failed",
            "return_code": post_result["return_code"],
            "note": post_result["stderr_tail"] or post_result["stdout_tail"] or "post-train flow executed",
            "cmd": post_result["cmd"],
        })
        payload["status"] = "completed" if post_result["ok"] else "failed"
        exit_code = EXIT_OK if post_result["ok"] else EXIT_POST_FAILED
        return exit_code
    finally:
        release_lock(lock_path)
        sanitized = sanitize_value(payload, root)
        write_json(report_out, sanitized)
        write_text(report_out.with_suffix(".md"), summarize_md(sanitized))
        print(json.dumps({
            "status": payload["status"],
            "mode": payload["mode"],
            "train_readiness_status": payload.get("train_readiness_status"),
            "decision_reason_code": payload.get("decision_reason_code"),
        }, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
