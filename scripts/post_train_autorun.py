#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Callable


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS = ROOT / "reports"
SYNC_MANIFEST_ARGS = [
    "scripts/sync_manifest.py",
    "--root",
    ".",
    "--manifest",
    "reports/release_manifest.json",
    "--structure",
    "docs/PROJECT_STRUCTURE.md",
    "--matrix",
    "reports/file_sync_matrix.json",
    "--sync-report",
    "reports/project_structure_sync_report.json",
    "--policy-report",
    "reports/policy_sync_report.json",
]


@dataclass(frozen=True)
class Step:
    name: str
    kind: str
    description: str
    checkpoint_required: bool = False


FULL_STEPS = [
    Step("checkpoint_resolution", "internal", "Resolve best/latest checkpoint for post-train tasks.", checkpoint_required=True),
    Step("benchmarks_internal", "command", "Run internal benchmark suite on the resolved checkpoint.", checkpoint_required=True),
    Step("golden_eval", "command", "Run golden sample evaluation on the resolved checkpoint.", checkpoint_required=True),
    Step("demo_bundle_manifest", "internal", "Build demo bundle manifest and summary.", checkpoint_required=False),
    Step("mobile_export", "command", "Refresh ONNX/mobile export artifacts.", checkpoint_required=False),
    Step("logbook_build", "command", "Append run information to the unified logbook.", checkpoint_required=False),
    Step("readme_sync", "command", "Refresh manifest/doc sync and claim alignment.", checkpoint_required=False),
    Step("closure_governance_pack", "command", "Refresh grouped closure and truth reports.", checkpoint_required=False),
    Step("release_build30", "command", "Refresh release zip package and release snapshot.", checkpoint_required=False),
    Step("artifact_release_zip", "command", "Refresh tracked artifact release zip.", checkpoint_required=False),
    Step("training_outputs_bundle", "command", "Build the downloadable training outputs bundle zip plus manifest.", checkpoint_required=False),
    Step("evidence_pack", "internal", "Write the current evidence pack summary.", checkpoint_required=False),
]

MODES: dict[str, list[str]] = {
    "full": [step.name for step in FULL_STEPS],
    "bench-only": ["checkpoint_resolution", "benchmarks_internal", "golden_eval"],
    "export-only": ["checkpoint_resolution", "mobile_export"],
    "demo-only": ["checkpoint_resolution", "demo_bundle_manifest", "evidence_pack"],
    "readme-update-only": ["readme_sync", "closure_governance_pack", "evidence_pack"],
    "check-only": ["checkpoint_resolution"],
}


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


def detect_python(root: Path) -> str:
    env_py = os.environ.get("TITAN_PYTHON", "").strip()
    if env_py:
        return env_py
    venv_py = root / ".titan-venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable or "python3"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def run_command(root: Path, cmd: list[str], env: dict[str, str] | None = None) -> dict:
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=root,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        env=env,
        check=False,
    )
    return {
        "cmd": sanitize_text(" ".join(cmd), root),
        "return_code": proc.returncode,
        "ok": proc.returncode == 0,
        "elapsed_sec": round(time.time() - started, 3),
        "stdout_tail": sanitize_text(proc.stdout[-4000:], root),
        "stderr_tail": sanitize_text(proc.stderr[-4000:], root),
    }


def maybe_cfg(root: Path):
    sys.path.insert(0, str(root))
    try:
        from config.config import cfg  # type: ignore

        return cfg
    except (ImportError, AttributeError) as exc:
        print(f"[warn] maybe_cfg: config.config import/load failed: {exc!r}", file=sys.stderr)
        return None
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass


def resolve_checkpoint(root: Path, explicit: str | None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_ckpt = os.environ.get("TITAN_POST_CKPT", "").strip() or os.environ.get("BENCHMARK_CKPT", "").strip()
    if env_ckpt:
        candidates.append(Path(env_ckpt).expanduser())

    cfg = maybe_cfg(root)
    save_dir = root / "checkpoints"
    model_name = None
    canonical_kaggle_ckpt_dirs = [
        root / "checkpoints" / "kaggle_onefile_build30",
        root / "checkpoints" / "kaggle_onecell_t4_build30",
    ]
    if cfg is not None:
        save_dir = root / str(getattr(cfg, "save_dir", "checkpoints"))
        model_name = str(getattr(cfg, "model_name", "") or "").strip() or None

    if model_name:
        candidates.extend(
            [
                save_dir / f"{model_name}_latest.pt",
                save_dir / f"{model_name}_best.pt",
            ]
        )
    if save_dir.exists():
        candidates.extend(sorted(save_dir.glob("*_latest.pt"), key=lambda p: p.stat().st_mtime, reverse=True))
        candidates.extend(sorted(save_dir.glob("*_best.pt"), key=lambda p: p.stat().st_mtime, reverse=True))
        candidates.extend(sorted(save_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True))

    for canonical_kaggle_ckpt_dir in canonical_kaggle_ckpt_dirs:
        candidates.extend(
            [
                canonical_kaggle_ckpt_dir / "latest.pt",
                canonical_kaggle_ckpt_dir / "best.pt",
            ]
        )

    root_ckpts = root / "checkpoints"
    if root_ckpts.exists() and root_ckpts != save_dir:
        candidates.extend(sorted(root_ckpts.rglob("*_latest.pt"), key=lambda p: p.stat().st_mtime, reverse=True))
        candidates.extend(sorted(root_ckpts.rglob("*_best.pt"), key=lambda p: p.stat().st_mtime, reverse=True))
        candidates.extend(sorted(root_ckpts.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True))

    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def build_contract_text() -> str:
    return dedent(
        """
        # Post-Train Automation Contract

        This contract is the canonical post-train closure state machine for the current working tree.

        ## Modes

        - `full`: benchmark -> golden eval -> demo manifest -> export -> logbook -> docs sync -> governance pack -> release zip -> training outputs bundle -> evidence pack.
        - `bench-only`: checkpoint resolution plus benchmark and golden eval.
        - `export-only`: checkpoint resolution plus export refresh.
        - `demo-only`: checkpoint resolution plus demo manifest and evidence pack summary.
        - `readme-update-only`: manifest sync, doc claim gate, governance refresh, and evidence pack summary.
        - `check-only`: checkpoint resolution only.
        - `plan-only`: emit this contract and the state machine without running commands.
        - `dry-run`: emit the execution plan with current checkpoint resolution but do not run mutating commands.

        ## Failure Policy

        - Missing checkpoint is an error for benchmark-driven modes unless `--allow-missing-checkpoint` is set.
        - If a command step fails in `full` mode, remaining steps are skipped and the JSON status is marked `failed`.
        - Internal manifest-writing steps still run in `demo-only` and `readme-update-only` modes even if no checkpoint is available, but they mark the missing checkpoint explicitly.
        - This script never claims trained evidence exists if the checkpoint is missing.
        """
    ).strip()


def build_state_machine_text() -> str:
    lines = [
        "# Post-Train State Machine",
        "",
        "| Order | Step | Kind | Checkpoint Required | Purpose |",
        "| --- | --- | --- | --- | --- |",
    ]
    for idx, step in enumerate(FULL_STEPS, start=1):
        lines.append(
            f"| {idx} | `{step.name}` | `{step.kind}` | `{str(step.checkpoint_required).lower()}` | {step.description} |"
        )
    return "\n".join(lines)


def _demo_asset_kind(path: Path, checkpoint: Path | None) -> str:
    """Honest per-asset classification.

    The snake demo is a pygame heuristic autoplayer (NOT model-driven), and the
    drone flow is a software-in-the-loop simulation — neither is a trained-model
    output, so they must never be tagged ``trained_demo`` even when a checkpoint
    exists. Only genuinely checkpoint-bound assets get ``trained_demo``.
    """
    name = path.name.lower()
    if "snake_demo" in name:
        return "ui_demo"
    if "drone_sitl" in name:
        return "sitl_sim_demo"
    return "trained_demo" if checkpoint and path.exists() else "repo_proof_or_placeholder"


def build_demo_bundle(root: Path, reports_dir: Path, checkpoint: Path | None) -> dict:
    assets = [
        root / "assets" / "snake_demo_proof.mp4",
        root / "assets" / "snake_demo_preview.gif",
        root / "reports" / "drone_sitl_demo.md",
        root / "reports" / "drone_sitl_demo_TR.md",
    ]
    manifest = {
        "schema": "demo_bundle_manifest_v1",
        "generated_utc": utc_now(),
        "checkpoint": sanitize_text(str(checkpoint), root) if checkpoint else None,
        "items": [
            {
                "path": str(path.relative_to(root)),
                "exists": path.exists(),
                "kind": _demo_asset_kind(path, checkpoint),
            }
            for path in assets
        ],
        "note": (
            "A trained checkpoint was resolved for post-train closure."
            if checkpoint
            else "No trained checkpoint resolved; this manifest only records current repo-side proof/demo assets."
        ),
    }
    json_path = reports_dir / "demo_bundle_manifest.json"
    md_path = reports_dir / "demo_bundle.md"
    ensure_parent(json_path)
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "# Demo Bundle",
        "",
        f"- generated_utc: `{manifest['generated_utc']}`",
        f"- checkpoint: `{manifest['checkpoint'] or 'none'}`",
        f"- note: {manifest['note']}",
        "",
        "## Items",
        "",
    ]
    for item in manifest["items"]:
        lines.append(f"- `{item['path']}` | exists=`{item['exists']}` | kind=`{item['kind']}`")
    write_text(md_path, "\n".join(lines))
    return manifest


def build_evidence_pack(root: Path, reports_dir: Path, checkpoint: Path | None, steps: list[dict], mode: str) -> None:
    evidence_refs = [
        "reports/train_readiness_decision.md",
        "reports/final_truth_matrix.md",
        "reports/final_backlog_classification.md",
        "reports/demo_bundle_manifest.json",
        "reports/training_outputs_bundle_manifest.json",
        "reports/release_manifest.json",
        "reports/one_command_full_sop_summary.md",
    ]
    existing = [ref for ref in evidence_refs if (root / ref).exists()]
    lines = [
        "# Final Evidence Pack",
        "",
        f"- generated_utc: `{utc_now()}`",
        f"- mode: `{mode}`",
        f"- checkpoint: `{sanitize_text(str(checkpoint), root) if checkpoint else 'none'}`",
        "",
        "## Included Evidence",
        "",
    ]
    if existing:
        lines.extend(f"- `{ref}`" for ref in existing)
    else:
        lines.append("- none")
    lines.extend(["", "## Step Status", ""])
    for step in steps:
        lines.append(
            f"- `{step['name']}`: status=`{step['status']}` return_code=`{step.get('return_code', 'n/a')}`"
        )
    write_text(reports_dir / "final_evidence_pack.md", "\n".join(lines))


def command_builders(root: Path, py: str, checkpoint: Path | None) -> dict[str, Callable[[], dict]]:
    env = os.environ.copy()
    if checkpoint:
        env.setdefault("BENCHMARK_CKPT", str(checkpoint))
        env.setdefault("TITAN_POST_CKPT", str(checkpoint))

    def bench() -> dict:
        if checkpoint is None:
            return {"return_code": 2, "ok": False, "stdout_tail": "", "stderr_tail": "checkpoint missing", "cmd": "<skipped>", "elapsed_sec": 0.0}
        return run_command(root, [py, "scripts/benchmarks_internal.py", "--run", "--samples", os.environ.get("BENCHMARK_SAMPLES", "0"), "--ckpt", str(checkpoint)], env=env)

    def golden() -> dict:
        if checkpoint is None:
            return {"return_code": 2, "ok": False, "stdout_tail": "", "stderr_tail": "checkpoint missing", "cmd": "<skipped>", "elapsed_sec": 0.0}
        return run_command(
            root,
            [
                py,
                "scripts/golden_score.py",
                "--run-model",
                "--ckpt",
                str(checkpoint),
                "--predictions",
                "reports/benchmarks/golden_outputs.jsonl",
                "--summary",
                "reports/benchmarks/golden_summary.json",
            ],
            env=env,
        )

    return {
        "benchmarks_internal": bench,
        "golden_eval": golden,
        "mobile_export": lambda: run_command(root, [py, "scripts/mobile_export.py"], env=env),
        "logbook_build": lambda: run_command(root, [py, "scripts/logbook_build.py", "--append"], env=env),
        "readme_sync": lambda: run_command(root, [py, *SYNC_MANIFEST_ARGS], env=env),
        "closure_governance_pack": lambda: run_command(root, [py, "scripts/build_closure_governance_pack.py"], env=env),
        "release_build30": lambda: run_command(root, ["bash", "scripts/release_build30.sh"], env=env),
        "artifact_release_zip": lambda: run_command(root, ["bash", "scripts/build_artifacts_release_zip.sh"], env=env),
        "training_outputs_bundle": lambda: run_command(root, [py, "scripts/build_training_outputs_bundle.py"], env=env),
    }


def summarize_md(payload: dict) -> str:
    lines = [
        "# Post-Train Autorun Status",
        "",
        f"- schema: `{payload['schema']}`",
        f"- mode: `{payload['mode']}`",
        f"- status: `{payload['status']}`",
        f"- generated_utc: `{payload['generated_utc']}`",
        f"- checkpoint: `{payload.get('checkpoint') or 'none'}`",
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


def run_mode(args: argparse.Namespace) -> dict:
    root = Path(args.project_root).resolve() if args.project_root else ROOT
    reports_dir = Path(args.reports_dir).resolve() if args.reports_dir else DEFAULT_REPORTS
    py = args.python or detect_python(root)

    contract_path = reports_dir / "post_train_automation_contract.md"
    state_machine_path = reports_dir / "post_train_state_machine.md"
    write_text(contract_path, build_contract_text())
    write_text(state_machine_path, build_state_machine_text())

    mode = "full"
    for candidate in ("bench_only", "export_only", "demo_only", "readme_update_only", "check_only"):
        if getattr(args, candidate):
            mode = candidate.replace("_", "-")
    if args.post_only:
        mode = "full"

    checkpoint = resolve_checkpoint(root, args.checkpoint)
    selected_names = MODES[mode]
    selected = [step for step in FULL_STEPS if step.name in selected_names]

    payload = {
        "schema": "post_train_autorun_status_v1",
        "generated_utc": utc_now(),
        "mode": "plan-only" if args.plan_only else "dry-run" if args.dry_run else mode,
        "status": "planned" if args.plan_only else "dry-run" if args.dry_run else "completed",
        "checkpoint": str(checkpoint) if checkpoint else None,
        "reports_dir": str(reports_dir),
        "steps": [],
    }

    if args.plan_only:
        for step in selected:
            payload["steps"].append({
                "name": step.name,
                "status": "planned",
                "note": step.description,
                "checkpoint_required": step.checkpoint_required,
            })
        return payload

    commands = command_builders(root, py, checkpoint)

    for step in selected:
        step_entry = {
            "name": step.name,
            "checkpoint_required": step.checkpoint_required,
        }
        if step.name == "checkpoint_resolution":
            if checkpoint is None and step.checkpoint_required and not args.allow_missing_checkpoint:
                step_entry.update({
                    "status": "failed",
                    "return_code": 2,
                    "note": "No checkpoint resolved for the requested mode.",
                })
                payload["steps"].append(step_entry)
                payload["status"] = "failed"
                break
            step_entry.update({
                "status": "completed" if checkpoint else "warning",
                "return_code": 0 if checkpoint or args.allow_missing_checkpoint else 2,
                "note": str(checkpoint) if checkpoint else "Checkpoint missing; continuing only because allow_missing_checkpoint or non-benchmark mode applies.",
            })
            payload["steps"].append(step_entry)
            continue

        if step.name == "demo_bundle_manifest":
            manifest = build_demo_bundle(root, reports_dir, checkpoint)
            step_entry.update({
                "status": "completed",
                "return_code": 0,
                "note": manifest["note"],
            })
            payload["steps"].append(step_entry)
            continue

        if step.name == "evidence_pack":
            build_evidence_pack(root, reports_dir, checkpoint, payload["steps"], mode)
            step_entry.update({
                "status": "completed",
                "return_code": 0,
                "note": "reports/final_evidence_pack.md refreshed.",
            })
            payload["steps"].append(step_entry)
            continue

        if args.dry_run:
            step_entry.update({
                "status": "planned",
                "return_code": 0,
                "note": step.description,
            })
            payload["steps"].append(step_entry)
            continue

        result = commands[step.name]()
        step_entry.update({
            "status": "completed" if result["ok"] else "failed",
            "return_code": result["return_code"],
            "note": result["stderr_tail"] or result["stdout_tail"] or step.description,
            "cmd": result["cmd"],
        })
        payload["steps"].append(step_entry)
        if not result["ok"]:
            payload["status"] = "failed"
            break

    if payload["status"] != "failed" and not args.plan_only and not args.dry_run:
        payload["status"] = "completed"
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical post-train autorun state machine.")
    parser.add_argument("--python", help="Python interpreter to use for child commands.")
    parser.add_argument("--project-root", help="Override project root for testing or controlled runs.")
    parser.add_argument("--reports-dir", help="Override reports directory.")
    parser.add_argument("--report-out", help="Write JSON status to this path instead of the default report path.")
    parser.add_argument("--checkpoint", help="Explicit checkpoint path.")
    parser.add_argument("--allow-missing-checkpoint", action="store_true", help="Do not fail when the checkpoint cannot be resolved.")
    parser.add_argument("--plan-only", action="store_true", help="Only write the contract and planned steps.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve plan and checkpoint, but do not run mutating commands.")
    parser.add_argument("--post-only", action="store_true", help="Run the full post-train chain.")
    parser.add_argument("--bench-only", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--demo-only", action="store_true")
    parser.add_argument("--readme-update-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_mode(args)

    root = Path(args.project_root).resolve() if args.project_root else ROOT
    reports_dir = Path(args.reports_dir).resolve() if args.reports_dir else DEFAULT_REPORTS
    report_out = Path(args.report_out).resolve() if args.report_out else reports_dir / "post_train_autorun_status.json"
    md_out = report_out.with_suffix(".md")

    ensure_parent(report_out)
    sanitized = sanitize_value(payload, root)
    report_out.write_text(json.dumps(sanitized, indent=2), encoding="utf-8")
    write_text(md_out, summarize_md(sanitized))
    print(json.dumps({"status": payload["status"], "mode": payload["mode"], "checkpoint": payload.get("checkpoint")}, ensure_ascii=False))
    return 0 if payload["status"] in {"planned", "dry-run", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
