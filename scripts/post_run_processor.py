#!/usr/bin/env python3
"""Process Ocean evidence-proof outputs into conservative handoff reports.

This script is intentionally claim-boundary first. It verifies that the expected
post-run artifacts exist, checks archive/hash coverage, extracts step/eval
signals, and writes a small report that separates operational evidence from
learning or capability claims.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
import zipfile


FINAL_EVIDENCE_ZIP = "mertformer_final_evidence_proof_outputs.zip"
REQUIRED_BASENAMES = {
    FINAL_EVIDENCE_ZIP,
    "proof_decision.json",
    "proof_decision.md",
    "final_math_h200_manifest.json",
    "final_math_h200_report.json",
    "sha256.txt",
}
REQUIRED_DIR_NAMES = {"logs", "eval", "checkpoints"}
CHECKPOINT_BASENAMES = {"last.pt", "final.pt"}


def sha256_file(path: Path) -> str:
    """Return SHA256 hex digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _find_first(root: Path, names: Iterable[str]) -> Path | None:
    name_set = set(names)
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name in name_set:
            return path
    return None


def _find_all(root: Path, names: Iterable[str]) -> list[Path]:
    name_set = set(names)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name in name_set)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _read_jsonl_events(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        start = raw.find("{")
        if start < 0:
            continue
        try:
            item = json.loads(raw[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            events.append(item)
    return events


def _read_eval_history(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"rows": [], "best_exact_accuracy": None, "last_exact_accuracy": None}

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(dict(row))

    exact_values: list[float] = []
    for row in rows:
        for key in ("held_out_exact_acc", "exact_accuracy", "exact_acc", "accuracy"):
            raw = row.get(key)
            if raw in (None, ""):
                continue
            try:
                exact_values.append(float(raw))
            except ValueError:
                pass
            break

    return {
        "rows": rows,
        "best_exact_accuracy": max(exact_values) if exact_values else None,
        "last_exact_accuracy": exact_values[-1] if exact_values else None,
    }


def _validate_zip(zip_path: Path | None) -> dict[str, Any]:
    if zip_path is None or not zip_path.exists():
        return {"present": False, "ok": False, "bad_member": None, "member_count": 0}
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad_member = zf.testzip()
        member_count = len(zf.infolist())
    return {
        "present": True,
        "ok": bad_member is None,
        "bad_member": bad_member,
        "member_count": member_count,
    }


def _parse_sha256_manifest(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    entries: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            continue
        entries[parts[1].lstrip("*").strip()] = parts[0]
    return entries


def _build_file_inventory(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        files.append(
            {
                "path": _relative(path, root),
                "basename": path.name,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return files


def _extract_event_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    step_events = [event for event in events if event.get("event") == "step"]
    checkpoints = [
        event
        for event in events
        if event.get("event") in {"checkpoint_save", "checkpoint_written", "final_checkpoint_written"}
    ]
    failures = [
        event
        for event in events
        if event.get("event") in {"error", "fatal", "oom", "nan_detected"}
        or str(event.get("reason", "")).upper() in {"OOM", "NAN"}
    ]
    return {
        "training_start_seen": any(event.get("event") == "training_start" for event in events),
        "step_count": len(step_events),
        "first_step": step_events[0].get("step") if step_events else None,
        "last_step": step_events[-1].get("step") if step_events else None,
        "checkpoint_event_count": len(checkpoints),
        "failure_event_count": len(failures),
        "failure_events": failures[:10],
    }


def _required_artifact_status(root: Path, files: list[dict[str, Any]]) -> dict[str, Any]:
    present_basenames = {item["basename"] for item in files}
    present_dirs = {path.name for path in root.rglob("*") if path.is_dir()}
    missing_files = sorted(REQUIRED_BASENAMES - present_basenames)
    missing_dirs = sorted(REQUIRED_DIR_NAMES - present_dirs)
    checkpoints = _find_all(root, CHECKPOINT_BASENAMES)
    return {
        "missing_required_files": missing_files,
        "missing_required_dirs": missing_dirs,
        "checkpoint_files": [_relative(path, root) for path in checkpoints],
        "last_checkpoint_present": any(path.name == "last.pt" for path in checkpoints),
        "final_checkpoint_present": any(path.name == "final.pt" for path in checkpoints),
        "required_files_ok": not missing_files,
        "required_dirs_ok": not missing_dirs,
    }


def process_ocean_output(outputs_dir: str | Path, dest_dir: str | Path) -> dict[str, Any]:
    """Process an Ocean output folder into post-run summary artifacts."""
    src = Path(outputs_dir).expanduser().resolve()
    dst = Path(dest_dir).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"outputs_dir not found: {src}")
    if not src.is_dir():
        raise NotADirectoryError(f"outputs_dir is not a directory: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    zip_path = _find_first(src, [FINAL_EVIDENCE_ZIP])
    decision_json_path = _find_first(src, ["proof_decision.json"])
    decision_md_path = _find_first(src, ["proof_decision.md"])
    manifest_path = _find_first(src, ["final_math_h200_manifest.json", "manifest.json"])
    report_path = _find_first(src, ["final_math_h200_report.json", "report.json"])
    eval_history_path = _find_first(src, ["eval_history.csv"])
    jsonl_path = _find_first(src, ["final_math_h200.jsonl"])
    sha256_path = _find_first(src, ["sha256.txt"])
    github_audit_path = _find_first(src, ["github_visibility_audit_20260514.md"])

    files = _build_file_inventory(src)
    artifact_status = _required_artifact_status(src, files)
    zip_status = _validate_zip(zip_path)
    decision = _read_json(decision_json_path)
    manifest = _read_json(manifest_path)
    report = _read_json(report_path)
    eval_history = _read_eval_history(eval_history_path)
    events = _read_jsonl_events(jsonl_path)
    event_summary = _extract_event_summary(events)
    sha_entries = _parse_sha256_manifest(sha256_path)
    sha_text = _read_text(sha256_path)

    sha_coverage = {
        basename: basename in sha_text for basename in sorted(REQUIRED_BASENAMES)
    }
    checkpoint_files = artifact_status["checkpoint_files"]
    checkpoint_ok = artifact_status["last_checkpoint_present"] and artifact_status["final_checkpoint_present"]

    operational_pass = bool(
        zip_status["ok"]
        and artifact_status["required_files_ok"]
        and artifact_status["required_dirs_ok"]
        and checkpoint_ok
        and event_summary["training_start_seen"]
        and event_summary["step_count"] > 0
        and event_summary["failure_event_count"] == 0
    )
    learning_signal_pass = bool(
        decision.get("learning_signal_pass")
        or str(decision.get("verdict", "")).upper() == "LEARNING_SIGNAL_PASS"
        or report.get("learning_signal_pass")
    )
    capability_claim_eligible = bool(
        decision.get("capability_claim_eligible")
        or str(decision.get("verdict", "")).upper() == "CAPABILITY_CLAIM_ELIGIBLE"
    )

    missing_hash_coverage = [
        name for name, covered in sha_coverage.items() if not covered and name != FINAL_EVIDENCE_ZIP
    ]
    fail_codes: list[str] = []
    if not zip_status["ok"]:
        fail_codes.append("FINAL_ARCHIVE_INVALID_OR_MISSING")
    if artifact_status["missing_required_files"]:
        fail_codes.append("REQUIRED_ARTIFACTS_MISSING")
    if artifact_status["missing_required_dirs"]:
        fail_codes.append("REQUIRED_ARTIFACT_DIRS_MISSING")
    if not event_summary["training_start_seen"]:
        fail_codes.append("TRAINING_START_NOT_SEEN")
    if event_summary["step_count"] == 0:
        fail_codes.append("NO_STEP_PROGRESS")
    if not checkpoint_ok:
        fail_codes.append("NO_CHECKPOINT_WRITTEN")
    if event_summary["failure_event_count"]:
        fail_codes.append("FATAL_EVENT_SEEN")
    if missing_hash_coverage:
        fail_codes.append("SHA256_COVERAGE_INCOMPLETE")

    summary = {
        "schema_version": 1,
        "source_root_name": src.name,
        "destination_root_name": dst.name,
        "required_artifacts": artifact_status,
        "zip": {
            **zip_status,
            "path": _relative(zip_path, src) if zip_path else None,
            "sha256": sha256_file(zip_path) if zip_path else None,
        },
        "paths": {
            "proof_decision_json": _relative(decision_json_path, src) if decision_json_path else None,
            "proof_decision_md": _relative(decision_md_path, src) if decision_md_path else None,
            "manifest": _relative(manifest_path, src) if manifest_path else None,
            "report": _relative(report_path, src) if report_path else None,
            "eval_history": _relative(eval_history_path, src) if eval_history_path else None,
            "jsonl_log": _relative(jsonl_path, src) if jsonl_path else None,
            "sha256": _relative(sha256_path, src) if sha256_path else None,
            "github_visibility_audit": _relative(github_audit_path, src) if github_audit_path else None,
        },
        "sha256_manifest": {
            "entry_count": len(sha_entries),
            "required_basename_coverage": sha_coverage,
            "missing_required_coverage": missing_hash_coverage,
        },
        "events": event_summary,
        "eval_history": {
            "row_count": len(eval_history["rows"]),
            "best_exact_accuracy": eval_history["best_exact_accuracy"],
            "last_exact_accuracy": eval_history["last_exact_accuracy"],
        },
        "decision": {
            "verdict": decision.get("verdict"),
            "reason_codes": decision.get("reason_codes", []),
            "claim_boundary": decision.get("claim_boundary"),
        },
        "claim_boundary": {
            "operational_evidence_supported": operational_pass,
            "learning_signal_supported": learning_signal_pass,
            "capability_claim_supported": capability_claim_eligible,
            "allowed_statement": (
                "This run may be described as operational evidence only when the "
                "exact artifacts, logs, checkpoint files, and proof_decision output "
                "in this folder are cited."
            ),
            "blocked_statement": (
                "Do not claim trained capability, benchmark verification, production "
                "readiness, or security from this processor alone."
            ),
        },
        "post_run_fail_codes": fail_codes,
        "files": files,
    }

    summary_path = dst / "post_run_summary.json"
    claim_report_path = dst / "post_run_claim_report.md"
    evidence_manifest_path = dst / "evidence_pack_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    generate_post_run_claim_report(summary, claim_report_path)
    evidence_manifest_path.write_text(
        json.dumps({"schema_version": 1, "files": files}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def generate_post_run_claim_report(summary: dict[str, Any], output_path: str | Path) -> None:
    """Write a conservative measured-claim boundary report."""
    claim = summary["claim_boundary"]
    decision = summary["decision"]
    events = summary["events"]
    eval_history = summary["eval_history"]
    required = summary["required_artifacts"]
    lines = [
        "# Post-Run Claim Boundary",
        "",
        "## Verdict Inputs",
        f"- Proof decision verdict: {decision.get('verdict') or 'missing'}",
        f"- Proof decision reason codes: {decision.get('reason_codes') or 'missing'}",
        f"- Training start seen: {events['training_start_seen']}",
        f"- Step count seen: {events['step_count']}",
        f"- Last step seen: {events['last_step']}",
        f"- Checkpoints: {', '.join(required['checkpoint_files']) if required['checkpoint_files'] else 'missing'}",
        f"- Final archive valid: {summary['zip']['ok']}",
        f"- Eval rows: {eval_history['row_count']}",
        f"- Best exact accuracy: {eval_history['best_exact_accuracy']}",
        "",
        "## Claim Boundary",
        f"- Operational evidence supported: {claim['operational_evidence_supported']}",
        f"- Learning signal supported: {claim['learning_signal_supported']}",
        f"- Capability claim supported: {claim['capability_claim_supported']}",
        "",
        "## Allowed Statement",
        f"- {claim['allowed_statement']}",
        "",
        "## Not Supported",
        f"- {claim['blocked_statement']}",
        "",
        "## Post-Run Fail Codes",
        f"- {', '.join(summary['post_run_fail_codes']) if summary['post_run_fail_codes'] else 'none'}",
    ]
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["process_ocean_output", "generate_post_run_claim_report", "sha256_file"]


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 scripts/post_run_processor.py <outputs_dir> <dest_dir>")
    result = process_ocean_output(sys.argv[1], sys.argv[2])
    print(json.dumps({"post_run_fail_codes": result["post_run_fail_codes"]}, sort_keys=True))
