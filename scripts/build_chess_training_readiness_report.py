#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
REPORT_JSON = ROOT / "reports" / "chess_training_readiness_report.json"
REPORT_MD = ROOT / "reports" / "chess_training_readiness_report.md"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def display_path(path: Path) -> str:
    resolved = path.resolve()
    root_resolved = ROOT.resolve()
    if resolved == root_resolved:
        return "<REPO_ROOT>"
    try:
        rel = resolved.relative_to(root_resolved)
        rel_text = rel.as_posix()
        return f"<REPO_ROOT>/{rel_text}" if rel_text else "<REPO_ROOT>"
    except ValueError:
        return str(resolved)


def build_payload() -> dict[str, Any]:
    canonical_onefile = ROOT / "scripts" / "chess_5080_onefile.py"
    export_share = ROOT / "scripts" / "export_chess_5080_share.py"
    windows_builder = ROOT / "scripts" / "build_chess_5080_windows_delivery.py"
    gui_script = ROOT / "apps" / "chess_gui" / "play_mertformer_chess_web.py"
    gui_sync = load_json(ROOT / "reports" / "chess_gui_onefile_sync_report.json")
    teaching = load_json(ROOT / "reports" / "chess_teaching_contract_report.json")
    extension = load_json(ROOT / "reports" / "chess_onefile_extension_report.json")

    checks = [
        {
            "id": "canonical_onefile",
            "status": canonical_onefile.exists(),
            "required": True,
            "detail": display_path(canonical_onefile),
        },
        {
            "id": "share_export",
            "status": export_share.exists(),
            "required": True,
            "detail": display_path(export_share),
        },
        {
            "id": "windows_builder",
            "status": windows_builder.exists(),
            "required": True,
            "detail": display_path(windows_builder),
        },
        {
            "id": "gui_app",
            "status": gui_script.exists(),
            "required": True,
            "detail": display_path(gui_script),
        },
        {
            "id": "gui_sync_state",
            "status": gui_sync.get("status") in {"up_to_date", "canonical_fallback_ready"},
            "required": True,
            "detail": gui_sync.get("status") or "missing_report",
        },
        {
            "id": "teaching_contract_smoke",
            "status": bool(teaching.get("summary", {}).get("all_green")),
            "required": True,
            "detail": str(teaching.get("summary", {}).get("all_green", False)).lower(),
        },
        {
            "id": "onefile_extension_report",
            "status": bool(extension.get("summary", {}).get("all_green")),
            "required": True,
            "detail": extension.get("final_status") or "missing_report",
        },
        {
            "id": "stockfish_anchor_optional",
            "status": True,
            "required": False,
            "detail": "Stockfish stays optional for training start and required later for anchor benchmark runs.",
        },
    ]
    required_green = all(check["status"] for check in checks if check["required"])
    return {
        "schema": "chess_training_readiness_report_v1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "final_status": "READY_FOR_TRAINING" if required_green else "NOT_READY",
        "summary": {
            "required_total": sum(1 for check in checks if check["required"]),
            "required_green": sum(1 for check in checks if check["required"] and check["status"]),
            "all_required_green": required_green,
        },
        "checks": checks,
        "canonical_train_command": "python3 scripts/chess_5080_onefile.py --mode train",
        "canonical_verify_command": "python3 scripts/chess_5080_onefile.py --mode verify",
        "notes": [
            "This report covers repo-side training readiness only.",
            "Real training, checkpoints, and benchmark outputs remain post-run evidence.",
            "A missing local GUI onefile copy is acceptable when repo-canonical fallback is available.",
        ],
    }


def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Chess Training Readiness Report",
        "",
        f"- final_status: `{payload['final_status']}`",
        f"- required_green: `{payload['summary']['required_green']}/{payload['summary']['required_total']}`",
        f"- canonical_train_command: `{payload['canonical_train_command']}`",
        f"- canonical_verify_command: `{payload['canonical_verify_command']}`",
        "",
        "## Checks",
        "",
        "| Check | Required | Status | Detail |",
        "| --- | --- | --- | --- |",
    ]
    for check in payload["checks"]:
        lines.append(
            f"| `{check['id']}` | `{check['required']}` | `{check['status']}` | {check['detail']} |"
        )
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in payload["notes"])
    return "\n".join(lines)


def main() -> int:
    payload = build_payload()
    REPORT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT_MD.write_text(build_markdown(payload).rstrip() + "\n", encoding="utf-8")
    print(json.dumps({"final_status": payload["final_status"], "all_required_green": payload["summary"]["all_required_green"]}))
    return 0 if payload["summary"]["all_required_green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
