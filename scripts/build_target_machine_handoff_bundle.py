#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"
MANIFEST_JSON = REPORTS / "target_machine_handoff_manifest.json"
MANIFEST_MD = REPORTS / "target_machine_handoff_manifest.md"
BUNDLE_ZIP = ARTIFACTS / "target_machine_handoff_bundle.zip"
BUNDLE_SHA256 = ARTIFACTS / "target_machine_handoff_bundle.zip.sha256"
README_NAME = "TARGET_MACHINE_README.md"
TRANSFER_FILE_CANDIDATES = [
    "zero_touch_start.sh",
    "run.sh",
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


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(ROOT)
        rel_text = rel.as_posix()
        return "<REPO_ROOT>" if not rel_text else f"<REPO_ROOT>/{rel_text}"
    except ValueError:
        return str(resolved)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_transfer_files() -> list[str]:
    decision = load_json(REPORTS / "start_gate_operator_decision.json")
    candidates = decision.get("required_transfer_files") or list(TRANSFER_FILE_CANDIDATES)
    resolved: list[str] = []
    for rel in candidates:
        if not isinstance(rel, str):
            continue
        if (ROOT / rel).exists() and rel not in resolved:
            resolved.append(rel)
    return resolved


def build_operator_steps(decision: dict) -> list[str]:
    recommended_path = str(decision.get("recommended_path") or "offline_clean")
    online_hint = "HF_TOKEN=... TITAN_OFFLINE=0 TITAN_INSTALL=1 TITAN_PROFILE=stable bash zero_touch_start.sh"
    return [
        "Copy or extract this bundle onto the target training machine.",
        "Run `bash zero_touch_start.sh --check-only` first.",
        "If the target machine start gate remains green, launch the canonical path immediately.",
        f"Canonical offline-clean launcher: `TITAN_INSTALL=1 TITAN_PROFILE=stable bash zero_touch_start.sh`",
        f"Optional online teacher lane only if intentionally chosen: `{online_hint}`",
        f"Current repo-side recommended_path is `{recommended_path}`.",
    ]


def build_bundle_readme(decision: dict, files: list[str], steps: list[str]) -> str:
    lines = [
        "# Target Machine Handoff Bundle",
        "",
        f"- generated_utc: `{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`",
        f"- next_action: `{decision.get('next_action', 'UNKNOWN')}`",
        f"- train_allowed: `{decision.get('train_allowed', False)}`",
        f"- decision_reason_code: `{decision.get('decision_reason_code') or 'none'}`",
        f"- recommended_path: `{decision.get('recommended_path') or 'none'}`",
        "",
        "## Exact Steps",
        "",
    ]
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(steps, start=1))
    lines.extend(["", "## Included Files", ""])
    lines.extend(f"- `{item}`" for item in files)
    return "\n".join(lines)


def build_manifest(decision: dict, files: list[str], steps: list[str]) -> dict:
    entries = []
    for rel in files:
        full = ROOT / rel
        entries.append(
            {
                "path": rel,
                "size_bytes": full.stat().st_size,
                "sha256": sha256_file(full),
            }
        )
    return {
        "schema": "target_machine_handoff_manifest_v1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "next_action": decision.get("next_action", "UNKNOWN"),
        "train_allowed": bool(decision.get("train_allowed", False)),
        "decision_reason_code": decision.get("decision_reason_code"),
        "recommended_path": decision.get("recommended_path"),
        "bundle_path": display_path(BUNDLE_ZIP),
        "bundle_sha256_path": display_path(BUNDLE_SHA256),
        "transfer_files": entries,
        "operator_steps": steps,
    }


def build_manifest_md(manifest: dict) -> str:
    lines = [
        "# Target Machine Handoff Manifest",
        "",
        f"- next_action: `{manifest['next_action']}`",
        f"- train_allowed: `{manifest['train_allowed']}`",
        f"- decision_reason_code: `{manifest.get('decision_reason_code') or 'none'}`",
        f"- recommended_path: `{manifest.get('recommended_path') or 'none'}`",
        f"- bundle_path: `{Path(manifest['bundle_path']).name}`",
        f"- bundle_sha256_path: `{Path(manifest['bundle_sha256_path']).name}`",
        "",
        "## Operator Steps",
        "",
    ]
    lines.extend(f"{idx}. {step}" for idx, step in enumerate(manifest["operator_steps"], start=1))
    lines.extend(["", "## Transfer Files", ""])
    for item in manifest["transfer_files"]:
        lines.append(f"- `{item['path']}` · `{item['size_bytes']}` bytes · sha256=`{item['sha256']}`")
    return "\n".join(lines)


def write_bundle(files: list[str], readme_text: str, manifest: dict) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="target_machine_handoff_") as tmp:
        stage = Path(tmp)
        for rel in files:
            src = ROOT / rel
            dest = stage / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        (stage / README_NAME).write_text(readme_text.rstrip() + "\n", encoding="utf-8")
        (stage / "reports").mkdir(parents=True, exist_ok=True)
        (stage / "reports" / MANIFEST_JSON.name).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (stage / "reports" / MANIFEST_MD.name).write_text(build_manifest_md(manifest).rstrip() + "\n", encoding="utf-8")
        with zipfile.ZipFile(BUNDLE_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in sorted(stage.rglob("*")):
                if item.is_file():
                    zf.write(item, item.relative_to(stage).as_posix())
    write_text(BUNDLE_SHA256, f"{sha256_file(BUNDLE_ZIP)}  {BUNDLE_ZIP.name}")


def main() -> int:
    decision = load_json(REPORTS / "start_gate_operator_decision.json")
    files = resolve_transfer_files()
    steps = build_operator_steps(decision)
    manifest = build_manifest(decision, files, steps)
    readme_text = build_bundle_readme(decision, files, steps)
    write_text(MANIFEST_JSON, json.dumps(manifest, indent=2, ensure_ascii=False))
    write_text(MANIFEST_MD, build_manifest_md(manifest))
    write_bundle(files, readme_text, manifest)
    print(
        json.dumps(
            {
                "bundle": display_path(BUNDLE_ZIP),
                "sha256": display_path(BUNDLE_SHA256),
                "manifest_json": display_path(MANIFEST_JSON),
                "manifest_md": display_path(MANIFEST_MD),
                "file_count": len(files),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
