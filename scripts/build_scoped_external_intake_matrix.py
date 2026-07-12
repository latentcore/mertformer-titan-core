#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

HOME = Path.home()
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
REPORT_JSON = REPORTS / "scoped_external_intake_matrix.json"
REPORT_MD = REPORTS / "scoped_external_intake_matrix.md"
IMMUTABLE_JSON = REPORTS / "immutable_evidence_register.json"
IMMUTABLE_MD = REPORTS / "immutable_evidence_register.md"

PROJECT_KEYWORDS = (
    "mertformer",
    "onyxstorm",
    "mertformerstream",
    "mertformer_outputs",
    "mertos_core",
    "chess_5080",
    "build30",
)
IMMUTABLE_MARKERS = (
    "immutable",
    "output",
    "result_",
    "snapshot",
    "smart_dump",
    "vault",
    "evidence",
    "history",
    "linkedin_run",
    "failed_",
)
IGNORE_NAMES = {
    ".git",
    ".idea",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".titan-venv",
    ".lint-venv",
    "node_modules",
}

SCOPED_PATTERNS = [
    HOME / "Desktop" / "MertFormer_45K_Launch_Bundle_20260401_2130",
    HOME / "Desktop" / "MertFormer_45K_Launch_Bundle_20260401_2130.zip",
    HOME / "Desktop" / "MertFormer_45K_Launch_Bundle_20260401_2130.zip.sha256",
    HOME / "Desktop" / "MertFormerStream",
    HOME / "Documents" / "mertformer_outputs_LINKEDIN_run_20260220_175540.zip",
    HOME / "Documents" / "mertformer_outputs_LINKEDIN_run_20260220_175540.zip.sha256",
    HOME / "Documents" / "mertformer-titan-core.zip",
    HOME / "Documents" / "mertformer-titan-core.zip.sha256",
    HOME / "Documents" / "mertformer_outputs",
    HOME / "Documents" / "MertFormer_Strategy_Vault_2026_04_27",
    HOME / "Downloads" / "MertOS_Core" / "mertformer_cleanup_20260110_015202.log",
    HOME / "Downloads" / "MertOS_Core" / "mertformer_cleanup_20260110_015332.log",
    HOME / "Downloads" / "content" / "mertformer_outputs",
    Path("/Applications") / "mertformer-titan-core.zip",
    Path("/Applications") / "mertformer-titan-core.zip.sha256",
    Path("/Applications") / "MertFormerChessDownload",
]

SCAN_ROOTS = [
    {"label": "Desktop", "path": HOME / "Desktop", "max_depth": 3, "mutation_policy": "project_only"},
    {"label": "Documents", "path": HOME / "Documents", "max_depth": 4, "mutation_policy": "project_only"},
    {"label": "Downloads", "path": HOME / "Downloads", "max_depth": 4, "mutation_policy": "project_only"},
    {"label": "Applications", "path": Path("/Applications"), "max_depth": 3, "mutation_policy": "audit_only_unrelated"},
    {"label": "UserApplications", "path": HOME / "Applications", "max_depth": 3, "mutation_policy": "audit_only_unrelated"},
]


def now_local() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def safe_sha256_file(path: Path) -> Optional[str]:
    """sha256_file guarded against unreadable external files (e.g. Downloads-folder
    entries this process lacks OS permission to open). This scanner walks
    scoped-but-external locations it does not own; one unreadable file must not
    abort the entire scoped-intake audit (and, transitively, verify_all.sh)."""
    try:
        return sha256_file(path)
    except OSError as exc:
        print(f"WARN: skip unreadable scoped-intake file {path}: {exc}", file=sys.stderr)
        return None


def sanitize_path(path: Path) -> str:
    resolved = path.resolve()
    root_resolved = ROOT.resolve()
    home_resolved = HOME.resolve()
    resolved_str = str(resolved)
    root_str = str(root_resolved)
    home_str = str(home_resolved)
    if resolved_str == root_str:
        return "<REPO_ROOT>"
    if resolved_str.startswith(root_str + "/"):
        return resolved_str.replace(root_str, "<REPO_ROOT>", 1)
    if resolved_str == home_str:
        return "<HOME>"
    if resolved_str.startswith(home_str + "/"):
        return resolved_str.replace(home_str, "<HOME>", 1)
    return resolved_str


def matches_project(path: Path) -> bool:
    lowered = str(path).lower()
    return any(keyword in lowered for keyword in PROJECT_KEYWORDS)


def is_inside_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT.resolve())
        return True
    except ValueError:
        return False


def is_project_safe_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path in {HOME / "Desktop", HOME / "Documents", HOME / "Downloads", Path("/Applications"), HOME / "Applications"}:
        return False
    return matches_project(path)


def is_immutable_evidence(path: Path) -> tuple[bool, str]:
    lowered = str(path).lower()
    for marker in IMMUTABLE_MARKERS:
        if marker in lowered:
            return True, marker
    return False, ""


def canonical_source_for(path: Path) -> Path | None:
    lowered = str(path).lower()
    if path.name.lower() == "mertformer-titan-core.zip":
        return ARTIFACTS / "mertformer_release.zip"
    if path.name.lower() == "mertformer-titan-core.zip.sha256":
        return ARTIFACTS / "mertformer_release.zip.sha256"
    if path.name == "MertFormer_Build30_Max_Closure_Handoff.md":
        return REPORTS / "repo_external_handoff.md"
    if path.name == "chess_5080_onefile.py" and "mertformerchessdownload" in lowered:
        return ROOT / "scripts" / "chess_5080_onefile.py"
    return None


def walk_project_hits(root: Path, max_depth: int) -> Iterable[Path]:
    if not root.exists():
        return []

    hits: list[Path] = []
    root_depth = len(root.parts)
    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        depth = len(current_path.parts) - root_depth
        dirs[:] = [name for name in dirs if name not in IGNORE_NAMES]
        if depth >= max_depth:
            dirs[:] = []
        for name in dirs + files:
            if name in IGNORE_NAMES:
                continue
            candidate = current_path / name
            if is_inside_repo(candidate):
                continue
            if matches_project(candidate):
                hits.append(candidate)
    return hits


def classify(path: Path, duplicate_rank: int, immutable: bool, canonical_source: Path | None) -> tuple[str, str]:
    if immutable:
        return "preserve_immutable_evidence", "preserve_immutable"
    if canonical_source is not None:
        return "sync_with_canonical_source", "project_sync"
    if path.is_dir() and is_project_safe_dir(path):
        return "keep_project_directory", "project_safe_cleanup"
    if duplicate_rank > 1 and path.suffix == ".zip":
        return "delete_as_stale_generated", "project_safe_cleanup"
    if path.suffix == ".sha256":
        return "keep_sidecar_metadata", "audit_only"
    if path.is_dir():
        return "keep_as_external_artifact", "audit_only"
    return "keep_as_external_artifact", "audit_only"


def relative_source_root(path: Path) -> str:
    for spec in SCAN_ROOTS:
        root = Path(spec["path"]).resolve()
        try:
            path.resolve().relative_to(root)
        except ValueError:
            continue
        return str(spec["label"])
    return "explicit"


def collect_entries() -> List[Dict[str, Any]]:
    collected: list[Path] = []
    for path in SCOPED_PATTERNS:
        if path.exists():
            collected.append(path)
    for spec in SCAN_ROOTS:
        root = Path(spec["path"])
        max_depth = int(spec["max_depth"])
        collected.extend(walk_project_hits(root, max_depth))

    dedup: dict[str, Path] = {}
    for path in collected:
        if not path.exists():
            continue
        if is_inside_repo(path):
            continue
        dedup[str(path.resolve())] = path
    present = sorted(dedup.values(), key=lambda item: str(item).lower())

    zip_hash_groups: Dict[str, List[Path]] = defaultdict(list)
    for path in present:
        if path.is_file() and path.suffix == ".zip":
            zip_sha = safe_sha256_file(path)
            if zip_sha is not None:
                zip_hash_groups[zip_sha].append(path)

    entries: List[Dict[str, Any]] = []
    for path in present:
        sha = None
        size = None
        duplicate_rank = 1
        immutable, immutable_reason = is_immutable_evidence(path)
        canonical_source = canonical_source_for(path)
        if path.is_file():
            size = path.stat().st_size
            sha = safe_sha256_file(path)
            if sha is not None and path.suffix == ".zip" and canonical_source is None and not immutable:
                duplicate_rank = len(zip_hash_groups[sha])
        disposition, mutation_policy = classify(path, duplicate_rank, immutable, canonical_source)
        entries.append(
            {
                "path": sanitize_path(path),
                "source_root": relative_source_root(path),
                "kind": "dir" if path.is_dir() else "file",
                "exists": True,
                "size_bytes": size,
                "sha256": sha,
                "duplicate_rank": duplicate_rank,
                "duplicate_group_size": duplicate_rank if path.suffix == ".zip" else 1,
                "disposition": disposition,
                "mutation_policy": mutation_policy,
                "immutable_evidence": immutable,
                "immutable_reason": immutable_reason or None,
                "canonical_source": sanitize_path(canonical_source) if canonical_source is not None else None,
            }
        )
    return entries


def write_sidecar(zip_path: Path) -> None:
    sidecar = zip_path.with_suffix(zip_path.suffix + ".sha256")
    sidecar.write_text(f"{sha256_file(zip_path)}  {zip_path.name}\n", encoding="utf-8")


def run_unlock_command(args: list[str]) -> None:
    try:
        subprocess.run(args, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        # Linux runners do not ship macOS chflags; unlock remains best-effort.
        return


def unlock_target(target: Path) -> None:
    if not target.exists():
        return
    for args in (
        ["chflags", "nouchg", str(target)],
        ["chflags", "noschg", str(target)],
        ["chflags", "nouchg,noschg", str(target)],
    ):
        run_unlock_command(args)
    try:
        current_mode = target.stat().st_mode
        os.chmod(target, current_mode | 0o200)
    except OSError:
        # best-effort chmod; tutulamayan izin/erişim hatalarını sessizce geç
        pass


def copy_file(source: Path, target: Path) -> str | None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        unlock_target(target)
        shutil.copy2(source, target)
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def build_handoff_dir_name() -> str:
    stamp = datetime.now().strftime("%Y_%m_%d")
    return f"MertFormer_Final_PreTraining_Closure_{stamp}"


def readiness_payload() -> dict:
    path = REPORTS / "train_readiness_decision.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def render_final_truth_summary(readiness: dict) -> str:
    blockers = [str(item) for item in readiness.get("blockers", []) if str(item).strip()]
    blocker_text = ", ".join(blockers) if blockers else "none"
    return "\n".join(
        [
            "# Final Truth Summary",
            "",
            f"- generated_local: `{now_local()}`",
            f"- final_status: `{readiness.get('final_status', 'UNKNOWN')}`",
            f"- decision_reason_code: `{readiness.get('decision_reason_code', 'UNKNOWN')}`",
            f"- recommended_path: `{readiness.get('recommended_path', 'none')}`",
            "- strict_local_lane: `offline_clean` remains the strict precomputed-KD lane and is not the active recommended path for this pass.",
            f"- remaining_non_winning_blockers: `{blocker_text}`",
            "- post_run_missing: trained final weights, best/latest checkpoint proof, checkpoint-bound benchmark outputs, trained demo bundle, and trained export/device measurements.",
            "",
            "## Exact Boundary",
            "- Repo-side readiness and trained evidence are not the same thing.",
            "- `remote_bootstrap` is the active recommended lane for rented-machine start.",
            "- `offline_clean` remains valid only as the strict local precomputed-KD lane.",
            "- `online_teacher` remains an explicit HF-token-gated lane rather than the default start path.",
        ]
    ) + "\n"


def render_gpu_operator_quickstart(readiness: dict) -> str:
    return "\n".join(
        [
            "# GPU Operator Quick Start",
            "",
            "1. Copy `target_machine_handoff_bundle.zip` onto the target training machine.",
            "2. Extract it and run `bash zero_touch_start.sh --check-only` first.",
            "3. Inject `HF_TOKEN` on the target machine for the active `remote_bootstrap` lane.",
            "4. Start the owned run with:",
            "",
            "```bash",
            "HF_TOKEN=... TITAN_OFFLINE=0 TITAN_INSTALL=1 TITAN_PROFILE=stable bash zero_touch_start.sh",
            "```",
            "",
            f"5. Confirm the repo-side contract still shows `{readiness.get('final_status', 'UNKNOWN')}` / `{readiness.get('decision_reason_code', 'UNKNOWN')}` before the long run.",
        ]
    ) + "\n"


def sync_external_artifacts(entries: list[dict], sync_mode: str) -> dict:
    release_zip = ARTIFACTS / "mertformer_release.zip"
    release_sha = ARTIFACTS / "mertformer_release.zip.sha256"
    handoff_zip = ARTIFACTS / "target_machine_handoff_bundle.zip"
    handoff_sha = ARTIFACTS / "target_machine_handoff_bundle.zip.sha256"
    handoff_md = REPORTS / "repo_external_handoff.md"
    sop_summary = REPORTS / "one_command_full_sop_summary.md"
    truth_matrix = REPORTS / "final_truth_matrix.md"
    repo_audit = REPORTS / "final_repo_audit.md"
    readiness_md = REPORTS / "train_readiness_decision.md"
    handoff_manifest_md = REPORTS / "target_machine_handoff_manifest.md"
    agi_gap_en = ROOT / "INTERNAL_AGI_GAP.md"
    agi_gap_tr = ROOT / "INTERNAL_AGI_GAP_TR.md"

    audit_rows: list[dict[str, Any]] = []
    apply_errors: list[str] = []
    readiness = readiness_payload()

    sync_rules = [
        {
            "id": "documents_release_zip",
            "source": release_zip,
            "target": HOME / "Documents" / "mertformer-titan-core.zip",
            "kind": "zip",
        },
        {
            "id": "applications_release_zip",
            "source": release_zip,
            "target": Path("/Applications") / "mertformer-titan-core.zip",
            "kind": "zip",
        },
        {
            "id": "desktop_repo_handoff",
            "source": handoff_md,
            "target": HOME / "Desktop" / "MertFormer_Build30_Max_Closure_Handoff.md",
            "kind": "file",
        },
        {
            "id": "applications_chess_onefile",
            "source": ROOT / "scripts" / "chess_5080_onefile.py",
            "target": Path("/Applications") / "MertFormerChessDownload" / "chess_5080_onefile.py",
            "kind": "file",
        },
    ]

    for rule in sync_rules:
        source = Path(rule["source"])
        target = Path(rule["target"])
        if not source.exists():
            audit_rows.append(
                {
                    "id": rule["id"],
                    "source": sanitize_path(source),
                    "target": sanitize_path(target),
                    "status": "missing_source",
                    "changed": False,
                }
            )
            if sync_mode == "apply":
                apply_errors.append(f"missing source for sync rule {rule['id']}: {source}")
            continue

        source_sha = sha256_file(source)
        target_sha = sha256_file(target) if target.exists() and target.is_file() else None
        needs_copy = target_sha != source_sha
        status = "match" if not needs_copy else "mismatch"

        if sync_mode == "apply" and needs_copy:
            error = copy_file(source, target)
            if error is None:
                if rule["kind"] == "zip":
                    write_sidecar(target)
                status = "copied"
                target_sha = sha256_file(target)
            else:
                status = "copy_failed"
                apply_errors.append(f"sync rule {rule['id']} failed for {sanitize_path(target)}: {error}")
        elif sync_mode == "apply" and rule["kind"] == "zip" and target.exists():
            write_sidecar(target)

        audit_rows.append(
            {
                "id": rule["id"],
                "source": sanitize_path(source),
                "target": sanitize_path(target),
                "status": status,
                "changed": status == "copied",
                "source_sha256": source_sha,
                "target_sha256": target_sha,
            }
        )

    handoff_dir = HOME / "Documents" / build_handoff_dir_name()
    handoff_files = [
        (handoff_zip, handoff_zip.name),
        (handoff_sha, handoff_sha.name),
        (release_zip, release_zip.name),
        (release_sha, release_sha.name),
        (truth_matrix, truth_matrix.name),
        (handoff_md, handoff_md.name),
        (repo_audit, repo_audit.name),
        (readiness_md, readiness_md.name),
        (handoff_manifest_md, handoff_manifest_md.name),
        (agi_gap_en, agi_gap_en.name),
        (agi_gap_tr, agi_gap_tr.name),
    ]

    handoff_copies: list[dict[str, Any]] = []
    if sync_mode == "apply":
        handoff_dir.mkdir(parents=True, exist_ok=True)
        for source, target_name in handoff_files:
            if not source.exists():
                apply_errors.append(f"missing handoff source: {source}")
                continue
            target = handoff_dir / target_name
            error = copy_file(source, target)
            if error is not None:
                apply_errors.append(f"handoff copy failed for {sanitize_path(target)}: {error}")
                continue
            handoff_copies.append(
                {
                    "source": sanitize_path(source),
                    "target": sanitize_path(target),
                    "sha256": sha256_file(target),
                }
            )
        (handoff_dir / "FINAL_TRUTH_SUMMARY.md").write_text(render_final_truth_summary(readiness), encoding="utf-8")
        (handoff_dir / "GPU_OPERATOR_QUICKSTART.md").write_text(render_gpu_operator_quickstart(readiness), encoding="utf-8")
        if sop_summary.exists():
            copy_file(sop_summary, handoff_dir / "CLOSURE_EXECUTION_SUMMARY.md")

    return {
        "sync_mode": sync_mode,
        "audit_rows": audit_rows,
        "handoff_dir": sanitize_path(handoff_dir),
        "handoff_dir_real": str(handoff_dir),
        "handoff_copies": handoff_copies,
        "apply_errors": apply_errors,
    }


def write_immutable_register(entries: list[dict]) -> None:
    immutable_entries = [
        {
            "path": entry["path"],
            "kind": entry["kind"],
            "immutable_reason": entry["immutable_reason"],
            "disposition": entry["disposition"],
        }
        for entry in entries
        if entry.get("immutable_evidence")
    ]
    payload = {
        "generated_local": now_local(),
        "entry_count": len(immutable_entries),
        "entries": immutable_entries,
    }
    IMMUTABLE_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Immutable Evidence Register",
        "",
        f"- generated_local: `{payload['generated_local']}`",
        f"- entry_count: `{payload['entry_count']}`",
        "",
        "| Path | Kind | Reason | Disposition |",
        "| --- | --- | --- | --- |",
    ]
    for entry in immutable_entries:
        lines.append(
            f"| `{entry['path']}` | `{entry['kind']}` | `{entry['immutable_reason']}` | `{entry['disposition']}` |"
        )
    if not immutable_entries:
        lines.append("| none | `none` | `none` | `none` |")
    IMMUTABLE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Scoped External Intake Matrix",
        "",
        f"- generated_local: `{payload['generated_local']}`",
        f"- sync_mode: `{payload['sync_mode']}`",
        f"- scope: {payload['scope']}",
        "",
        "## Audited Roots",
        "",
        "| Label | Path | Mutation Policy | Exists |",
        "| --- | --- | --- | --- |",
    ]
    for root in payload["audited_roots"]:
        lines.append(
            f"| `{root['label']}` | `{root['path']}` | `{root['mutation_policy']}` | `{str(root['exists']).lower()}` |"
        )

    lines.extend(
        [
            "",
            "## Project-Related Entries",
            "",
            "| Path | Root | Kind | Disposition | Mutation Policy | Immutable | Canonical Source | SHA256 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for entry in payload["entries"]:
        lines.append(
            f"| `{entry['path']}` | `{entry['source_root']}` | `{entry['kind']}` | `{entry['disposition']}` | `{entry['mutation_policy']}` | `{str(entry['immutable_evidence']).lower()}` | `{entry['canonical_source'] or ''}` | `{entry['sha256'] or ''}` |"
        )

    sync = payload["sync_report"]
    lines.extend(
        [
            "",
            "## External Sync Audit",
            "",
            f"- handoff_dir: `{sync['handoff_dir']}`",
            "",
            "| Rule | Source | Target | Status | Changed |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in sync["audit_rows"]:
        lines.append(
            f"| `{row['id']}` | `{row['source']}` | `{row['target']}` | `{row['status']}` | `{str(row['changed']).lower()}` |"
        )

    if sync["handoff_copies"]:
        lines.extend(["", "## Final Documents Handoff Folder", ""])
        for item in sync["handoff_copies"]:
            lines.append(f"- `{item['target']}` <= `{item['source']}` (sha256=`{item['sha256']}`)")

    if sync["apply_errors"]:
        lines.extend(["", "## Apply Errors", ""])
        lines.extend(f"- {item}" for item in sync["apply_errors"])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build scoped external intake matrix and optional sync report")
    parser.add_argument("--json-out", default=str(REPORT_JSON))
    parser.add_argument("--md-out", default=str(REPORT_MD))
    parser.add_argument("--sync-mode", choices=["audit", "apply"], default="audit")
    args = parser.parse_args()

    json_out = Path(args.json_out)
    md_out = Path(args.md_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)

    entries = collect_entries()
    write_immutable_register(entries)

    audited_roots = [
        {
            "label": str(spec["label"]),
            "path": sanitize_path(Path(spec["path"])),
            "mutation_policy": str(spec["mutation_policy"]),
            "exists": Path(spec["path"]).exists(),
        }
        for spec in SCAN_ROOTS
    ]
    sync_report = sync_external_artifacts(entries, args.sync_mode)
    handoff_dir_real = Path(sync_report.pop("handoff_dir_real"))

    payload = {
        "generated_local": now_local(),
        "sync_mode": args.sync_mode,
        "scope": "Desktop/Documents/Downloads plus application roots for project-related artifacts; unrelated applications remain audit-only.",
        "repo_root": sanitize_path(ROOT),
        "audited_roots": audited_roots,
        "entry_count": len(entries),
        "immutable_evidence_count": sum(1 for entry in entries if entry.get("immutable_evidence")),
        "entries": entries,
        "sync_report": sync_report,
        "immutable_report_json": sanitize_path(IMMUTABLE_JSON),
        "immutable_report_md": sanitize_path(IMMUTABLE_MD),
    }
    json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(md_out, payload)

    if args.sync_mode == "apply":
        if handoff_dir_real.exists():
            extra_targets = [
                (md_out, handoff_dir_real / md_out.name),
                (json_out, handoff_dir_real / json_out.name),
            ]
            for source, target in extra_targets:
                error = copy_file(source, target)
                if error is not None:
                    sync_report["apply_errors"].append(f"handoff sync failed for {sanitize_path(target)}: {error}")
                    continue
                sync_report["handoff_copies"].append(
                    {
                        "source": sanitize_path(source),
                        "target": sanitize_path(target),
                        "sha256": sha256_file(target),
                    }
                )
            payload["sync_report"] = sync_report
            json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            write_markdown(md_out, payload)

    print(
        json.dumps(
            {
                "json": sanitize_path(json_out),
                "md": sanitize_path(md_out),
                "entries": len(entries),
                "sync_mode": args.sync_mode,
                "handoff_dir": sync_report["handoff_dir"],
                "apply_errors": len(sync_report["apply_errors"]),
            },
            indent=2,
        )
    )
    return 1 if args.sync_mode == "apply" and sync_report["apply_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
