#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORTS = ROOT / "reports"
DEFAULT_ARTIFACTS = ROOT / "artifacts"
BUNDLE_NAME = "mertformer_training_outputs_bundle.zip"
SHA_NAME = f"{BUNDLE_NAME}.sha256"
MANIFEST_JSON = "training_outputs_bundle_manifest.json"
MANIFEST_MD = "training_outputs_bundle_manifest.md"
EXCLUDE_PARTS = {
    ".git",
    ".titan-venv",
    ".lint-venv",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}
EXCLUDE_NAMES = {".env"}


@dataclass(frozen=True)
class BundleRoot:
    rel_path: str
    kind: str
    description: str


ROOT_SPECS = [
    BundleRoot("logs", "dir", "training and runtime logs"),
    BundleRoot("checkpoints", "dir", "generated checkpoints"),
    BundleRoot("reports", "dir", "generated reports and evidence documents"),
    BundleRoot("artifacts", "dir", "artifact-side release outputs"),
    BundleRoot("packages", "dir", "package-side release outputs"),
    BundleRoot("datasets/logits", "dir", "precomputed logits shards"),
    BundleRoot("repro/cuda.lock", "file", "target-machine CUDA lock artifact"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_text(text: str, project_root: Path) -> str:
    return text.replace(str(project_root), "<REPO_ROOT>")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _should_exclude(rel_path: str, path: Path, excluded_relpaths: set[str]) -> bool:
    if rel_path in excluded_relpaths:
        return True
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return True
    name = path.name.lower()
    if name in EXCLUDE_NAMES or name.startswith(".env."):
        return True
    return False


def collect_entries(project_root: Path, excluded_relpaths: set[str]) -> tuple[list[dict], list[dict], list[str]]:
    entries: list[dict] = []
    root_rows: list[dict] = []
    missing_roots: list[str] = []

    for spec in ROOT_SPECS:
        target = project_root / spec.rel_path
        row = {
            "path": spec.rel_path,
            "kind": spec.kind,
            "description": spec.description,
            "exists": target.exists(),
            "included_files": 0,
        }
        if not target.exists():
            missing_roots.append(spec.rel_path)
            root_rows.append(row)
            continue

        if spec.kind == "file":
            rel_path = str(target.relative_to(project_root))
            if not _should_exclude(rel_path, target, excluded_relpaths):
                stat = target.stat()
                entries.append(
                    {
                        "path": rel_path,
                        "size_bytes": int(stat.st_size),
                        "sha256": sha256_file(target),
                    }
                )
                row["included_files"] = 1
            root_rows.append(row)
            continue

        count = 0
        for candidate in sorted(target.rglob("*")):
            if not candidate.is_file():
                continue
            rel_path = str(candidate.relative_to(project_root))
            if _should_exclude(rel_path, candidate, excluded_relpaths):
                continue
            stat = candidate.stat()
            entries.append(
                {
                    "path": rel_path,
                    "size_bytes": int(stat.st_size),
                    "sha256": sha256_file(candidate),
                }
            )
            count += 1
        row["included_files"] = count
        root_rows.append(row)

    entries.sort(key=lambda item: item["path"])
    return entries, root_rows, missing_roots


def verify_zip(zip_path: Path, project_root: Path) -> dict[str, object]:
    with zipfile.ZipFile(zip_path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"zip CRC check failed for member: {bad_member}")
        names = archive.namelist()

    unzip_available = shutil.which("unzip")
    unzip_result: dict[str, object]
    if unzip_available:
        proc = subprocess.run(
            [unzip_available, "-t", str(zip_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        unzip_result = {
            "available": True,
            "return_code": proc.returncode,
            "stdout_tail": sanitize_text(proc.stdout[-2000:], project_root),
            "stderr_tail": sanitize_text(proc.stderr[-2000:], project_root),
        }
        if proc.returncode != 0:
            raise RuntimeError(f"`unzip -t` failed for {zip_path.name}")
    else:
        unzip_result = {
            "available": False,
            "return_code": None,
            "stdout_tail": "",
            "stderr_tail": "unzip command unavailable; used zipfile CRC validation only",
        }

    return {
        "entry_count": len(names),
        "unzip_t": unzip_result,
    }


def build_manifest_md(payload: dict) -> str:
    lines = [
        "# Training Outputs Bundle Manifest",
        "",
        f"- generated_utc: `{payload['generated_utc']}`",
        f"- project_root: `{payload['project_root']}`",
        f"- bundle_zip: `{payload['bundle_zip']}`",
        f"- bundle_sha256_file: `{payload['bundle_sha256_file']}`",
        f"- bundle_sha256: `{payload['bundle_sha256']}`",
        f"- included_files: `{payload['included_files']}`",
        f"- missing_roots: `{len(payload['missing_roots'])}`",
        "",
        "## Root Coverage",
        "",
    ]
    for row in payload["roots"]:
        lines.append(
            f"- `{row['path']}` | exists=`{row['exists']}` | kind=`{row['kind']}` | included_files=`{row['included_files']}` | {row['description']}"
        )
    lines.extend(["", "## Missing Roots", ""])
    if payload["missing_roots"]:
        lines.extend(f"- `{item}`" for item in payload["missing_roots"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- zipfile_crc_ok: `{payload['integrity']['zipfile_crc_ok']}`",
            f"- unzip_t_available: `{payload['integrity']['unzip_t']['available']}`",
            f"- unzip_t_return_code: `{payload['integrity']['unzip_t']['return_code']}`",
            "",
            "## Included Files",
            "",
        ]
    )
    if payload["entries"]:
        lines.extend(
            f"- `{entry['path']}` | size_bytes=`{entry['size_bytes']}` | sha256=`{entry['sha256']}`"
            for entry in payload["entries"]
        )
    else:
        lines.append("- none")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the canonical run-output bundle zip.")
    parser.add_argument("--project-root", default=str(ROOT), help="Project root to bundle.")
    parser.add_argument("--reports-dir", help="Override reports directory.")
    parser.add_argument("--artifacts-dir", help="Override artifacts directory.")
    return parser.parse_args()


DASHBOARD_NAME = "training_dashboard.png"


def ensure_training_dashboard(project_root: Path, reports_dir: Path) -> dict:
    """Render reports/training_dashboard.png from the newest run log, if it is stale.

    [2026-07-30] The bundle is the ONE choke point every lane passes through, so the
    "a finished run ships with its charts" guarantee is attached here rather than to a
    single launcher. Before this, ``scripts/plot_training_log.py`` was wired only into
    ``scripts/one_command_full_sop.sh`` behind an off-by-default flag -- and AFTER the
    bundle step, so even there a fresh dashboard could not reach the zip. Neither
    ``launch_ocean_45k.sh`` (which calls this script directly) nor ``launch_8xb300.sh``
    (which ``exec``s zero_touch_start.sh and has no post-run hook at all) rendered
    anything, so a completed 45K run produced no visualisation.

    Rendering here happens BEFORE the archive is walked, so reports/ -- which is bundled
    recursively -- picks the image up automatically.

    Best-effort by construction: any failure is recorded in the manifest and the bundle
    proceeds. A charting problem must never cost the operator the outputs of a run that
    consumed real GPU hours. Set TITAN_BUNDLE_SKIP_PLOT=1 to opt out entirely.
    """
    status: dict = {"attempted": False, "rendered": False, "reason": None}

    if os.environ.get("TITAN_BUNDLE_SKIP_PLOT", "0").strip().lower() in {"1", "true", "yes"}:
        status["reason"] = "disabled_by_TITAN_BUNDLE_SKIP_PLOT"
        return status

    logs_dir = project_root / "logs"
    candidates = sorted(logs_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True) \
        if logs_dir.is_dir() else []
    if not candidates:
        status["reason"] = "no_run_log_found"
        return status

    newest_log = candidates[0]
    dashboard = reports_dir / DASHBOARD_NAME
    if dashboard.exists() and dashboard.stat().st_mtime >= newest_log.stat().st_mtime:
        status["reason"] = "already_current"
        status["source_log"] = str(newest_log.relative_to(project_root))
        return status

    status["attempted"] = True
    status["source_log"] = str(newest_log.relative_to(project_root))
    plotter = project_root / "scripts" / "plot_training_log.py"
    if not plotter.exists():
        status["reason"] = "plot_training_log.py_missing"
        return status

    try:
        proc = subprocess.run(
            [sys.executable, str(plotter), str(newest_log), "--out", str(dashboard)],
            cwd=str(project_root), capture_output=True, text=True, timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        status["reason"] = f"render_failed: {exc}"
        return status

    if dashboard.exists():
        status["rendered"] = True
        status["reason"] = "rendered"
        status["path"] = str(dashboard.relative_to(project_root))
    else:
        # plot_training_log.py exits 0 with a warning when matplotlib is unavailable.
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        status["reason"] = f"no_image_produced (rc={proc.returncode}): {tail[-1] if tail else ''}"
    return status


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    reports_dir = Path(args.reports_dir).resolve() if args.reports_dir else project_root / "reports"
    artifacts_dir = Path(args.artifacts_dir).resolve() if args.artifacts_dir else project_root / "artifacts"

    reports_dir.mkdir(parents=True, exist_ok=True)
    dashboard_status = ensure_training_dashboard(project_root, reports_dir)
    print(f"training_dashboard: {dashboard_status.get('reason')}")
    bundle_zip = artifacts_dir / BUNDLE_NAME
    bundle_sha = artifacts_dir / SHA_NAME
    manifest_json = reports_dir / MANIFEST_JSON
    manifest_md = reports_dir / MANIFEST_MD

    ensure_parent(bundle_zip)
    ensure_parent(manifest_json)

    temp_zip = artifacts_dir / f".{BUNDLE_NAME}.tmp"
    if temp_zip.exists():
        temp_zip.unlink()

    excluded_relpaths = {
        str(bundle_zip.relative_to(project_root)),
        str(bundle_sha.relative_to(project_root)),
        str(manifest_json.relative_to(project_root)),
        str(manifest_md.relative_to(project_root)),
        str(temp_zip.relative_to(project_root)),
    }
    entries, root_rows, missing_roots = collect_entries(project_root, excluded_relpaths)

    with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for entry in entries:
            archive.write(project_root / entry["path"], arcname=entry["path"])

    verify_result = verify_zip(temp_zip, project_root)
    os.replace(temp_zip, bundle_zip)

    bundle_digest = sha256_file(bundle_zip)
    write_text(bundle_sha, f"{bundle_digest}  {bundle_zip.name}")

    payload = {
        "schema": "training_outputs_bundle_manifest_v1",
        "generated_utc": utc_now(),
        "project_root": "<REPO_ROOT>",
        "bundle_zip": str(bundle_zip.relative_to(project_root)),
        "bundle_sha256_file": str(bundle_sha.relative_to(project_root)),
        "bundle_sha256": bundle_digest,
        "included_files": len(entries),
        "entries": entries,
        "roots": root_rows,
        "missing_roots": missing_roots,
        "integrity": {
            "zipfile_crc_ok": True,
            "zip_entry_count": verify_result["entry_count"],
            "unzip_t": verify_result["unzip_t"],
        },
        "exclusions": {
            "explicit_relpaths": sorted(excluded_relpaths),
            "exclude_parts": sorted(EXCLUDE_PARTS),
            "exclude_names": sorted(EXCLUDE_NAMES),
        },
        "boundary_note": "This bundle contains generated/run outputs only. Raw training corpora and secret-bearing surfaces are intentionally excluded.",
    }
    manifest_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_text(manifest_md, build_manifest_md(payload))
    print(json.dumps({"bundle_zip": payload["bundle_zip"], "included_files": payload["included_files"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
