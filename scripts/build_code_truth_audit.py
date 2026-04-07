#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent

EVIDENCE_COLUMNS = [
    "code_path",
    "canonical_command",
    "verification",
    "artifact",
]

MARKER_PATTERNS: dict[str, re.Pattern[str]] = {
    "TODO": re.compile(r"\bTODO\b", re.IGNORECASE),
    "FIXME": re.compile(r"\bFIXME\b", re.IGNORECASE),
    "TBD": re.compile(r"\bTBD\b", re.IGNORECASE),
    "scaffold": re.compile(r"\bscaffold\b", re.IGNORECASE),
    "fallback": re.compile(r"\bfallback\b", re.IGNORECASE),
}

SCAN_EXTENSIONS = {
    ".py",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".md",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".sh",
    ".txt",
}

SCAN_DIRS = [
    ROOT / "layers",
    ROOT / "mertformer_sdk",
    ROOT / "scripts",
    ROOT / "tests",
    ROOT / "datasets",
]

SCAN_FILES = [
    ROOT / "AGENTS.md",
    ROOT / "README.md",
    ROOT / "README_TR.md",
    ROOT / "MODEL_CARD.md",
    ROOT / "MODEL_CARD_TR.md",
]

EXCLUDED_DIR_NAMES = {
    ".git",
    ".titan-venv",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}

EXCLUDED_SCAN_PATHS = {
    "scripts/build_code_truth_audit.py",
}

SURFACE_GROUPS = {
    "no-touch": [
        "AGENTS.md",
        "reports/final_backlog_classification.md",
        "reports/final_freeze_manifest.md",
        "reports/source_of_truth_map.md",
        "reports/final_truth_matrix.md",
    ],
    "high-risk": [
        "layers/bitlinear.py",
        "mertformer_sdk/kernels/dispatcher.py",
        "mertformer_sdk/kernels/triton_ternary.py",
        "mertformer_sdk/kernels/cpp/bitnet_cpu.cpp",
        "mertformer_sdk/kernels/metal/engine.py",
        "scripts/chess_5080_onefile.py",
    ],
    "closure": [
        "zero_touch_start.sh",
        "scripts/verify_all.sh",
        "scripts/final_orchestrator.py",
        "scripts/post_train_autorun.py",
        "scripts/build_closure_governance_pack.py",
    ],
    "research": [
        "scripts/train_tpu_turbo.py",
        "reports/phase2_carryover.md",
        "CHESS_5080_POC_INTERNAL_TR.md",
        "OFFLINE_4060_DEMO.md",
    ],
    "desktop-hygiene": [
        "scripts/build_workspace_hygiene_manifest.py",
        "interfaces/workspace_hygiene_manifest_v1.schema.json",
        "reports/workspace_hygiene_manifest.md",
        "reports/workspace_hygiene_manifest.json",
    ],
}

TECHNICAL_SURFACES = [
    {
        "path": "layers/bitlinear.py",
        "lane": "bitnet-kernel",
        "surface_class": "living",
        "maturity": "tested_fallback",
        "canonical_command": [
            "python3 -m pytest -q tests/test_kernel_dispatcher.py tests/test_kernel_equivalence.py tests/test_cpp_kernel_loader.py"
        ],
        "verification": [
            "tests/test_kernel_dispatcher.py",
            "tests/test_kernel_equivalence.py",
            "tests/test_cpp_kernel_loader.py",
        ],
        "artifact": [
            "reports/code_truth_delta_audit.md",
            "reports/final_truth_matrix.md",
        ],
        "source_of_truth": [
            "AGENTS.md",
            "reports/code_truth_contract.md",
            "reports/final_truth_matrix.md",
        ],
        "notes": "BitLinear is correctness-first and dispatch-aware, but production-depth performance claims still depend on backend-specific measurement.",
    },
    {
        "path": "mertformer_sdk/kernels/dispatcher.py",
        "lane": "bitnet-kernel",
        "surface_class": "maintained",
        "maturity": "tested_fallback",
        "canonical_command": [
            "python3 -m pytest -q tests/test_kernel_dispatcher.py"
        ],
        "verification": [
            "tests/test_kernel_dispatcher.py",
        ],
        "artifact": [
            "reports/code_truth_delta_audit.md",
        ],
        "source_of_truth": [
            "reports/code_truth_contract.md",
            "reports/final_truth_matrix.md",
        ],
        "notes": "Dispatcher routing is deterministic and test-covered; it is a contract surface rather than a speed claim by itself.",
    },
    {
        "path": "mertformer_sdk/kernels/triton_ternary.py",
        "lane": "bitnet-performance",
        "surface_class": "living",
        "maturity": "tested_fallback",
        "canonical_command": [
            "python3 -m pytest -q tests/test_kernel_equivalence.py"
        ],
        "verification": [
            "tests/test_kernel_equivalence.py",
        ],
        "artifact": [
            "reports/code_truth_delta_audit.md",
        ],
        "source_of_truth": [
            "reports/code_truth_contract.md",
            "reports/final_truth_matrix.md",
        ],
        "notes": "The Triton kernel is explicitly experimental; correctness is partially covered, but it is not yet a release-grade performance claim surface.",
    },
    {
        "path": "mertformer_sdk/kernels/cpp/bitnet_cpu.cpp",
        "lane": "cpu-reference",
        "surface_class": "maintained",
        "maturity": "reference_safe",
        "canonical_command": [
            "python3 -m pytest -q tests/test_cpp_kernel_loader.py"
        ],
        "verification": [
            "tests/test_cpp_kernel_loader.py",
        ],
        "artifact": [
            "reports/code_truth_delta_audit.md",
        ],
        "source_of_truth": [
            "reports/code_truth_contract.md",
            "reports/surface_lifecycle_matrix.md",
        ],
        "notes": "This file is a minimal CPU reference kernel and should remain a parity/debug surface, not a production-depth speed claim.",
    },
    {
        "path": "mertformer_sdk/kernels/metal/engine.py",
        "lane": "mps-metal",
        "surface_class": "living",
        "maturity": "tested_fallback",
        "canonical_command": [
            "python3 -m pytest -q tests/test_kernel_dispatcher.py"
        ],
        "verification": [
            "tests/test_kernel_dispatcher.py",
        ],
        "artifact": [
            "reports/code_truth_delta_audit.md",
        ],
        "source_of_truth": [
            "reports/code_truth_contract.md",
            "reports/surface_lifecycle_matrix.md",
        ],
        "notes": "Metal currently routes through deterministic PyTorch fallback math and must not be narrated as a custom optimized kernel path.",
    },
    {
        "path": "scripts/chess_5080_onefile.py",
        "lane": "chess-proof",
        "surface_class": "living",
        "maturity": "tested_fallback",
        "canonical_command": [
            "python3 -m pytest -q tests/test_chess_5080_onefile.py tests/test_export_chess_5080_share.py"
        ],
        "verification": [
            "tests/test_chess_5080_onefile.py",
            "tests/test_export_chess_5080_share.py",
        ],
        "artifact": [
            "reports/code_truth_delta_audit.md",
            "CHESS_5080_POC_INTERNAL_TR.md",
        ],
        "source_of_truth": [
            "CHESS_5080_POC_INTERNAL_TR.md",
            "reports/code_truth_contract.md",
        ],
        "notes": "The chess onefile is a real code path with tests and delivery helpers, but it remains a proof/product baseline rather than a solved final product lane.",
    },
    {
        "path": "scripts/export_chess_5080_share.py",
        "lane": "chess-proof",
        "surface_class": "maintained",
        "maturity": "tested_fallback",
        "canonical_command": [
            "python3 -m pytest -q tests/test_export_chess_5080_share.py"
        ],
        "verification": [
            "tests/test_export_chess_5080_share.py",
        ],
        "artifact": [
            "reports/code_truth_delta_audit.md",
        ],
        "source_of_truth": [
            "CHESS_5080_POC_INTERNAL_TR.md",
            "reports/code_truth_contract.md",
        ],
        "notes": "Delivery/export logic is implemented and tested, but external product-grade distribution still depends on trained outputs and operator validation.",
    },
]

DOC_CLAIM_CROSSWALK = [
    {
        "doc": "README.md",
        "claim": "Canonical verification and train-end entrypoints are real code paths.",
        "code_path": [
            "scripts/verify_all.sh",
            "zero_touch_start.sh",
            "scripts/final_orchestrator.py",
        ],
        "canonical_command": [
            "bash scripts/verify_all.sh",
            "bash zero_touch_start.sh --check-only",
        ],
        "verification": [
            "reports/train_readiness_decision.json",
            "reports/start_gate_report.json",
        ],
        "artifact": [
            "reports/final_truth_matrix.md",
            "reports/canonical_entrypoint.md",
        ],
    },
    {
        "doc": "MODEL_CARD.md",
        "claim": "Measured versus target boundaries remain explicit.",
        "code_path": [
            "scripts/check_doc_claim_consistency.py",
            "scripts/build_closure_governance_pack.py",
        ],
        "canonical_command": [
            "python3 scripts/check_doc_claim_consistency.py",
            "python3 scripts/build_closure_governance_pack.py",
        ],
        "verification": [
            "reports/claim_registry.json",
            "reports/final_truth_matrix.md",
        ],
        "artifact": [
            "reports/code_truth_delta_audit.md",
        ],
    },
    {
        "doc": "reports/final_truth_matrix.md",
        "claim": "Closure-critical claims map to evidence instead of living only as prose.",
        "code_path": [
            "scripts/build_closure_governance_pack.py",
            "scripts/build_code_truth_audit.py",
        ],
        "canonical_command": [
            "python3 scripts/build_closure_governance_pack.py",
            "python3 scripts/build_code_truth_audit.py",
        ],
        "verification": [
            "reports/claim_registry.json",
            "reports/code_truth_delta_audit.json",
        ],
        "artifact": [
            "reports/final_truth_matrix.md",
            "reports/code_truth_delta_audit.md",
        ],
    },
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / path


def file_exists(path: str) -> bool:
    return resolve_repo_path(path).exists()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def marker_hits_for_text(text: str) -> list[str]:
    hits: list[str] = []
    for name, pattern in MARKER_PATTERNS.items():
        if pattern.search(text):
            hits.append(name)
    return hits


def iter_scan_paths() -> list[Path]:
    paths: list[Path] = []
    for directory in SCAN_DIRS:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
                continue
            if path.suffix.lower() not in SCAN_EXTENSIONS:
                continue
            if rel(path) in EXCLUDED_SCAN_PATHS:
                continue
            paths.append(path)
    for path in SCAN_FILES:
        if path.exists():
            if rel(path) in EXCLUDED_SCAN_PATHS:
                continue
            paths.append(path)
    dedup: dict[str, Path] = {}
    for path in paths:
        dedup[str(path.resolve())] = path
    return sorted(dedup.values(), key=lambda item: rel(item) if item.is_relative_to(ROOT) else str(item))


def scan_marker_hits(limit: int = 80) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in iter_scan_paths():
        text = read_text(path)
        hits = marker_hits_for_text(text)
        if not hits:
            continue
        entry = {
            "path": rel(path),
            "markers": hits,
            "summary": ", ".join(hits),
        }
        results.append(entry)
        if len(results) >= limit:
            break
    return results


def build_surface_record(entry: dict[str, Any]) -> dict[str, Any]:
    path = entry["path"]
    resolved = resolve_repo_path(path)
    marker_hits = marker_hits_for_text(read_text(resolved)) if resolved.exists() else []
    evidence = {
        "code_path": [path],
        "canonical_command": list(entry["canonical_command"]),
        "verification": list(entry["verification"]),
        "artifact": list(entry["artifact"]),
    }
    return {
        "path": path,
        "exists": resolved.exists(),
        "lane": entry["lane"],
        "surface_class": entry["surface_class"],
        "maturity": entry["maturity"],
        "markers": marker_hits,
        "evidence": evidence,
        "evidence_complete": all(bool(evidence[column]) for column in EVIDENCE_COLUMNS),
        "source_of_truth": list(entry["source_of_truth"]),
        "notes": entry["notes"],
    }


def build_payload(marker_limit: int = 80) -> dict[str, Any]:
    surfaces = [build_surface_record(entry) for entry in TECHNICAL_SURFACES]
    maturity_counts: dict[str, int] = {}
    for item in surfaces:
        maturity = item["maturity"]
        maturity_counts[maturity] = maturity_counts.get(maturity, 0) + 1

    return {
        "schema": "code_truth_delta_audit_v1",
        "generated_utc": utc_now(),
        "evidence_contract": {
            "required_columns": list(EVIDENCE_COLUMNS),
            "done_rule": "A closure-critical item is only done when code path, canonical command, verification, and artifact/report evidence all exist together.",
        },
        "surface_groups": SURFACE_GROUPS,
        "technical_surfaces": surfaces,
        "doc_claim_crosswalk": DOC_CLAIM_CROSSWALK,
        "marker_scan": scan_marker_hits(limit=marker_limit),
        "summary": {
            "technical_surface_count": len(surfaces),
            "maturity_counts": maturity_counts,
            "marker_hit_count": len(scan_marker_hits(limit=marker_limit)),
        },
    }


def markdown_table_paths(paths: list[str]) -> str:
    if not paths:
        return "- none"
    return "\n".join(f"- `{path}`" for path in paths)


def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Code-Truth Delta Audit",
        "",
        "Current repo-side code-truth audit for the closure pass. This report does not upgrade any claim by rhetoric; it labels maturity, evidence shape, and marker risk explicitly.",
        "",
        "## Done Rule",
        payload["evidence_contract"]["done_rule"],
        "",
        "Required evidence columns:",
        *(f"- `{column}`" for column in payload["evidence_contract"]["required_columns"]),
        "",
        "## Surface Groups",
    ]
    for group_name, paths in payload["surface_groups"].items():
        lines.append(f"### {group_name}")
        lines.append(markdown_table_paths(paths))
        lines.append("")

    lines.extend(
        [
            "## Technical Surface Maturity",
            "",
            "| Path | Lane | Surface Class | Maturity | Markers | Evidence Complete | Notes |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in payload["technical_surfaces"]:
        markers = ", ".join(f"`{marker}`" for marker in item["markers"]) if item["markers"] else "none"
        lines.append(
            "| {path} | `{lane}` | `{surface_class}` | `{maturity}` | {markers} | `{complete}` | {notes} |".format(
                path=f"`{item['path']}`",
                lane=item["lane"],
                surface_class=item["surface_class"],
                maturity=item["maturity"],
                markers=markers,
                complete=str(item["evidence_complete"]).lower(),
                notes=item["notes"],
            )
        )

    lines.extend(
        [
            "",
            "## Doc-to-Code Crosswalk",
            "",
            "| Doc | Claim | Code Path | Canonical Command | Verification | Artifact |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in payload["doc_claim_crosswalk"]:
        lines.append(
            "| `{doc}` | {claim} | {code_path} | {command} | {verification} | {artifact} |".format(
                doc=item["doc"],
                claim=item["claim"],
                code_path=", ".join(f"`{value}`" for value in item["code_path"]),
                command=", ".join(f"`{value}`" for value in item["canonical_command"]),
                verification=", ".join(f"`{value}`" for value in item["verification"]),
                artifact=", ".join(f"`{value}`" for value in item["artifact"]),
            )
        )

    lines.extend(
        [
            "",
            "## Marker Scan",
            "",
            "Marker hits are review prompts, not automatic bug declarations. In particular, dataset compliance tables may intentionally retain `TBD` placeholders until legal/compliance review finishes.",
            "",
            "| Path | Markers |",
            "| --- | --- |",
        ]
    )
    for hit in payload["marker_scan"]:
        lines.append(
            f"| `{hit['path']}` | {', '.join(f'`{marker}`' for marker in hit['markers'])} |"
        )
    if not payload["marker_scan"]:
        lines.append("| none | none |")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the repo code-truth delta audit.")
    parser.add_argument("--out-json", default="reports/code_truth_delta_audit.json")
    parser.add_argument("--out-md", default="reports/code_truth_delta_audit.md")
    parser.add_argument("--marker-limit", type=int, default=80)
    args = parser.parse_args()

    payload = build_payload(marker_limit=args.marker_limit)
    out_json = ROOT / args.out_json
    out_md = ROOT / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text(build_markdown(payload).rstrip() + "\n", encoding="utf-8")

    print("OK: code-truth delta audit refreshed")
    print(f" - {rel(out_md)}")
    print(f" - {rel(out_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
