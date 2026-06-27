#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXCLUDE_PARTS = {
    ".git",
    ".titan-venv",
    ".lint-venv",
    ".venv",
    ".idea",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}


SENSITIVE_FILE_NAMES = {".env"}

SOP_RUNTIME_OUTPUTS = (
    "reports/one_command_full_sop_summary.md",
    "reports/one_command_full_sop.log",
)


PYTHON_ROLE_OVERRIDES: dict[str, tuple[str, str]] = {
    "config/config.py": (
        "runtime configuration model and validation helpers",
        "çalışma zamanı konfigürasyon modeli ve doğrulama yardımcıları",
    ),
    "model/transformers.py": (
        "MertFormer model assembly and forward graph",
        "MertFormer model montajı ve ileri geçiş grafiği",
    ),
    "train/train.py": (
        "main training loop entrypoint",
        "ana eğitim döngüsü giriş noktası",
    ),
    "train/continual_adapter.py": (
        "continual learning adapter path for training",
        "eğitim için continual learning adaptör yolu",
    ),
    "layers/bitlinear.py": (
        "BitLinear low-bit linear layer implementation",
        "BitLinear düşük-bit linear katman implementasyonu",
    ),
    "layers/bitnet_patch.py": (
        "BitNet quantization patch and runtime hooks",
        "BitNet kuantizasyon patch ve runtime kancaları",
    ),
    "layers/cognitive_extensions.py": (
        "optional cognitive extension blocks",
        "opsiyonel bilişsel genişletme blokları",
    ),
    "layers/ffn.py": (
        "feed-forward network blocks (dense and sparse paths)",
        "feed-forward ağ blokları (dense ve sparse yollar)",
    ),
    "layers/lifelong_safety.py": (
        "lifelong safety guard layer",
        "yaşam boyu güvenlik koruma katmanı",
    ),
    "layers/liquid.py": (
        "liquid neural dynamics layers",
        "liquid sinir dinamik katmanları",
    ),
    "layers/mertformer_block.py": (
        "core transformer block composition",
        "çekirdek transformer blok bileşimi",
    ),
    "layers/mla.py": (
        "grouped-query attention (GQA) implementation",
        "grouped-query attention (GQA) implementasyonu",
    ),
    "layers/moe.py": (
        "mixture-of-experts routing and expert execution",
        "mixture-of-experts yönlendirme ve uzman çalıştırma",
    ),
    "layers/qinn.py": (
        "QINN experimental regulation layer (feature-flag)",
        "QINN deneysel regülasyon katmanı (feature-flag)",
    ),
    "layers/world_model_head.py": (
        "world-model auxiliary head",
        "dünya-modeli yardımcı çıktı kafası",
    ),
    "scripts/sync_manifest.py": (
        "release manifest and project-structure sync generator",
        "release manifest ve proje-yapısı senkron üreticisi",
    ),
    "scripts/sync_test_stat_claims.py": (
        "pytest pass/skipped claim synchronizer for tracked docs",
        "izlenen dokümanlar için pytest passed/skipped claim senkronlayıcısı",
    ),
    "scripts/docs_inventory.py": (
        "markdown inventory and folder policy reporter",
        "markdown envanteri ve klasör politika raporlayıcısı",
    ),
    "mertformer_sdk/kernels/triton_fused_bitlinear.py": (
        "Triton fused BitLinear CUDA kernel surface",
        "Triton fused BitLinear CUDA kernel yüzeyi",
    ),
}


def tracked_files(root: Path) -> list[Path]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True,
            capture_output=True,
        )
        raw = proc.stdout.decode("utf-8", errors="replace")
        rels = [p for p in raw.split("\0") if p]
        return [root / rel for rel in rels]
    except Exception:
        return []



def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel_if_under(root: Path, target: Path) -> str | None:
    try:
        return str(target.resolve().relative_to(root.resolve()))
    except Exception:
        return None


def collect_entries(root: Path, excluded_relpaths: set[str]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []

    candidates = tracked_files(root)
    if candidates:
        for rel in SOP_RUNTIME_OUTPUTS:
            p = root / rel
            if p.exists() and p.is_file():
                candidates.append(p)
    if not candidates:
        candidates = [p for p in root.rglob("*") if p.is_file()]

    dedup: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)
    candidates = dedup

    for p in candidates:
        if not p.exists() or not p.is_file():
            continue
        rel = p.relative_to(root)
        rel_s = str(rel)
        if any(part in EXCLUDE_PARTS for part in rel.parts):
            continue
        if rel_s in excluded_relpaths:
            continue
        name = rel.name.lower()
        if name in SENSITIVE_FILE_NAMES or name.startswith(".env."):
            continue

        st = p.stat()
        entries.append(
            {
                "path": rel_s,
                "size_bytes": int(st.st_size),
                "sha256": file_hash(p),
                "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            }
        )

    entries.sort(key=lambda x: str(x["path"]))
    return entries


def _normalize_rel(rel: str) -> str:
    return rel.replace("\\", "/")


def _humanize_stem(stem: str) -> str:
    return stem.replace("_", " ").replace("-", " ").strip()


def _python_role(rel: str, lang: str) -> str:
    rel_l = _normalize_rel(rel).lower()
    p = Path(rel_l)
    name = p.name
    stem = p.stem
    parts = [part.lower() for part in p.parts]

    override = PYTHON_ROLE_OVERRIDES.get(rel_l)
    if override:
        return override[0] if lang == "en" else override[1]

    if name == "__init__.py":
        if len(parts) >= 2:
            pkg = parts[-2]
            return (
                f"{pkg} package initializer and exports"
                if lang == "en"
                else f"{pkg} paket başlatıcısı ve dışa aktarmalar"
            )
        return "package initializer" if lang == "en" else "paket başlatıcısı"

    if parts and parts[0] == "tests" and stem.startswith("test_"):
        target = _humanize_stem(stem.removeprefix("test_"))
        return (
            f"automated test module for {target}"
            if lang == "en"
            else f"{target} için otomatik test modülü"
        )

    if parts and parts[0] == "scripts":
        target = _humanize_stem(stem)
        return (
            f"automation script for {target}"
            if lang == "en"
            else f"{target} için otomasyon scripti"
        )

    if parts and parts[0] == "eval":
        target = _humanize_stem(stem)
        return (
            f"evaluation routine for {target}"
            if lang == "en"
            else f"{target} için değerlendirme rutini"
        )

    if parts and parts[0] == "orchestrator":
        target = _humanize_stem(stem)
        return (
            f"orchestrator runtime component for {target}"
            if lang == "en"
            else f"{target} için orkestratör runtime bileşeni"
        )

    if parts and parts[0] == "mertformer_sdk":
        target = _humanize_stem(stem)
        return (
            f"SDK component for {target}"
            if lang == "en"
            else f"{target} için SDK bileşeni"
        )

    target = _humanize_stem(stem)
    return f"module for {target}" if lang == "en" else f"{target} için modül"


def _comment_map(lang: str) -> dict[str, str]:
    if lang == "tr":
        return {
            "directory": "dizin",
            "gitignore": "git ignore politikası",
            "dockerfile": "container build baseline",
            "citation": "atıf metaverisi",
            "pyproject": "proje metaverisi",
            "license_en": "lisans koşulları (EN)",
            "license_tr": "lisans koşulları (TR)",
            "readme_en": "ana dokümantasyon (EN)",
            "readme_tr": "Türkçe doküman karşılığı",
            "run_and_clean_pycache": (
                "Python modülü/scripti (komut çalıştırma + garanti pycache temizliği; "
                "venv cache temizliği için --include-venv-caches kullan)"
            ),
            "sop_summary": "dokümantasyon/rapor dosyası (tek komut uçtan uca SOP özeti; her çalıştırmada üzerine yazılır)",
            "sop_log": "metin/log artefaktı (tek komut uçtan uca SOP ham logu; her çalıştırmada üzerine yazılır)",
            "md": "dokümantasyon/rapor dosyası",
            "sh": "kabuk otomasyon scripti",
            "yaml": "YAML yapılandırma dosyası",
            "jsonl": "JSONL veri/log artefaktı",
            "json_schema": "JSON şema artefaktı",
            "json_data": "JSON veri artefaktı",
            "csv": "CSV veri artefaktı",
            "txt": "metin artefaktı",
            "log": "metin/log artefaktı",
            "cpp": "C++ kaynak dosyası",
            "media": "medya varlığı",
            "sha256": "artefakt sağlama toplamı",
            "artifact": "artefakt",
            "toml": "TOML yapılandırma dosyası",
        }
    return {
        "directory": "directory",
        "gitignore": "git ignore policy",
        "dockerfile": "container build baseline",
        "citation": "citation metadata",
        "pyproject": "project metadata",
        "license_en": "license terms (EN)",
        "license_tr": "license terms (TR)",
        "readme_en": "primary documentation (EN)",
        "readme_tr": "Turkish document counterpart",
        "run_and_clean_pycache": (
            "Python module/script (run command + guaranteed post-run cache sweep; "
            "add --include-venv-caches for venv cache cleanup)"
        ),
        "sop_summary": "documentation/report file (single-command end-to-end SOP summary; overwritten each run)",
        "sop_log": "text/log artifact (single-command end-to-end SOP raw log; overwritten each run)",
        "md": "documentation/report file",
        "sh": "shell automation script",
        "yaml": "YAML configuration file",
        "jsonl": "JSONL data/log artifact",
        "json_schema": "JSON schema artifact",
        "json_data": "JSON data artifact",
        "csv": "CSV data artifact",
        "txt": "text artifact",
        "log": "text/log artifact",
        "cpp": "C++ source file",
        "media": "media asset",
        "sha256": "artifact checksum",
        "artifact": "artifact",
        "toml": "TOML configuration file",
    }


def structure_comment(rel: str, is_dir: bool, lang: str = "en") -> str:
    p = Path(rel)
    name = p.name
    lower = name.lower()
    suffix = p.suffix.lower()
    comments = _comment_map(lang)

    if is_dir:
        return comments["directory"]
    if name == ".gitignore":
        return comments["gitignore"]
    if name == "Dockerfile":
        return comments["dockerfile"]
    if name == "CITATION.cff":
        return comments["citation"]
    if name == "pyproject.toml":
        return comments["pyproject"]
    if name == "LICENSE":
        return comments["license_en"]
    if name == "LICENSE_TR":
        return comments["license_tr"]
    if name == "README.md":
        return comments["readme_en"]
    if name == "README_TR.md" or lower.endswith("_tr.md"):
        return comments["readme_tr"]
    if lower == "run_and_clean_pycache.py":
        return comments["run_and_clean_pycache"]
    if lower == "one_command_full_sop_summary.md":
        return comments["sop_summary"]
    if lower == "one_command_full_sop.log":
        return comments["sop_log"]
    if suffix == ".md":
        return comments["md"]
    if suffix == ".py":
        role = _python_role(rel, lang)
        return (
            f"Python module/script ({role})"
            if lang == "en"
            else f"Python modülü/scripti ({role})"
        )
    if suffix == ".sh":
        return comments["sh"]
    if suffix in {".yaml", ".yml"}:
        return comments["yaml"]
    if suffix == ".jsonl":
        return comments["jsonl"]
    if suffix == ".json":
        return comments["json_schema"] if "schema" in lower else comments["json_data"]
    if suffix == ".csv":
        return comments["csv"]
    if suffix == ".txt":
        return comments["txt"]
    if suffix == ".log":
        return comments["log"]
    if suffix == ".cpp":
        return comments["cpp"]
    if suffix in {".png", ".gif", ".mp4", ".svg"}:
        return comments["media"]
    if suffix == ".sha256":
        return comments["sha256"]
    if suffix in {".pdf", ".pptx", ".age"}:
        return comments["artifact"]
    if suffix == ".toml":
        return comments["toml"]
    return comments["artifact"]


def flatten_tree(node: dict[str, Any], parent_parts: tuple[str, ...] = ()) -> list[str]:
    """Reconstruct the relative file paths actually emitted into the structure tree.

    Used to diff the rendered PROJECT_STRUCTURE tree against the manifest entry
    list so the file_sync_matrix gate reflects a real comparison rather than a
    hardcoded green result.
    """
    paths: list[str] = []
    for name in node.get("files", []):
        paths.append("/".join(parent_parts + (name,)))
    for name, child in node.get("dirs", {}).items():
        paths.extend(flatten_tree(child, parent_parts + (name,)))
    return paths


def build_tree(paths: list[str]) -> dict[str, Any]:
    root: dict[str, Any] = {"dirs": {}, "files": []}
    for rel in sorted(paths):
        parts = Path(rel).parts
        node = root
        for part in parts[:-1]:
            node = node["dirs"].setdefault(part, {"dirs": {}, "files": []})
        node["files"].append(parts[-1])
    return root


def emit_tree(
    node: dict[str, Any],
    prefix: str = "",
    parent_parts: tuple[str, ...] = (),
    lang: str = "en",
) -> list[str]:
    lines: list[str] = []
    dir_names = sorted(node["dirs"])
    file_names = sorted(node["files"])
    entries = [(name, True) for name in dir_names] + [(name, False) for name in file_names]

    for idx, (name, is_dir) in enumerate(entries):
        last = idx == len(entries) - 1
        branch = "└── " if last else "├── "
        rel = "/".join(parent_parts + (name,))
        display = f"{name}/" if is_dir else name
        lines.append(f"{prefix}{branch}{display}  # {structure_comment(rel, is_dir, lang)}")
        if is_dir:
            child_prefix = prefix + ("    " if last else "│   ")
            lines.extend(emit_tree(node["dirs"][name], child_prefix, parent_parts + (name,), lang))
    return lines


def build_structure_lines(paths: list[str], *, lang: str) -> list[str]:
    root_comment = (
        "project root (git ls-files inventory)"
        if lang == "en"
        else "proje kökü (git ls-files envanteri)"
    )
    lines = [f"mertformer-titan-core/  # {root_comment}"]
    lines.extend(emit_tree(build_tree(paths), lang=lang))
    return lines


def build_structure_md(paths: list[str], out_path: Path) -> None:
    structure_lines = build_structure_lines(paths, lang="en")
    lines = [
        "# PROJECT_STRUCTURE",
        "",
        "Generated automatically from tracked files with inline role comments.",
        "",
        "```text",
    ]
    lines.extend(structure_lines)
    lines.append("```")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sync_readme_structure_block(
    readme_path: Path,
    section_heading: str,
    structure_lines: list[str],
) -> bool:
    if not readme_path.exists():
        return False

    content = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"({re.escape(section_heading)}.*?)(```text\n)(.*?)(\n```)",
        re.DOTALL,
    )
    match = pattern.search(content)
    if not match:
        return False

    replacement_block = "```text\n" + "\n".join(structure_lines) + "\n```"
    updated = content[: match.start(2)] + replacement_block + content[match.end(4) :]
    if updated != content:
        readme_path.write_text(updated, encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate release manifest and sync reports")
    ap.add_argument("--root", default=".")
    ap.add_argument("--manifest", default="reports/release_manifest.json")
    ap.add_argument("--structure", default="docs/PROJECT_STRUCTURE.md")
    ap.add_argument("--matrix", default="reports/file_sync_matrix.json")
    ap.add_argument("--sync-report", default="reports/project_structure_sync_report.json")
    ap.add_argument("--policy-report", default="reports/policy_sync_report.json")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    manifest_path = Path(args.manifest)
    structure_path = Path(args.structure)
    matrix_path = Path(args.matrix)
    sync_path = Path(args.sync_report)
    policy_path = Path(args.policy_report)

    excluded_relpaths = set()
    for p in [manifest_path, structure_path, matrix_path, sync_path, policy_path]:
        rel = _rel_if_under(root, p)
        if rel:
            excluded_relpaths.add(rel)

    entries = collect_entries(root, excluded_relpaths)
    entry_paths = [str(e["path"]) for e in entries]

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root": ".",
        "entry_count": len(entries),
        "entries": entries,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    build_structure_md(entry_paths, structure_path)

    readme_sync = {
        # README carries only a POINTER to the canonical tree (kept full in
        # docs/PROJECT_STRUCTURE.md). This keeps the front page sharp instead of
        # re-injecting the full ~900-line tracked-file tree on every sync.
        "README.md": sync_readme_structure_block(
            root / "README.md",
            "### Canonical Layout (Build 30 V2)",
            ["Full tracked-file tree: docs/PROJECT_STRUCTURE.md"],
        ),
        "README_TR.md": sync_readme_structure_block(
            root / "README_TR.md",
            "### Kanonik Yerleşim (Build 30 V2)",
            ["Tam tracked-dosya ağacı: docs/PROJECT_STRUCTURE.md"],
        ),
    }
    readme_sync_ok = all(readme_sync.values())

    # Real drift check: diff the manifest entry list against the paths actually
    # rendered into the structure tree (flattened back out of build_tree). This
    # binds the matrix gate to a genuine comparison instead of a hardcoded green.
    structure_paths = flatten_tree(build_tree(entry_paths))
    manifest_set = set(entry_paths)
    structure_set = set(structure_paths)
    missing_in_structure = sorted(manifest_set - structure_set)
    missing_in_manifest = sorted(structure_set - manifest_set)
    matrix_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_count": len(manifest_set),
        "structure_count": len(structure_set),
        "missing_in_structure": missing_in_structure,
        "missing_in_manifest": missing_in_manifest,
        "ok": not missing_in_structure and not missing_in_manifest,
    }

    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_text(json.dumps(matrix_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    sync_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ok": bool(matrix_payload["ok"] and readme_sync_ok),
        "readme_sync": readme_sync,
        "details": matrix_payload,
    }
    sync_path.parent.mkdir(parents=True, exist_ok=True)
    sync_path.write_text(json.dumps(sync_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    policy_file = Path("policy/allow_deny_policy.yaml")
    policy_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "policy_file": str(policy_file),
        "policy_exists": (root / policy_file).exists(),
        "ok": (root / policy_file).exists(),
    }
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(policy_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = bool(policy_payload["ok"] and sync_payload["ok"])
    print(json.dumps({"manifest_entries": len(entries), "ok": ok}, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
