"""Repo secret scanner.

Scans *tracked* files for common secret patterns and exits non-zero if any are found.
When git metadata is unavailable, scans a bounded package file walk instead. This
keeps transferred/package-only copies gateable without weakening the secret check.
The scanner must not print secret values.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


PATTERNS: dict[str, re.Pattern[str]] = {
    # Hugging Face user tokens start with `hf_` and are long alnum strings.
    # Use a higher minimum length to avoid false positives like `hf_candidates`.
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{24,}\b"),
    "wandb_token": re.compile(r"\bwandb_[A-Za-z0-9]{10,}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9]{10,}\b"),
    # Covers PAT (ghp_), OAuth (gho_), user-to-server (ghu_), server-to-server (ghs_), refresh (ghr_).
    "github_token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"),
    "github_fine_grained_pat": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{60,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # Common 40-hex API keys (heuristic) but only when the line contains key-like context.
    # Avoid false positives from git SHAs or checksums.
    "hex40_context": re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password)\b\s*[:=]\s*['\"]?[0-9a-f]{40}['\"]?\b"),
    # W&B API keys are 40-hex; catch explicit env-style assignments.
    "wandb_api_key": re.compile(r"(?i)\bWANDB_API_KEY\b\s*[:=]\s*['\"]?[0-9a-f]{40}['\"]?\b"),
}

SKIP_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".zip",
    ".pdf",
    ".pptx",
    ".docx",
    ".onnx",
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".mp4",
    ".mov",
    ".tar",
    ".tgz",
    ".gz",
    ".zst",
}

SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".titan-venv",
    ".lint-venv",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "site-packages",
    "checkpoints",
    "results",
}

DEFAULT_MAX_SCAN_BYTES = 2_000_000


def _max_scan_bytes() -> int:
    raw = os.environ.get("MERTFORMER_SECRET_SCAN_MAX_BYTES", str(DEFAULT_MAX_SCAN_BYTES))
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_SCAN_BYTES


def _should_skip_path(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    if any(part in SKIP_DIR_NAMES or part.endswith((".dist-info", ".egg-info")) for part in rel.parts):
        return True
    if path.suffix.lower() in SKIP_EXTS:
        return True
    try:
        if path.stat().st_size > _max_scan_bytes():
            return True
    except OSError:
        return True
    return False


def _git_tracked_files(root: Path = ROOT) -> list[Path]:
    out = subprocess.check_output(["git", "ls-files"], cwd=root, stderr=subprocess.DEVNULL, text=True)
    paths = []
    for line in out.splitlines():
        p = root / line
        if _should_skip_path(p, root):
            continue
        paths.append(p)
    return paths


def _package_files(root: Path = ROOT) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _should_skip_path(path, root):
            continue
        paths.append(path)
    return sorted(paths)


def discover_scan_files(root: Path = ROOT, *, package_mode: bool = False) -> tuple[str, list[Path]]:
    if not package_mode:
        try:
            files = _git_tracked_files(root)
            return "git_tracked", files
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return "package_walk", _package_files(root)


def _redact_line(line: str) -> str:
    redacted = line
    for rx in PATTERNS.values():
        redacted = rx.sub("<REDACTED>", redacted)
    return redacted


def _display_path(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def scan_paths(paths: list[Path], root: Path = ROOT) -> list[tuple[str, Path, int, str]]:
    hits: list[tuple[str, Path, int, str]] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for idx, line in enumerate(text.splitlines(), start=1):
            for name, rx in PATTERNS.items():
                if rx.search(line):
                    hits.append((name, _display_path(path, root), idx, _redact_line(line).strip()))
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan repository or package files for secret-like patterns.")
    parser.add_argument("--root", default=str(ROOT), help="Repository/package root to scan.")
    parser.add_argument("--package-mode", action="store_true", help="Force package file-walk mode instead of git ls-files.")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    mode, paths = discover_scan_files(root, package_mode=args.package_mode)
    hits = scan_paths(paths, root)

    if not hits:
        print(f"OK: no secret patterns detected in {mode} files. scanned={len(paths)}")
        return 0

    print(f"ERROR: potential secrets detected in {mode} files:")
    # Do not print raw matches.
    for name, path, idx, safe in hits[:200]:
        print(f"- {name}: {path}:{idx} :: {safe}")
    if len(hits) > 200:
        print(f"... and {len(hits) - 200} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
