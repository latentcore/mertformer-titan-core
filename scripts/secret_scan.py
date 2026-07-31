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
    # Anthropic keys: `sk-ant-...` / service-account `sk-svcac-...`. The generic openai_key
    # pattern above does NOT catch these — its tail class `[A-Za-z0-9]` excludes the hyphen
    # inside "ant-"/"svcac-", so the match breaks before accumulating 10 chars. Confirmed gap
    # (2026-07-13 home-dir secret scan); fixed here with a tail class that allows `-`/`_`.
    "anthropic_key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    "anthropic_service_key": re.compile(r"\bsk-svcac-[A-Za-z0-9_-]{20,}\b"),
    # Covers PAT (ghp_), OAuth (gho_), user-to-server (ghu_), server-to-server (ghs_), refresh (ghr_).
    "github_token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"),
    "github_fine_grained_pat": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{60,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # Google API key (e.g. Maps/Cloud): fixed `AIza` prefix + 35 alnum/`_`/`-` chars.
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    # Google OAuth 2.0 client secret: fixed `GOCSPX-` prefix.
    "google_oauth_client_secret": re.compile(r"\bGOCSPX-[A-Za-z0-9_-]{20,}\b"),
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

# --------------------------------------------------------------------------
# Optional home-directory scan (opt-in via --include-home-dirs; NOT part of
# the default git_tracked/package_walk scan verify_all.sh runs on every pass).
# Personal secret exports (WhatsApp/AI-chat exports, downloaded credential
# files) tend to live in ~/Documents and ~/Downloads, outside any git tree
# this scanner otherwise ever sees. Path.home() resolves correctly on
# macOS/Linux/Windows, so no OS-specific branching is needed here.
# --------------------------------------------------------------------------
HOME_SCAN_DIR_NAMES = ("Documents", "Downloads")


def _home_scan_roots(home: Path | None = None) -> list[Path]:
    base = home if home is not None else Path.home()
    roots: list[Path] = []
    for name in HOME_SCAN_DIR_NAMES:
        candidate = base / name
        if candidate.is_dir():
            roots.append(candidate)
    return roots


def _walk_files_safe(root: Path) -> list[Path]:
    """os.walk-based (not Path.rglob): a single permission-denied subdirectory
    under an arbitrary home folder must not abort the whole scan -- the same
    failure class scripts/build_scoped_external_intake_matrix.py's
    safe_sha256_file() wrapper was built to survive. onerror=no-op means one
    unreadable directory is skipped, not fatal."""
    paths: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(str(root), onerror=lambda _exc: None):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES and not d.startswith(".")]
        for fname in filenames:
            p = Path(dirpath) / fname
            if _should_skip_path(p, root):
                continue
            paths.append(p)
    return paths


def discover_home_directory_files(home: Path | None = None) -> list[Path]:
    paths: list[Path] = []
    for root in _home_scan_roots(home):
        paths.extend(_walk_files_safe(root))
    return sorted(paths)


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
    out = subprocess.check_output(["git", "ls-files"], cwd=root, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace")
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
    parser.add_argument(
        "--include-home-dirs",
        action="store_true",
        help="Also scan ~/Documents and ~/Downloads for secret-like patterns (opt-in; "
        "NOT run by verify_all.sh -- for manual personal-export secret sweeps, e.g. "
        "leaked tokens in AI-chat/WhatsApp exports saved outside the repo).",
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    mode, paths = discover_scan_files(root, package_mode=args.package_mode)
    hits = scan_paths(paths, root)

    if not hits:
        print(f"OK: no secret patterns detected in {mode} files. scanned={len(paths)}")
    else:
        print(f"ERROR: potential secrets detected in {mode} files:")
        # Do not print raw matches.
        for name, path, idx, safe in hits[:200]:
            print(f"- {name}: {path}:{idx} :: {safe}")
        if len(hits) > 200:
            print(f"... and {len(hits) - 200} more")

    home_hits: list[tuple[str, Path, int, str]] = []
    if args.include_home_dirs:
        home_paths = discover_home_directory_files()
        home_hits = scan_paths(home_paths, root)
        if not home_hits:
            print(
                "OK: no secret patterns detected in home-directory files "
                f"(~/Documents, ~/Downloads). scanned={len(home_paths)}"
            )
        else:
            print("ERROR: potential secrets detected in home-directory files (~/Documents, ~/Downloads):")
            for name, path, idx, safe in home_hits[:200]:
                print(f"- {name}: {path}:{idx} :: {safe}")
            if len(home_hits) > 200:
                print(f"... and {len(home_hits) - 200} more")

    return 1 if (hits or home_hits) else 0


if __name__ == "__main__":
    raise SystemExit(main())
