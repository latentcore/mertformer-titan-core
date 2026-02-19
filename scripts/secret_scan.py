"""Repo secret scanner (best-effort).

Scans *tracked* files for common secret patterns and exits non-zero if any are found.
This is designed to run in CI and must not print secret values.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


PATTERNS: dict[str, re.Pattern[str]] = {
    # Hugging Face user tokens start with `hf_` and are long alnum strings.
    # Use a higher minimum length to avoid false positives like `hf_candidates`.
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{24,}\b"),
    "wandb_token": re.compile(r"\bwandb_[A-Za-z0-9]{10,}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9]{10,}\b"),
    "github_token": re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # Common 40-hex API keys (heuristic) but only when the line contains key-like context.
    # Avoid false positives from git SHAs or checksums.
    "hex40_context": re.compile(r"(?i)\\b(?:api[_-]?key|secret|token|password)\\b\\s*[:=]\\s*['\\\"]?[0-9a-f]{40}['\\\"]?\\b"),
    # W&B API keys are 40-hex; catch explicit env-style assignments.
    "wandb_api_key": re.compile(r"(?i)\\bWANDB_API_KEY\\b\\s*[:=]\\s*['\\\"]?[0-9a-f]{40}['\\\"]?\\b"),
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
}


def _tracked_files() -> list[Path]:
    out = subprocess.check_output(["git", "ls-files"], text=True)
    paths = []
    for line in out.splitlines():
        p = Path(line)
        if p.suffix.lower() in SKIP_EXTS:
            continue
        paths.append(p)
    return paths


def _redact_line(line: str) -> str:
    redacted = line
    for rx in PATTERNS.values():
        redacted = rx.sub("<REDACTED>", redacted)
    return redacted


def main() -> int:
    hits: list[tuple[str, Path, int, str]] = []
    for path in _tracked_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for idx, line in enumerate(text.splitlines(), start=1):
            for name, rx in PATTERNS.items():
                if rx.search(line):
                    hits.append((name, path, idx, _redact_line(line).strip()))

    if not hits:
        print("OK: no secret patterns detected in tracked files.")
        return 0

    print("ERROR: potential secrets detected in tracked files:")
    # Do not print raw matches.
    for name, path, idx, safe in hits[:200]:
        print(f"- {name}: {path}:{idx} :: {safe}")
    if len(hits) > 200:
        print(f"... and {len(hits) - 200} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
