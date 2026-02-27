from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List


DEFAULT_ZIP = "packages/MertFormer_Titan_OnyxStorm_v1.0_B30_Release.zip"

DENY_PATH_PATTERNS = [
    re.compile(r"(^|/)__pycache__/"),
    re.compile(r"(^|/)\.pytest_cache/"),
    re.compile(r"(^|/)\.ruff_cache/"),
    re.compile(r"(^|/)\.mypy_cache/"),
    re.compile(r"(^|/)\.git/"),
    re.compile(r"(^|/)\.titan-venv/"),
    re.compile(r"(^|/)\.venv/"),
    re.compile(r"(^|/)venv/"),
    re.compile(r"(^|/)checkpoints/"),
    re.compile(r"(^|/)logs/"),
    re.compile(r"(^|/)data/"),
    re.compile(r"(^|/)datasets/stage"),
    re.compile(r"(^|/)datasets/logits/"),
    re.compile(r"(^|/)\.env$"),
    re.compile(r"(^|/)packages/"),
]

SECRET_PATTERNS: Dict[str, re.Pattern[str]] = {
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{24,}\b"),
    "wandb_token": re.compile(r"\bwandb_[A-Za-z0-9]{10,}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9]{10,}\b"),
    "github_token": re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}

TEXT_EXTS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".csv",
    ".jsonl",
    ".sh",
}



def _is_text_candidate(path: str) -> bool:
    p = Path(path)
    if p.suffix.lower() in TEXT_EXTS:
        return True
    if p.suffix == "":
        return True
    return False



def audit_zip(path: Path, max_text_bytes: int = 512_000) -> Dict[str, object]:
    deny_hits: List[str] = []
    secret_hits: List[Dict[str, object]] = []

    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        for name in names:
            for rx in DENY_PATH_PATTERNS:
                if rx.search(name):
                    deny_hits.append(name)
                    break

        for name in names:
            if not _is_text_candidate(name):
                continue
            try:
                info = zf.getinfo(name)
                if info.file_size > int(max_text_bytes):
                    continue
                data = zf.read(name)
                text = data.decode("utf-8", errors="ignore")
            except Exception:
                continue
            for key, rx in SECRET_PATTERNS.items():
                if rx.search(text):
                    secret_hits.append({"pattern": key, "path": name})

    ok = not deny_hits and not secret_hits
    return {
        "zip_path": str(path),
        "ok": ok,
        "deny_hits": deny_hits,
        "secret_hits": secret_hits,
        "deny_count": len(deny_hits),
        "secret_count": len(secret_hits),
    }



def main() -> int:
    ap = argparse.ArgumentParser(description="Audit release zip against denylist + secret patterns")
    ap.add_argument("--zip", dest="zip_path", default=DEFAULT_ZIP, help="Release zip path")
    ap.add_argument("--max-text-bytes", type=int, default=512_000, help="Skip text files larger than this size")
    args = ap.parse_args()

    zpath = Path(args.zip_path)
    if not zpath.exists():
        print(json.dumps({"ok": False, "error": "zip_not_found", "zip_path": str(zpath)}, ensure_ascii=False, indent=2))
        return 2

    report = audit_zip(zpath, max_text_bytes=int(args.max_text_bytes))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not bool(report.get("ok", False)):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
