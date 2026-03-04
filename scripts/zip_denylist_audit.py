from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, List

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

DEFAULT_ZIP = "packages/MertFormer_Titan_OnyxStorm_v1.0_B30_Release.zip"
DEFAULT_POLICY = "policy/allow_deny_policy.yaml"

TEXT_EXTS = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv", ".jsonl", ".sh"}


def load_policy(policy_path: Path) -> tuple[List[re.Pattern[str]], Dict[str, re.Pattern[str]]]:
    deny_regex = [
        r"(^|/)__pycache__/",
        r"(^|/)\.pytest_cache/",
        r"(^|/)\.ruff_cache/",
        r"(^|/)\.mypy_cache/",
        r"(^|/)\.git/",
        r"(^|/)\.titan-venv/",
        r"(^|/)\.venv/",
        r"(^|/)venv/",
        r"(^|/)\.env$",
    ]
    secret = {
        "hf_token": r"\bhf_[A-Za-z0-9]{24,}\b",
        "wandb_token": r"\bwandb_[A-Za-z0-9]{10,}\b",
        "openai_key": r"\bsk-[A-Za-z0-9]{10,}\b",
        "github_token": r"\bghp_[A-Za-z0-9]{20,}\b",
    }

    if policy_path.exists() and yaml is not None:
        try:
            data = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
            deny_regex = list((data.get("deny", {}) or {}).get("path_regex", deny_regex))
            secret = dict((data.get("deny", {}) or {}).get("secret_regex", secret))
        except Exception:
            pass

    return [re.compile(x) for x in deny_regex], {k: re.compile(v) for k, v in secret.items()}


def _is_text_candidate(path: str) -> bool:
    p = Path(path)
    return p.suffix.lower() in TEXT_EXTS or p.suffix == ""


def audit_zip(
    path: Path,
    deny_patterns: List[re.Pattern[str]] | None = None,
    secret_patterns: Dict[str, re.Pattern[str]] | None = None,
    max_text_bytes: int = 512_000,
) -> Dict[str, object]:
    # Backward-compatible defaults for existing callers/tests.
    if deny_patterns is None or secret_patterns is None:
        _deny, _secret = load_policy(Path(DEFAULT_POLICY))
        deny_patterns = _deny if deny_patterns is None else deny_patterns
        secret_patterns = _secret if secret_patterns is None else secret_patterns

    deny_hits: List[str] = []
    secret_hits: List[Dict[str, object]] = []

    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        for name in names:
            for rx in deny_patterns:
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
                text = zf.read(name).decode("utf-8", errors="ignore")
            except Exception:
                continue
            for key, rx in secret_patterns.items():
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
    ap = argparse.ArgumentParser(description="Audit release zip against canonical policy denylist + secret patterns")
    ap.add_argument("--zip", dest="zip_path", default=DEFAULT_ZIP, help="Release zip path")
    ap.add_argument("--policy", default=DEFAULT_POLICY, help="Policy YAML path")
    ap.add_argument("--max-text-bytes", type=int, default=512_000, help="Skip text files larger than this size")
    args = ap.parse_args()

    zpath = Path(args.zip_path)
    if not zpath.exists():
        print(json.dumps({"ok": False, "error": "zip_not_found", "zip_path": str(zpath)}, ensure_ascii=False, indent=2))
        return 2

    deny_patterns, secret_patterns = load_policy(Path(args.policy))
    report = audit_zip(zpath, deny_patterns=deny_patterns, secret_patterns=secret_patterns, max_text_bytes=int(args.max_text_bytes))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if bool(report.get("ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
