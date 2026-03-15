from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List


def _iter_md_files(root: Path) -> List[Path]:
    skip_parts = {".git", ".titan-venv", ".lint-venv", "site-packages", "__pycache__"}
    files: List[Path] = []
    for path in root.rglob("*.md"):
        if any(part in skip_parts for part in path.parts):
            continue
        files.append(path)
    return files


def audit_numbers(root: Path) -> Dict[str, object]:
    root_label = "<REPO_ROOT>"
    # Parameter-size claims in B format (e.g., 3.70B)
    b_any = re.compile(r"\b~?\d+(?:[.,]\d+)?B\b")
    b_ok = re.compile(r"\b~?\d+\.\d{2}B\b")
    findings: List[Dict[str, str]] = []

    for path in _iter_md_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        rel_path = str(path.relative_to(root))
        for idx, line in enumerate(text.splitlines(), start=1):
            for match in b_any.finditer(line):
                token = match.group(0)
                if not b_ok.fullmatch(token):
                    findings.append(
                        {
                            "file": rel_path,
                            "line": str(idx),
                            "token": token,
                            "context": line.strip()[:200],
                        }
                    )

    return {
        "schema": "claim_number_audit_v1",
        "root": root_label,
        "b_format_expected": "~X.XXB",
        "issue_count": len(findings),
        "issues": findings,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    report = audit_numbers(root)
    out = root / "reports" / "claim_number_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["issue_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
