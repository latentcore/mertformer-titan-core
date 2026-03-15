from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.zip_denylist_audit import audit_zip, load_policy  # noqa: E402


def main() -> int:
    root = ROOT
    zip_path = root / "artifacts" / "mertformer_release.zip"
    policy_path = root / "policy" / "allow_deny_policy.yaml"
    report_path = root / "reports" / "artifacts_zip_denylist_audit.json"

    if not zip_path.exists():
        payload = {
            "ok": False,
            "error": "zip_not_found",
            "zip_path": str(zip_path),
        }
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    deny_patterns, secret_patterns = load_policy(policy_path)
    report = audit_zip(zip_path, deny_patterns=deny_patterns, secret_patterns=secret_patterns)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if bool(report.get("ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
