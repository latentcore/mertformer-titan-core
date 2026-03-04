#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def pip_freeze() -> list[str]:
    result = subprocess.run([
        ".titan-venv/bin/python", "-m", "pip", "freeze"
    ], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    out = Path("reports/sbom.cdx.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    components = []
    for line in pip_freeze():
        if "==" in line:
            name, ver = line.split("==", 1)
        else:
            name, ver = line, "unknown"
        components.append({
            "type": "library",
            "name": name,
            "version": ver,
            "purl": f"pkg:pypi/{name}@{ver}",
        })

    payload = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": {"type": "application", "name": "mertformer-titan-core"},
        },
        "components": components,
    }

    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SBOM written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
