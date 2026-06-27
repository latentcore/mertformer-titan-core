#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Reproducibility hash kapsami: asagidaki sabit dosya listesiyle SINIRLIDIR.
# DIKKAT: Bu liste manuel olarak bakimi yapilir. Repo evrildikce yeni kritik
# dosyalar eklenmezse hash kapsami sessizce daralir (yeni dosyalardaki
# degisiklikler "ok" sonucunu etkilemez). Kritik bir dosya eklendiginde bu
# listeyi elle guncelleyin. (Dinamik manifest/glob kesfi kasitli olarak
# kullanilmadi: cikti hash sozlesmesini deterministik tutmak icin.)
TARGETS = [
    "pyproject.toml",
    "requirements.txt",
    "config/config.py",
    "train/train.py",
    "layers/bitlinear.py",
    "mertformer_sdk/kernels/dispatcher.py",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_out(*args: str) -> str:
    r = subprocess.run(args, capture_output=True, text=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def main() -> int:
    rows = []
    for t in TARGETS:
        p = Path(t)
        if p.exists():
            rows.append({"path": t, "sha256": sha256(p), "size_bytes": p.stat().st_size})
        else:
            rows.append({"path": t, "missing": True})

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": cmd_out("git", "rev-parse", "HEAD"),
        "python": cmd_out(".titan-venv/bin/python", "-V"),
        "targets": rows,
        "ok": all("sha256" in x for x in rows),
    }

    out = Path("reports/repro_build_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": payload["ok"], "target_count": len(rows)}))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
