#!/usr/bin/env python3
"""Generate ``datasets/validation_provenance.json`` — fingerprint-exclusion evidence.

The held-out validation set (``datasets/validation.jsonl``) must never leak into training.
This records the set's identity (file SHA256, byte size, row count) and a per-row text
fingerprint set (SHA256 of each row's extracted text via the repo's single
``train.packing.extract_row_text``) so that any training-data build can *prove* it excluded
these rows. It is provenance/exclusion evidence — not a quality or capability claim.

Run:
    python scripts/gen_validation_provenance.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "datasets" / "validation.jsonl"
OUT = ROOT / "datasets" / "validation_provenance.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build() -> dict:
    from train.packing import extract_row_text

    if not CORPUS.exists():
        raise FileNotFoundError(f"validation corpus not found: {CORPUS}")

    fingerprints: list[str] = []
    rows_total = 0
    with CORPUS.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows_total += 1
            text = extract_row_text(obj) if isinstance(obj, dict) else ""
            if text:
                fingerprints.append(hashlib.sha256(text.encode("utf-8")).hexdigest())

    unique_sorted = sorted(set(fingerprints))
    set_anchor = hashlib.sha256("\n".join(unique_sorted).encode("utf-8")).hexdigest()
    return {
        "schema": "validation_provenance_v1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": str(CORPUS.relative_to(ROOT)),
        "file_sha256": _sha256_file(CORPUS),
        "file_bytes": CORPUS.stat().st_size,
        "rows_total": rows_total,
        "rows_with_text": len(fingerprints),
        "unique_text_fingerprints": len(unique_sorted),
        "text_fingerprint_algorithm": "sha256(utf8(train.packing.extract_row_text(row)))",
        "fingerprint_set_sha256": set_anchor,
        "text_fingerprints": unique_sorted,
        "claim_boundary": (
            "Identity + per-row text fingerprints of the held-out validation set. A "
            "training-data build should exclude any training row whose text fingerprint "
            "appears here. This file is provenance/exclusion evidence, not a quality or "
            "capability claim."
        ),
    }


def main() -> int:
    data = build()
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[validation_provenance] wrote {OUT.relative_to(ROOT)} "
        f"rows={data['rows_total']} unique_fingerprints={data['unique_text_fingerprints']} "
        f"file_sha256={data['file_sha256'][:12]}..."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
