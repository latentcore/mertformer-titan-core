"""Local dataset registry helpers.

The repository maintains a single source-of-truth snapshot registry:
`datasets/hashes.json`.

We use it to pin Hugging Face dataset revisions (git SHA) whenever we call
`datasets.load_dataset(...)`, so that the training/eval pipeline is reproducible.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HASHES_PATH = _PROJECT_ROOT / "datasets" / "hashes.json"


@lru_cache(maxsize=1)
def _load_hashes() -> dict:
    if not _HASHES_PATH.exists():
        print(
            f"⚠️ dataset hash registry missing ({_HASHES_PATH}) — dataset revisions/snapshots are unpinned.",
            file=sys.stderr,
        )
        return {}
    try:
        obj = json.loads(_HASHES_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - unreadable registry is non-fatal, treat as unpinned
        print(
            f"⚠️ dataset hash registry unreadable ({_HASHES_PATH}): {exc} — treating as unpinned.",
            file=sys.stderr,
        )
        return {}
    sources = obj.get("sources", obj) if isinstance(obj, dict) else {}
    if not sources:
        print(
            f"⚠️ dataset hash registry has no sources ({_HASHES_PATH}) — dataset revisions/snapshots are unpinned.",
            file=sys.stderr,
        )
    return obj if isinstance(obj, dict) else {}


def get_hf_revision(dataset_id: str) -> str | None:
    """Return the pinned HF dataset revision (git SHA) if present."""
    obj = _load_hashes()
    sources = obj.get("sources", {}) if isinstance(obj, dict) else {}
    entry = sources.get(dataset_id) if isinstance(sources, dict) else None
    if not isinstance(entry, dict):
        return None
    rev = entry.get("revision")
    return str(rev) if rev else None


def get_snapshot_sha256(dataset_id: str) -> str | None:
    """Return the registry SHA256 fingerprint for the dataset snapshot, if present."""
    obj = _load_hashes()
    sources = obj.get("sources", {}) if isinstance(obj, dict) else {}
    entry = sources.get(dataset_id) if isinstance(sources, dict) else None
    if not isinstance(entry, dict):
        return None
    h = entry.get("sha256")
    return str(h) if h else None

