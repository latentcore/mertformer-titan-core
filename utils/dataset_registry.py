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


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HASHES_PATH = _PROJECT_ROOT / "datasets" / "hashes.json"


@lru_cache(maxsize=1)
def _load_hashes() -> dict:
    if not _HASHES_PATH.exists():
        return {}
    try:
        return json.loads(_HASHES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


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

