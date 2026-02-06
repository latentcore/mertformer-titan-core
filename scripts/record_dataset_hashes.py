"""Record dataset snapshot hashes for reproducible training (HF manifest fingerprint).

This script is intentionally **online**: it queries Hugging Face Hub to pin dataset
revisions and to compute a stable SHA256 fingerprint of the repository file manifest.

It writes `datasets/hashes.json` which is used as the single source of truth for:
- revision pinning (`revision` / `snapshot`)
- air-gapped verification (manifest SHA256 over filenames + LFS sha256 when available)

NOTE:
- This does NOT download full dataset content.
- For very large datasets, the file manifest can still be large; expect a few seconds.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _sha256_hexdigest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_inventory(repo_root: Path) -> list[str]:
    inv = repo_root / "datasets" / "inventory.json"
    if not inv.exists():
        raise FileNotFoundError(f"Missing inventory: {inv}")
    obj = json.loads(inv.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unexpected inventory format: {inv}")

    items = obj.get("items", [])
    if not isinstance(items, list):
        raise RuntimeError(f"Unexpected inventory format (items not list): {inv}")

    ds_ids = [str(it["dataset"]) for it in items if isinstance(it, dict) and it.get("dataset")]
    if not ds_ids:
        raise RuntimeError(f"No datasets found in inventory: {inv}")
    return sorted(set(ds_ids))


def _manifest_from_dataset_info(repo_id: str, info: Any) -> dict[str, Any]:
    # Keep this canonical and stable; only include fields we can rely on.
    files: list[dict[str, Any]] = []
    lfs_sha_count = 0
    for sib in list(getattr(info, "siblings", None) or []):
        entry: dict[str, Any] = {"path": getattr(sib, "rfilename", None)}
        size = getattr(sib, "size", None)
        if size is not None:
            entry["size"] = int(size)
        blob_id = getattr(sib, "blob_id", None)
        if blob_id:
            entry["blob_id"] = str(blob_id)
        lfs = getattr(sib, "lfs", None)
        if lfs is not None:
            lfs_sha = getattr(lfs, "sha256", None)
            if lfs_sha:
                entry["lfs_sha256"] = str(lfs_sha)
                lfs_sha_count += 1
        files.append(entry)

    files = sorted(files, key=lambda x: str(x.get("path") or ""))
    revision = getattr(info, "sha", None)
    gated = getattr(info, "gated", None)
    return {
        "repo_id": repo_id,
        "revision": str(revision) if revision else None,
        "gated": gated,
        "files": files,
        "files_count": len(files),
        "files_lfs_sha256_count": lfs_sha_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(PROJECT_ROOT / "datasets" / "hashes.json"))
    parser.add_argument("--token", default=None, help="Optional HF token (or set HF_TOKEN env).")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")

    from huggingface_hub import HfApi  # type: ignore

    api = HfApi()
    ds_ids = _load_inventory(PROJECT_ROOT)

    generated_at = _utc_now_iso()
    sources: dict[str, Any] = {}

    for ds in ds_ids:
        info = api.dataset_info(ds, token=token, files_metadata=True)
        manifest = _manifest_from_dataset_info(ds, info)
        manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        sources[ds] = {
            "kind": "hf_repo_manifest",
            "snapshot": f"hf://{ds}@{manifest['revision']}",
            "revision": manifest["revision"],
            "snapshot_date_utc": generated_at,
            "sha256": _sha256_hexdigest(manifest_json),
            "status": "verified",
            "ref_url": f"https://huggingface.co/datasets/{ds}",
            "files_count": manifest["files_count"],
            "files_lfs_sha256_count": manifest["files_lfs_sha256_count"],
            "gated": manifest["gated"],
        }

    # Internal, tracked tiny files (these are safe to hash directly).
    for rel in ["datasets/validation.jsonl", "datasets/golden_samples.jsonl"]:
        p = PROJECT_ROOT / rel
        if not p.exists():
            continue
        sources[f"internal:{rel}"] = {
            "kind": "local_file",
            "snapshot": str(p.as_posix()),
            "revision": None,
            "snapshot_date_utc": generated_at,
            "sha256": _sha256_file(p),
            "status": "verified",
            "ref_url": None,
            "bytes": p.stat().st_size,
        }

    # Internal, local stage snapshots (gitignored artifacts, but hashable for air-gapped runs).
    stage_candidates = [
        "datasets/stage1/stage1_data.jsonl",
        "datasets/stage2/stage2_data.jsonl",
        "datasets/stage3/stage3_data.jsonl",
        "datasets/stage4_soul/stage4_data.jsonl",
        "datasets/stage5_tools/stage5_data.jsonl",
        "datasets/training_data.jsonl",  # legacy fallback
    ]
    for rel in stage_candidates:
        p = PROJECT_ROOT / rel
        if not p.exists():
            continue
        sources[f"internal:{rel}"] = {
            "kind": "local_file",
            "snapshot": str(p.as_posix()),
            "revision": None,
            "snapshot_date_utc": generated_at,
            "sha256": _sha256_file(p),
            "status": "verified",
            "ref_url": None,
            "bytes": p.stat().st_size,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at_utc": generated_at,
                "note": "For HF datasets, `sha256` is a fingerprint of the pinned repository file manifest (includes filenames + sizes + LFS sha256 when available). For local files, `sha256` is the file content hash.",
                "sources": sources,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote: {out} ({len(sources)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
