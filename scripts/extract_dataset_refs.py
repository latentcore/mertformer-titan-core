"""Extract dataset references from the codebase (best-effort).

Outputs:
  - datasets/inventory.json
  - datasets/inventory.md (EN)
  - datasets/inventory_TR.md (TR)

This is a heuristic scanner (regex-based) designed for provenance/compliance work.
It must not require network by default. Metadata fetch is opt-in.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


SCAN_DIRS = ("scripts", "eval", "train", "orchestrator")

RX_LOAD_DATASET = re.compile(r"""\bload_dataset\(\s*["']([^"']+)["']""")
RX_LOAD_DATASET_SAFE = re.compile(r"""\bload_dataset_safe\(\s*["']([^"']+)["']""")
RX_DATASET_FIELD = re.compile(r"""["']dataset["']\s*:\s*["']([^"']+)["']""")
RX_DATASET_INFO = re.compile(r"""\bdataset_info\(\s*["']([^"']+)["']""")


@dataclass
class Ref:
    dataset: str
    file: str
    line: int
    kind: str
    snippet: str


def _iter_py_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for d in SCAN_DIRS:
        p = root / d
        if not p.exists():
            continue
        out.extend(sorted(p.rglob("*.py")))
    return out


def _scan_file(path: Path) -> list[Ref]:
    refs: list[Ref] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return refs

    for i, line in enumerate(text.splitlines(), start=1):
        for rx, kind in (
            (RX_LOAD_DATASET, "load_dataset"),
            (RX_LOAD_DATASET_SAFE, "load_dataset_safe"),
            (RX_DATASET_FIELD, "pipeline_source"),
            (RX_DATASET_INFO, "dataset_info"),
        ):
            m = rx.search(line)
            if not m:
                continue
            ds = m.group(1).strip()
            refs.append(
                Ref(
                    dataset=ds,
                    file=str(path.as_posix()),
                    line=i,
                    kind=kind,
                    snippet=line.strip()[:200],
                )
            )
    return refs


def _fetch_metadata(dataset_id: str, token: str | None) -> dict[str, Any]:
    # Best-effort Hugging Face dataset metadata. Never fails the scan.
    try:
        from huggingface_hub import HfApi  # type: ignore

        api = HfApi()
        info = api.dataset_info(dataset_id, token=token)
        card = getattr(info, "cardData", None) or {}
        return {
            "license": card.get("license") or card.get("licenses") or None,
            "homepage": card.get("homepage") or None,
            "gated": bool(getattr(info, "gated", False)),
            "sha": getattr(info, "sha", None),
            "private": bool(getattr(info, "private", False)),
            "hf_url": f"https://huggingface.co/datasets/{dataset_id}",
        }
    except Exception:
        return {"license": None, "homepage": None, "gated": None, "sha": None, "private": None, "hf_url": f"https://huggingface.co/datasets/{dataset_id}"}


def _write_md(path: Path, title: str, items: list[dict[str, Any]], lang: str) -> None:
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    if lang == "tr":
        lines.append("Bu dosya, kod tabaninda referanslanan dataset kimliklerinin otomatik envanteridir (best-effort).")
        lines.append("Lisans/provenans dogrulamasi icin `datasets/SOURCES*.md` ve `datasets/LICENSES*.md` dosyalarini referans alin.")
    else:
        lines.append("This file is an auto-generated inventory of dataset identifiers referenced by the codebase (best-effort).")
        lines.append("For license/provenance work, see `datasets/SOURCES*.md` and `datasets/LICENSES*.md`.")
    lines.append("")
    lines.append("| Dataset | License (best-effort) | HF URL | Refs |")
    lines.append("| --- | --- | --- | --- |")
    for it in items:
        ds = it["dataset"]
        lic = it.get("metadata", {}).get("license")
        lic_str = "TBD" if not lic else (lic if isinstance(lic, str) else ", ".join(lic))
        url = it.get("metadata", {}).get("hf_url") or f"https://huggingface.co/datasets/{ds}"
        refs = str(it.get("ref_count", 0))
        lines.append(f"| `{ds}` | {lic_str} | {url} | {refs} |")
    lines.append("")
    lines.append("## Reference Details")
    lines.append("")
    for it in items:
        lines.append(f"### `{it['dataset']}`")
        for ref in it.get("refs", [])[:50]:
            lines.append(f"- {ref['file']}:{ref['line']} ({ref['kind']})")
        if len(it.get('refs', [])) > 50:
            lines.append(f"- ... and {len(it['refs']) - 50} more")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-json", default="datasets/inventory.json")
    parser.add_argument("--out-md", default="datasets/inventory.md")
    parser.add_argument("--out-md-tr", default="datasets/inventory_TR.md")
    parser.add_argument("--fetch-metadata", action="store_true", help="Fetch HF metadata (license/gated/sha).")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    refs: list[Ref] = []
    for p in _iter_py_files(root):
        refs.extend(_scan_file(p))

    by_ds: dict[str, list[Ref]] = {}
    for r in refs:
        by_ds.setdefault(r.dataset, []).append(r)

    token = os.environ.get("HF_TOKEN") if args.fetch_metadata else None

    items: list[dict[str, Any]] = []
    for ds in sorted(by_ds.keys()):
        ds_refs = by_ds[ds]
        meta = _fetch_metadata(ds, token=token) if args.fetch_metadata else {"license": None, "hf_url": f"https://huggingface.co/datasets/{ds}"}
        items.append(
            {
                "dataset": ds,
                "ref_count": len(ds_refs),
                "refs": [asdict(r) for r in ds_refs],
                "metadata": meta,
            }
        )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"generated_from": list(SCAN_DIRS), "items": items}, indent=2), encoding="utf-8")

    _write_md(Path(args.out_md), "Dataset Inventory (Auto)", items, lang="en")
    _write_md(Path(args.out_md_tr), "Dataset Envanteri (Otomatik)", items, lang="tr")

    print(f"Wrote: {out_json} ({len(items)} datasets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
